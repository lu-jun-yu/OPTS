"""CPU tests: PYTHONPATH=LLM:LLM/verl python -m unittest discover -s LLM/test -p test_response_boundary.py."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoTokenizer
from verl import DataProto

from trainer.opts_ttpo.ray_trainer import (
    TreeSearchState,
    prepare_next_round_input,
    refresh_tree_search_states,
    select_next_states,
    selected_to_branch_points,
)
from utils.response_boundary import decode_response_strs, find_first_boxed_token, find_last_boxed_token
from utils.reward_fn import extract_answer


class ResponseBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        models = Path(__file__).resolve().parents[1] / "models"
        cls.tokenizers = [AutoTokenizer.from_pretrained(models / name, local_files_only=True)
                          for name in ("Qwen3-1.7B", "Qwen3-1.7B-Base", "Qwen3-8B-Base")]
        cls.tokenizer = cls.tokenizers[0]

    def make_batch(self, suffixes, inherited=None):
        tokenizer = self.tokenizer
        inherited = inherited or [[] for _ in suffixes]
        # A box in the original question must never become a response boundary.
        raw_prompt = tokenizer.encode(r"Question: \boxed{999}?", add_special_tokens=False)
        response_len = max(len(a) + len(b) for a, b in zip(inherited, suffixes)) + 2
        prompt_len = len(raw_prompt) + response_len + 4
        n = len(suffixes)
        inputs = torch.full((n, prompt_len + response_len), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(inputs)
        response_mask = torch.zeros((n, response_len), dtype=torch.long)
        for i, (prefix, suffix) in enumerate(zip(inherited, suffixes)):
            prompt = raw_prompt + prefix
            inputs[i, prompt_len - len(prompt):prompt_len + len(suffix)] = torch.tensor(prompt + suffix)
            attention[i, prompt_len - len(prompt):prompt_len + len(suffix)] = 1
            response_mask[i, :len(suffix)] = 1
        return DataProto.from_dict(tensors={
            "input_ids": inputs,
            "attention_mask": attention,
            "responses": inputs[:, prompt_len:],
            "response_mask": response_mask,
            "advantages": torch.ones((n, response_len)),
            "state_branches": torch.ones((n, response_len)),
        }, non_tensors={
            "uid": np.array(["tree"] * n, dtype=object),
            "rid": np.array([f"r{i}" for i in range(n)], dtype=object),
            "pid": np.array([None] * n, dtype=object),
            "cid": np.array([{} for _ in range(n)], dtype=object),
            "branch_pos": np.full(n, -1),
            "raw_prompt_len": np.full(n, len(raw_prompt)),
        })

    def assert_branch_prefix(self, batch, expected_answer_prefix):
        response_len = batch.batch["responses"].shape[1]
        prompt_len = batch.batch["input_ids"].shape[1] - response_len
        state = refresh_tree_search_states(batch, ["tree"], {}, 1.0, prompt_len,
                                           tokenizer=self.tokenizer)["tree"]
        row = list(batch.non_tensor_batch["rid"]).index(state.candidate_rid)
        branches = selected_to_branch_points({"tree": (row, state.candidate_pos)}, batch)
        next_batch = prepare_next_round_input(batch, branches, self.tokenizer.pad_token_id)
        prompt = next_batch.batch["input_ids"][0][next_batch.batch["attention_mask"][0].bool()].tolist()
        raw_len = int(batch.non_tensor_batch["raw_prompt_len"][0])
        self.assertEqual(prompt[raw_len:], expected_answer_prefix)
        self.assertNotIn("first_boxed_token_pos", next_batch.non_tensor_batch)
        self.assertNotIn("last_boxed_token_pos", next_batch.non_tensor_batch)
        return state

    def test_search_first_box_and_reward_last_box_token_offsets(self):
        cases = [
            (r"\boxed{1}", r"\boxed{1}", "1"),
            (r"\boxed {1}", r"\boxed {1}", "1"),
            ("\\boxed{1} 中文🙂 \\boxed \t\n {2}", "\\boxed \t\n {2}", "2"),
            (r"\boxed{1} then \boxed {\frac{2}{{3}}}", r"\boxed {\frac", r"\frac{2}{{3}}"),
            (r"\boxed{1} then \boxed {2", r"\boxed{1}", "1"),
            (r"\boxed{1} repeated \boxed{1}", r"\boxed{1}", "1"),
            (r"wrong $\boxed{1}$; revised \(\boxed{2}\)", r"\boxed{2}", "2"),
            (r"中文🙂：\boxed{1}</think>最终\boxed{\frac{1}{{2}}}", r"\boxed{\frac", r"\frac{1}{{2}}"),
            (r"\boxed{1} then \boxed{\left\{2,3\right\}}", r"\boxed{\left", r"\left\{2,3\right\}"),
            (r"\boxed{1} then \boxed{\frac{2}{3}", r"\boxed{1}", "1"),
            (r"\boxed{1} then \boxed{}", r"\boxed{}", ""),
            (r"\boxed{1} then \boxed", r"\boxed{1}", "1"),
            (r"\boxed{\frac{1}{2}", None, None),
            ("no answer", None, None),
        ]
        for tokenizer in self.tokenizers:
            for text, marker, answer in cases:
                with self.subTest(model=tokenizer.name_or_path, text=text):
                    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                    expected = -1 if marker is None else next(
                        i for i, (start, end) in enumerate(encoded["offset_mapping"])
                        if start <= text.rindex(marker) < end
                    )
                    self.assertEqual(find_last_boxed_token(tokenizer, encoded["input_ids"]), expected)
                    first = text.find(r"\boxed")
                    expected_first = -1 if first < 0 else next(
                        i for i, (start, end) in enumerate(encoded["offset_mapping"])
                        if start <= first < end
                    )
                    self.assertEqual(find_first_boxed_token(tokenizer, encoded["input_ids"]), expected_first)
                    self.assertEqual(extract_answer(text), answer)

    def test_original_token_splits_and_special_tokens(self):
        tokenizer = self.tokenizer
        prefix = tokenizer.encode("中文🙂 reasoning ", add_special_tokens=False)
        # Deliberately split the marker differently from normal BPE encoding.
        suffix = sum((tokenizer.encode(s, add_special_tokens=False)
                      for s in ("\\", "bo", "xed", "{", "2", "}")), [])
        ids = [tokenizer.eos_token_id] + prefix + suffix + [tokenizer.eos_token_id]
        with patch.object(tokenizer, "decode", side_effect=AssertionError("unexpected decode")), \
             patch.object(tokenizer, "encode", side_effect=AssertionError("unexpected encode")):
            self.assertEqual(find_first_boxed_token(tokenizer, torch.tensor(ids)), len(prefix) + 1)

    def test_cached_text_and_old_boundary_cache(self):
        tokenizer = self.tokenizer
        prefix = tokenizer.encode(r"reasoning \boxed{1} revised ", add_special_tokens=False)
        suffix = tokenizer.encode(r"\boxed{2} done", add_special_tokens=False)
        batch = self.make_batch([suffix], [prefix])
        text = tokenizer.decode(prefix + suffix, skip_special_tokens=True)
        batch.non_tensor_batch["full_response_str"] = np.array([text], dtype=object)
        batch.non_tensor_batch["last_boxed_token_pos"] = np.array([len(prefix)])
        width = batch.batch["responses"].shape[1]
        prompt_width = batch.batch["input_ids"].shape[1] - width
        with patch.object(tokenizer, "decode", side_effect=AssertionError("decoded cached text")):
            self.assertEqual(decode_response_strs(batch, tokenizer, prompt_width, width), [text])
            self.assertEqual(batch.non_tensor_batch["first_boxed_token_pos"][0],
                             find_first_boxed_token(tokenizer, prefix + suffix))
            with patch("utils.response_boundary.find_first_boxed_token", side_effect=AssertionError("recomputed cache")):
                self.assertEqual(decode_response_strs(batch, tokenizer, prompt_width, width), [text])

    def test_legacy_decode_once_and_no_box_in_answer(self):
        tokenizer = self.tokenizer
        suffix = tokenizer.encode("no boxed answer", add_special_tokens=False)
        batch = self.make_batch([suffix])
        width = batch.batch["responses"].shape[1]
        prompt_width = batch.batch["input_ids"].shape[1] - width
        with patch.object(tokenizer, "decode", wraps=tokenizer.decode) as decode:
            decode_response_strs(batch, tokenizer, prompt_width, width)
            decode_response_strs(batch, tokenizer, prompt_width, width)
            self.assertEqual(decode.call_count, 1)
        self.assertEqual(batch.non_tensor_batch["first_boxed_token_pos"].tolist(), [-1])

    def test_root_search_replaces_first_box(self):
        tokenizer = self.tokenizer
        ids = tokenizer.encode(r"\boxed{1} revise to \boxed{2} extra explanation.", add_special_tokens=False)
        batch = self.make_batch([ids])
        batch.batch["advantages"][0, len(ids) - 1] = -100
        pos = find_first_boxed_token(tokenizer, ids)
        state = self.assert_branch_prefix(batch, ids[:pos])
        self.assertEqual(state.candidate_pos, pos)
        self.assertGreater(state.raw_candidate_pos, pos)
        self.assertIsNone(extract_answer(tokenizer.decode(ids[:pos])))

    def test_search_can_still_select_before_the_first_box(self):
        ids = self.tokenizer.encode(r"reasoning \boxed{1} revised \boxed{2}", add_special_tokens=False)
        batch = self.make_batch([ids])
        batch.batch["advantages"][0, 0] = -100
        state = self.assert_branch_prefix(batch, [])
        self.assertEqual(state.candidate_pos, 0)

    def test_xi_changes_position_using_length_normalized_score(self):
        batch = self.make_batch([[100, 101, 102]])
        batch.batch["advantages"].zero_()
        batch.batch["advantages"][0, 0] = -1.0
        batch.batch["advantages"][0, 1] = -9.0
        prompt_width = batch.batch["input_ids"].shape[1] - batch.batch["responses"].shape[1]

        unpenalized = refresh_tree_search_states(
            batch, ["tree"], {}, 1.0, prompt_width, tokenizer=self.tokenizer, xi=0.0
        )["tree"]
        normalized = refresh_tree_search_states(
            batch, ["tree"], {}, 1.0, prompt_width, tokenizer=self.tokenizer, xi=1.0
        )["tree"]

        self.assertEqual(unpenalized.candidate_pos, 0)
        self.assertEqual(normalized.candidate_pos, 1)
        self.assertAlmostEqual(normalized.candidate_perf_diff, 9.0, places=6)
        self.assertAlmostEqual(normalized.candidate_selection_score, 3.0, places=6)

    def test_baseline_and_cross_tree_ranking_use_candidate_score(self):
        batch = self.make_batch([[100], [101]])
        batch.non_tensor_batch["uid"] = np.array(["a", "b"], dtype=object)

        def state(rid, candidate_score, unconstrained_score):
            return TreeSearchState(
                terminal_rid=rid,
                terminal_pos=1,
                raw_candidate_rid=rid,
                raw_candidate_pos=0,
                raw_perf_diff=unconstrained_score,
                raw_selection_score=unconstrained_score,
                candidate_rid=rid,
                candidate_pos=0,
                candidate_perf_diff=candidate_score,
                candidate_selection_score=candidate_score,
                updated_round=0,
            )

        selected = select_next_states(
            batch=batch,
            search_count={},
            max_perf_diffs={},
            max_search_per_tree=1,
            tree_search_state_by_uid={
                "a": state("r0", candidate_score=1.0, unconstrained_score=100.0),
                "b": state("r1", candidate_score=2.0, unconstrained_score=0.0),
            },
            max_searched_tree_ratio=1.0,
            search_batch_size=2,
            perf_diff_baseline_mode="mean",
        )
        self.assertEqual(set(selected), {"b"})

    def test_search_replaces_spaced_box(self):
        ids = self.tokenizer.encode("reasoning \\boxed \n {1} revised \\boxed{2} tail", add_special_tokens=False)
        batch = self.make_batch([ids])
        batch.batch["advantages"][0, len(ids) - 1] = -100
        pos = find_first_boxed_token(self.tokenizer, ids)
        state = self.assert_branch_prefix(batch, ids[:pos])
        self.assertEqual(state.candidate_pos, pos)
        self.assertIsNone(extract_answer(self.tokenizer.decode(ids[:pos])))

    def test_unfinished_first_box_also_limits_search(self):
        ids = self.tokenizer.encode(r"reasoning then unfinished \boxed{2", add_special_tokens=False)
        batch = self.make_batch([ids])
        batch.batch["advantages"][0, len(ids) - 1] = -100
        pos = find_first_boxed_token(self.tokenizer, ids)
        state = self.assert_branch_prefix(batch, ids[:pos])
        self.assertEqual(state.candidate_pos, pos)

    def test_no_box_keeps_existing_search_limit(self):
        ids = self.tokenizer.encode("reasoning without an answer marker", add_special_tokens=False)
        batch = self.make_batch([ids])
        batch.batch["advantages"][0, len(ids) - 1] = -100
        state = self.assert_branch_prefix(batch, ids[:-1])
        self.assertEqual(state.candidate_pos, len(ids) - 1)

    def test_child_uses_terminal_answer_not_discarded_parent(self):
        tokenizer = self.tokenizer
        prefix = tokenizer.encode("reasoning ", add_special_tokens=False)
        root_suffix = tokenizer.encode(r"discarded \boxed{999} trailing text", add_special_tokens=False)
        child_suffix = tokenizer.encode(r"continue thinking then \boxed{\frac{1}{{2}}} trailing text", add_special_tokens=False)
        batch = self.make_batch([prefix + root_suffix, child_suffix], [[], prefix])
        batch.non_tensor_batch["pid"][1] = "r0"
        batch.non_tensor_batch["branch_pos"][1] = len(prefix) - 1
        batch.non_tensor_batch["cid"][0] = {len(prefix) - 1: ["r1"]}
        batch.batch["advantages"][1] = 2
        batch.batch["advantages"][1, len(child_suffix) - 1] = -100
        pos = find_first_boxed_token(tokenizer, prefix + child_suffix)
        state = self.assert_branch_prefix(batch, (prefix + child_suffix)[:pos])
        self.assertEqual(state.terminal_rid, "r1")
        self.assertEqual(state.candidate_pos + len(prefix), pos)

    def test_first_box_in_inherited_prefix_clamps_back_to_ancestor(self):
        tokenizer = self.tokenizer
        prefix = tokenizer.encode(r"reasoning \boxed{1} revised ", add_special_tokens=False)
        root_suffix = tokenizer.encode("discarded tail", add_special_tokens=False)
        child_suffix = tokenizer.encode(r"\boxed{2} trailing text", add_special_tokens=False)
        batch = self.make_batch([prefix + root_suffix, child_suffix], [[], prefix])
        batch.non_tensor_batch["pid"][1] = "r0"
        batch.non_tensor_batch["branch_pos"][1] = len(prefix) - 1
        batch.non_tensor_batch["cid"][0] = {len(prefix) - 1: ["r1"]}
        batch.batch["advantages"][1] = 2
        batch.batch["advantages"][1, len(child_suffix) - 1] = -100
        pos = find_first_boxed_token(tokenizer, prefix + child_suffix)
        state = self.assert_branch_prefix(batch, prefix[:pos])
        self.assertEqual(state.terminal_rid, "r1")
        self.assertEqual(state.candidate_rid, "r0")
        self.assertEqual(state.candidate_pos, pos)

    def test_box_spanning_ancestor_and_child(self):
        tokenizer = self.tokenizer
        before = tokenizer.encode("reasoning ", add_special_tokens=False)
        prefix = before + tokenizer.encode(r"\bo", add_special_tokens=False)
        root_suffix = tokenizer.encode("bad discarded tail", add_special_tokens=False)
        child_suffix = tokenizer.encode("xed{2} trailing text", add_special_tokens=False)
        batch = self.make_batch([prefix + root_suffix, child_suffix], [[], prefix])
        batch.non_tensor_batch["pid"][1] = "r0"
        batch.non_tensor_batch["branch_pos"][1] = len(prefix) - 1
        batch.non_tensor_batch["cid"][0] = {len(prefix) - 1: ["r1"]}
        batch.batch["advantages"][1] = 2
        batch.batch["advantages"][1, len(child_suffix) - 1] = -100
        state = self.assert_branch_prefix(batch, before)
        self.assertEqual(state.terminal_rid, "r1")
        self.assertEqual(state.candidate_rid, "r0")


if __name__ == "__main__":
    unittest.main()
