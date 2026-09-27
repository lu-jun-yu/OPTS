"""CPU tests: PYTHONPATH=LLM python -m unittest discover -s LLM/test -p test_main_eval.py."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from trainer import main_eval
from utils.reward_fn import compute_score_sync as reward_score


class MainEvalTests(unittest.TestCase):
    def test_training_validation_omits_predictions_before_metric_computation(self):
        root = Path(__file__).resolve().parents[1]
        for file in (root / "trainer/opts_ttpo/ray_trainer.py", root / "verl/verl/trainer/ppo/ray_trainer.py"):
            with self.subTest(file=file):
                tree = ast.parse(file.read_text())
                method = next(node for node in ast.walk(tree)
                              if isinstance(node, ast.FunctionDef) and node.name == "_validate")
                call = next(node for node in ast.walk(method) if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name) and node.func.id == "process_validation_metrics")
                expression = ast.Expression(body=call.args[2])
                infos = {"acc": [1, 0], "reward": [1, 0], "pred": ["answer1", "answer2"]}
                filtered = eval(compile(expression, str(file), "eval"), {"reward_extra_infos_dict": infos})
                self.assertEqual(filtered, {"acc": [1, 0], "reward": [1, 0]})
                target = next(node.value for node in ast.walk(method) if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "target_acc_metric_names" for t in node.targets))
                self.assertEqual(eval(compile(ast.Expression(target), str(file), "eval"), {"target_n": 32}),
                                 ("avg@32", "pass@32"))

    def test_validation_skips_consensus_but_test_mode_retains_it(self):
        frame = pd.DataFrame([{
            "responses": [r"\boxed{1}", r"\boxed{0}"],
            "reward_model": {"ground_truth": "1"},
            "data_source": "math500",
        }])
        with patch.object(main_eval.pd, "read_parquet", return_value=frame), \
             patch.object(main_eval, "tqdm", side_effect=lambda values, **kwargs: values), \
             patch.object(main_eval, "cons_at_k", side_effect=AssertionError("consensus must not run")), \
             redirect_stdout(StringIO()):
            result = main_eval.evaluate_pregenerated_parquet("unused.parquet", ["avg", "pass"], [32])
        self.assertEqual(result["_all"], {"avg@32": 0.5, "pass@32": 1.0})
        with patch.object(main_eval.pd, "read_parquet", return_value=frame), \
             patch.object(main_eval, "tqdm", side_effect=lambda values, **kwargs: values), \
             patch.object(main_eval, "cons_at_k", return_value=1.0) as consensus, \
             redirect_stdout(StringIO()):
            result = main_eval.evaluate_pregenerated_parquet("unused.parquet", ["avg", "pass", "cons"], [32])
        consensus.assert_called_once()
        self.assertEqual(result["_all"]["cons@32"], 1.0)

    def test_training_and_eval_correctness_match(self):
        cases = [
            (r"\boxed{36}</think>no later answer", "36", 1.0),
            (r"<think>\boxed{36}</think>", "36", 1.0),
            (r"\boxed{36}\boxed{9}</think>explanation 36", "36", 0.0),
            (r"\boxed{36}</think>\boxed{9}", "36", 0.0),
            (r"\boxed{9}</think>\boxed{36}</think>", "36", 1.0),
            (r"\boxed{9} corrected to \boxed{36}", "36", 1.0),
            (r"\boxed{\frac{1}{{2}}}", "0.5", 1.0),
            (r"\boxed{\sqrt{\frac{1}{4}}}", "0.5", 1.0),
            (r"\boxed{\sqrt{\frac{1}{4}}}", "0.25", 0.0),
            (r"\boxed{10\%}", "10", 0.0),
            (r"\boxed{1+10\%}", "1.1", 1.0),
            (r"\boxed{1+10\%}", "0.11", 0.0),
            (r"\boxed{0} then \boxed {42}", "42", 1.0),
            (r"\boxed{\frac{-1}{3}}", "-1./3", 1.0),
            (r"\boxed{\arcsin(10/13)}", "np.arcsin(10/13)", 1.0),
            (r"\boxed{36} then \boxed{}", "36", 0.0),
            (r"\boxed{36} then \boxed{9", "36", 1.0),
            (r"answer: 36", "36", 0.0),
            (r"\boxed{36", "36", 0.0),
        ]
        for response, truth, expected in cases:
            with self.subTest(response=response, truth=truth):
                self.assertEqual(reward_score("test", response, truth)["acc"], expected)
                self.assertEqual(main_eval.compute_score(response, truth), expected)
                self.assertEqual(main_eval.is_answer_correct(response, truth), bool(expected))
                self.assertEqual(main_eval.cons_at_k([response], truth, 1), expected)

    def test_parquet_metrics_keep_shared_answer_rules(self):
        frame = pd.DataFrame([{
            "responses": [r"\boxed{36}</think>no later answer", r"\boxed{9}", r"\boxed{36}"],
            "reward_model": {"ground_truth": "36"},
            "data_source": "math500",
        }])
        with patch.object(main_eval.pd, "read_parquet", return_value=frame), \
             patch.object(main_eval, "tqdm", side_effect=lambda values, **kwargs: values), \
             redirect_stdout(StringIO()):
            result = main_eval.evaluate_pregenerated_parquet("unused.parquet", ["avg", "pass", "cons"], [3])
        self.assertEqual(result["_all"], {"avg@3": 2 / 3, "pass@3": 1.0, "cons@3": 1.0})

    def test_generation_entry_preserves_full_response_and_metrics(self):
        responses = [
            [r"\boxed{36}</think>no later answer", r"\boxed{36}\boxed{9}</think>explanation 36"],
            [r"\boxed{4}</think>\boxed{\frac{1}{{2}}}", "answer: 0.5"],
        ]
        frame = pd.DataFrame([{
            "prompt": [{"role": "user", "content": "test question"}],
            "reward_model": {"ground_truth": truth},
        } for truth in ("36", "0.5")])
        for n in (1, 2):
            with self.subTest(n=n), TemporaryDirectory(prefix="opts_test_main_eval_") as output_dir:
                llm = Mock()
                llm.get_tokenizer.return_value.apply_chat_template.return_value = "test prompt"
                llm.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text=text) for text in row[:n]])
                                             for row in responses]
                fake_vllm = SimpleNamespace(LLM=Mock(return_value=llm), SamplingParams=Mock())
                argv = ["main_eval", "--model_path", "unused-model", "--datasets", "math500",
                        "--n", str(n), "--output_dir", output_dir]
                with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"vllm": fake_vllm}), \
                     patch.object(main_eval, "load_test_data", return_value={"math500": frame}), \
                     patch.object(main_eval, "tqdm", side_effect=lambda values, **kwargs: values), \
                     redirect_stdout(StringIO()):
                    main_eval.main()
                details = json.loads((Path(output_dir) / f"math500_n{n}.json").read_text())
                self.assertEqual([row["scores"] for row in details], [[1.0, 0.0][:n]] * 2)
                self.assertEqual([row["responses"] for row in details], [row[:n] for row in responses])
                summary = json.loads((Path(output_dir) / f"summary_n{n}.json").read_text())
                expected = {"total": 2, "accuracy": 1.0, "correct": 2} if n == 1 else {
                    "total": 2, "pass@1": 0.5, "pass@2": 1.0,
                }
                self.assertEqual(summary["results"]["math500"], expected)


if __name__ == "__main__":
    unittest.main()
