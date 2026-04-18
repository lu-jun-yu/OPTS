以下采用论文初稿阶段的可读格式；正式投稿时可进一步统一为 NeurIPS BibTeX。  
对于 Hugging Face 数据集/模型条目，访问日期统一记为 `2026-04-18`。  
当前共 `38` 条参考文献/资源条目。

## A. 强化学习与搜索基础

1. Richard S. Sutton. *The Bitter Lesson*. 2019. http://www.incompleteideas.net/IncIdeas/BitterLesson.html
2. John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR 2016. https://arxiv.org/abs/1506.02438
3. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. *Proximal Policy Optimization Algorithms*. arXiv, 2017. https://arxiv.org/abs/1707.06347
4. David Silver et al. *Mastering the game of Go with deep neural networks and tree search*. Nature, 2016. https://www.nature.com/articles/nature16961
5. David Silver et al. *Mastering the game of Go without human knowledge*. Nature, 2017. https://www.nature.com/articles/nature24270
6. David Silver et al. *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play*. Science, 2018. https://doi.org/10.1126/science.aar6404
7. Julian Schrittwieser et al. *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*. Nature, 2020. https://www.nature.com/articles/s41586-020-03051-4
8. Thomas Anthony, Zheng Tian, and David Barber. *Thinking Fast and Slow with Deep Learning and Tree Search* (Expert Iteration). NeurIPS 2017. https://papers.neurips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search

## B. LLM 强化学习与 RLVR

9. DeepSeek-AI et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv, 2024. https://arxiv.org/abs/2402.03300
10. DeepSeek-AI et al. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv, 2025. https://arxiv.org/abs/2501.12948
11. Arash Ahmadian et al. *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs*. ACL 2024. https://aclanthology.org/2024.acl-long.662/
12. Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. *GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning*. arXiv, 2025. https://arxiv.org/abs/2504.02546
13. Qiying Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv, 2025. https://arxiv.org/abs/2503.14476
14. Guangming Sheng et al. *HybridFlow: A Flexible and Efficient RLHF Framework* (VeRL). arXiv, 2024. https://arxiv.org/abs/2409.19256
15. Philipp Moritz et al. *Ray: A Distributed Framework for Emerging AI Applications*. arXiv, 2017. https://arxiv.org/abs/1712.05889
16. An Yang et al. *Qwen3 Technical Report*. arXiv, 2025. https://arxiv.org/abs/2505.09388

## C. 树结构搜索、树结构 RL 与 Agent 相关工作

17. Ziyu Wan et al. *AlphaZero-Like Tree-Search can Guide Large Language Model Decoding and Training*. ICML 2024. https://proceedings.mlr.press/v235/wan24c.html
18. Zhenyu Hou, Ziniu Hu, Yujiang Li, Rui Lu, Jie Tang, and Yuxiao Dong. *TreeRL: LLM Reinforcement Learning with On-Policy Tree Search*. ACL 2025. https://aclanthology.org/2025.acl-long.604/
19. Yizhi Li et al. *TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling*. arXiv, 2025. https://arxiv.org/abs/2508.17445
20. Wanxin Tian et al. *SEEA-R1: Tree-Structured Reinforcement Fine-Tuning for Self-Evolving Embodied Agents*. arXiv, 2025. https://arxiv.org/abs/2506.21669
21. Xinyu Guan et al. *rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking*. ICML 2025. https://proceedings.mlr.press/v267/guan25f.html
22. Guanting Dong et al. *Agentic Reinforced Policy Optimization*. arXiv, 2025. https://arxiv.org/abs/2507.19849
23. Yuxiang Ji et al. *Tree Search for LLM Agent Reinforcement Learning*. arXiv, 2025. https://arxiv.org/abs/2509.21240

## D. 过程监督与 Test-Time Scaling

24. Hunter Lightman et al. *Let's Verify Step by Step*. ICLR 2024. https://openreview.net/forum?id=v8L0pN6EOi
25. Peiyi Wang et al. *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*. ACL 2024. https://aclanthology.org/2024.acl-long.510/
26. Liangchen Luo et al. *Improve Mathematical Reasoning in Language Models by Automated Process Supervision*. arXiv, 2024. https://arxiv.org/abs/2406.06592
27. MiniMax et al. *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*. arXiv, 2025. https://arxiv.org/abs/2506.13585

## E. 实验平台、数据集与工程基础

28. Shengyi Huang, Rousslan Fernand Julien Dossa, Chang Ye, and Jeff Braga. *CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms*. arXiv, 2021. https://arxiv.org/abs/2111.08819
29. Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. *The Arcade Learning Environment: An Evaluation Platform for General Agents*. *Journal of Artificial Intelligence Research*, 2013. https://arxiv.org/abs/1207.4708
30. Emanuel Todorov, Tom Erez, and Yuval Tassa. *MuJoCo: A physics engine for model-based control*. IROS 2012. https://doi.org/10.1109/IROS.2012.6386109
31. Dan Hendrycks et al. *Measuring Mathematical Problem Solving With the MATH Dataset*. arXiv, 2021. https://arxiv.org/abs/2103.03874
32. OpenAI. `prm800k`. GitHub repository accompanying *Let's Verify Step by Step*. https://github.com/openai/prm800k
33. `hiyouga/math12k`. Hugging Face dataset. https://huggingface.co/datasets/hiyouga/math12k
34. `nlile/NuminaMath-1.5-RL-Verifiable`. Hugging Face dataset. https://huggingface.co/datasets/nlile/NuminaMath-1.5-RL-Verifiable
35. `math-ai/minervamath`. Hugging Face dataset. https://huggingface.co/datasets/math-ai/minervamath
36. `math-ai/amc23`. Hugging Face dataset. https://huggingface.co/datasets/math-ai/amc23
37. `math-ai/aime25`. Hugging Face dataset. https://huggingface.co/datasets/math-ai/aime25
38. `Qwen/Qwen3-1.7B`. Hugging Face model card. https://huggingface.co/Qwen/Qwen3-1.7B
