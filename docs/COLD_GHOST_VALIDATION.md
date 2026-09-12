# Cold/Ghost 交付验证记录

验证日期：2026-09-12。以下为实际执行结果，不是预测的 benchmark 表现。

## 1. 原代码完整性

原始代码已先上传到同一仓库的 main 分支：

- 基线提交：`8e3e48d40d65e9b7d09a4c2682dd374f243b1c21`。
- 原文件数量：81。
- SHA256 清单：`environments/cold_ghost/original_files_sha256.json`。
- 完整性检查结果：81 个文件内容全部一致，无删除、无修改。
- 新方法、训练、评测、检查、环境和文档全部以新增文件交付。

新增安装脚本会把原 grounding 补丁应用到独立安装的 lmms-eval；它不修改本仓库原文件。

## 2. 实际测试环境

本地 Windows x86_64、Python 3.12.14、torch 2.11.0+cpu、torchvision 0.26.0+cpu、transformers 5.4.0、Hydra 1.3.2、pytest 8.4.2。CPU wheel 与远程指南的 CUDA wheel 分开管理，核心版本一致。

服务器目标环境是 Linux、Python 3.10、CUDA 12.8。没有登录用户的远程服务器，也没有执行真实 7B GPU 训练或全量评测。

## 3. 完整检查

实际运行：

```bash
python scripts/cold_ghost/check_project.py
```

检查报告：

| 检查 | 结果 |
|---|---|
| 原 81 文件 SHA256 | 全部一致 |
| Python AST 语法，含原 models/scripts/profiler 与新增模块/测试 | 40 文件通过 |
| 新增 Python 的目标 3.10 语法解析 | 33 文件通过 |
| Bash -n，含原 shell 与新增安装脚本 | 6 文件通过 |
| 新增 CLI --help | 11 入口通过 |
| 运行指南 Bash 代码块 | 32 段 bash -n 通过 |
| 实际 CLI dry-run | 训练 12/60 项、准确率 62 项通过 |
| 直接调用 doctor，实际导入 torch/torchvision/transformers 与新增方法 | 通过 |
| CPU pytest | **66 passed**，2 条第三方 SWIG 弃用警告，无失败 |

完整检查默认输出 `experiments/cold_ghost/checks/report.json`。报告含各命令 stdout/stderr、退出码和完整性结果，实验目录不加入源码版本。

测试期间设定 `CUDA_VISIBLE_DEVICES=""`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。模型测试使用 transformers 的真实随机微型 LLaVA 与 Qwen2.5-VL 实例，不下载模型、不生成虚构 benchmark 分数。

## 4. 运行行为覆盖

- 原 router 的 decision layer、score、top-k、预算和 38 配置的矩阵枚举。
- 24 GC 设置、12 个 full checkpoint、60 个全消融训练计划；stagewise 不重复训练。
- compact 与 stagewise 的视觉状态、Full 选择和模型输出一致性。
- Ghost 为零更新时与原 Reroute 的一致性。
- Ghost/Cold 分类、固定预算、末次 decision 后全 Cold、age Full 清零和 Ghost 递增。
- Qwen MRoPE、3D position_ids/4D position embeddings、可变视觉 token 数及新 prefill 重置。
- 实际 forward、两步 cached decode 和 generate。
- 激活检查点中的梯度传播、warmup identity、rollout、冻结 backbone 不更新而 Ghost 更新。
- 四项损失、未来标签、freshness 监督、teacher 捕获与防止答案输入。
- checkpoint 配置/hash/完成状态校验、部分恢复、优化器/RNG 恢复及错误权重拒绝。
- 数据固定划分、排除清单、内容去重、内部验证分离和缓存恢复。
- 真实原 `run_lmms_eval` API 集成：测试中仅替代外部模型工厂和 evaluator，检查原模型参数、Qwen max_pixels/SDPA、12 任务、prompt/bbox preset 与 generation 默认传递。
- 矩阵失败汇总、非零退出、缺失/损坏结果处理、训练完成项跳过和部分项续训。
- FLOPs 计数覆盖融合 SDPA：CPU 已知形状算例实际得到 960 FLOPs；Ghost 子项不重复计入总量。
- 直接 CLI 进程导入：修复并回归验证 `profile.py` 遮蔽 Python 标准库的问题。

## 5. 依赖解析检查

原 Hydra 使用 antlr 4.9.x，而较新的 legacy latex2sympy2 固定 antlr 4.7.2，会导致完整评测框架安装冲突。新增约束固定：

```text
antlr4-python3-runtime==4.9.3
latex2sympy2==1.5.4
math-verify==0.9.0
latex2sympy2-extended==1.11.0
```

已根据官方包元数据核对版本范围，实际执行小组合安装、Hydra compose、数学解析器调用、pip check，并完成 lmms-eval 0.7.1 的完整依赖 dry-run。该解析在本地 Python 3.12 CPU 环境执行，不等同于 Linux/CUDA 实装。

随后已在同一 CPU 测试环境实际安装完整评测依赖，并按服务器方案从官方 v0.7.1 标签（commit `88b23e2bfa16a1edbc16e9e238ed82130b3a4f56`）editable 安装 lmms-eval，仅对独立测试环境应用原三个补丁。在 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` 下实际执行 `doctor --require-eval`，结果 `ok=true, errors=[]`：真实 evaluator、LLaVA/Qwen 两个模型工厂及 12 个任务定义全部通过，`pip check` 无依赖冲突。没有实例化 7B 模型或加载数据集。

核验中发现 PyPI wheel 缺少部分无扩展名任务模板；指南固定使用官方 Git 标签源码 editable 安装，已验证该路径正常。

新增 `doctor --require-eval` 额外检查实际 evaluator 和两个模型工厂导入，不实例化模型、不下载数据；服务器安装最后执行该检查。`doctor --require-gpu` 在服务器实际执行 CUDA 张量并记录硬件信息。

## 6. 未执行的部分

未训练真实 LLaVA-1.5-7B/Qwen2.5-VL-7B Ghost 权重；未运行全部真实评测数据；未测量服务器的 GPU TFLOPs、延迟或峰值显存；未报告训练收益。

这些结果需按 [从零运行指南](COLD_GHOST_从零运行指南.md) 在服务器生成。正式推理会拒绝 smoke、部分训练和不匹配的权重。交付提供实现、数据准备、训练、实验和验收入口，不把 CPU 测试通过表述为全量 GPU 实验已完成。
