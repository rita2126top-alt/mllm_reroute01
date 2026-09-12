# 全量实验入口验证记录

日期：2026-09-12。本记录对应在 `9daee69` 版本之上新增的完整正式实验总脚本及全量指南。

## 实际检查结果

| 检查 | 实际结果 |
|---|---|
| 原 81 文件 SHA256 | 全部保持一致 |
| Python AST | 45 个文件通过 |
| 本次 Python 的目标 3.10 语法 | 通过 |
| Bash -n | 9 个脚本通过 |
| 新增项目 Python CLI --help | 12 个入口通过 |
| 全量指南 Bash 代码块 | 21 段全部通过语法检查，实验命令无缩短参数 |
| 完整 CPU pytest | **85 passed，2 条第三方 SWIG 弃用警告，无失败** |
| 正式计划实际生成 | 247 个调度项；生成计划未加载模型/数据，也未创建实验结果根目录 |
| 真实原 lmms-eval 评分提取器离线观察 | 恢复成功的重试正常接受；实际随机回退准确识别 |

执行环境为本地 Windows、Python 3.12.14、torch 2.11.0+cpu、transformers 5.4.0、lmms-eval v0.7.1 源码 editable 安装。远程目标环境仍是 Linux、Python 3.10、CUDA 12.8；没有实际运行用户远程 GPU。

## 完整计划核对

| 阶段 | 计划项 |
|---|---|
| train | 1 个训练矩阵任务，内部串行 60 checkpoint |
| accuracy | 62 设置 × 12 原子任务，744 个设置/任务组合 |
| profile | 62 设置，3 图 × 1 pass、0 warmup |
| benchmark | 62 设置，3 图 × 5 passes、2 warmup、64 tokens |
| ablation | 48 compact 设置 × 4 完整任务，192 个设置/任务组合 |
| diagnostics | 12 设置 × 512 张内部验证图片 |

247 个上层调度项对应 60 次训练 + 246 次实验设置执行，即 306 个底层训练/实验执行项。数据下载/准备不计入这些数量。

## 行为验证

新增总调度测试覆盖：完整计划枚举；串行执行；失败后保留旧 attempt 并创建新 attempt；已完成项验证后跳过；结果、checkpoint、manifest、配置内容变更使旧完成记录失效；训练失败阻止依赖阶段；单独评测已有权重；正式样本数完整性；原效率样本 ID 与次数；judge 预检和失败标记；GPU 选择传递；同目录操作系统排他锁和自动释放。

数据准备另做 10 项无网络模拟检查：首次正式清单生成、重复保留原 bytes/mtime、不同 hash 拒绝覆盖、多个训练问题文件拒绝、显式合法/非法路径、模型下载失败、排除导出失败、清单生成失败、错误图像目录。没有下载实际模型或数据。

使用真实 `full_env.sh` 的 shell 模拟验证了：默认 all、all --resume、prepare、各单阶段参数；Conda 前提检查；非法和缩短运行参数拒绝；Python 版本/judge/doctor/子进程失败码传播；包含空格的目录。

MMBench 观察器保持原评分函数返回值不变，在原函数丢弃随机回退原因之前读取该原因。原结果会保留，但最终随机回退使正式评测非零退出；中途失败后恢复成功不会误判。实际提取器验证仅使用本地响应替身，没有发送 API 请求，也没有输出凭据。

## 本地报告位置

运行 `python scripts/cold_ghost/check_project.py` 可复现项目级检查，其报告写入 `experiments/cold_ghost/checks/report.json`。本次还有：

- `experiments/cold_ghost/checks/full_runbook_audit.json`
- `experiments/cold_ghost/checks/prepare_full_data_audit.json`

实验和测试临时产物保持 Git 忽略，本文记录实际验证结论。数据脚本不改原项目，已有训练数据身份不被静默替换。

## 尚未执行

没有运行真实 60 个 7B checkpoint 训练、完整 benchmark、正式 judge API 请求或 GPU 性能测量。因此这里不报告模型精度、吞吐、显存或方法提升。服务器实际全量执行命令见 [完整全量运行指南](COLD_GHOST_全量运行指南.md)。
