# MLLM-Reroute + Cold/Ghost

在原 MLLM-Reroute 上增加 skip token 状态更新。原 81 个文件完整保留；decision layer、原评分、Full top-k、预算与评测任务沿用原项目。Cold 保持跳过状态，Ghost 用低维残差预测更新部分 skip token。

**从这里开始：[远程服务器 Conda 从零运行指南](docs/COLD_GHOST_从零运行指南.md)。**

- [最终方法与代码对应](docs/COLD_GHOST_METHOD.md)
- [交付验证记录](docs/COLD_GHOST_VALIDATION.md)
- [原项目 README](README.md)
- [先前完整方案文本](docs/MLLM-Reroute-Ghost-Cold-完整方案.md)

```text
cold_ghost/                   # 方法、训练、数据、评测、效率和诊断
scripts/cold_ghost/           # 安装、检查、单项与矩阵命令
environments/cold_ghost/      # Conda、约束、原文件 SHA256
tests/cold_ghost/             # CPU 实际微型模型和行为验证
```

支持 38 个原设置和 24 个 GC 设置，覆盖 POPE、GQA、MMBench、MME、8 个 grounding 条目和 TFLOPs/KV/运行时间。12 个 full checkpoint 对应两种 backbone、两套原 Reroute 决策方案、三个预算；compact/stagewise 共用同设置权重。另含 full/self/context/score/no_fresh 固定消融和三类内部诊断。

GC 需要按指南训练辅助参数。正式评测拒绝随机、部分或 smoke checkpoint。交付已完成 CPU 真实微型模型测试，未在远程 GPU 上训练 7B，也未提供训练完成权重或虚构 benchmark 分数。
