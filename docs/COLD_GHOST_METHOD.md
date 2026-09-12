# MLLM-Reroute-GC：已实现的方法与实验约束

本文对应仓库新增的 `cold_ghost/` 与 `scripts/cold_ghost/` 实现。原 MLLM-Reroute 的代码、38 个实验配置和原实验入口保留。新的默认方法是 **Full + Ghost + Cold**；Ghost 只更新原路由已经跳过的视觉 token。

这是一份实现说明。真实 7B backbone 的训练收益、benchmark 精度、GPU 延迟和峰值显存需要运行训练及实验后取得，本文不预设性能提升。

## 1. 原路由具体保留什么

所有 decoder layer index 使用 **0-based**。`N` 是本样本进入语言模型的原始视觉 token 数；Qwen 的 `N` 随原 processor 的图像处理变化。

| 原属性 | 实际实现 |
|---|---|
| Importance | 调用原 `models/patching.py` 的 attention scoring：原 normalization、Q/K projection、RoPE/MRoPE，取最后一个 prompt token 对视觉区域的 attention，并在 heads 上平均 |
| Full Top-k | 调用原 `models/router.py`；保留原 Top-k 的选中规则与原始位置顺序 |
| Full 数量 | `min(N, max(1, floor(N × keep_ratio)))`；比例始终相对原始 `N` |
| 非单调重选 | 原 `monotonic=false`，在下一次原定 decision layer 从全部视觉位置重新选择 |
| 阶段内部 | 沿用本阶段缓存的 Full 集合和 scores，不增加 decision layer |
| 非视觉位置 | 始终进入 Full branch，不参与 Ghost/Cold 分类 |
| Backbone | LLaVA-1.5-7B 和 Qwen2.5-VL-7B，原 processor、SDPA 与原加载精度 |
| 评测 | 原 lmms-eval v0.7.1 任务、split、prompt、指标及生成默认值 |

| 原 schedule | LLaVA | Qwen2.5-VL |
|---|---|---|
| `ours_vs_fastv` | `[3,7,15,23]` | `[3,6,13,20]` |
| `ours_vs_pdrop` | `[2,7,15,23]` | `[2,6,13,20]` |

| 档位 | `ours_vs_fastv` ratios | `ours_vs_pdrop` ratios |
|---|---|---|
| `avg192` | `[0.2644,0.2644,0.2644,0.2644]` | `[0.4502,0.4502,0.2251,0.1126]` |
| `avg128` | `[0.1418,0.1418,0.1418,0.1418]` | `[0.2655,0.2655,0.1328,0.0664]` |
| `avg64` | `[0.01916,0.01916,0.01916,0.01916]` | `[0.1392,0.1392,0.0696,0.0348]` |

更新后的 hidden states 会影响后续原 scoring，所以新旧方法实际选中的 token 身份可以不同。保持固定的是 scoring 算法、决策层、Full Top-k 规则和预算。

## 2. 三类视觉 token 与执行顺序

令第 `l` 层的原 Full 集合为 \(\mathcal A_l\)，其补集为 \(\mathcal U_l\)。新增 selector 只在 \(\mathcal U_l\) 中选出 \(\mathcal G_l\)，剩余为 \(\mathcal C_l\)。

\[
\mathcal G_l\subseteq\mathcal U_l,
\qquad \mathcal C_l=\mathcal U_l\setminus\mathcal G_l.
\]

先 gather Full visual + 全部非视觉位置，执行冻结的原 decoder block，再获得本层实际 compact Active residual：

\[
d_j^l=v_j^{l+1}-v_j^l,\qquad j\in\mathcal A_l.
\]

随后更新 Ghost，Cold 使用 identity bypass：

\[
v_i^{l+1}=\begin{cases}
\left[F_l(\operatorname{Gather}(X^l,\mathcal K_l))\right]_i,&i\in\mathcal A_l,\\
v_i^l+\widehat\Delta_i^l,&i\in\mathcal G_l,\\
v_i^l,&i\in\mathcal C_l.
\end{cases}
\]

因此 Ghost 不追溯改变当前层已经完成的 Full outputs；它影响后续状态及下一次原决策。

`compact_route` 每层使用可微分 `index_copy` 写回完整状态。`compact_route_stagewise` 保持原阶段内部的 compact tensor 传递方式，将 Ghost 写入当前 deferred buffer；到下一 decision layer 才重建完整状态。两个执行版本共享同一 checkpoint。

## 3. Ghost 的选择：预测下一次原决策

下一次决策固定为：

\[
d^+(l)=\min\{d\in\mathcal D:d>l\}.
\]

有下一次决策时，每层最多选择 128 个 Ghost：

\[
K_G^l=\min(128,|\mathcal U_l|).
\]

无下一次决策时，`K_G=0`；最后一个 decision layer 完成 Full 选择后，全部 skip token 固定为 Cold，不再执行 Ghost selector 或 residual predictor。

设当前阶段的 Full 最低选中 score 为 \(\tau_l\)，selector 对每个 skip token 使用：

\[
z_i^l=W_{\rm down}^l\operatorname{LN}_0(v_i^l),
\]

\[
x_i^{\rm react}=\left[z_i^l;
\frac{s_i^l}{\tau_l+\epsilon};
\frac{s_i^l-\tau_l}{\tau_l+\epsilon};
\frac{\min(a_i^l,8)}8;
\frac{l}{L-1};
\frac{d^+(l)-l}{L-1}\right],
\]

\[
\ell_i^l=(w_{\rm react}^l)^\top x_i^{\rm react}+b_{\rm react}^l,
\qquad r_i^l=\sigma(\ell_i^l).
\]

实现直接按 logits \(\ell_i^l\) 排序；sigmoid 单调，因此它表示相同的概率优先级，并避免先计算 sigmoid 的饱和舍入。hard indices 停止梯度，同 logits 按原视觉索引升序。Ghost/Cold 每层可以重新划分，Full 集合仍只在原 decision layers 更新。

Age 记录进入本层前连续缺少 Full computation 的层数：

\[
a_i^0=0,\qquad
a_i^{l+1}=\begin{cases}0,&i\in\mathcal A_l,\\a_i^l+1,&i\in\mathcal U_l.\end{cases}
\]

Ghost 同样属于 skip，更新其数值不会将 age 清零。真实 age 使用未截断整数；只有网络输入的 age 编码截断为 8。

## 4. Self、Context 与 gate

固定 bottleneck 为 32，prototype 数为 8。新增无 affine LayerNorm 的 `epsilon=1e-6`，不会替换原 backbone normalization。

Self residual：

\[
\widehat\Delta_{i,\rm self}^l=W_{\rm up}^l\operatorname{SiLU}(z_i^l).
\]

Context 只读取当前 student 的 Active 输入与实际 compact residual：

\[
c_j^l=W_R^l d_j^l,\qquad
k_j^{P,l}=W_K^{P,l}\operatorname{LN}_0(v_j^l),
\]

\[
\beta_{mj}^l=\operatorname{softmax}_{j\in\mathcal A_l}
\left(\frac{(q_m^{P,l})^\top k_j^{P,l}}{\sqrt{32}}\right),
\qquad p_m^l=\sum_{j\in\mathcal A_l}\beta_{mj}^l c_j^l.
\]

由此得到本样本、本层的 \(8\times32\) prototype bank。Ghost 使用自己的 query 读取这些 prototypes：

\[
q_i^{G,l}=W_Q^{G,l}\operatorname{LN}_0(v_i^l),\quad
k_m^{G,l}=W_K^{G,l}p_m^l,\quad
u_m^{G,l}=W_V^{G,l}p_m^l,
\]

\[
\alpha_{im}^l=\operatorname{softmax}_m
\left(\frac{(q_i^{G,l})^\top k_m^{G,l}}{\sqrt{32}}\right),
\quad h_{i,\rm ctx}^l=\sum_m\alpha_{im}^l u_m^{G,l},
\quad\widehat\Delta_{i,\rm ctx}^l=W_{\rm ctx}^l h_{i,\rm ctx}^l.
\]

最后使用统一 gate：

\[
g_i^l=\sigma\left((w_g^l)^\top[z_i^l;h_{i,\rm ctx}^l;\widetilde a_i^l]+b_g^l\right),
\]

\[
\widehat\Delta_i^l=g_i^l
\left(\widehat\Delta_{i,\rm self}^l+\widehat\Delta_{i,\rm ctx}^l\right).
\]

每连续四层共享同组新增参数，组编号为 \(\lfloor l/4\rfloor\)。动态 prototype bank、scores、age、hidden states 不跨层或跨样本共享。没有 Ghost 执行机会的参数组不加入优化器。

所有新增参数和投影计算保持 FP32，写回 hidden 时转换为原 backbone dtype。两套升维权重使用标准差 `1e-4` 的正态初始化；prototype queries 的标准差为 `0.02`；其余矩阵使用 Xavier uniform；gate 权重为 0、bias 为 -2；reactivation 权重和 bias 为 0。

## 5. 数据选择与排除评测图像

正式训练只读取 GQA `train_balanced_questions.json`，每张图选择数值 question ID 最小的问题。答案不进入训练 manifest，也不进入 decoder 输入。

训练数据准备执行以下确定规则：

1. 用原 lmms-eval v0.7.1 任务加载器读取原评测任务实际使用的 test/validation split，经 `doc_to_visual` 获取图片；加入原 bundled efficiency cohort 的三张图片。
2. 对经过 EXIF 方向校正的 RGB 图像，以尺寸和像素内容计算 SHA256；保存各任务 scope 的排除哈希。自动导出支持按 dataset fingerprint 和 document cursor 恢复，不通过中途进度标记完成整个导出。
3. 对 GQA 每个 image ID 确定最小 question ID，将 image ID 按 `SHA256("42:" + image_id)` 升序排列。
4. 按此顺序剔除与评测排除集内容相同的图片，以及已保留图片的重复内容。相同内容的不同 image ID 以该顺序中第一个为代表。
5. 前 8192 张不同内容的图片作为训练集，接着 512 张作为内部验证集；不足 8704 张时报错。

排除 scope 完整包含 POPE、GQA、MMBench、MME、RefCOCO、RefCOCO+、RefCOCOg 和 efficiency。RefCOCO/RefCOCO+ 覆盖 val/testA/testB，RefCOCOg 覆盖 val/test。

这里的内容去重是**完全相同的解码 RGB 像素及尺寸**检测，无法保证识别重新缩放、裁剪或有损重编码后的近重复图像。manifest 同时记录图片文件哈希、像素哈希、原问题文件哈希、排除 manifest 哈希和最终 manifest 哈希。读取训练/验证图像时会核验文件是否改变。

两个 backbone 和所有消融共用同一份正式 manifest。训练 prompt 使用 GQA 原后缀 `\nAnswer the question using a single word or phrase.`，再经对应 backbone 原 processor/chat template 构造。teacher 和 student 使用同一份已处理图片与 prompt tensors。

## 6. Dense teacher 与四项监督

Teacher 使用**同一个冻结 backbone 的原始 dense forward**。每个样本先临时恢复原 decoder forward，停止梯度收集视觉输入与 dense residual，然后恢复 Ghost patch 执行 student。不会加载第二套 7B teacher 权重。

\[
\Delta_i^{l,*}=v_i^{l+1,*}-v_i^{l,*}.
\]

这监督 dense 轨迹上的单层变化，不把过时 student 到下一层 teacher 的总差值冒充单层 residual。teacher 只提供状态监督，不提供 student 的 Full Top-k 标签。

Residual loss 对实际 Ghost token 事件按 hidden dimension 取 MSE，再按 token 事件数平均：

\[
\mathcal L_\Delta=\frac1{\max(1,Z_G)}\sum_l\sum_{i\in\mathcal G_l}
\frac1D\|\widehat\Delta_i^l-\Delta_i^{l,*}\|_2^2.
\]

Direction loss 对 teacher residual 范数至少为 `1e-6` 的 Ghost 事件计算 \(1-\cos(\widehat\Delta,\Delta^*)\)。范数下界为 `1e-6`，点积、范数与 reduction 全部为 FP32。

Reactivation 标签在**同一次 student forward 完成后**从下一次实际原路由决策产生：

\[
y_i^{l,\rm react}=\mathbf1[i\in\mathcal A_{d^+(l)}],\qquad i\in\mathcal U_l.
\]

对全部 skip 候选计算 BCE-with-logits，并按候选事件数平均。标签与 hard Top-k 停止梯度，标签不参与本层 Ghost 选择。

Freshness 监督实际 skip→Full 事件，并读取**重新激活层执行 Full computation 之前的输入**：

\[
\mathcal R=\{(i,l_r):l_r\in\mathcal D,
i\notin\mathcal A_{l_r-1},i\in\mathcal A_{l_r}\},
\]

\[
\mathcal L_{\rm fresh}=\frac1{\max(1,|\mathcal R|)}
\sum_{(i,l_r)\in\mathcal R}\frac1D\|v_i^{l_r}-v_i^{l_r,*}\|_2^2.
\]

Freshness 可经冻结 Full layers 和之前的 Ghost 更新反向传播。冻结原参数不会对整个 student 使用 `no_grad()`。各项空集合返回零；没有真实重激活事件就没有 freshness 监督。

## 7. 两个连续训练阶段

| 阶段 | 状态轨迹 | Loss | 轮数 |
|---|---|---|---|
| Warmup | `apply_updates=False`，Ghost 产生预测但不写回；后续层始终读取原 identity-bypass Reroute 轨迹 | \(\mathcal L_\Delta+0.1\mathcal L_{\rm dir}+0.1\mathcal L_{\rm react}\) | 1 epoch |
| Rollout | `apply_updates=True`，Ghost 实际写回并影响后续层及未来原路由 | \(\mathcal L_\Delta+0.1\mathcal L_{\rm dir}+0.1\mathcal L_{\rm react}+\mathcal L_{\rm fresh}\) | 1 epoch |

当前实现按样本在线生成 teacher 和 student 轨迹；warmup 在状态层面等价于先记录原 Reroute 轨迹再做单层拟合，不要求用户预存整套多层 hidden states。

固定使用 AdamW、学习率 `1e-4`、betas `(0.9,0.999)`、optimizer epsilon `1e-8`、weight decay `0.01`、micro batch 1、梯度累积 16、梯度裁剪范数 1.0。矩阵参数使用 decay；prototype queries、gate/reactivation 向量 heads 和 bias 不 decay。两个阶段连续使用同一 optimizer state 和恒定学习率。

初始化 seed 为 42。warmup 和 rollout 分别使用 42、43 对训练 manifest 做确定性打乱。模型训练始终使用 `compact_route`、`use_cache=False`；原 backbone 保持 eval，新增模块处于 train。

当前只对冻结的原 decoder 计算启用 activation checkpointing，使用 `use_reentrant=False`；Ghost predictor 保持普通可微分执行。激活重算与 routing/age/state 管理分离。不调用会重算整个有状态 patched forward 的 backbone `gradient_checkpointing_enable()`。训练状态写回采用可微分的新 tensor，不原地覆盖反传需要的旧 hidden state。

每阶段结束后在独立 512 样本内部验证集报告四项 loss。交付第二阶段结束的权重，不根据 benchmark 或验证 loss 选择“最佳 epoch”。

## 8. 固定消融

| 名称 | 实际变化 |
|---|---|
| 原 Reroute | skip 全部 identity bypass，使用原实验入口 |
| `self` | learned Ghost selector + Self + gate；Context code 置零，Context 模块冻结且不执行 |
| `context` | learned selector + Context + gate；保留 selector/gate 所需 `z`，Self 升维模块冻结且不执行 |
| `score` | skip 按原 cached importance 排序选 Ghost，同分按原位置；保留 Self、Context、gate，reactivation head 冻结，不计算 BCE |
| `no_fresh` | 完整模型，freshness loss 固定为零 |
| `full` | 完整 selector、Self、Context、gate 和全部适用监督 |

Warmup 的 freshness 对所有实验均关闭。`self` 和 `context` 的 rollout 使用全部适用四项 loss；`score` 使用 residual、direction、freshness。各行独立训练，数据、预算、轮数、初始化种子保持一致。最终默认方法为 `full`。

## 9. Checkpoint 与正式评测

完整方法需要 `2 backbones × 2 schedules × 3 tiers = 12` 个 checkpoint。每个文件对应一个模型、schedule 和 tier，同时服务对应 compact/stagewise 两种执行形式。五种新增训练消融共计 60 个 checkpoint。

文件名为 `{model}__{schedule}__{tier}__{ablation}.pt`，例如 `llava15__ours_vs_fastv__avg192__full.pt`。

Checkpoint 保存 Ghost `state_dict`、optimizer state、stage/cursor、随机数状态及元数据，不保存原 backbone 权重。元数据包含 backbone ID、hidden size、decoder 层数、decision layers、keep ratios、Ghost 参数、ablation 和训练 manifest SHA256。

断点仅在完成梯度累积的边界保存，没有未保存的 pending gradients。Resume 要求实验、manifest、消融、累积设置和 smoke 标记一致，恢复同一 optimizer 与样本顺序。

`--smoke` 与 `--steps-per-stage` 用于执行检查。截断运行或 smoke 权重不能通过正式评测加载校验。正式矩阵禁止随机初始化 Ghost 后直接报告成绩。

## 10. 验证与诊断边界

CPU 测试使用本地随机初始化的真实 tiny LLaVA/Qwen Transformers 模型，不下载预训练权重。覆盖原 Reroute 等价性、compact/stagewise 一致性、Cold 不变、Ghost 写回、下一层可见状态、新样本重置、decode 不更新 Ghost、跨层梯度、两阶段训练、teacher 恢复、checkpoint 恢复和数据哈希校验。

内部验证诊断报告：

- 原 Reroute 与完整方法在实际重激活输入上的 MSE/cosine error，按 Full skip age 分组；共同事件另做配对，配对两侧统一按原 Reroute age 分组。
- Dense residual 在真实 Active IDs 所张成的 rank 8/32 子空间上的重构误差，并注明有效 rank；另外比较独立训练的 Self 与完整方法在真实部署轨迹上的 Ghost residual error。
- 下一次会/不会重激活的 skip 状态误差，Ghost selection precision/recall，以及含同分处理的 reactivation average precision；无正样本时 AP/recall 记为 `null`。

当前方法范围是单图、无 padding、batch size 1 的完整 image/question prompt prefill。新样本清空路由与辅助状态；自回归 decode 沿用原 KV 路径，不为历史 Ghost/Cold 补写 KV。Context 聚合当前 Active 视觉 residual，不将答案 token 作为上下文来源，也不声称与原 causal decoder 的逐位置信息依赖完全等价。

Full keep schedule 相同不代表新增方法总 FLOPs 相同。Ghost projections、prototype 运算、selector、gate、gather/scatter 和辅助状态有额外开销。真实精度与效率结论应由原任务矩阵及对应 profile/runtime 输出共同判断。
