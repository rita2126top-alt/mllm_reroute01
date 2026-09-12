下面给出基于当前 **MLLM-Reroute 项目** 的完整修改方案，方法命名为：

$$
\boxed{\textbf{MLLM-Reroute-GC：Reactivation-Aware Ghost Evolution}}
$$

其中 GC 表示 **Ghost / Cold**。

方案保留项目原有的 **decision layers、attention importance、Full-token Top-k、keep ratios、非单调重选机制、模型和评测设置**。新增方法只处理已经被原路由判定为 skip 的视觉 token：一部分执行 Ghost 状态更新，其余执行 Cold identity bypass。

下面沿用原文的编号、分节、公式和逐步解释方式。每一个参数第一次出现，说明：

$$
\boxed{
\text{它表示什么}
\quad+\quad
\text{维度是多少}
\quad+\quad
\text{是不是可训练的}
\quad+\quad
\text{为什么需要它}
}
$$

本文是一套确定的新增方法设计。涉及训练、性能提升和误差下降的内容均为待实现、待验证的方案，不是当前仓库已经完成的功能或实验结果。

项目行为已按本地 [router.py](D:/mllm_compression_code/mllm-reroute-main/models/router.py)、[patching.py](D:/mllm_compression_code/mllm-reroute-main/models/patching.py) 和 [run_paper_table.sh](D:/mllm_compression_code/mllm-reroute-main/scripts/run_paper_table.sh) 核对。

---

# 1. 先不要设计模块，先把问题说透

假设语言模型部分一共有：

$$
\boxed{L=\text{LLM decoder 的总层数}}
$$

个 Transformer layers。

本文统一使用与项目代码一致的 **从 0 开始的 layer index**：

$$
l\in\{0,\ldots,L-1\}.
$$

因此配置中的 layer index 3 表示第 4 个 decoder block。

项目的两个模型分别是：

$$
\boxed{
\begin{aligned}
\text{LLaVA-1.5-7B}:&\quad L=32,\ D=4096,\\
\text{Qwen2.5-VL-7B}:&\quad L=28,\ D=3584.
\end{aligned}
}
$$

这里：

$$
\boxed{D=\text{语言模型 hidden dimension}}
$$

是固定模型属性，不是新增超参数。Qwen 的维度和层数对应其[官方模型配置](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)；实际实现从所加载模型的配置读取。

图片经过原视觉编码器和 projector / merger 后得到：

$$
\boxed{N=\text{当前样本输入语言模型的视觉 token 数}}
$$

个 visual tokens。

LLaVA-1.5 在项目常规单图设置下为：

$$
N=576.
$$

Qwen 沿用项目的：

$$
\boxed{\texttt{max\_pixels}=451584}
$$

以及原 processor，实际 \(N\) 随图像预处理结果变化，不能把每张图都写成固定 576。

第 \(l\) 层输入时，第 \(i\) 个 visual token 的 hidden state 定义为：

$$
\boxed{v_i^l\in\mathbb R^D},
\qquad i\in\{1,\ldots,N\}.
$$

执行第 \(l\) 层后得到 \(v_i^{l+1}\)。因此 hidden-state 的深度下标范围为 \(0,\ldots,L\)。

---

# 2. MLLM-Reroute 在第 \(l\) 层干了什么？

首先定义：

$$
\boxed{\mathcal D=\{d_0,\ldots,d_{S-1}\}}
$$

其中：

$$
\boxed{
\mathcal D=\text{原实验配置指定的 decision-layer 索引集合},
\quad S=|\mathcal D|
}
$$

它们都是原配置的固定属性，不参与训练。

MLLM-Reroute **只在 \(l\in\mathcal D\) 时产生新选择**。

具体来说，项目在当前 decision layer 的输入上调用原打分流程：

$$
\text{原 input norm}
\rightarrow Q/K\text{ projection}
\rightarrow\text{RoPE/MRoPE}
\rightarrow QK^\top
\rightarrow\text{softmax}.
$$

定义：

$$
\boxed{
s_i^{d_s}
=\frac1{n_h}\sum_{h=1}^{n_h}
\operatorname{Attn}^{d_s}_{h,q_{\rm last},\,\operatorname{pos}(i)}
}
$$

这里：

$$
\boxed{
\begin{aligned}
n_h&=\text{原模型 attention head 数，固定属性},\\
q_{\rm last}&=\text{完整输入序列的最后一个 token 位置，动态索引},\\
\operatorname{pos}(i)&=\text{视觉 token }i\text{ 的原始序列位置，动态整数},\\
s_i^{d_s}&\in\mathbb R=\text{该决策层的视觉 importance score}.
\end{aligned}
}
$$

这与项目一致：取**最后一个序列 token 对视觉区域的 attention，并在 heads 上平均**。

然后读取原配置该阶段的 keep ratio：

$$
\boxed{\rho_s=\text{原 YAML 第 }s\text{ 阶段的 keep ratio}}.
$$

\(\rho_s\) 是固定超参数，不重新学习或重新搜索。

Full token 数定义为：

$$
\boxed{
K_F^{d_s}
=\min\left(N,\max\left(1,\lfloor N\rho_s\rfloor\right)\right)
}.
$$

这里 \(N\) 是该样本最初的视觉 token 数。Reroute 保留原始视觉位置，所以不会逐阶段把比例乘到上一次剩余数量上。

选中集合为：

$$
\boxed{
\mathcal A_{d_s}
=\operatorname{TopK}
\left(\{s_i^{d_s}\}_{i=1}^{N},K_F^{d_s}\right)
}.
$$

使用项目原来的 \(\texttt{scores.topk}\)，然后将选中索引按原始 token 位置升序排列。原 Full Top-k 的同分处理也保持原实现。

在两个 decision layers 之间：

$$
\boxed{
\mathcal A_l=\mathcal A_{d_s},\quad
s_i^l=s_i^{d_s},\quad
K_F^l=K_F^{d_s},
\qquad d_s\le l<d_{s+1}
}.
$$

也就是说，阶段内沿用缓存的 decision 和 scores，不增加重选层。

第一个 decision layer 之前：

$$
\mathcal A_l=\{1,\ldots,N\}.
$$

最后定义：

$$
\boxed{
\mathcal U_l=\{1,\ldots,N\}\setminus\mathcal A_l
}
$$

为当前被原 MLLM-Reroute 跳过的 visual token 集合。

---

# 3. Active token 和 skipped token 的区别

定义：

$$
\boxed{F_l=\text{冻结的第 }l\text{ 个原始 Transformer decoder layer}}.
$$

原始 layer 包含原模型的 normalization、self-attention、FFN 和 residual connections。

另外定义：

$$
\boxed{\mathcal T=\text{所有非视觉 token 的原始位置集合}},
\qquad
\boxed{M=|\mathcal T|}.
$$

\(M\) 是当前样本的输入属性。它包括需要保留的 system、text 和其他非视觉位置。

视觉索引 \(i\) 与完整序列位置属于不同索引空间。先定义：

$$
\boxed{
\mathcal K_l
=\{\operatorname{pos}(i):i\in\mathcal A_l\}\cup\mathcal T
}
$$

为当前 Full branch 的完整序列位置集合。完整序列位置统一从 0 开始计数，范围为 \(0,\ldots,N+M-1\)。

MLLM-Reroute 按 \(\mathcal K_l\) 的原位置升序 gather 成 compact sequence，再执行完整的 \(F_l\)。

因此对 \(i\in\mathcal A_l\)：

$$
\boxed{
v_i^{l+1}
=\left[
F_l\left(
\operatorname{Gather}(X^l,\mathcal K_l)
\right)
\right]_i
}.
$$

这里 \(X^l\in\mathbb R^{(N+M)\times D}\) 表示按原顺序排列的完整 hidden states；\([\cdot]_i\) 表示取出对应原视觉索引 \(i\) 的输出。

这个写法说明：完整 Transformer 是对 compact sequence 进行联合计算，不能把 self-attention 理解为独立的单 token 函数。

对 \(i\in\mathcal U_l\)，当前项目则使用：

$$
\boxed{v_i^{l+1}=v_i^l}.
$$

即：

> 不进入该层完整 attention / FFN，保留 hidden state，等待后续 decision layer 再次被选择。

---

# 4. 问题到底发生在哪里？

以 LLaVA 的原 \(\texttt{ours\_vs\_fastv}\) 设置为例：

$$
\mathcal D=\{3,7,15,23\}.
$$

假设某个视觉 token 在索引 7 的 decision layer 未被选中，那么在索引 7 到 14 的整个阶段都被 skip：

$$
v_i^{8}=v_i^7,\quad
v_i^{9}=v_i^8,\quad\ldots,\quad
v_i^{15}=v_i^{14}.
$$

于是：

$$
\boxed{v_i^{15}=v_i^7}.
$$

与此同时，持续 Active 的 token 已经完成索引 7 到 14 的八个 layer 更新。

当索引 15 的 decision layer 重新在全部原始视觉位置上打分时，旧状态 \(v_i^{15}\) 会参与 scoring；如果被重新选中，它还会直接作为第 15 层的输入。

这就是：

$$
\boxed{\text{Skipped-State Staleness}}
$$

即：

$$
\boxed{\text{被跳过的 token 缺少中间层更新，representation 可能逐渐过时}}.
$$

“可能”表示这是需要测量的现象。连续跳层并不在数学上保证误差单调增加。

---

# 5. 我们真正想预测的量是什么？

先考虑同一个输入在未压缩原模型中的计算。

用：

$$
\boxed{v_i^{l,*}\in\mathbb R^D}
$$

表示 dense teacher 在第 \(l\) 层的输入状态。

上标 \(*\) 专门表示：

$$
\boxed{\text{冻结的原始未压缩模型提供的监督值}}.
$$

定义：

$$
\boxed{
\Delta_i^{l,*}
=v_i^{l+1,*}-v_i^{l,*}
\in\mathbb R^D
}.
$$

它叫做：

> **Layer-wise State Evolution Residual**。

也就是：

> dense 模型在这一层实际把该视觉 token 改变了多少。

原 MLLM-Reroute 对 skipped token 施加的更新是零：

$$
v_i^{l+1}-v_i^l=0.
$$

我们的出发点是：

$$
\boxed{
\text{当前没有进入 Full Top-k}
\not\Rightarrow
\text{后续重新使用它时，冻结状态仍然合适}
}.
$$

---

# 6. 所以我们的真正目标变得非常明确

对需要补偿的 skipped token，不重新运行完整 Transformer，而是预测：

$$
\boxed{\widehat\Delta_i^l\in\mathbb R^D}.
$$

这里帽子 \(\widehat{\phantom{\Delta}}\) 表示预测值。

单层监督目标为：

$$
\widehat\Delta_i^l\approx\Delta_i^{l,*}.
$$

状态更新改为：

$$
\boxed{
v_i^{l+1}=v_i^l+\widehat\Delta_i^l
}.
$$

但必须分清：

$$
\Delta_i^{l,*}
=v_i^{l+1,*}-v_i^{l,*}
$$

是 dense 轨迹上的单层变化，并不等于已经过时的 student 状态到 teacher 下一层状态的全部差值。

因此单层 residual 拟合之外，后面还固定加入重新激活时的 **Freshness Loss**，约束累计误差。

---

# 7. 现在问题变成：怎样便宜地预测 \(\widehat\Delta_i^l\)？

单独根据 token 自身预测：

$$
\widehat\Delta_i^l=f(v_i^l)
$$

只能利用当前局部状态。

原 Transformer 中某个视觉 token 的变化还依赖该位置允许读取的上下文：

$$
\boxed{
\Delta_i^{l,*}
=f_l\left(
v_i^{l,*},
\{x_j^{l,*}:j\le\operatorname{pos}(i)\}
\right)
}.
$$

这里 \(x_j^{l,*}\) 是 dense 序列第 \(j\) 个位置的输入状态。

对于 causal decoder，位于视觉 token 后面的 question tokens 并不会被该视觉 token 直接读取。

因此不能将“所有视觉 residual 都天然包含完整问题信息”作为方法依据。

MLLM-Reroute 的问题相关性主要通过：

$$
\boxed{
\text{最后一个 prompt token 的 attention}
\rightarrow\text{原 Full-token selection}
}
$$

进入路由。新增状态更新分支在这一固定 prompt 条件下工作。

---

# 8. 将 \(\widehat\Delta\) 分成两个来源

固定采用两个组成部分：

$$
\boxed{
\widetilde\Delta_i^l
=\widehat\Delta_{i,\mathrm{self}}^l
+\widehat\Delta_{i,\mathrm{ctx}}^l
}.
$$

其中：

$$
\boxed{
\widehat\Delta_{i,\mathrm{self}}^l
=\text{根据 token 自身状态预测的变化}
}
$$

以及：

$$
\boxed{
\widehat\Delta_{i,\mathrm{ctx}}^l
=\text{根据当前 Active-token residual 预测的变化}
}.
$$

两者都属于 \(\mathbb R^D\)。

\(\widetilde\Delta_i^l\) 表示尚未经过稳定性 gate 的预测；最终 \(\widehat\Delta_i^l\) 在第 37 节统一定义。

---

# 9. 第一部分：Self Evolution Predictor

输入为：

$$
v_i^l\in\mathbb R^D.
$$

固定设置：

$$
\boxed{d_b=32}
$$

为 Ghost predictor 的 bottleneck dimension。这是超参数，不参与训练。

定义降维矩阵：

$$
\boxed{W_{\mathrm{down}}^l\in\mathbb R^{d_b\times D}}.
$$

它是可训练参数，将 \(D\) 维状态映射为 32 维。

得到：

$$
\boxed{
z_i^l=W_{\mathrm{down}}^l\operatorname{LN}_0(v_i^l)
\in\mathbb R^{d_b}
}.
$$

这里：

$$
\boxed{\operatorname{LN}_0=\text{无可训练 affine 参数的 LayerNorm}}
$$

只属于新增分支，不替换原模型的 RMSNorm 或其他 normalization。其数值稳定常数固定为 \(10^{-6}\)。

归一化后，低维 predictor 更容易适应不同 token 的输入幅度。

固定使用：

$$
\boxed{\phi=\operatorname{SiLU}}
$$

作为激活函数。

再定义：

$$
\boxed{W_{\mathrm{up}}^l\in\mathbb R^{D\times d_b}}
$$

为可训练升维矩阵。

于是：

$$
\boxed{
\widehat\Delta_{i,\mathrm{self}}^l
=W_{\mathrm{up}}^l\operatorname{SiLU}(z_i^l)
\in\mathbb R^D
}.
$$

整个分支为：

$$
\boxed{D\rightarrow32\rightarrow D}.
$$

---

# 10. 这两个矩阵到底学什么？

\(W_{\mathrm{down}}^l\) 和 \(W_{\mathrm{up}}^l\) 都参与训练，原 LVLM 参数全部冻结。

它们学习：

> 一个具有当前 hidden representation 的视觉 token，在相应深度继续演化时，通常需要怎样的状态变化。

因此：

$$
\boxed{
\widehat\Delta_{i,\mathrm{self}}^l
\text{ 建模 token-local state evolution}
}.
$$

这里使用上标 \(l\) 表示该层调用的参数。实际存储采用第 45 节的每四层共享规则，不为每层单独复制完整模块。

---

# 11. 但到这里仍然不够

两个样本中的局部 patch 即使具有相似状态，它们所在图像的结构、可见前缀、当前选中集合仍然可能不同。

只读 \(v_i^l\) 的 Self predictor 无法直接知道：

$$
\boxed{\text{当前样本中，真正经过本层计算的视觉 token 正在怎样变化}}.
$$

因此固定加入：

$$
\widehat\Delta_{i,\mathrm{ctx}}^l.
$$

这个分支读取本层实际观察到的 Active residual，不额外读取未来答案，也不在推理时调用 dense teacher。

---

# 12. 这里出现方案的关键想法：借用 Active Token 的真实变化

对：

$$
j\in\mathcal A_l
$$

原 MLLM-Reroute 本来就会执行 compact Transformer。

其输入和输出分别为：

$$
v_j^l,\qquad v_j^{l+1}.
$$

因此定义：

$$
\boxed{
d_j^l=v_j^{l+1}-v_j^l\in\mathbb R^D
}.
$$

这里：

$$
\boxed{
d_j^l=\text{当前压缩 student 实际观察到的 Active-token residual}
}.
$$

它与：

$$
\Delta_j^{l,*}
$$

不同，后者来自 dense teacher。

由于 compact attention 的上下文已经改变，即使模型权重相同，也不能假设：

$$
d_j^l=\Delta_j^{l,*}.
$$

获取 \(d_j^l\) 只需要保留本层 Active 输入并做一次减法，不需要再跑一遍完整 decoder layer。

---

# 13. 为什么 Active token 的 \(d_j^l\) 能帮助 skipped token？

\(d_j^l\) 是当前样本、当前层、当前 compact context 下实际发生的变化。

因此它提供：

$$
\boxed{
\text{sample-conditioned}
+\text{layer-conditioned}
+\text{routing-conditioned residual samples}
}.
$$

原 router 使用完整 prompt 的最后位置进行选择，所以提供 residual 的 Active 集合受到 prompt 影响。

但单个 \(d_j^l\) 是否直接包含问题内容，仍然受原 causal attention 的位置关系限制。

新增分支固定聚合本层全部 Active **视觉** residual。它属于对已知完整 prompt 的额外信息混合，不声称与原 decoder 的逐位置因果依赖完全等价。

本方案的适用范围固定为：

$$
\boxed{\text{已知完整图片和问题的 prompt prefill}}.
$$

训练和推理均不将答案 token 放入 Ghost 的上下文来源。

---

# 14. 将 Active residual 压缩为 Residual Prototype Bank

项目不同预算下的 \(K_F^l\) 不同。

如果每个 Ghost token 都直接读取全部 Active residual，读取规模会随着 \(K_F^l\) 增长。

因此固定采用：

$$
\boxed{\text{Residual Prototype Bank}}
$$

先聚合，再供 Ghost tokens 读取。

这个 bank 在每个需要 Ghost 更新的 layer、每个样本上重新生成，不跨样本复用，也不跨层复用旧 residual。

---

# 15. 什么是 Residual Prototype？

定义：

$$
\boxed{M_p=8}
$$

为每层的 residual prototype 数量。

\(M_p\) 是固定超参数，不是向量，也不是学习出来的 token 数。

给定：

$$
\{d_j^l:j\in\mathcal A_l\},
$$

构造：

$$
\boxed{\{p_m^l\}_{m=1}^{M_p}}.
$$

其中每个：

$$
p_m^l\in\mathbb R^{d_b}
$$

都是由当前样本动态计算的低维 residual 摘要。

例如在 LLaVA 的 \(\texttt{ours\_vs\_fastv/avg192}\) 中：

$$
N=576,\qquad K_F^l=\lfloor576\times0.2644\rfloor=152.
$$

于是：

$$
152\text{ 个 Active residual}
\rightarrow8\text{ 个 prototypes}.
$$

---

# 16. Prototype 首先也要降维

定义：

$$
\boxed{W_R^l\in\mathbb R^{d_b\times D}}
$$

为可训练 residual compression 矩阵。

对每个 Active token：

$$
\boxed{
c_j^l=W_R^ld_j^l\in\mathbb R^{d_b}
}.
$$

\(c_j^l\) 表示 **compressed residual code**，是动态 feature。

这里统一使用 \(c_j^l\)，将后面 \(r_i^l\) 专门留给 reactivation probability，避免符号混用。

---

# 17. 怎样把 Active residual codes 变成 8 个 prototypes？

第 \(m\) 个 prototype 配置一个可训练 query：

$$
\boxed{q_m^{P,l}\in\mathbb R^{d_b}},
\qquad m\in\{1,\ldots,M_p\}.
$$

上标 \(P\) 表示 Prototype。

同时定义可训练矩阵：

$$
\boxed{W_K^{P,l}\in\mathbb R^{d_b\times D}}
$$

并将 Active 输入投影为 key：

$$
\boxed{
k_j^{P,l}=W_K^{P,l}\operatorname{LN}_0(v_j^l)
\in\mathbb R^{d_b}
}.
$$

prototype \(m\) 对 Active token \(j\) 的相关性为：

$$
e_{mj}^l
=\frac{(q_m^{P,l})^\top k_j^{P,l}}{\sqrt{d_b}}
\in\mathbb R.
$$

在当前 Active 视觉集合上归一化：

$$
\boxed{
\beta_{mj}^l
=\frac{\exp(e_{mj}^l)}
{\sum_{j'\in\mathcal A_l}\exp(e_{mj'}^l)}
}.
$$

满足：

$$
\sum_{j\in\mathcal A_l}\beta_{mj}^l=1.
$$

最后：

$$
\boxed{
p_m^l
=\sum_{j\in\mathcal A_l}\beta_{mj}^lc_j^l
\in\mathbb R^{d_b}
}.
$$

训练参数是 query 和投影矩阵；\(\beta_{mj}^l\)、\(c_j^l\)、\(p_m^l\) 都是当前输入产生的动态结果。

---

# 18. 现在拥有了什么？

原来拥有：

$$
K_F^l\text{ 个 }D\text{ 维 Active residual}.
$$

现在得到：

$$
\boxed{
P_l=
\begin{bmatrix}
(p_1^l)^\top\\
\vdots\\
(p_{M_p}^l)^\top
\end{bmatrix}
\in\mathbb R^{M_p\times d_b}
}.
$$

\(P_l\) 是第 \(l\) 层的动态 Residual Prototype Bank，不是独立的可训练常量。

固定尺寸为：

$$
\boxed{8\times32}.
$$

它概括本层已观察到的几种 residual 模式。训练没有给每个 prototype 预先指定“物体”“文字”或“位置”等语义，也不能把这些语义解释当成已经验证的结论。

---

# 19. 接下来 Ghost token 怎样读取这些 prototypes？

对最终进入 Ghost 集合的 token \(i\)，定义：

$$
\boxed{
q_i^{G,l}
=W_Q^{G,l}\operatorname{LN}_0(v_i^l)
\in\mathbb R^{d_b}
}.
$$

其中：

$$
\boxed{W_Q^{G,l}\in\mathbb R^{d_b\times D}}
$$

是可训练矩阵，上标 \(G\) 表示 Ghost branch。

每个 prototype 生成 key 和 value：

$$
\boxed{
k_m^{G,l}=W_K^{G,l}p_m^l,\qquad
u_m^{G,l}=W_V^{G,l}p_m^l
}.
$$

这里：

$$
\boxed{
W_K^{G,l},W_V^{G,l}\in\mathbb R^{d_b\times d_b}
}
$$

均为可训练参数；\(k_m^{G,l},u_m^{G,l}\in\mathbb R^{d_b}\) 是动态 feature。

---

# 20. Ghost Attention

Ghost token \(i\) 与 prototype \(m\) 的匹配分数为：

$$
\ell_{im}^l
=\frac{(q_i^{G,l})^\top k_m^{G,l}}{\sqrt{d_b}}.
$$

这里 \(\ell_{im}^l\in\mathbb R\) 为动态标量，不与后面的 skip age \(a_i^l\) 共用符号。

定义：

$$
\boxed{
\alpha_{im}^l
=\operatorname{softmax}_m(\ell_{im}^l)
},
\qquad
\sum_{m=1}^{M_p}\alpha_{im}^l=1.
$$

于是：

$$
\boxed{
h_{i,\mathrm{ctx}}^l
=\sum_{m=1}^{M_p}\alpha_{im}^l u_m^{G,l}
\in\mathbb R^{d_b}
}.
$$

\(\alpha_{im}^l\) 表示该 Ghost token 对第 \(m\) 个 residual prototype 的读取权重。

这一步只使用 8 个低维 prototypes，不增加原 Transformer 的视觉 attention 序列长度。

---

# 21. 再把 context code 升回原 hidden dimension

定义：

$$
\boxed{W_{\mathrm{ctx}}^l\in\mathbb R^{D\times d_b}}
$$

为可训练升维矩阵。

得到：

$$
\boxed{
\widehat\Delta_{i,\mathrm{ctx}}^l
=W_{\mathrm{ctx}}^lh_{i,\mathrm{ctx}}^l
\in\mathbb R^D
}.
$$

它表示：

> 根据当前真正经过完整计算的视觉 tokens 的 residual，预测 Ghost token 的上下文相关状态变化。

---

# 22. 至此核心 Predictor 已经完整了

Self branch：

$$
\boxed{
\widehat\Delta_{i,\mathrm{self}}^l
=W_{\mathrm{up}}^l
\operatorname{SiLU}\left(
W_{\mathrm{down}}^l\operatorname{LN}_0(v_i^l)
\right)
}.
$$

Context branch：

$$
\boxed{
\widehat\Delta_{i,\mathrm{ctx}}^l
=W_{\mathrm{ctx}}^l
\sum_{m=1}^{M_p}\alpha_{im}^l u_m^{G,l}
}.
$$

未门控预测为：

$$
\boxed{
\widetilde\Delta_i^l
=\widehat\Delta_{i,\mathrm{self}}^l
+\widehat\Delta_{i,\mathrm{ctx}}^l
}.
$$

本方案固定保留两个分支，并在第 37 节使用统一 gate 产生最终更新。

---

# 23. Residual Transport 的作用是什么？

Self-only predictor 的输入为：

$$
v_i^l.
$$

完整 predictor 的输入为：

$$
\boxed{
\left(v_i^l,\{d_j^l:j\in\mathcal A_l\}\right)
}.
$$

Active tokens 提供本层已经观察到的 transformation samples，Ghost token 根据自己的状态读取这些 samples 的低维摘要。

因此这部分称为：

$$
\boxed{\text{Residual Transport}}.
$$

它是否显著优于单独的 Self predictor，必须通过第 57 节规定的消融实验检验，不能仅从模块数量推断。

---

# 24. 从数学角度，它更接近学习一个 transformation field

把深度 \(l\) 看作离散时间：

$$
v_i^{l+1}=v_i^l+\Delta_i^l.
$$

\(\Delta_i^l\) 可以直观理解为 representation 沿深度方向的变化量。

当前层观察到 Active 残差：

$$
\{d_j^l:j\in\mathcal A_l\}.
$$

通过 prototype aggregation 和 Ghost retrieval，估计其他位置的：

$$
\widehat\Delta_i^l.
$$

因此方法可描述为：

$$
\boxed{\text{Residual Field Interpolation}}.
$$

这里的“场”和“插值”是建模解释，不构成精确恢复 dense Transformer 或误差必然下降的理论证明。

---

# 25. 到这里出现第二个问题：需要给所有 skipped token 补吗？

本方案固定只补偿其中一部分。

例如 LLaVA 的 \(\texttt{ours\_vs\_fastv/avg192}\)：

$$
N=576,\qquad K_F^l=152,
$$

所以：

$$
|\mathcal U_l|=424.
$$

其中一些 token 可能在后续 decision layer 重新进入 Full 集合，另一些则可能始终不再进入 Full computation。

但不能提前把“未来未重激活”简单等同于“完全没有影响”：在后续 decision layer，它们仍会参与原项目的全量 scoring，并可能影响选择。

因此新增的选择目标明确限定为：

$$
\boxed{\text{优先维护下一次原定 decision layer 最可能重新被使用的 skipped tokens}}.
$$

这就是：

# Reactivation-Aware Ghost Routing

---

# 26. 先定义“重激活”

MLLM-Reroute 只在 decision layer 改变 Active 集合，因此重激活事件定义为：

$$
\boxed{
i\notin\mathcal A_{l_r-1},
\quad i\in\mathcal A_{l_r},
\quad l_r\in\mathcal D
}.
$$

其中 \(l_r\) 为重新激活层的 0-based 索引。

例如：

$$
i\notin\mathcal A_7,\ldots,
i\notin\mathcal A_{14},
\qquad i\in\mathcal A_{15}.
$$

表示该 token 跳过索引 7 到 14 的完整 layer 计算，在索引 15 的原定决策层回来。

阶段内部不会因为新增 Ghost predictor 而提前进入 Full 集合。

---

# 27. 定义 Future Horizon \(H\)

本方案固定：

$$
\boxed{H=1\text{ 个未来 decision event}}.
$$

这里 \(H\) 的单位是“决策次数”，不是连续 decoder 层数。

定义下一次 decision layer：

$$
\boxed{
d^+(l)=\min\{d\in\mathcal D:d>l\}
}.
$$

\(d^+(l)\) 是由原配置确定的动态索引，不是可训练参数。

如果该集合为空，说明之后没有任何重选机会。

因此：

> 第 \(l\) 层的 Ghost selector 预测当前 skipped token 在 \(d^+(l)\) 是否会重新进入原 Full Top-k。

例如在索引 7 到 14，目标都是索引 15 的下一次决策。

这样无需为适配 Ghost 方法改变原 decision-layer 间隔。

---

# 28. Reactivation label 怎么产生？

训练时记录一条完整 student 路由轨迹，然后构造：

$$
\boxed{
y_i^{l,\mathrm{react}}
=\mathbf1[i\in\mathcal A_{d^+(l)}],
\qquad i\in\mathcal U_l
}.
$$

\(\mathbf1[\cdot]\) 是 indicator function，条件成立为 1，否则为 0。

这里的未来集合来自**同一条实际 student forward 的原 MLLM-Reroute 决策**。

单层预训练阶段使用原 identity-bypass Reroute 轨迹；rollout 阶段使用当前 Reroute-GC 的实际轨迹。

这些 label 只在 forward 完成以后用于 loss，作为停止梯度的监督值。它们不进入当前层 Ghost selector 的输入。

dense teacher 只提供 hidden states 和 residual targets，不替 student 指定未来 Top-k。

没有 \(d^+(l)\) 的层不计算 reactivation loss，也不执行 Ghost 更新。

---

# 29. Predictor 输入哪些信息？

第一项是当前阶段缓存的原始 importance：

$$
s_i^l.
$$

第二项是与 Full 选择边界的距离。

定义：

$$
\boxed{
\tau_l=\min_{j\in\mathcal A_l}s_j^l
}
$$

为该阶段 Full Top-k 的最低选中 score，属于动态标量。

再定义：

$$
\boxed{m_i^l=s_i^l-\tau_l\in\mathbb R}.
$$

如果 \(m_i^l\) 接近 0，说明这个 skipped token 的 score 接近当前选择边界。

这里的边界来自原 Full Top-k，Ghost selector 不修改 \(\tau_l\)、\(s_i^l\) 或 \(\mathcal A_l\)。

为稳定尺度，固定使用：

$$
\boxed{
\widetilde s_i^l=\frac{s_i^l}{\tau_l+\epsilon},
\qquad
\widetilde m_i^l=\frac{s_i^l-\tau_l}{\tau_l+\epsilon}
},
\qquad
\boxed{\epsilon=10^{-6}}.
$$

\(\epsilon\) 为固定数值常数，不参与训练。

非 decision layer 沿用这些 cached scores 和 threshold，不额外运行原模型的全量 attention scoring。

---

# 30. 再引入 Skip Age

定义：

$$
\boxed{
a_i^l
=\text{进入第 }l\text{ 层前，token }i
\text{ 已连续多少层没有执行 Full computation}
}.
$$

\(a_i^l\) 是动态非负整数，不是参数。

初始化：

$$
a_i^0=0.
$$

每层结束后更新：

$$
\boxed{
a_i^{l+1}
=\begin{cases}
0,&i\in\mathcal A_l,\\
a_i^l+1,&i\in\mathcal U_l.
\end{cases}
}
$$

Ghost 也属于 \(\mathcal U_l\)，所以 Ghost 更新不会把 age 清零。

原因是这里统计“距离上一次完整 Transformer 更新多久”，不是“多久没有发生任何数值变化”。

固定截断编码：

$$
\boxed{
\widetilde a_i^l
=\frac{\min(a_i^l,A_{\max})}{A_{\max}},
\qquad A_{\max}=8
}.
$$

真实计数不截断；只对送入网络的 age feature 截断。

---

# 31. 为什么 Skip Age 有用？

两个 token 即使具有相近的 cached score，也可能经历不同长度的 Full bypass。

例如：

$$
s_1^l\approx s_2^l,
\qquad a_1^l=1,\quad a_2^l=7.
$$

它们的状态历史不同，因此新增 predictor 需要知道已经缺少了多少次完整计算。

但：

$$
\boxed{a_i^l\text{ 是状态历史特征，不是真实 representation error}}.
$$

尤其 Ghost token 已经得到近似更新，不能直接说 age 更大的 token 必然更加过时。

误差是否随 age 增加，由诊断实验测量。

---

# 32. Reactivation Predictor 具体怎么写？

对所有当前 skipped tokens，先计算：

$$
z_i^l=W_{\mathrm{down}}^l\operatorname{LN}_0(v_i^l).
$$

因此 Cold token 也需要这一低维投影和分类开销；只有 Ghost 才继续执行 residual 升维更新。

为了适应原项目不同的决策间隔，再定义两个固定规则产生的标量：

$$
\boxed{
\widetilde l=\frac{l}{L-1},
\qquad
\widetilde h_l=\frac{d^+(l)-l}{L-1}
}.
$$

它们分别表示当前相对深度和距离下一次决策的相对层数，都不是训练参数。

拼接输入：

$$
\boxed{
x_i^{\mathrm{react}}
=[
z_i^l;
\widetilde s_i^l;
\widetilde m_i^l;
\widetilde a_i^l;
\widetilde l;
\widetilde h_l
]
\in\mathbb R^{d_b+5}
}.
$$

固定 \(d_b=32\)，所以该输入是 37 维。

定义：

$$
\boxed{
r_i^l
=\sigma\left(
(w_{\mathrm{react}}^l)^\top x_i^{\mathrm{react}}
+b_{\mathrm{react}}^l
\right)
}.
$$

其中：

$$
\boxed{
w_{\mathrm{react}}^l\in\mathbb R^{d_b+5},
\quad b_{\mathrm{react}}^l\in\mathbb R
}
$$

均为可训练参数；

$$
\sigma(x)=\frac1{1+e^{-x}}
$$

为 sigmoid。

最后：

$$
\boxed{
r_i^l\in(0,1)
=\text{token }i\text{ 在下一次原定决策中重新激活的预测概率}
}.
$$

---

# 33. 到这里可以变成三类 token

当前视觉 token 分为：

$$
\boxed{\text{Full / Ghost / Cold}}.
$$

### Full

$$
\mathcal A_l
$$

由原 MLLM-Reroute 决定，执行原 compact Transformer。

### Ghost

$$
\mathcal G_l\subseteq\mathcal U_l
$$

只执行新增低成本状态更新。

### Cold

$$
\mathcal C_l=\mathcal U_l\setminus\mathcal G_l
$$

保持本层输入 hidden state。

满足：

$$
\boxed{
\mathcal A_l\cup\mathcal G_l\cup\mathcal C_l
=\{1,\ldots,N\}
}
$$

且三者两两不相交。

所有非视觉 tokens 始终进入原 Full branch，不参与 Ghost / Cold 划分。

---

# 34. 怎么决定哪些 skipped token 进入 Ghost？

固定：

$$
\boxed{K_G=128}
$$

为每层 Ghost 更新数量的上限。

定义实际数量：

$$
\boxed{
K_G^l
=\begin{cases}
\min(128,|\mathcal U_l|),&d^+(l)\text{ 存在},\\
0,&d^+(l)\text{ 不存在}.
\end{cases}
}
$$

如果没有 skipped token，则 \(K_G^l=0\)。

然后：

$$
\boxed{
\mathcal G_l
=\operatorname{TopK}
\left(\{r_i^l:i\in\mathcal U_l\},K_G^l\right)
}.
$$

新增 Ghost Top-k 在概率相同时按原视觉索引升序打破平局，保证该新增分支可复现。

最后：

$$
\boxed{\mathcal C_l=\mathcal U_l\setminus\mathcal G_l}.
$$

Ghost / Cold 每层可以重新划分，但这不会改变阶段内固定的 Full 集合。

最后一个 decision layer 的 Full 选择完成后，所有未选 token 都设为 Cold。此后既没有重选，也不会进入任何后续 Full attention / KV，因此停止对它们补偿。

---

# 35. Ghost priority 固定为 \(r_i^l\)

本方案的新增选择规则唯一确定为：

$$
\boxed{\text{Ghost priority}=r_i^l}.
$$

不另外定义 residual magnitude、手工风险项或第二套 Full importance。

这使两个决策层次清晰对应各自职责：

$$
\boxed{
\begin{aligned}
\text{原 MLLM-Reroute score }s_i^l
&\rightarrow\text{决定谁执行 Full},\\
\text{新增概率 }r_i^l
&\rightarrow\text{决定 skipped tokens 中谁执行 Ghost}.
\end{aligned}
}
$$

改变 Ghost hidden states 后，后续原 decision layer 的 scores 和实际选中身份可能随之变化。这是状态变化的自然结果。

保持不变的是**打分公式、决策时机、Top-k 规则和数量预算**，不是强行让修改后的模型永远选中与原模型完全相同的 token 身份。

---

# 36. 那么完整 token 更新公式终于可以写出来

对任意视觉 token：

$$
\boxed{
v_i^{l+1}
=\begin{cases}
\left[F_l\big(\operatorname{Gather}(X^l,\mathcal K_l)\big)\right]_i,
&i\in\mathcal A_l,\\[2mm]
v_i^l+\widehat\Delta_i^l,
&i\in\mathcal G_l,\\[2mm]
v_i^l,
&i\in\mathcal C_l.
\end{cases}
}
$$

三种行为分别是：

$$
\boxed{
\begin{aligned}
\text{Full}&:\text{执行原完整 decoder layer},\\
\text{Ghost}&:\text{使用小 predictor 更新状态},\\
\text{Cold}&:\text{保持状态}.
\end{aligned}
}
$$

Ghost 更新在本层 Full branch 完成之后执行，所以不会追溯改变本层已经计算完的 Full outputs。

它影响后续层的状态，以及后续原 decision layer 的打分和重选。

---

# 37. 但是直接相加还存在一个稳定性问题

新增 predictor 的更新幅度需要受到调节。

定义：

$$
\boxed{
x_i^{G}
=[z_i^l;h_{i,\mathrm{ctx}}^l;\widetilde a_i^l]
\in\mathbb R^{2d_b+1}
}.
$$

固定 \(d_b=32\)，所以该输入是 65 维。

可训练 gate 参数为：

$$
\boxed{
w_g^l\in\mathbb R^{2d_b+1},
\quad b_g^l\in\mathbb R
}.
$$

产生：

$$
\boxed{
g_i^l=\sigma\left((w_g^l)^\top x_i^G+b_g^l\right)\in(0,1)
}.
$$

最终 residual 定义为：

$$
\boxed{
\widehat\Delta_i^l
=g_i^l
\left(
\widehat\Delta_{i,\mathrm{self}}^l
+\widehat\Delta_{i,\mathrm{ctx}}^l
\right)
}.
$$

\(g_i^l\) 是可学习的幅度系数，不解释为已经校准的“可信概率”。Sigmoid 限制的是系数，不能单独保证向量范数有界。

固定初始化：

$$
W_{\mathrm{up}}^l,W_{\mathrm{ctx}}^l
\sim\mathcal N(0,10^{-8}),
\qquad w_g^l=0,\quad b_g^l=-2.
$$

即两套升维矩阵的标准差均为 \(10^{-4}\)，初始更新较小。其余线性矩阵采用 Xavier uniform，prototype queries 使用标准差 0.02 的正态初始化，reactivation 权重和 bias 初始化为零。

所有新增矩阵投影不使用 bias；只有明确写出的两个标量 head 使用 bias。

---

# 38. 现在开始讲训练：Ground Truth 从哪里来？

## 训练阶段使用冻结的 Dense Teacher

teacher 就是同一个未压缩原始 LVLM，不是另一套更大的模型。

对相同的图片和问题：

$$
(I,Q)
$$

使用与 student 完全一致的图像预处理、tokenization、prompt 和原始视觉位置。

得到：

$$
v_i^{l,*},\qquad v_i^{l+1,*},
$$

并计算：

$$
\boxed{
\Delta_i^{l,*}=v_i^{l+1,*}-v_i^{l,*}
}.
$$

teacher forward 使用停止梯度。

student 的 prototype 输入始终是当前 compact branch 的 \(d_j^l\)，不能在训练时把它偷偷替换成 dense residual。

新增训练数据固定来自 [GQA 官方训练集](https://cs.stanford.edu/people/dorarad/gqa/download.html)的 balanced questions，只使用图片和问题，不把标准答案接入 decoder。

具体数据划分和训练轮数在第 46、47 节给出。这是新增模块必需的训练步骤，原仓库的评测脚本并不已经包含这一步。

---

# 39. 第一项 Loss：Residual Reconstruction Loss

对当前 forward 实际执行 Ghost 更新的 token，定义数量：

$$
\boxed{Z_G=\sum_l|\mathcal G_l|}.
$$

固定使用按 hidden dimension 归一化的均方误差：

$$
\boxed{
\mathcal L_\Delta
=\frac1{\max(1,Z_G)}
\sum_l\sum_{i\in\mathcal G_l}
\frac1D
\left\|
\widehat\Delta_i^l-\Delta_i^{l,*}
\right\|_2^2
}.
$$

这里 \(\|\cdot\|_2\) 是欧氏范数。

损失的作用是让单层 Ghost residual 接近 dense 轨迹的单层变化。

当 \(Z_G=0\) 时，该项定义为零，不产生除零或无效梯度。

---

# 40. 第二项 Loss：Direction Loss

teacher residual 范数过小时，方向没有稳定含义，因此先定义有效项集合：

$$
\boxed{
\mathcal I_{\mathrm{dir}}
=\{(l,i):i\in\mathcal G_l,\ \|\Delta_i^{l,*}\|_2\ge\epsilon\},
\qquad
Z_{\mathrm{dir}}=|\mathcal I_{\mathrm{dir}}|
}.
$$

\(\mathcal I_{\mathrm{dir}}\) 是动态索引集合，\(Z_{\mathrm{dir}}\) 是有效项数，都不是参数。

定义：

$$
\boxed{
\mathcal L_{\mathrm{dir}}
=\frac1{\max(1,Z_{\mathrm{dir}})}
\sum_{(l,i)\in\mathcal I_{\mathrm{dir}}}
\left[
1-
\frac{
(\widehat\Delta_i^l)^\top\Delta_i^{l,*}
}{
\max(\|\widehat\Delta_i^l\|_2,\epsilon)
\max(\|\Delta_i^{l,*}\|_2,\epsilon)
}
\right]
}.
$$

其中：

$$
\epsilon=10^{-6}.
$$

这一项约束预测 residual 与 teacher residual 的方向。

全部候选都无效时，该 loss 为零。

范数、点积和 loss reduction 固定使用 FP32，避免低精度下的小范数计算不稳定。

---

# 41. 第三项 Loss：Reactivation Prediction Loss

对存在下一次决策的层，定义：

$$
\boxed{
Z_R=\sum_{l:d^+(l)\text{ 存在}}|\mathcal U_l|
}.
$$

使用：

$$
\boxed{
\mathcal L_{\mathrm{react}}
=-\frac1{\max(1,Z_R)}
\sum_{l:d^+(l)\text{ 存在}}
\sum_{i\in\mathcal U_l}
\left[
y_i^{l,\mathrm{react}}\log r_i^l
+(1-y_i^{l,\mathrm{react}})\log(1-r_i^l)
\right]
}.
$$

实现直接使用 logits 和 numerically stable 的 BCE-with-logits，而不是先算 sigmoid 再手工取 log。

这一 loss 覆盖 Ghost 和 Cold 候选，避免只在已选 Ghost 上训练 reactivation head。

hard Top-k 和未来路由标签均停止梯度，不用未来标签代替当前 Ghost 选择。

---

# 42. 最重要的 Loss 还要监督“重新回来之前”

我们关心的时刻是：token 被重新选中，**尚未执行重新激活层的 Full computation** 时，其输入是否已经接近 dense 状态。

定义实际重激活事件集合：

$$
\boxed{
\mathcal R
=\{(i,l_r):
l_r\in\mathcal D,\ 
i\notin\mathcal A_{l_r-1},\
i\in\mathcal A_{l_r}\}
}.
$$

固定使用：

$$
\boxed{
\mathcal L_{\mathrm{fresh}}
=\frac1{\max(1,|\mathcal R|)}
\sum_{(i,l_r)\in\mathcal R}
\frac1D
\left\|
v_i^{l_r}-v_i^{l_r,*}
\right\|_2^2
}.
$$

这项 loss 约束的是多层 Ghost / Cold 轨迹累积后的实际输入状态。

如果某条轨迹没有重激活事件，则该项为零。

它与单层 residual loss 互补：单层更新接近 teacher 不代表累计状态已经接近 teacher。

---

# 43. 总 Loss 怎么组合？

固定：

$$
\boxed{
\mathcal L_{\mathrm{total}}
=\mathcal L_\Delta
+0.1\mathcal L_{\mathrm{dir}}
+0.1\mathcal L_{\mathrm{react}}
+\mathcal L_{\mathrm{fresh}}
}.
$$

即：

$$
\boxed{
\lambda_\Delta=1,\quad
\lambda_{\mathrm{dir}}=0.1,\quad
\lambda_{\mathrm{react}}=0.1,\quad
\lambda_{\mathrm{fresh}}=1
}.
$$

四个权重是固定超参数，不参与训练，也不通过测试集调参。

单层预训练阶段尚未将 Ghost 更新串入完整轨迹，因此固定不启用 Freshness Loss；完整 rollout 阶段使用上面的四项总损失。

这两步是连续训练阶段，不是两个待选方案。

---

# 44. 哪些参数训练，哪些不训练？

原视觉编码器、projector / merger、token embedding、LLM decoder 和输出 head 全部：

$$
\boxed{\text{Frozen}}.
$$

原 MLLM-Reroute 的 scorer、decision-layer schedule、keep ratios、Full Top-k 和 \(\texttt{monotonic=false}\) 保持原样。

原 router 是规则和 attention 计算，不应描述成新增的可训练 selector。

新增可训练参数只有：

$$
\boxed{
\begin{aligned}
&W_{\mathrm{down}},W_{\mathrm{up}},W_R,\\
&q_m^P,W_K^P,\\
&W_Q^G,W_K^G,W_V^G,W_{\mathrm{ctx}},\\
&w_g,b_g,w_{\mathrm{react}},b_{\mathrm{react}}.
\end{aligned}
}
$$

rollout 训练中，冻结原权重不等于对整个 student 使用 \(\texttt{no\_grad}\)。

需要保留从后续状态损失，经冻结 Full layers，到前面 Ghost 更新的梯度路径；否则 Freshness Loss 无法训练更早层的补偿。

---

# 45. 但是每层一套参数会不会很多？

固定采用：

# Block-shared Ghost Predictor

定义：

$$
\boxed{G_{\mathrm{share}}=4}
$$

为共享同一组 Ghost 参数的连续 layer 数。

层 \(l\) 使用：

$$
\boxed{b(l)=\left\lfloor\frac l4\right\rfloor}
$$

号参数组。

例如：

$$
l=0,1,2,3
$$

共享第 0 组；

$$
l=4,5,6,7
$$

共享第 1 组。

因此 LLaVA 分配：

$$
\lceil32/4\rceil=8
$$

组，Qwen 分配：

$$
\lceil28/4\rceil=7
$$

组。没有 Ghost 执行机会的尾部参数组不参与 forward 或优化，并在参数统计中区分“分配参数”和“实际使用参数”。

同一组内共享前面列出的全部新增矩阵和 heads，但不共享动态 prototypes、scores、age 或 hidden states。

两个 backbone 分别训练。不同原 keep schedule / budget 分别训练对应 checkpoint；compact 与 stagewise 使用同一套 checkpoint。

---

# 46. 完整训练流程重新从头走一次

### Stage 0：固定训练数据并准备 teacher supervision

固定从 GQA balanced training split 建立 8,704 个不同图片的 image-question 样本，每张图只取一个问题。

确定性选择规则为：

1. 先按图像内容去重，并剔除与本项目全部评测集合及 bundled efficiency cohort 重复的图像。
2. 每张图在该训练 split 内选择数值 question ID 最小的问题。
3. 按 \(\operatorname{SHA256}(\texttt{"42:"}+\text{image ID})\) 升序排列。
4. 前 8,192 张图作为训练集，接下来的 512 张图作为内部验证集。

固定保存 manifest、图像哈希和顺序；若符合条件的数据不足 8,704 张，则数据准备报错，不从评测集合补样本。

只给模型输入 \((I,Q)\)，不接入答案。两个 backbone 使用相同样本 manifest，但保留各自原 processor 和 prompt。

teacher 提供：

$$
V^{l,*},\quad\Delta_i^{l,*}.
$$

### Stage 1：产生原 MLLM-Reroute 轨迹

每个原 Reroute schedule / tier 独立运行 identity-bypass student，记录：

$$
V^{l,\mathrm R},\quad
\mathcal A_l,\quad
s_i^l,\quad
a_i^l,\quad
d_j^l.
$$

其中 \(\mathrm R\) 表示原 Reroute。

再根据实际下一次 decision 的集合产生：

$$
y_i^{l,\mathrm{react}}.
$$

### Stage 2：单层 Ghost 预训练

固定训练 1 个 epoch。

输入使用上一步真实 Reroute 状态和 compact residual，计算 reactivation probability，在 skip 集合内选出预测的 Ghost Top-k，再计算 Self、Context 和 gate。

使用：

$$
\boxed{
\mathcal L_\Delta
+0.1\mathcal L_{\mathrm{dir}}
+0.1\mathcal L_{\mathrm{react}}
}.
$$

此阶段的各层输入来自记录轨迹，不把本层新预测写进下一层训练输入。完成后进入下一节的 rollout 训练。

---

# 47. 为什么还需要第二阶段 Rollout Training？

单层预训练看到的是原 Reroute 状态。部署时，后续层会读到前面 Ghost 自己产生的状态。

因此固定继续训练 1 个 epoch 的完整 Reroute-GC rollout：

$$
0\rightarrow1\rightarrow\cdots\rightarrow L-1.
$$

这一次将 Ghost 更新实际写回 token state。

原 decision layers 使用当前 student states 重新打分，保持原 Full Top-k 和预算；同阶段 Full 集合仍固定。

整次 forward 完成后，用该次 forward 下一次 decision 的实际选中集合监督 reactivation head，并使用四项总 loss。

固定训练配置为：

$$
\boxed{
\begin{aligned}
\text{optimizer}&=\operatorname{AdamW},\\
\text{learning rate}&=10^{-4},\\
(\beta_1,\beta_2)&=(0.9,0.999),\\
\text{optimizer epsilon}&=10^{-8},\\
\text{weight decay}&=0.01,\\
\text{micro batch}&=1,\\
\text{gradient accumulation}&=16,\\
\text{gradient clipping norm}&=1.0,\\
\text{random seed}&=42.
\end{aligned}
}
$$

使用恒定学习率。两个阶段连续使用同一个 optimizer state；每阶段开始时以种子 \(42+\text{阶段编号}\) 对训练 manifest 确定性打乱。

矩阵参数施加 weight decay，prototype queries、向量 heads 和标量 bias 不施加 decay。

原模型保留项目对应的加载精度；新增参数使用 FP32 master weights，projection 和 loss 的数值计算使用 FP32，写回时转换成原 hidden dtype。

训练统一使用 \(\texttt{compact\_route}\) 执行形式，使用 \(\texttt{use\_cache=false}\)。

activation checkpointing 只包裹无状态的原 decoder 计算和纯 tensor Ghost predictor，固定使用 \(\texttt{use\_reentrant=false}\)。routing cache、age 和状态写回在 checkpoint 外每层只执行一次，反向重算复用该次 forward 已确定的 hard indices。

不能直接 checkpoint 整个带状态修改的 patched forward，否则反向重算可能重复推进 age 或覆盖路由缓存。teacher 和 hard routing labels 停止梯度，student 的跨层 hidden-state 梯度保留。

原 backbone 始终保持 eval 模式，只有新增模块处于 train 模式。训练写回使用可微分的 functional scatter / index_copy 构造新状态，不原地覆盖反向传播需要的输入或 deferred tensor。

固定交付第二阶段结束的 checkpoint，内部验证用于报告四项 loss 和 freshness，不根据测试集选择训练轮数或 checkpoint。

原项目共有两种 Reroute schedule、三个预算档、两个 backbone，因此训练：

$$
\boxed{2\times3\times2=12}
$$

个 Ghost checkpoint。每个 checkpoint 同时用于对应的 compact 和 stagewise 执行版本。

---

# 48. 完整 inference 流程

## 第一步：视觉编码

输入图片 \(I\)，使用原视觉编码器及 projector / merger 得到：

$$
V^0=\{v_1^0,\ldots,v_N^0\}.
$$

图像尺寸控制和预处理完全沿用原模型配置。

## 第二步：文本编码

输入问题 \(Q\)，使用原 tokenizer、chat template 和 embeddings，保留原序列位置。

初始化当前样本的 routing cache、skip ages 和 deferred states，不读取上一样本的缓存。

## 第三步：第 \(l\) 层的原 MLLM-Reroute selection

第一个 decision layer 之前执行原 dense 层。

在 decision layer：

$$
\text{完整当前状态}
\rightarrow s_i^l
\rightarrow\text{原 Top-k}
\rightarrow\mathcal A_l.
$$

非 decision layer 直接使用当前阶段缓存的 \(\mathcal A_l\) 和 scores。

## 第四步：Full branch 先执行

将原序列位置集合 \(\mathcal K_l\) 按升序 gather，使用原 RoPE/MRoPE 和原 compact decoder 计算，得到 Full outputs。

## 第五步：得到本层真实 residual samples

如果存在下一次决策且有 skipped tokens，取：

$$
d_j^l=v_j^{l+1}-v_j^l,\qquad j\in\mathcal A_l.
$$

## 第六步：对 skipped token 预测未来 reactivation

对所有 \(i\in\mathcal U_l\) 计算 \(z_i^l\)、标量输入和 \(r_i^l\)，在 \(\mathcal U_l\) 内选择：

$$
\mathcal G_l=\operatorname{TopK}(r_i^l,K_G^l).
$$

剩余为 \(\mathcal C_l\)。

## 第七步：构造 Residual Prototype Bank

有 Ghost token 时，用本层 Active inputs 和实际 residual 构造：

$$
P_l\in\mathbb R^{8\times32}.
$$

没有 Ghost token 时跳过整个 bank 和 Ghost predictor。

## 第八步：Ghost update

对 \(i\in\mathcal G_l\)：

$$
\widehat\Delta_i^l
=g_i^l
\left(
\widehat\Delta_{i,\mathrm{self}}^l+
\widehat\Delta_{i,\mathrm{ctx}}^l
\right),
$$

然后：

$$
\boxed{v_i^{l+1}=v_i^l+\widehat\Delta_i^l}.
$$

## 第九步：Cold token

对 \(i\in\mathcal C_l\)：

$$
\boxed{v_i^{l+1}=v_i^l}.
$$

更新所有视觉 token 的 skip ages，继续下一层。

最后一个决策层完成后，不再运行 Ghost selector 或 Ghost update。prefill 结束后沿用原项目的 autoregressive decode，不为历史 Ghost / Cold token 补写 KV cache。

---

# 49. 最后把三部分重新拼回来

逻辑上，第 \(l+1\) 层完整视觉状态为：

$$
\boxed{
V^{l+1}
=\operatorname{Scatter}
\left(
V_{\mathrm{Full}}^{l+1},
V_{\mathrm{Ghost}}^{l+1},
V_{\mathrm{Cold}}^{l+1}
\right)
}.
$$

Scatter 表示按原视觉 token index 放回对应位置，逻辑上的视觉位置总数仍为 \(N\)。

对于项目的 \(\texttt{compact\_route}\)，每层将 Full 和 Ghost 输出写回完整 hidden tensor，Cold 位置保持输入。

对于 \(\texttt{compact\_route\_stagewise}\)，保留原来的阶段内 compact 执行方式：Full tensor 在层间传递，Ghost 更新写入当前 deferred-state buffer，Cold buffer 项保持原值。

到下一个 decision layer，使用：

$$
\boxed{
\text{最新 Full states}
\cup\text{最新 Ghost states}
\cup\text{保留的 Cold states}
}
$$

重建完整输入，再调用原 decision-layer scoring。

不能只更新临时 Ghost tensor，却让下一阶段继续读取阶段开始时的旧 deferred snapshot。

stagewise 和 compact 使用相同的数学更新、索引映射、Ghost tie-break 和 checkpoint。它们是同一方法的两种原有执行形式，是否达到预期数值一致性需通过实现测试核对。

---

# 50. 这时一个 token 的完整 trajectory 可以长这样

使用原 LLaVA schedule：

$$
\mathcal D=\{3,7,15,23\}.
$$

假设 token 271 在索引 7 未进入 Full，在索引 15 重新激活。

一条允许的轨迹为：

$$
\boxed{
\begin{array}{c|ccccccccc}
l&7&8&9&10&11&12&13&14&15\\
\hline
\text{状态}&G&G&C&G&G&G&C&G&F
\end{array}
}.
$$

例如：

$$
v_{271}^{8}=v_{271}^{7}+\widehat\Delta_{271}^{7},
$$

$$
v_{271}^{9}=v_{271}^{8}+\widehat\Delta_{271}^{8},
$$

$$
v_{271}^{10}=v_{271}^{9}
$$

对应 Ghost、Ghost、Cold。

在索引 15 进入 Full 之前：

$$
\boxed{
v_{271}^{15}
=v_{271}^{7}
+\sum_{\substack{l=7,\ldots,14\\271\in\mathcal G_l}}
\widehat\Delta_{271}^{l}
}.
$$

原 Reroute 在相同持续 skip 区间则保持：

$$
v_{271}^{15}=v_{271}^{7}.
$$

这条示例只说明状态更新方式，不能保证实际模型一定选择该 token 或产生这些 Ghost / Cold 状态。

---

# 51. 为什么这种方法有望比普通 Adapter 更有效？

普通 Self Adapter 主要利用：

$$
\widehat\Delta_i^l=f(v_i^l).
$$

完整方法利用：

$$
\boxed{
\widehat\Delta_i^l
=f\left(
v_i^l,
\{d_j^l:j\in\mathcal A_l\},
a_i^l
\right)
}.
$$

它同时获得局部状态、实际层变化样本和 Full bypass 历史。

此外，补偿预算由下一次原决策中的重激活概率分配。

这给出明确的可检验假设：

$$
\boxed{
\text{在相同 Ghost 预算下，Active residual context 能降低重激活输入误差}
}.
$$

该优势必须由训练后的对照实验支持，不能写成已有定论。

---

# 52. 这个设计的另一个重要特点：不是所有 skipped token 都补

在 LLaVA 的 \(\texttt{ours\_vs\_fastv/avg192}\) 且仍有下一次决策的层：

$$
N=576,\quad K_F^l=152,\quad K_G^l=128.
$$

所以：

$$
|\mathcal C_l|=576-152-128=296.
$$

视觉计算分配为：

$$
\boxed{
152\times\text{Full}
+128\times\text{Ghost Update}
+296\times\text{Identity}
}.
$$

此外，全部 424 个 skipped tokens 都参与低维 reactivation scoring。

最后一个 decision layer 以后则为：

$$
152\times\text{Full}
+424\times\text{Identity},
$$

不继续花费 Ghost 分类或状态更新计算。

这里的 152 来自该项目原 keep ratio；不能把所有 \(\texttt{avg192}\)、\(\texttt{avg128}\)、\(\texttt{avg64}\) 档位直接当成每层 Full token 数。

---

# 53. Ghost branch 到底比完整 Transformer 便宜多少？

设当前层：

$$
U_l=|\mathcal U_l|,\qquad
A_l=|\mathcal A_l|,\qquad
G_l=|\mathcal G_l|.
$$

这里 \(U_l,A_l,G_l\) 是集合大小的动态整数，不是新增参数。

所有 skip 候选的低维投影和 reactivation head 开销为：

$$
O(U_lDd_b).
$$

Active residual compression、prototype key 和 aggregation 开销为：

$$
O(A_lDd_b+A_lM_pd_b).
$$

Ghost 的 Self 升维、query 投影、Context 升维和 prototype retrieval 开销为：

$$
O(G_lDd_b+G_lM_pd_b).
$$

prototype 小矩阵变换还包含：

$$
O(M_pd_b^2).
$$

因此新增主要复杂度为：

$$
\boxed{
O\left(
U_lDd_b+
A_lDd_b+
A_lM_pd_b+
G_lDd_b+
G_lM_pd_b+
M_pd_b^2
\right)
}.
$$

还必须计算 Top-k、normalization、age 更新、残差减法、gather/scatter 和内存读写。

按本文独立投影逐项计数，主要矩阵乘的 multiply-accumulate 次数近似为：

$$
\boxed{
(U_l+2A_l+3G_l)Dd_b
+O\left(M_p(A_l+G_l)d_b+M_pd_b^2\right)
}.
$$

这里一次 multiply-accumulate 是一次乘法加一次加法；换算为 FLOPs 时必须与原 profiler 的计数口径保持一致。

不能只报告 \(O(M_pd_b)\)，因为 \(D\leftrightarrow d_b\) 投影通常是新增分支的重要开销。

Full Transformer 仍然包含原 Q/K/V/O projections、compact attention 和大 FFN。新增 predictor 的结构规模较小，但实际延迟受 kernel 数、FP32 运算和 memory bandwidth 影响。

固定报告：

$$
\boxed{
\text{总 FLOPs}
+\text{prefill latency}
+\text{end-to-end latency}
+\text{KV bytes}
+\text{辅助状态内存}
+\text{峰值显存}
}.
$$

原 decision-layer 全量 Q/K scoring 开销也必须计入总成本。

保留原 Full keep schedule 后，新方法的总成本通常增加，不能继续自动声称与原 FastV / PDrop 对照严格 FLOPs-matched。

---

# 54. 主实验与 MLLM-Reroute 项目保持一致

固定使用项目的两个 backbone：

$$
\boxed{
\text{LLaVA-1.5-7B}
\quad+\quad
\text{Qwen2.5-VL-7B}
}.
$$

固定保留三个原档位：

$$
\boxed{\texttt{avg192},\quad\texttt{avg128},\quad\texttt{avg64}}.
$$

主评测完整覆盖：

| 类别 | 原项目任务 |
|---|---|
| VQA / 综合能力 | POPE、GQA、MMBench English dev、MME |
| RefCOCO | val、testA、testB |
| RefCOCO+ | val、testA、testB |
| RefCOCOg | val、test |

保持 batch size 1、SDPA、原 task definitions、原数据 split、原 prompt、原指标和原图像预处理。

LLaVA grounding 使用原 \(\texttt{bbox=normalized,prompt=full}\)；Qwen 使用原 \(\texttt{bbox=pixel,prompt=nuwa}\)。

accuracy evaluation 的 generation 参数沿用项目调用的 lmms-eval v0.7.1 task/model 默认值，不另设统一 temperature 或生成长度。

原来的 \(\texttt{paper\_main}\) 任务组包含 POPE 和 8 个 grounding splits；完整复现还要覆盖 \(\texttt{vqa}\) 任务组中的 GQA、MMBench 和 MME。

原 38 个 baseline / routing configs 全部保留。新增 Reroute-GC 结果与对应原 Reroute config 成对报告，新增权重训练成本单独记录。

## 诊断一：MLLM-Reroute 是否存在 State Staleness？

在内部验证集上，对实际重激活事件记录：

$$
\boxed{
E_i^{l_r}
=1-\cos\left(v_i^{l_r},v_i^{l_r,*}\right)
}.
$$

按重激活前的 Full skip age \(a_i^{l_r}\) 分组，同时报告每组样本数、均方误差和 cosine error。

对比原 Reroute 与 Reroute-GC 的实际轨迹，并在共同重激活事件上另报配对结果，避免把“不同 token 被选回来”误解为纯状态改善。

是否存在 age 与误差的关系，以测量结果为准。

---

# 55. 诊断二：Active residual 能不能预测 skipped residual？

固定在 512 个内部验证样本上测量。

第一步，在 dense residual 上分析同层共享结构：

$$
\{\Delta_j^{l,*}:j\in\mathcal A_l\}
\rightarrow
\Delta_i^{l,*}.
$$

这里的 \(\mathcal A_l\) 使用对应 student 的真实选择。报告秩 8 和秩 32 的 residual reconstruction error。

这只是 teacher 空间中的结构诊断。

第二步，使用部署时真正可获得的：

$$
\boxed{\{d_j^l:j\in\mathcal A_l\}}
$$

构造 prototypes，比较 Self-only 与 Self+Context 的 Ghost residual error。

只有第二步才能直接检验“实际 compact residual 能否帮助部署中的 skipped-token 更新”。

不能用 dense Active residual 的重构能力代替真实 student residual 的有效性证据。

---

# 56. 诊断三：Future-reactivated token 是否更需要补偿？

按下一次原定决策将当前 skipped tokens 分成：

$$
\boxed{
\mathcal U_l^+
=\{i\in\mathcal U_l:y_i^{l,\mathrm{react}}=1\}
}
$$

和：

$$
\boxed{
\mathcal U_l^-=\mathcal U_l\setminus\mathcal U_l^+
}.
$$

二者分别表示“下次重新激活”和“下次未重新激活”，不把负类写成“此后永远不重要”。

固定报告：

$$
\boxed{
\text{两类状态误差}
+\text{Ghost selection precision / recall}
+\text{reactivation average precision}
}.
$$

没有正样本的分组明确报告样本数，相关比例记为不适用，不伪造为零性能。

这些诊断服务于解释主方法；主评测数据集和原预算设置不因诊断结果而改变。

---

# 57. 最重要的 ablation 应该是什么？

消融固定使用项目原 \(\texttt{ablation}\) 任务集合：

$$
\boxed{\text{GQA、MMBench、RefCOCO testA、RefCOCO testB}}.
$$

在两个 backbone、三个预算和两种原 Reroute schedule 上执行下表。accuracy ablation 使用 compact 形式，stagewise 在完整方法上验证精度一致性与效率。

| 编号 | 实验行 | 确定的修改 |
|---|---|---|
| A0 | 原 MLLM-Reroute | 全部 skipped token 使用 identity bypass |
| A1 | Reroute + Self | 使用固定 Ghost 预算和 learned reactivation selector，只保留 Self residual 与 gate；使用全部适用 losses |
| A2 | Reroute + Context | 相同预算和 selector，只保留 Context residual 与 gate；使用全部适用 losses |
| A3 | Reroute + Self + Context，去掉 learned Ghost selection | skipped token 按原 cached importance 从高到低选择 Ghost，同分按原索引升序；保留 residual、direction 和 freshness losses |
| A4 | Reroute-GC，去掉 Freshness Loss | 完整 Ghost selector、Self、Context、gate；训练时固定令 freshness 权重为 0 |
| A5 | 完整 Reroute-GC | 本文确定的全部模块与四项 loss |

A1 的 gate 将 context code 置零；A2 保留生成 selector / gate 所需的低维 \(z_i^l\)，不执行 Self 升维输出。

每行分别训练相应模块，保持训练 manifest、训练轮数、原 Full schedule、Ghost 上限和初始化种子一致。

消融是验证各组成部分的固定实验设计，不是部署方法的待选菜单；最终部署使用 A5。

---

# 58. 怎样判断 Residual Transport 是否带来实际贡献？

首先比较：

$$
\boxed{\text{A5 完整方法}\quad\text{与}\quad\text{A1 Self-only}}.
$$

二者使用相同 Full 预算、Ghost 上限、训练数据和训练轮数，主要差别是 Active residual context。

同时比较：

$$
\text{A5 与 A3}
$$

检验 learned Ghost selection；

$$
\text{A5 与 A4}
$$

检验 Freshness Loss。

固定联合考察：

$$
\boxed{
\text{原 benchmark 指标}
+\text{重激活输入误差}
+\text{实际新增延迟}
}.
$$

如果某模块没有可测收益，实验结论如实报告。方法在测量前不宣称必然优于 Self Adapter，也不预填准确率或加速倍数。

---

# 59. 本方案的全部固定参数与项目对接位置

## 原 decision layers：逐项保持

| 原 Reroute 配置 | LLaVA-1.5-7B | Qwen2.5-VL-7B |
|---|---|---|
| ours_vs_fastv | \([3,7,15,23]\) | \([3,6,13,20]\) |
| ours_vs_pdrop | \([2,7,15,23]\) | \([2,6,13,20]\) |

全部使用 0-based 索引。对应 stagewise 配置使用相同 schedule。

原 physical-delete 对照保持不变：FastV 在索引 3 单次选择；PDrop 使用原四阶段 schedule 和其原 keep ratios，不接入 Ghost / Cold。

## 原 Reroute keep ratios：逐项保持

| 档位 | ours_vs_fastv 四阶段 ratios | ours_vs_pdrop 四阶段 ratios |
|---|---|---|
| avg192 | \([0.2644,0.2644,0.2644,0.2644]\) | \([0.4502,0.4502,0.2251,0.1126]\) |
| avg128 | \([0.1418,0.1418,0.1418,0.1418]\) | \([0.2655,0.2655,0.1328,0.0664]\) |
| avg64 | \([0.01916,0.01916,0.01916,0.01916]\) | \([0.1392,0.1392,0.0696,0.0348]\) |

两个 backbone 沿用同一组原 ratios，不为 Ghost 重新校准。

三个 avg 名称沿用原项目的预算档位标签。Qwen 的 ratios 沿用 LLaVA 参考值，不保证每个样本实际按层平均的视觉 token 数恰好等于档位名称中的数字。

当 \(N=576\) 时，实际 Full 数量为：

| 档位 | ours_vs_fastv 每阶段 \(K_F\) | ours_vs_pdrop 四阶段 \(K_F\) |
|---|---|---|
| avg192 | \([152,152,152,152]\) | \([259,259,129,64]\) |
| avg128 | \([81,81,81,81]\) | \([152,152,76,38]\) |
| avg64 | \([11,11,11,11]\) | \([80,80,40,20]\) |

## 新增参数：唯一确定的设置

$$
\boxed{
\begin{aligned}
K_G&=128,\\
d_b&=32,\\
M_p&=8,\\
H&=1\text{ 个未来 decision event},\\
A_{\max}&=8,\\
G_{\mathrm{share}}&=4,\\
\epsilon&=10^{-6}.
\end{aligned}
}
$$

最后一个 decision layer 完成选择后，所有 skip token 固定为 Cold。

## 代码对接位置

下面是确定的实现落点，并不表示这些修改已在当前仓库完成：

| 项目位置 | 对接内容 |
|---|---|
| models/router.py | 原 score、decision cache、Top-k、keep ratios、monotonic 逻辑原样保留 |
| 新增 models/ghost.py | 存放 block-shared Self / Context predictor、prototype bank、gate 和 skip 内 Ghost selector |
| models/patching.py 的 RoutingContext | 增加每样本 age、Ghost / Cold 记录和模块引用；新 prefill 时清空动态状态 |
| _forward_compact_route | 保留 compact Full 输入和输出；在原 scatter 阶段加入 Ghost 写回 |
| _forward_compact_route_stagewise | decision 后更新当前 deferred buffer 中的 Ghost 行 |
| _forward_compact_in_stage | 每层以本层 Active 输入 / 输出产生 residual，更新同一 deferred buffer |
| 新增训练脚本 | 执行 teacher supervision、单层预训练、完整 rollout 和独立 Ghost checkpoint 保存 |
| scripts/run_eval.py 与 profiler 的模型加载入口 | 加载对应 Ghost checkpoint；原模型、任务和 routing 配置不变 |
| 新增 GC 实验配置 | 继承对应原配置，只增加 Ghost checkpoint 和固定 GC 参数，结果目录单独记录 |

checkpoint 必须包含 backbone、\(D\)、decision layers、keep ratios、Ghost 超参数和训练 manifest 哈希。加载时严格匹配对应原配置，禁止随机初始化 Ghost 模块后直接报告正式评测成绩。

## 实现验收

固定验证以下语义：

1. 关闭新增模块时回到原 Reroute；人工令 Ghost residual 为零时，Full outputs、states 和 decisions 与原路径一致。
2. 同一阶段 Full 集合固定，Ghost / Cold 都是 skip 子集；Cold hidden 不变，非视觉位置不进入新增分支。
3. 下一次 decision 读到更新后的 deferred states；新样本清空缓存，decode 不执行历史视觉 Ghost 更新。
4. compact 与 stagewise 在相同权重、输入和正常执行路径上具有一致的状态与选择结果，允许按原 dtype 设置合理数值容差。
5. 覆盖空 skip 集合、skip 数小于 128、无下一次 decision、可变 \(N\)、无重激活事件和梯度跨层传播。

## 效率实验：保留原执行参数

使用原 bundled manifest 的 3 张图片及原 prompt。

原 \(\texttt{run\_profile.sh}\) 默认执行：

$$
1\text{ pass},\qquad0\text{ warmup}.
$$

原 \(\texttt{run\_runtime\_bench.sh}\) 默认执行：

$$
\boxed{
5\text{ timed passes},
\quad2\text{ warmup passes},
\quad64\text{ generated tokens}
}.
$$

runtime 保留：

$$
\texttt{min\_new\_tokens=max\_new\_tokens=64},
\quad\texttt{do\_sample=false},
\quad\texttt{use\_cache=true}.
$$

profile / runtime 的 backbone dtype 保持 LLaVA FP16、Qwen BF16；新增模块 FP32 的成本如实计入。

沿用原 CUDA-event 计时边界，CPU 图像预处理不计入 GPU generation 时间。额外增加 GC 模块 FLOPs 和辅助内存统计时，矩阵运算和 functional attention 不能漏计。

KV cache 仍只写 Full visual + nonvisual tokens。Ghost / Cold 的 hidden-state buffer 是辅助状态内存，必须与 KV bytes 分开报告。

---

# 60. 把所有主要符号汇总一次

| 符号 | 含义及维度 | 类型 |
|---|---|---|
| \(L,D\) | 原 decoder 层数、hidden dimension | 固定模型属性 |
| \(l\) | 从 0 开始的当前 layer index | 索引 |
| \(N,M\) | 当前样本视觉、非视觉 token 数 | 输入属性 |
| \(X^l\) | 原顺序完整状态，\((N+M)\times D\) | 动态 feature |
| \(v_i^l\) | 当前视觉状态，\(D\) 维 | 动态 feature |
| \(\mathcal T\) | 非视觉位置集合 | 动态集合 |
| \(\mathcal K_l\) | Active 视觉位置映射到完整序列后，与非视觉位置的并集 | 动态集合 |
| \(\mathcal D,d_s\) | 原 decision-layer 集合及第 \(s\) 次决策位置 | 固定配置 |
| \(\rho_s\) | 原阶段 keep ratio | 固定配置 |
| \(s_i^l\) | 原决策层 score，阶段内缓存 | 动态标量 |
| \(K_F^l\) | 原配置确定的本层 Full 视觉数量 | 动态整数 |
| \(\mathcal A_l,\mathcal U_l\) | 原 Full 集合、原 skip 集合 | 动态集合 |
| \(F_l\) | 原 decoder layer | 冻结模块 |
| \(v_i^{l,*}\) | dense teacher hidden，\(D\) 维 | 监督信号 |
| \(\Delta_i^{l,*}\) | dense 单层 residual，\(D\) 维 | 监督信号 |
| \(d_j^l\) | 实际 compact Active residual，\(D\) 维 | 动态 feature |
| \(d_b\) | Ghost bottleneck，32 | 固定超参数 |
| \(\operatorname{LN}_0\) | 新增无 affine LayerNorm | 固定运算 |
| \(W_{\mathrm{down}},W_{\mathrm{up}}\) | \(d_b\times D,\ D\times d_b\) | 可训练 |
| \(z_i^l\) | Self / selector 低维输入，\(d_b\) 维 | 动态 feature |
| \(W_R,c_j^l\) | residual 投影矩阵、\(d_b\) 维 residual code | 参数 / 动态 feature |
| \(M_p\) | prototype 数，8 | 固定超参数 |
| \(q_m^P,W_K^P\) | prototype query、Active key projection | 可训练 |
| \(\beta_{mj}^l\) | prototype 对 Active residual 的聚合权重 | 动态标量 |
| \(p_m^l,P_l\) | \(d_b\) 维 prototype、\(M_p\times d_b\) bank | 动态 feature |
| \(W_Q^G,W_K^G,W_V^G\) | Ghost query / prototype key / value 投影 | 可训练 |
| \(\alpha_{im}^l\) | Ghost 对 prototype 的读取权重 | 动态标量 |
| \(h_{i,\mathrm{ctx}}^l\) | context code，\(d_b\) 维 | 动态 feature |
| \(W_{\mathrm{ctx}}\) | Context 升维，\(D\times d_b\) | 可训练 |
| \(\widehat\Delta_{i,\mathrm{self}}^l,\widehat\Delta_{i,\mathrm{ctx}}^l\) | Self / Context 预测，均为 \(D\) 维 | 动态 feature |
| \(g_i^l\) | Ghost update 幅度 gate | 动态标量 |
| \(w_g,b_g\) | gate 权重，\(2d_b+1\) 维及标量 | 可训练 |
| \(\widehat\Delta_i^l\) | 门控后的最终 Ghost residual，\(D\) 维 | 动态输出 |
| \(\tau_l,m_i^l\) | 原 Full threshold、score margin | 动态标量 |
| \(a_i^l,A_{\max}\) | 距上次 Full 的层数、编码上限 8 | 动态整数 / 超参数 |
| \(H\) | 未来 horizon，固定 1 次决策 | 固定超参数 |
| \(d^+(l)\) | 下一次原 decision layer | 动态索引 |
| \(r_i^l\) | 下一次决策重新激活的预测概率 | 动态标量 |
| \(w_{\mathrm{react}},b_{\mathrm{react}}\) | reactivation head，\(d_b+5\) 维及标量 | 可训练 |
| \(K_G,K_G^l\) | Ghost 上限 128、实际 Ghost 数 | 超参数 / 动态整数 |
| \(\mathcal G_l,\mathcal C_l\) | Ghost / Cold 集合 | 动态集合 |
| \(y_i^{l,\mathrm{react}}\) | 实际 student 下一次决策产生的 label | 停止梯度监督 |
| \(\mathcal R\) | 实际 skip→reactivation 事件集合 | 动态监督集合 |
| \(G_{\mathrm{share}},b(l)\) | 共享层数 4、参数组编号 | 超参数 / 索引 |

---

# 61. 最后把整套算法浓缩成一条因果链

原 MLLM-Reroute 的 skip 路径为：

$$
\boxed{
\text{原 decision-layer attention score}
\rightarrow
\text{原 Full Top-k}
\rightarrow
\text{未选 token 保持 hidden state}
\rightarrow
\text{等待后续原决策层重选}
}.
$$

新增方法保留前两步，将 skip 路径改成：

$$
\boxed{
\begin{array}{c}
\text{原 MLLM-Reroute decision layers、scores、Top-k、keep ratios}\\
\downarrow\\
\text{原 Full 集合 }\mathcal A_l
\quad+\quad
\text{原 skip 集合 }\mathcal U_l\\
\downarrow\\
\text{只在 }\mathcal U_l\text{ 中预测下一次原决策的 reactivation}\\
\downarrow\\
\text{Full / Ghost / Cold}\\
\downarrow\\
\text{原 Full branch 产生真实 Active residual}\\
\downarrow\\
\text{Residual Prototype Bank + Self Evolution + Gate}\\
\downarrow\\
i\in\mathcal G_l:\quad v_i^{l+1}=v_i^l+\widehat\Delta_i^l\\
i\in\mathcal C_l:\quad v_i^{l+1}=v_i^l\\
\downarrow\\
\text{更新后的状态等待下一次原定 decision layer}\\
\downarrow\\
\text{沿用原 scoring 和 Full Top-k 完成重选}
\end{array}
}.
$$

整个工作的核心问题是：

$$
\boxed{
\textbf{Skipping computation should not necessarily mean skipping representation evolution.}
}
$$

对应到当前项目，最终方法明确为：

$$
\boxed{
\text{MLLM-Reroute 原路由}
+
\text{skip 内 Ghost / Cold 划分}
+
\text{Active Residual Prototype 驱动的 Ghost 状态更新}
}.
$$

原模型、原 Full-token 决策机制和原评测体系继续作为基础；新增训练和计算开销单独记录，效果由相同实验条件下的实际结果判断。
