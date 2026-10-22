# 对 Claude 审阅的数学修正与实现约束

日期：2026-10-22。审阅对象：[PR #1 评论](https://github.com/Yunbo-max/multichoice/pull/1#issuecomment-6349668026)及提交 `c712daced0b313da6c5ad31022578cc8c8aca68f` 中的 [原审阅](review_and_simplification_2026-10-22.md)。

**结论：可以把单参数 count law 作为新实验臂，但原审阅中有公式错误、缺失条件和不可直接执行的对照建议。下面的修正优先于原审阅的相应说法。**

本次是文档与审阅回复更新。当前代码仍实现独立 count head；下文的 mixture、power tilt 和 empirical-map 是待实现方案，不代表已经存在的 CLI 或完成的实验。这里指出的是审阅提案的问题，不是把所有问题都归因于当前实现。

## 0. 统一符号与损失尺度

$K$ 是候选数量，$C$ 是推理链数量，$\mathcal M_x$ 是输入协议允许的答案数量。教师联合分布为 $Q$，其统计量为

$$q_i=\mathbb E_Q[\mathbf1(i\in S)],\qquad t_m=Q(|S|=m).$$

学生为

$$Z_m(s)=\sum_{|A|=m}\exp\!\left(\sum_{i\in A}s_i\right),\qquad
P_{\rho,s}(S)=\rho_{|S|}\frac{\exp(\sum_{i\in S}s_i)}{Z_{|S|}(s)}.$$

对任意合法的数量分布 $\rho$，集合交叉熵是

$$\boxed{L=-q^\top s+\sum_{m\in\mathcal M_x}t_m\log Z_m(s)-\sum_{m\in\mathcal M_x}t_m\log\rho_m.}$$

本文使用按候选求和的 BCE 和未除以 $K$ 的集合损失。当前代码按题除以 $K$；比较公式、梯度或基线时，必须对两边采用相同缩放。混合不同 $K$ 时，这也决定了题目之间的相对权重。

## R1. “alpha=0 严格等于 BCE”缺少答案协议条件

原审阅的结论仅在允许全部数量 $0,\ldots,K$ 时成立。令

$$\mathrm{PB}_m(s)=\frac{Z_m(s)}{\prod_i(1+e^{s_i})},\qquad
C_{\mathcal M}(s)=\sum_{m\in\mathcal M}\mathrm{PB}_m(s).$$

必须先把隐式数量分布限制到合法范围：

$$r_m^{\mathcal M}(s)=\frac{\mathrm{PB}_m(s)}{C_{\mathcal M}(s)}
=\frac{Z_m(s)}{\sum_{n\in\mathcal M}Z_n(s)},\qquad m\in\mathcal M.$$

建议的合法 mixture 是

$$\boxed{\rho_{\alpha,m}=(1-\alpha)r_m^{\mathcal M}(s)+\alpha\pi_m,\qquad 0\le\alpha\le1,}$$

其中先验 $\pi$ 也必须在同一 $K$/任务协议的允许数量上归一化。于是

$$\boxed{L_{\alpha=0}=-q^\top s+\log\sum_{m\in\mathcal M}Z_m(s)
=L_{\mathrm{BCE}}+\log C_{\mathcal M}(s).}$$

| 输入协议 | 正确端点 |
|---|---|
| 允许空集，$\mathcal M=\{0,\ldots,K\}$ | 普通逐项 BCE |
| 非空多选，$\mathcal M=\{1,\ldots,K\}$ | 以答案非空为条件的 BCE |
| 单选，$\mathcal M=\{1\}$ | 候选 softmax 交叉熵 |

反例：$K=2,s=(0,0),Y=\{A\}$ 时，三种损失分别为 $\log4$、$\log3$、$\log2$。因此测试不能对所有协议都断言等于普通 BCE。

若直接混合未条件化的 PB 与合法先验，再丢弃不合法数量，总质量为 $(1-\alpha)C_{\mathcal M}+\alpha$，通常不等于 1。先混合再重新条件化也可构成合法模型，但先验有效权重变为

$$\alpha_{\mathrm{effective}}=\frac{\alpha}{(1-\alpha)C_{\mathcal M}+\alpha},$$

不再是原来声称的固定混合强度。

单选时 $\rho_1=1$，$\alpha$ 不可辨识且不改变数量分布。单选提升不能直接证明数量修正有效。

## R2. 原审阅的完整 score 梯度漏了 -q_i

令

$$u_{mi}=\partial_{s_i}\log Z_m(s)=P_s(i\in S\mid |S|=m).$$

正确的完整偏导是

$$\boxed{\partial_{s_i}L=-q_i+\sum_m t_mu_{mi}
-\sum_m\frac{t_m}{\rho_m}\partial_{s_i}\rho_m.}$$

原审阅漏掉了 $-q_i$。在不限制数量、$\alpha=0$ 的情形，正确梯度必须回到 $\sigma(s_i)-q_i$；缺项会错误地得到 $\sigma(s_i)$。

对 R1 的条件化 mixture，固定 $\alpha$、固定先验时，记 $\nu_i=\sum_mr_m^{\mathcal M}u_{mi}$，则

$$\partial_{s_i}\rho_{\alpha,m}=(1-\alpha)r_m^{\mathcal M}(u_{mi}-\nu_i),$$

$$\partial_\alpha L=-\sum_mt_m\frac{\pi_m-r_m^{\mathcal M}}{\rho_{\alpha,m}}.$$

若 $\alpha=\alpha(x;\theta)$ 是可学习 probe，求骨干参数梯度时还要包含经由 $\alpha$ 的链式法则。目标和训练集先验保持冻结；使用完整标量损失的 autograd，不手工遗漏依赖路径。

当前独立 head 的 score 梯度已经含有 $-q_i$；本条纠正的是新提案中的公式。

## R3. 一阶矩匹配不推出“损失差等于教师全部 total correlation”

设独立 BCE 学生输出 $b_i$，$h$ 为伯努利熵，定义

$$I(Q)=\sum_i h(q_i)-H(Q).$$

一般恒等式为

$$\boxed{L_{\mathrm{BCE}}(q;b)-L_{\mathrm{set}}(Q;P)
=I(Q)+\sum_iD_{\mathrm{KL}}(\operatorname{Ber}(q_i)\|\operatorname{Ber}(b_i))
-D_{\mathrm{KL}}(Q\|P).}$$

即使 BCE 已经匹配边缘概率 $b=q$，仍然有

$$\boxed{L_{\mathrm{BCE}}(q;q)-L_{\mathrm{set}}(Q;P)
=I(Q)-D_{\mathrm{KL}}(Q\|P).}$$

所以原审阅的等式还要求 $P=Q$，不能仅凭一阶矩匹配。结构模型的实际边缘概率通常也不等于 $\sigma(s_i)$，两者不能混用。

### 四选项反例

教师 $Q_A$ 以各 $1/2$ 的概率输出 `{AB}`、`{CD}`；教师 $Q_B$ 均匀输出全部六个二元素集合。两者都满足

$$q=(1/2,1/2,1/2,1/2),\qquad t_2=1.$$

当前损失只读取 $q,t$，因此任何学生参数下都无法区分它们。理想拟合都得到六个二元素集合均匀的 $P^*$。

| 量 | $Q_A$ | $Q_B$ |
|---|---:|---:|
| 最优 BCE | $4\log2$ | $4\log2$ |
| 最优集合损失 | $\log6$ | $\log6$ |
| 损失差 | $\log(8/3)$ | $\log(8/3)$ |
| 教师 $I(Q)$ | $3\log2$ | $\log(8/3)$ |
| 残余 $D_{\mathrm{KL}}(Q\|P^*)$ | $\log3$ | $0$ |

### 可采用的正确命题

对单个输入，假设教师统计量可行，学生的 score 与独立数量分布可自由优化，并允许必要的边界极限。最优 $P^*$ 是合法答案空间上、匹配 $q,t$ 的最大熵分布：

$$P^*=\arg\max_R H(R)\quad\text{s.t.}\quad
\mathbb E_R[\mathbf1(i\in S)]=q_i,\quad R(|S|=m)=t_m.$$

拉格朗日乘子给出 $R(S)\propto\exp(a_{|S|}+\sum_{i\in S}s_i)$，正是当前独立 head 模型族。由于 $Q$ 与 $P^*$ 匹配这些充分统计量，

$$-\mathbb E_Q\log P^*=H(P^*),\qquad
D_{\mathrm{KL}}(Q\|P^*)=H(P^*)-H(Q).$$

因此

$$\boxed{\min_bL_{\mathrm{BCE}}-\min_{\rho,s}L_{\mathrm{set}}=I(P^*),\qquad
I(Q)=I(P^*)+D_{\mathrm{KL}}(Q\|P^*).}$$

**准确表述：当前目标蒸馏选项边缘概率和答案数量分布，以及满足这些统计约束所必需的依赖结构；它不能恢复任意选项特异的共现关系。**

这是逐题、容量充分的性质。共享 Transformer 参数、受限的 alpha-mixture 族、未达到最优的训练，都不能直接套用该最优等式。它也不是准确率提升或新颖性的证明。

## R4. 随意 shuffled-count 可能使目标无下界

合法的联合教师统计量至少满足

$$\sum_iq_i=\sum_mmt_m.$$

若把另一道题的数量目标 $\widetilde t$ 放入集合损失所有位置，却保留原来的 $q$，则可能根本不存在对应的教师联合分布。

令 $F(s)=-q^\top s+\sum_m\widetilde t_m\log Z_m(s)$。共同平移分数给出

$$F(s+c\mathbf1)-F(s)=c\left(\sum_mm\widetilde t_m-\sum_iq_i\right).$$

对固定 $\alpha>0$，若先验在目标支撑上为正，count CE 沿此方向有界：

$$0\le\operatorname{CE}(\widetilde t,\rho_\alpha(s))
\le-\log\alpha-\sum_m\widetilde t_m\log\pi_m.$$

因此平均数量不一致时，总损失可沿某个平移方向趋向负无穷。具体反例：非空 $K=2$，$q=(1,1)$，$\widetilde t=\delta_1$，$s=(c,c)$，有

$$L=-c+\log2-\log\rho_{\alpha,1}(s)\longrightarrow-\infty.$$

只匹配平均数量也不充分。将 $q$ 降序排列后，合法统计量还需满足

$$\sum_{j=1}^kq_{(j)}\le\sum_mt_m\min(k,m),\qquad k=1,\ldots,K.$$

例如 $q=(1,1,0)$ 与 $t=\tfrac12\delta_1+\tfrac12\delta_3$ 均值都为 2，但在 $k=2$ 时有 $2>1.5$，仍不可行。当前目标构建器已有一致性与可行性检查；不能为了运行 shuffle 对照绕过这些检查。

一个明确的替代对照是仅打乱单独一项 count CE 的目标：

$$L_{\mathrm{control}}=-q^\top s+\sum_mt_m^{\mathrm{original}}\log Z_m(s)
-\sum_mt_m^{\mathrm{shuffled}}\log\rho_m.$$

前半部分保留合法教师的条件集合监督，后半部分是单独的数量分类监督。它是拆分目标的扰动对照，不能称为对同一个教师联合分布的 CE。若未来 rho 依赖 scores，后半部分仍须向 scores 正常反传；“单独一项”不表示梯度独立。shuffle 应匹配 $K$/协议；即使该对照有收益，也不能排除全局数量先验的作用。

`BCE(q) + CE(t,rho)` 应称为“解耦的边缘与数量监督”，而不是字面意义上的“只传数量”。

## R5. empirical-map 没有恢复教师的完整联合不确定性

逐链求

$$S_c=\arg\max_{S:|S|\in\mathcal M}\sum_{i\in S}\operatorname{logit}(b_{ci})$$

恰好是独立 Bernoulli 在合法答案范围内的 MAP 规则。其经验分布

$$Q_{\mathrm{emp}}=\frac1C\sum_c\delta_{S_c}$$

是合法联合分布，链间可以有相关性；但它把每条链的软不确定性压成一个硬集合，不能从单链边缘概率恢复原来未知的完整教师联合分布。

准确表述应为“由逐链硬集合构造经验目标”，而非“完全去掉独立性假设”。若要把样本解释为教师原生集合策略的样本，应让教师直接产生并验证完整集合，而不是把边缘概率的加性 MAP 误写成任意联合教师的 MAP。

全部数量都合法时，加性 MAP 与 0.5 阈值选择相同（零 log-odds 的平局规则除外）；非空约束主要在阈值结果为空时改变输出。两者不应被包装成普遍不同的算法。

实现时继续保留每个 view 等权、view 内有效链等权的约定；有效链数不同，不能悄悄改成所有链直接等权。$K$ 表示候选数，$C$ 表示链数。比较教师目标时固定学生；比较数量头时固定教师目标。

## R6. power tilt 的 log normalizer 符号写反了

对

$$\rho_{\gamma,m}=\frac{Z_m(s)^\gamma}{\sum_{n\in\mathcal M}Z_n(s)^\gamma},$$

正确负对数似然是

$$\boxed{L_\gamma=-q^\top s+(1-\gamma)\sum_mt_m\log Z_m(s)
+\log\sum_{n\in\mathcal M}Z_n(s)^\gamma.}$$

原审阅把 log normalizer 写成负号。负号属于 log probability；负对数似然中是正号。$\gamma=1$ 的端点仍取决于 R1 的协议。

“不能产生分离的双峰数量分布”需要限定 $\gamma>0$ 及这里的连续数量支撑；允许负指数时该结论不成立，例如 $K=2,s=0,\gamma=-1$ 得到 $(0.4,0.2,0.4)$。

## R7. 相关性、合法性和 alpha 的解释不能过度推广

1. **“数量约束只带来负相关”错误。** 当前加权模型的固定数量切片有非正成对协方差，但混合数量后可以正相关。例如 $K=2,s=0,\rho=(0.49,0.02,0.49)$ 时，$P(00)=P(11)=0.49$、$P(01)=P(10)=0.01$，协方差为 $0.24>0$。限制在于不能表达任意选项特异的依赖，不在于只能负相关。
2. **“mixture/alpha 是唯一合法 count law”错误。** 当前独立 head 只要在合法数量上归一化，就是合法模型。合法性取决于概率归一化、支撑和教师统计量一致性。
3. **alpha 接近 0 不证明数量信息没用。** 它只能说明在当前先验、模型和优化设置下，这个混合修正没有得到明显使用。alpha 接近 1 也不能证明所有任务都不需要逐题数量预测；需要匹配对照。
4. **“scores 改不动 count”是固定 rho 时的说法。** 一旦 rho 通过 PB 依赖 scores，scores 就会改变数量分布；恒等式 $P(|S|=m)=\rho_m$ 仍成立。
5. 一个全局 alpha 才是一个新增参数。线性 scalar probe 有 $d+1$ 个参数；骨干模型仍然微调。

## R8. 新分支的实现约束与实验顺序

- 先增加全局 alpha 实验臂，保留当前完整 head；没有验证前不要把 mixture 改成唯一默认路径。
- prior 只从 train 估计，并按 $K$/协议归一化；为需要支持的数量保留正质量。teacher 平均数量先验与 gold 数量先验是两种明确的对照，不能使用 dev/test 标签估计 prior。
- 固定 alpha=0 是端点基线。若用 sigmoid 参数化可学习 alpha，有限参数不能精确初始化为 0；使用明确的内部初值，如 0.1 或 0.5，或者另行说明投影边界参数化。
- 保持损失缩放、合法数量、训练步数和解码器一致，分别报告普通 BCE、协议条件化基线、全局 alpha、alpha(x)、完整 head。
- **不能直接沿用完整 head 的 score 平移不变性。** 若 $s_i=u_i+c$，则 $\log Z_m(s)=\log Z_m(u)+mc$。计算 PB 或 mixture 的数量概率时必须恢复 $mc$，并使用原始 scores 的 softplus 归一化；否则随意中心化会改变模型。within-count 概率本身可以使用中心化分数。
- 原独立 head 的自由 score/count-logit 凸性证明，不自动适用于把 rho 与 scores 耦合后的 mixture。不要沿用未经证明的凸性表述。
- 学生 mixture 的集合 MAP 仍比较 $\log\rho_m+\sum_{i\in\mathrm{Top}_m}s_i-\log Z_m(s)$；不能误用教师条件乘积模型的纯加性 log-odds 解码。
- 先固定教师目标比较 count law，再固定学生比较软条件 Bernoulli 目标和经验硬集合目标。保留 group split、同预算、teacher quality 和非 oracle 的评估要求。

中心化的精确恢复公式为 $s=\widetilde s+c\mathbf1$ 时

$$\log\mathrm{PB}_m(s)=\log Z_m(\widetilde s)+mc-\sum_i\operatorname{softplus}(s_i),\qquad
r_m^{\mathcal M}(s)=\operatorname{softmax}_{m\in\mathcal M}[\log Z_m(\widetilde s)+mc].$$

mixture 非凸也有直接反例：允许空集的 $K=1$，固定 $\alpha=1/2$、$\pi_1=1/2$，标签为正，则

$$L(s)=-\log(1/4+\tfrac12\sigma(s)),\qquad L''(-\log3)=-1/16<0.$$

所以这里不是仅仅缺少证明；全局凸性本身不成立。

## 9. 可直接采用的方法表述

> 我们研究无显式推理的集合决策读出：从同一冻结模型的多种提示视角和推理链，构造相互一致的选项边缘概率及答案数量分布，并将其蒸馏到合法答案集合上的结构化学生。当前模型保留这些统计量约束下的集合结构，不声称恢复教师任意联合相关性。单参数数量混合是待验证的简化方案，必须与协议匹配的基线和完整数量头比较。

如果老师都来自同一个 checkpoint，应称为 multi-view self-distillation；不同提示和多条链不等于多个独立模型。上述性质不构成“全新概率模型”或性能提升的证据。

## 10. 验证证据的边界

[Claude 的原评论](https://github.com/Yunbo-max/multichoice/pull/1#issuecomment-6349668026)报告在其 PyTorch 环境中 85/85 测试通过，并完成真实 Qwen3.5 tokenizer 的 Yes/No 单 token 与 2,040 条 dev 行检查。本次文档更新没有重新运行该环境，也没有附加取得原始执行日志。

原开发环境的 82 通过、3 跳过仍是独立的本地记录；不能把外部报告改写为本次亲自运行的结果。真实模型前向、FSDP 导出/重载、GPU 教师与学生训练、以及新方法的准确率和校准收益仍未验证。
