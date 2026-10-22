# Review + mathematical simplification proposal

> **2026-10-22 数学修正：本页是原审阅的历史记录，不应直接作为实现依据。**
> 后续核对发现：完整 score 梯度漏了 `-q_i`；`alpha=0 -> BCE` 缺少答案协议条件；total-correlation 等式缺少残余 KL；任意 shuffled-count 可能构造不合法且无下界的目标。power-tilt 归一化项的符号、empirical-map 的独立性解释和 alpha 的实验解读也需要修正。
> **请以 [审阅修正与实现约束](review_corrections_2026-10-22.md) 中的正确公式、反例和实验约束为准。**
> 原文保留在下方以便逐项对照。原审阅报告的验证结果保持其原始归属；新增 mixture/empirical-map 方案尚未实现。

（这条评论是给 ChatGPT 改代码用的：先给审阅结论，再给一个“去掉 count head、用一参数数学变形”的具体方案和实现清单。）

## 0. 我已经做过的验证

- 拉取了 `codex/multiteacher-set-distillation`，读了 `docs/multiteacher_set_distillation.md`、`set_distillation.py`、`build_multiteacher_targets.py`、`sii_style_teacher.py`、trainer/eval 的改动。
- 用项目自带 torch 环境跑 `python -m unittest discover -s tests -v`：**85/85 passed**，包括 PR 里说“缺 PyTorch 跳过”的 3 个 autograd / count-head 测试。
- 用真实 Qwen3.5 tokenizer 跑 `--validate-only --q-transform membership`：Yes=9175、No=2665，均为单 token；2040 条 dev 行全部合法。
- 仍未验证：真实模型前向、FSDP 导出/重载、GPU teacher/student 训练。PR 对此的边界说明是准确的。

---

## 1. 新颖性判断（直说）

**数学本身不算新。** `P(S) ∝ ρ(|S|) ∏_{i∈S} w_i` 是 cardinality potential / conditional Bernoulli / Poisson-binomial conditioned on count 这一族；`Z_m` 的 DP、marginal、MAP 都是标准工具（Tarlow 2012、conditional Bernoulli 1994）。CE 也是标准。不要用“新概率模型”作为卖点。

**组合和应用有新意。** “multi-view self-teacher → candidate-level Yes/No → count-constrained set law → structured student readout → exact set MAP”这个组合，我没有看到完全相同的构造；作为 application/system 方法它是干净的。但顶会 main track 光靠这个组合不够，需要：

1. 一个可证明/可量化的理论点；
2. 在重建的 group split 上，相对 matched controls 的稳健实验优势。

**最强的理论包装**是把 set KD 与 mean-marginal BCE 的差写成 multi-information：

```
L_BCE(q) - L_set(Q) = I(Q)   （学生匹配一阶矩时）
```

即独立 sigmoid BCE 是最大熵投影，丢掉的正是 teacher set 分布的 total correlation；你的方法 transfer 的正是 count 诱导的这部分。这个命题应该正式写进 docs，`{A}` vs `{A,B,C}` 只是它的特例。

**Reviewer 最可能攻击的点：**

1. count 不是独立问出来的，而是从 per-option Yes/No 经条件独立 + count 约束推导的；“同时 distill which 和 how many”这个说法略强。
2. per-chain conditional independence 是强假设；count 约束只引入一种负相关，无法表示 `{AB,CD}` vs `{AC,BD}`。
3. teacher quality 是瓶颈：必须先报 teacher 的 count NLL/Brier、set EM、marginal calibration，否则整套 DP 没意义。
4. 缺 count-only 和 shuffled-count 对照。
5. set MAP vs thresholded marginals 都要报；旧 58.63% 不能当新方法证据。

---

## 2. 关键数学点：count 边际就是 ρθ，不能直接删 head

在

$$P_\theta(S)=\rho_\theta(|S|;x)\frac{e^{\sum_{i\in S}s_i}}{Z_{|S|}(s)},\qquad Z_m(s)=\sum_{|A|=m}e^{\sum_{i\in A}s_i}$$

里，对固定 $m$ 求和：

$$\sum_{|S|=m}P_\theta(S)=\rho_\theta(m;x)\frac{\sum_{|S|=m}e^{\sum_{i\in S}s_i}}{Z_m(s)}=\rho_\theta(m;x).$$

**所以 count 边际完全由 $\rho_\theta$ 决定，scores 改不动它。** 固定/全局 prior 会让所有题共享同一个 count 分布，per-question “有几个答案”直接丢失。因此：

- 不能“删掉 head + 用固定 prior”作为主方法；
- head 的唯一职责是产生 input-dependent 的 $\rho_\theta(m|x)$；
- 正确的化简是把 head 降成**一个参数的数学变形**，而不是删掉。

---

## 3. 提议：一参数 count law

### 3.1 α-mixture（推荐先做）

令

$$\mathrm{PB}_m(s)=\frac{Z_m(s)}{\prod_i(1+e^{s_i})}$$

是现有 scores 的隐式 Poisson-binomial。把学习到的 head 换成

$$\boxed{\rho_\theta(m|x)=(1-\alpha)\,\mathrm{PB}_m(s)+\alpha\,\bar\rho(m)}$$

其中 $\bar\rho$ 是训练集上的全局 count 先验（K+1 个数，不是模块），$\alpha\in[0,1]$ 是**一个标量**（全局参数，或标量 probe $\alpha(x)=\sigma(w^\top h+b)$）。

**α=0 严格退化为 BCE：**

$$\log\rho_\theta(m)=\log Z_m(s)-\sum_i\log(1+e^{s_i}),$$

$$\mathcal L=-\sum_m\rho_T(m)\log\rho_\theta(m)-\sum_iq_i s_i+\sum_m\rho_T(m)\log Z_m(s)
=\sum_i\log(1+e^{s_i})-\sum_iq_i s_i,$$

正好是 `BCEWithLogits(q, s)`。**α=1** 时 count 固定为 $\bar\rho$，只学 within-count membership。

梯度（供实现参考）：

$$\frac{\partial\mathcal L}{\partial s_i}
=-\sum_m\frac{\rho_T(m)}{\rho_\theta(m)}\frac{\partial\rho_\theta(m)}{\partial s_i}
+\sum_m\rho_T(m)P_\theta(i\in S\mid|S|=m),$$

$$\frac{\partial\rho_\theta(m)}{\partial s_i}=(1-\alpha)\,\mathrm{PB}_m(s)\Big[P_\theta(i\in S\mid|S|=m)-p_i\Big],\qquad p_i=\sigma(s_i).$$

所有量都能从现有 DP 拿到（$Z_m$、条件 marginal、$\mathrm{PB}_m$），不需要新模块。$\alpha$ 的梯度是 likelihood-ratio：

$$\frac{\partial\mathcal L}{\partial\alpha}
=-\sum_m\rho_T(m)\frac{\bar\rho(m)-\mathrm{PB}_m(s)}{\rho_\theta(m)}.$$

**这个构造的意义**：$\alpha$ 是一个可解释的插值参数。$\alpha=0$ 是纯 BCE，$\alpha=1$ 是固定 count 先验，中间是数据驱动的 count 修正。它直接回答“count 信息到底有没有用、需不需要 per-question”。

### 3.2 备选：power tilt

$$\rho_\theta(m)\propto \mathrm{PB}_m(s)^\gamma,\qquad \gamma=1\Rightarrow\text{BCE}.$$

代回 set CE 后主项是 $(1-\gamma)\sum_m\rho_T(m)\log Z_m(s)+\gamma\sum_i\log(1+e^{s_i})$，再加归一化项 $-\log\sum_m\mathrm{PB}_m^\gamma$；$\gamma$ 一个标量。它更“指数族”，但只能调 over/under-dispersion，**表示不了双峰 count**（例如 $\rho=(0,0.5,0,0.5)$）。所以推荐先做 α-mixture。

### 3.3 零新参数备选：复用冻结 label head

把 255 个单 token 标签中一段没被 option 用到的（例如最后 22 个）当 count 类，$\rho_\theta=\mathrm{softmax}(\text{这些 logits})$。不新增参数，但要求 backbone 学会对齐固定的 label 方向；风险最高，作为最后 ablation。

---

## 4. 教师侧化简：经验 chain-set 分布

现在的 per-chain conditional-independence + count conditioning 是最弱的假设。K=8 时可以直接用每条链的 **MAP set**：

$$S_c=\arg\max_{S:\,|S|\in\mathcal M_x}\sum_{i\in S}\log\frac{b_i}{1-b_i},$$

（固定 $m$ 时就是 top-$m$ log-odds；再在 allowed counts 里选最好的 $m$；$b_i\in\{0,1\}$ 用端点处理，tie 按 option index。）

然后取经验分布

$$Q_T=\frac1C\sum_c\delta_{S_c},\qquad
q_i=\frac1C\sum_c\mathbf1[i\in S_c],\qquad
\rho_m=\frac1C\sum_c\mathbf1[|S_c|=m].$$

loss 变成 observed sets 的 listwise CE：

$$\mathcal L=-\frac1C\sum_c\left[\log\rho_\theta(|S_c|)+\sum_{i\in S_c}s_i-\log Z_{|S_c|}(s)\right].$$

优点：

- 去掉教师侧 conditional-independence 假设和教师侧 DP；
- K=8 的链间分歧直接变成经验 count 分布，正好利用 variance/agreement；
- 学生侧 $Z_m$/DP 保留，exact set MAP 不变。

原来的 DP target 保留为 ablation（`--aggregate conditional_bernoulli` vs `--aggregate empirical_map`）。

---

## 5. 化简后的完整方法

1. **Teacher**：per-chain membership $b$ → per-chain MAP set → 经验 $Q_T=(q,\rho)$（K=8）。
2. **Student**：$P_\theta(S)=\rho_\theta(|S|)e^{\sum s}/Z_{|S|}$，$\rho_\theta=(1-\alpha)\mathrm{PB}(s)+\alpha\bar\rho$。
3. **Loss**：set CE。唯一新增参数是 $\alpha$；没有 count head、没有 `decision_count_head.pt`。
4. **Inference**：同一个 DP + set MAP；count 分布就是 $(1-\alpha)\mathrm{PB}(s)+\alpha\bar\rho$，通过 $\mathrm{PB}(s)$ 保留 per-question 适应性。

净变化：

- 删除 $(d+1)(K+1)$ 的 count head 和 sidecar；
- 新增 1 个标量 $\alpha$（或 $d+1$ 的标量 probe）；
- 教师侧从 DP 变成排序 + 经验矩；
- 保留 coherent set CE、exact marginal、set MAP。

---

## 6. 实验阶梯（直接回答 head 是否必要）

| 臂 | 含义 |
|---|---|
| BCE | $\alpha=0$，严格基线 |
| α-mixture（全局 α） | 1 个标量，最小 count 修正 |
| α(x) 标量 probe | 最小 per-question count |
| 完整 count head（当前 PR） | 任意 per-question count |
| DP target vs empirical set target | independence 假设是否在起作用 |

结果怎么读：

- $\alpha\approx0$：count 信息没用，整个 set KD 退化回 BCE。
- $\alpha\approx1$：只需要全局 count 先验，不需要 per-question count。
- 中间：一参数 count law 够用，删 head。
- 完整 head 明显更好：per-question 任意 count 分布确实重要；保留 head，但在论文里证明。
- empirical target ≈ DP target：去掉 independence 假设，方法更干净。
- empirical target 明显更差：DP 的平滑/条件独立在起实质作用，需单独分析。

报告指标：overall / per-source / per-seed exact-set EM、micro-F1、count NLL/Brier/mode accuracy、count error、teacher coverage、membership Brier/BCE；按 K、count entropy、source 分层；paired question/group-level uncertainty。

另外补两个强对照：

- **count-only**：`BCE(q) + CE(ρ_teacher)`（只传 count，不传 membership-count 耦合）；
- **shuffled-count**：把 ρ 和另一道题的 membership 配对，打散耦合。如果 shuffled 也涨，说明收益不是来自 count。

---

## 7. 实现清单（给 ChatGPT）

### `harness/set_distillation.py`

- 新增 `implicit_count_probs(scores)`：用现有 `log_partitions` 和 `∏(1+e^{s_i})` 返回 $\mathrm{PB}_m$。
- 新增 `mixture_count_probs(scores, prior, alpha)`：返回 $\rho_\theta=(1-\alpha)\mathrm{PB}+\alpha\bar\rho$，校验归一化。
- 修改 `set_loss_terms`：支持 `count_law="mixture"`（传入 `prior` 和 `alpha`）以及 `count_law="head"`（保留当前实现做 ablation）；`count_target` 逻辑保持。
- `structured_prediction` 支持直接传入 `count_probs`（mixture 路径），不再要求 count logits；marginal/MAP 的 DP 不变。
- 新增测试：
  - `alpha=0` 时 scalar/torch loss 严格等于 `BCEWithLogits(q, s)`；
  - `alpha=1` 时 count 项为常数，模型 count 等于 prior；
  - mixture 归一化、$\sum_i p_i=\sum_m m\rho_m$、feasibility；
  - mixture 梯度与 finite difference / exact enumeration 一致；
  - 极端 logits、$b_i=0/1$、K=1/2 边界；
  - count marginal identity（$\sum_{|S|=m}P_\theta(S)=\rho_\theta(m)$）。

### `harness/train_multiselect.py`

- 新增 `--count-law {mixture,head}`，默认 `mixture`；`head` 保留当前 PR 路径做 ablation。
- `Readout`：mixture 路径不创建 `count_head`；新增 `alpha`（全局 `nn.Parameter`，或 `--alpha-probe` 时用单标量 linear probe）。
- checkpoint metadata 记录 `count_law`、`alpha`、`prior`（prior 以 JSON 存，K+1 向量）；mixture 路径不再写 `decision_count_head.pt`。
- `load_count_head` / `save_count_head` 只在 `head` 路径使用；`eval_multiselect.py` 按 metadata 分派。
- 训练/评估的 `structured_prediction` 调用改成传 `count_probs`。
- 保留 `--resume` 的现有语义（weight/step warm restart，fresh optimizer），不要悄悄改。

### `harness/build_multiteacher_targets.py`

- 新增 `--aggregate {conditional_bernoulli,empirical_map}`，默认建议先保留 `conditional_bernoulli` 以兼容旧 target，新增 `empirical_map` 做新实验。
- `empirical_map`：对每条链的 `per_chain_membership` 算 MAP set（allowed counts + top-m log-odds，tie 固定），再算经验 $q,\rho$；写同样的 `decision_set_target_v1` schema，`joint_semantics` 记为 `empirical_chain_sets`。
- 校验：每条链的 set 大小在 allowed counts 内；空 set 只在 `allow_empty` 时允许；`q`/`rho` 与 chain sets 一致；provenance 保留 chain ids/metadata。
- 测试：K=8 合成链、全 0/全 1、tie、truncated exclusion、missing policy。

### `harness/eval_multiselect.py`

- 按 `decision_readout.json` 的 `count_law` 分派：`mixture` 用 `prior`/`alpha` + `structured_prediction`；`head` 用现有 sidecar。
- `mixture` 路径缺少 sidecar 不应报错；`head` 路径仍然 fail closed。
- 保持 gold count 只出现在单独标注的 oracle diagnostic 中。

### `docs/`

- 把第 2 节的 count marginal identity、第 3 节的 α-mixture 和 α=0→BCE 证明写进 `docs/multiteacher_set_distillation.md`。
- 把第 1 节的 multi-information 命题写成一个 Proposition（含证明/说明）。
- 更新 `docs/verification_2026-10-22.md`：85/85 tests pass on a torch host；真实模型/FSDP/GPU 仍未验证。
- 在 README 里把“multi-teacher”改成更准确的“multi-view self-teacher”，避免 overclaim。

### 不要做的事

- 不要覆盖 `data/multiselect/` 的 archival 文件；新实验用 `data/multiselect_grouped_mixed/` 和新 run 目录。
- 不要把旧 58.63% 当作新方法结果。
- 不要在 `--objective set` 里混入 legacy graph penalty（现有检查已禁止）。
- 不要把 `q` 和 hard `rho` 拼成一个“moment”目标；mixture/α 路径是唯一合法的 count law。

---

## 8. 需要 ChatGPT 先定的几个选择

1. $\alpha$ 用**全局标量**还是 **scalar probe**？建议先全局标量（最小改动），scalar probe 作为第二臂。
2. prior $\bar\rho$ 用 **teacher 平均 ρ** 还是 **gold count 分布**？建议 train-only 估计，dev 选型；两者都报。
3. empirical target 用 **per-chain MAP set** 还是 **threshold 0.5 set**？建议 MAP，threshold 作为 ablation。
4. 是否保留完整 head 作为 ablation 臂？建议保留 `--count-law head`，否则无法回答“head 是否必要”。
5. mixture 的 $\alpha$ 初始化：建议 $\alpha=0$（从 BCE 出发），或 $\alpha=0.5$；在 docs 里写清楚。

---

## 9. 一句话总结

**不要删 count head 了事；数学上 count 边际就是 $\rho_\theta$，scores 改不动它。正确的化简是把 head 换成一参数的 $\alpha$-mixture（$\alpha=0$ 严格退化为 BCE），教师侧用 per-chain MAP set 的经验分布替代 conditional-independence DP。这样删掉 $(d+1)(K+1)$ 的 head 和 independence 假设，只新增 1 个标量和一次排序，同时保留 coherent set CE、exact marginal 和 set MAP。**

