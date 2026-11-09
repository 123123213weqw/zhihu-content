---
title: 投机解码全景:LLM 推理加速的数学、结构与六代演进
topics: []
status: draft
url:
id:
created: 2026-11-09
published:
tags: [deep-learning, inference, speculative-decoding]
---

为什么 DeepSeek 能一边把模型做大、一边把价格打到地板?为什么 vLLM、SGLang 的配置里 speculative decoding 一栏的选项越来越多——EAGLE、MTP、DFlash、DSpark?

这些名字共享同一个内核:**别让大模型一次只算一个 token**。本文把这个内核拆到公式级——每个结论给推导,每个机制给结构,每个数字给出处。比喻只做锚点,一句话,不铺陈。

**记号**:target 模型分布 $p$,draft 模型 $q$,草稿长度 $\gamma$,条件接受率 $\alpha$,接受长度 $\tau$,大模型单次前向耗时 $c$,draft 每 token 耗时 $\beta c$,模型参数量 $N$,权重字节数 $|W|$。

## 一、解码为什么慢:访存账的完整推导

### 1.1 从自回归到算术强度

自回归生成:第 $t{+}1$ 个 token 依赖第 $t$ 个输出,每步一次完整前向。batch $= b$ 时,每步

$$\text{FLOPs} \approx 2Nb, \qquad \text{Bytes} = |W| + \underbrace{2Lhd\,b\,t}_{\text{KV cache 读取}}$$

$L$ 为层数、$h$ 为头数、$d$ 为头维。KV 项随上下文 $t$ 线性增长——长上下文下 decode 会先撞 KV 访存墙,这是另一条优化线(PagedAttention、MLA)的领地;本文先看 $b=1$、短上下文的主体项 $|W|$。算术强度:

$$\text{AI}(b) = \frac{2Nb}{|W| + 2Lhd\,bt} \;\xrightarrow{\;b=1,\;t\ll|W|/2Lhd\;}\; \approx 1\;\text{FLOP/byte}$$

对照 H100 的机器平衡点 $989\text{TFLOPS} / 3.35\text{TB/s} \approx 295$:解码是教科书级 memory-bound,GPU 计算单元闲置约

$$1 - \frac{1}{295} \approx 99.7\%$$

速度上限由带宽封顶:$v_{\max} = \text{BW}/|W|$。70B fp16($|W|{=}140$GB)在单卡 H100 上 $v_{\max} \approx 24$ tok/s,与算力无关——堆算力零收益,只有省访存次数才有收益。

![prefill 一次访存并行算出全部位置;decode 一次访存只算 1 个位置](images/figA_membound.png)

### 1.2 三个直接推论(后文全部围绕它们)

**推论 1(prefill 是 compute-bound)**:prefill 一次前向算 $L_s$ 个输入位置,$\text{AI} \approx L_s \gg 295$,同一块 GPU 两种工况,decode 的闲置算力是"免费午餐"。

**推论 2(batch 的两面)**:$\text{AI}(b) \propto b$(短上下文近似),batch 越大越接近 compute-bound。所以**高并发服务本身就缓解 memory-bound,投机解码的边际收益随 batch 增大而衰减**;而低并发(单用户对话、agent 串行调用、端侧)是投机解码的主场。很多框架因此做动态开关:负载高时自动关投机。这一条是理解 DSpark"吞吐调度"的钥匙,§5.3 回收。

**推论 3(verify 近似免费)**:一次前向算 $\gamma{+}1$ 个位置(GEMM 变胖,$\text{AI}$ 提高 $\gamma{+}1$ 倍),在 memory-bound 区内耗时几乎不变——**一次访存,顺便多验证 $\gamma$ 个位置,不要白不要**。这就是投机解码的全部物理基础。

## 二、内核:猜与验的数学

### 2.1 三步流程

draft:小模型 $q$ 连采 $\gamma$ 个 token $x_1..x_\gamma$(链)/一棵树/一个块;verify:$p$ 一次前向算出 $t{+}1..t{+}\gamma{+}1$ 的分布;accept:接受一致前缀,首个分歧处按修正分布重采样一个,丢弃其后草稿。每轮一次大模型访存,期望产出 $\mathbb{E}[\tau]+1$ 个 token(重采样贡献 $+1$)。

![上排 4 次访存产 4 token;下排 1 次访存产 3 token](images/figB_timeline.png)

### 2.2 无损性证明(rejection sampling)

**命题**:草稿 $x\sim q$,以 $\min(1, p(x)/q(x))$ 概率接受;拒绝时从 $r(x) = \mathrm{norm}\big(\max(0, p(x)-q(x))\big)$ 重采样。则输出严格服从 $p$。

**证明**:总拒绝率

$$Z = 1 - \sum_x q(x)\min\left(1,\frac{p(x)}{q(x)}\right) = \sum_x \max\big(0, p(x)-q(x)\big)$$

$x$ 经两条互斥路径产出:

$$\Pr(x) = \underbrace{\min\big(q(x), p(x)\big)}_{\text{accept}} + \underbrace{Z\,r(x)}_{\text{reject \& resample}}$$

$p(x)\ge q(x)$ 时 $r(x)=\frac{p(x)-q(x)}{Z}$,$\Pr(x)=q(x)+p(x)-q(x)=p(x)$;$p(x)<q(x)$ 时 $r(x)=0$,$\Pr(x)=p(x)$。故 $\Pr\equiv p$。**证毕。**

两个工程推论:(i) $q$ 与 $p$ 差距只影响速度,不影响分布——**draft 可以放心大胆地小**;(ii) Medusa 后来的 typical acceptance 放宽了这条:接受准则改为截断的典型集采样,容忍小偏差换更高接受率,换来"近似无损"(原文实测质量损失可忽略)——严格与激进之间有谱系可选。

### 2.3 期望接受长度与加速比的完整模型

逐位置条件接受率 $\alpha_j$、独立近似下:

$$\mathbb{E}[\tau] = \sum_{j=1}^{\gamma}\prod_{i\le j}\alpha_i \approx \frac{\alpha(1-\alpha^{\gamma})}{1-\alpha}$$

加速比($c$ = target 前向,$\beta c$ = draft 每 token):

$$S(\gamma) = \frac{1+\mathbb{E}[\tau]}{1+\gamma\beta}, \qquad \frac{\partial S}{\partial \gamma} = 0 \Rightarrow \text{最优 } \gamma \text{ 存在内点}$$

代数字:$\alpha=0.8,\beta=0.02,\gamma=4$:$\mathbb{E}[\tau]\approx2.95$,$S\approx3.7\times$。三个可检验的极限:

- $\beta \to 0$(draft 免费):最优 $\gamma \to \infty$,但收益饱和于 $\frac{1+\alpha/(1-\alpha)}{1} = \frac{1}{1-\alpha}$($\alpha=0.8$ 时 5×)
- $\alpha \to 1$: $S \to \gamma{+}1$(完美草稿,验证即产出)
- 高并发(推论 2):$c$ 本身下降,verify 不再近似免费,$S \to 1$

![接受前缀 τ=3、拒绝位重采样、其后丢弃](images/figC_accept.png)

**KV cache 的隐藏账**:verify 那次前向会给 $\gamma{+}1$ 个位置都算出 KV;接受前缀之后,被拒位置的 KV 必须丢弃(回滚)。工程上用"先只保留共享前缀的 KV + 增量补算"或直接重算,各框架实现不同,这是投机解码系统的第一大坑。

## 三、草稿从哪来:六代演进

验证是死数学,变化的是 drafter 的结构与训练。四种挂法的结构对照:

![MTP 焊进预训练、Medusa 嫁接、EAGLE 旁听特征、dFlash 独立配对](images/figG_drafter.png)

### 3.1 独立小模型(2022-23)

直接配一个小模型。$\alpha$ 低(分布不对齐)、部署双模型。T5-XXL + T5-small 约 2-3×(Leviathan et al.)。

### 3.2 Medusa(2023.09):嫁接 $k$ 个头

冻结主干,顶部接 $k$ 个头(单层 transformer + 线性),第 $i$ 头直接预测 $x_{t+i}$(跳过中间 token 的串行依赖——每个头都从同一份顶层特征出发):

$$\mathcal{L} = \sum_{i=1}^{k}\text{CE}_i, \qquad \text{Medusa-2:} \quad \mathcal{L} + \lambda_{\text{conn}}\|\Delta h\|$$

Medusa-1 只训头;Medusa-2 解冻主干联合微调,$\mathcal{L}_{\text{conn}}$ 约束主干输出不漂移。2-3×。局限:头与主干特征"不够亲",深层头的预测质量掉得快。

### 3.3 EAGLE(2024):特征层自回归

关键观察:token 是离散有损投影,特征空间信息量更大、映射更平滑。drafter 单层,自回归预测**下一位置的主干特征**(不是 token):

$$\tilde h_{t+1} = f_\theta\big([\,h_t^{(L-1)},\;\text{Emb}(x_t)\,]\big),\quad x_{t+1} = \text{LMhead}_p(\tilde h_{t+1})$$

训练目标(主干冻结):对齐特征 + 蒸馏输出分布

$$\mathcal{L} = \big\|\tilde h - h\big\|^2 + \text{CE}\big(\text{LMhead}_p(\tilde h),\, x^*\big)$$

两阶段:static(target 前向收集 $(h_t, h_{t+1})$ 对)→ draft 阶段(drafter 自采样训练,贴近推理时"已跑偏"的状态分布)。EAGLE-2 加动态树(§4),EAGLE-3 改多层特征融合 + 对比目标,3-4×,成为开源事实标准。

### 3.4 MTP(2024-25):焊进预训练

Meta(Gloeckle et al. 2024)提出时是训练技术:多步前瞻缓解教师强制的暴露偏差。DeepSeek-V3 把它工程化为推理加速:主干倒数第二层特征 $h^0$ 与 token embedding 融合,过一个独立 transformer 层,共享主干 LM head:

$$h^{k} = \text{MTP}_k\big(\text{Proj}([h^{0}, \text{Emb}(x)])\big)$$

预训练联合损失,深度加权(浅头权重大、深头权重小,余弦衰减):

$$\mathcal{L} = \mathcal{L}_{\text{main}} + \lambda\sum_{k=1}^{\gamma}\frac{\gamma+1-k}{\gamma}\,\text{CE}_k$$

推理配置单层($\gamma=1$):主头出 $t{+}1$、MTP 头出 $t{+}2$,一次验证两位置。代入 §2.3:

$$S = \frac{1+\alpha}{1+\beta} \approx \frac{1.9}{1.05} \approx 1.8\times \quad (\alpha\approx0.9)$$

与 DeepSeek 官方 SGLang 部署的 1.8× 严丝合缝。**为什么推理只用一层**:单层 MTP 与主干共享 embedding/head,增量显存 $\approx$ 一个 transformer 层;$k$ 层 MTP 的收益受 $\alpha^k$ 衰减,而 KV 管理与验证宽度成本线性涨,1-2 层是甜点。Qwen3、GLM-4.5 原生携带 MTP 权重;Jetson 实测 Qwen 27B depth 3:13→35 tok/s(2.7×)。出厂预装,零配对成本——代价是与 checkpoint 绑死、上限低。

### 3.5 dFlash(2026):块并行

前四代 draft 都串行(第 $i$ 个依赖第 $i{-}1$ 的实际取值,延迟 $O(\gamma)$)。dFlash(Chen, Liang & Liu)改成完形填空:输入 = 1 个 anchor + $\gamma$ 个共享 mask 向量,块内**双向注意力**(每个空位看得见上下文和其他空位的存在,看不见它们的取值),一次前向全部位置同时出 token;KV 注入把 target 多层隐状态投影进 drafter 每层:

```python
x = concat(anchor_emb, mask_emb.repeat(gamma))    # γ+1 个输入位
h = drafter(x, context=kv_inject(target_hidden))  # 块内双向,一次前向
d = sample(lm_head(h[:, 1:]))                     # d_1 … d_γ 同时落地
```

注意力 mask 的矩阵形式(块内全可见,块外因果):

$$M[i,j] = \begin{cases}1 & j \le t \;\text{(prefix)}\\ 1 & i,j \in \text{block}\\ 0 & \text{else}\end{cases}$$

延迟 $O(1)$、与 $\gamma$ 无关 → 敢用深 drafter 和大 $\gamma$(推荐 depth 15)。代价是联合分布被因式分解:

$$q_{\text{dflash}}(x_1..x_\gamma) = \prod_{i=1}^{\gamma} q(x_i \mid x_{\le t}) \quad\text{vs 真值}\quad \prod_i q(x_i \mid x_{\le t},\, x_{t+1:i-1})$$

每个空位填的时候不知道别的空最终填了什么——像同一排人各写各的,意思可能打架(这就是"后缀衰减":块尾条件接受率掉到 0.63-0.78,§6 R2 图)。最优 $\gamma$ 的一阶条件也变了:draft 边际成本 $\approx 0$,$\gamma$ 上限只受"验证宽度成本 + 生存概率 $\prod\alpha_i$ 衰减"约束,所以块系天然押深。

![dFlash 机制:切块→mask 一次前向并行出块→逐位判决截断→回环](images/figD_dflash.png)

训练侧:mask 调度随机化(数量、位置、块长、预填比例全随机——推理只摆一种姿势,训练必须见过一百种),监督是 on-policy 的(用 target 跑语料拿隐状态与分布)。便宜了推理,费在训练管线,且专机专用。

### 3.6 DSpark(2026):并行 + 串行修正 + 置信调度

DSpark(北大 & DeepSeek,arXiv:2607.05147)在 dFlash 骨干上叠三个件:

**串行修正头**(Markov/RNN,把块内依赖补回来):

$$\hat x_i \sim g_\phi(x_i \mid \hat x_{i-1},\dots,\hat x_{1},\; \text{block}_i)$$

并行骨干管速度,一层串行头管一致性——开销仅全轮延迟的 0.2%-1.3%,接受长度比 dFlash 高 16-18%、比 EAGLE-3 高 27-31%。

**置信头** + 顺序温度缩放校准(STS):

$$c_i = \sigma(w^\top h_i + b), \qquad \tilde c_i = c_i^{1/T_i} \Big/ \text{norm},\quad T_i \text{ 仅依赖位置 } i$$

**硬件感知调度器**(§5.3 形式化)。

## 四、候选的组织:链、树、块

**链**:全对概率 $\alpha^\gamma$ 指数衰减。

**树**:每层 top-$k$ 扇出。树注意力把整棵树摊进一次前向,mask 只放行祖先:

$$\text{mask}(v,u) = \mathbf{1}\big[u \in \text{Anc}(v) \cup \{v\}\big]$$

期望接受长度 = 最深全对路径深度。$b_l$ 条近似独立(独立近似)的深度 $l$ 路径,至少一条全对:

$$\mathbb{E}[\tau_{\text{tree}}] = \sum_{l\ge1}\Pr\big(\exists\ \text{accepted path} \ge l\big) \approx \sum_{l\ge1}\Big(1-(1-\alpha^l)^{b_l}\Big)$$

同样预算下远高于链——多条路,总有一条走得远(直观锚点就这一句)。

![draft 树与树注意力 mask:每个位置只看祖先](images/figE_tree.png)

谱系:**Medusa** 多头笛卡尔积 + TOMD;**EAGLE-2** 动态树——按 drafter 置信度决定扩展形状(易句浅宽快走、难句深窄重押,接受长度 4.26);**Sequoia**(2024.02)把树形做成约束优化:

$$\max_{T}\ \mathbb{E}[\tau(T)] \quad \text{s.t.} \quad |T| \le B$$

用 DP 离线求解,结构结论:**最优树上宽下窄**(浅层接受率高、值得多押)。块系(dFlash/DSpark)不是树——块内无父子分支,是"扁平并行"。

## 五、深度的调度:静态与自适应

### 5.1 固定 $\gamma$ 必然次优

接受率随上下文难度波动:模板文本近乎确定,创作文本处处分歧。同一 $\gamma$ 对易上下文太浅(收益没吃满)、对难上下文太深(白 draft + 验证浪费):

![易上下文押深仍有收益,难上下文早饱和](images/figF_adaptive.png)

### 5.2 最优停止的形式化

继续押第 $j{+}1$ 个位置的边际期望产出是生存概率 $\times$ 置信:$a_j \cdot c_{j+1}$(其中 $a_j=\prod_{i\le j}c_i$)。最优停止条件:

$$a_j \cdot c_{j+1} \ge \kappa \quad \Leftrightarrow \quad \text{继续押;否则停}$$

$\kappa$ 是摊销的验证成本比。各方法的差别只在"怎么估 $c$":

| 方法 | $c$ 的估计 | 代价 |
|---|---|---|
| SVIP / AdaEDL | drafter 输出熵 $H(q_i)$(熵高=猜虚了) | 免训练;AdaEDL 比静态快 10-57% |
| SpecDec++ / DISCO | 学习 stopping head / 分类器 | 需训练 |
| DSpark | 置信头 + STS 校准(直接预测 $c_i$) | 一个头 |

### 5.3 DSpark 的系统级调度

把"逐请求深浅"与"引擎吞吐"联合优化。引擎实测每秒步数 $SPS(B)$ 随总验证宽度 $B$ 上升而下降(每步更胖),系统吞吐:

$$\Theta = \hat\tau(B)\cdot SPS(B), \qquad \hat\tau = \mathbb{E}_r\big[1+\textstyle\sum_j a_{r,j}\big]$$

调度器以 $\Theta$ 为目标贪心分配每条请求的验证宽度:高置信请求押深($a_{r,j}$ 缓降),低置信早验。生产(DeepSeek-V4)验证预算从 MTP-1 静态 2 动态扩到 4-6 token;同等吞吐 per-user +60~85%(V4-Flash)、+57~78%(V4-Pro);80 tok/s/user SLA 下系统吞吐 +51%。**这就是推论 2 的工程闭环:负载和深浅一起调度,而不是全局一个 $\gamma$。**

## 六、实测记分牌(论文原生数字)

接受长度(DSpark 论文 Table 1,九基准平均):

![DSpark 4.95-5.00 vs EAGLE-3 3.68-3.80 vs DFlash 4.26-4.29](images/figR1_accept.png)

位置衰减(同论文 Figure 2,Qwen3-4B):

![dFlash 块尾掉到 0.63-0.78;EAGLE-3 前低后高;DSpark 全程平稳](images/figR2_decay.png)

EAGLE-3 的"前低后高"值得展开:串行 drafter 首位置要吸收上下文、越猜越准;块并行首位置最准、越往后越散——两条路线的本质差异在数据上的指纹,同向交叉。

六代公开加速(各论文原生口径):

![从 2-3× 到生产 +60-85%,口径已标注](images/figR3_speed.png)

**口径警告**:离线低并发基准、生产同吞吐对比、端侧实测,三种口径不可直接横比——横比看量级,不排名次。

## 七、工程选择指南

- 模型自带 MTP 权重 → 开 MTP,depth 3 起步(零成本 2-3×)
- 要最广 checkpoint / llama.cpp → dFlash,depth 15 起步
- vLLM + 极限吞吐 → DSpark,depth 4 起步(llama.cpp 暂不支持)
- 自训模型 → EAGLE-3 管线最成熟

铁律:draft 必须与 target **精确变体**配对;接受率在真实 prompt 分布上实测;depth 在自己负载下调优——高并发场景注意 §1.2 推论 2,必要时让框架的动态开关干活。

## 八、总结:一个内核,三个维度

$$\underbrace{\text{draft \& verify}}_{\text{rejection sampling, 无损}} \times \underbrace{\text{drafter}}_{\text{small} \to \text{head} \to \text{feature} \to \text{MTP} \to \text{block}} \times \underbrace{\text{layout}}_{\text{chain} \to \text{tree} \to \text{block}} \times \underbrace{\text{depth}}_{\text{static} \to \text{adaptive}}$$

即:**猜-验内核**(三行证明保证无损)× **草稿结构**(旁听→嫁接→血脉相连→独立配对)× **候选组织**(链的指数衰减→树的总有一条路→块的深度零延迟)× **深度调度**(全局固定→逐请求置信×硬件联合调度)。物理本质一句话——

$$\text{speedup} = \frac{\text{idle compute, now used}}{\text{memory traffic saved}}$$

树系与块系的对决打到中盘,训练侧与推理侧的合流刚刚开始。唯一不变的是那个三行证明的承诺:**快,不牺牲任何一个 token 的正确性。**

---

*参考:Leviathan et al. 2023;Chen et al. 2023(Speculative Sampling);Cai et al. 2024(Medusa);Li et al. 2024-25(EAGLE 1-3);Gloeckle et al. 2024(MTP);DeepSeek-AI 2024(V3 报告);Chen et al. 2024(Sequoia);Cheng et al. 2026(DSpark, arXiv:2607.05147);Kai et al. 2024(AdaEDL);Jetson AI Lab(Speculative Decoding on Jetson)*
