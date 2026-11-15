---
title: argmax:AI 每说一个字,都是它在做决定
topics: []
status: draft
url:
id:
created: 2026-11-15
published:
tags: [math, deep-learning, argmax, softmax]
---

"取最大值,返回下标。"

"就这?"

"就这——但 GPT 每生成一个 token,最后一层都是它;分类模型每次预测,输出是它;你训练时不能直接用它的原因,也写在它的定义里。"

argmax 大概是被名字耽误得最厉害的函数:五个字母,三个坑。这篇文章把三个坑填了,再顺手讲清它和 softmax 那点事。

## 坑一:它返回的不是最大值,是"在哪里取得"

先分家。max 和 argmax 是两个问题:

- **max**:这座山**多高**?
- **argmax**:山顶**在哪**?

![max 标的是高度,argmax 标的是坐标](images/fig1_max_vs_argmax.png)

$$\max_{x \in [0,10]} f(x) \approx 1.98 \quad \text{(a number)} \qquad \arg\max_{x \in [0,10]} f(x) \approx 6.8 \quad \text{(a location)}$$

arg 是 argument(自变量)的缩写——argmax 回答的是"哪个自变量让函数最大"。

所以模型推理的最后一层,输出不是概率最高的那个概率值,而是**概率最高的那个词的下标**。拿到下标,查词表,吐出字。

你每天用的分类模型,`model.predict(x)` 的返回值,骨子里就是 `argmax(logits)`。

## 坑二:它可能不存在,也可能不唯一

数学上较真起来,argmax 其实是个"集合值"的东西:

$$\arg\max_{x \in X} f(x) \;=\; \{x \in X \;:\; f(x) = \sup_{t \in X} f(t)\}$$

什么时候这个集合非空?**Weierstrass 定理**:紧集上的连续函数必有最大值。(紧集可以粗糙理解成"有界且包含边界"——闭区间就是。)

两个经典的翻车现场:

**不存在**:$f(x) = x$ 在开区间 $(0, 1)$ 上。永远能更大,但永远够不着——上确界 1 取不到,argmax 是空集。

**不唯一**:$f(x) = x^2$ 在 $\{-1, +1\}$ 上。两个点打平,argmax 返回 $\{-1, +1\}$ 两个元素。

打平了怎么办?工程界的办法很朴素:**tie-breaking 约定**。NumPy 和 PyTorch 都默认"返回第一个遇到的"(平台间不保证一致,跨设备复现实验时这里埋过无数人的深夜)。

(顺便吐槽:这个"多个并列最大"的情况有个高大上的名字叫 non-unique maximizer,听起来很唬人,意思就是打平了。)

## 坑三:它不可导——这是 softmax 存在的全部理由

argmax 的输出是 one-hot:

$$\text{argmax}([1.0,\; 2.5,\; 4.0,\; 3.2]) = [0,\; 0,\; 1,\; 0]$$

输入动一点点,输出要么纹丝不动,要么整个跳一格。**阶梯函数,在跳变点没有导数。**

而训练需要梯度。

于是深度学习发明(准确说是从统计力学借来)了 softmax——argmax 的可导替身:

$$\text{softmax}(x_i) = \frac{e^{x_i/\tau}}{\sum_j e^{x_j/\tau}}$$

那个 τ 是温度。**温度越低,分布越冷**:高处的概率越滚越大,低处的越缩越小。

![τ 从 2 一路降到 0.1,概率质量逐步坍缩到最大值那一个格](images/fig2_temperature.png)

证明它真的收敛到 argmax,只要三行。设 $x^*$ 是唯一最大元,对任意 $i \ne *$:

$$\text{softmax}(x_i) = \frac{e^{x_i/\tau}}{e^{x^*/\tau} + \sum_j e^{x_j/\tau}} = \frac{1}{e^{(x_i - x^*)/\tau} \cdot (1 + \cdots)} \;\xrightarrow{\;\tau \to 0\;}\; 0$$

因为 $x_i - x^* < 0$,除以越来越小的 τ,指数奔向 $-\infty$。与此同时 $\text{softmax}(x^*) \to 1$。

**τ→0 的 softmax,就是 argmax。** 一个是硬邦邦的阶跃,一个是可导的软坡——温度旋钮控制它有多软。

训练时用 τ=1 的 softmax 留住梯度,推理时用 argmax 一锤定音。这不是两个不相干的操作,是同一个决定的两副面孔。

## 冷的时候:采样、温度与"AI 的性格"

生成式模型推理时有个旋钮叫 temperature,转的就是这个 τ。

- **τ = 0**:贪心解码,每步都 argmax。AI 变成复读机,同一个问题永远同一个答案
- **τ 升高**:分布变平,小概率词也有机会被抽中。AI 开始"发散",偶尔冒出没试过的表达

低温保守,高温冒险——这是人性,也是采样。

顺带一句:之前那篇投机解码里"验证后取最大概率路径"的贪心选择,底层同样是 argmax。它不显山露水,但所有生成系统的最后一厘米都是它。

## 尾声:一个最小的函数,一条分界线

argmax 之美在于它是条分界线:

$$\text{continuous (training, gradients, softmax)} \;\xrightarrow{\;\tau \to 0\;}\; \text{discrete (inference, decision, argmax)}$$

神经网络活在左边,你说出的每个字活在右边。

下次模型回复你一段话,可以想到:那段话的每一个 token,都经过了一次 argmax——几十万维的词表里,一次冷静的一锤定音。

**思考题**:如果允许你把 τ 设成负数会发生什么?(提示:看看 $e^{x_i/\tau}$ 里谁会爆炸。)

---

*参考:Weierstrass 极值定理(任意实分析教材);Goodfellow et al.《Deep Learning》第 6 章(softmax 与数值稳定);PyTorch 文档 torch.argmax 的 tie-breaking 说明。*
