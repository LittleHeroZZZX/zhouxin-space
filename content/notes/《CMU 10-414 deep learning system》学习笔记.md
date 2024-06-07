---
title: 《CMU 10-414 deep learning system》学习笔记
tags:
  - CUDA
  - 深度学习系统
date: 2024-05-28T12:24:00+08:00
lastmod: 2024-06-07T18:34:00+08:00
publish: true
dir: notes
slug: notes on cmu 10-414 deep learning system
images:
  - https://pics.zhouxin.space/202406041918872.png?x-oss-process=image/quality,q_90/format,webp
math: "true"
---

# Lecture 1: Introduction and Logistics

## 课程的目标

本课程的目标是学习现代深度学习系统，了解包括自动微分、神经网络架构、优化以及 GPU 上的高效操作在内的技术的底层原理。作为实践，本课程将实现一个 needle（deep learning library）库，类似 PyTorch。

## 为什么学习深度学习系统？

为什么学习？深度学习这一概念很早就存在了，但直到 PyTorch、TensorFlow 此类现代深度学习框架发布，深度学习才开始迅速发展。简单易用的自动差分库是深度学习发展的最大动力。

除了使用这些库，我们为什么还要学习深度学习系统？

- 为了构建深度学习系统  
如果想要从事深度学习系统的开发，那毫无疑问得先学习它。目前深度学习框架并没完全成熟，还有很多开发新功能，乃至新的框架的机会。

- 为了能够更高效地使用现有系统  
了解现有系统的内部实现，可以帮助我们写出更加高效的深度学习代码。如果想要提高自定义算子的效率，那必须先了解相关操作是如何实现的。

- 深度学习系统本身就很有趣  
尽管这个系统看上去很复杂，但是其核心算法的原理确实相当简单的。两千行左右的代码，就可以写出一个深度学习库。

## 预备知识

- systems programming
- 线性代数
- 其他数学知识：计算、概率、简单的证明
- Python 和 C++ 经验
- 机器学习的相关经验

# Lecture 2: ML Refresher & Softmax Regression

## 机器学习基础

深度学习是由数据驱动的，所谓数据驱动，这意味着当我们想要写一个用于识别手写数字的模型时，我们关注的不是某个数字形状上有什么特点，如何通过编程识别该特点，而是直接将数据集喂给模型，模型自动训练并识别数字类别。

深度学习模型由三部分组成：

- 假说模型：模型的结构，包括一系列参数，其描述了模型从输入到输出的映射关系；
- 损失函数：指定了对模型的评价，损失函数值越小，说明该模型在指定任务上完成得更好；
- 优化方法：用于对模型中参数进行优化，使得损失函数最小的方法。

## Softmax 回归

以经典的 softmax 回归模型为例，简单回顾一下 ML 模型。

考虑一个 k 分类任务，其中数据集为 $x^{(i)} \in R^n\ ,\  y^{(i)} \in \{ 1,...,k\}\   \ \  i = 1,...,m$，$n$ 标识输入数据集的维度，$k$ 标识标签类别数，$m$ 标识数据集样本数量。

一个假说模型就是将一个 $n$ 维的输入映射到一个 $k$ 维的输出，即：$h: R^n \rightarrow R^k$。注意，模型并不会直接输出类别的序号，而是通过输出一个 $k$ 维向量 $h(x)$，其中第 $i$ 个元素 $h_i(x)$ 表示是第 $i$ 个类别的概率。

对于线性模型来说，使用 $\theta \in R^{n\times k}$ 这个模型中的参数，那么 $h_\theta(x) = \theta^T x$。

如果一次输入多个数据，那么输入数据就可以组织成一个矩阵，相比起多个向量操作，矩阵的操作通常效率更高，我们在代码实现中一般也是用矩阵操作。数据集可以表示为：

{{< math_block >}}
X\in R^{m\times n}=\left[ \begin{array}{c}
	x^{(1)T}\\
	\vdots\\
	x^{\left( m \right) T}\\
\end{array} \right] ,  y\in \left\{ 1,...,k \right\} ^m=\left[ \begin{array}{c}
	y^{\left( 1 \right)}\\
	\vdots\\
	y^{\left( m \right)}\\
\end{array} \right]
{{< /math_block >}}

数据集的矩阵是一个个样本转置后堆叠 stack 起来的。那么输出可以表示为：

{{< math_block >}}
h_{\theta}\left( X \right) =\left[ \begin{array}{c}
	h_{\theta}\left( x^{\left( 1 \right)} \right) ^T\\
	\vdots\\
	h_{\theta}\left( x^{\left( m \right)} \right) ^T\\
\end{array} \right] =\left[ \begin{array}{c}
	x^{\left( 1 \right) T}\theta\\
	\vdots\\
	x^{\left( m \right) T}\theta\\
\end{array} \right] =X\theta
{{< /math_block >}}

关于损失函数 $l_{err}$，一种朴素的想法是将模型预测错误的模型数据量作为损失函数，即如果模型预测的正确率最高的那个类别与真实类别不相同，则损失函数为 1，否则为 0：

{{< math_block >}}
l_{err}\left( h\left( x \right) , y \right) \,\,=\,\,\left\{ \begin{aligned}
	0 \ &\mathrm{if} \ \mathrm{argmax} _i\,\,h_i\left( x \right) =y\\
	1 \ &\mathrm{otherwise}\\
\end{aligned} \right.
{{< /math_block >}}

遗憾的是，这个符合直觉函数是不可微分的，难以对参数进行优化。更合适的做法是使用交叉熵损失函数。

在此之前，我们将先讲输出过一个 softmax 函数，使之的行为更像一个概率——各个类别的概率之和为 1：

{{< math_block >}}
z_i=p\left( \mathrm{label}=i \right) =\frac{\exp \left( h_i\left( x \right) \right)}{\sum_{j=1}^k{\exp \left( h_j\left( x \right) \right)}}
{{< /math_block >}}

那么交叉熵损失函数就可以定义为：

{{< math_block >}}
l_{ce}\left( h\left( x \right) ,y \right) =-\log p\left( \mathrm{label}=y \right) =-h_y\left( x \right) +\log \sum_{j=1}^k{\exp \left( h_j\left( x \right) \right)}
{{< /math_block >}}

注意在计算交叉熵时，通过运算进行了化简，这使得我们可以省去计算 softmax 的过程，直接计算最终的结果。不但如此，交叉熵的计算中，如果 $h_i(x)$ 的值很小，那么取对数会出现很大的值，化简后的计算则避免了这种情况。

所有的深度学习问题，都可以归结为一下这个最优化问题：{{< math_block >}}
\mathop {\mathrm{minimize}} \limits_{\theta}\ \ \frac{1}{m}\sum_{i=1}^m{l(h_{\theta}(x^{(i)}),y^{(i)}))}
{{< /math_block >}}
我们使用梯度下降法对该问题进行优化。在此之前，首先介绍一下关于梯度。我们的优化目标可以看作一个关于$\theta \in R^{n\times k}$的函数$f$，那么其在$\theta_0$处的梯度可以表示为：
{{< math_block >}}
\nabla _{\theta}f\left( \theta _0 \right) \in R^{n\times k}=\left[ \begin{matrix}  
	\frac{\partial f\left( \theta _0 \right)}{\partial \theta _{11}}&		\cdots&		\frac{\partial f\left( \theta _0 \right)}{\partial \theta _{k1}}\\  
	\vdots&		\ddots&		\vdots\\  
	\frac{\partial f\left( \theta _0 \right)}{\partial \theta _{n1}}&		\cdots&		\frac{\partial f\left( \theta _0 \right)}{\partial \theta _{nk}}\\  
\end{matrix} \right]
{{< /math_block >}}
其中，第$i$行第$j$个元素表示除$\theta_{ij}$之外的参数都被当作常数，对$\theta_{ij}$求偏导。

梯度下降，就是沿着梯度方向不断进行迭代，以求找到最佳的$\theta$使得目标函数值最小。
{{< math_block >}}
\theta :=\theta _0-\alpha \nabla f\left( \theta _0 \right)
{{< /math_block >}}
上式中，$\alpha$被称为学习率或者步长。

事实上，在现代深度学习中，并不是使用的传统梯度下降的方案，因为其无法将所有训练集一次性读入并计算梯度。现代使用的是随机梯度下降（Stochastic Gradient Descent，SGD）

首先将m个训练集样本划分一个个小batch，每个batch都有B条数据。那每一batch的数据表示为$X\in R^{B\times n}$，更新参数$\theta$的公式变为：
{{< math_block >}}
\theta :=\theta _0-\frac{\alpha}{B}\nabla f\left( \theta _0 \right)
{{< /math_block >}}
我们的梯度变成了每个小batch对全体样本梯度的估计。

那如何计算梯度表达式呢？梯度矩阵中每个元素都是一个偏导数，我们就先从计算偏导数开始。假设$h$是个向量，我们来计算偏导数$\frac{\partial l_{ce}\left( h,y \right)}{\partial h_i}$：
{{< math_block >}}
\begin{align}  
\frac{\partial l_{ce}\left( h,y \right)}{\partial h_i}&=\frac{\partial}{\partial h_i}\left( -h_y+\log \sum_{j=1}^k{\exp h_j} \right)  
\\  
&=-1\left\{ i=y \right\} +\frac{\exp \left( h_j \right)}{\sum_{j=1}^k{\exp h_j}}  
\\  
&=-1\left\{ i=y \right\} +\mathrm{softmax} \left( h \right)  
\\  
&=z-e_y  
\end{align}
{{< /math_block >}}

如果$h$是个向量，那么梯度$\nabla_h l_{ce}(h,y)$就能够以向量的形式表示为：
{{< math_block >}}
\nabla_h l_{ce}(h,y) = z-e_y
{{< /math_block >}}
这里我们将对$h$进行softmax标准化记为$z$，$e_y$表示对应的单位向量。

事实上，我们要计算的梯度是关于$\theta$的，具体来说，表达式为$\nabla_\theta l_{ce}(\theta^Tx,y)$，其中，$\theta$是个矩阵。或许，可以使用链式法则进行求解，但是太麻烦了，这里还涉及矩阵对向量的求导。我们需要一种更加通用的求导方案。

有两个解决办法：
- 正确且官方的做法：使用矩阵微分学、雅可比矩阵、克罗内克积和向量化等知识进行求解。
- 一个hacky、登不上台面、但大家都在用的方案：将所有的矩阵和向量当作标量，使用链式法则求解，并进行转置操作使得结果的size符合预期，最后检查数值上结果是否正确。

按照第二个方法的逻辑，过程为：
{{< math_block >}}
\begin{align}  
\frac{\partial}{\partial \theta}l_{ce}\left( \theta ^Tx,y \right) &=\frac{\partial l_{ce}\left( \theta ^Tx,y \right)}{\partial \theta ^Tx}\cdot \frac{\partial \theta ^Tx}{\partial \theta}  
\\  
&=\left[ z-e_y \right] _{k\times 1}\cdot x_{n\times 1}  
\\  
&=x\cdot \left[ z-e_y \right]  
\end{align}
{{< /math_block >}}
其中，$z=\text{softmax}(\theta^Tx)$。注意，倒数第二步求出的结果是两个列向量相乘，不能运算。又已知结果应该是$n\times k$的矩阵，调整向量之间的顺序即可。

照猫画虎，可以写出batch的情况，$X\in R^{B\times n}$：
{{< math_block >}}
\begin{align}  
\frac{\partial}{\partial \theta}l_{ce}\left( \theta ^TX,y \right) &=\frac{\partial l_{ce}\left( \theta ^TX,y \right)}{\partial \theta ^TX}\cdot \frac{\partial \theta ^TX}{\partial \theta}  
\\  
&=\left[ Z-E_y \right] _{B\times k}\cdot X_{B\times n}  
\\  
&=X^T\cdot \left[ Z-E_y \right]  
\end{align}
{{< /math_block >}}

# Lecture 3: Manual Neural Networks
这节课，我们将人工实现全连接神经网络，之后的课程，将引入自动微分技术。
## 从线性模型转变为非线性模型
![image.png](https://pics.zhouxin.space/202406071816708.png?x-oss-process=image/quality,q_90/format,webp)

如上图所示，线性模型本质上是将样本空间划分为线性的几个部分，这样的模型性能十分有限，因此很多不满足这样分布的实际问题就不能被解决。

一种解决方案是，在将样本输入到线性分类器前，先人工挑选出某些特征，即对$X$应用一个函数$\phi$，其将$X$映射到$\phi(X)$上，映射后的空间可以被线性划分。一方面，它确实是早期实践中行之有效的方案；另一方面，人工提取特征的泛化性能有限，受限于具体问题和研究人员的对问题的洞察程度。

如果我们使用线性网络提取特征，并直接接上一个线性分类头，这两个线性层等效为一个线性层，并不能做到非线性化的要求（基础知识，此处不再解释）。

因此，在使用线性网络提取特征后，需要再接上一个非线性函数$\sigma$，即$\phi = \sigma (W^T X)$。
## 神经网络
上文提到的使用非线性函数后的模型，就可以视作一种最简单的神经网络。所谓神经网络，值得是机器学习中某一类特定的假说模型，其由多层组成，每一层都有大量可以微分的参数。

神经网络最初的确起源于模拟人类神经元这一动机，但随着其发展，越来越多的神经网络模型出现，与人类大脑神经网络越来越不相关。

以双层神经网络为例，其形式化表示为$h_\theta(x) = W_2^T \sigma(W_1^T x)$，所有可学习的参数使用$\theta$表示。以batch的矩阵形式表示为：
{{< math_block >}}
h_\theta(X) = \sigma(XW_1)W_2
{{< /math_block >}}
接下来给出L层多层感知机（a.k.a. MLP、前馈神经网络、全连接层）的形式化表达：
{{< math_block >}}
\left\{\begin{array}{l}  
Z_{i+1} = \sigma_i(Z_iW_i), i=1,...,L  \\  
Z_1 = X\\  
h_\theta(X) =Z_{L+1}\\  
[Z_i\in R^{m\times n_i}, W_i \in R^{n_i\times n_{i+1}}]\\  
\sigma_i:R\rightarrow R

\end{array} \right.
{{< /math_block >}}
每一层的输入为$Z_i$，输出为$Z_{i+1}$。

为什么要是用深度网络而不是宽度网络？没有很完美的解释，但最好并且最现实的解释是：经验证明，当参数量固定时，深度网络性能优于宽度网络。
## 反向传播（梯度计算）
与Lecture 2一致，使用交叉熵作为损失函数，使用SGD作为优化算法，唯一的区别是，这次要对MLP网络求解梯度。

对于两层神经网络$h_\theta(X) = \sigma(XW_1)W_2$，待求的梯度表达式为：
{{< math_block >}}
\nabla_{\{W_1, W_2\}}l_{ce}(\sigma(XW_1)W_2,y)
{{< /math_block >}}
对于$W_2$的梯度，其与Lecture 2的计算类似：
{{< math_block >}}
\begin{align}  
\frac{\partial l_{ce}(\sigma(XW_1)W_2,y)}{\partial W_2}&=\frac{\partial l_{ce}(\sigma(XW_1)W_2,y)}{\partial \sigma(XW_1)W_2} \cdot \frac{\partial\sigma(XW_1)W_2}{\partial W_2}\\  
&=(S-I_y)_{m\times k}\cdot \sigma(XW_1)_{m\times d}\\  
&=\sigma(XW_1)^T\cdot (S-I_y)\\  
&[S=\text{softmax}(\sigma(XW_1))]  
\end{align}
{{< /math_block >}}

对于$W_1$的梯度，其需要多次应用链式法则，但并不难计算：
{{< math_block >}}
\begin{align}  
\frac{\partial l_{ce}(\sigma(XW_1)W_2,y)}{\partial W_1}&=\frac{\partial l_{ce}(\sigma(XW_1)W_2,y)}{\partial \sigma(XW_1)W_2} \cdot \frac{\partial\sigma(XW_1)W_2}{\partial \sigma(XW_1)}\cdot \frac{\partial \sigma(XW_1)}{\partial XW_1}\cdot\frac{\partial XW_1}{\partial X_1}\\  
&=(S-I_y)_{m\times k}\cdot [W_2]_{d\times k}\cdot \sigma\prime(XW_1)_{m\times d}\cdot X_{m\times n}\\  
&=X^T\cdot [\sigma\prime(XW_1)\odot((S-I_y)\cdot W_2^T)]\\  
&[S=\text{softmax}(\sigma(XW_1))]  
\end{align}
{{< /math_block >}}
以上公式中$\odot$表示逐元素乘法。至于为啥这么算，俺也不知道。

接下来将其推广到一般情况，即$L$层的MLP中对$W_i$求导：
{{< math_block >}}
\begin{align}  
\frac{\partial l(Z_{l+1},y)}{\partial W_i} &=\frac{\partial l}{\partial Z_{l+1}}\cdot \frac{\partial Z_{l+1}}{\partial Z_{l}}\cdot...\cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}}\cdot\frac{\partial Z_{i+1}}{\partial W_{i}}\\  
&=G_{i+1}\cdot\frac{\partial Z_{i+1}}{\partial W_{i}}=\frac{\partial l}{\partial Z_{i+1}}\cdot \frac{\partial Z_{i+1}}{W_i}\\

\end{align}
{{< /math_block >}}

由上述公式，我们可以得到一个反向迭代计算的$G_i$，即：
{{< math_block >}}
\begin{align}  
G_i &= G_{i+1}\cdot \frac{Z_{i+1}}{Z_i} \\  
&=G_{i+1}\cdot \frac{\partial \sigma(Z_iW_i)}{\partial Z_iW_i}\cdot\frac{\partial Z_iW_i}{Z_i}\\  
&=G_{i+1}\cdot \sigma\prime(Z_iW_i)\cdot W_i\\  
\end{align}
{{< /math_block >}}

上面的计算都是将矩阵当作标量进行的，接下来我们考虑其维度。已知，$Z_i \in R^{m\times n_i}$是第$i$层的输入，$G_i = \frac{\partial l}{\partial Z_{i}}$，其维度如何呢？$G_i$每个元素表示损失函数$l$对第$i$层输入的每一项求偏导，也可以记作是$l$对$Z_i$求梯度，即$\nabla_{Z_i} l$，其维度显然是$m\times n_i$，继续计算前文$G_i$：
{{< math_block >}}
\begin{align}  
G_i &=[G_{i+1}]_{m\times n_{i+1}}\cdot \sigma\prime(Z_iW_i)_{m\times n_{i+1}}\cdot [W_i]_{n_i\times n_{i+1}}\\  
&= [G_{i+1}\odot \sigma\prime(Z_iW_i)]W_i^T  
\end{align}
{{< /math_block >}}

有了$G_i$，就可以继续计算$l$对$W_i$的偏导数了：
{{< math_block >}}
\begin{align}  
\frac{\partial l(Z_{l+1},y)}{\partial W_i} &=G_{i+1}\cdot\frac{\partial Z_{i+1}}{\partial W_{i}} \\  
&=G_{i+1}\cdot \frac{\partial\sigma(Z_iW_i)}{\partial Z_iW_i}\cdot\frac{\partial Z_iW_i}{\partial W_i}\\  
&=[G_{i+1}]_{m\times n_{i+1}}\cdot \sigma\prime(Z_iW_i)_{m\times n_{i+1}}\cdot [Z_i]_{m\times n_i}\\  
&=Z_i^T\cdot[G_{i+1}\odot\sigma\prime(Z_iW_i)]  
\end{align}
{{< /math_block >}}

至此，每个小组件都已制造完毕，让我们来把它装起来吧！
- 前向传播
	- 初始化：$Z_1 = X$
	- 迭代：$Z_{i+1} = \sigma(Z_iW_i)$ 直至$i=L$（注意，最后一层没有非线性部分，此处没有展示出来）
- 反向传播
	- 初始化：$G_{L+1} = S-I_y$
	- 迭代：$G_i=[G_{i+1}\odot \sigma\prime(Z_iW_i)]W_i^T$ 直至$i=1$
值得注意的是，在反向传播中，需要用到前向传播的中间结果$Z_i$。为了更高效地计算梯度，不得不以牺牲内存空间为代价，即空间换时间。

> 许多课程，讲到这里就结束了，但对我们这门课来说，才刚刚开始...

# Lecture 4: Automatic Differentiation
## 基本工具
- 计算图
计算图是自动微分中常用的一种工具。计算图是一张有向无环图，每个节点表示（中间结果）值，每条边表示输入输出变量。例如，$y=f(x_1, x_2) = \ln(x_1)+x_1x_2-\sin x_2$对应的计算图为：
![](https://pics.zhouxin.space/202406071612073.webp?x-oss-process=image/quality,q_90/format,webp)
按照拓扑序列遍历这张图，就可以得到对应表达式的值。
## 对自动微分方法的简单介绍
深度学习中，一个核心内容就是计算梯度。这里介绍集中计算梯度的方案：
- 偏导数定义
- 
梯度是由一个个偏导数组成的，可以直接根据偏导数的定义来计算梯度：
{{< math_block >}}
\frac{\partial f(\theta)}{\partial \theta_i} = \lim_{\epsilon \to 0}\frac{f(\theta + \epsilon e_i) - f(\theta)}{\epsilon}
{{< /math_block >}}
其中，$e_i$是表示第$i$个方向上的单位向量。

- 数值求解
根据上述定义，我们可以选取一个很小的量代入$\epsilon$，得到数值计算偏导的方法：
{{< math_block >}}
\frac{\partial f(\theta)}{\partial \theta_i} = \frac{f(\theta + \epsilon e_i) - f(\theta - \epsilon e_i)}{2\epsilon} + o(\epsilon^2)
{{< /math_block >}}
这里并不是直接使用第一项的公式，即分子不是$f(\theta + \epsilon e_i) - f(\theta)$，并且误差项是$\epsilon^2$，这是由于泰勒展开：
{{< math_block >}}
\begin{align}  
f(\theta+\delta) = f(\theta)+f^\prime (\theta)\delta+\frac{1}{2}f^{\prime \prime}(\theta)\delta^2+o(\delta^3)\\  
f(\theta-\delta) = f(\theta)+f^\prime (\theta)\delta-\frac{1}{2}f^{\prime \prime}(\theta)\delta^2+o(\delta^3)  
\end{align}
{{< /math_block >}}
上述两式作差，即可得到数值计算$f^\prime(\theta)$的方法。

这个方法的问题在于存在误差，并且效率低下（这里要计算两次f），该方法常用于验证其它方法的具体实现是否出错。具体来说，验证如下等式是否成立：
{{< math_block >}}
\delta^T \nabla_\theta f(\theta) = \frac{f(\theta + \epsilon \delta) - f(\theta - \epsilon \delta)}{2 \epsilon} + o(\epsilon^2)
{{< /math_block >}}
其中$\delta$是单位球上的某个向量，$\nabla_\theta f(\theta)$是使用其它方法计算得到的梯度。等式左边是其它方法计算的梯度在$\delta$上的投影，右侧是使用数值求解得到的梯度值，验证该等式是否成立就可以判断左侧梯度是否计算错误。

- 符号微分
符号微分，就是根据微分的计算规则使用符号手动计算微分。部分规则为：
{{< math_block >}}
\begin{align}  
&\frac{\partial (f(\theta) + g(\theta))}{\partial \theta} = \frac{\partial f(\theta)}{\partial \theta} + \frac{\partial g(\theta)}{\partial \theta}\\  
&\frac{\partial (f(\theta) g(\theta))}{\partial \theta} = g(\theta) \frac{\partial f(\theta)}{\partial \theta} + f(\theta) \frac{\partial g(\theta)}{\partial \theta}\\  
&\frac{\partial f(g(\theta))}{\partial\theta}=\frac{\partial f(g(\theta))}{\partial g(\theta)}\frac{\partial g(\theta)}{\partial\theta}  
\end{align}
{{< /math_block >}}
根据该公式，可以计算得到$f(\theta) = \prod_{i=1}^{n} \theta_i$的梯度表达式为：$\frac{\partial f(\theta)}{\partial \theta_k} = \prod_{j \neq k}^{n} \theta_j$。如果我们根据该公式来计算梯度，会发现需要计算$n(n-2)$次乘法才能得到结果。这是因为在符号运算的过程中，我们忽略了可以反复利用的中间结果。

- 正向模式自动微分 forward mode automatic differentiation
沿着计算图的拓扑序列，同样可以计算出输出关于输入的导数，还是以$y=f(x_1, x_2) = \ln(x_1)+x_1x_2-\sin x_2$为例，其计算图为：
![image.png](https://pics.zhouxin.space/202406071612328.png?x-oss-process=image/quality,q_90/format,webp)


整个梯度计算过程如下，在此过程中应用到了具体函数的求导公式：
{{< math_block >}}
\begin{aligned}  
&\dot\nu_{1} =1 \\  
&\dot\nu_{2} =0 \\  
&\dot{\nu}_{3} =v_{1}/v_{1}=0.5 \\  
&\dot{\nu}_{4} =\hat{v}_{1}v_{2}+v_{2}v_{1}=1\times5+0\times2=5 \\  
&\dot\nu_{5} =\dot{v_{2}}\cos v_{2}=0\times\cos5=0 \\  
&\dot{\nu}_{6} =v_{3}+v_{4}=0.5+5=5.5 \\  
&\dot{\nu}_{7} =\dot{v_{6}}-\dot{v_{5}}=5.5-0=5.5  
\end{aligned}
{{< /math_block >}}

对于$f:\mathbb{R}^n \to \mathbb{R}^k$，前向传播需要$n$次前向计算才能得到关于每个输入的梯度，这就意味前向传播适合$n$比较小、$k$比较大的情况。但是在深度学习中，通常$n$比较大、$k$比较小。

- 反向模式自动微分
定义$\text{adjoint}:\overline{v_i}=\frac{\partial y}{\partial v_i}$,其表示
整个计算过程如下所示，需要注意的是$\overline{v_2}$的计算过程，其在计算图上延伸出了两个节点，因此梯度也由两部分相加：
{{< math_block >}}
\begin{align}  
&\overline{v_{7}}=\frac{\partial y}{\partial v_{7}}=1\\  
&\overline{v_{6}}=\overline{v_{7}}\frac{\partial v_{7}}{\partial v_{6}}=\overline{v_{7}}\times1=1\\  
&\overline{v_{5}}=\overline{v_{7}}\frac{\partial v_{7}}{\partial v_{5}}=\overline{v_{7}}\times(-1)=-1\\  
&\overline{v_{4}}=\overline{v_{6}}\frac{\partial v_{6}}{\partial v_{4}}=\overline{v_{6}}\times1=1\\  
&\overline{v_{3}}=\overline{v_{6}}\frac{\partial v_{6}}{\partial v_{3}}=\overline{v_{6}}\times1=1\\  
&\overline{v_{2}}=\overline{v_{5}}\frac{\partial v_{5}}{\partial v_{2}}+\overline{v_{4}}\frac{\partial v_{4}}{\partial v_{2}}=\overline{v_{5}}\times\cos v_{2}+\overline{v_{4}}\times v_{1}\\  
&\overline{v_{1}}=\overline{v_{4}} \frac{\partial v_{4}}{\partial v_{1}}+\overline{v_{3}} \frac{\partial v_{3}}{\partial v_{1}}=\overline{v_{4}}\times v_{2}+ \overline{v_{3}} \frac{1}{v_{1}}=5+\frac{1}{2}=5.5

\end{align}
{{< /math_block >}}

接下来我们讨论一下为什么前文中$\overline{v_2}$由两部分组成。考虑如下一个计算图：
![image.png](https://pics.zhouxin.space/202406071612078.png?x-oss-process=image/quality,q_90/format,webp)

$y$可以被视作关于$v_2$和$v_3$的函数，即$y = f(v_2, v_3)$，那么：
{{< math_block >}}
\overline{v_{1}}=\frac{\partial y}{\partial v_{1}}=\frac{\partial f(v_{2},v_{3})}{\partial v_{2}}\frac{\partial v_{2}}{\partial v_{1}}+\frac{\partial f(v_{2},v_{3})}{\partial v_{3}} \frac{\partial v_{3}}{\partial v_{1}}=\overline{v_{2}} \frac{\partial v_{2}}{\partial v_{1}}+\overline{v_{3}} \frac{\partial v_{3}}{\partial v_{1}}
{{< /math_block >}}
因此，定义partial adjoint $\overline{v_{i\to j}} = \overline{v_j} \frac{\partial v_j}{\partial v_i}$，那么$\overline{v_i}$可以表示为：
{{< math_block >}}
\overline{\nu_i}=\sum_{j\in next(i)}\overline{\nu_{i\rightarrow j}}
{{< /math_block >}}

## 反向模式微分算法的实现
基于以上分析，可以写出如下的实现反向模式微分算法的伪代码：
![image.png](https://pics.zhouxin.space/202406071612188.png?x-oss-process=image/quality,q_90/format,webp)

其中`node_to_grad`是一个字典，保存着每个节点的partial adjoint值。由于是按照逆拓扑序列遍历的节点，因此可以保证当遍历到$i$时，所有以$i$为输入的节点（k节点所在的集合）都已被遍历完毕，即$\overline{v_k}$已经计算出来。

那么partial adjoint值使用什么数据结构保存呢？一个常见的思路是使用邻接矩阵，但是这个矩阵中有大量元素是不存在了，空间浪费很大。我们可以在原有计算图的基础上进行拓展来保存partial adjoint和adjonitzhi之间的计算关系。

如下图所示，黑色部分是原表达式的计算图，红色部分是将adjoint和partial adjount的计算图：
![image.png](https://pics.zhouxin.space/202406071611419.png?x-oss-process=image/quality,q_90/format,webp)




使用计算图，除了能够节省内存外，还能清楚的看到正向计算的中间结果和反向计算之间的依赖关系，进而优化计算。

## 反向模式ad和反向传播的区别
![image.png](https://pics.zhouxin.space/202406071817738.png?x-oss-process=image/quality,q_90/format,webp)

反向传播：
- 在反向计算过程中使用与前向传播完全相同的计算图
- 应用于第一代深度学习框架

反向AD：
- 为adjoint在计算图中创建独立的节点
- 被应用于现代深度学习框架

现代普遍应用反向AD的原因：
- 某些损失函数是关于梯度的函数，这种情况下需要计算梯度的梯度，但反向传播就不能计算此类情况，而在反向AD中只要增加一个节点后在此计算梯度即可；
- 反向AD优化空间更大。

## 考虑Tensor的反向模式AD
前面都是在假设中间变量是标量的基础上讨论的，接下来我们将其推广到Tensor上。

首先推广adjoint，定义对于一个Tensor$Z$，其adjoint$\overline{Z}$为：
{{< math_block >}}
=\begin{bmatrix}\frac{\partial y}{\partial Z_{1,1}}&...&\frac{\partial y}{\partial Z_{1,n}}\\...&...&...\\\frac{\partial y}{\partial Z_{m,1}}&...&\frac{\partial y}{\partial Z_{m,n}}\end{bmatrix}
{{< /math_block >}}
鉴于
{{< math_block >}}
\begin{aligned}Z_{ij}&=\sum_kX_{ik}W_{kj}\\v&=f(Z)\end{aligned}
{{< /math_block >}}
那么在计算$\overline{X_{i,k}}$时，需要将所有计算图上以$X_{i,k}$为输入的节点都找出来，即$Z$的第$i$行的每个元素。因此$\overline{X_{i,k}}$的计算公式为：
{{< math_block >}}
\overline{X_{i,k}}=\sum_{j}\frac{\partial Z_{i,j}}{\partial X_{i,k}}\bar{Z}_{i,j}=\sum_{j}W_{k,j}\bar{Z}_{i,j}
{{< /math_block >}}
上述公式记为矩阵形式为：
{{< math_block >}}
\overline X = \overline Z W^T
{{< /math_block >}}

# 参考文档