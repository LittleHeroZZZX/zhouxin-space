---
title: 《CMU 10-414 deep learning system》学习笔记
tags:
  - CUDA
  - 深度学习系统
date: 2024-05-28T12:24:00+08:00
lastmod: 2024-05-29T12:49:00+08:00
publish: true
dir: notes
slug: notes on cmu 10-414 deep learning system
images:
  - https://pics.zhouxin.space/202406041918872.png?x-oss-process=image/quality,q_90/format,webp
math: "true"
---

# 资源存档

| 课程主页 | [Deep Learning Systems](https://dlsyscourse.org/) |
| ---- | ------------------------------------------------- |
|      |                                                   |

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

$$
X\in R^{m\times n}=\left[ \begin{array}{c}
	x^{(1)T}\\
	\vdots\\
	x^{\left( m \right) T}\\
\end{array} \right] ,  y\in \left\{ 1,...,k \right\} ^m=\left[ \begin{array}{c}
	y^{\left( 1 \right)}\\
	\vdots\\
	y^{\left( m \right)}\\
\end{array} \right] 
$$

数据集的矩阵是一个个样本转置后堆叠 stack 起来的。那么输出可以表示为：

$$
h_{\theta}\left( X \right) =\left[ \begin{array}{c}
	h_{\theta}\left( x^{\left( 1 \right)} \right) ^T\\
	\vdots\\
	h_{\theta}\left( x^{\left( m \right)} \right) ^T\\
\end{array} \right] =\left[ \begin{array}{c}
	x^{\left( 1 \right) T}\theta\\
	\vdots\\
	x^{\left( m \right) T}\theta\\
\end{array} \right] =X\theta 
$$

关于损失函数 $l_{err}$，一种朴素的想法是将模型预测错误的模型数据量作为损失函数，即如果模型预测的正确率最高的那个类别与真实类别不相同，则损失函数为 1，否则为 0：

$$
l_{err}\left( h\left( x \right) , y \right) \,\,=\,\,\left\{ \begin{aligned}
	0 \ &\mathrm{if} \ \mathrm{argmax} _i\,\,h_i\left( x \right) =y\\
	1 \ &\mathrm{otherwise}\\
\end{aligned} \right. 
$$

遗憾的是，这个符合直觉函数是不可微分的，难以对参数进行优化。更合适的做法是使用交叉熵损失函数。

在此之前，我们将先讲输出过一个 softmax 函数，使之的行为更像一个概率——各个类别的概率之和为 1：

$$
z_i=p\left( \mathrm{label}=i \right) =\frac{\exp \left( h_i\left( x \right) \right)}{\sum_{j=1}^k{\exp \left( h_j\left( x \right) \right)}}
$$

那么交叉熵损失函数就可以定义为：

$$
l_{ce}\left( h\left( x \right) ,y \right) =-\log p\left( \mathrm{label}=y \right) =-h_y\left( x \right) +\log \sum_{j=1}^k{\exp \left( h_j\left( x \right) \right)}
$$

注意在计算交叉熵时，通过运算进行了化简，这使得我们可以省去计算 softmax 的过程，直接计算最终的结果。不但如此，交叉熵的计算中，如果 $h_i(x)$ 的值很小，那么取对数会出现很大的值，化简后的计算则避免了这种情况。

所有的深度学习问题，都可以归结为一下这个最优化问题：$$  
\mathop {\mathrm{minimize}} \limits_{\theta}\ \ \frac{1}{m}\sum_{i=1}^m{l(h_{\theta}(x^{(i)}),y^{(i)}))}

$$
我们使用梯度下降法对该问题进行优化。在此之前，首先介绍一下关于梯度。我们的优化目标可以看作一个关于$\theta \in R^{n\times k}$的函数$f$，那么其在$\theta_0$处的梯度可以表示为：
$$

\nabla _{\theta}f\left( \theta _0 \right) \in R^{n\times k}=\left[ \begin{matrix}  
	\frac{\partial f\left( \theta _0 \right)}{\partial \theta _{11}}&		\cdots&		\frac{\partial f\left( \theta _0 \right)}{\partial \theta _{k1}}\\  
	\vdots&		\ddots&		\vdots\\  
	\frac{\partial f\left( \theta _0 \right)}{\partial \theta _{n1}}&		\cdots&		\frac{\partial f\left( \theta _0 \right)}{\partial \theta _{nk}}\\  
\end{matrix} \right] 

$$
其中，第$i$行第$j$个元素表示除$\theta_{ij}$之外的参数都被当作常数，对$\theta_{ij}$求偏导。

梯度下降，就是沿着梯度方向不断进行迭代，以求找到最佳的$\theta$使得目标函数值最小。
$$

\theta :=\theta _0-\alpha \nabla f\left( \theta _0 \right) 

$$
上式中，$\alpha$被称为学习率或者步长。

事实上，在现代深度学习中，并不是使用的传统梯度下降的方案，因为其无法将所有训练集一次性读入并计算梯度。现代使用的是随机梯度下降（Stochastic Gradient Descent，SGD）

首先将m个训练集样本划分一个个小batch，每个batch都有B条数据。那每一batch的数据表示为$X\in R^{B\times n}$，更新参数$\theta$的公式变为：
$$

\theta :=\theta _0-\frac{\alpha}{B}\nabla f\left( \theta _0 \right) 

$$
我们的梯度变成了每个小batch对全体样本梯度的估计。

那如何计算梯度表达式呢？梯度矩阵中每个元素都是一个偏导数，我们就先从计算偏导数开始。假设$h$是个向量，我们来计算偏导数$\frac{\partial l_{ce}\left( h,y \right)}{\partial h_i}$：
$$

\frac{\partial l_{ce}\left( h,y \right)}{\partial h_i}=\frac{\partial}{\partial h_i}\left( -h_y+\log \sum_{j=1}^k{\exp h_j} \right)  
\\  
=-1\left\{ i=y \right\} +\frac{\exp \left( h_j \right)}{\sum_{j=1}^k{\exp h_j}}  
\\  
=-1\left\{ i=y \right\} +\mathrm{softmax} \left( h \right)  
\\  
=z-e_y

$$

如果$h$是个向量，那么梯度$\nabla_h l_{ce}(h,y)$就能够以向量的形式表示为：
$$

\nabla_h l_{ce}(h,y) = z-e_y

$$
这里我们将对$h$进行softmax标准化记为$z$，$e_y$表示对应的单位向量。

事实上，我们要计算的梯度是关于$\theta$的，具体来说，表达式为$\nabla_\theta l_{ce}(\theta^Tx,y)$，其中，$\theta$是个矩阵。或许，可以使用链式法则进行求解，但是太麻烦了，这里还涉及矩阵对向量的求导。我们需要一种更加通用的求导方案。

有两个解决办法：
- 正确且官方的做法：使用矩阵微分学、雅可比矩阵、克罗内克积和向量化等知识进行求解。
- 一个hacky、登不上台面、但大家都在用的方案：将所有的矩阵和向量当作标量，使用链式法则求解，并进行转置操作使得结果的size符合预期，最后检查数值上结果是否正确。

按照第二个方法的逻辑，过程为：
$$

\frac{\partial}{\partial \theta}l_{ce}\left( \theta ^Tx,y \right) =\frac{\partial l_{ce}\left( \theta ^Tx,y \right)}{\partial \theta ^Tx}\cdot \frac{\partial \theta ^Tx}{\partial \theta}  
\\  
=\left[ z-e_y \right] _{k\times 1}\cdot x_{n\times 1}  
\\  
=x\cdot \left[ z-e_y \right] 

$$
其中，$z=\text{softmax}(\theta^Tx)$。注意，倒数第二步求出的结果是两个列向量相乘，不能运算。又已知结果应该是$n\times k$的矩阵，调整向量之间的顺序即可。

照猫画虎，可以写出batch的情况，$X\in R^{B\times n}$：
$$

\frac{\partial}{\partial \theta}l_{ce}\left( \theta ^TX,y \right) =\frac{\partial l_{ce}\left( \theta ^TX,y \right)}{\partial \theta ^TX}\cdot \frac{\partial \theta ^TX}{\partial \theta}  
\\  
=\left[ Z-E_y \right] _{B\times k}\cdot X_{B\times n}  
\\  
=X^T\cdot \left[ Z-E_y \right] 

$$

# Lecture 3: Manual Neural Networks
这节课，我们将人工实现全连接神经网络，之后的课程，将引入自动微分技术。
## 从线性模型转变为非线性模型
![image.png](https://pics.zhouxin.space/202406041247481.png?x-oss-process=image/quality,q_90/format,webp)
如上图所示，线性模型本质上是将样本空间划分为线性的几个部分，这样的模型性能十分有限，因此很多不满足这样分布的实际问题就不能被解决。

一种解决方案是，在将样本输入到线性分类器前，先人工挑选出某些特征，即对$X$应用一个函数$\phi$，其将$X$映射到$\phi(X)$上，映射后的空间可以被线性划分。一方面，它确实是早期实践中行之有效的方案；另一方面，人工提取特征的泛化性能有限，受限于具体问题和研究人员的对问题的洞察程度。

如果我们使用线性网络提取特征，并直接接上一个线性分类头，这两个线性层等效为一个线性层，并不能做到非线性化的要求（基础知识，此处不再解释）。

因此，在使用线性网络提取特征后，需要再接上一个非线性函数$\sigma$，即$\phi = \sigma (W^T X)$。
## 神经网络

## 反向传播（梯度计算）




# 参考文档