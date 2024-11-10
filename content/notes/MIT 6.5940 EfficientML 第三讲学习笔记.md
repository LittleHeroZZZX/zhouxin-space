---
title: MIT 6.5940 EfficientML 第三讲学习笔记
tags:
  - EfficientML
date: 2024-11-09T14:01:00+08:00
lastmod: 2024-11-10T16:12:00+08:00
publish: true
dir: notes
slug: notes on mit efficientml 3rd lecture
math: "true"
---

> MIT 6.5940 EfficientML 第三讲学习笔记，主要介绍剪枝的定义、效果和粗细程度，并详细介绍了多种剪枝标准。

# Lecture 3: Pruning and sparsity 剪枝和稀疏性

## 剪枝的动机

在上一讲提到，内存操作的代价相当昂贵，因此为了加速模型的运行，一个思路就是减少模型中一切内存的占用，包括减小模型大小、减小激活层大小和数量。

## 剪枝定义

剪枝的数学定义如下所示：  
![剪枝的数学定义](https://pics.zhouxin.space/202411091410114.png?x-oss-process=image/quality,q_90/format,webp)  
具体来说，剪枝指的是移除神经网络中的某一些参数，使得神经网络成为一个相对稀疏网络。

## 剪枝效果

如下图所示，经过剪枝后的模型，其性能损失并不显著；相反，如果在剪枝后对模型进行微调，甚至迭代剪枝和微调的步骤，可以实现仅使用几分之一的参数量达到相同或者更优的准确率。  
![剪枝效果图](https://pics.zhouxin.space/202411091418290.png?x-oss-process=image/quality,q_90/format,webp)

这一节的效果多少有点震撼到我：经过剪枝后，模型的参数量可以减少 90%，并且不掉点，甚至性能更优！

## 剪枝细粒程度

根据剪枝实施的细粒程度，可以对其进行分类。
- 细粒度剪枝  
细粒度剪枝允许对任意单个权重进行剪枝，其优点是剪枝程度更高，确定是参数张量作为稀疏矩阵，在硬件上的加速更加难以实现。

- 粗粒度剪枝  
粗粒度剪枝只能都对参数矩阵中的某一行进行全部剪枝，缺点是不够灵活，优点是经过剪枝后的参数仍旧是一个非稀疏矩阵（剪枝后的矩阵变得更小），易于加速。  
![全连接层剪枝示意图](https://pics.zhouxin.space/202411091436763.png?x-oss-process=image/quality,q_90/format,webp)

前面举的是全连接层的例子，接下来讨论卷积操作的剪枝。卷积层的参数张量形状为 $[c_o, c_i, k_h, k_w]$。由于其具有四个维度，因此其剪枝的细粒程度种类多得多：

![卷积核可视化记号示意图](https://pics.zhouxin.space/202411091440974.png?x-oss-process=image/quality,q_90/format,webp)

![卷积核不同细粒程度示意图](https://pics.zhouxin.space/202411091439970.png?x-oss-process=image/quality,q_90/format,webp)

第一种方式是剪枝最细的，往往可以实现数量级的剪枝，然而其一般需要在特定硬件上才能取得比较好的推理速度，在 GPU 上设计算法比较困难。

第二种方式是基于模式的，采用固定的模式对卷积核进行剪枝。一种典型的模式是 N:M，即在连续 M 个参数中选择 N 的进行剪枝。

最后一种方式是对整个通道进行剪枝，其优点在于经过剪枝后仍旧是通用矩乘，缺点是剪枝率很比较低。

## 剪枝标准

- 基于大小的剪枝  
一种符合直觉、并且效果很好的剪枝方式是根据参数的绝对值来决定其重要性，剪掉那些绝对值很小的参数。如果要进行按行剪枝，则每一行的权重对于这一行向量的 L1 Norm，即这一行向量的绝对值的和。

- 基于缩放参数的剪枝  
例如在卷积的每一层中，有多个通道，可以给每个通道配置一个可学习的参数，作为该通道的放缩参数。通过训练来确定每个通道放缩参数，即该通道的重要性，并进行剪枝。

- 基于二阶的剪枝  
我们可以使用泰勒展开来表示经过剪枝后模型的误差：

{{< math_block >}}
\delta L = L(\mathbf{x}; \mathbf{W}) - L(\mathbf{x}; \mathbf{W}_\rho = \mathbf{W} - \delta\mathbf{W}) = \sum_i g_i \delta w_i + \frac{1}{2} \sum_i h_{ii} \delta w_i^2 + \frac{1}{2} \sum_{i\neq j} h_{ij} \delta w_i \delta w_j + O(\|\delta\mathbf{W}\|^3)
{{< /math_block >}}

其中，$g_i = \frac{\partial L}{\partial w_i}$，$h_{i,j} = \frac{\partial ^2 L}{\partial w_i \partial w_j}$。

基于二阶的剪枝方法假设：  
损失函数是近似二次的，因此高阶误差项可以忽视；  
神经网络在训练过程中已经收敛，因此 L 对 w 的一阶导为 0；  
对不同参数的剪枝操作引发的误差是彼此独立的，因此 $\frac{1}{2} \sum_{i\neq j} h_{ij} \delta w_i \delta w_j$ 为 0。

因此，剪枝后模型的误差为：

{{< math_block >}}
\delta L = L(\mathbf{x}; \mathbf{W}) - L(\mathbf{x}; \mathbf{W}_\rho) \approx \frac{1}{2} \sum_i h_{ii} \delta w_i^2
{{< /math_block >}}

为了最小化剪枝后的误差，应当保留权重更大的参数，因此重要性表示为：

{{< math_block >}}
\text{importance}_{w_i} = \frac{1}{2}h_{ii}w_{i}
{{< /math_block >}}

其中，$h_{ii}$ 是 Hessian 矩阵。

- 对激活层进行剪枝  
对激活层进行剪枝其本质上就是粗粒度的权重剪枝。如下图所示，如果我们需要移除激活层的某个节点，其进行的操作就是在 FC 网络中移除权重举证的某一行，或者在卷积中移除某个通道。  
![权重剪枝与激活层剪枝之间的关系](https://pics.zhouxin.space/202411091608696.png?x-oss-process=image/quality,q_90/format,webp)

- 基于 0 概率的剪枝  
ReLU 层的输出的激活层有概率为零，通过统计 batch 中每个位置的输出为 0 的频率，可以对 0 概率高的位置进行剪枝。

注意，这里是对激活层进行剪枝，而非之前提到的直接对参数进行剪枝。我的理解是，对激活层进行剪枝相当于直接对这激活层对应的两层网络的参数进行剪枝。

- 基于回归的剪枝  
如果直接评估整个模型剪枝前后的误差，这一代价可能很高。基于回归的剪枝对网络逐层评估和剪枝。

下图展示了对全连接层输入通道的剪枝算法示意图。首先将矩乘结果视为多个通道外积结果的和，为每个通道设置一个缩放系数 $\beta_{c}$，缩放系数越接近 0 说明该通道越不重要。

![基于回归的剪枝算法](https://pics.zhouxin.space/202411101545235.png?x-oss-process=image/quality,q_90/format,webp)

优化算法为：首先固定权重，对 $\beta$ 进行优化，对系数小的参数进行剪枝；然后固定 $\beta$，对 $W$ 进行优化。还可以对上述过程进行重复和迭代。

