---
title: MIT 6.5940 EfficientML 第二讲学习笔记
tags: 
date: 2024-11-05T22:33:00+08:00
lastmod: 2024-11-08T19:03:00+08:00
publish: true
dir: notes
slug: notes on mit efficientml 2nd lecture
---

如无另外说明，本文图片截取自 [EfficientML](efficient.ml) 课程幻灯片。

# Lecture 2: Basics of neural networks 神经网络基础

## 神经网络

- 基本术语  
如下图所示，我们使用术语 Synapses（突触？）、权重、参数来指代网络中的参数，使用术语神经元、特征、激活层来指代网络中每一层的计算结果。  
![三层神经网络示意图](https://pics.zhouxin.space/202411060903228.png?x-oss-process=image/quality,q_90/format,webp)

模型的宽度指的是隐藏层的维度，对于相同的参数量，宽而浅的模型相比窄而深的模型计算效率更高，因为其核函数调用次数更少，并且能够充分进行并行计算。然而后者在准确率上往往表现得更好，这需要进行折中。

- 全连接层  
全连接层是对输入进行加权求和并加上偏执项，如下所示：  
![全连接层示意图](https://pics.zhouxin.space/202411060913049.png?x-oss-process=image/quality,q_90/format,webp)

- 2D 卷积  
![2D卷积示意图](https://pics.zhouxin.space/202411060918066.png?x-oss-process=image/quality,q_90/format,webp)

基本术语这一节略，大多是 DL 的入门知识。

## 神经网络效率的评价指标

- latency 延迟  
神经网络完成某一特定任务需要的时间。延迟主要取决于计算时间和内存时间，即：

{{< math_block >}}
\begin{align*}
\text{latency} &\approx \max(T_{\text{computation}}, T_{\text{memory}})\\
T_{\text{compucation}} &\approx \frac{\text{number of ops in model}}{\text{number of ops that processor can process per second}}\\
T_{\text{mem}} &\approx T_{\text{data movement of activations}} + T_{\text{data movement of weights}}\\
T_{\text{data movement of activations}} &\approx \frac{\text{input and output actication size}}{\text{mem bandwith of processor}}\\
T_{\text{data movement of weights}} &\approx \frac{\text{model size}}{\text{mem bandwith of processor}}
\end{align*}
{{< /math_block >}}

内存操作消耗的资源比计算多的多的多，如下图所示 32 位的访存操作消耗的能量是常见其它计算操作的数百倍。  
![不同32比特操作消耗能量对比图](https://pics.zhouxin.space/202411081715363.png?x-oss-process=image/quality,q_90/format,webp)

- throuthput 吞吐量  
单位时间内神经网络能够处理的数据量。

由于 batch 的存在，延迟很高的模型可能能够同时处理多个 batch，因此高延迟≠低吞吐量。在移动设备上更关心延迟，而在高性能设备上更关心吞吐量。

- number of parameters 参数量  
不同层的参数量计算公式如下图所示：  
![不同层参数量计算公式](https://pics.zhouxin.space/202411081832724.png?x-oss-process=image/quality,q_90/format,webp)
- model size 模型大小  
模型大小取决于参数量与参数字长，即：

{{< math_block >}}
\text{model size} = \text{number of parameters} \times \text{bit width}
{{< /math_block >}}

- total / peak number of activations 激活层总量和峰值大小  
在模型推理中，通常瓶颈在于激活层大小而非参数量，如下图所示，相同性能的不同模型，模型参数可以进行大幅度优化，但参数量几乎差不多。而激活层的峰值大小则决定了这个模型在推理过程中消耗的最大内存数。  
![推理中参数大小与激活层大小对比图](https://pics.zhouxin.space/202411081838915.png?x-oss-process=image/quality,q_90/format,webp)  
在模型训练中，激活层更是瓶颈，其大小是参数量的数倍。  
![训练中参数大小与激活层大小对比图](https://pics.zhouxin.space/202411081841611.png?x-oss-process=image/quality,q_90/format,webp)  
在 CNN 的训练中，由于前期图像分辨率高，瓶颈在于激活层；后期特征通道数较高，瓶颈在于权重。  
![image.png](https://pics.zhouxin.space/202411081844392.png?x-oss-process=image/quality,q_90/format,webp)

- MAC  
MAC 操作指的是 multiply-accumulate，即一次 $a\leftarrow a+b\cdot c$ 操作，在 GPU 上其通常可以被翻译为一条指令。

在如下所示的举证向量乘法即 MV 中，MAC 数为 $m\cdot n$，即每个结果都需要 $n$ 次 MAC 操作。 而在矩乘即 GEMM 中，MAC 数为 $m\cdot n\cdot k$，即每个结果需要 $k$ 次 MAC 操作。  
![矩阵-向量乘法和通用矩乘示意图](https://pics.zhouxin.space/202411081854699.png?x-oss-process=image/quality,q_90/format,webp)  
不同层的 MAC 数如下所示：  
![不同层的MAC数](https://pics.zhouxin.space/202411081859325.png?x-oss-process=image/quality,q_90/format,webp)

- FLOP 浮点操作数  
一个 MAC 相当于两个 FLOP

- FLOPs 每秒浮点计算次数

- OP 操作数  
模型并不一定总是用浮点数进行表示和计算，对于非浮点计算，我们称之为 OP。

- OPs 每秒操作计算次数

