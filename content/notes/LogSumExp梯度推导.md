---
title: LogSumExp梯度推导
tags:
  - 深度学习系统
date: 2024-07-20T11:08:00+08:00
lastmod: 2024-07-24T14:20:00+08:00
publish: true
dir: notes
slug: gradient of log sum exp
math: "true"
---

# 前言

在 CMU 10-414/714 Deep Learning System 第二个 homework 有一个小任务要对数值稳定形式的 LogSumExp 的梯度进行推导，查阅了不少资料 [^1]，琢磨好半天才搞懂，特此记录。

# 推导过程

## 符号说明

推导过程中使用的符号说明如下：

{{< math_block >}}
\begin{align*}
 z &\in \mathbb{R}^n\\
 z_k &= \max{z}\\
 \hat{z} &= z - \max{z}\\
 f &= \log{\sum_{i=1}^n{\exp{(z_i - \max{z})}}+\max{z}}\\
 &=\log{\sum_{i=1}^n\exp\hat{z}_i}+z_k

\end{align*}
{{< /math_block >}}

## 非最大情况推导

当 $z_j\neq z_k$ 时，$\frac{\partial{f}}{\partial{z_j}}$ 推导如下：

{{< math_block >}}
\begin{align*}
\frac{\partial{f}}{\partial{z_j}} 
&=\frac{\partial{(\log{\sum_{i=1}^n\exp\hat{z}_i)}}}{\partial z_j} + \frac{\partial z_k}{\partial{z_j}} \\
&= \frac{\partial{(\log{\sum_{i=1}^n\exp\hat{z}_i)}}}{\sum_{i=1}^n\exp\hat{z}_i}\cdot \frac{\sum_{i=1}^n\exp\hat{z}_i}{\partial{z_j}}+0 \\
&=\frac{1}{\sum_{i=1}^n\exp\hat{z}_i}\cdot(\sum_{i\neq j} \frac{\partial\exp{\hat z_i}}{\partial z_j}+\frac{\partial \exp{\hat z_j}}{\partial z_j}) \\ 
&=\frac{1}{\sum_{i=1}^n\exp\hat{z}_i}\cdot(0+\exp{\hat{z}_j}) \\ 
&=\frac{\exp{\hat{z}_j}}{\sum_{i=1}^n\exp\hat{z}_i}
\end{align*}
{{< /math_block >}}

## 最大情况推导

当 $z_j= z_k$ 时，$\frac{\partial{f}}{\partial{z_j}}$ 推导如下：

{{< math_block >}}
\begin{align*}
\frac{\partial{f}}{\partial{z_j}} 
&=\frac{\partial{(\log{\sum_{i=1}^n\exp\hat{z}_i)}}}{\partial z_j} + \frac{\partial z_k}{\partial{z_j}} \\
&= \frac{\partial{(\log{\sum_{i=1}^n\exp\hat{z}_i)}}}{\sum_{i=1}^n\exp\hat{z}_i}\cdot \frac{\sum_{i=1}^n\exp\hat{z}_i}{\partial{z_j}}+1 \\
&=\frac{1}{\sum_{i=1}^n\exp\hat{z}_i}\cdot [\sum_{z_i \neq z_k}{\frac{\partial \exp{(z_i-z_k)}}{\partial z_j}}+\sum_{z_i=z_k}{\frac{\partial \exp{(z_i-z_k)}}{\partial z_j}}]+1\\
&\text{注意，上式中有}z_j=z_k\\
&=\frac{1}{\sum_{i=1}^n\exp\hat{z}_i}\cdot[\sum_{z_i \neq z_k}{-\exp(z_i-z_k)}+0]+1 \\
&= 1-\frac{\sum_{z_i \neq z_k}{\exp(z_i-z_k)}}{\sum_{i=1}^n\exp\hat{z}_i} \\
&=\frac{\exp{\hat{z}_j}}{\sum_{i=1}^n\exp\hat{z}_i}
\end{align*}
{{< /math_block >}}

## 一般情况

注意到无论 $z_j$ 是不是最大值，都有：

{{< math_block >}}
\frac{\partial{f}}{\partial{z_j}}=\frac{\exp{\hat{z}_j}}{\sum_{i=1}^n\exp\hat{z}_i}
{{< /math_block >}}

这里我们讨论的是 $f\in \mathbb{R}$ 且 $z\in\mathbb{R}^n$ 的情况，实际情况中，$f$ 和 $z$ 都是高维张量，我们要求 $z$ 关于 $z$ 的梯度，即 $\nabla_z f$。

# 代码实现

首先感谢 [yofufufufu](https://github.com/yofufufufu) 的不吝赐教，代码实现主要参考他的解释 [^2]。我们继续来化简公式：

{{< math_block >}}
\begin{align*}
\frac{\partial{f}}{\partial{z_j}}
&=\frac{\exp{\hat{z}_j}}{\sum_{i=1}^n\exp\hat{z}_i}\\
&=\exp(z_j - \log \sum_{i=1}^n\exp\hat{z}_i)\\
&=\exp(z_j - f)
\end{align*}
{{< /math_block >}}

惊喜地发现，LogSumExp 这个函数的梯度可以用其输入和输出来表示，那在代码实现中，只要获取该节点的输入和输出就可以计算出梯度，即在 cmu10414 课程，该节点实现如下：

```python
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=self.axes)) + max_z.squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(node.inputs[0].shape)))
        z = node.inputs[0]
        shape = [1 if i in self.axes else z.shape[i] for i in range(len(z.shape))]
        gradient = exp(z - node.reshape(shape).broadcast_to(z.shape))
        return out_grad.reshape(shape).broadcast_to(z.shape)*gradient
```

# 参考资料

[^1]: [logsumexp 反向传播推导\_logsumexp (lse)-CSDN博客](https://blog.csdn.net/u010043946/article/details/134408424)
[^2]: [hw2 LogSumExp梯度公式推导 · Issue #4 · kcxain/dlsys · GitHub](https://github.com/kcxain/dlsys/issues/4#issuecomment-2242385479)