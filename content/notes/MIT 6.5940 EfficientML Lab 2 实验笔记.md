---
title: MIT 6.5940 EfficientML Lab 2 实验笔记
tags: 
date: 2025-02-25T15:35:00+08:00
lastmod: 2025-02-25T19:24:00+08:00
publish: true
dir: notes
slug: notes on mit efficientml Lab 2
math: "true"
---

> 本问记录为 EfficientML Lab 2 实验笔记，包含 K-Means 量化、K-Means QAT、线性量化等内容，难度不大，内容丰富。

# Part 1: K-Means Quantization

## Qustion 1

第一个问题是实现 K-means 量化的核心算法，其中 K-means 本身是调库实现的。

第一小问求簇数，n bit 可以表示 `2^n` 个簇。第二小问是根据已有的 Codebook 表示量化后的张量，使用 Tensor 的索引表示即可。需要注意的是，codebook 可能是调用者传入的而非一定是由我们计算得到的，因此在表示的时候要使用 codebook 的成员来获取 `centroids` 和 `labels`，否则在后面会报错。

```python
    if codebook is None:
        ############### YOUR CODE STARTS HERE ###############
        # get number of clusters based on the quantization precision
        # hint: one line of code
        n_clusters = 2 ** bitwidth
        ############### YOUR CODE ENDS HERE #################
        # use k-means to get the quantization centroids
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    ############### YOUR CODE STARTS HERE ###############
    # decode the codebook into k-means quantized tensor for inference
    # hint: one line of code
    quantized_tensor = codebook.centroids[codebook.labels].view(fp32_tensor.shape)
    ############### YOUR CODE ENDS HERE #################
```

## Question 2

略

# Part 2: Trained K-Means Quantization

## Question 3

在低比特量化后模型掉点很厉害，因此要进行 QAT。量化后的权重的梯度推导为：

{{< math_block >}}
\frac{\partial \mathcal{L} }{\partial C_k} = \sum_{j} \frac{\partial \mathcal{L} }{\partial W_{j}} \frac{\partial W_{j} }{\partial C_k} = \sum_{j} \frac{\partial \mathcal{L} }{\partial W_{j}} \mathbf{1}(I_{j}=k)
{{< /math_block >}}

但在本实验中，简单起见，我们使用相同簇的原始权重的均值作为量化后的该簇更新后的值。代码实现就一行：

```python
codebook.centroids[k] = fp32_tensor[codebook.labels == k].mean()
```

最终不同量化位数得到的性能指标如下表所示，2 bits 掉点很夸张，QAT 后也难以恢复到原始性能。

| 量化位数   | 掉点率    | 微调后掉点率 | 微调轮数 |
| ------ | ------ | ------ | ---- |
| 8 bits | 0.17%  | 0.17%  | 0    |
| 4 bits | 13.87% | 0.49%  | 1    |
| 2 bits | 82.95% | 1.75%  | 5    |

# Part 3: Linear Quantization 

## Question 4

本问实现的是线性量化的核心函数，即给定张量、量化位宽、缩放系数、零点，计算量化后的张量。根据线性量化公式：

{{< math_block >}}
q = r/S + Z
{{< /math_block >}}

计算即可：

```python
############### YOUR CODE STARTS HERE ###############
    # Step 1: scale the fp_tensor
    scaled_tensor = fp_tensor / scale
    # Step 2: round the floating value to integer value
    rounded_tensor = scaled_tensor.round()
    ############### YOUR CODE ENDS HERE #################

    rounded_tensor = rounded_tensor.to(dtype)

    ############### YOUR CODE STARTS HERE ###############
    # Step 3: shift the rounded_tensor to make zero_point 0
    shifted_tensor = rounded_tensor + zero_point
    ############### YOUR CODE ENDS HERE #################
```

值得一提的是，在 step 4 中，执行了一步将溢出的结果压缩到范围之内的操作。

## Question 5

计算缩放系数和零点的公式分别为：

{{< math_block >}}
\begin{align*}
S&=(r_{\mathrm{max}} - r_{\mathrm{min}}) / (q_{\mathrm{max}} - q_{\mathrm{min}})\\
Z &= \mathrm{int}(\mathrm{round}(q_{\mathrm{min}} - r_{\mathrm{min}} / S))
\end{align*}
{{< /math_block >}}

代码照抄即可。

权重有一个特殊的性质：分布通常都是关于 0 点对称的，因此权重量化的零点可以直接设置为 0。

此外，经验表明，对卷积核进行量化时，按照输出通道逐通道量化能够取得更好的表现

## Question 6-8

在此之前，实验文档首先推导了考虑线性量化的全连接层和卷积层的表达式，推导过程进行了一系列代入、假设和化简，主要包括：

{{< math_block >}}
\begin{align*}
Z_{\mathrm{weight}}&=0\\
r_{\mathrm{weight}} &= S_{\mathrm{weight}}q_{\mathrm{weight}}\\
Z_{\mathrm{bias}} &= 0\\
S_{\mathrm{bias}} &= S_{\mathrm{input}} \cdot S_{\mathrm{weight}}
\end{align*}
{{< /math_block >}}

最终得到的结论为：

{{< math_block >}}
\begin{align*}
q_{\mathrm{output}} &= (\mathrm{CONV}[q_{\mathrm{input}}, q_{\mathrm{weight}}] + Q_{\mathrm{bias}}) \cdot (S_{\mathrm{input}}S_{\mathrm{weight}} / S_{\mathrm{output}}) + Z_{\mathrm{output}}\\
q_{\mathrm{output}} &= (\mathrm{Linear}[q_{\mathrm{input}}, q_{\mathrm{weight}}] + Q_{\mathrm{bias}})\cdot (S_{\mathrm{input}} \cdot S_{\mathrm{weight}} / S_{\mathrm{output}}) + Z_{\mathrm{output}}\\
\text{其中，}Q_{\mathrm{bias}} &= q_{\mathrm{bias}} - \mathrm{Linear}[Z_{\mathrm{input}}, q_{\mathrm{weight}}]
\end{align*}
{{< /math_block >}}

Q7 和 Q8 的代码实现相同，需要注意的是在对 output 进行缩放时，由于是逐通道量化的，因此权重的缩放系数是个张量，需要处理好形状以便进行广播。

```python
    ############### YOUR CODE STARTS HERE ###############
    # Step 2: scale the output
    #         hint: 1. scales are floating numbers, we need to convert output to float as well
    #               2. the shape of weight scale is [oc, 1, 1, 1] while the shape of output is [batch_size, oc]
    output = output * (input_scale * weight_scale / output_scale).swapaxes(0, 1)

    # Step 3: shift output by output_zero_point
    #         hint: one line of code
    output = output + output_zero_point
    ############### YOUR CODE ENDS HERE #################
```

## Question 9

要干的活文档基本都干好了，只剩下一个对输入进行量化的活需要我们完成。照猫画虎，使用 `get_quantization_scale_and_zero_point` 计算缩放系数和零点，使用 `linear_quantize` 进行线性量化即可。

```python
############### YOUR CODE STARTS HERE ###############
x_scale, x_zero_point = get_quantization_scale_and_zero_point(x, 8)
return linear_quantize(x, 8, x_scale, x_zero_point)
############### YOUR CODE ENDS HERE #################
```

量化后的模型精度为 92.21% 几乎没掉点。

**为什么在线性量化中没有 ReLU 层**：ReLU 层被融合到前一层中网络中，可以减少数据的搬运次数。

## Question 10

回答来自 deepseek：

|**量化方法**|**核心优势**|**核心劣势**|**适用场景**|
|---|---|---|---|
|**K-means 量化**|高精度（非均匀数据）|计算复杂、硬件支持差|数据分布复杂、对精度敏感、专用硬件场景|
|**线性量化**|低延迟、硬件友好、易部署|对非均匀数据精度低|实时推理、通用硬件、动态范围稳定场景|

# 小结

实验文档本身体量很大，知识点也很丰富，但是大多数代码都已经给出了，每个回答只需要写一行或者两行代码，并且周围也给出了充足的提示。这使得实验本身缺乏挑战性，有点鸡肋。