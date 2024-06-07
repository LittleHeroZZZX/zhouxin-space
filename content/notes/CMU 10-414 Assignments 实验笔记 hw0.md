---
title: CMU 10-414 Assignments 实验笔记 hw0
tags: 
date: 2024-06-06T13:28:00+08:00
lastmod: 2024-06-07T11:09:00+08:00
publish: true
dir: notes
slug: notes on cmu 10-414 assignments hw0
math: "true"
---

# 前言

本文记录了完成《CMU 10-414/714 Deep Learning System》配套 Assignments 的过程和对应笔记。共有 6 个 hw，循序渐进地从头实现了一个深度学习框架，并利用搭建 DL 中厂常见的网络模型，包括 CNN、RNN、Transformer 等。

实验环境为 Ubuntu 24 @ WSL2。

由于官方自动评分系统目前不再接受非选课学生注册，因此本代码仅保证能够通过已有测试样例。

# 资源存档

源码来自官方：[Assignments](https://dlsyscourse.org/assignments/)

所有代码均上传至 [cmu10-414-assignments: cmu10-414-assignments](https://gitee.com/littleherozzzx/cmu10-414-assignments)，如官网撤包，可通过 git 回滚获取原始代码。

# hw0

第一个 homework 共需完成 7 个函数，第一个很简单，用于熟悉评测系统，直接从第二个函数开始。

## parse_mnist

这个函数签名为：`parse_mnist(image_filename, label_filename)`，用于读取 MNIST 手写数据集。[官网](http://yann.lecun.com/exdb/mnist/) 对数据集格式有详细介绍，直接下拉到 FILE FORMATS FOR THE MNIST DATABASE 这部分即可。

整个数据集分为训练集和测试集，包括数字图像和标签。标签文件内前 8Byte 记录了 magic number 和 number of items，之后按照每个样本占 1Byte 的格式组织。图像文件内前 16Byte 记录了非图像数据，之后按照行优先的顺序按照每个像素占 1Byte 的格式以此排布，每个图片共有 28×28 个像素点。

具体实现中，使用 gzip 库按字节读取数据文件，注意整个数据集需要进行标准化，即将每个像素的灰度值除以 255。完整实现为：

```python
def parse_mnist(image_filename, label_filename):
    image_file_handle = gzip.open(image_filename, 'rb')
    label_file_handle = gzip.open(label_filename, 'rb')
    image_file_handle.read(16)
    label_file_handle.read(8)
    image_data = image_file_handle.read()
    label_data = label_file_handle.read()
    image_file_handle.close()
    label_file_handle.close()
    X = np.frombuffer(image_data, dtype=np.uint8).reshape(-1, 28*28).astype(np.float32)
    X = X / 255.0
    y = np.frombuffer(label_data, dtype=np.uint8)
    return X, y
```

## softmax_loss

这个函数签名为：`softmax_loss(Z, y)`，需要注意的是它计算的是 softmax 损失，或者说是交叉熵损失，而不是进行 softmax 归一化。

照着公式写两行代码即可，不用再赘述：

```python
def softmax_loss(Z, y):
    rows = np.arange(Z.shape[0])
    return -np.mean(Z[rows, y] - np.log(np.sum(np.exp(Z), axis=1)))
```

## softmax_regression_epoch

这个函数签名为：`softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100)`，要实现的是 softmax 回归一个 epoch 上的训练过程。

首先计算出总的 batch 数，并进行这么多次的循环。在每个循环内，先从 X 和 y 中取出对应样本，然后根据公式计算即可。这里涉及到将 label 转换为独热编码的一个小技巧：`E_batch = np.eye(theta.shape[1])[y_batch]`，其它则比较简单：

```python
def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    total_batches = (X.shape[0] + batch - 1) // batch
    for i in range(total_batches):
        X_batch = X[i*batch:(i+1)*batch]
        y_batch = y[i*batch:(i+1)*batch]
        E_batch = np.eye(theta.shape[1])[y_batch]
        logits = X_batch @ theta
        Z_batch = np.exp(logits)
        Z_batch /= np.sum(Z_batch, axis=1, keepdims=True)
        gradients = X_batch.T @ (Z_batch - E_batch) / batch
        theta -= lr * gradients
```

## nn_epoch

这个函数签名为：`nn_epoch(X, y, W1, W2, lr = 0.1, batch=100)`，要实现一个双层感知机在一个 epoch 上的训练过程。

跟着公式写代码计算即可，需要注意的两个点：

- ReLU 激活函数可以使用 max 函数进行实现：`Z1_batch = np.maximum(X_batch @ W1, 0)`
- 除以 batch_size 这一步应该提前在计算 G2 的过程，如果放在最后更新 $\theta$ 这一步，存在精度误差，不能通过测试点。

完整代码为：

```python
def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    total_batches = (X.shape[0] + batch - 1) // batch
    for i in range(total_batches):
        X_batch = X[i*batch:(i+1)*batch]
        y_batch = y[i*batch:(i+1)*batch]
        E_batch = np.eye(W2.shape[1])[y_batch]
        Z1_batch = np.maximum(X_batch @ W1, 0)
        G2_batch = np.exp(Z1_batch @ W2)
        G2_batch /= np.sum(G2_batch, axis=1, keepdims=True)
        G2_batch -= E_batch
        G2_batch /= batch
        G1_batch = (Z1_batch > 0) * (G2_batch @ W2.T)
        gradients_W1 = X_batch.T @ G1_batch
        gradients_W2 = Z1_batch.T @ G2_batch
        W1 -= lr * gradients_W1
        W2 -= lr * gradients_W2
```

## softmax_regression_epoch_cpp

这个函数签名为：`void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k, float lr, size_t batch)`，这是一个 softmax 回归在 cpp 上的实现版本。

与 Python 自动处理数组索引越界不同，cpp 版本要分开考虑完整的 batch 和最后一轮不完整的 batch。计算 logits 时，需要使用三轮循环模拟矩阵乘法。cpp 版本的实现可以不写出 $E_y$ 矩阵，梯度计算不用使用矩阵计算，直接使用两层循环，判断 class_idx 是否为正确的 label：`softmax[sample_idx * k + class_idx] -= (y[start_idx + sample_idx] == class_idx);`。

完整的代码为：

```cpp
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    size_t total_batches = (m + batch - 1) / batch;

    for(size_t i = 0; i < total_batches; i++) {
        size_t start_idx = i * batch;
        size_t end_idx = std::min(start_idx + batch, m);
        size_t current_batch_size = end_idx - start_idx;

        // Allocate memory for logits and softmax
        float* logits = new float[current_batch_size * k]();
        float* softmax = new float[current_batch_size * k]();

        // Compute logits
        for(size_t sample_idx = 0; sample_idx < current_batch_size; sample_idx++) {
            for(size_t class_idx = 0; class_idx < k; class_idx++) {
                for(size_t feature_idx = 0; feature_idx < n; feature_idx++) {
                    logits[sample_idx * k + class_idx] += X[(start_idx + sample_idx) * n + feature_idx] * theta[feature_idx * k + class_idx];
                }
            }
        }

        // Compute softmax
        for(size_t sample_idx = 0; sample_idx < current_batch_size; sample_idx++) {
            float max_logit = *std::max_element(logits + sample_idx * k, logits + (sample_idx + 1) * k);
            float sum = 0;
            for(size_t class_idx = 0; class_idx < k; class_idx++) {
                softmax[sample_idx * k + class_idx] = exp(logits[sample_idx * k + class_idx] - max_logit);
                sum += softmax[sample_idx * k + class_idx];
            }
            for(size_t class_idx = 0; class_idx < k; class_idx++) {
                softmax[sample_idx * k + class_idx] /= sum;
            }
        }

        // Compute gradient
        for(size_t sample_idx = 0; sample_idx < current_batch_size; sample_idx++) {
            for(size_t class_idx = 0; class_idx < k; class_idx++) {
                softmax[sample_idx * k + class_idx] -= (y[start_idx + sample_idx] == class_idx);
            }
        }

        // Update theta
        for(size_t feature_idx = 0; feature_idx < n; feature_idx++) {
            for(size_t class_idx = 0; class_idx < k; class_idx++) {
                float gradient = 0;
                for(size_t sample_idx = 0; sample_idx < current_batch_size; sample_idx++) {
                    gradient += X[(start_idx + sample_idx) * n + feature_idx] * softmax[sample_idx * k + class_idx];
                }
                theta[feature_idx * k + class_idx] -= lr * gradient / current_batch_size;
            }
        }

        // Free allocated memory
        delete[] logits;
        delete[] softmax;
    }
}
```

## hw0 小结

hw0 理应是在 Lecture 2 课前完成的，初学者看到一堆公式应该会很懵逼，但整个 hw 比较简单，照着公式一步步走就能完成（除了双层感知机中奇怪的精度错误），主要还是用来熟悉 NumPy 和基本的 DL 模型。

# 参考文档