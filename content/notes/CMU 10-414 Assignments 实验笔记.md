---
title: CMU 10-414 Assignments 实验笔记
tags:
  - CUDA
  - 深度学习系统
date: 2024-06-06T13:28:00+08:00
lastmod: 2024-07-24T18:25:00+08:00
publish: true
dir: notes
slug: notes on cmu 10-414 assignments
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

# hw1

第一个 homework 共有六个小问：正向计算、反向梯度、拓扑排序反向模式自动微分、softmax 损失、双层感知机的 SGD 算法。

## Implementing forward & backward computation

前两个小问就放在一起讨论了。第一问是通过 NumPy 的 API 实现一些常用的算子，第二问则是通过第一问的算子实现常用算子的梯度实现。

需要注意的是，notebook 中强调了第一问操作的对象是 `NDArray`，第二问是 `Tensor`。前者模拟的事这些算子的低层实现，后者则是通过调用这个算子来实现梯度计算，或者说是将梯度计算封装为另一个算子，这样就可以求梯度看作一个普通运算，进而自动求出梯度的梯度。详细解释请看 Lecture 4。

- PowerScaler  
这个算子作用是对张量逐元素求幂。幂指数作为不可学习的参数，在算子实例化时就固定了，因此不用考虑对幂指数的偏导数。这个很简单，应用幂函数的求导公式即可：

```python
class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return self.scalar * (power_scalar(a, self.scalar-1)) * out_grad
```

- EWiseDiv  
这个算子的作用是对张量逐元素求商。梯度计算很简单，即 $a/b$ 分别对 $a$ 和 $b$ 求偏导：

```python
class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.true_divide(a, b)

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad/b , -a/b/b*out_grad
```

- DivScalar  
这个算子的作用是将整个张量同除 scalar，和 `PowerScalar` 一样，scalar 是不要考虑梯度的：

```python
class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.true_divide(a, self.scalar)

    def gradient(self, out_grad, node):
        return out_grad/self.scalar
```

- MatMul  
这个算子的作用是矩阵乘法。这是这门课程到现在第一个具有挑战性的任务。在计算梯度时，根据课程给出的方法，可以得到如下两个表达式：

```python
adjoint1 = out_grad @ transpose(b)
adjoint2 = transpose(a) @ out_grad
```

但但但是，以上只是理论推导。在实际应用中，存在两个问题：1) 矩阵乘法可能是高维矩阵而非二维矩阵相乘，例如 shape 为 (2, 2, 3, 4) 和 (2, 2, 4, 5) 的两个张量相乘；2) 张量乘法过程可能存在广播的情况，这种情况下的梯度怎么处理。

第一个问题，NumPy 基本都为我们处理好了，只要两个张量的**倒数两个维度**符合二维矩阵乘法且**其余维度**（也称为批量维度）相等，或者某个批量维度为 1（会进行广播），它们就可以进行张量乘法运算。

天下没有免费的午餐，自动广播带来便利的同时，也带来了第二个问题。求出的 adjoint 或者说偏导，应该和输入参数的维度一致，但根据公式计算得到的梯度的维度和广播后的维度一样，因此要进行 reduce 操作。

以下是我不严谨且非形式化的 reduce 操作推导：假设矩阵 $A_{m\times n}$ 经过广播后是 $A_{p\times n\times n}^\prime$，实际上参与计算的就是这个 $A^\prime$。首先直接假设在计算图上用 $A^\prime$ 替代 $A$，当 $A^\prime @B$（该节点记为 $f(x_1,...)$）的某个输入节点 $x_1$ 需要计算梯度时，就会需要计算张量 $\partial f/ \partial x_1$ 和张量 $A^\prime$ 求得的偏导之间的乘积。接下来我们把 $A$ 还原，相对应的，$f(x_1, ...)$ 这个节点计算梯度就要将 $p$ 维度上的偏导数全部加起来，这体现在 $A_{p\times n\times n}^\prime$ 也是将其 $p$ 维度上的元素全部加起来，得到 $A^\prime_{m\times n}$。

上面这段描述不太清晰，总而言之就是要将广播出来的维度全部 sum 掉。

NumPy 中广播新增的维度只会放在最前面，因此只需要计算出要 sum 掉维度的个数，然后取前 $n$ 个维度即可，具体见代码：

```python
class MatMul(TensorOp):
    def compute(self, a, b):
        return a@b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        adjoint1 = out_grad @ transpose(b)
        adjoint2 = transpose(a) @ out_grad
        adjoint1 = summation(adjoint1, axes=tuple(range(len(adjoint1.shape) - len(a.shape))))
        adjoint2 = summation(adjoint2, axes=tuple(range(len(adjoint2.shape) - len(b.shape))))
        return adjoint1, adjoint2
```

- Summation  
这个算子的作用是对张量的指定维度求和。设带求和的张量 $X$ 的维度为 $s_1\times s_2\times ... \times s_n$，那么求和之后的维度就是移除掉 $axes$ 中指示的维度，形式化表达为：

{{< math_block >}}
\text{SUM}(X_{s_1\times s_2\times ... \times s_n}, axes) = [\sum_{s_i \in axes} X]_{\{s_j | j\notin axes \}}
{{< /math_block >}}

假设一个输入为的 shape 为 $3\times 2\times 4 \times 5$，在第 0 和 2 的维度上做 summation，输出的 shape 为 $2\times 5$。反向传播的过程就是先把 `out_grad` 扩展到 $1\times 2 \times 1\times 5$，然后广播到输入的 shape。

埋个坑，这部分还没有理解，不知道怎么形式化表达求和运算与并对其求导，误打误撞以下代码通过了测试：

```python
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        shape = list(a.shape)
        axes = self.axes
        if axes is None:
            axes = list(range(len(shape)))
        for _ in axes:
            shape[_] = 1
        return broadcast_to(reshape(out_grad, shape), a.shape)
```

- BroadcastTo  
这个算子的作用是将张量广播到指定的 shape。所谓广播，就是将数据在不存在或者大小为 1 的维度上复制多份，使之与目标 shape 相匹配。

关于广播算子正向和梯度运算的分析，可查看 MatMul 算子，其对广播过程有详细讨论。本算子实现代码为：

```python
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        ret = summation(out_grad, tuple(range(len(out_grad.shape) - len(input_shape))))
        for i, dim in enumerate(input_shape):
            if dim == 1:
              ret = summation(ret, axes=(i,))
        return reshape(ret, input_shape)
```

- Reshape  
这个算子的作用是将张量重整至指定 shape。反向运算则是将张量重整至输入张量的 shape。其代码实现相当简单：

```python
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)
```

- Negate  
这个算子作用是将整个张量取相反数，反向运算则是再取一次相反数。其代码实现为：

```python
class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return negate(out_grad)
```

- Transpose  
这个算子的作用是交换指定的两个轴，如果没指定则默认为最后两个轴。注意，这个算子的行为与 `np.transpose` 不一致，需要调用 API 是 `np.swapaxes`。反向运算则是再次交换这两个轴：

```python
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        else:
            return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)
```

## Topological sort

这一小问要求实现拓扑排序，涉及的知识点都是数据结构的内容，包括图的拓扑排序、后序遍历和 dfs 算法。

在问题说明中明确要求使用树的后序遍历对算法图求解其拓扑序列，简单来说就是如果本节点存在未访问的子节点（inputs），则先访问子节点，否则访问本节点。所谓访问本节点，就是将其标记为已访问，并将其放入拓扑序列。

结合 dfs 算法，求拓扑序列的代码为：

```python
def find_topo_sort(node_list: List[Value]) -> List[Value]:
    visited = dict()
    topo_order = []
    for node in node_list:
        if not visited.get(node, False):
            topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited: dict, topo_order):
    sons = node.inputs
    for son in sons:
        if not visited.get(son, False):
            topo_sort_dfs(son, visited, topo_order)
    visited[node] = True
    topo_order.append(node)
```

## Implementing reverse mode differentiation

终于开始组装我们的自动微分算法了！核心就是理论课中介绍的反向模式 AD 的算法为代码：  
![image.png](https://pics.zhouxin.space/202406152001945.webp?x-oss-process=image/quality,q_90/format,webp)  
其中有几个注意点：

- `autograd.py` 文件最后一部分提供了一个助手函数 `sum_node_list(node_list)`，用于在不创造冗余节点的情况下，对一系列 node 求和，对应伪代码中对 $\overline{v_i}$ 求和的部分；
- 只有存在输入的节点才要计算梯度，初始 input 节点是没法计算梯度的，要进行判断；
- `node.op.gradient` 返回值类型未 `Tuple | Tensor`，要分类处理。

在写代码之前，最好复习一遍理论；在 debug 的过程中，可以自己画一下计算图，会有奇效。反向模式 AD 具体实现为：

```python
def compute_gradient_of_variables(output_tensor, out_grad) -> None:
    for node in reverse_topo_order:
        node.grad = sum_node_list(node_to_output_grads_list[node])
        if len(node.inputs) > 0:
            gradient = node.op.gradient(node.grad, node)
            if isinstance(gradient, tuple):
                for i, son_node in enumerate(node.inputs):
                    node_to_output_grads_list.setdefault(son_node, [])
                    node_to_output_grads_list[son_node].append(gradient[i])
            else:
                node_to_output_grads_list.setdefault(node.inputs[0], [])
                node_to_output_grads_list[node.inputs[0]].append(gradient)
```

## Softmax loss

本问题先要完成对数函数和指数函数的前向和反向计算，然后再完成 softmax 损失，也就是交叉熵损失函数。

根据说明，这里传入的 y 已经转为了独热编码。具体实现根据说明中的公式一点点写即可，没有要特别说明的：

```python
def softmax_loss(Z, y_one_hot):
    batch_size = Z.shape[0]
    lhs = ndl.log(ndl.exp(Z).sum(axes=(1, )))
    rhs = (Z * y_one_hot).sum(axes=(1, ))
    loss = (lhs - rhs).sum()
    return loss / batch_size
```

## SGD for a two-layer neural network

最后一问，利用前面的组件，实现一个双层感知机及其随机梯度下降算法。注意事项：

- 这里传入的 y 的值是其 label，需要转为独热编码；
- 一定要仔细看题！在计算两个权重的更新值时，应该使用 NumPy 计算，再转为 Tensor。如果直接使用 Tensor 算子计算，每次更新都会在计算图上新增好几个节点，并指数级增长，这会导致后面一些要 600 多 batch 的测试点要跑十几分钟，实际上只要几秒钟就能跑完。如果你遇到了同样的问题，请再读一遍题目要求。  
代码为：

```python
def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    batch_cnt = (X.shape[0] + batch - 1) // batch
    num_classes = W2.shape[1]
    one_hot_y = np.eye(num_classes)[y]
    for batch_idx in range(batch_cnt):
        start_idx = batch_idx * batch
        end_idx = min(X.shape[0], (batch_idx+1)*batch)
        X_batch = X[start_idx:end_idx, :]
        y_batch = one_hot_y[start_idx:end_idx]
        X_tensor = ndl.Tensor(X_batch)
        y_tensor = ndl.Tensor(y_batch) 
        first_logits = X_tensor @ W1 # type: ndl.Tensor
        first_output = ndl.relu(first_logits) # type: ndl.Tensor
        second_logits = first_output @ W2 # type: ndl.Tensor
        loss_err = softmax_loss(second_logits, y_tensor) # type: ndl.Tensor
        loss_err.backward()
        
        new_W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        new_W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
        W1, W2 = new_W1, new_W2

    return W1, W2

```

## hw 1 小结

明显感觉到，这个 hw 的强度上来了。由于不太熟悉 NumPy 的运算，中间查了不少资料和别人的实现。感谢 [@# xx要努力](https://www.zhihu.com/people/xiao-xiong-34-11) 的文章 [^1]，不少都是参考他的实现。

最后双层感知机的调试，由于使用了 Tensor 算子来实现，跑了十几分钟，最后才发现题干已经要求使用 NumPy 运算。长了个很大的教训，下次一定好好读题。

# hw2

## Q1: Weight Initialization 

Q1 实现的是几种不同的生成参数初始值的方法，结合 `init_basic.py` 中的辅助函数，照抄 notebook 中给的公式实现，比较简单。注意把 `kwargs` 传递给辅助函数，里面有 `dtype`、`device` 等信息。

```python
def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    if nonlinearity == "relu":
        gain = math.sqrt(2)
    ### BEGIN YOUR SOLUTION
    bound = gain * math.sqrt(3 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    if nonlinearity == "relu":
        gain = math.sqrt(2)
    ### BEGIN YOUR SOLUTION
    std = gain / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION
```

## Q2: nn_basic

在 Q2，我们将实现几个最基本的 Module 组件。在 Debug 过程中，我遇到了两个很奇怪问题：

- 所有输入和参数都是 `float32` 类型，但有一个输出是 `float64` 类型，导致过不了测试点
- 反向传播中，有一个 node 接收到的 `out_grad` 的 shape 比该节点的输入的 shape 大，但理论上来说二者应该是一致的  
经过漫长的调试追踪，发现第一个问题是因为在实现 `DivScalar` 即除法时，如果输入是一个实数而非一个矩阵，`numpy` 进行除法运算的结果默认为 `float64`，解决方案是显式调用 `np.true_divide` 进行除法运算，并使用关键字 `dtype='float32'` 指定返回值类型。

第二个问题是因为 `numpy` 中许多运算都会进行自动广播，但是该广播操作对我们的 `needle` 库是不可见的，也无法添加到计算图中，因此导致了反向传播过程的 shape 不匹配。解决方案是修改**修改 Q1 中基础算子的实现**，在计算前检查 shape 是否匹配。修改后的 `ops_mathematic.py` 文件内容为：

```python
"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        assert a.shape == b.shape , "The shape of lhs {} and rhs {} should be the same".format(a.shape, b.shape)
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        assert a.shape == b.shape, "The shape of two tensors should be the same"
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return self.scalar * (power_scalar(a, self.scalar-1)) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        assert a.shape == b.shape, "The shape of two tensors should be the same"
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        assert a.shape == b.shape, "The shape of two tensors should be the same"
        return array_api.true_divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad/b , -a/b/b*out_grad
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.true_divide(a, self.scalar, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        else:
            return array_api.swapaxes(a, *self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        expect_size = 1
        for i in self.shape:
            expect_size *= i
        real_size = 1
        for i in a.shape:
            real_size *= i
        assert expect_size == real_size , "The reshape size is not compatible"
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        assert len(self.shape) >= len(a.shape), \
            "The target shape's dimension count {} should be greater than \
                or equal to the input shape's dimension count {}".format(len(self.shape), len(a.shape))
        for i in range(len(a.shape)):
            assert a.shape[-1 - i] == self.shape[-1 - i] or a.shape[-1 - i] == 1, \
                "The input shape {} is not compatible with the target shape {}".format(a.shape, self.shape)
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        ret = summation(out_grad, tuple(range(len(out_grad.shape) - len(input_shape))))
        for i in range(len(input_shape)):
            if input_shape[-1 - i] == 1 and self.shape[-1 - i] != 1:
                ret = summation(ret, (len(input_shape) - 1 - i,))
        return reshape(ret, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        shape = list(a.shape)
        axes = self.axes
        if axes is None:
            axes = list(range(len(shape)))
        for _ in axes:
            shape[_] = 1
        return broadcast_to(reshape(out_grad, shape), a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a@b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        adjoint1 = out_grad @ transpose(b)
        adjoint2 = transpose(a) @ out_grad
        adjoint1 = summation(adjoint1, axes=tuple(range(len(adjoint1.shape) - len(a.shape))))
        adjoint2 = summation(adjoint2, axes=tuple(range(len(adjoint2.shape) - len(b.shape))))
        return adjoint1, adjoint2
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        relu_mask = Tensor(node.inputs[0].cached_data > 0)
        return out_grad * relu_mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
```

万事俱备，接下来可以开始完成 Q2 了。

- Linear  
首先要实现一个线性层，其公式为：

{{< math_block >}}
Y = XW + B
{{< /math_block >}}

注意 `weight` 和 `bias` 都是 `Parameter` 类型，如果定义为 `Tensor` 类型，会导致后面实现优化器过不了测试点。该模块代码为：

```python
class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(self.weight, device=device, dtype=dtype)
        self.bias = None
        if bias:
            self.bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = self.bias.transpose()
            self.bias = Parameter(self.bias, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias.shape != (1, self.out_features):
            self.bias = self.bias.reshape((1, self.out_features))
        y = ops.matmul(X, self.weight)
        if self.bias:
            y += self.bias.broadcast_to(y.shape)
        return y
        ### END YOUR SOLUTION
```

- ReLU  
这个模块很简单，调用 `ops.relu` 即可。

- Sequential  
这个模块的作用是将多个模块封装进一个模块，由其负责将输入在内部按需计算，并给出最终输出。其实现为：

```python
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        for module in self.modules:
            y = module(y)
        return y
        
        ### END YOUR SOLUTION
```

- LogSumExp  
这里要实现的是数值稳定版本的 LogSumExp 算子。文档中直接给出了公式，这里我们给出推导过程：

{{< math_block >}}
\begin{align*}
\log \sum_i \exp(z_i)
&= \log \sum_i \exp(z_i - \max z + \max z)\\
&=\log \sum_i[\exp(z_i - \max z) \cdot \exp(\max z)] \\
&= \log [\sum_i \exp(z_i -\max z)\cdot\exp(\max z)] \\
&=\log \sum_i \exp(z_i -\max z) + \max z
\end{align*}
{{< /math_block >}}

通过恒等变换，避免了 $\exp$ 指数运算可能导致的数值上溢的问题。

显然，数值稳定版本的梯度和原始公式的梯度一致，直接求导或者根据文章 [LogSumExp梯度推导]({{< relref "LogSumExp%E6%A2%AF%E5%BA%A6%E6%8E%A8%E5%AF%BC.md" >}}) 得到其梯度计算公式为：

{{< math_block >}}
\begin{align*}
\frac{\partial{f}}{\partial{z_j}}
&=\frac{\exp{\hat{z}_j}}{\sum_{i=1}^n\exp\hat{z}_i}\\
&=\exp(z_j - \log \sum_{i=1}^n\exp\hat{z}_i)\\
&=\exp(z_j - f)
\end{align*}
{{< /math_block >}}

惊喜地发现，LogSumExp 这个函数的梯度可以用其输入和输出来表示，那在代码实现中，只要获取该节点的输入和输出就可以计算出梯度，即：

```python
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        self.max_z = max_z
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

- SoftmaxLoss  
这里实现其是计算 Softmax 损失的模块，在实现过程中可以调用前面实现的数值稳定版本的 LogSumExp，其公式为：

{{< math_block >}}
\begin{align*}
\ell_\text{softmax}(z,y) = \log \sum_{i=1}^k \exp z_i - z_y
\end{align*}
{{< /math_block >}}

代码骨架中已经提供了一个将标签转换为度和编码的辅助函数，同时记得求的损失应该是在 batch 上的均值，记得做平均。

```python
class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, label_size = logits.shape
        one_hot_y = init.one_hot(label_size, y)
        true_logits = ops.summation(logits * one_hot_y, axes=(1,))
        return (ops.logsumexp(logits, axes=(1, )) - true_logits).sum()/batch_size
        ### END YOUR SOLUTION
```

- LayerNorm1d  
这是第一个比较有挑战性的模块，其中涉及大量的 reshape 和广播操作，必须对每个变量的形状都了如指掌。注意，可以默认输入的 shape 为 `(batch_size, feature_size)`。计算公式为：

{{< math_block >}}
\begin{align*}
y = w \circ \frac{x_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})} + b
\end{align*}
{{< /math_block >}}

根据公式照抄即可，但是要注意中间变量的 shape：

```python
class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype), device=device, dtype=dtype)
        ### BEGIN YOUR SOLUTION
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feature_size = x.shape
        mean = (x.sum(axes=(1, )) / feature_size).reshape((batch_size, 1)).broadcast_to(x.shape)
        var = (((x - mean) ** 2).sum(axes=(1, )) / feature_size).reshape((batch_size, 1)).broadcast_to(x.shape)
        std_x = (x - mean) / ops.power_scalar(var + self.eps, 0.5)
        weight = self.weight.broadcast_to(x.shape)
        bias = self.bias.broadcast_to(x.shape)
        return std_x * weight + bias
        ### END YOUR SOLUTION
```

- Flatten  
本模块的作用是保留第一个维度为 batchsize，展平剩下维度。使用 `ops.resahpe` 实现即可：

```python
class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        assert len(X.shape) >= 2
        elem_cnt = 1
        for i in range(1, len(X.shape)):
            elem_cnt *= X.shape[i]
        return X.reshape((X.shape[0], elem_cnt))
        ### END YOUR SOLUTION
```

- BatchNorm1d  
LayerNorm 是在每一个 batch 内部进行标准化操作，而 BatchNorm 是在每一个 feature 内部进行标准化操作。这就导致了每个样本都会对其他样本的推理结果产生影响，因此在推理时应动态计算均值和方差，以供推理时使用。`nn.Module` 中有一个 `training` 字段用于标识是否在训练。

与 LayerNorm 类似，在实现过程中运用了大量 reshape 和广播操作，要留意中间变量的形状。

```python
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype), device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.weight.shape != (1, self.dim):
            self.weight = self.weight.reshape((1, self.dim))
        if self.bias.shape != (1, self.dim):
            self.bias = self.bias.reshape((1, self.dim))
        if self.training:
            batch_size, feature_size = x.shape
            mean = (x.sum(axes=(0, )) / batch_size).reshape((1, feature_size))
            var = (((x - mean.broadcast_to(x.shape)) ** 2).sum(axes=(0, )) / batch_size).reshape((1, feature_size))
            self.running_mean = self.running_mean *(1 - self.momentum) + mean.reshape(self.running_mean.shape) * ( self.momentum)
            self.running_var = self.running_var *(1 - self.momentum) + var.reshape(self.running_var.shape) * (self.momentum)
            mean = mean.broadcast_to(x.shape)
            var = var.broadcast_to(x.shape)
            std_x = (x - mean) / ops.power_scalar(var + self.eps, 0.5)
            weight = self.weight.broadcast_to(x.shape)
            bias = self.bias.broadcast_to(x.shape)
            return std_x * weight + bias
        else:
            std_x = (x - self.running_mean.broadcast_to(x.shape)) / ops.power_scalar(self.running_var.broadcast_to(x.shape) + self.eps, 0.5)
            return std_x * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
```

- Dropout  
Dropout 说白了就是以概率 p 随机丢弃一部分输入，并把剩下的输入进行缩放，以确保下一层的输入期望不变。代码骨架提供了 `init.randb` 用于生成服从二项分布的布尔序列。代码实现为：

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        mask = init.randb(*x.shape, p=1 - self.p)
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION
```

- Residual  
残差模块就是将其它模块的输出和输入的和作为新的输出，实现比较简单：

```python
class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
```

## Q3: Optimizer Implementation

在本问题中，我们将实现优化器模块。优化器模块的作用是根据 `loss.backward()` 计算出的梯度，更新模型的参数。

需要注意的是，本模块默认启用 l2 正则化或者说 weight decay，因此梯度等于 `param.grad + weight_decay * param`。

- SGD  
首先要实现的优化器是随机梯度下降，注意在更新参数时要先使用 `data` 方法创建该参数的副本，以避免计算图越来越大。这里还使用了移动平均来计算梯度，初始值默认为 0。代码实现如下：

```python
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is not None:
                if param not in self.u:
                    self.u[param] = ndl.zeros_like(param.grad, requires_grad=False)
                self.u[param] = self.momentum * self.u[param].data + (1 - self.momentum) * (param.grad.data + self.weight_decay * param.data)
                param.data = param.data - self.lr * self.u[param]
        ### END YOUR SOLUTION
```

- Adam  
没什么好说的，照抄公式就行：

```python
class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                if param not in self.m.keys():
                    self.m[param] = ndl.zeros_like(param.grad, requires_grad=False)
                if param not in self.v.keys():
                    self.v[param] = ndl.zeros_like(param.grad, requires_grad=False)
                grad = param.grad.data + self.weight_decay * param.data
                self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad.data
                self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * grad.data * grad.data
                u_hat = self.m[param].data / (1 - self.beta1 ** self.t)
                v_hat = self.v[param].data / (1 - self.beta2 ** self.t)
                param.data = param.data - self.lr * u_hat.data / (ndl.ops.power_scalar(v_hat.data, 0.5) + self.eps).data
                
        

        ### END YOUR SOLUTION
```

## Q4: DataLoader Implementation

在本问题中，我们将实现一些数据处理、Dataset 和 DataLoader 类。Dataset 类用于提供标准接口来访问数据集，DataLoader 类是从数据集读取一个 batch 的迭代器。

- RandomFlipHorizontal  
这个方法是按照概率 p 反转一张图片。注意输入数据的格式是 `H*W*C`，因此只要使用 `np.flip` 对 W 轴进行翻转即可。

```python
class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            img = np.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION
```

- RandomCrop  
这个方法是对原图进行随机裁剪。其实现裁剪的流程是：先在上下左右填充 `padding` 个空白像素，然后根据上下偏移量 `shift_y` 和左右偏移量 `shift_y`，在填充图中裁切出与原图大小相同的图片。

```python
class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        img_size = img.shape
        img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        img = img[self.padding + shift_x:self.padding + shift_x + img_size[0], self.padding + shift_y:self.padding + shift_y + img_size[1], :]
        return img
        ### END YOUR SOLUTION
```

- MNISTDataset  
这里要实现针对 MNIST 数据集的 Dataset 子类，作为其子类，要实现三个方法：`__init__` 方法初始化图片、标签和数据处理函数、`__len__` 返回数据集样本数、`__getitem__` 方法获取指定下标的数据集。

要注意的是：1) 使用之前实现的 `parse_mnist` 方法来解析 MNIST 数据集；2) `Dataset` 父类提供了 `apply_transforms` 方法对图片进行处理；3) `__getitem__` 方法最好支持以列表指定的多下标以批量读取数据集;4) 图片处理函数接受的数据格式是 `H*W*C`，但 `__getitem__` 返回值的格式应当为 `batch_size*n`。

代码实现为：

```python
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.X, self.y = parse_mnist(image_filename, label_filename)
        
        ### END YOUR SOLUTION
    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x = self.apply_transforms(self.X[index].reshape(28, 28, -1))
        return x.reshape(-1, 28*28), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
```

- Dataloader  
Dataloader 类是一个迭代器，也挺简单的，见码知义：

```python
class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.index = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.index >= len(self.ordering):
            raise StopIteration
        else:
            batch = [Tensor.make_const(x) for x in self.dataset[self.ordering[self.index]]]
            self.index += 1
            return batch
        ### END YOUR SOLUTION
```

## Q5: MLPResNet Implementation

到此为止，我们的 needle 库的各基本组件都实现好了，在本问题中，我们将使用他们拼出 MLP ResNet，并在 MNIST 数据集上进行训练。

- Residual Block  
首先是实现一个残差块，按照下图将这一块块积木拼出来就行：  
![image.png](https://pics.zhouxin.space/202407241649801.png?x-oss-process=image/quality,q_90/format,webp)

```python
def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION
```

- MLP ResNet  
同样是拼积木，注意这里面有 `num_blocks` 个 Residual Block。  
![image.png](https://pics.zhouxin.space/202407241652831.png?x-oss-process=image/quality,q_90/format,webp)

```python
def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION
```

- Epoch  
`Epoch` 方法用来执行一个 epoch 的训练或者推理，并返回平均错误率或者平均损失，这个函数的逻辑是：实例化损失函数 - 从 DataLoader 获取输入 - 模型推理 - 计算损失 - 重置梯度 - 反向传播 - 更新参数 - 计算错误率。

```python
def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    error_count = 0
    loss = 0
    for x, y in dataloader:
        if opt is None:
            model.eval()
        else:
            model.train()
        y_pred = model(x)
        batch_loss = loss_func(y_pred, y)
        loss += batch_loss.numpy() * x.shape[0]
        if opt is not None:
            opt.reset_grad()
            batch_loss.backward()
            opt.step()
        y = y.numpy()
        y_pred = y_pred.numpy()
        y_pred = np.argmax(y_pred, axis=1)
        error_count += np.sum(y_pred != y)
    return error_count / len(dataloader.dataset), loss / len(dataloader.dataset)
    ### END YOUR SOLUTION
```

- Train MNIST  
本方法用于在 MNIST 数据集上训练一个 MLP ResNet，本方法的逻辑是：实例化 Dataset- 实例化 DataLoader- 实例化模型 - 实例化优化器 - 迭代 epoch

```python
def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(data_dir+"/train-images-idx3-ubyte.gz", data_dir+"/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(data_dir+"/t10k-images-idx3-ubyte.gz", data_dir+"/t10k-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size)
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
        test_error, test_loss = epoch(test_dataloader, model)
        # print(f"Epoch {i+1}/{epochs} Train Error: {train_error:.4f} Train Loss: {train_loss:.4f} Test Error: {test_error:.4f} Test Loss: {test_loss:.4f}")
    return train_error, train_loss, test_error, test_loss
    
    ### END YOUR SOLUTION
```

## hw2 小结

到这里，hw2 就已经完结啦。拖拖拖，拖了一个月才做完，本课程的 test 不是很严格，在 Debug hw2 的过程中发现了不少 hw1 中的错误。遇到问题除了自己调试，也建议参考一下别人的实现，能够提升找到问题所在的效率。

# 参考文档

[^1]: [zhuanlan.zhihu.com/p/579465666](https://zhuanlan.zhihu.com/p/579465666)