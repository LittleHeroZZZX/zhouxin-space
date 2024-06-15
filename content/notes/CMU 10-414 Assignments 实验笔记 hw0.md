---
title: CMU 10-414 Assignments 实验笔记 hw0
tags:
  - CUDA
  - 深度学习系统
date: 2024-06-06T13:28:00+08:00
lastmod: 2024-06-15T20:23:00+08:00
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

# 参考文档

[^1]: [zhuanlan.zhihu.com/p/579465666](https://zhuanlan.zhihu.com/p/579465666)