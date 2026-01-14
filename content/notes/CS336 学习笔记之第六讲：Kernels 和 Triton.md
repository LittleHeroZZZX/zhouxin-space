---
title: CS336 学习笔记之第六讲：Kernels 和 Triton
tags:
  - CS336
  - LLM
date: 2026-01-13T12:53:00+08:00
lastmod: 2026-01-13T20:33:00+08:00
publish: true
dir: notes
slug: notes on cs336 lecture 6 kernels and triton
---

---

> TL;DR 本讲是 CS336 系列笔记的第六讲。本节从算子融合的必要性出发，横向对比了手写 CUDA、使用 Triton 以及 PyTorch 2.0 编译技术三种实现方式。重点解析了 Triton 如何通过“块级（Block-level）”抽象简化显存管理，并以 GeLU、Softmax 和 Matmul 为例，展示了利用共享内存（SRAM）和分块（Tiling）技术打破访存墙的关键技巧。

## 算子融合 (Kernel Fusion)

### 动机：Warehouse vs Factory

老师有一个非常经典的类比：
- DRAM (显存) 就像是一个巨大的仓库（Warehouse），容量大但存取慢。
- SRAM (共享内存/寄存器) 就像是工厂（Factory）的车间，容量小但处理快。

在执行深度学习模型时，如果不做算子融合，数据需要在仓库和工厂之间反复搬运：
1. 从 DRAM 读取数据 $x$ -> 计算乘法 -> 写回 DRAM
2. 从 DRAM 读取数据 -> 计算加法 -> 写回 DRAM
3. ...

这种模式下，内存带宽（Memory Bandwidth） 成为了绝对瓶颈。算子融合的核心思想就是：将数据一次性搬运到工厂，完成一系列计算（乘、加、Tanh 等）后，再统一写回仓库。

### 实例：GeLU

以 GeLU 激活函数为例，其数学表达式包含乘法、加法、Tanh 等多个操作。

- Manual 实现：使用 PyTorch 基础算子拼凑（`x * 0.5 * ...`），每次操作都会触发一次 Kernel Launch 和全局显存读写。
- Fused 实现：PyTorch 官方的 `F.gelu` 是经过融合的，只触发一次 Kernel Launch。

性能测试显示，Fused 版本比 Manual 版本快得多（源码中约为 7-8 倍差距），且 Profiling 结果显示 Manual 版本充斥着大量琐碎的 Kernel 调用。

## CUDA Kernels：打开黑盒

为了追求极致性能，我们可以直接编写 CUDA C++ 代码。

### 执行模型

CUDA 的执行层级映射了硬件结构：
- Grid：对应整个计算任务，由多个 Thread Block 组成。
- Thread Block：对应一个 SM，块内线程可以共享 Shared Memory 并同步。
- Thread：最小执行单元，处理单个数据点。

### 代码与限制

通过 `torch.utils.cpp_extension.load_inline` 可以方便地在 Python 中内联 CUDA 代码。虽然手动管理线程索引（`blockIdx`, `threadIdx`）和内存能够带来性能收益，但其开发门槛极高：

- 必须手动处理内存合并访问（Coalescing）。
- 必须手动管理 Shared Memory 的数据搬运。
- 代码冗长且容易出错（Off-by-one error）。

## Triton：Python 时代的 GPU 编程

OpenAI 于 2021 年推出的 Triton 旨在降低 GPU 编程门槛。它引入了 Block-level 的编程抽象，让开发者关注“数据块”而非“单个线程”。

### Triton vs CUDA

Triton 编译器自动处理了许多 CUDA 中需要手动优化的痛点：

|特性|CUDA|Triton|
|---|---|---|
|内存合并访问 (Coalescing)|手动|自动|
|共享内存管理 (Shared Mem)|手动|自动|
|SM 内部调度|手动|自动|
|SM 间调度|手动|手动|

### 实现 GeLU

在 Triton 中，我们通过 `tl.program_id` 获取 Block ID，并计算出该 Block 需要处理的数据指针偏移量。计算过程完全向量化，代码非常接近 Python 原生写法。

```Python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. 计算当前 Block 处理的数据范围
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 2. 加载数据 (Triton 自动处理合并访问)
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # 3. 计算 (会被编译为高效的 PTX 指令)
    y = 0.5 * x * (1 + ... ) 
    
    # 4. 写回
    tl.store(y_ptr + offsets, y, mask=mask)
```

查看生成的 PTX 汇编代码可以看到，Triton 编译器自动进行了 Thread Coarsening（线程粗化），即一个线程可能处理多个元素以提高指令级并行度。

## PyTorch Compilation

这是 PyTorch 2.0 的杀手级特性。
- 原理：通过 `torch.compile(model)`，PyTorch 会捕获计算图，分析算子间的依赖关系，并自动调用 Triton 后端生成融合后的 Kernel。
- 效果：在 GeLU 的例子中，`torch.compile` 生成的 Kernel 性能与手写 Triton 几乎一致，远超 Manual 实现。
- Profiling：在 Profiler 中可以看到类似 `triton_poi_fused_add_mul_tanh...` 的名称，这标志着自动融合生效了。

## 进阶计算：Softmax 与 Matmul

### Triton Softmax：Reduce 操作

Softmax 是一个典型的 Row-wise 操作：$y_i = \frac{e^{x_i}}{\sum e^{x_j}}$。
- 朴素实现：需要多次遍历显存（求 Max -> 减 Max -> 求 Exp -> 求 Sum -> 除法）。对于 $M \times N$ 的矩阵，读写次数高达 $5MN + M$。
- Triton 实现：
    1. 每个 Block 处理矩阵的一行（Row）。
    2. 将整行数据加载到 SRAM。
    3. 在 SRAM 中完成 Max、Exp、Sum 的计算（利用 `tl.max`, `tl.sum`）。
    4. 写回结果。
    - 收益：显存读写降低至 $MN$ 次，实现数倍加速。

### Triton Matmul：分块

矩阵乘法（$C = A \times B$）是计算密集型任务，但对于大矩阵，显存带宽依然可能成为瓶颈。
- 朴素做法：计算 $C$ 的每个元素都需要读取 $A$ 的一行和 $B$ 的一列，导致大量重复读取。
- Tiling 策略：
    1. 将 $A$ 和 $B$ 分割成小块（Tiles）。
    2. 将 Tiles 加载到 Shared Memory 中。
    3. 复用 Shared Memory 中的数据计算出 $C$ 的一部分局部和。

通过 Tiling，我们将对慢速 DRAM 的访问转换为了对快速 Shared Memory 的访问。此外，Triton 还支持 L2 Cache 优化（Grouped Ordering），通过调整 Block 的执行顺序，尽可能提高 L2 缓存的命中率。
