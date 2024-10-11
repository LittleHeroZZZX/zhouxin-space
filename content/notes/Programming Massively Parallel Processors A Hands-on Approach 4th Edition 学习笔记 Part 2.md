---
title: Programming Massively Parallel Processors A Hands-on Approach 4th Edition 学习笔记 Part 2
tags:
  - CUDA
date: 2024-10-10T20:09:00+08:00
lastmod: 2024-10-11T23:54:00+08:00
publish: true
dir: notes
slug: note on Programming Massively Parallel Processors A Hands-on Approach 4th Edition part 2
---

若无另外声明，本文图片均截取自原书。

# Chapter 07: Convolution 卷积

本章主要介绍 2D 卷积实现，从朴素版本开始，分别使用常量内存、分块共享内存和 cache 技术依次进行优化。

## 7.1 Background 背景

卷积的定义此处不再赘述，简单来说就是对某个元素及其相邻元素进行加权求和。

## 7.2 Parallel convolution: a basic algorithm 并行卷积

本节将以 2D 卷积为例进行学习。

注意到卷积运算彼此独立，因此可以按照每个线程负责一个元素计算的方式写出并行版本的卷积核。首先确定参数列表：输入矩阵指针 `N`，卷积核指针 `F`，输出矩阵指针 `P`，卷积核半径 `r`，输入矩阵高宽 `height` 和 `width`。

然后确定线程和输出元素之间的映射关系。鉴于输出矩阵是个二维矩阵，因此可以将线程也组织为二维形式，并且每个线程负责计算一个元素。每个 block 最多有 1024 个线程，因此最多计算 1024 个元素。对应核函数为：

```c
__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P,
    int r, int width, int height) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            inRow = outRow - r + fRow;
            inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow*(2*r+1) + fCol]*N[inRow*width + inCol];
            }
        }
    }
    P[outRow*width + outCol] = Pvalue;
}
```

该核函数通过两层循环对感受野进行遍历，使用寄存器变量 `Pvalue` 进行暂存，使用一个 `if` 进行感受野边界判断。

不难发现，上述代码存在控制流分歧。处理四周边界的线程在条件判断中存在分歧。分歧的影响程度取决于矩阵的大小，对于较大输入和较小卷积核，分歧的比例很小，反之影响很大。

另一个更为严峻的影响因素是内存带宽，上述代码浮点操作数和访存量的带宽比值为 0.25 OP/B（第 11 行的两次计算比上两次 8 字节浮点数访存）。这使得访存大大拖累了计算过程。

## 7.3 Constant memory and caching 常量内存和缓存

在卷积中，卷积核有三个良好性质：1️⃣ 卷积核通常都比较小，其半径不超过 7，即便是 3D 卷积中权重数量也不超过 7 的立方即 343 个元素；2️⃣ 在卷积过程中，卷积核权重不会变化；3️⃣ 所有线程都按照相同的次序访问同一个卷积核。

上述三个特性使得卷积核非常适合放在常量内存和缓存中。常量内存在核函数执行过程中不能被修改，且只有 64KB 大小。常量内存需要在主机端进行申请和拷贝，假设使用编译时常量 `FILTER_RADIUS` 来指定核函数半径，则使用如下代码声明常量内存：

```c
#define FILTER_RADIUS 2
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
```

需要注意的是常量内存必须在全局作用域中声明，即不能在主机函数中进行声明。

使用 `cudaMemcpyToSymbol` 函数将数据从主机拷贝到常量内存中：

```c
cudaMemcpyToSymbol(F, F_h, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float))
```

其中 `F_h` 表示主机上的 F。

保存在常量内存上的变量是全局变量，因此不需要将卷积核作为参数传给核函数，因此相比第一版核函数，除了函数签名外，几乎不需要修改：

```c
__global__ void convolution_2D_const_mem_kernel(float *N, float *P, int r,
                                                int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow][fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

```

CUDA C 中变量作用域遵循 C 语言规则，因此如果分文件声明和引用全局变量，需要使用 `extern` 关键字进行外部引用。

常量内存变量也保存在 DRAM 中，但是由于已知该变量在核函数运行时不可变，因此运行时将指导硬件对其采用更激进的 cache 策略。

与共享内存或者寄存器不同，cache 对程序员是不可见的，其由硬件和运行时控制。cache 成本相当昂贵，尤其是，如果需要支持写操作。而常量变量不可写入且比较小的特性，使得在硬件上能够以较低的代价实现常量缓存即 constant cache。

在引入常量内存之后，浮点操作数和访存量的带宽比值翻了个翻，达到了 0.5 OP/B。

## 7.4 Tiled convolution with halo cells 带有边界单元的分块卷积

分块卷积可以缓解内存瓶颈。首先来定义输入和输出分块的概念。输出矩阵中的一块指的是一个 block 中所有线程计算的元素的集合，如果由输出矩阵每个元素有一个线程负责计算，每个 block 包含 16 个线程，那么输出矩乘就是按照每块 4×4 进行分块。当然在实际中每个 block 至少要有一个线程束那么多线程，以便最大化占用率和数据复用率。

自然地，输入块就被定义为计算一个输出块需要用到的元素集合。如下图所示，如果卷积核半径为 2，那么输入块为蓝色部分（深蓝和浅蓝），输出块为绿色部分。其中，浅蓝色被称为 halo cells 即边界单元。  
![image.png](https://pics.zhouxin.space/202410112036856.png?x-oss-process=image/quality,q_90/format,webp)

进行分块时候，就首先由同一个 block 内的进行将数据通过合并访存将其读入共享内存。注意到输入内存和输出内存大小存在差异，有两种线程组织方式来应对这一差异。第一种是启动与输入块元素数量相同的线程，这种方式便于加载数据，但在计算时需要闲置部分线程。另一种方式是启动与输出块相同的线程，这种方式在加载数据阶段较为复杂，但是整体线程利用率更高。本书将以方式一为例。

```c
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))
__constant__ float F_c[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P,
                                                      int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    // Loading input tile
    __shared__ N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    // Turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0
            && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Pvalue += F[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

```

代码如上所示，这部分代码做到了 self-explain，不再解释。

接下来计算上述代码的浮点操作数和访存量的带宽比值，这里也将边界线程不用加载 ghost cell 当作一次访存。在每一个 block 内部，其需要加载 `IN_TILE_DIM*IN_TILE_DIM` 个浮点数到共享内存中，进行了 `OUT_TILE_DIM*OUT_TIME_DIM*(2*FILTER_RADIUM+1)*(2*FILTER_RADIUM+1)` 浮点数运算。对于 32×32 的输入和 5×5 的卷积核，比值为 9.57 OP/B。

![image.png](https://pics.zhouxin.space/202410112305262.png?x-oss-process=image/quality,q_90/format,webp)

上表展示了不同输入维度对应的浮点操作数和访存量的带宽比值，不难发现卷积核越大，该比值越高。

## 7.5 Tiled convolution using caches for halo cells 为边界元素使用缓存的空洞卷积

注意到如下事实：一块的 halo cells 可能是另一块的内部元素，因此当一块在试图访问其 halo cells 时，很有可能其已经被加载到 L2 cache 中。应用如上特性，本章将介绍一种具有相同输入和输入 tile size 的分块卷积算法，其只把内部元素加载到共享内存，而不显式加载 halo cells。

代码如下：

```c
#define TILE_DIM 32
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N,
                                                            float *P, int width, int height) {
    int col = blockIdx.x*TILE_DIM + threadIdx.x;
    int row = blockIdx.y*TILE_DIM + threadIdx.y;
    //loading input tile
    __shared__ N_s[TILE_DIM][TILE_DIM];
    if(row<height && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    // Calculating output elements
    // turning off the threads at the edges of the block
    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                if (threadIdx.x-FILTER_RADIUS+fCol >= 0 &&
                    threadIdx.x-FILTER_RADIUS+fCol < TILE_DIM &&
                    threadIdx.y-FILTER_RADIUS+fRow >= 0 &&
                    threadIdx.y-FILTER_RADIUS+fRow < TILE_DIM) {
                    Pvalue += F[fRow][fCol]*N_s[threadIdx.y+fRow][threadIdx.x+fCol];
                }
                else {
                    if (row-FILTER_RADIUS+fRow >= 0 &&
                        row-FILTER_RADIUS+fRow < height &&
                        col-FILTER_RADIUS+fCol >=0 &&
                        col-FILTER_RADIUS+fCol < width) {
                        Pvalue += F[fRow][fCol]*
                            N[(row-FILTER_RADIUS+fRow)*width+col-
FILTER_RADIUS+fCol];
                    }
                }
            }
        }
        P[row*width+col] = Pvalue;
    }
}
```

在代码中既要进行 ghost cells 判断，也要进行 halo cells 判断。通过两层 for 循环遍历感受野，在循环内部首先判断是否为内部元素，如果是 halo cells，则继续判断是否为 ghost cells。
