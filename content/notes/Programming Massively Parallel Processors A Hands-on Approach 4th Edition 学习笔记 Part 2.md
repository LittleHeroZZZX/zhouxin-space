---
title: Programming Massively Parallel Processors A Hands-on Approach 4th Edition 学习笔记 Part 2
tags:
  - CUDA
date: 2024-10-10T20:09:00+08:00
lastmod: 2024-10-24T23:16:00+08:00
publish: true
dir: notes
slug: note on Programming Massively Parallel Processors A Hands-on Approach 4th Edition part 2
math: "true"
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

# Chapter 08: Stencil 模板计算

注意，本章中 Stencil 模板指的是一种计算模式，常用于科学计算领域，与 C++ 中的 template 是完全不同的两个概念。stencil 用于计算一系列具有物理意义的离散量，其与卷积操作有相通之处，即同意一个元素及其周围元素计算新值。与之不同的是，用于计算的元素和对应的权重由微分方程。此外，在迭代过程中，输出值可能取决于边界条件，stencil 计算可能具有依赖性，并且科学计算往往要求更高的浮点精度。这些区别决定了 stencil 和卷积具有不同的优化技术。

## 8.1 Background 背景

使用计算机进行数值计算的第一步就是将其离散化。我们使用结构化网格对 n 维欧式空间进行规则划分，在一维中使用线段、二维使用矩形、三维使用长方体。下图中对一维函数 $y=\sin (x)$ 按照长度为 $\pi/6$ 进行了划分。  
![image.png](https://pics.zhouxin.space/202410150922860.png?x-oss-process=image/quality,q_90/format,webp)

在离散表示中，不再网格点上的值要使用例如线性插值、样条插值技术通过周围网格点计算得出。计算精度取决于网格的密度，密度越大越精确。精度还取决于数据表示的精度，例如双精度浮点数的精度大于半精度浮点数，但更高的精度意味着消耗更多的片上内存，可能构成计算瓶颈。

模板制定了如何通过一点及其周围点的值通过有限差分的方法计算该点的其它数学量，而偏微分方程则制定了该数学量的具体表达式。例如，计算一维函数的一阶导数有一个经典的方法是：

{{< math_block >}}
f^\prime(x) = \frac{f(x+h)-f(x-h)}{2h} + O(h^2)
{{< /math_block >}}

其中 $O(h^2)$ 是误差项，从中可以看出，误差取决于网格划分的密度。

假设 `F[i]` 是保存函数值的数组，需要计算一阶导数 `FD[i]`，显然可以通过表达式 `FD[i] = (F[i+1]-f[i-1])/(2*h)` 进行迭代计算，进一步地，可以等价转换为 `FD[i] = F[i+1]/(2*h)-F[i-1]/(2h)`，上述表达式可以记为对 `[i-1, i, i+1]` 按照权重 `[-1/2h, 0, 1/2h]` 进行 stencil 操作。

显然，如果要计算偏微分方程，则需要使用多维网格进行划分和计算。

在本章中，我们主要关注一种计算模式：stencil 将被应用到全局以计算全局所有数学量的值，这类计算模式被称为模板扫描 stencil sweep。

## 8.2 Parallel stencil: a basic algorithm 一种基本算法：并行模板

假定在一次 stencil sweep 中输出元素之间彼此独立，并且网格边界元素保存了这个微分方程的边界值，在单个 sweep 中不会修改。例如，在下图中输出部分的阴影就是所谓的边界值，其在 sweep 中不会被修改。上述假设是有意义的，因为 stencil 主要用于有边界的微分方程问题。  
![image.png](https://pics.zhouxin.space/202410151057594.png?x-oss-process=image/quality,q_90/format,webp)

下述代码展示了一个计算 3d stencil 的核函数，每个 block 负责计算 output 的一个 tile，每个 thread 负责计算 tile 中的一个元素。

```c
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i*N*N + j*N + k] = c0*in[i*N*N + j*N + k]
            + c1*in[i*N*N + j*N + (k - 1)]
            + c2*in[i*N*N + j*N + (k + 1)]
            + c3*in[i*N*N + (j - 1)*N + k]
            + c4*in[i*N*N + (j + 1)*N + k]
            + c5*in[(i - 1)*N*N + j*N + k]
            + c6*in[(i + 1)*N*N + j*N + k];
    }
}

```

上述代码的浮点操作与访存比为：13/(7\*4) = 0.46 OP/B。

## 8.3 Shared memory tiling for stencil sweep 为模板扫描进行共享内存分块

在 stencil 上进行共享内存分块与卷积类似，但也有一些微妙的不同。下图展示了计算一个 output 中的 tile 中 stencil 涉及到的输入，与卷积不同的是，四个角落并不需要被使用。在进行寄存器分片时，这点尤其重要。对于共享内存分块，这一特性也会导致共享内存优化效果弱于卷积版本，这是由于不同线程复用的元素个数相比卷积更少了。  
![image.png](https://pics.zhouxin.space/202410151129310.png?x-oss-process=image/quality,q_90/format,webp)

共享内存的优化的上限，会随着维度和阶数（类似于卷积中的半径 radius）显著减小。例如，对于 2d stencil 来说，一阶对应 3\*3 卷积，理论上限分别为 2.5 OP/B 和 4.5 OP/B，二阶对应 5\*5 卷积，理论上限分别为 4.5 OP/B 和 12.5 OP/B，三阶对应 7\*7 卷积，理论上限分别为 6.5 OP/B 和 24.5 OP/B。而对于 3d stencil，这一效应要显著得多得多，3d 三阶 stencil 对应半径为 7 的 3d 卷积，理论上限分别为 9.5 OP/B 和 171.5 OP/B。

使用共享内存优化后的代码如下所示：

```c
__global void stencil_kernel(float* in, float* out, unsigned int N) {
    int i = blockIdx.z*OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if(i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    }
    __syncthreads();
    if(i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM-1 && threadIdx.y >= 1
           && threadIdx.y<IN_TILE_DIM-1 && threadIdx.x>=1 && threadIdx.x<IN_TILE_DIM-1){
            out[i*N*N + j*N + k] = c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                + c1*in_s[threadIdx.z][threadIdx.y][threadIdx.x-1]
                + c2*in_s[threadIdx.z][threadIdx.y][threadIdx.x+1]
                + c3*in_s[threadIdx.z][threadIdx.y-1][threadIdx.x]
                + c4*in_s[threadIdx.z][threadIdx.y+1][threadIdx.x]
                + c5*in_s[threadIdx.z-1][threadIdx.y][threadIdx.x]
                + c6*in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
        }
    }
}
```

上述代码中，`ijk` 标识了本线程负责加载的元素在 `in` 矩阵中的索引，同时也标识着本线程负责计算的元素在 `out` 中的索引，其中负责加载 halo 和 ghost cell 的 thread 不需要计算输出元素。第 10 行 `if` 用于排除计算边界值的线程，第 11 行 `i`f 用于排除加载 halo 和 ghost cell 的线程。

上述代码的 OP/B 值计算过程为：假设 input tile 每个维度的 length 为 T，那么 output tile 的每个维度的 length 为 T-2，每个 block 负责计算 (T-2)^3 个元素，共有 13\*(T-2)^3 个浮点运算；而每个 block 需要加载 T^3 个元素，因此 OP/B 值为 $\frac{13}{4}\times (1-\frac{2}{T})^3$。

T 越大，OP/B 值越大，理论上限为 13/4 = 3.25 OP/B。由于 block 中线程数量限制，T 最大取 8，此时尚未考虑共享内存限制。当 T 为 8 时，OP/B 仅为 1.37，这是由于 halo 元素在 3d 模板扫描中占比过大，halo 元素的复用率远低于内部元素。

T 较小的另一个缺陷是无法充分利用内存合并访问技术，对于 8×8×8 的 tile 来说，每个线程束都会加载来自 input 不同行的元素，而无法利用内存合并访问。

## 8.4 Thread coarsening 线程粗化

上节提到，共享内存技术在 stencil sweep 上加速效果并不显著，这是由于线程之间复用元素的比例小。本节，将通过线程粗化技术，提高粗化后的线程间的元素复用比例以克服原有缺陷。

假设输入 tile 为 6×6×6，如下图左所示（上面、前面、左面的一层被移除），输出 tile 为 4×4×4，如下图右绿色所示。

![image.png](https://pics.zhouxin.space/202410172002187.png?x-oss-process=image/quality,q_90/format,webp)

每个 block 中线程的数量与 x-y 平面中元素数量相同，即有 4\*4=16 个线程。对应核函数的实现代码为：

```c
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z*OUT_TILE_DIM;
    int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    if(iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1)*N*N + j*N + k];
    }
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
    }
    for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1)*N*N + j*N + k];
        }
        __syncthreads();
        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1
               && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i*N*N + j*N + k] = c0*inCurr_s[threadIdx.y][threadIdx.x]
                    + c1*inCurr_s[threadIdx.y][threadIdx.x-1]
                    + c2*inCurr_s[threadIdx.y][threadIdx.x+1]
                    + c3*inCurr_s[threadIdx.y+1][threadIdx.x]
                    + c4*inCurr_s[threadIdx.y-1][threadIdx.x]
                    + c5*inPrev_s[threadIdx.y][threadIdx.x]
                    + c6*inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}
```

上述代码中，在 z 方向上进行迭代，使用三个共享内存分别保存计算计算当前元素在 z 方向上需要的三层元素。

通过将 z 轴上多个线程合并为一个线程来实现线程粗化，这使得每个 block 需要的线程数从 T^3 减少为 T^2，因此 T 可以取到更大的值，例如 32。此时 OP/B 值达到了 2.68 OP/B，对共享内存的需求也从完整的 tile 减少为 tile 中的三层。

## 8.5 Register tiling 寄存器分片

观察上一节代码中 out 的计算公式，不难发现 `inPrev` 和 `inNext` 这两个共享内存各自只被访问了一个元素，因此，我们只需要使用两个寄存器变量保存二者即可。此外，额外使用一个寄存器变量用于保存 `inCurr_s[threadIdx.y][threadIdx.x]`，以加快两个寄存器变量的更新。

```c
out[i*N*N + j*N + k] = c0*inCurr_s[threadIdx.y][threadIdx.x]
                    + c1*inCurr_s[threadIdx.y][threadIdx.x-1]
                    + c2*inCurr_s[threadIdx.y][threadIdx.x+1]
                    + c3*inCurr_s[threadIdx.y+1][threadIdx.x]
                    + c4*inCurr_s[threadIdx.y-1][threadIdx.x]
                    + c5*inPrev_s[threadIdx.y][threadIdx.x]
                    + c6*inNext_s[threadIdx.y][threadIdx.x];
```

使用寄存器分片优化后的代码如下所示：

```c
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z*OUT_TILE_DIM;
    int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;
    float inPrev;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inCurr;
    float inNext;
    if(iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev = in[(iStart - 1)*N*N + j*N + k];
    }
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr = in[iStart*N*N + j*N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }
    for(int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext = in[(i + 1)*N*N + j*N + k];
        }
        __syncthreads();
        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1
               && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i*N*N + j*N + k] = c0*inCurr
                    + c1*inCurr_s[threadIdx.y][threadIdx.x-1]
                    + c2*inCurr_s[threadIdx.y][threadIdx.x+1]
                    + c3*inCurr_s[threadIdx.y+1][threadIdx.x]
                    + c4*inCurr_s[threadIdx.y-1][threadIdx.x]
                    + c5*inPrev
                    + c6*inNext;
            }
        }
        __syncthreads();
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s;
    }
}
```

寄存器优化减少了三分之二的共享内存的使用量，但是并没有减少对全局内存的访存次数。

# Chapter 09: Parallel histogram 并行直方图

本章以直方图计算为例，引入了结果输出位置与数据相关的计算模式，介绍了原子操作及其优劣，使用私有化、粗化和聚合等优化技术进行优化。

## 9.1 Background 背景

对直方图📊的介绍略。

直方图的顺序计算代码如下所示，比较简单：

```c
void histogram_sequential(char *data, unsigned int length,
                          unsigned int *histo) {
  for(unsigned int i = 0; i < length; ++i) {
    int alphabet_position = data[i] - 'a';
    if(alphabet_position >= 0 && alphabet_position < 26)
      histo[alphabet_position/4]++;
  }
}
}
```

## 9.2 Atomic operations and a basic histogram kernel 原子操作和一个基本的直方图核函数

最简单的直方图核函数就是起与元素个数数量相等的线程，每个线程负责对其对应的元素进行归类，这种情况下多个线程可能需要同一个输出参数进行更新，这种冲突被称为输出干扰。此时涉及到了原子操作和条件竞争的概念。

条件竞争指的是多线程同时对结果进行更新，这使得结果取决于这些线程的执行顺序。原子操作指的是独占式地完成 read-modefy-wirte 操作。本节花了大段用于说明什么是条件竞争和原子操作，在 OS 中学过这些概念，此处省略。

CUDA 中提供了一系列支持原子操作的内建函数，其以 `atomicXxx` 进行命名。

现代编译器中往往提供了一系列特殊指令用于支持某些特定功能，例如原子操作或者向量化，其对于程序员来说可能以库函数的形式被调用，但在编译层面该库函数调用不存在函数调用过程，而是直接被编译为对应的编译器指令。

应用原子操作后的直方图核函数如下所示：

```c
__global__ void histo_kernel(char *data, unsigned int length,
    unsigned int *histo) {
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < length) {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alpha_position < 26) {
        atomicAdd(&(histo[alphabet_position/4]), 1);
    }
}
}
```

## 9.3 Latency and throughput of atomic operations 原子操作的延迟和吞吐量

在前几章我们了解到，对全局内存的访问很慢很慢，但只要有足够的线程，我们就可以通过零开销上下文切换技术来隐藏这一延迟，并将延迟转移到 DRAM 带宽。上述操作的前提都是**有足够数量的线程并行访问内存**。遗憾的是，当我们使用对全局内存进行原子操作时，线程对全局内存的读写操作转换为顺序操作，

🙋‍♀️🌰，对于具有 8 通道、64 比特数据位宽、频率为 1G、访问延迟为 200 个时钟周期的 DRAM，其峰值吞吐量为 8 byte\* 2（每个周期传输两次）\*1G\*8 通道=128 GB/s。如果每个元素大小为 4 字节，那么每秒将能够读写 32G 个元素。

与之相反，每次具有一个读、一个写的原子操作的访问周期是 400 个时钟周期，那么每秒做多进行 2.5M 次原子操作。

当然，并非所有的原子操作都在对同一个位置进行修改，但即便数据均匀分布，那么理论上限为 2.5 M \*7 = 17.5M。但在现实中，由于单词中的字母分布并不均匀，实际加速系数也达不到这么高。

增加原子操作吞吐量的一个手段是减少单词访存延迟，可以使用缓存进行优化。因此，原子操作支持对末级缓存进行操作，末级缓存由所有流多处理器共享。对末级缓存的访存时延相较 DRAM 少了一个数量级。

## 9.4 Privatization 私有化

私有化也是增加原子操作吞吐量的一个技术。私有化指的是线程将频繁访问的数据结构拷贝到私有内存中，计算结束后再合并到原数据结构中。

在直方图中，我们可以为每个 block 应用私有化，并在计算结束后将其合并。代码如下：

```c
__global__ void histo_private_kernel(char *data, unsigned int length,
                                     unsigned int *histo) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[blockIdx.x*NUM_BINS + alphabet_position/4]), 1);
        }
    }
    if(blockIdx.x > 0) {
        syncthreads();
        for(unsigned int bin=threadIdx.x; bin<NUM_BINS; bin += blockDim.x) {
            unsigned int binValue = histo[blockIdx.x*NUM_BINS + bin];
            if(binValue > 0) {
                atomicAdd(&(histo[bin]), binValue);
            }
        }
    }
}
```

以块为单位进行私有化的好处是当我们需要进行同步时（合并前要确保使用同一块副本的线程都计算结束）可以直接调用 `syncthreads`。此外，如果直方图的长度够小，还可以在共享内存中声明副本。

## 9.5 Coarsening 粗化

在 CPU 中，我们常常让粗化后的线程对数据进行连续访问，这是为了充分利用 CPU 的缓存机制。

在 GPU 中，由于内存合并访问技术，不应该让线程内部顺序访问连续数据，而是应该让一个线程束内线程单次连续访存。这种分区方式被称为交错分区 interleave partition

```c
__global__ void histo_private_kernel(char* data, unsigned int length,
                                     unsigned int* histo) {
    // Initialize privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin=threadIdx.x; bin<NUM_BINS; bin += blockDim.x) {
        histo_s[binIdx] = 0u;
    }
    
    __syncthreads();
    // Histogram
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i = tid; i < length; i += blockDim.x*gridDim.x) {
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }
    }
    
    __syncthreads();
    // Commit to global memory
    for(unsigned int bin = threadIdx.x; bin<NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[binIdx];
        if(binValue > 0) {
            atomicAdd(&(histo[binIdx]), binValue);
        }
    }
}
```

## 9.6 Aggregation 聚合

在数据中可能存在局部大量重复区域的情况，这种情况下可能导致线程一起对某个位置同时进行原子操作，为了避免这一情况，我们可以聚合这些局部重复结果，即使用一个变量记录当前的类别和该类别对应的数量，知道计算出不同的类别时才将上一个类别的数量添加到公用变量中。上述技术可以将给予大量重复区域的更新事务合并为一个事务，减少了公用变量的访存密度。

```c
__global__ void histo_private_kernel(char* data, unsigned int length,
                                     unsigned int* histo){

    // Initialize privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        histo_s[bin] = 0u;
    }

    __syncthreads();
    // Histogram
    unsigned int accumulator = 0;
    int prevBinIdx = -1;
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i = tid; i < length; i += blockDim.x*gridDim.x) {
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26) {
            int bin = alphabet_position/4;
            if(bin == prevBinIdx) {
                ++accumulator;
            } else {
                if(accumulator > 0) {
                    atomicAdd(&(histo_s[prevBinIdx]), accumulator);
                }
                accumulator = 1;
                prevBinIdx = bin;
            }
        }
    }
    if(accumulator > 0) {
        atomicAdd(&(histo_s[prevBinIdx]), accumulator);
    }
    __syncthreads();
    // Commit to global memory
    for(unsigned int bin = threadIdx.x; bin<NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if(binValue > 0) {
            atomicAdd(&(histo[bin]), binValue);
        }
    }
}
```

# Chapter 10: Reduction 归约

本章介绍了对于归约核函数的一系列优化技术，使用了包括最小化控制流分歧、最小化内存访问分歧、最少全局内存访问、线程粗化等技术。

## 10.1 Background 背景

对归约的介绍略。其中介绍了一个术语 identity value，GPT 将其翻译为单位值，类似于单位元，即在归约运算中（归约都是二元运算），某个数与单位值进行归约操作，结果仍是该数。

## 10.2 Reduction tree 归约树

以 max 算子为例，其归约过程可以用如下一棵归约树进行描述：  
![image.png](https://pics.zhouxin.space/202410191415104.png?x-oss-process=image/quality,q_90/format,webp)

如果想要用上图所示归约树的过程进行归约，这要求归约算子具备结合律。此外，下一节使用的优化技术还要求归约算子具备交换律。

## 10.3 A simple reduction kernel 一个简单的归约核函数

由于在归约过程中不同的线程之间需要进行数据交互，我们首先从单一 block 开始。由于一个 block 中最多有 1024 个线程，因此我们最多能够处理 2048 个元素。  
![image.png](https://pics.zhouxin.space/202410191436804.png?x-oss-process=image/quality,q_90/format,webp)

求和归约核函数如上所述，`i` 表示当前线程被分配到的元素下标，只有对应这个下标的线程才能够对这个元素进行写入。使用 `stride` 表示第 `i` 个元素归约操作的另一个参数的距离。在第一轮中，`stride` 为 1，第 `i` 个元素（即所有偶数位索引）需要与 `i+1` 的元素进行相加操作；第二轮中，`stride` 为 2，只有 `i` 为 4 的整数倍的元素需要与 `i+2` 进行相加操作；...；第 `n` 轮中，`stride` 为 2^(n-1)，只有 `i` 为 2^n 的整数倍，即 `threadIdx.x` 为 `stride` 的整数倍才需要进行归约操作，另一个归约元素为 `i+stride`。归约树如下所示：

![image.png](https://pics.zhouxin.space/202410191537773.png?x-oss-process=image/quality,q_90/format,webp)

## 10.4 Minimizing control divergence 最小化控制流分歧

上一节的实现代码具有严重的控制流分歧，在后几轮迭代中，只有 2 的幂的整数倍的线程才会被激活。控制流分歧会导致低硬件资源利用率。控制分歧的思路是在每轮迭代中，尽可能将被激活的线程集中在一起，如下图所示：

![image.png](https://pics.zhouxin.space/202410191553796.png?x-oss-process=image/quality,q_90/format,webp)  
不难发现，激活线程数量每次都是上一次的一半（向上取整），假设本轮迭代激活了 n 个线程，那么这其中第 `i` 个线程归约运算的两个元素的下标分别为 `i` 和 `n+i`。基于此规律，我们可以写出最小化控制流分歧版本的归约核函数：

```c
__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        *output = input[0];
    }
}
```

## 10.5 Minimizing memory divergence 最小化内存分歧

在 10.3 节中给出的代码还有一个缺陷是内存访问分歧，无法启用合并内存访问技术。而上一节中我们无意中修复了这个问题，所有活动线程在每轮迭代中都是连续访问内存的。

## 10.6 Minimizing global memory accesses 最小化全局内存访问

通过共享内存可以避免频繁访问全局内存，代码也挺简单：

```c
__global__ void SharedMemorySumReductionKernel(float* input) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}
```

## 10.7 Hierarchical reduction for arbitrary input length 任意输入长度的分层归约

在此之前，我们假设输入的长度小于一个 block 内的线程数，这是由于我们仅能够对一个 block 内的线程进行同步。当输入长度在一个 block 内放不下时，就需要将其划分到多个 block。由于缺乏 block 间的同步机制，我们选择在每个 block 内部独立进行归约，并将结果通过原子操作归约到全局结果中。如下图所示：  
![image.png](https://pics.zhouxin.space/202410200849301.png?x-oss-process=image/quality,q_90/format,webp)  
相应代码为：

```c
__global__ SegmentedSumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i] + input[i + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}
```

## 10.8 Thread coarsening for reduced overhead 线程粗化

之前的代码都是一个元素对应一个线程，并采取多个 block。如果硬件资源不够，那么这些线程和 block 将以线程束为单位，并且**顺序**执行。前面提到，在每个 block 中，随着归约迭代的进行，一个 block 中的线程很多将空闲下来，而在迭代后期，每个线程束中的线程中的控制流分歧效应愈发显著。如果所有 block 都是并行执行的，那么上述开销难以避免。但如果这些执行块是顺序执行的，则完全没有必要。

```c
__global__ CoarsenedSumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = input[i];
    for(unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile) {
        sum += input[i + tile*BLOCK_DIM];
    }
    input_s[t] = sum;
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}
```

代码如上所示，每个元素在初始迭代中负责将 `2*COARSE_FACTOR` 个元素而非 2 两个元素相加，其余大致相同。

# Chapter 11: Prefix sum (scan) 前缀和

## 11.1 Background 背景

数学上 inclusive scan 的定义为输入一个具有结合律的算子 $\oplus$ 和一个长度为 n 的向量，输出为：

{{< math_block >}}
[x_0, x_0\oplus x_1,...,(x_0\oplus x_1\oplus...\oplus x_{n-1})]
{{< /math_block >}}

上述公式被称为 inclusive 是由于输出元素中包含了对应位置的输入元素，与之相反的是 exclusive scan，其输出表示为：

{{< math_block >}}
[i, x_0, x_0\oplus x_1,...,(x_0\oplus x_1\oplus...\oplus x_{n-2})]
{{< /math_block >}}

显然二者可以轻易进行转换，因此本章将以 inclusive scan 进行编程。

## 11.2 Parallel scan with the Kogge-Stone algorithm 使用 Kogge-Stone 算法的并行扫描

在并行算法中，如果我们让每个线程负责一个元素计算，其并不会比顺序算法更快，这是由于计算最后一个元素的线程仍需要进行完整的前缀和计算，相当于跑了一次顺序算法。如果硬件资源不足以支撑所有线程并发执行，那么并行算法将比顺序算法更慢，时间复杂度达到 $O(n^2)$

想要提高并行计算速率，就必须在线程之间共享中间结果。这里介绍 Kogge-Stone 算法。算法示意图如下所示：  
![image.png](https://pics.zhouxin.space/202410221112153.png?x-oss-process=image/quality,q_90/format,webp)

上图展示了一种就地算法，即图中的 $y_i$ 和 $x_i$ 表示的是同一个位置。上述代码的输入为数组 `XY`，在迭代开始时，`XY` 中包记录了所有 `x[i]`，在进行 `k` 轮迭代之后，`XY[i]` 中记录了 `x[i]` 即其往前最多共 `2^k` 个元素的和。例如，在进行了 2 轮迭代后，`XY[i] = x[i]+x[i-1]+x[i-2]+x[i-3]`。

整个算法流程为：在迭代开始时，`XY[i] = x[i]`；在第一轮迭代中，`XY[0]` 已经符合要求，不需要计算，除第 0 位以为所有元素都加上其前一个元素，即 `XY[i] = XY[i]+XY[i-1]`；在第二轮迭代中，`XY[0,1]` 均已符合要求，不需要计算，除第 0 和 1 位元素外所有元素都加上与其距离为 2 的元素，即 `XY[i]=XY[i]+XY[i-2]`；在第 `k` 轮迭代中，前 `2^(k-1)` 个元素已经符合结果，除此以外的所有元素都加上与其距离为 `2^(k-1)` 的元素，即 `XY[i] = XY[i]+XY[i-2^(k-1)]`。相应核函数实现为：

```c
__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N){
	__shared__ float XY[SECTION_SIZE];
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N){
		XY[threadIdx.x] = X[i];
	} else {
		XY[threadIdx.x] = 0.0f;
	}
	for(unsigned int stride=1; stride < blockDim.x; stride*=2){
		__syncthreads();
		float temp;
		if(threadIdx.x >= stride)
			temp = XY[threadIdx.x] + XY[threadIdx.x-stride];
		__syncthreads();
		if(threadIdx.x >= stride)
			XY[threadIdx.x] = temp;
	}
	if(i<N){
		Y[i] = XY[threadIdx.x];
	}
}
```

不难发现在公式 `XY[i] = XY[i]+XY[i-stride]` 中具备明显的条件竞争，在上述代码使用一个中间变量 `temp` 暂存计算结果，使用同步确保所有线程都计算完毕后再进行写入操作。

## 11.3 Speed and work efficiency consideration 速度和任务效率考量

并行算法的一个性能指标是任务效率 work efficiency，其表示并行算法的计算量与理论最小计算量的逼近程度。例如，求前缀和最少需要 N-1 即 O(n) 次加法。

Kogge-Stone 算法的任务效率计算公式为：最多有 $\log_2 N$ 轮迭代，每轮迭代需要计算 $N-\text{stride}$ 次加法，总计为：

{{< math_block >}}
\text{work efficiency} = \sum_{\text{stride}} (N-\text{stride}),\ \ \text{for stride}=1,2,4,...,N//2
{{< /math_block >}}

第一项与求和变量 `stride` 无关，求和为 $N\log_2 N$，第二项为等差数列求和，近似于 $N-1$，最终计算效率与二者之和，即 $N\log_2 N - (N-1)$。

好消息是其性能比起 O(n^2) 的朴素算法要好，坏消息是比不上顺序算法 O(n)。尽管其计算操作数比顺序算法多，但是其仅需要 $log_2 N$ 轮迭代即可计算结束，而在顺序算法中需要 $N$ 轮迭代才能算完。在实际中，由于线程束控制流分歧的存在，并不能达到我们的理论任务效率，实际中的任务效率约为 $N\log_2 N$。

我们可以使用迭代次数来比较不同的并行算法，但是迭代次数少的算法并不一定运行速度就快，由于算法消耗的资源和具体硬件资源的限制，可能出现单轮迭代无法并行执行的情况。

Kogge-Stone 算法的缺点是在硬件资源受限的情况下其执行效率很低，并且由于加法次数仍未优化到最低，这些额外的加法也会带来功耗开销。而其优点是，在硬件资源充足的情况下性能很高。

## 11.4 Parallel scan with the Brent-Kung algorithm 使用 Brent-Kung 算法的并行扫描

本文停更。
