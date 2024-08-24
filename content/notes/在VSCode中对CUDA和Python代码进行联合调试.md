---
title: 在VSCode中对CUDA和Python代码进行联合调试
tags:
  - CUDA
  - 调试
date: 2024-08-24T19:29:00+08:00
lastmod: 2024-08-24T21:59:00+08:00
publish: true
dir: notes
slug: joint debgugging of cuda and python in vscode
---

在 cmu10414 hw3 的最后实现矩阵乘法的算子的时候靠肉眼和 printf 实在是调不通，研究了一下怎么在 VSCode 中联合调试 CUDA 和 Python 代码，特此记录。

# 项目准备

原项目中将 CUDA 代码编译为 so 动态链接库供 Python 调用，使用 cmake 进行构建。这里我们来构建一个最小样例进行调试。

整个项目的目录树为：

```text
.
├── CMakeLists.txt
├── python
│   └── test_cuda_hello.py
└── src
    ├── cuda_hello.cu
    └── pybind_wrapper.cpp
```

其中，`cuda_hello.cu` 是待调试的 CUDA 代码，里面定义了一个核函数和一个主机端调用接口：

```cpp
#include <stdio.h>

__global__ void cuda_hello_kernel() {
    printf("Hello from CUDA kernel!\n");
}

extern "C" void launch_cuda_hello() {
    cuda_hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

```

`pybind_wrapper.cpp` 使用 pybind11 将这个函数注册到 Python 中：

```cpp
#include <pybind11/pybind11.h>

extern "C" void launch_cuda_hello();

PYBIND11_MODULE(cuda_hello, m) {
    m.def("hello", &launch_cuda_hello, "A function that launches a CUDA kernel to print Hello");
}
```

在 `test_cuda_hello.py` 文件中，我们将通过动态链接库导入 `hello_cuda` 这个包，并调用其中的 `launch_cuda_hello` 函数：

```python
import sys
import os

# 将 build 目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import cuda_hello

cuda_hello.hello()
```

注意我们编译出的动态链接库文件在 `build` 目录下，因此要先将该目录添加到 Python 的搜索路径再导入。

`CMakeLists.txt` 文件内容为，各代码含义见注释：

```cmake
# 设置 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 设置构建类型为 Debug
set(CMAKE_BUILD_TYPE Debug)

# 设置 CUDA 主机编译器为 g++
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)

# 定义项目名称和支持的语言
project(CudaHello CUDA CXX)

# 设置 C++ 标准为 C++14
set(CMAKE_CXX_STANDARD 14)

# 设置 CUDA 标准为 C++14
set(CMAKE_CUDA_STANDARD 14)

# 启用 CUDA 语言支持
enable_language(CUDA)

# 设置 CUDA 架构（根据 GPU 调整这个值）
set(CUDA_ARCHITECTURES 89)

# 查找 Python 解释器和开发组件
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 查找 pybind11 包
find_package(pybind11 CONFIG REQUIRED)

# 添加 CUDA 文件并创建共享库
add_library(cuda_functions SHARED src/cuda_hello.cu)

# 设置目标属性，指定 CUDA 架构
set_target_properties(cuda_functions PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# 如果是 Debug 模式，为 CUDA 编译器添加调试选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_functions PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
endif()

# 创建 pybind11 模块
pybind11_add_module(cuda_hello src/pybind_wrapper.cpp)

# 将 CUDA 函数库链接到 pybind11 模块
target_link_libraries(cuda_hello PRIVATE cuda_functions)

```

有几个需要注意的点：`set(CUDA_ARCHITECTURES 89)` 显卡架构的参数应该根据自己显卡的型号的 CC 来填，各显卡 CC 值见 NVIDIA 官网：[CUDA GPUs - Compute Capability | NVIDIA Developer](https://developer.nvidia.com/cuda-gpus)；`target_compile_options(cuda_functions PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)` 用于在给 nvcc 指定编译参数 `-g -G`，确保其编译出的主机端和设备端代码都包含调试信息。

准备完以上文件，执行如下命令编译共享库：

```bash
mkdir build
cd build
cmake ..
make
```

编译结束后，在 `build` 文件夹应该会出现一个文件名类似于 `cuda_hello.cpython-3x-x86_64-linux-gnu.so`（Windows 平台后缀为 `.pyd`）的共享库，说明编译成功。

然后执行 `test_cuda_hello.py` 文件，应该就能看到来自 GPU 的输出 `Hello from CUDA kernel!`。

万事俱备，接下来开始调试！

# 手动调试

NVIDIA 提供了 cuda-gdb 工具对 cuda 代码进行调试，具体调试过程为：

1. 在终端输入 `cuda-gdb python --quite`，启动 cuda-gdb 调试器，对 Python 解释器进行调试；
2. 在 cuda-gdb 交互终端中设置断点，例如 `break cuda_hello_kernel` 为 `cuda_hello_kernel` 函数设置断点，或者 `break src/cuda_hello.cu:4` 在 `cuda_hello.cu` 文件的第 4 行打上断点；
3. 在交互终端输入 `run python/test_cuda_hello.py` 执行 Python 解释器，并将 py 文件作为参数传递给它。稍等一会，程序将在断点处停下，并提示：`CUDA thread hit Breakpoint 1, cuda_hello_kernel<<<(1,1,1),(1,1,1)>>> ()`

之后按照正常的 gdb 工具调试即可。

# 配置 VSCode 进行调试

前面已经实现了使用 cuda-gdb 工具进行调试，但我对 gdb 工具不太了解，只会使用基于 GUI 的调试工具。接下来我们配置 VSCode，使之支持对 CUDA 和 Python 代码联合调试。

首先安装插件 [Nsight Visual Studio Code Edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition)，此插件由 NVIDIA 开发，用于在 VSCode 中支持对 CUDA 代码的调试 [^1]。

编辑 `.vscode/launch.json` 文件，输入如下内容，并修改其中 Python 解释器路径为正确值：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Launch",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        },
        {
            "name": "CUDA GDB Server: Launch",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "path/to/python", //修改为Python路径
            "args": ["${file}"],
            "debuggerPath": "/usr/local/cuda/bin/cuda-gdb", // 确认cuda-gdb路径正确
        }
    ],
    "compounds": [
    {
        "name": "Python and CUDA",
        "configurations": ["Python: Launch", "CUDA GDB Server: Launch"]
    }
]
}
```

上面这个文件由三部分组成，第一部分定义了 Python 调试器的相关配置，第二部分定义 cuda-gdb 调试器的配置，第三部分使用 compounds 将两个调试配置组装成一个，在调试时将同时启动这两个调试器。

接下来在 VSCode 中切换到 Run and Debug 面板，并修改调试配置为 `Python and CUDA`，如下图所示：

![image.png](https://pics.zhouxin.space/202408242150909.png?x-oss-process=image/quality,q_90/format,webp)

然后在 py 和 CUDA 文件中打上断点，在**py 文件中**按下快捷键 `F5` 开始调试，代码将在断点处停下：  
![image.png](https://pics.zhouxin.space/202408242155725.png?x-oss-process=image/quality,q_90/format,webp)  
继续运行，其将在 CUDA 断点处停下：  
![image.png](https://pics.zhouxin.space/202408242158812.png?x-oss-process=image/quality,q_90/format,webp)

# 参考文档

[^1]: [NVIDIA Nsight VSCE Documentation](https://docs.nvidia.com/nsight-visual-studio-code-edition/)