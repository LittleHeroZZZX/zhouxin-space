---
title: 如何在VsCode中“优雅”地配置cmake —— 以PaddlePaddle为例
tags: 
date: 2024-11-15T11:09:00+08:00
lastmod: 2024-11-17T14:41:00+08:00
publish: true
dir: notes
slug: 
---

通过本文，你将了解如何在 VSCode 中配置 CMake 项目，包括但不限于语法高亮、代码跳转、CMake 配置、构建、测试。

## 环境说明

本文使用 WSL Ubuntu 22.04 作为演示环境，VSCode 版本为 `1.95.2`，使用项目为 [PaddlePaddle](https://github.com/PaddlePaddle/paddle)。

VSCode 中需要安装如下插件：
- [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)：配合 clangd server 实现对 C/C++ 的代码高亮、补全、跳转、重构等
- [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)：为 CMake 项目的提供支持
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)：为项目提供 Pythjon 支持，包括高亮、跳转、调试等

还需要安装如下工具：
- [clangd](https://clangd.llvm.org/installation)：clangd sever，为 C/C++ 的代码解析提供支持

CMake、编译器、调试器等工具默认可用。

## 配置 CMake 项目

在 VSCode 中安装 CMake Tools 插件后第一次打开 CMake 项目，VSCode 默认会自动进行配置，即默认执行 `CMake: Configure` 命令。如果检测到多个编译器，会提示用户选择一个。此时 CMake 插件还没有做任何配置，这时候进行 Configure 大概率是不符合用户预期的，我们可以使用 `ESC` 退出 Configure 过程。  
![编译器选择页面](https://pics.zhouxin.space/202411151536207.webp)

我们首先对 VSCode 插件进行配置。在 VSCode 中打开 `settings-Workspace`，在 workspace 修改的设置内容将以 `.vscode/settings.json` 文件的形式保存在项目根文件夹中。通过为每个项目保存不同的配置，可以方便地且“优雅”地在不同项目之间切换。依据下图，在设置中找到 CMake Tools 插件的设置。

![image.png](https://pics.zhouxin.space/202411151546650.webp)

其中有几项值得关注，可以根据自己需要进行修改：
- Build Directory：指定 CMake 构建目录路径
- Build Environment & Configure Environment：指定配置和构建阶段环境变量
- Build Args & Configure Args：指定配置和构建阶段额外命令行参数
- Cmake Path：指定 Cmake 可执行文件路径
- Generator：指定生成器，例如 Ninja

插件配置完成后，在 `.vscode/settings.json` 文件中就可以看到对应的修改：

```json
{
    "cmake.configureArgs": [
        "-DPY_VERSION=3.12",
        "-DWITH_GPU=OFF",
        "-DWITH_TESETING=ON",
        "-DPYTHON_EXECUTABLE=/home/zhouxin/miniconda3/envs/paddle-dev/bin/python"
    ],
    "cmake.configureSettings": {
        "CMAKE_EXPORT_COMPILE_COMMANDS": true
    },
    "cmake.buildDirectory": "${workspaceFolder}/build_mask",
    "cmake.automaticReconfigure": false,
    "cmake.configureOnOpen": false,
    "cmake.configureOnEdit": false,
    "cmake.generator": "Ninja",
}
```

接下来可以为整个项目指定编译工具链，使用快捷键 `Ctirl+Shift+P` 唤起 VSCode 命令面板，并搜索 `cmake: kits` 可以找到 `CMake: Edit User-Local CMake Kits`，在该文件内，可以配置多个不同的编译工具链，相关说明见：[vscode-cmake-tools/docs/kits.md at main · microsoft/vscode-cmake-tools · GitHub](https://github.com/microsoft/vscode-cmake-tools/blob/main/docs/kits.md)。然后使用命令 `CMake: Select a Kit` 选择为本项目选择一套合适的工具链。

一切就绪，接下来就可以对 CMake 项目进行 Configure 操作。在命令面板找到 `CMake Configure`，执行之。在 VSCode Output 面板中切换到 CMake，可以看到输出的日志，如果有错误，可以根据错误 Debug。

![CMake 日志输出面板](https://pics.zhouxin.space/202411171337718.webp)

一般来说，在排除 CMakeList 文件本身出错和环境没准备妥当之后，大概率是某些环境变量的问题。可以在 `.vscode/settings.json` 设置文件中修改某些环境变量值，或者传入某些参数以指定某些工具的路径就可以解决。

## Code Intelligence

Code Intelligence 指的是一系列语法高亮、代码跳转、自动补全、错误检测等等功能的集合，一言以蔽之，就是让 IDE 理解你的代码。对于没有配置好 Code Intelligence 的项目，随意打开一个文件，可能存在大量头文件找不到的报错，函数调用之间跳转基本也都是失败的，IDE[^2] 无法定位到函数源码的位置。

而构建工具，例如 CMake，完全掌握着文件之间的依赖关系 [^1]。在 CMake 配置过程中，可以使用参数 `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` 或者在插件设置中添加如下内容以要求 CMake 在配置过程在构建目录生成包含文件依赖信息的文件 `compile_commands.json`。

```json
"cmake.configureSettings": {
	"CMAKE_EXPORT_COMPILE_COMMANDS": true
},
```

构建工具给出信息之后，还得告诉 clangd 这些“信息”的具体位置。在 `.vscode/settings.json` 文件中添加如下内容：

```json
"clangd.arguments": [
    "--compile-commands-dir=${workspaceFolder}/build", // 指定编译信息所在目录
    "-j=20",                                        // 设置并行任务数为20
    "--background-index",                           // 启用后台索引
    "--pch-storage=memory",                         // 将预编译头存储在内存中
    "--limit-results=500",                          // 限制结果数量为500
    "--log=info"                                    // 设置日志级别为info
],
```

其中第一行指定 `compile_commands.json` 文件所在目录，默认为 CMake 的构建目录。其余为推荐使用的一些其它配置，加速大型项目的解析。

接下来重新执行 `CMake: Configure`，并在配置完成后重启 clangd 服务器 `clang: restart language server`。经过此番折腾，IDE 已经能够对大部分头文件进行解析，并正确实现文件中的跳转。

然而，在 Paddle 的项目中，仍有许多函数无法被正确解析。这是由于在 `Paddle/third_party` 文件中有众多第三方工具的源码。这些工具将在 CMake build 过程中被安装到构建目录之内，这些目前仍无法解析的函数依赖这些第三方工具。使用命令 `CMake: Build` 构建整个项目，完成之后所有函数就能够被正常解析了！

## Debug

Debug 取决于具体的项目，在 VSCode 使用 `.vscode/launch.json` 对 Debug 进行配置。关于对 Python 和 CUDA/C/C++ 代码进行联合调试的内容，可以查看另一篇文章：[在VSCode中对CUDA和Python代码进行联合调试 | 周鑫的个人博客](https://www.zhouxin.space/notes/joint-debgugging-of-cuda-and-python-in-vscode/)。

## 测试

在 VSCode 中，使用 Test Explorer 对测试进行管理和配置 [^3]，VSCode 本身不提供特定语言的测试配置，而是以插件的形式扩展特定语言的测试支持。在插件中搜索 `@category:"testing"` 可以查看所有测试插件。`Python` 和 `CMake Tools` 插件似乎自带对 Python Test 和 CTest 的支持，不需要额外安装测试插件。

在 VSCode 的测试面板，可以看到所有测试项目。下图展示了来自 CMake 的 CTest 和来自 Python 的 Python Tests 项目，Paddle 的那个测试还没确定怎么来的。Anyway，能跑就行😎。

![测试面板](https://pics.zhouxin.space/202411171438048.webp)

# 参考

[^1]: [JSON Compilation Database Format Specification — Clang 20.0.0git documentation](https://clang.llvm.org/docs/JSONCompilationDatabase.html)
[^2]: 一般不认为 VSCode 是一个集成开发环境，但是 VSCode 配合一系列插件说是 IDE 也不为过，并且是具有高度可定制能力的 IDE
[^3]: [Testing in Visual Studio Code](https://code.visualstudio.com/docs/editor/testing)