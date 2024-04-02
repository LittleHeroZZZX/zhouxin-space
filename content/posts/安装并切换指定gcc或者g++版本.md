---
名称: 安装并切换指定gcc或者g++版本
tags:
  - Ubuntu
创建时间: 2024-04-01 10:58
修改时间: 2024-04-02 11:04
publish: true
---

# 资源存档

# 知其然

**注意：** 该方式将从 PPA 下载 gcc/g++，国内访问很慢，建议参考 [《为apt配置代理》](../../%E4%B8%BAapt%E9%85%8D%E7%BD%AE%E4%BB%A3%E7%90%86.md) 这篇文章，配置好 apt 的代理。
以安装 `g++ 13` 版本（不支持指定小版本号）为例，以下给出用到的命令 [^1]：

``` bash
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y && sudo apt update
sudo apt install gcc-13 g++-13 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 --slave /usr/bin/g++ g++ /usr/bin/g++-13
sudo update-alternatives --config gcc
```

注意，上面第四条指令中的 gcc/g++ 后面的版本号需要根据自己的需要修改。以后一条指令用于可视化调整 gcc 各个版本的优先级。

# 知其所以然

上述过程可以理解为：
1. 添加 PPA 源
2. 安装指定版本的 gcc
3. 使用 `update-alternatives` 工具调整优先级，使得 `gcc` 默认指向 `gcc-13`

## PPA 源

PPA 指的是 Personal Package Archive，即个人软件包存档，其是相对官方仓库的一个概念。Ubuntu 提供了一个官方软件仓库以及该仓库的镜像仓库，该仓库会进行兼容性检查，因此更新较慢 [^2]。
为此，引入了 PPA，即让开发人员自己搭建的非官方软件仓库，以此获取最新的软件版本。
在这里，为了安装 `gcc 13`，我们使用 `add-apt-repository` 命令添加 ppa 仓库 `ppa:ubuntu-toolchain-r/test`。在此之前我们还安装了 `software-properties-common` 工具，以确保正确使用 `add-apt-repository` 命令。

## 安装 gcc

添加 PPA 仓库之后，就可以使用 `apt` 命令正常安装 `gcc`，这里我们使用 `gcc-13` 来指定版本号。注意，只能指定大版本号，该方式不支持指定小版本号。

## update-alternatives 调整优先级

`update-alternatives` 是 Ubuntu 提供的一个维护符号链接的工具，其通过更新符号链接来实现程序在多个版本之间的切换。其使用“替代方案”这一概念，一个替代方案指的是一组可以相互替代的命令，例如 `gcc-10` 和 `gcc-12` 就是 `gcc` 的替代方案。添加替代方案的命令为：

``` bash
update-alternatives --install <link> <name> <path> <priority>
```

`link` 指的是将被创建或者更新的符号链接的地址，例如 `/usr/bin/gcc`；
`name` 指的是替代方案的标识名称，例如 `gcc`；
`path` 指的是符号链接指向的在替代方案中希望使用的具体程序版本或者实现，例如 `/usr/bin/gcc-12`；
`prioritity` 指的是该 `path` 在方案中的优先级，是整数，优先级越高数字越大，在本例中我们根据 `gcc` 版本号给定相应的优先级。

你可能注意到了，我们实际使用的命令是 `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 --slave /usr/bin/g++ g++ /usr/bin/g++-13`，后半部分还有一个参数 `--slave /usr/bin/g++ g++ /usr/bin/g++-13`，这个命令的作用是为主方案添加多个从属方案，即当我们切换 `gcc` 时，自动切换相对应的从属方案 `g++`，其语法是：

``` bash
update-alternatives  --install <link> <name> <path> <priority> [--slave <link> <name> <path>] ...
```

在从属方案中，优先级与主方案一致，不需要指定优先级。

在为一个替代方案提供了多个候选项的情况下，可以使用 `sudo update-alternatives --config <name>` 命令，通过交互界面选择方案。

# 参考文档

[^1]: [How to Install GCC Compiler on Ubuntu 22.04](https://www.dedicatedcore.com/blog/install-gcc-compiler-ubuntu/)
[^2]: [ubuntu ppa源管理 - 简书](https://www.jianshu.com/p/6aa5575e8a34)