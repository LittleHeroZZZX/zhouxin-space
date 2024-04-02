---
title: GUN Make
tags:
  - Make
  - Cpp
date: 2024-03-29T12:23:00+08:00
lastmod: 2024-04-02T20:24:00+08:00
publish: true
---

# 概述

## 编译流程

> 总结一下，源文件首先会生成中间目标文件，再由中间目标文件生成可执行文件。在编译时，编译 器只检测程序语法和函数、变量是否被声明。如果函数未被声明，编译器会给出一个警告，但可以生成 Object File。而在链接程序时，链接器会在所有的 Object File 中找寻函数的实现，如果找不到，那到就 会报链接错误码（Linker Error），在 VC 下，这种错误一般是：Link 2001 错误，意思说是说，链接器 未能找到函数的实现。你需要指定函数的 Object File

编译只需要给出声明即可，具体定义在链接阶段确定。

# makefile 介绍

## makefile 的基本规则

```make
target ... : prerequisites ...
    recipe
    ...
    ...
```

**target：** 目标文件、可执行文件、或者标签  
**prerequisites：** 生成该 target 所依赖的 target 或者文件  
**recipe：** 该 target 执行的 shell 命令

直白地说，prerequisites 中如果有一个以上的文件比 target 文件要新的话，recipe 所定义的命令就会被执行。

## 声明和使用变量

``` make
// 声明变量
objs = main.o dep.o

// 使用变量
edit : $(objs)
	cc -o edit $(objs)
```

## make 自动推导

GNU 的 make 支持根据 `.o` 文件自动推导相同文件名的 `.c` 依赖，可以省略不写；此外，还支持自动推导出编译 `.o` 的命令，同样可以省略不写。

# 书写规则

## 规则语法

```make
targets ... : prerequisites ...; command
    command
    ...
    ...
```

`targets` 是文件名，可以有多个目标，用空格分隔开；`command` 是需要执行的命令行，如果与 `targets` 在同一行，使用分号分隔开，否则需要前导 `Tab`。

## 使用通配符

通配符包括 *，？和~。  
波浪号~指的是当前用户的家目录，`~User/...` 指的是指定用户 `User` 的家目录。

## 文件搜寻

makefile 提供了一个特殊变量 `VPATH` 用来指示源文件所在目录。`VPATH` 可以赋值多个路径，使用 `:` 隔开即可。make 搜索的第一优先级为当前目录，其次是 `VPATH` 指向的目录。

此外，make 还提供了关键字 `vpath`，用于灵活指定搜索路径，其有三种用法：  
`vpath <pattern> <dirs>`：为符合模式 `pattern` 的文件指定搜索目录 `dirs`  
`vpath <pattern>`：为符合模式 `pattern` 的文件清除搜索目录  
`vpath`：清除所有文件的搜索目录

`vpath` 使用 `%` 来指定匹配模式，`%` 可以匹配零个或者若干字符。

如果一个文件同时匹配多个模式，则根据 `vpath` 出现的顺序依次搜索。

## 伪目标

有些时候，一个目标不需要生成一个文件，只需要执行特定操作，例如 `clean` 清除所有已编译文件。为了避免这些目标与某个文件重名，可以显式使用标记 `.PHONY` 进行声明，即：

``` make
.PHONY : clean
	clean : rm *.o temp
```

## 静态模式

静态模式用于批量指定构建的依赖，其语法为：

```make
<targets> ...: <target-pattern>: <prereq-patterns> ...
	recipe
```

其作用是，根据 `target-pattern` 从 `targets` 中匹配目标文件，并根据规则 `preerq-patterns` 为每个匹配的文件生成其依赖。例如下面的代码就实现了为每个 `.o` 文件添加同名 `.c` 依赖，并编译相应的目标文件。

```make
objects := src1.o src2.o

all: myprogram

myprogram: $(objects)
    cc -o myprogram $(objects)

$(objects): %.o: %.c
    cc -c $< -o $@
```

## 自动生成依赖

许多编译器支持自动检测文件的依赖，例如 `gcc -MM main.c` 就会输出 `main.c` 的依赖文件。GNU 组织建议将每一个源文件的依赖存放在同名 `.d` 文件中，可以使用如下的规则来生成对应 `.d` 文件：

```make
%.d: %.c 
	@set -e; rm -f $@; \
	$(CC) -M $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
```

# 书写命令

## 命令显示

一般而言，make 会把正在执行的命令打印到前台，但在命令前添加 `@` 可禁止这一行为。此外，编译时添加参数 `-s` 可全局禁止命令显示。

## 命令执行

默认情况下，每条命令都使用全新 `shell` 执行，如果要在前一条的基础上执行，可在同一行使用分号 `;` 或者 `&&` 连接两条命令。

## 错误忽略

为了忽略错误，可在命令前加一个减号 `-`.