---
名称: CS144 Lab 实验笔记
tags:
  - 计算机网络
  - CS144
  - TCP
  - Cpp
date: 2023-03-30T19:33:00+08:00
lastmod: 2024-04-02T12:23:00+08:00
publish: true
images: "![](https://pics-zhouxin.oss-cn-hangzhou.aliyuncs.com/Lab0%20%E5%8F%AF%E9%9D%A0%E5%AD%97%E8%8A%82%E6%B5%81.png)"
dir: 学习笔记
---

# 资源存档

本次实验使用的课程代码版本为 CS144 Winter 2024，鉴于 CS144 官方要求禁止公开代码以防止抄袭，我将我的题解和原始代码存档放在了 Gitee 上（外国学生应该不知道这个平台吧），有需要可自取：[CS144: CSS144 Winter 2024 Labs.](https://gitee.com/littleherozzzx/CS144)。另外，我还托管了课程主页的镜像，各个资源链接如下：

| 名称         | 链接                                                                                                             | 备注                                                             |
| ---------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 原始代码和题解    | [CS144: CSS144 Winter 2024 Labs.](https://gitee.com/littleherozzzx/CS144)                                      | 原始代码在 archive 分支，题解在 main 分支                                   |
| 课程主页镜像     | [CS 144: Introduction to Computer Networking](https://littleherozzzx.github.io/cs144Winter2024.github.io/)                             |                                                                |
| 虚拟机镜像和配置过程 | [Setting up your CS144 VM using VirtualBox](https://web.stanford.edu/class/cs144/vm_howto/vm-howto-image.html) | 百度云链接：https://pan.baidu.com/s/1s7xWKn5ccph64--rdJOz6g?pwd=ozb0 |

## 虚拟机镜像

在 WSL2 上对 Lab 0 进行测试时，发现 `make check webget` 报错 `make[3]: ../tun.sh: Command not found`，这个 `tun.sh` 应该是构建过程中自动生成的文件，怀疑是 wsl 兼容性问题。CS144 官网给出了 Virtual Box 镜像及相应配置过程：[Setting up your CS144 VM using VirtualBox](https://web.stanford.edu/class/cs144/vm_howto/vm-howto-image.html)。

# Lab 0



## 环境配置

我使用的是 `Ubuntu 22.04 @ WSL2`，原文档给出了一个环境配置命令：

``` sh
sudo apt update && sudo apt install git cmake gdb build-essential clang clang-tidy clang-format gcc-doc pkg-config glibc-doc tcpdump tshark
```

文档中提到测试环境是 `Ubuntu 23.10 LTS`+`g++ 13.2`，而上述命令并不能安装对应版本的 gcc，可以参考这篇文章安装最新的 `g++`：[安装并切换指定gcc或者g++版本](../posts/%E5%AE%89%E8%A3%85%E5%B9%B6%E5%88%87%E6%8D%A2%E6%8C%87%E5%AE%9Agcc%E6%88%96%E8%80%85g++%E7%89%88%E6%9C%AC.md)，在 Ubuntu 22 上最新只能安装 13.1 版本的 g++。后续实验均在此基础上进行。

## 现代 C++

实验要求使用现代 C++ 风格进行编程，基本理念是：每个对象都只设计尽可能少的公共接口、内部存在各种安全检查、使用结束后应该正确回收垃圾，避免使用成对的关键字（例如 `new` 和 `delete`）。相反，通过构造函数和析构函数来获取和释放资源，即基于“资源获取即初始化”RAII 理念。

具体来说，对于编码风格有以下要求：
- 在编码过程中参考文档 [cppreference.com](https://en.cppreference.com/w/)
- 不要使用 `malloc`、`free`、`new` 或者 `delete` 关键字
- 不要使用原始指针，使用智能指针
- 不要使用模板、线程、锁或者虚函数
- 不要使用 `C` 风格字符串 `char*` 或者相关函数 `strlen()` 等
- 不要使用 `C` 风格类型转换，使用 `C++` 的 `static_cast` 进行转换
- 函数形参尽可能使用 `const` 关键字
- 变量和函数都尽可能使用 `const` 关键字修饰
- 避免使用全局变量，每个变量的作用域都应该尽可能小
- 在提交前，使用 `cmake --build build --target tidy` 获取关于代码风格修改的建议，使用 ` cmake --build build --target format` 对代码进行格式化。

## Writing webget

忽略前面通过 `telnet` 刚问网页和发送邮件的内容，第一个编码任务是完成 `Webget`，使之能够获取网页。这个任务比较简单，涉及到一点网络编程的知识。
整个任务的流程是：根据形参获取初始化主机地址，建立与该主机的 TCP 连接，发送 HTTP 请求报文（包含形参中的资源路径），打印响应报文，关闭 TCP 连接。
实现的代码为：

```C++
void get_URL( const string& host, const string& path )
{
  Address addr = Address(host, "http");
  TCPSocket sock = TCPSocket();
  sock.connect(addr);
  string message = "GET " + path +" HTTP/1.1\r\n" + "Host: "+host + "\r\n" +"Connection: close\r\n\r\n";
  sock.write(message);
  while(!sock.eof()){
    string response;
    sock.read(response);
    cout << response;
  }
  sock.close();
  cerr << "Function called: get_URL(" << host << ", " << path << ")\n";
  cerr << "Warning: get_URL() has not been implemented yet.\n";
}
```

## An in-memory reliable byte stream

第二个任务是实现可靠的内存字节流，有以下几个要求：
- 输出端和输入端数据顺序一致，以 EOF 结尾
- 流量控制，即该字节流存在一个容量上限
- 容量上限指的是字节流中存在的数据的上限，而非发送者发送的字节流的上限。显然，我在实现时直接截断了超过剩余容量的输入
- 单线程使用，不需要考虑并发读写

任务要求实现如下接口：

``` C++
class Writer : public ByteStream
{
public:
  void push( std::string data ); // Push data to stream, but only as much as available capacity allows.
  void close();                  // Signal that the stream has reached its ending. Nothing more will be written.

  bool is_closed() const;              // Has the stream been closed?
  uint64_t available_capacity() const; // How many bytes can be pushed to the stream right now?
  uint64_t bytes_pushed() const;       // Total number of bytes cumulatively pushed to the stream
};

class Reader : public ByteStream
{
public:
  std::string_view peek() const; // Peek at the next bytes in the buffer
  void pop( uint64_t len );      // Remove `len` bytes from the buffer

  bool is_finished() const;        // Is the stream finished (closed and fully popped)?
  uint64_t bytes_buffered() const; // Number of bytes currently buffered (pushed and not popped)
  uint64_t bytes_popped() const;   // Total number of bytes cumulatively popped from stream
};
```

为了记录累计读写量、维护剩余容量和端口是否关闭，在 `ByteStream` 添加了如下成员变量（别忘了在构造函数中初始化）：

``` C++
  std::queue<char> buffer_; // 缓冲区
  uint64_t amount_; // 剩余容量
  uint64_t total_pushed_; // 总写入量
  uint64_t total_poped_; // 总读取量
  bool close_; // 端口状态
```

具体实现比较简单，维护一个双端队列进行读写操作，这里就不放代码了。值得一提的是在 `Writer::push` 的实现中，如果待写入数据超过了缓冲区剩余容量，则直接截断即可。

最终吞吐量为 0.63 Gbit/s，处于能接受的水平。
![Lab0 可靠字节流](https://pics-zhouxin.oss-cn-hangzhou.aliyuncs.com/Lab0%20%E5%8F%AF%E9%9D%A0%E5%AD%97%E8%8A%82%E6%B5%81.png)