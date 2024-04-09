---
title: CS144 Lab 实验笔记
tags:
  - 计算机网络
  - CS144
  - TCP
  - Cpp
date: 2023-03-30T19:33:00+08:00
lastmod: 2024-04-09T20:58:00+08:00
publish: true
dir: 技术笔记
---

# 资源存档

本次实验使用的课程代码版本为 CS144 Winter 2024，鉴于 CS144 官方要求禁止公开代码以防止抄袭，我将我的题解和原始代码存档放在了 Gitee 上（外国学生应该不知道这个平台吧），有需要可自取：[CS144: CSS144 Winter 2024 Labs.](https://gitee.com/littleherozzzx/CS144)。另外，我还托管了课程主页的镜像，各个资源链接如下：

| 名称         | 链接                                                                                                             | 备注                                                             |
| ---------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 原始代码和题解    | [CS144: CSS144 Winter 2024 Labs.](https://gitee.com/littleherozzzx/CS144)                                      | 原始代码在 archive 分支，题解在 main 分支                                   |
| 课程主页镜像     | [CS 144: Introduction to Computer Networking](https://littleherozzzx.github.io/cs144Winter2024.github.io/)                             |                                                                |
| 虚拟机镜像和配置过程 | [Setting up your CS144 VM using VirtualBox](https://web.stanford.edu/class/cs144/vm_howto/vm-howto-image.html) | 百度云链接：<https://pan.baidu.com/s/1s7xWKn5ccph64--rdJOz6g?pwd=ozb0> |

## 虚拟机镜像

CS144 官网给出了 Virtual Box 镜像及相应配置过程：[Setting up your CS144 VM using VirtualBox](https://web.stanford.edu/class/cs144/vm_howto/vm-howto-image.html)。

# Lab 0



## 环境配置

我使用的是 `Ubuntu 22.04 @ WSL2`，原文档给出了一个环境配置命令：

``` sh
sudo apt update && sudo apt install git cmake gdb build-essential clang clang-tidy clang-format gcc-doc pkg-config glibc-doc tcpdump tshark
```

文档中提到测试环境是 `Ubuntu 23.10 LTS`+`g++ 13.2`，而上述命令并不能安装对应版本的 gcc，可以参考这篇文章安装最新的 `g++`：[安装并切换指定gcc或者g++版本](./%E5%AE%89%E8%A3%85%E5%B9%B6%E5%88%87%E6%8D%A2%E6%8C%87%E5%AE%9Agcc%E6%88%96%E8%80%85g++%E7%89%88%E6%9C%AC.md)，在 Ubuntu 22 上最新只能安装 13.1 版本的 g++。后续实验均在此基础上进行。

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

# Lab 1 

## Putting substrings in sequence

这个模块要求实现一个 TCP 包重组模块，我感觉就是实现计网中 GBN 算法中的接受窗口，缓存收到的处于接收窗口内的 TCP 包、对其按序重组，并及时写入 Lab 0 中实现的可靠内存字节流中。做下来发现这个任务有以下几个要求：

- 实现包重组，包括乱序、重复、过期、截断等
- 该模块缓冲区不得大于内存字节流中的可用缓冲区大小
	- 如果包过长，则截断保存

每个包到达时，有三个字段标识数据内容 `data`、包序号 `first_index` 和是否为最后一个包 `is_last_substring`，对于乱序到达的数据报，我们要暂存这些信息，我使用如下一个结构体保存每一个数据报：

```C++
struct reassembler_item{
  std::string data;
  uint64_t first_index;
  uint64_t last_index; // 左闭右开
  bool is_last;

  bool operator < (const reassembler_item& x) const{
    return first_index < x.first_index;
  }

  reassembler_item(std::string data1, uint64_t first_index1, uint64_t last_index1, bool is_last1)
    : data(std::move(data1)),
    first_index(first_index1),
    last_index(last_index1),
    is_last(is_last1) {}
};
```

为了方便比较，我引入了一个字段用于表示这个包的数据表示的序号范围，采用左闭右开区间是因为存在一些空串（用来标识数据已经发送结束），其右闭区间为 -1，对于无符号数下溢了。

使用 `vector` 暂存收到的乱序数据报，并维护保证其始终有序且不存在重复元素。具体来说，在每次插入数据报时，使用 `std::lower_bound` 二分查找其待插入位置。找到插入位置后，待插入数据报可能向后覆盖了好几个已收到的数据报（例如，新收到的数据范围为 100~200，但是 110~120、190~210 范围的数据报在此之间已经收到并且保存在本模块缓冲区中），因此检查待插入位置后面可能被覆盖的元素，被待插入数据报完全覆盖的数据报直接扔掉，不完全覆盖的数据报则先拼接到待插入的数据报中，然后再扔掉。同样地，待插入数据报也有可能被待插入位置前的数据报覆盖，如果被完全覆盖了，则直接扔掉待插入数据报；如果被不完全覆盖，则拼接到前一个数据报后再扔掉。只有没被覆盖的数据报才需要被单独插入到模块内部暂存区中。  
注意，上文所说的覆盖包含无重叠但相邻的情况，即 [1,200) 和 [200,300) 这两个数据包也是可以合并的。这可以保证如果有字符串可以向内存缓冲区写入，则这个字符串一定是且仅是暂存区的第一个数据包。  

只有当暂存区新插入数据包时，才需要检查暂存区数据能否写入内存缓冲区。暂存区 `insert` 方法实现如下：

```C++
void Reassembler::insert( uint64_t first_index, string data, bool is_last_substring )
{
  // Your code here.
  uint64_t capacity = output_.writer().available_capacity();
  // 可以接受的序号范围为[current_index, current_index+capacity)  左闭右开
  // data中数据的序号范围为[first_index, first_index+data.size())
  // 二者取交集，若为空说明该串过期或者太早到来
  uint64_t left_bound = max(first_index, current_index_);
  uint64_t right_bound = min(current_index_+capacity, first_index+data.size());
  if(right_bound < left_bound) { // 相等为空串，也能接受（可能标志了last_string）
    return; // 对于buffer_没有更新操作，后续不会向缓冲区写入
  }

  reassembler_item item = reassembler_item(
    data.substr( left_bound-first_index, right_bound-left_bound),
    left_bound, right_bound, is_last_substring && right_bound == first_index+data.size());
  pending_size_ += item.data.size(); // 先全部加进去，后面根据覆盖的内容再移除
  auto insert_iter = lower_bound(buffer_.begin(), buffer_.end(), item);
  // 先判断item是否向后覆盖了其它已插入buffer_的数据,如果有则合并
  auto iter = insert_iter;
  while (iter != buffer_.end() && item.last_index >= iter->first_index ){
    if(item.last_index < iter->last_index) { // 只有部分覆盖才要合并，全覆盖直接erase即可
      item.data += iter->data.substr(item.last_index-iter->first_index);
      // 覆盖长度为item_last-iter_first
      pending_size_ -= item.last_index - iter->first_index;
      item.last_index = iter->last_index;
      item.is_last |= iter->is_last;
    }
    else {
      pending_size_ -= iter->data.size();
    }
    iter = buffer_.erase(iter);
  }
  // 再判断前一个数据是否覆盖了item
  // 被前一个覆盖直接在前一个元素中修改，而不需要再插入item了
  if(insert_iter != buffer_.begin()){
    iter = insert_iter - 1;
    if(iter->last_index >= item.first_index){
      if(iter->last_index < item.last_index){ // 非完全覆盖
        iter->data += item.data.substr(iter->last_index-item.first_index);
        pending_size_ -= iter->last_index - item.first_index;
        iter->last_index = item.last_index;
        iter->is_last |= item.is_last;
      } else { // 完全覆盖
        pending_size_ -= item.data.size();
      }
      // 没插入，不需要删除的代码
      // 直接return，不要运行后面插入insert代码
      return;
    }
  }
  // insert item into buffer_
  buffer_.insert(insert_iter, item);
  // 只有插入了新的item，才有可能需要向缓冲区写入
  if(buffer_[0].first_index == current_index_){
    auto& to_write_item = buffer_[0];
    output_.writer().push(to_write_item.data);
    pending_size_ -= to_write_item.data.size();
    current_index_ = to_write_item.last_index;
    if(to_write_item.is_last){
      output_.writer().close();
    }
    buffer_.erase(buffer_.begin());
  }


}
```