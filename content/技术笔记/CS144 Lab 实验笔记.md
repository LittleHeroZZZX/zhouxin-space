---
title: CS144 Lab 实验笔记
tags:
  - 计算机网络
  - CS144
  - TCP
  - Cpp
date: 2023-03-30T19:33:00+08:00
lastmod: 2024-04-11T12:46:00+08:00
publish: true
dir: 技术笔记
math: "true"
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

文档中提到测试环境是 `Ubuntu 23.10 LTS`+`g++ 13.2`，而上述命令并不能安装对应版本的 gcc，可以参考这篇文章安装最新的 `g++`：[安装并切换指定gcc或者g++版本](../../Ubuntu/%E5%AE%89%E8%A3%85%E5%B9%B6%E5%88%87%E6%8D%A2%E6%8C%87%E5%AE%9Agcc%E6%88%96%E8%80%85g++%E7%89%88%E6%9C%AC.md)，在 Ubuntu 22 上最新只能安装 13.1 版本的 g++。后续实验均在此基础上进行。

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

具体实现比较简单，维护一个队列 `vector<string>` 进行读写操作。在 `Writer::push` 的实现中，如果待写入数据超过了缓冲区剩余容量，则直接截断即可。指的注意的是 pop 采用了一种“lazy pop”的机制，即每次 pop 一个字节时，不要直接删除队头字符串的第一个字符，而是使用一个变量记录对头字符串还剩多少字节没有被 pop。

`byte_stream.cc` 的实现如下：

```C++
#include "byte_stream.hh"
#include "iostream"
using namespace std;

ByteStream::ByteStream( uint64_t capacity ) :
  capacity_( capacity ), buffer_(), amount_(0), total_pushed_(0),
  total_poped_(0), first_string_left_size(0), close_( false ), error_( false )  {}

bool Writer::is_closed() const
{
  // Your code here.
  return close_;
}

void Writer::push( string data )
{
  // Your code here.
  uint64_t free_capacity = available_capacity();
  uint64_t to_push_size = min(free_capacity, data.size());
  if(to_push_size == 0)  return;
  data.resize(to_push_size);
  buffer_.emplace(std::move(data));
  if(buffer_.size() == 1)
    first_string_left_size = to_push_size;
  total_pushed_ += to_push_size;
  amount_ += to_push_size;
  return;
}

void Writer::close()
{
  // Your code here.
    close_ = true;
}

uint64_t Writer::available_capacity() const
{
  // Your code here.
  return capacity_ - amount_;
}

uint64_t Writer::bytes_pushed() const
{
  // Your code here.
  return total_pushed_;
}

bool Reader::is_finished() const
{
  // Your code here.
  return amount_ == 0 && close_;
}

uint64_t Reader::bytes_popped() const
{
  // Your code here.
  return total_poped_;
}

string_view Reader::peek() const
{
  // Your code here.
  if(amount_ == 0 || buffer_.empty()){
    return string_view{};
  }
  const string& front = buffer_.front();
//  return string_view(front.data()+front.size()-first_string_left_size,1);
//  return string_view(&front[front.size()-first_string_left_size]);
  return string_view(front).substr(front.size()-first_string_left_size);
}


void Reader::pop( uint64_t len )
{
  // Your code here.
  total_poped_ += len;
  amount_ -= len;
  while(len){
    if(len >= first_string_left_size){
      len -= first_string_left_size;
      buffer_.pop();
      first_string_left_size = buffer_.front().size();
    } else{
      first_string_left_size -= len;
      len = 0;
    }
  }
}

uint64_t Reader::bytes_buffered() const
{
  // Your code here.
  return amount_;
}

```

最终吞吐量最高跑到了 34 Gbit/s。  
![Lab0 实验结果](http://pics.zhouxin.space/20240410101229.png)

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

最终重组模块吞吐量最高跑到了 10 Gbit/s。  
![Lab1 实验结果](http://pics.zhouxin.space/20240410101326.png)

# Lab 2

到此为止，我们已经完成了内存可靠字节流和 TCP 包重组模块，重组模块将收到的 TCP 包进行重组，并及时写入内存字节流。接下来，我们需要写一个 TCP 接收器模块，接收来自 peer 发送方的消息，并回复 ACK 和接收窗口大小。

在此之前，有一个数据格式问题：在前两个模块中，我们使用 `uint64` 来标记序列号，可是在 TCP 的数据包只有 32 位用于记录序号，并且初始包（SYN）的序号可能是随机的。因此，我们首先要实现一个 32 位 TCP 包序号和 64 位绝对序号互相转换的模块。前者开始序号随机，并不断自增取余；后者固定从 0 开始自增，且我们认为总数据量不可能超过 2^64Byte，即 2^34GB。

## Translating between 64-bit indexes and 32-bit seqnos

![TCP包序号、绝对序号和流序号之间的对应关系](http://pics.zhouxin.space/20240411101104.png)  
根据上图定义，不难发现 seqno 和 abs seqno 存在如下对应关系：

$$
seqno = (absSeqno+zeroPoint) \% 2^{32}
$$

从 64 位转 32 位根据上式转换即可，其中对 2^32 取余是不必要的，因为 32 位数自动截断高 32 位。

从 32 位向 64 位转换，我们需要分开考虑其高低 32 位。首先是低 32 位，低 32 位标识了这个包的是整个序列的第 $absSeq\%2^{32}$ 个包。那怎么通过 $seqno$ 计算它是整个序列的第几个包呢？$seqno$ 在自增过程中会不断取余，若不取余，记其为 $seqno'$，那么这个包是整个序列的第 $seqno'-zeroPoint$ 个包，而 $seqno'=seqno+n\times 2^{32}$，即：

$$
\begin{aligned}
absSeq \% 2^{32} &= (seqno'-zeroPoint)\%2^{32} \\
&= (seqno+n\times 2^{32} - zeroPoint)\%2^{32} \\ 
&= (seqno-zeroPoint + 2^{32}) \% 2^{32}
\end{aligned}
$$

上式即为计算绝对序号低 32 位的方法。得到低 32 位后，就要根据 checkPoint 得到高 32 位。显然，为了接近 checkPoint，高 32 位也是越接近越好，因此高 32 位可以为 checkPoint 的高 32 位或者在此基础上±1，然后比较这三个方案哪个更接近 checkPoint 即可。

`wrapping_integers.cc` 实现为：

```C++
#include "wrapping_integers.hh"

using namespace std;

Wrap32 Wrap32::wrap( uint64_t n, Wrap32 zero_point )
{
  // Your code here.
  return Wrap32 { Wrap32(n) + zero_point.raw_value_  };
}

uint64_t Wrap32::unwrap( Wrap32 zero_point, uint64_t checkpoint ) const
// 转换为从0开始的绝对编号
{
  // Your code here.
  // checkpoint = 前32位+left
  // 与checkpoint最近的可能有两个数（分布在checkpoint一左一右） 其中一个必定是 前32位+offset
  // 如果offset < left  那么另一个必定比checkpoint大 等于前32位+zero_point+0x1 0000 0000
  // 那么就要看checkpoint 更接近前32位+zero_point 还是前32位+zero_point+0x1 0000 0000
  // 两边同减去前32位和zero_point 就是看 left-point 更接近0 还是0x 1 0000 0000
  uint64_t offset = (raw_value_+0x1'0000'0000-zero_point.raw_value_)%0x1'0000'0000;
  uint32_t left = checkpoint % 0x1'0000'0000;
  uint64_t high32 = checkpoint - left;
//  if( offset == left) {
//    return high32+checkpoint;
//  } else if ( offset < left){
  if(offset < left){
    if(left- offset <= 0x8000'0000) { // 更接近前32位+zero_point
      return high32+ offset;
    } else {
      return high32+ offset +0x1'0000'0000;
    }
  } else {
  // 同上，offset > left 那么另一个一定比check_point 小 等于前32位+zero_point-0x1 0000 0000
  if( high32 == 0 || offset -left <= 0x8000'0000) { // 更接近前32位+zero_point
    return high32+ offset;
  } else {
    return high32+ offset -0x1'0000'0000;
  }
  }

}

```

## Implementing the TCP receiver

接下来我们就可以实现 TCP receiver 了，实验过程中注意区分五个序号的概念，很容易搞混。另有几个关键逻辑值得一提：

- 如果收到 RST，需要将向内存字节流报告出错（很奇怪为啥 `set_eroor` 方法是 `Reader` 而不是 `Writer` 的）；
- 收到 SYN 后更新 `zero_point` 和 `ack_`；
- 只有收到 SYN 后才能开始接收数据；
- 向包重组器发送数据后，根据内存中写入的数据量可以得到第一个待接收的数据的序号，进而更新 `ack_`；
- 如果数据全部接收完毕，`ack_` 更新时还要额外 +1（FIN 占了一个序号），接收完毕需要根据 `writer.is_closed` 来判断;

`TCP_receiver` 实现如下：

```C++
#include "tcp_receiver.hh"

using namespace std;

void TCPReceiver::receive( TCPSenderMessage message )
{
  // Your code here.
  if(message.RST) {
    reader().set_error();
    return;
  }
  if(message.SYN){
    zero_point_ = Wrap32(message.seqno);
    ack_.emplace(message.seqno);
  }
  if(ack_.has_value()) {
    const uint64_t check_point = writer().bytes_pushed()+1;
    uint64_t first_index
      = Wrap32( message.SYN ? message.seqno + 1 : message.seqno ).unwrap( zero_point_, check_point )-1;
    reassembler_.insert( first_index, std::move(message.payload), message.FIN );
    ack_ = ack_->wrap(writer().bytes_pushed()+1+writer().is_closed() , zero_point_);
  }
}

TCPReceiverMessage TCPReceiver::send() const
{
  // Your code here.
  return {ack_,
           static_cast<uint16_t>(min(reassembler_.writer().available_capacity(), static_cast<uint64_t>(UINT16_MAX))),
           reader().has_error()};
}

```
