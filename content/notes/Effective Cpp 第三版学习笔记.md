---
title: Effective Cpp 第三版学习笔记
tags:
  - Cpp
date: 2024-04-17T18:23:00+08:00
lastmod: 2024-04-19T15:08:00+08:00
publish: true
dir: notes
slug: notes on effective cpp 3rd ed
---

# 前言

本文是我在学习 Scott Meyers 的著作《Effective C++》第三版的笔记，鉴于豆瓣对于本书中文翻译褒贬不一，我直接看的英文原著。PDF 链接：[Books-3/Effective C++ 3rd ed.pdf at master · GunterMueller/Books-3 · GitHub](https://github.com/GunterMueller/Books-3/blob/master/Effective%20C%2B%2B%203rd%20ed.pdf)

> For Nancy, without whom nothing would be much worth doing.  
> Wisdom and beauty form a very rare combination.

作者在扉页留下了这句话，打破了我内心对程序员的一点点小刻板印象，我本以为这类话只会在文青写的书里出现呢。

# 正文

## Introduction

### lhs & rhs

> Two of my favorite parameter names, for example, are lhs and rhs. They stand for “left-hand side” and “right- hand side,” respectively.

lhs 和 rhs 是作者常用的二元运算符的形参命名，分别表示运算符左边和右边的参数。

## Accustoming yourself to C++

### Item 1: View C++ as a federation of languages.

> ✦ Rules for effective C++ programming vary, depending on the part of C++ you are using.

C++ 诞生于 C，但很快成长为一门“联合”的语言。它支持多种编程范式，包括面向过程、面向对象、泛型、函数式和元编程。

GPT 对以上几种编程范式的介绍：

> 1. 面向过程编程 (Procedural Programming)  
> 面向过程编程是一种编程范式，它侧重于编写一系列计算步骤或过程的方法。这种范式通过函数或过程来操作数据，强调的是程序的动作而非数据的表现形式。这种方式适合于简单的任务，能够让程序员更直接地控制程序流和状态。
> 
> 优点：简单直接，容易理解和实现。  
> 缺点：随着程序规模的扩大，维护和扩展变得困难，难以管理大量的数据和函数。
> 
> 2. 面向对象编程 (Object-Oriented Programming, OOP)  
> 面向对象编程是基于“对象”的概念，其中对象是数据和操作这些数据的函数（称为方法）的集合。OOP 强调将现实世界的实体模型化为程序中的对象，每个对象都有其属性和行为。
> 
> 优点：提高了代码的复用性、封装性和可维护性。  
> 缺点：可能会导致程序结构和实现过于复杂，性能上有时也不如面向过程编程。
> 
> 3. 泛型编程 (Generic Programming)  
> 泛型编程是一种编程风格，通过抽象和重用来增强代码的灵活性和可重用性。它允许程序员编写与数据类型无关的代码。在泛型编程中，数据类型作为参数传递给算法或数据结构。
> 
> 优点：代码复用性高，类型安全。  
> 缺点：可能增加编程的复杂性，理解和设计泛型结构需要更深的抽象思维。
> 
> 4. 函数式编程 (Functional Programming, FP)  
> 函数式编程是一种把计算视为数学函数的求值，并避免使用程序状态及可变数据的编程范式。FP 强调无副作用的函数，支持高阶函数和函数组合，以及懒计算等特性。
> 
> 优点：代码更简洁，更容易推理，便于并行计算。  
> 缺点：与传统的命令式编程思维差异大，学习曲线较陡峭。
> 
> 5. 元编程 (Metaprogramming)  
> 元编程是一种编程技术，允许程序在运行时或编译时改变其自身结构。这通常涉及到代码生成代码的情况。
> 
> 优点：极大的灵活性和动态性。  
> 缺点：可能导致代码难以理解、维护和调试。
> 
> 6. 模板元编程 (Template Metaprogramming)  
> 模板元编程是一种在编译时执行计算的技术，常见于 C++ 的模板机制中。它通过模板实例化时的类型推导来执行算法，从而在编译时生成高度优化的代码。
> 
> 优点：生成高度优化的代码，无运行时开销。

**可以认为 C++ 由四门语言构成，每一门子语言本身都很简单**：
- C。C++ 兼容 C 的语法，因此使用 C 语言完成的大多数任务都可以用 C++ 来完成，但得益于另外三个 C 不具备的子语言，C++ 可以完成得更得心应手。
-  Object-Oriented C++。这指的就是我们熟悉的“C with class”，即在 C++ 中引入的面向对象模块。
- Template C++。这指的是 C++ 中泛型编程的部分，这还催生了一种全新的编程范式：模板元编程。
- STL。STL 对于容器、迭代器、算法和函数对象的实现有其自洽的一套逻辑，如果我们要使用 STL 的内容，那也要遵循这套逻辑。

不同子语言之间可能有不同的行为准则，例如 C 的内建类型按值传递相比引用传递更高效，但对于对象而言恰恰相反；又例如 STL 的迭代器行为类似于 C 中的指针，这种情况下又要使用值传递。

### Item 2: Prefer consts, enums, and inlines to defines.

> ✦ For simple constants, prefer const objects or enums to `#defines`.  
> ✦ For function-like macros, prefer inline functions to `#defines`.

这一条可以简写为：尽量让编译器去处理而非在预处理阶段替换。

一个理由是，对于编译器而言，其可能无法得知在预处理阶段被替换的常量符号，因而这些符号不会出现在符号表中。如果这些常量导致了出错或者警告，在错误信息中提示的就是常量的值而非代码中给定的常量名，这降低了错误信息的可读性。

第二个理由是，`const` 关键字定义的常量可以控制作用域，而 `#define` 关键字则不可以。

关于把 `#define` 替换为常量，有几点需要注意：

- 如果需要定义一个指向常量的指针，大部分情况这个这个指针本身也是不可更改指向的，即指向常量的常量指针，需要两个 `const` 关键字，即：`const char* const name = "Name"`。
- 对于类成员是常量的情况，还要声明为静态变量以防止在内存中存在多个常量拷贝，即：

```C++ {hl_lines=["3"]}
class GamePlayer{
private:
	static const  int NumTurns = 5;
	int scores[NumTurns]; 
	...
};
```

部分很老的编译器可能不允许在类声明中定义静态变量的值，更加通用的做法是在类实现的文件中给出静态成员的值。但有例外：即编译器在编译这个类时就需要知道其静态变量的值，例如上述代码中，编译器需要知道 `scores` 数组的长度，因此要么在声明时就给出静态变量的值，要么使用曲线救国的方案：

``` C++
class GamePlayer{ 
	private: 
		enum { NumTurns = 5 };
		int scores[NumTurns];
		...
};
```

上述方案被称为“the enum hack”，了解它的价值在于：

- the enum hack 相比 `const` 更像传统的 `#define`，其不能取地址。
- 出于实践的考虑：确实有很多代码使用了这个技巧

另一个“尽量让编译器去处理而非在预处理阶段替换”的理由是：人们使用宏在不需要函数调用开销的情况下实现类似函数的功能，然而这种宏函数无法执行类型检查并且每个变量都要用括号扩起来。C++ 提供了 `inline` 关键字用于实现类似的效果，inline 函数会在原地展开，免去了函数调用的开销；同时，其又支持像常规函数一样的语法和类型检查。

### Item 3: Use const whenever possible.

> ✦ Declaring something const helps compilers detect usage errors. const can be applied to objects at any scope, to function parameters and return types, and to member functions as a whole.  
> ✦ Compilers enforce bitwise constness, but you should program using logical constness.  
> ✦ When const and non-const member functions have essentially identi- cal implementations, code duplication can be avoided by having the non-const version call the const version.

尽可能使用 `const` 关键字，它可以让编译器帮助防止变量被调用者或者其他代码修改。

当 `const` 关键字和指针相遇，有多种情况：

```C++
char greeting[] = "Hello";
char *p = greeting; // non-const pointer, non-const data
const char *p = greeting; // non-const pointer, const data
char * const p = greeting; // const pointer, non-const data
const char * const p = greeting; // const pointer, const data
```

上述规则可以总结为：如果 `const` 出现在 `*` 的左边，那么指向的数据本身是不可变的；如果 `const` 出现在 `*` 的右边，那么指针是不可变的。

对于 `const` 在 `*` 的左边的情况，其相对类型的位置又有两种情况，二者是完全等价的，即：

```C++
const int a;
int const b;
// a和b均表示一个不可修改的int
```

STL 中的迭代器如果被声明为 `const`，那么说明这个迭代器本身是不可修改的，而非这个迭代器指向了不可修改的数据。如果需要一个指向不可修改数据的迭代器，需要使用 `const_iterator` 类型。

在函数声明中，`const` 关键字可以用来修饰返回值类型、参数类型和整个函数（仅限成员函数）。

通常而言，没有理由将返回值声明为 `const`，但有的时候这么做也可能减少调用者的错误。例如，假设实现了一个实数类 `Rational` 并重载了其 `operator *` 以实现乘法，如果不将返回值声明为 `const`，那么下列代码就是符合语法但无意义的：

```C++
Rational a, b, c;
(a*b) = c; // 将c的值赋给临时变量(a*b)
if(a*b = c); // 漏打了一个等号
```

将一个成员函数声明为 `const` 有助于提高编码效率，一方面它可以帮助调用者区分哪些方法会修改对象哪些不会，另一方面，在使用 const 引用传参的情况下，只能调用该对象的 const 方法。此外，除了声明为 `const` 之外其他签名均相同的两个成员函数在 C++ 中也被视为重载。

对于 `const` 有两种哲学理念：

- bitwise constness：const 成员函数不得修改对象内的任何数据，这是一种比较严格其方便编译器实现的理念，也是 C++ 所采用的。
- logic constness: const 成员函数允许以客户无法感知的形式修改对象内的数据，例如私有变量。

logic constness 的存在也是合理的。例如，如果我们想实现一个 `String` 类及其 `size()` 方法，我们使用一个私有变量 `length` 缓存其长度，那么将 `size()` 声明为 `const` 显然是合理的（否则 `const String` 将无法获取长度），但在实现 `size()` 的过程中，第一次访问 `size()` 不可避免要修改 `length` 值，这违反了 bitwise constness 理念，但又是符合程序员直觉的一个需求。这种情况下，我们可以使用 `mutable` 来修饰变量，这样就可以在 const 成员函数中修改他们。

前面提到，const 可以用来重载成员函数，那我们可能会有如下两个重载函数的声明：

``` C++
class Vector {
...
	const Element& operator [](size_t index) const{
	...// 越界检查、身份校验等
	return data[i];
	}
	
	Element& operator [](size_t index){
	...// 越界检查、身份校验等
	return data[i];
	}
}
```

不难发现，const 和非 const 版本的两个 `[]` 下边访问方法的实现完全相同，但为了让 const 对象可以获取可修改的数据引用和非 const 对象获取不可修改的引用，我们不得不重复两次。

为了减少这种无意义的重复，我们可以在非 const 方法中调用 const 方法，并使用 `const_cast` 关键字将其转换为非 const 对象。即：

``` C++
class Vector {
...
	const Element& operator [](size_t index) const{
	...// 越界检查、身份校验等
	return data[i];
	}
	
	Element& operator [](size_t index){
	return const_cast<Element&>( // 将const element& 转换为 element&
		static_cast<const Vector&>(*this)[index]) // 将this转换为const对象，以调用const方法
	}
}
```

### Item 4: Make sure that objects are initialized before they’re used.

> ✦ Manually initialize objects of built-in type, because C++ only some- times initializes them itself.  
> ✦ In a constructor, prefer use of the member initialization list to as- signment inside the body of the constructor. List data members in the initialization list in the same order they’re declared in the class.  
> ✦ Avoid initialization order problems across translation units by re- placing non-local static objects with local static objects.

在 C++ 中，当你定义一个变量时，有一套复杂的规则来决定编译器是否会为你进行默认初始化。然而，试图读取一个未被初始化的变量是一个未定义行为，可能导致程序崩溃或者复杂的 debug。最好的方案是每次在定义时就进行初始化。

对于非成员的内建变量类型，需要手动进行初始化：

```C++
int x=0;
const char* text = "Hello World!";
double d;
cin >> d;
```

除此以外几乎所有的情况，初始化的任务由构造函数完成。规则很简单：每一个成员变量都要在构造函数中被初始化。

注意区分构造函数中的初始化和赋值：

```C++ {hl_lines=["12-15"]}
class PhoneNumber { ... };
class ABEntry { // ABEntry = “Address Book Entry” 
public: 
	ABEntry(const std::string& name, const std::string& address, const std::list<PhoneNumber>& phones); 
private:
	std::string theName; 
	std::string theAddress; 
	std::list<PhoneNumber> thePhones; 
	int numTimesConsulted; 
};
ABEntry::ABEntry(const std::string& name, const std::string& address, const std::list<PhoneNumber>& phones){ 
	theName = name;
	theAddress = address;
	thePhones = phones;
	numTimesConsulted = 0;
	// 以上都是赋值而非初始化
}
```

C++ 的类成员的初始化必须在构造函数主体前的初始化列表中完成。上述的赋值方法会先调用各个类的默认构造函数进行隐式初始化，然后再调用拷贝构造函数，使用初始化列表则可以直接调用拷贝构造函数进行初始化，省去了默认构造的时间。此外，内建类型的变量并不会进行默认初始化，必须在初始化列表或者构造函数主体中显式初始化。

类初始化的顺序为：基类先于派生类，类成员按照声明的顺序进行初始化。即便在初始化列表中指定了其它顺序，类内成员仍将按照声明的顺序进行初始化。

接下来讨论静态对象的初始化问题，静态对象包括：全局对象、命名空间中定义的对象、类/函数/文件内被声明为静态的对象。其中，函数内的静态对象被称为局部静态对象，其余被称为非局部静态对象。所有的静态对象在程序结束运行时销毁。

一个**翻译单元**指的是生成一个目标文件的源码，即单个源文件加上其包含的所有头文件。

接下来作者举了一个例子，可以抽象为：一个翻译单元 A 的非静态局部对象的初始化过程引用了来自另一个翻译单元 B 的非局部静态对象，但是编译器并不能保证当 A 初始化时 B 中的非局部静态变量已经初始化。为了解决这个问题，我们可以引入设计模式中的单例模式，在 B 中定义一个全局函数或者在类定义中定义一个成员函数，用于初始化一个局部静态对象并返回其引用。

但是，上述解决方案并不适用于多线程环境：同个线程可能同时初始化一个局部静态对象。可以通过在线程启动前手动调用每个返回局部静态对象的函数以完成初始化。  

# 参考文档