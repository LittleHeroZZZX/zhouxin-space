---
title: Effective Cpp 第三版学习笔记
tags:
  - Cpp
date: 2024-04-17T18:23:00+08:00
lastmod: 2024-09-09T23:09:00+08:00
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
> 
> ✦ For function-like macros, prefer inline functions to `#defines`.

这一条可以简写为：尽量让编译器去处理而非在预处理阶段替换。

一个理由是，对于编译器而言，其可能无法得知在预处理阶段被替换的常量符号，因而这些符号不会出现在符号表中。如果这些常量导致了出错或者警告，在错误信息中提示的就是常量的值而非代码中给定的常量名，这降低了错误信息的可读性。

第二个理由是，`const` 关键字定义的常量可以控制作用域，而 `#define` 关键字则不可以。

关于把 `#define` 替换为常量，有几点需要注意：

- 如果需要定义一个指向常量的指针，大部分情况这个这个指针本身也是不可更改指向的，即指向常量的常量指针，需要两个 `const` 关键字，即：`const char* const name = "Name"`。
- 对于类成员是常量的情况，还要声明为静态变量以防止在内存中存在多个常量拷贝，即：

```cpp {hl_lines=["3"]}
class GamePlayer{
private:
	static const  int NumTurns = 5;
	int scores[NumTurns]; 
	...
};
```

部分很老的编译器可能不允许在类声明中定义静态变量的值，更加通用的做法是在类实现的文件中给出静态成员的值。但有例外：即编译器在编译这个类时就需要知道其静态变量的值，例如上述代码中，编译器需要知道 `scores` 数组的长度，因此要么在声明时就给出静态变量的值，要么使用曲线救国的方案：

``` cpp
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
> 
> ✦ Compilers enforce bitwise constness, but you should program using logical constness.
>   
> ✦ When const and non-const member functions have essentially identi- cal implementations, code duplication can be avoided by having the non-const version call the const version.

尽可能使用 `const` 关键字，它可以让编译器帮助防止变量被调用者或者其他代码修改。

当 `const` 关键字和指针相遇，有多种情况：

``` cpp
char greeting[] = "Hello";
char *p = greeting; // non-const pointer, non-const data
const char *p = greeting; // non-const pointer, const data
char * const p = greeting; // const pointer, non-const data
const char * const p = greeting; // const pointer, const data
```

上述规则可以总结为：如果 `const` 出现在 `*` 的左边，那么指向的数据本身是不可变的；如果 `const` 出现在 `*` 的右边，那么指针是不可变的。

对于 `const` 在 `*` 的左边的情况，其相对类型的位置又有两种情况，二者是完全等价的，即：

``` cpp
const int a;
int const b;
// a和b均表示一个不可修改的int
```

STL 中的迭代器如果被声明为 `const`，那么说明这个迭代器本身是不可修改的，而非这个迭代器指向了不可修改的数据。如果需要一个指向不可修改数据的迭代器，需要使用 `const_iterator` 类型。

在函数声明中，`const` 关键字可以用来修饰返回值类型、参数类型和整个函数（仅限成员函数）。

通常而言，没有理由将返回值声明为 `const`，但有的时候这么做也可能减少调用者的错误。例如，假设实现了一个实数类 `Rational` 并重载了其 `operator *` 以实现乘法，如果不将返回值声明为 `const`，那么下列代码就是符合语法但无意义的：

``` cpp
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

``` cpp
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

``` cpp
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
> 
> ✦ In a constructor, prefer use of the member initialization list to as- signment inside the body of the constructor. List data members in the initialization list in the same order they’re declared in the class.  
> 
> ✦ Avoid initialization order problems across translation units by re- placing non-local static objects with local static objects.

在 C++ 中，当你定义一个变量时，有一套复杂的规则来决定编译器是否会为你进行默认初始化。然而，试图读取一个未被初始化的变量是一个未定义行为，可能导致程序崩溃或者复杂的 debug。最好的方案是每次在定义时就进行初始化。

对于非成员的内建变量类型，需要手动进行初始化：

``` cpp
int x=0;
const char* text = "Hello World!";
double d;
cin >> d;
```

除此以外几乎所有的情况，初始化的任务由构造函数完成。规则很简单：每一个成员变量都要在构造函数中被初始化。

注意区分构造函数中的初始化和赋值：

```cpp {hl_lines=["12-15"]}
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

## Constructors, Destructors, and Assignment Operators

### Item 5: Know what functions C++ silently writes and calls. 

> ✦ Compilers may implicitly generate a class’s default constructor, copy constructor, copy assignment operator, and destructor.

默认情况下，编译器在**必要时**会生成 public 且 inline 的默认构造函数、析构函数、拷贝构造函数和拷贝赋值函数。编译器为一个类生成的所有函数都是非虚函数，唯一的例外是一个派生类的基类有一个虚析构函数，那么编译器会为其生成一个虚析构函数。否则，将无法通过基类指针/引用销毁派生对象。

生成拷贝构造函数时，编译器会拷贝所有非静态成员。拷贝赋值函数原理与之类似，但是并非所有对象都可以被拷贝，例如私有对象、const 对象或者引用对象，这种情况下编译器就会拒绝生成拷贝构造函数。

### Item 6: Explicitly disallow the use of compiler- generated functions you do not want.

> ✦ Compilers may implicitly generate a class’s default constructor, copy constructor, copy assignment operator, and destructor.

有些类可能不允许有两个相同的对象，但语法/编译器并没有提供禁用生成拷贝构造和拷贝赋值的关键字。一种可能得实现是，将二者声明为私有的，这可以防止用户调用拷贝构造和赋值；此外，不要实现这两个私有函数，这可以防止友元函数和类成员函数调用。

调用声明但没有定义的函数会在链接期出错，一种将其提前到编译器的办法是，定义一个描述不可拷贝的类 `Uncopyable`，其它类派生于它：

```cpp
class Uncopyable { 
protected: // allow construction and destruction of derived objects... 
	Uncopyable() {}
	~Uncopyable() {}
private: // ...but prevent copying 
	Uncopyable(const Uncopyable&); 
	Uncopyable& operator=(const Uncopyable&); 
};

class UncopyableThing: private Uncopyable{
...
};
```

`UncopyableThing` 中并没有 `Uncopyable` 对象，但上述方法可以其作用是因为：当编译器尝试生成拷贝函数时，其会调用基类拷贝函数（无论有无该对象）。

### Item 7: Declare destructors virtual in polymorphic base classes.

> ✦ Polymorphic base classes should declare virtual destructors. If a class has any virtual functions, it should have a virtual destructor.  
> 
> ✦ Classes not designed to be base classes or not designed to be used polymorphically should not declare virtual destructors.

如果我们使用基类指针释放派生对象，并且基类没有虚析构函数，那么会造成 partially destroyed 问题，即派生对象的基类被释放，而其派生部分内存泄漏。

解决这个问题很简单，将基类析构函数声明为虚函数即可。含有虚方法的类大概率都是基类——这些方法都会在派生类中被重写，因此他们的析构函数必须为虚函数。

为不含虚方法的类声明虚析构函数不是个好主意。虚函数在实现时需要额外占用内存（虚函数表指针指向虚函数表），导致原本可以正好装入寄存器的增大一倍（指针长度通常等于机器字长），同时还失去了与 C 语言的兼容性。

值得注意的是，STL 中所有的容器的析构函数都是非虚的，因此不要把他们当做基类（C++11 中引入了 `final` 关键字）。

如果将析构函数声明为纯虚函数，则必须要在派生类中实现抽象基类的析构函数，这是由于当派生类析构调用结束后，会调用基类的析构函数。

### Item 8: Prevent exceptions from leaving destructors.

> ✦ Destructors should never emit exceptions. If functions called in a destructor may throw, the destructor should catch any exceptions, then swallow them or terminate the program.  
> 
> ✦ If class clients need to be able to react to exceptions thrown during an operation, the class should provide a regular (i.e., non-destruc- tor) function that performs the operation.

在析构函数中不应该引发异常，否则就会导致当销毁一个类数组，轮流调用对象的析构函数时，引发多个 active exception，这是未定义的行为，会导致程序终止。

但很多时候析构函数执行的代码（释放资源等）就是会抛出异常，如果在析构过程中捕获异常，有两种方案：

- 使用 `std::abort()` 终止程序，并记录日志。这可以避免程序出现未定义行为。
- 继续运行，并记录日志。这可能导致程序异常，毕竟有操作执行失败了。

上述两种方案都无法让用户根据异常信息做出反应，可以显式提供一个资源释放的接口，让用户手动释放资源并根据异常做出反应，析构函数同样可以帮用户“擦屁股”释放资源，但如果有异常不能转发给用户，使用前文的两种处理办法之一。

### Item 9: Never call virtual functions during construction or destruction.

> ✦ Don’t call virtual functions during construction or destruction, be- cause such calls will never go to a more derived class than that of the currently executing constructor or destructor.

不要在构造函数中调用析构函数。假设有个业务类 `Transtraction` 及其纯虚成员函数 `Transtraction::log`，在业务基类中调用这个日志函数，然后根据具体业务派生业务类。如果创建一个具体业务类，其会调用业务基类的日志函数，进而调用日志函数。但是，调用的日志函数**并非**具体业务中的日志，而是 `Transtraction::log`。这是由于，派生类的构造函数还没执行，其成员都还没进行初始化，因此如果虚函数被绑定在派生类上，那么其对于派生成员的调用都是未定义行为。

事实上，派生类在调用基类构造函数过程中，如果使用 runtime type 技术获取其类型，它不是派生类而是基类。

析构的过程也是如此，当进入基类的析构函数，这个对象类型就将被认为是基类而非派生类。

那怎么实现这个需求呢？将 `log` 声明为非虚函数，并要求传入 `string` 类型的日志信息，基类构造函数需要日志信息作为参数，并显式调用 `log`，派生类构造函数显式调用基类日志信息。这个日志信息可以使用派生类的静态私有函数生成，要求是静态函数是为了防止访问非静态成员（此时派生类成员还没有初始化）。 

### Item 10: Have assignment operators return a reference to *this.

> ✦ Have assignment operators return a reference to *this.

在重载赋值运算符时通过返回 `*this`，可以实现等号传递。这条比较简单，不赘述。

### Item 11: Handle assignment to self in operator=.

> ✦ Make sure operator= is well-behaved when an object is assigned to itself. Techniques include comparing addresses of source and target objects, careful statement ordering, and copy-and-swap.  
> 
> ✦ Make sure that any function operating on more than one object be- haves correctly if two or more of the objects are the same.

自己赋值自己看似是个很蠢的想法，但它确实经常出现，例如：

```cpp {hl_lines=["6"]}
vector<Widget> widgets;
...
for(int i=0; i<widgets.size(); i++){
	for(int j=0; j<widgets.size(); j++){
	...
	widgets[i] = widgets[j]
	}
}
```

当 i=j 时，就出现了自己赋值自己的情况。

自己赋值自己可能会出现很多意想不到的情况：如果类的赋值函数的逻辑是先释放资源，再复制资源，这种情况下就会出现复制已经被释放的资源的操作。

为了解决该问题，在赋值运算符实现前先判断两个资源地址是否相同即可：

```cpp {hl_lines=["2"]}
Widget& Widget::operator=(const Widget& rhs){
	if(this == &rhs)    return *this;
	...
}
```

### Item 12: Copy all parts of an object.

> ✦ Copying functions should be sure to copy all of an object’s data members and all of its base class parts.  
> 
> ✦ Don’t try to implement one of the copying functions in terms of the other. Instead, put common functionality in a third function that both call.

如果我们手动实现了一个类的拷贝函数，又处于某种原因添加了成员，记得及时更新拷贝函数和构造函数，编译器不会给出任何警告。

此外，对于派生类的拷贝函数必须显式调用基类的拷贝函数，否则会调用默认构造函数（对于拷贝构造）或者不拷贝基类成员（对于拷贝赋值）。

两个拷贝函数之间一方调用一方都是无意义的，一个用于初始化一个对象，一个用于拷贝一个对象。可以将重复的代码封装为一个成员函数再调用。

题外话，整本书作者都写得挺幽默的，也很喜欢把编译器拟人化。看下面这段，编译器就跟怨妇一样会抱怨你没听它的话：

> When you declare your own copying functions, you are indicating to compilers that there is something about the default implementations you don’t like. Compilers seem to take offense at this, and they retaliate in a curious fashion: they don’t tell you when your implementations are almost certainly wrong.
> 
>  That’s their revenge for your writing the copying functions yourself. You reject the copying functions they’d write, so they don’t tell you if your code is incomplete

## Resource Management

> Resource Management A resource is something that, once you’re done using it, you need to return to the system. If you don’t, bad things happen.

### Item 13: Use objects to manage resources.

> ✦ To prevent resource leaks, use RAII objects that acquire resources in their constructors and release them in their destructors.  
> 
> ✦ Two commonly useful RAII classes are tr1::shared_ptr and auto_ptr. tr1::shared_ptr is usually the better choice, because its behavior when copied is intuitive. Copying an auto_ptr sets it to null.

假设我们有个类用于使用资源，其有一个工厂函数用于得到一个资源对象，该函数的调用者 `f()` 负责释放该对象。即：

```cpp
class Investment { ... }; // 资源使用类

Investment* createInvestment(); // 工厂函数

void f() { 
	Investment *pInv = createInvestment(); 
	..
	delete pInv;
}
```

然而，世事并不遂人愿。`f` 在执行过程中，可能由于 return 语句、异常等导致控制流走不到指针释放的语句，导致对象内存泄漏和资源得不到释放。光凭借人力来手动维护是费时且易出错的。

因此，我们可以把资源交由一个对象来管理，当对象创建，资源随之申请，当对象析构，资源随之释放，即 RAII 模式。可以使用智能指针来管理该资源，即：

```cpp
void f() {
	std::unique_ptr<Investment> pInv(createInvestment());
	...
}
```

上述代码阐明了使用对象管理资源的两个要点：

- 一旦资源成功获取，立即移交给管理者对象。
- 管理者对象通过析构函数来确保资源被正确释放。如果资源在释放过程中引发了异常，参考 [ > Item 8 Prevent exceptions from leaving destructors.](.md#item-8-prevent-exceptions-from-leaving-destructors.)

### Item 14: Think carefully about copying behavior in resource-managing classes.

> ✦ Copying an RAII object entails copying the resource it manages, so the copying behavior of the resource determines the copying behav- ior of the RAII object. 
> 
> ✦ Common RAII class copying behaviors are disallowing copying and performing reference counting, but other behaviors are possible.

对于使用一个资源管理对象来拷贝/构造另一个资源管理对象，可以有如下几种行为：

- 禁止拷贝。
- 引用计数。例如 `shared_ptr`。单个资源被多个管理类管理，他们共享一个引用计数器。存在的问题是，`shared_ptr` 当引用计数为 0 时默认行为是调用资源的析构函数，但像 mutex 锁这类的资源，正确的行为是释放这个锁。好在 `shared_ptr` 提供了设置删除函数的接口，即在初始化时额外传入一个删除函数。
- 拷贝资源。有些资源是可拷贝的（例如内存），这种情况也能深拷贝这些资源。
- 移交所有权。

### Item 15: Provide access to raw resources in resource-managing classes.

> ✦ APIs often require access to raw resources, so each RAII class should offer a way to get at the resource it manages. 
> 
> ✦ Access may be via explicit conversion or implicit conversion. In gen- eral, explicit conversion is safer, but implicit conversion is more con- venient for clients.

围绕一个资源，会有许许多多可以调用的 API，我们不可能在管理类中封装这些 API，因此管理类必须提供一个用于获取原始资源的显式或者隐式方法。

显式方法可以是提供一个接口用户获取被管理资源，或者重载 `*` 或者 `->` 运算符，使得可以直接通过这两个运算符访问资源。

隐式方法是提供类型转换函数，使得资源管理对象可以隐式转换为资源对象，这使得程序员可以像使用资源一样直接把资源管理对象传入资源 API，但与此同时的隐式类型转换也带了一些隐藏疑难的问题。

有人可能会觉得直接访问资源破坏了资源管理类对资源的封装，这点我觉得作者解释得很好：**并非所有类都是用来封装的，资源管理类是用来管理资源的获取和释放的**。

> AII classes don’t exist to encapsulate something; they exist to ensure that a particular action — resource release — takes place.

### Item 16: Use the same form in corresponding uses of new and delete.

> ✦ If you use [] in a new expression, you must use [] in the correspond- ing delete expression. If you don’t use [] in a new expression, you mustn’t use [] in the corresponding delete expression.

new 做的事情：申请一片空间，调用构造函数。delete 做的事情：调用析构函数，释放一片空间。

对于创建一个对象数组，编译器会将数组长度记录在某个位置（许多保存在空间前的地址），在释放对象数组时，必须使用 `delete []` 显式告知编译器要删除的是数组，否则是未定义行为（编译器大概率会将其视为单个对象）。

因此 `new` 和 `delete`、`new []` 和 `delete []` 要配套使用。

### Item 17: Store newed objects in smart pointers in standalone statements.

> ✦ Store newed objects in smart pointers in standalone statements. Failure to do this can lead to subtle resource leaks when exceptions are thrown.

即便使用了智能指针，也可能由于意外导致内存泄漏：

```cpp
processWidget(std::tr1::shared_ptr<Widget>(new Widget), priority());
```

编译器在对上述函数调用的参数求值的时候，标准并未规定其顺序，因此可能先 `new Widget`，并在 `priority()` 中引发异常，只能指针此时还没构造函数调用就异常结束了，进而导致内存泄漏。

解决这个问题的方案也很简单，使用单独的语句保存构造这个智能指针，然后再传入函数调用。

## Designs and Declarations

这一章主要讨论如何设计和实现 C++ 接口。

### Item 18: Make interfaces easy to use correctly and hard to use incorrectly.

> ✦ Good interfaces are easy to use correctly and hard to use incorrectly. You should strive for these characteristics in all your interfaces. 
> 
> ✦ Ways to facilitate correct use include consistency in interfaces and behavioral compatibility with built-in types. 
> 
> ✦ Ways to prevent errors include creating new types, restricting opera- tions on types, constraining object values, and eliminating client re- source management responsibilities. 
> 
> ✦ tr1::shared_ptr supports custom deleters. This prevents the cross - DLL problem, can be used to automatically unlock mutexes (see Item 14), etc.

一个理想的接口实现是：如果接口调用正常运行，说明一切都按照调用者预期进行，否则给出相应反馈。

例如，如果实现一个日期类，构造函数需要传入年月日，相比直接接收三个 `int` 参数，一种更好的方案是分别定义年月日的类，这样可以防止用户混淆了月和日。此外，还可以定义 12 个月常量，不要使用枚举定义，而是在月份的类里定义 12 个静态函数，返回这 12 个月的常量。使用静态函数而非静态常量是为了避免 [ > Item 4 Make sure that objects are initialized before they’re used.](.md#item-4-make-sure-that-objects-are-initialized-before-they’re-used.) 提到的初始化非局部静态常量的问题。

为了防止用户犯错，另一个方案是严格约束一个类可以支持的操作，例如将 `operator *` 的返回值声明为 `const`，或者尽可能用 `const` 修饰函数。这样编译器就可以识别出如下的笔误：

```cpp
if (obj1 + obj2 = obj3){ // 本意是obj1 + obj2 == obj3
						//但写成了将一个变量赋值给另一个临时变量
	...
}
```

我们定义的类最好与内建的类型表现出一致的行为，上面这条规则实际上是本条的特例。尽量与内建类型表现一致有助于减少用户的记忆量和犯错的几率。 

接口不应该让用户一定要做什么收尾的事情（例如释放资源），因此工厂函数最好不要返回野指针，让用户自行封装，而是直接返回智能指针。该方案还能避免 cross dll 问题（申请和释放内容的代码不在同一个 dll），所有资源都是由申请者进行释放。

### Item 19: Treat class design as type design.

> ✦ Class design is type design. Before defining a new type, be sure to consider all the issues discussed in this Item.

好的类型应该有自然的语法、符合直觉的语义和高效的实现。设计类时，要回答好这几个问题：

- 你的对象要怎么构造和销毁？这个问题决定了如何实现构造和析构函数，以及相关的内存申请和释放的函数。
- 对象的初始化和对象赋值有什么区别？这个问题回答了构造函数和拷贝运算符的区别，不要混淆二者。
- 如果你的对象按值传递，会发生什么？按值传递将调用拷贝构造函数，这一过程应该符合预期。
- 你的对象的有效取值有哪些？根据有效值，可以在构造、setter 方法、成员函数中检查是否为有效值。
- 你的类能否正确处理继承关系和被继承？作为派生类，你需要实现虚函数；作为基类，你需要声明虚函数。
- 你的类可以转换为什么类型？如果你的类可以隐式转为其它类型，你要么在那个类中声明一个接受你的类的非显式构造函数，或者在你的类中声明一个那个类的类型转换函数。如果你的类只能显式转换为其它类型，你就不能声明类型转换函数或者声明只有一个参数的非显式构造函数，你要么提供一个方法用于转换为其它类型，或者将其他类型的相对应的构造函数声明为 explicit。
- 哪些函数和运算符对你的类来说是有意义的？这个问题回答了你要实现哪些运算符和函数。
- 你应该禁用哪些编译器可能会生成的函数？如果你不想让编译器生成某些函数，应该显式将其声明为私有的。
- 你的成员访问权限应该是怎么样的？这决定了成员的访问权限，以及友元函数和友元类。
- 你的类有哪些“未声明的接口”？所谓未声明的接口，就是指出了表现出的接口之外，你的类还做出了哪些承诺和保证？例如性能、异常、资源使用等。
- 你的类泛化性能如何？如果你的类想要泛化出一系列类，那你应该定义模板类。
- 你真的需要一个类嘛？如果几个函数就能解决你的问题，那你实际上并不需要一个类。

### Item 20: Prefer pass-by-reference-to-const to pass-by- value.

> ✦ Prefer pass-by-reference-to-const over pass-by-value. It’s typically more efficient and it avoids the slicing problem. 
> 
> ✦ The rule doesn’t apply to built-in types and STL iterator and func- tion object types. For them, pass-by-value is usually appropriate.

默认情况下，函数参数的传递方式为值传递，即实参通过拷贝构造作为形参传递给函数，当函数调用结束时，还需要调用形参的析构函数。这一过程需要浪费大量的时间。

使用 const 引用传递可以避免上述重复的操作，即：

```cpp
int foo(const class_name& param);
```

`const` 关键字可以确保调用者传入的参数不被修改。引用则可以实现虚函数的动态绑定。

对于大部分编译器而言，引用传递是通过指针来实现的，因此，对于一些内建类型，使用值传递的性能可能要优于引用传递。同样，对于 STL 中的迭代器，按值传递的性能优于引用传递。

并不是说，一个类很小，所以它就适合按值传递。一个很小的类其拷贝构造函数也可能很耗时。例如，一个 vector 的指针，拷贝构造函数可能要执行深拷贝，它的运行代价是非常非常昂贵的。

即便拷贝构造函数执行得很快，也并不意味着它适合按值传递。一些编译器区别对待内建类型和用户定义的类，后者即便再小也不允许被保存在一个寄存器中，这就隐含了性能问题。

### Item 21: Don’t try to return a reference when you must return an object.

> ✦ Never return a pointer or reference to a local stack object, a refer- ence to a heap-allocated object, or a pointer or reference to a local static object if there is a chance that more than one such object will be needed. (Item 4 provides an example of a design where returning a reference to a local static is reasonable, at least in single-threaded environments.)

引用传递可以提高传递效率，但这并不意味着所有的函数传递都应该使用引用传递。使用引用传递的前提是被传递的对象确实存在。假设实现了一个有理数类 `Rational`，如果将 `operator *` 的返回类型定义为引用传递，那么在调用 `operator *` 前这个对象肯定是不存在的，这就要让函数来创建这个对象。

函数有两种方式来创建一个对象：在栈上或者在堆上，前者会导致返回的引用对象会被销毁，后者会导致需要调用者手动销毁。即便用户记得销毁，如下代码仍然存在内存泄露：

```cpp
Rational x, y, z, product;
product = x * y * z; // x*y返回的临时对象（在堆上）没有被释放
...
delete product;
```

接下来介绍一种奇淫巧技，通过静态变量来解决内存泄露问题：

```cpp
const Rational& operator*(const Rational& lhs, const Rational& rhs){
	static Rational result;
	result = ...
	return result;
}
```

上面这段代码很“巧妙”地规避了内存泄露问题，但除了很常见的静态变量多线程不安全问题外，`(a * b) == (c * d)` 这个表达式结果是恒 true 的！！

### Item 22: Declare data members private.

> ✦ Declare data members private. It gives clients syntactically uniform access to data, affords fine-grained access control, allows invariants to be enforced, and offers class authors implementation flexibility. 
> 
> ✦ protected is no more encapsulated than public.

为什么不把数据类型声明为 public/protected：

- 语法一致性：用户在调用接口/数据时，无需区分调用的是函数还是直接获取了成员变量。
- 读写权限设置：通过函数获取/写入成员变量时，可以控制每个成员变量的读写权限。
- 封装：通过对 getter 进行封装，如果需要修改 getter 的实现，用户代码也不需要更改。
- 便于维护数据：可以防止客户程序直接修改数据变量，破坏结构。
- 保留了修改的余地：如果后期需要重构这个类，只要保证仍提供相关接口即可，而不需要确保数据成员一定要存在。

### Item 23: Prefer non-member non-friend functions to member functions.

> ✦ Prefer non-member non-friend functions to member functions. Do- ing so increases encapsulation, packaging flexibility, and functional extensibility.

先聊聊封装。一个类封装程度越高，意味着其对外暴露的内容越少，同时意味着我们修改一个类的灵活性也就越高（因为只需要维护对外暴露的内容）。提高我们的灵活性，这就是为什么我们要进行封装。

一个数据成员的封装程度越高，意味着它对外暴露得越少。评判一个数据成员对外暴露的程度，就是统计有类成员方法和友元方法引用了这个成员。

因此，当一个需求既可以使用成员函数实现也可以使用非成员且非友元函数实现，最好使用后者，因为这不会降低数据成员的封装程度。

假设我们实现了一个浏览器类 `WebBrowser`，及相应的清理历史记录、cookies、下载的文件等成员函数。如果我们想些一个 `clearAll` 函数，根据上面的原则，不应该使用成员函数来实现。

就是说，我们可以定义一个函数来实现 clearAll，或者定义一个工具类并实现一个静态函数 clearAll，这在 Java 中更为常见。在 C++ 中，更地道的方法是将 `clearAll` 和 `WebBrowser` 定义在同一个 `namespace` 中：

```cpp
namespace WebBrowserStuff { 
	class WebBrowser { ... }; 
	void clearBrowser(WebBrowser& wb); 
	... 
}
```

得益于 `namespace` 跨文件的特性，可以将不同的类似 `clearAll` 的工具函数声明在不同的头文件中。

### Item 24: Declare non-member functions when type conversions should apply to all parameters.

> ✦ If you need type conversions on all parameters to a function (includ- ing the one that would otherwise be pointed to by the this pointer), the function must be a non-member.

一般来说，让类支持隐式类型转换并不是个好主意，但凡事都有例外。例如，一个数值型的类要支持来自 `int` 的隐式转换是合理的。

接下来，当我们实现加法时，多个选项摆在了面前：重载定义成员函数、定义非成员函数、定义友元函数。

如果我们把他定义成一个成员函数，那么允许来自 `int` 的隐式转换时，`Rational * int` 是可以通过编译的，但是 `int * Rational` 是不可以的，因为 `int` 类型的 `operator *` 并不支持类型 `Rational` 的参数。这显然不够优雅，违反了乘法的交换律。

一种解决方案定义非成员函数 `const Rational operator*(const Rational& lhs, const Rational& rhs)`，当任意一个参数为 `int` 时，编译器会将其隐式转换为 `Rational`。

需求实现了，那么问题来了，要不要声明其为友元函数呢？如果可以，就不要声明为友元，因为友元会降低类的封装程度。

### Item 25: Consider support for a non-throwing swap.

> ✦ Provide a swap member function when std::swap would be inefficient for your type. Make sure your swap doesn’t throw exceptions. 
> 
> ✦ If you offer a member swap, also offer a non-member swap that calls the member. For classes (not templates), specialize std::swap, too. 
> 
> ✦ When calling swap, employ a using declaration for std::swap, then call swap without namespace qualification. 
> 
> ✦ It’s fine to totally specialize std templates for user-defined types, but never try to add something completely new to std.

`swap` 自从在 STL 中引入，就是一个异常安全的函数。其一种经典的实现是：

```cpp
namespace std { 
	template<typename T> 
	void swap(T& a, T& b){ 
		T temp(a); 
		a = b; 
		b = temp; 
	} 
}
```

只要类实现了构造函数和拷贝构造函数，上面这个模板函数就用于该类的交换。然而，默认的 `swap` 函数调用了一次拷贝构造函数和两次拷贝赋值函数，我们可能想根据自己的类定制一个更 fancy 的交换函数。

对于存在类指针数据成员的类来说，拷贝函数进行的深拷贝是不必要的，我们可以在自定义交换函数中执行浅交换，即只要交换指针。注意，这一过程可以通过模板特化进行，而不是完全自定义一个 `swap` 函数。

但是，模板特化也不能访问私有指针，一种做法是将特化的版本声明为友元函数。然而，更传统的做法是在类中声明一个公有接口 `swap`，并在模板特化中调用该接口。STL 的容器就是这么实现的。

但是，上述方案并不适用于模板类。具体来说，模板类中存在模板类型 `T`，在对 `swap` 进行特化时只能进行部分特化，但 C++ 中模板函数不支持部分特化：

```cpp
namespace std{
	template<typename T>
	void swap<Widget<T>>(Widget<T>& a, Widget<T>& b){  // 对swap部分特化是不允许的
		a.swap(b);
	}
}
```

一种方案是对 `swap` 进行重载（删除 `<Widget<T>>` 即可），但很遗憾，C++ 标准规定 std 命名空间只能由 C++ 标准委员会进行修改，而重载属于修改，是不被允许的。

似乎所有路都被堵死了？其实没有！别忘记，我们不一定要重载或者特化 `std::swap`，我们可以直接在 `Widget` 的命名空间中声明 `swap` 并使用。得益于 ADL 机制，编译器会自动调用 `Widget` 所在命名空间的 `swap`。

上述方案是万能的嘛？很遗憾，又不是。如下的一段代码，当执行交换时，调用的是哪个函数呢？`std::swap` 还是使用 `T` 特化的版本？又或者某个命名空间中针对类型 `T` 的 `swap`。

```cpp
template<typename T> 
void doSomething(T& obj1, T& obj2) {
	... 
	swap(obj1, obj2); 
	... 
}
```

你可能想的是：如果有针对类型 `T` 的 swap，则优先调用，如果没有则回落到 `std::swap`，在 `doSomething` 中添加一行就能实现你的需求：

```cpp {hl_lines=["3"]}
template<typename T> 
void doSomething(T& obj1, T& obj2) {
	using std::swap;
	... 
	swap(obj1, obj2); 
	... 
}
```

当调用 swap 时，编译器首先会在全局空间或者 `T` 所在的命名空间寻找参数为 `T`swap 函数，如果找不到，则会在 `std` 空间中寻找特化的 swap，如果还是没有，则使用通用的 swap 实现。

本节内容有点多，小结一下：

- 如果通用的 swap 性能可以接受，则没必要自己实现。
- 如果要自己实现，步骤为：
	- 提供一个 swap 成员接口
	- 在类所在的明明空间提供一个 swap 非成员函数，调用 swap 成员函数接口
	- 如果你写的是类不是模板类，则为其特化一个 `std::swap`
- 当调用 swap 时，确保使用 using 语句，使得 `std::swap` 是可见的。

最后一点忠告：swap 成员函数不应该抛出异常。这是因为 swap 一个很重要的应用就是帮助类提供强异常安全的保证。这一约束仅用于成员函数，非成员函数不受此限制。

## Implementations

### Item 26: Postpone variable definitions as long as possible.

> ✦ Postpone variable definitions as long as possible. It increases pro- gram clarity and improves program efficiency.

对象的构造和析构过程需要时间，因此，尽可能推延变量的定义，知道接下来必须要使用这个变量。例如下面代码中，提前定义了需要返回的 `ret` 再判断异常逻辑。当触发异常时，`s` 的构造和析构是不必要的：

```cpp
std::string foo(string s){
	...
	string ret;
	if(s.size() == 0){
		throw logic_error("s is empty");
	}
	...
	return ret;
}
```

此外，上述代码会将 `ret` 初始化空串，这也是不必要的，之后对其赋值还会调用拷贝构造函数。更合适的做法是直接把计算出的返回值赋给 `ret`。

所谓“as long as possible”，不仅仅指的是延迟变量的定义，而是当明确了变量的值之后再定义这个变量。

对于循环中要使用的对象，一般在循环外定义更好，这可以避免多次调用构造和析构函数。

### Item 27: Minimize casting.

> ✦ Avoid casts whenever practical, especially dynamic_casts in perfor- mance-sensitive code. If a design requires casting, try to develop a cast-free alternative. 
> 
> ✦ When casting is necessary, try to hide it inside a function. Clients can then call the function instead of putting casts in their own code. 
> 
> ✦ Prefer C++-style casts to old-style casts. They are easier to see, and they are more specific about what they do.

C++ 支持如下格式的类型转换：

- C 风格：`(T) expression`
- 函数风格：`T(expression)`
- C++ 形式：
	- `const_cast<T>(expression)`：移除一个变量的 const 修饰，只有 `const_cast` 运算符支持该转换。
	- `dynamic_cast<T>(expression)`：进行“safe downcasting”，即判断一个基类对象能否安全转换为派生对象，该运算符有较大的运行时开销。
	- `reinterpret_cast<T>(expression)`：进行两个无关类型之间的转换，即按照比特位重新解析为另外一个对象。该转换除非是面向低层编码，否则不应该使用。
	- `static_cast<T>(expression)`：进行强制隐式类型转换  
建议使用新版的 C++ 形式进行类型转换，一方面这些类型转换语句在代码中更容易识别，另一方面新的四种类型转换功能更加细化，方便查找错误。

不同编译器和不同平台的内存排布可能不同，因此不要根据内存排布进行低层的类型转换。

`static_cast` 如果传入的派生类对象，会返回基类对象的拷贝；如果传入派生类指针或引用，会返回基类对象指针或引用。因此，如果要调用基类非 const 成员函数，需要先转换为基类引用或者基类指针，再调用，否则该函数对该对象的修改是不起作用的。

`dynamic_cast` 开销并不小，能避免就避免。可以使用虚函数的动态绑定机制，在不进行类型转换的情况下通过基类指针访问派生类的函数。

### Item 28: Avoid returning “handles” to object internals.

> ✦ Avoid returning handles (references, pointers, or iterators) to object internals. Not returning handles increases encapsulation, helps const member functions act const, and minimizes the creation of dangling handles.

一个成员变量的封装程度也与返回该对象的引用的成员函数的访问权限有关，如果公有函数返回了私有变量，那么这个变量的封装就被破坏为公有的。

如果一个对象内部的数据成员以指针的形式指向外部空间，并且该指针也可以被外部访问，那么即便这个对象被 const 修饰，其成员的内容还是会被修改。

指针、引用、迭代器等都会存在上述问题，他们可以统称为用于用于访问对象的句柄。

上面的两个问题指出了要遵守的规则：成员函数不得返回访问权限比自身更严格的成员变量/函数的句柄，除非有意为之并将返回值声明为 `const`。

此外，如果一个类的成员函数返回了类内部成员的引用，还可能诱发临时对象销毁后访问问题，即这个类的临时对象调用了这个成员函数，其返回值将在返回后被销毁。例如：

```cpp
class A{
	Data data_;
	const Data& get_data() const {
		return data_;
	}
};

const Data* const p_data = &(A().get_data());
```

### Item 29: Strive for exception-safe code.

> ✦ Exception-safe functions leak no resources and allow no data struc- tures to become corrupted, even when exceptions are thrown. Such functions offer the basic, strong, or nothrow guarantees. 
> 
> ✦ The strong guarantee can often be implemented via copy-and-swap, but the strong guarantee is not practical for all functions. 
> 
> ✦ A function can usually offer a guarantee no stronger than the weak- est guarantee of the functions it calls.

当一个异常被跑出，异常安全的函数应该做到：

- 没有资源泄露。资源泄露不仅仅是内存泄露，还包括锁等资源。这一点可以通过 [ > Item 13 Use objects to manage resources.](.md#item-13-use-objects-to-manage-resources.) 中的 RAII 做到。
- 数据结构没有被破坏。即需要维护的数据结构仍然保持维护的状态。

异常安全的函数满足以下三种特性之一：

- 最基本的保证：如果抛出异常，程序内的所有状态都是合法且有效的，但无法预知这些状态的取值。
- 强力保证：如果抛出异常，程序内所有的状态和函数调用前相同。这样的函数我们称之为原子函数。
- 不抛出异常保证：函数保证在执行过程中不会抛出异常。内建类型的所有操作都是这样的函数。

需要注意的是，类似 `void foo() noexcept;` 这样的函数声明并不意味着该函数保证不会抛出异常，这个声明意味着如果抛出了异常，那是致命的错误。相反，该函数甚至可能无法提供任何级别的异常安全保证。

函数是否是异常安全的并不取决于它的函数声明，而是取决于其具体实现。确保不抛出异常是很困难的，尤其是当使用 C++ 的各种库时，通常只要实现稍弱的两种保证即可。

要想提供异常安全的强力保证，通常会使用到 `swap and copy` 技术，即先对要修改的对象的拷贝进行修改，没有异常再交换二者。

一旦涉及到函数彼此调用，想要实现强力保证就很快困难，即便被调用的函数能够提供强力保证。在下面的代码中，`foo` 调用了 `f1` 和 `f2`，如果 `f1` 正常调用结束，但 `f2` 发生了异常而回退，此时需要由 `foo` 追踪 `f1` 的修改内容并进行回退——这显然相当困难。

```cpp
void foo(){
	...
	f1();
	f2();
	...
}
```

异常安全的强力保证需要消耗大量的资源和性能，并不适用于所有的场景。这种情况下，我们就要转向基本保证。

但基本保证也不是一件易事，仍旧考虑上面那个调用两个函数的例子，如果 `f1` 是异常不安全的，那么当其排除异常时，内部可能存在资源泄露，这对于调用者 `foo` 来说是无法定位并释放的。因此，如果一个函数调用了异常不安全的函数，那其也无法提供异常安全的保证。

同样的，对于一个系统来说，其要么是异常安全要么是异常危险的，不可能介于二者之间。一旦这个系统中有一个函数是异常危险的，这个系统就不可能是异常安全的。

### Item 30: Understand the ins and outs of inlining.

> ✦ Limit most inlining to small, frequently called functions. This facili- tates debugging and binary upgradability, minimizes potential code bloat, and maximizes the chances of greater program speed. 
> 
> ✦ Don’t declare function templates inline just because they appear in header files.

内联函数除了可以减少函数调用开销，还可以给予编译器更大的优化空间。

但是，启用内联，也会让目标文件变得更大（所有调用内链函数的地方都会被展开），增加换页次数、降低 cache 命中率。

`inline` 是向编译器建议，而不是强制要求编译器将该函数处理为内联函数。有两种方式向编译器提出建议：隐式，即在类中给出成员/友元函数的定义；显式，即在函数定义处使用 `inline` 关键字。

编译器要在编译器将内联函数调用原地展开，因此内链函数必须在头文件中给出。模板函数也是如此。但这并不意味着模板函数和内联函数之间存在什么充分必要关系。

库的设计者应该评估是否将一个接口声明为 `inline`，如果这样做，一旦需要对内联函数的实现进行修改，所有调用该函数的代码也需要被重新编译。修改一个普通函数，则仅仅需要重新链接。

### Item 31: Minimize compilation dependencies between files.

> ✦ The general idea behind minimizing compilation dependencies is to depend on declarations instead of definitions. Two approaches based on this idea are Handle classes and Interface classes. 
> 
> ✦ Library header files should exist in full and declaration-only forms. This applies regardless of whether templates are involved.

当我们修改一个类的具体实现后，所有直接和间接依赖这个类的文件都会被重新编译。这是因为 C++ 中的接口和实现没有很好地分离。

```cpp
#include "data.h"
class Person{
public:
	const Date& get_birthdate() const; // interface
private:
	Date birthdate_; // implementation detail
};
```

例如，`Person` 类中有接口 `get_birthdate`，其私有成员变量 `Date birthdate_` 就是一个实现，在编译 `Persion` 时，必须知道 `Date` 的具体实现，才能顺利编译。这是因为必须在 `Person` 中给 `Date` 成员预留出足够的空间，而不知道其具体实现，则无法获知其大小。

**解决方案一：句柄类**

在 Java 中，则不存在上述困扰。当在 Java 中定义一个类时，类成员以指针的形式保存在类中，而不为其预留完整空间。

可以使用 C++ 模拟这一过程，这被称为“pimpl idiom”（point to implementation），具体为：将原本 `Person` 在头文件中的定义分为接口 `Person` 和实现 `PersonImpl` 两个类，前者只声明对外的接口和一个指向具体实现类的指针，后者定义具体的数据成员和接口实现。即：

```cpp
// person.h
#include <memory>

class Date;

class Person{
public:
	const Date& get_birthdate() const; // interface
private:
	std::shared_ptr<PersonImpl> pImpl_; 
};
```

需要注意的是，这里使用了前向声明（forward declaration）技术

pimpl idiom 技术的核心理念是：将对实现的依赖转换为对声明的依赖。根据该理念，可以导出两个技巧：

- 如果能使用对象指针或者引用，就不要直接使用对象。声明一个对象需要该对象的定义，但是指针和引用只需要声明。
- 尽可能依赖声明而非实现。即便是某个函数的参数类型或者返回值类型，是可以直接声明为该类而不需要一定声明为指针或者引用的。
- 一个类分别要提供声明和定义两个头文件。调用者要包含声明的头文件而非前向声明某个类。

**解决方案二：接口类**  
除了 pimpl idiom，另一种处理方式是将 `Person` 声明为一种特殊的抽象基类——接口，其作用是为派生类指定必须实现公有函数接口。通常来说，接口没有数据成员，没有构造函数，一个虚拟析构函数和一系列纯虚函数。

C++ 中的接口不如 Java 中的限制严格，允许接口具有数据成员。

`Person` 接口可以声明为：

```cpp
class Date;
class Person{
public:
	virtual ~Person();

	virtual const Date& get_birthdate() const = 0;
};
```

注意，这个类的使用者只能使用 `Person` 的引用或者指针。按照这种方式实现的 `Person`，除非其接口有所改变，否则即便 `Person` 的实现修改调用者也不用重新编译。

接口的调用者需要一个用于创建对象的方法，常用的方式是提供一个静态工厂函数接口用于创建一个对象，并返回相应的智能指针。这个工厂函数可以工具参数返回这个接口的不同派生对象。

注意，由于工厂函数是一个静态函数，并不依赖于具体的数据成员或者方法，因此其所在的类仍旧是一个抽象类/接口。

当然，上述方案减少了头文件之间的依赖，代价是增大了对象的体积，略微减慢了运行速度。

句柄类的解决方案每次访问对象，都要进行一次指针访问操作；接口类的解决方案中，每个函数都是虚函数，每次访问接口函数，都有一次虚函数调用的开销。

## Inheritance and Object-Oriented Design

这一章将集中介绍 C++ 中面向对象相关的内容，包括继承、派生和虚函数。C++ 中的 OOP 遵循 OOP 的基本理念，但又与其他语言的 OOP 有所不同。只有正确理解 C++ 中的 OOP，才能把“所想”通过 C++ 变成“所得”。

### Item 32: Make sure public inheritance models “is-a.”

> ✦ Public inheritance means “is-a.” Everything that applies to base classes must also apply to derived classes, because every derived class object is a base class object.

**公有继承意味着“is-a”关系**，也就是说，类型 `D` 的所有对象也是类型 `B` 的对象。前面说的是 OOP 最基本的理念，必须要记住。

C++ 中，需要基类对象的地方也可以传入派生类对象，当且仅当是公有继承才允许。

is-a 关系很容易被直觉和不精确误导：众所周知，企鹅是一种鸟，并且鸟会飞，根据上述想法，不难写出如下代码：

```C++
class Bird{
	...
public:
	virtual void fly();
};

class Penguin: public Bird{
	...
}
```

但事实上，企鹅并不会飞。这一问题的根源在于并不是所有的鸟都会飞，语言的表述是不准确的。更合理的做法是，派生出一个 `FlyingBird` 类，并在该类中声明虚函数 `fly`。当然，一切取决于需求，如果不需要使用 `fly` 这个行为，就没必要派生出 `FlyingBird` 这个抽象类。

is-a 关系与数学上的特殊 - 一般关系也不相同，例如数学上正方形是一种特殊的长方形，但在 C++ 的公有继承不能这么实现。公有继承 is-a 关系指的是，派生类满足基类的一切性质，而正方形的长宽必须一致，这一特性导致长方形的某些方法并不适用于正方形。

### Item 33: Avoid hiding inherited names.

> ✦ Names in derived classes hide names in base classes. Under public inheritance, this is never desirable. 
> 
> ✦ To make hidden names visible again, employ using declarations or forwarding functions.

在类继承中，同样存在名称遮蔽，即派生类中的成员会遮蔽基类中的同名成员。对于成员变量来说，一切都符合直觉，但对于成员函数来说，就不是这么一回事了。

首先，成员函数的遮蔽是以函数名为标志的，也就是说，派生类中的成员函数除了会遮蔽基类中签名相同的同名函数，还会遮蔽基类中同名的重载函数。例如：

```cpp
class Base{
public:
	void fun1();
	void fun1(int x); // 重载
};

class Derived: public Base{
public:
	void fun1(); // 遮蔽了基类中所有名为fun1的成员函数
}

/**********************************/

int x=1;
Derived d;
d.fun1(x); // 不合法
```

C++ 的这一默认行为既不符合直觉，也不符合公有继承是 is-a 的关系。为了使重载函数仍旧在派生类中可见，可以在派生类中添加一行 using 语句：

```cpp
class Base{
public:
	virtual void fun1();
	virtual void fun1(int x); // 重载
};

class Derived: public Base{
public:
	using Base::fun1; // 基类中所有名为fun1的成员都在派生类中可见
	virtual void fun1(); 
};

/**********************************/

int x=1;
Derived d;
d.fun1(x); // 合法
```

“在派生类中只继承基类重载成员函数某几个版本”这一想法在**公有继承**中违反了 is-a 理念，但在**私有继承**中，这个需求是合理的。如果在上面代码中，私有继承的派生类只想继承 `fun1` 的无参版本，可以使用转发：

```cpp
class Base{
public:
	virtual void fun1();
	virtual void fun1(int x); // 重载
};

class Derived: public Base{
public:
	virtual void fun1()
	{Base::fun1();} // 转发
};

/**********************************/

int x=1;
Derived d;
d.fun1(x); // 不合法
d.fun1(); // 合法，调用的是Derived::fun1()
```

### Item 34: Differentiate between inheritance of interface and inheritance of implementation.

> ✦ Inheritance of interface is different from inheritance of implementa- tion. Under public inheritance, derived classes always inherit base class interfaces. 
> 
> ✦ Pure virtual functions specify inheritance of interface only. 
> 
> ✦ Simple (impure) virtual functions specify inheritance of interface plus inheritance of a default implementation. 
> 
> ✦ Non-virtual functions specify inheritance of interface plus inherit- ance of a mandatory implementation.

在 C++ 的继承过程中，需要区分继承一个接口和继承一个函数。前者指的是，只继承这个函数的声明，而不继承基类中的实现（通常也不存在该实现），后者指的是同时继承声明和实现，同时还要区分能否重写（override）该函数。

如果只需要继承来自基类的接口，可以在基类中将该接口声明为纯虚函数（事实上接口也就应该是纯虚函数）。一个冷知识是，纯虚函数同样可以在基类中给出定义，只是在调用时要显式指定，例如 `Base::fun()`。

如果需要继承一个实现，同时允许在派生类中重写该方法，可以在基类中将该方法声明为虚函数。在实践过程中，往往会由于一个基类的多个派生类的同一个方法具有相同的实现，因此将其作为基类的默认实现。但这也为未来埋下了隐患：之后派生出的某个类并不适用该实现，但是重写该方法了，在编译阶段不会发现这个错误。解决方案为：

```cpp
class Base{
public:
	virtual void fun() = 0;// 改为纯虚函数
protected:
	void default_fun() = 0;
};

class Derived: public Base{
public:
	virtual void fun() // 转发到默认函数
	{default_fun();}
};
```

解决方案就是将原函数声明为纯虚函数，并提供一个非同名默认实现函数，在需要使用该默认实现的派生类中，重写该方法，转发到基类默认实现。

有些人不喜欢上面将声明和实现写在两个函数中的方案，转而在基类中为纯虚函数提供一个定义来实现该需求。这是可行的，但在默认实现的权限控制上不如上面这个方案细粒度高。

如果需要继承一个实现，同时禁止在派生类中重写该方法，那么就应该将该方法声明为非虚函数，并使用 `final` 关键字。

### Item 35: Consider alternatives to virtual functions.

> ✦ Alternatives to virtual functions include the NVI idiom and various forms of the Strategy design pattern. The NVI idiom is itself an ex- ample of the Template Method design pattern. 
> 
> ✦ A disadvantage of moving functionality from a member function to a function outside the class is that the non-member function lacks ac- cess to the class’s non-public members. 
> 
> ✦ tr1::function objects act like generalized function pointers. Such ob- jects support all callable entities compatible with a given target sig- nature.

虚函数在实现过程中被尝尝使用，但实际上其也有几种替代品：

- 通过非虚接口实现模板方法模式  
这里的非虚接口来自一种理念：虚拟函数应该是私有的。所谓模板方法模式是一种设计模式，指的是在父类中定义了一个算法的框架，允许子类在不改变算法结构的情况下重写算法中的某些步骤。具体来说，在基类中提供一个非虚接口，其实现是调用某几个特定的私有虚函数，在派生类中，通过修改这几个私有虚函数的实现以修改派生类中的行为。

这一设计模式的好处是可以在公有接口中在调用私有接口前后添加一些自定义内容，例如初始化环境、打日志、检查返回值、申请释放锁等。这一模式是控制反转的提现：高层抽象类负责控制基本流程顺序，低层派生类负责控制每个流程的具体实现。

- 通过函数指针实现策略模式  
前面提到的模板方法的解决方案，仍旧用到了虚函数（尽管其是私有的），一种更灵活的解决方案是要求派生类在构造基类时传入一个函数指针，基类在实现相关方法时，将调用该函数。

其灵活性体现在，即便是同一派生类的不同实例，也可以具有不同的函数实现。

起问题在于，作为非成员函数，该函数无法访问类中的非公有变量。解决方案是降低这个类的封装程度，例如将该函数声明为友元函数，或者提供访问这些变量的公有方法。

- 通过 `std::function` 实现策略模式  
函数指针的实现方案灵活度不够高：参数必须完美匹配，并且只支持常规函数。对其稍加改造，使用 `std::function` 来替代函数指针，则支持各种可调用的对象（函数对象、lambda 函数、成员函数等），且支持自动类型转换。

### Item 36: Never redefine an inherited non-virtual function.

> ✦ Never redefine an inherited non-virtual function.

非虚函数使用的是静态绑定，即基类指针分别指向基类和派生类对象，调用同一个非虚函数，如果这个函数在派生类中被重新定义了，那么二者调用的版本是不同的。这并不符合面向对象的设计原则：

- 前面提到，非虚函数的含义是为该类指定了某种特定实现，这种实现不应该在派生类中修改。如果有修改的需求，应该将其指定为虚函数。
- 前面提到，公有继承是 is-a 关系，如果在派生类中要重定义某个函数，说明派生对象 is not a 基类对象，与 is-a 关系矛盾。

### Item 37: Never redefine a function’s inherited default parameter value.

> ✦ Never redefine an inherited default parameter value, because default parameter values are statically bound, while virtual functions — the only functions you should be redefining — are dynamically bound.

不要修改函数继承的默认参数，这个条款乍一看很奇怪，这实际上是 C++ 中为了更高效地实现虚函数而出现的一种特性，即：

```cpp
class Base{
public:
	virtual void show(string str="Base"){
		cout << "call Base::show "<< str << endl;
	}
};

class Derived: public Base{
public:
	virtual void show(string str="Derived"){
		cout << "call Derived::show "<< str << endl;
	}
};
/*********/

Derived d = Derived();
Base &pd = d;
pd.show(); // output: call Derived::show base
```

具有默认参数的虚函数在进行动态绑定时，其默认参数是静态绑定的。这就造成了上面这几行代码中，的确调用了派生类中重写了的 `show` 函数，但是传入的默认函数是来自 `pd` 静态的类型 `Base` 中对应方法的参数。这一特性是为了减少虚函数表中需要维护的内容，但也导致了其不符合直觉的行为。

这种情况下，在派生类中将待重写的虚函数的参数列表照抄基类中的列表也是不合适的（未来可能修改参数的默认值）。一种解决方案是使用前文提到过的非虚接口：

```cpp
class Base{
public:
	virtual void show(string str="Base"){
		do_show(str);
	}
private:
	virtual void do_show(string str){
		cout << "call Base::do_show" << str << endl;
	}
};

class Derived: public Base{
private:
	virtual void do_show(string str){
		cout << "call Derived::do_show" << str << endl;
	}
};
/*********/

Derived d = Derived();
Base &pd = d;
pd.show(); // output: call Derived::do_show base
```

由于非虚函数不可在派生类中重写/遮蔽，因此 `show` 的默认参数只能为值 base。

### Item 38: Model “has-a” or “is-implemented-in-terms- of” through composition.

> ✦ Composition has meanings completely different from that of public inheritance. 
> 
> ✦ In the application domain, composition means has-a. In the implementation domain, it means is-implemented-in-terms-of.

组合关系（composition）指的是一个物体由多个对象组合而来，或者一个对象包含了其他对象的关系。与公有继承意味着 is-a 类似，组合关系意味着 has-a 或者 is-implemented-in-terms-of（基于 xxx 而实现）。

组合关系的这两层含义，对应着两种不同领域：has-a 常用于对现实世界建模，is-implemented-in-terms-of 常用语纯粹的实现领域，例如实现锁、二叉树等等。

区别 has-a 和 is-a 比较简单，但区分 is-implemented-in-terms-of 和 is-a 就有说法了。例如，当我们需要使用链表来实现集合时，这是哪种关系呢？如果 D is-a B，那么对于 B 成立的说法，对 D 都应该成立，但是链表允许有重复值，集合则允许，因此不是 is-a 关系。

### Item 39: Use private inheritance judiciously.

> ✦ Private inheritance means is-implemented-in-terms of. It’s usually inferior to composition, but it makes sense when a derived class needs access to protected base class members or needs to redefine inherited virtual functions. 
> 
> ✦ Unlike composition, private inheritance can enable the empty base optimization. This can be important for library developers who strive to minimize object sizes

私有继承有如下两个影响：

- 派生类对象不允许被转换为基类对象；
- 基类成员在派生类中的访问权限为私有。  

上面两个特性决定了，私有继承的含义为 is-implemented-in-terms-of，它和组合的一种含义相同。只有在迫不得已时，才应该使用私有继承，通常应该使用组合。

迫不得已？例如要使用基类保护成员，或者要重写虚函数的情况。

### Item 40: Use multiple inheritance judiciously.

> ✦ Multiple inheritance is more complex than single inheritance. It can lead to new ambiguity issues and to the need for virtual inheritance. 
> 
> ✦ Virtual inheritance imposes costs in size, speed, and complexity of initialization and assignment. It’s most practical when virtual base classes have no data. 
> 
> ✦ Multiple inheritance does have legitimate uses. One scenario in- volves combining public inheritance from an Interface class with private inheritance from a class that helps with implementation.

如果使用多继承，很容易出现名称相同（歧义）的情况。C++ 在解析对重载函数的调用时，首先搜索最佳匹配的函数，然后再检查其权限。这就导致了，即使同名的两个函数一者是私有的，编译器仍旧不能正确对多继承中的同名函数正确解析。

为了解决这种歧义，在函数调用时必须显式指出调用的是哪个基类下的函数名。

在多继承中，同一个基类可能沿着不同的路径被继承了多次，这些数据在最终的派生类中可以有两套独立的副本，也可以共享一个副本（虚继承）。被虚继承的基类称为虚基类。

一般来说，所有的公有继承都应该是虚继承的。但是，虚继承本身存在性能代价：一方面，编译器需要为虚基类维护更多的信息，另一方面，在初始化时派生类的作者必须了解到有哪些虚基类，并为其手动初始化。

因此，虚基类能不用就不用，即便要用，虚基类中的数据成员能少就少。

多继承的一个合理的使用场景是：公有继承一个接口，同时私有继承一个类帮助实现这个接口。之所以要私有继承一个类，是因为要修改其的虚函数，否则使用组合即可。

## Templates and Generic Programming

从最初的容器开始，模板进入程序员的世界。后来人们发现模板的能力远不止于此，一种新的编程范式——模板变成应运而生。随后 C++ 中的模板又被证明为是图灵完备的，一种在编译期运行的程序——模板元变成又诞生了。

### Item 41: Understand implicit interfaces and compile-time polymorphism.

> ✦ Both classes and templates support interfaces and polymorphism. 
> 
> ✦ For classes, interfaces are explicit and centered on function signa- tures. Polymorphism occurs at runtime through virtual functions. 
> 
> ✦ For template parameters, interfaces are implicit and based on valid expressions. Polymorphism occurs during compilation through tem- plate instantiation and function overloading resolution.

在面向对象编程中，显式接口和运行时多态是重要的组成部分。在泛型编程中，二者仍然生效，但更重要的是隐式接口和编译时多态。

- 在泛型编程中，所谓隐式接口就是对类型 `T` 执行的所有操作、调用的所有方法
- 所谓编译时多态指的是对 `T` 的实例化类型参数的不同会导致调用不同的方法，这就实现了多态

通过各种表达式，可以为类型 `T` 声明其必须支持的接口有哪些。更严谨的说法是，`T` 必须要支持一些接口，使得这些表达式合法。例如：

```cpp
template <typename T>
void do(T &w){
	if(w.size() > 10 && w!=xxx){
	...
	}
}
```

这几行代码并非意味着 `T` 必须支持返回一个可以与 10 比较类型的值的 `size` 方法，实际上，它只需要返回一个支持/重载了运算符 `operator >` 且接受参数 10 的类型即可。同样的，`T` 也不一定要重载 `!=` 运算符，只要 `w` 可以转换为某个类型 `X` 并且 `xxx` 可以转换为某个类型 `Y`，且 `X!=Y` 这个运算符有定义即可。

### Item 42: Understand the two meanings of typename.

> ✦ When declaring template parameters, class and typename are inter- changeable. 
> 
> ✦ Use typename to identify nested dependent type names, except in base class lists or as a base class identifier in a member initializa- tion list.

在声明模板参数中，`typename` 和 `class` 是等价的。有些程序员会区分使用二者，例如只接受类的参数使用 `class`，接受一切类型的参数使用 `class`。

模板中，由于不知道参数的具体类型，很容易引发歧义：

```cpp {hl_lines=["3"]}
template<typename C>
void foo(const C& container){
	C::const_iterator *x;
}
```

上述代码的第四行中，本意是声明一个指向类型为 `C::const_iterator` 的指针 `x`，但如果类 `C` 中恰好存在一个名为 `const_iterator` 的静态成员变量，并且恰好存在一个名为 `x` 的全局变量，这样代码的含义就变为了两个表达式相乘。编译器必须考虑各种可能性，默认情况下，其不会将类中的名称，例如 `C::const_iterator` 视为一个类型名称。需要在前面使用 `typename` 关键字修饰，这样编译器将把其视为类型明对待：

```cpp
template<typename C>
void foo(const C& container){
	typename C::const_iterator *x;
}
```

注意：此处 `typename` 不可使用 `class` 替换。

“在模板参数类的内嵌类型名前需要使用 `typename` 修饰”这一规则的一个例外是：在类继承的基类列表名和类初始化列表中，不得使用 `typename` 修饰：

```cpp
template<typename>
// Derived继承了Base<T>中的一个内嵌类Nested
class Derived: public Base<T>::Nested{ // 基类列表，不可使用typename修饰
public:
	explicit Derived(int x)
	:Base<T>::Nested(x) // 初始化列表，不可使用typename修饰
	{ ... }
}
```

### Item 43: Know how to access names in templatized base classes.

> In derived class templates, refer to names in base class templates via a “this->” prefix, via using declarations, or via an explicit base class qualification.

当需要继承模板基类并访问其中的方法时，编译器会拒绝访问：

```cpp {hl_lines=["28"]}
class BaseA{
public:
	void do_foo1();
	void do_foo2();
};

// 模板基类，有foo1和foo2两个接口
template <typename BaseName>
class Base{
public:
	void foo1(){
		BaseName base;
		base.do_foo1();
	}

	void foo2(){
		BaseName base;
		base.do_foo2();
	}
};


// 派生类，调用foo2方法
template <typename BaseName>
class Derived: public Base<BaseName>{
public:
	void call_foo2(){
		do_foo2(); // invalid
	}
};
```

之所以第 28 行不能正确访问基类中的 `do_foo2` 方法，是因为模板特化的存在。在模板基类中的一个特化版本可能没有提供 `do_foo2` 方法，因此编译器拒绝编译该代码。

有如下三种解决方案：

- 在函数调用前使用 `this->`：

```cpp {hl_lines=["5"]}
template <typename BaseName>
class Derived: public Base<BaseName>{
public:
	void call_foo2(){
		this->do_foo2(); // valid
	}
};
```

- 使用 `using` 声明该方法：

```cpp {hl_lines=["4"]}
template <typename BaseName>
class Derived: public Base<BaseName>{
public:
	using MsgSender<BaseName>::do_foo2;
	void call_foo2(){
		do_foo2();
	}
};
```

- 显式指定调用基类中的方法：

```cpp {hl_lines=["5"]}
template <typename BaseName>
class Derived: public Base<BaseName>{
public:
	void call_foo2(){
		MsgSender<BaseName>::do_foo2();
	}
};
```

应该避免使用第三种解决方案，因为作用域限定符将会使得虚函数的动态绑定机制失效。

从名称可见性的角度看，这三个解决方案都做了一件事：向编译器保证 `do_foo2` 这个方法在任何模板特化中总是存在的。如果实际上不存在，那么在编译器该错误将被发现。

### Item 44: Factor parameter-independent code out of templates.

> ✦ Templates generate multiple classes and multiple functions, so any template code not dependent on a template parameter causes bloat. 
> 
> ✦ Bloat due to non-type template parameters can often be eliminated by replacing template parameters with function parameters or class data members. 
> 
> ✦ Bloat due to type parameters can be reduced by sharing implemen- tations for instantiation types with identical binary representations.

模板可以精简源码的大小，但也有可能在实例化的过程中增大生成的可执行文件的大小，一个原因就是模板被实例化次数过多了。例如，当我们想要实现一个支持转置的方阵：

```cpp
template<typename T, size_t n>
class SquareMatrix{
public:
	...
	void invert();
};
```

这个模板类有两个参数，一个类型参数 `T` 指示数据类型，一个非类型参数 `n` 指示矩阵大小。这是个常见的操作，但其会导致对于每个不同矩阵大小 `n`，即便数据类型相同，依旧会生成多份 `invert` 的实现代码。这显然是没必要的，需要再次进行抽象。

一种做法是抽象出一个模板基类，只接收一个类型参数 `T`，并提供一个 `void invert(size_T n)` 方法，让派生类将非类型参数转发到该方法。这样，相同类型的数据将共享相同的模板实例。

但是，抽象后的方案并不一定比原始方案更好。原始方案在编译期就确定了矩阵大小，编译器有更多的优化空间。另一方面，优化后的方案可执行文件更小，能够减少工作集的大小，提升程序的局部性，提升 cache 命中率。

### Item 45: Use member function templates to accept “all compatible types.”

> ✦ Use member function templates to generate functions that accept all compatible types. 
> 
> ✦ If you declare member templates for generalized copy construction or generalized assignment, you’ll still need to declare the normal copy constructor and copy assignment operator, too.

假设我们想实现一个智能指针类 `SmartPointer`，要求支持从任何兼容的类型（任意类型的裸指针）构造：

```cpp
template<typename T>
class SmartPointer{
public:
	SmartPointer(T *real_ptr);
...
};
```

接下来，要求不同类型的智能指针之间可以相互转换，可以使用通用复制构造函数 generalized copy constructors：

```cpp
template<typename T>
class SmartPointer{
public:
	template<typename U>
	SmartPointer(const SmartPointer<U>& other);
...
};
```

上述代码在模板类中使用了模板构造函数，以允许来自其它实例的构造参数。

接下来，要求这个智能指针能够像裸指针一样，支持隐式的类型转换，例如，派生类指针转换为基类指针，我们使用 cpp 内置的之间转换来实现：

```cpp
template<typename T>
class SmartPointer{
public:
	template<typename U>
	SmartPointer(const SmartPointer<U>& other)
	:held_ptr(other.get();) {};
	T* get() const {return held_ptr};
...
private:
	T* held_ptr;
};
```

需要注意的是，模板构造函数并不会组织编译器生成默认构造函数。因此，如果需要拷贝构造一个对象，编译器将生成拷贝构造函数，即下述代码将没有任何输出：

```cpp
#include "iostream"  
template<typename T>  
class SmartPointer{  
public:  
    template<typename U>  
    SmartPointer(const SmartPointer<U>& other)  
            :held_ptr(other.get()) {  
                std::cout << "Enter template copy constructor\n";  
            };  
    SmartPointer(T* p)  
            :held_ptr(p) {};  
    T* get() const {return held_ptr;};  
private:  
    T* held_ptr;  
};  
  
int main(){  
    SmartPointer<int> pint = {new int};  
    SmartPointer<int> pint2 = {pint}; // 调用默认拷贝构造函数，而非模板构造函数
}
```

### Item 46: Define non-member functions inside templates when type conversions are desired.

> ✦ When writing a class template that offers functions related to the template that support implicit type conversions on all parameters, define those functions as friends inside the class template.

在 [|Item 24](.md#item-24-declare-non-member-functions-when-type-conversions-should-apply-to-all-parameters.) 中，我们使用非成员函数来实现支持交换律的加法（自动类型转换），当我们将该技巧一应用到模板上时，发生了一些微妙的变化：

```cpp
template<typename T>
class Rational{
public:
	Rational(const T& numerator=0, const T& denominator=0);
	const T numerator() const;
	const T denominator() const;
	...
};

template<typename T>
const Rational<T> operator*(const Rational<T>& lhs, const Rational<T> rhs){...}

/*******************/

Rational<int> half(1, 2);
Rational<int> res = half*2; // error! won't compile!
```

怎会如此？！！原因在于，编译器要先对 `operator*` 进行实例化，但是，它不知道将该将 `T` 推导为哪个类型。`operator*` 接收了两个不同的参数类型，但在模板参数推导的过程中**隐式类型转换**不被考虑。

解决方案是，将 `operator*` 声明为友元函数，让其中模板类型参数随着类的实例化而一起实例化。需要注意的是，要在类中给出这个函数友元函数的定义，这个友元函数不是模板函数，在外部给出定义的函数是一个模板函数，**二者不是一个函数**：

```cpp
template<typename T>
class Rational{
public:
	Rational(const T& numerator=0, const T& denominator=0);
	const T numerator() const;
	const T denominator() const;

	friend const Rational operator*(const Rational& lhs, const Rational rhs) // 在模板类中可以简写，忽略尖括号中的内容
	...
};

template<typename T>
const Rational<T> operator*(const Rational<T>& lhs, const Rational<T> rhs){...} // 这是一个模板函数定义，类中声明的友元函数并非一个模板函数
```

### Item 47: Use traits classes for information about types.

> ✦ Traits classes make information about types available during com- pilation. They’re implemented using templates and template special- izations. 
> 
> ✦ In conjunction with overloading, traits classes make it possible to perform compile-time if...else tests on types.

我们来尝试实现 `advance` 模板函数，其作用是将一个指针或者迭代器移动指定距离。cpp 的所有迭代器中，有一部分支持随机访问，而另一部分仅支持连续访问。出于性能的考量，在实现 `advance` 时，我们需要分开实现这两种迭代器，即我们需要获知该迭代器的类型信息。由于我们还要支持对指针的操作，因此这一信息不应保存在迭代器的内部。这可如何是好？

好在，我们还有类型萃取 `traits`。`traits` 并非 cpp 中的关键字或者预定义的某个接口，它是一种技术的统称。鉴于类型信息不应保存在类型内部，标准做法是将其保存在一个模板，以及该模板的特化版本中。对于迭代器来说，标准库中的模板命名为 `iterator_traits`：

```cpp
template<typename iterT>
struct iterator_traits;
```

传统上，使用结构体来实现 `traits`。通过在结构体内声明一个名为 `iterator_category` 的 typedef，对于不同类型的 `iterT` 定义不同的值，来区分不同的迭代器类型。

具体来说，`iterator_traits` 由两部分组成。对于用户定义的迭代器，要求其必须内嵌一个名为 `iterator_category` 的 typedef，取值为标准库中的迭代器的分类 tag：

```cpp
template<...>
class deque{
public:
	class iterator{
	public:
		typedef random_access_iterator_tag iterator_category;
		...
	};
	...
};


/*******迭代器tag取值**************/
struct input_iterator_tag {};
struct output_iterator_tag {};
struct forward_iterator_tag : public input_iterator_tag {};
struct bidirectional_iterator_tag : public forward_iterator_tag {};
struct random_access_iterator_tag : public bidirectional_iterator_tag {};
```

例如，上文定义了一个双端队列中支持随机访问的迭代器。对于 `iterator_traits` 来说，其要做的就是将 `iterT` 中的 tag 再次声明为 `iterator_category`：

```cpp
template<typename iterT>
struct iterator_traits{
	typedef typename iterT::iterator_category iterator_category
};
```

至此，我们已经完成了对用户自定义类型支持。接下来我们支持对内建指针的支持。指针是一种支持随机访问的迭代器，使用部分模板特化对指针进行特化：

```cpp
template<typename T>
struct iterator_traits<T*>{
	typedef typename random_access_iterator_tag iterator_category;
};
```

当我们完成萃取类后，接下来就可以分类讨论来实现 `advance` 方法了：

```cpp
template <typename iterT, typename distT>
void advance(iterT& iter, distT d){
	if(typeid(typename iterator_traits<IterT>::iterator_category) == typeid(std::random_access_iterator_tag))
	...
}
```

遗憾的是，上面的代码存在编译错误，该问题将在下一个条款讨论。不但如此，上述代码的 `if` 语句应该在运行期执行，但事实上，条件语句在编译器就已经确定了结果，这降低了代码的执行效率。

编译器的条件语句？emmmm...似乎比较麻烦。别忘了我们还有函数重载！函数重载的就是根据不同的参数类型执行不同的代码！据此，我们可以重载不同迭代器类型对应的 `advance` 实现：

```cpp
template <typename iterT, typename distT>
void do_advance(iterT& iter, distT d, std::random_access_iterator_tag){
...
}

template <typename iterT, typename distT>
void do_advance(iterT& iter, distT d, std::input_iterator_tag){
...
}

...
```

有了以上代码，`advance` 函数仅需要调用他们即可。需要注意的是，在模板函数的重载中，可以有未命名形参。但是在调用函数的过程中，必须传入实参对象。好在，我们前面用来标识类型的标签是空的结构体，我们可以直接使用该结构体构造一个空对象作为形参传入：

```cpp
template <typename iterT, typename distT>
void advance(iterT& iter, distT d){
	do_advance(iter, d, typename std::iterator_traits<iter>::iterator_category);
}
```

以上我们总结出使用函数萃取类的过程：

- 创建一系列根据类型重载的“worker”函数。
- 创建一个“master”函数调用调用“worker”。

### Item 48: Be aware of template metaprogramming.

> ✦ Template metaprogramming can shift work from runtime to com- pile-time, thus enabling earlier error detection and higher runtime performance. 
> 
> ✦ TMP can be used to generate custom code based on combinations of policy choices, and it can also be used to avoid generating code in- appropriate for particular types.

模板元编程（TMP）就是书写在编译期运行的 cpp 代码的过程，其在编译期运行，输出结果再由编译器进行编译。

TMP 在上世纪九十年代被发现（⚠️不是发明），其有两个作用：

- 让某些不可能或者难以实现的事情变得可以实现；
- 将一些运行期的工作转移到编译期进行。  
第二个作用，可以把一些运行期的错误提前到编译期发现，并减小编译生成的可执行代码的文件大小，提高行效率。

前面我们在不使用函数重载实现 `advance` 的过程中，曾提到以下代码存在编译错误：

```cpp
template <typename iterT, typename distT>
void advance(iterT& iter, distT d){
	if(typeid(typename iterator_traits<IterT>::iterator_category) == typeid(std::random_access_iterator_tag))
	...
}
```

原因在于，如果我们传入一个不支持随机访问的迭代器，这个函数模板依旧会被完整展开，并且其中存在 `iter += d;` 这样的语句。尽管，该语句所在的 if 分支的条件永远为 `false`，这并不影响编译器对该语句进行编译检查。而不支持随机访问的迭代器并没有实现 `operator +=`，因此将会在编译期报错。

TMP 是图灵完备的，前一条款演示了在 TMP 中如何实现条件控制流。在 TMP 中，循环控制流则是通过递归来实现的，与常规 cpp 中递归调用函数不同，TMP 的递归是模板递归实例化。一个使用 TMP 编写的计算阶乘的代码为：

```cpp
template<unsigned n>
struct Factorial{
	enum {value = n*Factorial<n-1>::value};
};

template<>
struct Factorial<0>{
	enum {value = 1};
};
```

接下来作者举了 TMP 实际应用的几个领域，此处不在记录。总而言之，TMP 有其擅长的领域，但鉴于其反直觉的特性，以及作者写该书时相关工具链还很孱弱，需要谨慎使用。

## Customizing new and delete

现如今，很多语言都支持了自动垃圾回收。C++ 手动的回收方式似乎显得有些过时了。但是，许多系统的开发者选择 cpp，因为其允许他们手动管理内存。做到这一点，必须了解 cpp 中内存分配和释放例程的行为，这正是本章的重点内容。

在多线程环境下，内存管理的困难要大得多，因为堆和 new-handler 都是可修改的全局资源，容易受到竞争条件的影响。

### Item 49: Understand the behavior of the new-handler.、

> ✦ set_new_handler allows you to specify a function to be called when memory allocation requests cannot be satisfied. 
> 
> ✦ Nothrow new is of limited utility, because it applies only to memory allocation; associated constructor calls may still throw exceptions.

如果 `operator new` 无法分配足够内存，其将抛出异常（老版本将返回 `NULL`），但在此之前，其将调用一个名为 `new-handler` 的错误处理函数。标准库中提供了一个 `set_new_handler` 函数用于设置 `new-handler`：

```cpp
namespace std{
	typedef void (*new_handler)();
	new_handler set_new_handler(new_handler p) noexcept;
}
```

如上所示，`new_handler` 是一个输入参数和返回值均为空的函数指针类型，`set_new_handler` 接收这样一个指针，并将原处理函数指针返回。范例为：“

```cpp
void out_of_mem(){
	std::cerr << "Out of Mem\n";
	std::abort();
}

int main(){
	std::set_new_handler(out_of_mem);
	int *p = new int[100000000L]; // if fail, call out_of_mem and then abort
}
```

当 `new` 不能分配足够的内存时，其将不停调用 new-handler 直至有足够内存，或者停止。因此，new-handler 函数必须满足以下特性之一：

- 释放更多的内存空间。
- 设置另一个 new-handler 函数。
- 取消当前的 new-hander 函数。这将恢复 `new` 失败的默认行为，即抛出一个异常。
- 抛出异常。
- 不再返回，程序停止运行。

如果我们想为不同的类定制 new-handler，似乎也挺简单的：在每次 new 之前手动替换对应的 new-handler 函数。接下来，我们一起来尝试将这一理念付诸实践。

首先，既然要替换原有的 new-handler 函数，那必须有一个变量在类中记录对应的 new-handler。那自然也要提供一个设置 new-handler 的接口，用于保存原始和替换 new-handler。此外，还需要重载 `operator new`：

```cpp
class Widget{
public:
	static std::new_handler set_new_handler(std::new_hander p) noexcept;
	static void* operator new (std::size_t size);

private:
	static std::new_handler current_handler;
};
```

需要注意，静态成员变量需要在类的定义外部进行定义和初始化。重载的 `new` 应该做哪些事情呢？如下：

- 调用 `set_new_handler`，将 new-handler 设置为类提供的函数。
- 调用全局 `new` 实例化一个对象，如果失败，则应该恢复原始 new-handler 并抛出异常。为了确保其被正确恢复，应该使用资源管理类对 new-handler 进行管理。
- 如果 `new` 正常实例化了一个对象，则 new 应该返回对象指针。恢复 new-handler 的工作交由资源管理对象的析构函数负责。

首先来实现一个 RAII 资源管理类：

```cpp
class NewHandlerHolder{
public:
	explicit NewHandlerHolder(std::new_handler nh)
	:handler(nh){};
	~NewHandlerHolder(){
		std::set_new_handler(handler);
	}

private:
	std::new_handler handler;
	NewHandlerHolder(const NewHandlerHolder&);
	NewHandlerHolder& operator=(const NewHandlerHolder&); // 禁止拷贝构造和赋值
};
```

那么 `new` 可以重载为：

```cpp
void* operator new (std::size_t size){
	NewHandlerHolder h(std::set_new_handler(current_handler));
	return ::operator new(size);
};
```

看到这里，不禁感叹，真 TMD 优雅！屏住呼吸，还没结束呢！接下来，我们使用混合模式（Mixin-style），将其改造为模板类。详细来说，通过继承基类，派生类可以得到 `set_new_handler` 和 `operator new` 这俩成员，通过模板，则可以确保不同的类继承得到的静态成员是不同的。

```cpp
template <typename T>
class NewHandlerSupport{
public:
	static std::new_handler set_new_handler(std::new_hander p) noexcept;
	static void* operator new (std::size_t size);
private:
	static std::new_handler currentHandler;
};

template <typename T>
std::new_handler NewHandlerSupport<T>::set_new_handler(std::new_hander p) noexcept{
	std::new_handler oldHandler = currentHandler; 
	currentHandler = p; 
	return oldHandler;
}

template<typename T> 
void* NewHandlerSupport<T>::operator new(std::size_t size){ 
	NewHandlerHolder h(std::set_new_handler(currentHandler));
	return ::operator new(size);
}

template<typename T> 
std::new_handler NewHandlerSupport<T>::currentHandler = 0;
```

有了模板类，我们再实现 `Wiget` 就简单多了：

```cpp
class Widget: public NewHandlerSupport<Widget>{
...
};
```

值得注意的是，在模板类中，我们没有使用到参数类型 `T`，其存在的作用是为不同的类名，编译器都会创建一次代码副本，将他们的静态成员隔离开来。此外，我们的 `Widget` 继承了一个使用自己实例化的基类模板，这是合理的，这一技术的名字和它的行为一样古怪：奇异递归模板模式。

### Item 50: Understand when it makes sense to replace new and delete.

> ✦ There are many valid reasons for writing custom versions of new and delete, including improving performance, debugging heap usage er- rors, and collecting heap usage information.

为什么要替换编译器默认版本的 `new` 和 `delete` 运算符呢？一般来说，有以下三个理由：

- 检查使用错误。没有或者多次释放 new 的内存，都会引发错误。如果在 `new` 和 `delete` 中维护一张内存申请表，则可以检查出上述问题。又或者，还可以用于防止数据溢出，通过在内存的末尾写入一个签名，在 `new` 中检查该是否完好，可以判断是否出现了数据溢出写入。
- 提高效率。编译器自带的实现版本，需要兼容各种程序、各种内存大小的申请的情况，还要考虑内存碎片等等各种情况，通过自定义实现，可以避免这些开销。
- 收集使用数据。在研发阶段通过收集数据，可以分析出该程序使用动态内存的特点，并针对性进行优化。

接下来举个使用 `new` 检查内存是否存在溢出写入的例子：

```cpp
static const int signature = 0xDEADBEEF;
typedef unsigned char Byte;

void* operator new(std::size_t size){
	using namespace std;
	size_t real_size = size+2*sizeof(int);
	void *p_mem = malloc(real_size);
	if(!p_mem)
		throw bad_alloc();
	*(static_cast<int*>(pMem)) = signature;
	*(reinterpret_cast<int* >(static_cast<Byte* >(pMem)+realSize-sizeof(int))) = signature;
	return static_cast<Byte*>(pMem) + sizeof(int);
}
```

上述代码通过在申请的内存块两段放置额外的签名数据，以检测是否存在数据溢出写入的情况。当然，上述代码实际上存在很多问题。一方面，它不符合 cpp 关于 new 的规范，如果内存申请失败，应该循环调用 new-handler；另一方面，它没有考虑内存对齐的情况。

cpp 要求 `new` 返回的指针要满足内存对齐的要求，我们使用的 `malloc` 同样也是内存对齐的，但我们返回的偏移了一个 `int` 大小的指针，其不是对齐的。

内存对齐此类的小但确实重要的问题有很多很多，自定义一个完美的 `new` 的困难可见一斑。

### Item 51: Adhere to convention when writing new and delete.

> ✦ operator new should contain an infinite loop trying to allocate mem- ory, should call the new-handler if it can’t satisfy a memory request, and should handle requests for zero bytes. Class-specific versions should handle requests for larger blocks than expected. 
> 
> ✦ operator delete should do nothing if passed a pointer that is null. Class-specific versions should handle blocks that are larger than ex- pected.

这一条款将介绍在自定义 `new` 和 `delete` 时，需要遵守的几个规则。

首先是与 `new` 相关的几个要求：

- 返回正确的值。
- 当内存不足时，循环调用 new-handle 函数。
- 正确处理申请大小为 0 的情况。
- 避免遮蔽正常的 `new`。此要求将在下一条款讨论。

返回值听上去很简单，如果内存充足，则返回对应指针；否则，抛出异常。但也并非如此一蹴而就，如果内存不足，需要循环调用 new-handle 并再次申请内存，直至 new-handle 函数指针为空，抛出异常 `std::bad_alloc`。此外，cpp 规范还要求，即便申请了 0 字节大小的空间，也应该返回一个合法的指针。下面这段为代码，演示了一个 `new` 的行为：

```cpp
void *operator new(std::size_t size){
	using namespace std;
	if(size == 0){
		size = 1;
	}
	while(true){
	attemp to allocate size bytes;
	if(success)
		return pointer to mem;
	}
	new_handler global_handler = set_new_handler(0); // 获取new-handler
	set_new_handler(global_handler);
	if(global_handler){
		(*global_handler)();
	} else {
		throw std::bad::alloc();
	}
	
}
```

对于 0 字节，可以把它当作申请了一个字节来处理。由于没有获取 `new-handler` 的函数，因此只能通过手动将其设置为 `null` 然后再恢复的方法，获取 `new-handler` 函数指针。对于多线程的环境，可能需要上锁防止竞争。

通常，为某个类重写的 `new` 都是针对这个类大小的内存进行优化的版本，而不是用于其他类或者该类的派生类。然而，如果在派生类中没有重写 `new`，`new` 派生类对象时将调用基类中的 `new` 函数。为了防止此类问题，可以在先判断 `size == sizeof(Base)`，若不相等，调用全局 `new` 函数。

`delete` 函数就简单多了，唯一要注意的是：`delete` 要考虑指针为 NULL 的情况。

```cpp
void operator delete(void *rawMemory) noexcept{
	if(rawMemory == 0)
		return;
	归还已分配的内存;
}
```

成员函数的版本也简单，只要像前面的 `new` 一样，记得校验申请的内存大小是否与基类大小一致即可。

### Item 52: Write placement delete if you write placement new.

> ✦ When you write a placement version of operator new, be sure to write the corresponding placement version of operator delete. If you don’t, your program may experience subtle, intermittent memory leaks. 、
> 
> ✦ When you declare placement versions of new and delete, be sure not to unintentionally hide the normal versions of those functions.

`Widget *pw = new Widget;` 这样一句代码会执行两个操作：先调用 `operator new` 申请对应大小的内存，再调用构造函数进行初始化。如果在构造期间出错了，由于构造没有完成，用户得不到 `pw` 指针，因此用户无法对初始化失败的内存进行释放。为了防止内存泄露，该操作由编译器负责。

编译器负责释放内存时，其必须知道与申请内存 `new` 配套的 `delete` 函数是哪个。对于常见的只接受一个参数 `size_t size` 的 `new` 来说，其配套的 `delete` 也是如此。但是，有一类 `new` 可以接受不止一个参数，这类 `new` 我们称之为“placement new”，定位 new。

placement new 狭义上只得是 `void* operator new(std::size_t, void *pMemory)`，其接受一个额外的指针，表示在其指示的位置构造对象。广义上来说，所有参数列表不止是 `size_t size` 的 new 都可以被称为 placement new。狭义的含义更常见，通过语境很容易判断 placement new 的含义。

对于 placement new，如果其在构造的过程中出错了，运行时系统负责找到参数类型和数量一致的 placement delete 释放对应内存。如果找不到，则会导致内存泄露。

如果 new 对象的过程一切整成，那么使用 `delete` 删除时，将会调用非 placement 版本。这就意味着，当自定义 placement new 时，既要提供 placement delete 版本防止构造失败，也要提供默认 delete 版本用于正常销毁。

由于名称遮蔽的存在，如果在类中声明了一个成员 placement new，其将遮蔽默认 new。此外，还遮蔽了全局存在三个版本的 new：

```cpp
void* operator new(std::size_t);
void* operator new(std::size_t, void*);
void* operator new(std::size_t, const std::nothrow_t);
```

如果遮蔽之后想让他们仍然可用，记得让配套的 delete 也可用。

## Miscellany

这一章是杂项，胜利在望！

### Item 53: Pay attention to compiler warnings.

> ✦ Take compiler warnings seriously, and strive to compile warning- free at the maximum warning level supported by your compilers. 
> 
> ✦ Don’t become dependent on compiler warnings, because different compilers warn about different things. Porting to a new compiler may eliminate warning messages you’ve come to rely on.

大多数情况要，只要编译器没给出 error，程序都能跑起来。但在警告中，程序可能存在致命的错误，例如：

```cpp
class Base{
public:
	virtual void f() const;
};

class Derived: public Base{
public:
	virtual void f();
};
```

上述代码本意是在派生类中重新定义激烈函数 `f`，但遗漏了 `const` 修饰符，对编译器而言这意味着 `Derived::f` 遮蔽了 `Base::f`。

对于此类行为，编译器会给出警告，根据警告，很容易检查出相应的错误。

### Item 54: Familiarize yourself with the standard library, including TR1.

历史文件，现实意义尚不明确。跳过本条款。

### Item 55: Familiarize yourself with Boost.

不想学了，草草结束，这个坑以后再填！

## 终章

从 2024-04-17 到 2024-05-28，这本书耗费的时间比想象中多的多得多。anyway，收货还是颇丰的。阅读过程中时不时地会发出感叹：这也太细/优雅/牛逼了，作者很喜欢埋一些伏笔，读到后面恍然大悟，知识都串起来了。

作为第一本 cpp 深入的书籍，不错不错。缺点是对现代 cpp 的涉猎太少了，好在作者还有一本《Effective Mordern cpp》，安排上！

完结，撒花🎉