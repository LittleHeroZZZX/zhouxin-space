---
title: Effective Cpp 第三版学习笔记
tags:
  - Cpp
date: 2024-04-17T18:23:00+08:00
lastmod: 2024-04-23T16:12:00+08:00
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

# 参考文档