---
title: 百度飞桨「启航计划」小结——CINN后端Pass改造
tags: 
date: 2025-01-08T00:07:00+08:00
lastmod: 2025-01-08T15:13:00+08:00
publish: true
dir: thoughts
slug: baidu paddlepaddle starter plan summary
---

在过去八周时间里，我参加了由飞桨开源社区组织的 [飞桨启航计划集训营（第四期）](https://github.com/PaddlePaddle/Paddle/issues/69152)，认领并完成 [【开源任务】CINN编译器后端Pass改造](https://github.com/PaddlePaddle/Paddle/issues/69639) 系列任务。趁最近在准备期末考试，除了复习干啥都有意思，好好总结一下在启航里的收获。（逃 🤐

# Why 启航？

为什么选择了启航计划？在回答这个问题之前，先介绍一下背景：当时学习了 CMU 10414 DLSys 课程，准备学习 TVM 或者 MLIR，但相关基础欠缺，一直苦于找不到切入口。在互联网上🏄‍♀️的时候无意中发现了启航计划，了解到其对新手相当友好：没有面试筛选、任务比较简单、有专门答疑研发老师，当时第三期正在进行，遂订阅了第三期的 ISSUE，蹲第四期的活动。

# 启航计划安排

刚开始有三个打卡任务，分别是编译 Paddle、跑通 Paddle Mix 和 为 Paddle 添加文档。第一个任务用来熟悉本地编译 Paddle 和单测，第三个任务用来熟悉 GitHub 工作流程。

理论上，完成这三个任务就能够达到最低结营条件，但我们参加这个活动肯定不是为了这张结营证书，而是想要提升自己的。这三个任务对于提升自己的作用聊胜于无。下一步，就可以选择几个的专项团，尝试一些低星任务。

由于启航计划面向新手，任务比较简单。低星任务基本是照葫芦画瓢，即照着样例基本就能完成，通过低星任务可以理解这个专项团的总体目标。高星任务则是一些推广，或者逻辑比较复杂，但也基本不涉及从 0 到 1 的创作，本质上还是模仿。

当然，任务简单并不意味着可以很轻松地完成。对于我们这种零经验的开发者来说，极大概率需要花上几天时间才能理解“1+1=2”，后期还会发现理解是不完备的或者根本就是错的😭。在完成的过程中，可以反复阅读任务文档和观看任务讲解视频，多与导师沟通，很多时候他们都能一语点醒梦中人。特别感谢 [Hongqing-work](https://github.com/Hongqing-work) 老师，CINN Pass 改造基本都是在向她请教，老师周末和晚上都能不厌其烦地答疑解惑，太感动了😭。

在训练营中，每两周都需要提交周报。这既是一个让我们回顾过去两周产出、规划未来的好机会，也能够了解其他同学的进度，保证自己不掉队。按照我的经验，1-2 周用于完成打卡任务，开始尝试低星任务；3-4 周继续完成某个专项团的任务，此时已经可以冲击一些高星任务了；5-8 周，渐臻佳境，对于某个专项团的任务已经能够做到游刃有余，并且尝试其它专项团任务。

# CINN 后端 Pass 改造

在本次启航计划中，我一共完成 7 个 CINN 后端 Pass 改造任务。这里介绍一下这个专项团的收获。

## 背景

本次任务的背景是 CINN 升级了后端 IR 表示，将原来 Func-Expr 层级结构中的 Expr 进行了细化，重新划分为 Func-Block-Stmt-Expr，重新划分后的 IR 层次更加清晰。

与之对应地，后端 Pass 也被划分为 FuncPass、BlockPass、StmtPass、ExprPass 四个级别，使用配套的 PassManager 应用 Pass。其层次结构为：  
![新 IR 层次结构  图源：https://github.com/PaddlePaddle/Paddle/issues/69639](https://pics.zhouxin.space/202501081247817.webp)

此外，还提供了 IR 访问方法：

1. 类型不敏感的 Stmt 和 Block 级别的访问/修改方法，在遍历 Stmt 前后将会调用用户传入的回调方法：

```cpp
// Visitors
void Visit(const BlockRef &block,
           const std::function<void(const StmtRef &)> &pre_callback,
           const std::function<void(const StmtRef &)> &post_callback);

void Visit(const StmtRef &stmt,
           const std::function<void(const StmtRef &)> &pre_callback,
           const std::function<void(const StmtRef &)> &post_callback);
// Mutators
// ...
```

2. 类型敏感的 Stmt 和 Block 定制化访问模板类，用户可以通过重写 `virtual StmtRetTy VisitStmt(const StmtRef &stmt, Args... args)` 定制化访问不同的 Stmt：

```cpp
template <typename StmtRetTy = void,
          typename BlockRetTy = void,
          typename... Args>
class StmtVisitor {
 public:
  virtual StmtRetTy VisitStmt(const StmtRef &stmt, Args... args) {
    CINN_CHECK_STMT_DEFINED(stmt)
    switch (stmt->stmt_type()) {
#define __(stmt__)                                \
  case ir::StmtNodeTy::stmt__:                    \
    return VisitStmt(stmt.as<stmt__>(), args...); \
    break;

      NODETY_FORALL_STMT(__)

      default:
        PADDLE_THROW(::common::errors::InvalidArgument(
            "Deadcode, not supported StmtNodeTy"));
#undef __
    }
  }
...
```

## 为什么要升级 IR？

从后端 Pass 的角度来看，IR 升级主要有两个好处：1. Pass 编写更加清晰和规范；2. Pass 便于管理。

旧 IR 下的的 Pass 大都通过继承 IRMutator/Visitor 在遍历整个 IR 的过程中修改来实现 Pass 的功能，但实际上其只需要针对某个特定类型的 Stmt/Block 处理即可。旧 IR 下的 IRMutator 为了便于开发者使用，提供了对各种类型的 Expr/Stmt/Block 默认遍历，例如对于 IfThenElse 默认实现版本会遍历条件和两个分支：

```cpp
template <typename T>
void IRMutator<T>::Visit(const IfThenElse *expr, T op) {
  auto *node = op->template As<IfThenElse>();
  IRVisitorRequireReImpl<void, T>::Visit(&node->condition, &node->condition);
  IRVisitorRequireReImpl<void, T>::Visit(&node->true_case, &node->true_case);
  if (node->false_case.defined())
    IRVisitorRequireReImpl<void, T>::Visit(&node->false_case,
                                           &node->false_case);
}
```

这种默认实现在很多情况下是不必要的，比如在合并两个相同的 If 中，显然不需要对条件应用此 Pass，也不需要对 Expr 级别的表达式进行访问。

理论上说，开发者可以通过重写对应的 Visit 方法来及时进行截断，但一方面这样会使得 Pass 的代码比较臃肿，另一方面 Pass 在开发时并没有此规范，已经成为遗留问题。

在此次 IR 和 Pass 改造后，原有的 IRMutator 将只保留对于 Expr 级别的访问逻辑，对于 Stmt 和 Block 级别的遍历由 PassManager 完成。例如，StmtPassManager 将会遍历这个函数，并为每一条 Stmt 调用一次其管理的 StmtPass，而在 StmtPass 内部，其只需要处理符合其目标的逻辑。

此外，新版的 StmtVisitor 没有提供 `VisitStmt` 默认实现，这可以强迫开发者自定义遍历逻辑，并及时截断不需要的遍历。

## Pass 编写范式

升级后的 IR 的编写范式一般为：1. 继承对应级别的 Pass 基类；2. 使用一个内部类对 Func/Block/Stmt 进行遍历实现核心功能，这个类可以继承 StmtMutator/IRMutator 或者调用 Visit/Mutate 方法来实现遍历；3. 返回 Success。

1. 继承对应级别的 Pass 基类  
第一步就是分析原 Pass 是什么级别 Pass，核心要义是抓住原 Pass 需要什么级别的信息以及是什么级别的修改。例如：
- [IfFusionPass](https://github.com/PaddlePaddle/Paddle/pull/69611) 是合并两个多个条件相同的 If，其要识别和删除多个 If，只有拿到这个 If 所在的 Block 能够实现多个语句的识别和单个语句的删除，这是一个 Block 级别的 Pass；
- [RearrangeLoadInstruction](https://github.com/PaddlePaddle/Paddle/pull/70437) 是将一个函数内的 Load 放到最前面执行，以提高指令的并行并行程度，为了确保 Load 是**一个函数内**最先执行的语句，以及对**函数内所有**相同的 Load 替换为本地变量，这是一个 Func 级别的 Pass；
- [EliminateCommonFactorOfLocalIndex](https://github.com/PaddlePaddle/Paddle/pull/70619/files) 需要获取当前 For 的**嵌套信息**，那必须由本 Pass 负责对 IR 的遍历，否则无法获取当前 For 的嵌套级别信息，因此这是一个 Func 级别的 Pass。

我的经验是：如果一个 Pass 仅仅需要当前 Stmt 的内部信息、不需要删除或者替换当前 Stmt、并且对于当前 Stmt 的嵌套级别没有要求（例如不要求当前的 For 是最外层/最内层的 For），那么其是一个 Stmt 级别的 Pass；如果一个 Pass 需要跨语句的信息，或者需要删除/替换/添加一条 Stmt，那么其是一个 BlockPass；如果一个 Pass 需要自己控制对 IR 的遍历过程，或者需要当前的嵌套上下文，那么这是一个 Func 级别的 Pass。

2. 编写实现类  
一些比较简单的 Pass 就是一个继承了 IRMutator 的实现类，此类 Pass 一般只需要额外继承 StmtMutator，如果不涉及 Expr 层面，则去掉对于 IRMutator，然后将原有逻辑迁移到新 IR 下即可。可参考 [RemoveScheduleBlock](https://github.com/PaddlePaddle/Paddle/pull/70334)。新 IR 下，很多变量都被设置为私有变量，必须通过 getter 和 setter 进行读写。

一些比较复杂的 Pass 可能有多个 Mutator 对 IR 进行多次访问，一般第一次是收集全局信息，之后再进行修改。读懂源码后再照葫芦画瓢修改即可。

更复杂的是调用了一些旧 IR 的方法，例如 `ir::ir_utils::CollectIRNodesWithoutTensor`，这种情况下可以判断一下传入的参数是否是 Expr，如果是 Expr 则还可以调用该方法（因为对 Expr 是封闭的，Expr 中不会有 Stmt 或者 Block），否则要根据这些方法的逻辑在新 IR 下进行实现。

3. 返回 Success  
这个没啥好说，返回 `LogicalResult::success()` 即可。

## Tips

1. Pass 应该实现为无状态的  
无状态指的是 Pass 不应该依赖之前的信息，或者记录一些持久信息。例如，一个对于 For 进行处理的 Pass，其内部不应该记录当前 For 的名字以防止重复。如果想要避免重复访问，可以将其实现为 FuncPass 手动处理遍历逻辑。

2. PassManager 是按照 DFS 后序遍历的  
这一遍历顺序可以保证最内部的语句被最先访问。Pass 改造过程中是可以依赖这一行为的。

3. Pass 之间缺乏通信机制  
Pass 之间是缺乏通信机制的，一些 Pass 在应用是前是需要检查能否进行变换的，这些检查 Pass 可以作为变换 Pass 的内部的一部分，在变换 Pass 实例化一个 PassManager 应用检查的 Pass。

4. 可参考 [Halide文档](https://halide-lang.org/docs/)  
CINN 在很多设计上参考了 Halide 和 TVM，在如果碰到一些例如不知道 Stmt 的作用的疑问，可以参考这两个这两个文档更加丰富的社区，往往会有惊喜收获。

# 后记

作为第一次开源活动经历，我个人觉得还是收获颇丰的。纸上得来终觉浅，绝知此事要躬行，很多之前没有实操的技术都在这次活动中得到了锻炼，例如 Git 和 GitHub 的工作流、VSCode 和 CMake 的配套、GLOG 的使用等等，以及对于 CINN 中 Pass 改造的经验，更是很好的学习 AI Sys 的切入口。

鼓励没有尝试过的同学多多参加这类活动，一定能不虚此行！