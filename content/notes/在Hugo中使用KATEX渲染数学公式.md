---
title: 在Hugo中使用KATEX渲染数学公式
tags:
  - Hugo
  - KATEX
date: 2024-06-05T15:35:00+08:00
lastmod: 2024-06-05T16:56:00+08:00
publish: true
dir: notes
slug: using katex to render math in hugo
---

# 前言

在博文中插入公式是个挺常见的需求，不知道为啥 Hugo 对于公式渲染没有原生支持😞。网络上能找到两种解决方案：KATEX 和 MathJax，据说前者性能更好一点。本博客使用 KATEX 进行渲染。

网络上相关资料挺多，但大多浅尝辄止，我在将其整合进 Obsidian 的过程中遇到了不少错误，折腾了一个下午 + 一个晚上，目前终于跑通能用了。demo 参考博文：[《CMU 10-414 deep learning system》学习笔记 | 周鑫的个人博客](https://www.zhouxin.space/notes/notes-on-cmu-10-414-deep-learning-system/)，其中含有大量公式。

# 技术方案

目前含有数学公式的工作流为：  
Obsidian 编辑博文 -> Obsidian github publisher 插件进行正则替换 -> Obsidian github publisher 上传到 github -> 服务器进行部署

## 引入 KATEX 样式表和 JS 文件

为了在博文中渲染公式，需要引入 KATEX 的样式表 [^1]，具体来说，在 `<your_hugo_site>/layouts/partials/` 文件夹下创建一个 `math.html` 文件，并写入以下内容：

```html
<link
    rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css" 
    integrity="sha384-wcIxkf4k558AjM3Yz3BBFQUbk/zgIYC2R0QpeeYb+TwlBVMrlgLqwRjRtGZiK7ww" 
    crossorigin="anonymous"
/>

<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js" integrity="sha384-hIoBPJpTUs74ddyc4bFZSM1TVlQDA60VBbJS0oA934VSz82sBx1X7kSx2ATBDIyd" crossorigin="anonymous"></script>

<!-- To automatically render math in text elements, include the auto-render extension: -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js" integrity="sha384-43gviWU0YVjaDtb/GhzOouOXtZMP/7XUzwPTstBeZFe/+rCMvRwr4yROQP43s0Xk" crossorigin="anonymous"
    onload="
    window.addEventListener('DOMContentLoaded', function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\$$', right: '\\\\$$', display: false},
                {left: '\\$$', right: '\\\\$$', display: true}
            ]
        });
    });
"></script>
```

然后在 `<your_hugo_site>/layouts/partials/extend_head.html` 文件内追加以下内容：

```html
{{ if or .Params.math .Site.Params.math }}
{{ partial "math.html" . }}
{{ end }}
```

上述代码的含义是：如果当前页面 `math` 属性或者全局 `math` 属性为真，则将我们之前写入的 `math.html` 模板文件包含至每个网页页面的 head 部分。

我们可以只将需要渲染数学公式的博文的 metadata 区域 `math` 字段设置为真，以引入 KATEX 相关文件，防止不必要的性能开销。

至此，理论上来说，含有数学公式的博文已经能被正确渲染了，许多教程也到此结束了。但是我碰到了公式没能被正确渲染的情况，如下图所示：  
![红框没公式渲染出错](https://pics.zhouxin.space/202406051638640.png?x-oss-process=image/quality,q_90/format,webp)

## 与 Obsidian 整合

上图中公式渲染出错的情况，有以下两个原因：

- Markdown 语法和 KATEX 语法冲突，包括但不限于：符号转义、下划线含义冲突等
- 公式块和公式内容之间存在额外空格

第一个问题可以通过使用 `div` 块包围公式来解决，hugo 不会对 `div` 块内的代码进行二次转义。第二个问题可以通过正则表达式替换来解决。

事实上，第一个问题也是正则所擅长的领域，通过一次正则替换，就可以具体将 md 中的公式块使用 `div` 包裹，并且移除额外的空格。具体来说，需要将以下的 md 文档：

```md
梯度下降，就是沿着梯度方向不断进行迭代，以求找到最佳的$\theta$使得目标函数值最小。
$$

\theta :=\theta _0-\alpha \nabla f\left( \theta _0 \right)

$$
上式中，$\alpha$被称为学习率或者步长。
```

替换为：

```md
梯度下降，就是沿着梯度方向不断进行迭代，以求找到最佳的$\theta$使得目标函数值最小。
<div>$$
\theta :=\theta _0-\alpha \nabla f\left( \theta _0 \right)
$$</div>
上式中，$\alpha$被称为学习率或者步长。
```

相应的模式串为 `/\$\$(\s*)([\s\S]*?)(\s*)\$\$/gs`，替换串为 `<div>$$$$\n$2\n$$$$</div>`。使用 github publisher 插件进行替换即可。

# One More Thing

推荐两个网站，分别用于 KATEX 和正则表达式的 debug：

- [KaTeX – The fastest math typesetting library for the web](https://katex.org/)
- [regex101: build, test, and debug regex](https://regex101.com/)

# 参考文档

[^1]: [Browser · KaTeX](https://katex.org/docs/browser)