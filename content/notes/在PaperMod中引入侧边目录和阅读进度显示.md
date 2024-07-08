---
title: 在PaperMod中引入侧边目录和阅读进度显示
tags:
  - 博客搭建
date: 2024-07-08T20:04:00+08:00
lastmod: 2024-07-08T21:40:00+08:00
publish: true
dir: notes
slug: introduce side toc and reading percentage to papermod
---

# 概述

在 PaperMod 中，目录的默认行为是在文章前展示，在阅读过程中无法利用其帮助定位或者精确跳转到某一部分，侧边目录能够很好解决上述痛点。此外，阅读进度百分比也能够帮助读者定位阅读位置，还能让网页显得更灵动一点。

实现方案主要借鉴自 [Sulv's Blog](https://www.sulvblog.cn/)，其中侧边目录其博文 [^1] 介绍的方法对长目录支持不太友好，不会自动滚动到正在阅读的内容，本文对此进行了改进。百分比显示实现的方案来自其博客的 [源码](https://github.com/xyming108/sulv-hugo-papermod)。

实现效果如下图所示：

![image.png](https://pics.zhouxin.space/202407082117402.png?x-oss-process=image/quality,q_90/format,webp)

# 步骤

## 侧边目录

在 PaperMod 中，目录相关的 html 代码定义在 `layouts/partials/toc.html` 中，为了修改它，只要创建一个 `<your_hugo_site>/layouts/partials/toc.html` 覆盖即可，在其中粘贴如下代码：

```html
{{- $headers := findRE "<h[1-6].*?>(.|\n])+?</h[1-6]>" .Content -}}
{{- $has_headers := ge (len $headers) 1 -}}
{{- if $has_headers -}}
<aside id="toc-container" class="toc-container wide">
    <div class="toc">
        <details {{if (.Param "TocOpen") }} open{{ end }}>
            <summary accesskey="c" title="(Alt + C)">
                <span class="details">{{- i18n "toc" | default "Table of Contents" }}</span>
            </summary>

            <div class="inner">
                {{- $largest := 6 -}}
                {{- range $headers -}}
                {{- $headerLevel := index (findRE "[1-6]" . 1) 0 -}}
                {{- $headerLevel := len (seq $headerLevel) -}}
                {{- if lt $headerLevel $largest -}}
                {{- $largest = $headerLevel -}}
                {{- end -}}
                {{- end -}}

                {{- $firstHeaderLevel := len (seq (index (findRE "[1-6]" (index $headers 0) 1) 0)) -}}

                {{- $.Scratch.Set "bareul" slice -}}
                <ul>
                    {{- range seq (sub $firstHeaderLevel $largest) -}}
                    <ul>
                        {{- $.Scratch.Add "bareul" (sub (add $largest .) 1) -}}
                        {{- end -}}
                        {{- range $i, $header := $headers -}}
                        {{- $headerLevel := index (findRE "[1-6]" . 1) 0 -}}
                        {{- $headerLevel := len (seq $headerLevel) -}}

                        {{/* get id="xyz" */}}
                        {{- $id := index (findRE "(id=\"(.*?)\")" $header 9) 0 }}

                        {{- /* strip id="" to leave xyz, no way to get regex capturing groups in hugo */ -}}
                        {{- $cleanedID := replace (replace $id "id=\"" "") "\"" "" }}
                        {{- $header := replaceRE "<h[1-6].*?>((.|\n])+?)</h[1-6]>" "$1" $header -}}

                        {{- if ne $i 0 -}}
                        {{- $prevHeaderLevel := index (findRE "[1-6]" (index $headers (sub $i 1)) 1) 0 -}}
                        {{- $prevHeaderLevel := len (seq $prevHeaderLevel) -}}
                        {{- if gt $headerLevel $prevHeaderLevel -}}
                        {{- range seq $prevHeaderLevel (sub $headerLevel 1) -}}
                        <ul>
                            {{/* the first should not be recorded */}}
                            {{- if ne $prevHeaderLevel . -}}
                            {{- $.Scratch.Add "bareul" . -}}
                            {{- end -}}
                            {{- end -}}
                            {{- else -}}
                            </li>
                            {{- if lt $headerLevel $prevHeaderLevel -}}
                            {{- range seq (sub $prevHeaderLevel 1) -1 $headerLevel -}}
                            {{- if in ($.Scratch.Get "bareul") . -}}
                        </ul>
                        {{/* manually do pop item */}}
                        {{- $tmp := $.Scratch.Get "bareul" -}}
                        {{- $.Scratch.Delete "bareul" -}}
                        {{- $.Scratch.Set "bareul" slice}}
                        {{- range seq (sub (len $tmp) 1) -}}
                        {{- $.Scratch.Add "bareul" (index $tmp (sub . 1)) -}}
                        {{- end -}}
                        {{- else -}}
                    </ul>
                    </li>
                    {{- end -}}
                    {{- end -}}
                    {{- end -}}
                    {{- end }}
                    <li>
                        <a href="#{{- $cleanedID -}}" aria-label="{{- $header | plainify -}}">{{- $header | safeHTML -}}</a>
                        {{- else }}
                    <li>
                        <a href="#{{- $cleanedID -}}" aria-label="{{- $header | plainify -}}">{{- $header | safeHTML -}}</a>
                        {{- end -}}
                        {{- end -}}
                        <!-- {{- $firstHeaderLevel := len (seq (index (findRE "[1-6]" (index $headers 0) 1) 0)) -}} -->
                        {{- $firstHeaderLevel := $largest }}
                        {{- $lastHeaderLevel := len (seq (index (findRE "[1-6]" (index $headers (sub (len $headers) 1)) 1) 0)) }}
                    </li>
                    {{- range seq (sub $lastHeaderLevel $firstHeaderLevel) -}}
                    {{- if in ($.Scratch.Get "bareul") (add . $firstHeaderLevel) }}
                </ul>
                {{- else }}
                </ul>
                </li>
                {{- end -}}
                {{- end }}
                </ul>
            </div>
        </details>
    </div>
</aside>
<script>
    let activeElement;
    let elements;
    
    document.addEventListener('DOMContentLoaded', function (event) {
        checkTocPosition();
    
        elements = document.querySelectorAll('h1[id],h2[id],h3[id],h4[id],h5[id],h6[id]');
        if (elements.length > 0) {
            // Make the first header active
            activeElement = elements[0];
            const id = encodeURI(activeElement.getAttribute('id')).toLowerCase();
            document.querySelector(`.inner ul li a[href="#${id}"]`).classList.add('active');
        }
    
        // Add event listener for the "back to top" link
        const topLink = document.getElementById('top-link');
        if (topLink) {
            topLink.addEventListener('click', (event) => {
                // Prevent the default action
                event.preventDefault();
    
                // Smooth scroll to the top
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        }
    }, false);
    
    window.addEventListener('resize', function(event) {
        checkTocPosition();
    }, false);
    
    window.addEventListener('scroll', () => {
        // Get the current scroll position
        const scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
    
        // Check if the scroll position is at the top of the page
        if (scrollPosition === 0) {
            return;
        }
    
        // Ensure elements is a valid NodeList
        if (elements && elements.length > 0) {
            // Check if there is an object in the top half of the screen or keep the last item active
            activeElement = Array.from(elements).find((element) => {
                if ((getOffsetTop(element) - scrollPosition) > 0 && 
                    (getOffsetTop(element) - scrollPosition) < window.innerHeight / 2) {
                    return element;
                }
            }) || activeElement;
    
            elements.forEach(element => {
                const id = encodeURI(element.getAttribute('id')).toLowerCase();
                const tocLink = document.querySelector(`.inner ul li a[href="#${id}"]`);
                if (element === activeElement){
                    tocLink.classList.add('active');
    
                    // Ensure the active element is in view within the .inner container
                    const tocContainer = document.querySelector('.toc .inner');
                    const linkOffsetTop = tocLink.offsetTop;
                    const containerHeight = tocContainer.clientHeight;
                    const linkHeight = tocLink.clientHeight;
    
                    // Calculate the scroll position to center the active link
                    const scrollPosition = linkOffsetTop - (containerHeight / 2) + (linkHeight / 2);
                    tocContainer.scrollTo({ top: scrollPosition, behavior: 'smooth' });
                } else {
                    tocLink.classList.remove('active');
                }
            });
        }
    }, false);
    
    const main = parseInt(getComputedStyle(document.body).getPropertyValue('--article-width'), 10);
    const toc = parseInt(getComputedStyle(document.body).getPropertyValue('--toc-width'), 10);
    const gap = parseInt(getComputedStyle(document.body).getPropertyValue('--gap'), 10);
    
    function checkTocPosition() {
        const width = document.body.scrollWidth;
    
        if (width - main - (toc * 2) - (gap * 4) > 0) {
            document.getElementById("toc-container").classList.add("wide");
        } else {
            document.getElementById("toc-container").classList.remove("wide");
        }
    }
    
    function getOffsetTop(element) {
        if (!element.getClientRects().length) {
            return 0;
        }
        let rect = element.getBoundingClientRect();
        let win = element.ownerDocument.defaultView;
        return rect.top + win.pageYOffset;   
    }
    
</script>
{{- end }}
```

其中，后半部分为 js 代码，根据阅读内容滚动并加粗相应标题就由其实现。

然后，添加 css 样式的代码，创建 `<your_hugo_site>/assets/css/extended/toc.css` 文件，并拷贝以下内容：

```css
:root {
    --nav-width: 1380px;
    --article-width: 650px;
    --toc-width: 300px;
}

.toc {
    margin: 0 2px 40px 2px;
    border: 1px solid var(--border);
    background: var(--entry);
    border-radius: var(--radius);
    padding: 0.4em;
}

.toc-container.wide {
    position: absolute;
    height: 100%;
    border-right: 1px solid var(--border);
    left: calc((var(--toc-width) + var(--gap)) * -1);
    top: calc(var(--gap) * 2);
    width: var(--toc-width);
}

.wide .toc {
    position: sticky;
    top: var(--gap);
    border: unset;
    background: unset;
    border-radius: unset;
    width: 100%;
    margin: 0 2px 40px 2px;
}

.toc details summary {
    cursor: zoom-in;
    margin-inline-start: 20px;
    padding: 12px 0;
}

.toc details[open] summary {
    font-weight: 500;
}

.toc-container.wide .toc .inner {
    margin: 0;
}

.active {
    font-size: 110%;
    font-weight: 600;
}

.toc ul {
    list-style-type: circle;
}

.toc .inner {
    margin: 0 0 0 20px;
    padding: 0px 15px 15px 20px;
    font-size: 16px;

    /*目录显示高度*/
    max-height: 83vh;
    overflow-y: auto;
}

.toc .inner::-webkit-scrollbar-thumb {  /*滚动条*/
    background: var(--border);
    border: 7px solid var(--theme);
    border-radius: var(--radius);
}

.toc li ul {
    margin-inline-start: calc(var(--gap) * 0.5);
    list-style-type: none;
}

.toc li {
    list-style: none;
    font-size: 0.95rem;
    padding-bottom: 5px;
}

.toc li a:hover {
    color: var(--secondary);
}

```

到此为止，目录应该就能在侧边正确显示了🎉🎉。

## 阅读百分比

阅读百分比实现的核心思想就是每当发生滚动事件时，根据滚动条高度计算当前阅读进度。这里我们将进度的数字显示在 TOP 按钮上，TOP 按钮定义在 `footer.html` 中，因此我们要创建 `<your_hugo_site>/layouts/partials/footer.html`，将主题中对应位置的 `footer.html` 内容拷贝进去，然后修改 TOP 按钮相关的代码，原代码为：

```html
{{- if (not site.Params.disableScrollToTop) }}
<a href="#top" aria-label="go to top" title="Go to Top (Alt + G)" class="top-link" id="top-link" accesskey="g">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 6" fill="currentColor">
        <path d="M12 6H0l6-6z" />
    </svg>
</a>
{{- end }}
```

我们要在其中添加一个用于展示进度的 `span` 和更新进度的 js 代码，即修改为：

```html
{{- if (not .Site.Params.disableScrollToTop) }}
<a href="#top" aria-label="go to top" title="Go to Top (Alt + G)" class="top-link" id="top-link" accesskey="g">
    <span class="topInner">
        <svg class="topSvg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 6" fill="currentColor">
            <path d="M12 6H0l6-6z"/>
        </svg>
        <span id="read_progress"></span>
    </span>
</a>

<script>
    document.addEventListener('scroll', function (e) {
        const readProgress = document.getElementById("read_progress");
        const scrollHeight = document.documentElement.scrollHeight;
        const clientHeight = document.documentElement.clientHeight;
        const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
        readProgress.innerText = ((scrollTop / (scrollHeight - clientHeight)).toFixed(2) * 100).toFixed(0);
    })
</script>
{{- end }}
```

然后添加相关 css 代码，即创建 `<your_hugo_site>/assets/css/extended/top.css` 文件，并拷贝以下内容：

```css
/*top*/
.topInner {
    display: grid;
    align-items: baseline;
    justify-items: center;
    margin: 7px;
    font-weight: 900;
}

.topSvg {
    width: 20px;
}

.top-link {
    padding: unset;
}

/*到顶部*/
.top-link {
    background: var(--entry);

    -webkit-transition: box-shadow 0.4s ease, transform 0.4s ease;
    -moz-transition: box-shadow 0.4s ease, transform 0.4s ease;
    -o-transition: box-shadow 0.4s ease, transform 0.4s ease;

    transition: box-shadow 0.4s ease, transform 0.4s ease;
    box-shadow: 0px 2px 4px rgb(5 10 15 / 5%), 0px 7px 13px -3px rgb(5 10 15 / 30%);
}

.dark .top-link {
    background: var(--entry);

    -webkit-transition: box-shadow 0.4s ease, transform 0.4s ease;
    -moz-transition: box-shadow 0.4s ease, transform 0.4s ease;
    -o-transition: box-shadow 0.4s ease, transform 0.4s ease;

    transition: box-shadow 0.4s ease, transform 0.4s ease;
    box-shadow: 0px 2px 4px rgb(5 10 15 / 5%), 0px 7px 13px -3px rgb(5 10 15 / 30%);
}

.top-link:hover {
    color: rgb(108, 108, 108);

    /*-webkit-transform: scale(1.1);*/
    /*-moz-transform: scale(1.1);*/
    /*-ms-transform: scale(1.1);*/
    /*-o-transform: scale(1.1);*/
    /*transform: scale(1.1);*/

    transition: box-shadow 0.4s ease, transform 0.4s ease;
    box-shadow: 0px 4px 8px rgb(5 10 15 / 5%), 0px 7px 13px -3px rgb(5 10 15 / 30%);
}

.dark .top-link:hover {
    color: rgba(180, 181, 182, .8);

    /*-webkit-transform: scale(1.1);*/
    /*-moz-transform: scale(1.1);*/
    /*-ms-transform: scale(1.1);*/
    /*-o-transform: scale(1.1);*/

    /*transform: scale(1.1);*/

    transition: box-shadow 0.4s ease, transform 0.4s ease;
    box-shadow: 0px 4px 8px rgb(5 10 15 / 5%), 0px 7px 13px -3px rgb(5 10 15 / 30%);
}
```

到此为止，阅读进度应该就能在 TOP 按钮上正确显示了🎉🎉。

# 参考文档

[^1]: [Hugo博客目录放在侧边 | PaperMod主题 | Sulv's Blog](https://www.sulvblog.cn/posts/blog/hugo_toc_side/)