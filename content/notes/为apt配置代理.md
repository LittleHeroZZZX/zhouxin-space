---
title: 为apt配置代理
tags:
  - Ubuntu
date: 2024-04-01T10:50:00+08:00
lastmod: 2024-04-12T11:41:00+08:00
publish: true
dir: notes
slug: config proxy for apt
---

一般来说，apt 通过换源即可获得不错的体验，但有的时候不得不加入一些没被镜像的国外源例如 `PPA`，因此不得不琢磨怎么在 apt 中配置代理。  
apt 不会从环境变量获取代理配置，需要手动其配置文件 `/etc/apt/apt.conf` 中添加：

``` bash
# 配置格式
Acquire::http::Proxy "http://USERNAME:PASSWORD@SERVER:PORT";
Acquire::https::Proxy "https://USERNAME:PASSWORD@SERVER:PORT";
```

例如，对于不需要认证的代理，在 `/etc/apt/apt.conf` 添加以下内容：

``` bash
Acquire::http::Proxy "http://127.0.0.1:7890";
Acquire::https::Proxy "http://127.0.0.1:7890";  
```

# 参考文档

- [Configure proxy for APT? - Ask Ubuntu](https://askubuntu.com/a/920242)