---
title: 创建基于阿里云OSS的图床
tags:
  - 图床
  - OSS
  - obsidian
  - PicGo
  - 博客搭建
date: 2024-04-01T23:10:00+08:00
lastmod: 2024-04-12T11:42:00+08:00
publish: true
dir: logs
slug: image hosting based on aliyun oss
---

# 概述

最近在研究怎么使用 hugo 发布 obsidian 文档，对于图片和其他等附件的保存位置，有两种方案：直接保存到博客服务器或者保存到图床。考虑到服务器只买了 30GB 的硬盘，直接放服务器上可能会爆容量，还是选择基于阿里云的 OSS 服务搭建图床，也保留了以后使用阿里云 CDN 服务加速访问的可能性。  
本文参考整合了网络上多篇博客教程 [^1]。

# 图床搭建

图床搭建主要由三部分组成：购买阿里云 OSS 空间，创建存储空间 Bucket，绑定域名（可选），配置安全策略，配置图床插件。  
图床的费用由两部分构成：存储费用（40GB 9 元/年）和流量费用（0.5 元/GB)，正常情况下流量费用可以忽略不计。

## 购买 OSS 空间和创建 Bucket

在阿里云官网搜索 OSS 即可找到购买页面，按照如下配置购买即可：  
![OSS购买配置截图](https://pics-zhouxin.oss-cn-hangzhou.aliyuncs.com/OSS%E8%B4%AD%E4%B9%B0%E9%85%8D%E7%BD%AE%E6%88%AA%E5%9B%BE.png)  
购买完成后，在 OSS 管理页面可以创建 Bucket，按照如下配置进行设置：  
![Bucket创建配置](https://pics-zhouxin.oss-cn-hangzhou.aliyuncs.com/Bucket%E5%88%9B%E5%BB%BA%E9%85%8D%E7%BD%AE.png)

## 绑定域名

#todo

## 数据安全 - 防盗链

为了防止图床图片被第三方引用导致异常的流量费用，可以使用 OSS 提供的防盗链功能仅对白名单 `Referer` 内的请求响应，设置路径在 `Bucket控制台-数据安全-防盗链`，在白名单中保留允许访问的域名如：`*.aliyun.com`、`blog.example.com`。  
是否允许空 Referer 访问仁者见仁，如果禁止将导致在 obsidian、typora 等软件中无法正常加载 OSS 上的图片。

## 配置图床插件

首先在阿里云中为 PicGo 创建一个子用户，并授予其对 OSS 的完全管理权限。  
创建子用户：在阿里云中找到 `RAM访问控制-身份管理-用户-创建用户`，登录名称任意，勾选允许 `OpenAPI 调用访问`，创建完成后会得到一组 `AccessKey ID` 和 `AccessKey Secret`，需要保管好，后续会用到。  
然后在用户管理界面，为刚刚创建的用户添加权限 `AliyunOSSFullAccess`。  
![Pasted image 20240402090526](https://pics-zhouxin.oss-cn-hangzhou.aliyuncs.com/Pasted%20image%2020240402090526.png)  
![Pasted image 20240402090610](https://pics-zhouxin.oss-cn-hangzhou.aliyuncs.com/Pasted%20image%2020240402090610.png)

子账户配置完成后，在终端使用命令 `winget install picgo` 安装图床软件 PicGo，或者前往 [PicGo is Here | PicGo](https://picgo.github.io/PicGo-Doc/zh/guide/#%E4%B8%8B%E8%BD%BD%E5%AE%89%E8%A3%85) 下载。在 `PicGo-图床设置-阿里云OOS` 配置相应参数：  
设定 KeyId：子用户的 AccessKey ID  
设定 KeySecret：子用户对应的 AccessKey Secre  
设定 Bucket：之前创建的 Bucket 名称

# 参考文档

[^1]: [02.Hugo中使用阿里云OSS作为图床 - 知乎](https://zhuanlan.zhihu.com/p/638165744)