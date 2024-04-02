---
title: 搭建ZeroTier MOON服务器
tags: 
date: 2024-03-31T11:40:00+08:00
lastmod: 2024-04-02T20:24:00+08:00
publish: true
---

# 资源存档

原文链接：[ZeroTier-One搭建moon节点 | 一水轩](https://www.tpfuture.top/views/linux/net/ZerotierOneAddMoon.html)  
ZeroTier 官网：[ZeroTier Central](https://my.zerotier.com/)

# 搭建过程

## 在服务器上安装并配置 ZeroTier

### 安装 ZeroTier

```sh
curl -s https://install.zerotier.com | sudo bash

sudo systemctl start zerotier-one.service

sudo systemctl enable zerotier-one.service


sudo zerotier-cli join <network ID> # 此处填写你的网络的network ID
```

### 在控制台勾选服务器

前往对应网络控制台 [ZeroTier Central](https://my.zerotier.com/)，允许刚刚添加的设备。  


## 搭建 MOON 服务器

### 开放端口

MOON 默认使用 UDP 9993 端口，故需要在服务器控制台开放对应入站策略。

### 生成 `moon.json` 文件

```sh
cd /var/lib/zerotier-one/
sudo zerotier-idtool initmoon identity.public > moon.json
```

使用 `vim` 等文本编辑工具修改刚刚生成的 `moon.json` 中 `"stableEndpoints"` 的值为服务器的公网 IPv4 地址：

``` sh
{
 "id": "xxxxx", # 这个值后面用于其它设备配置moon
 "objtype": "world",
 "roots": [
  {
   "identity": "xxxx:0:eeee",
   "stableEndpoints": ["<IPv4 address>/9993"] # 修改这里<IPv4 address>替换为公网地址
  }
 ],
 "signingKey": "asdfasdfasdf",
 "signingKey_SECRET": "asdfasdfasdfasd",
 "updatesMustBeSignedBy": "asdfasdfasdf",
 "worldType": "moon"
}
```

注意，该文件的 `id` 字段唯一标识了这台设备，该 `id` 用于其它结点配置 moon。

### 生成签名文件

```sh
zerotier-idtool genmoon moon.json
```

该命令会生成一个 `.moon` 文件，通过这个文件，可以把 moon 节点加入网络。

### 将 moon 节点加入网络

``` sh
mkdir moons.d
mv *.moon moons.d/
sudo systemctl restart zerotier-one
```

## 其它设备配置

在需要使用 MOON 的设备上安装了 ZeroTier 并加入网络后，还需要手动配置 MOON 节点：

``` sh
sudo zerotier-cli orbit <id> <id>  # 或者在windows上需要管理员权限
```

其中 `id` 是 MOON 服务器的节点 id，可在 [[#生成`moon.json`文件]] 这一步生成的 `json` 中看见，或者在 ZeroTier 网络控制台也可以找到该设备的 id。
