---
title: 基于Webhook实现hugo博客自动构建部署
tags:
  - Webbook
  - Hugo
date: 2024-04-10T10:35:00+08:00
lastmod: 2024-04-10T11:48:00+08:00
publish: true
dir: 建站日志
---

# 博客发布流程

我的博文发布工作流可以参考这篇文章 [博客搭建日志 > 博客发布工作流](%E5%8D%9A%E5%AE%A2%E6%90%AD%E5%BB%BA%E6%97%A5%E5%BF%97.md#博客发布工作流)，其中最后两个步骤还需要手动完成，即登录服务器从 repo 中拉取，然后使用 hugo 命令构建。

询问 GPT 后得知，Github 提供了 Webhook 服务，配合服务器上的 Webhook 监听器，可以实现每当我向 repo 推送时，都在服务器上自动拉取并构建博客。

# 步骤

## 前置条件

首先需要创建一个 repo 项目保存博客相关文件，并把服务器的公钥添加到 Github 账户的 SSH 密钥中。这一过程可以参考：[博客搭建日志](%E5%8D%9A%E5%AE%A2%E6%90%AD%E5%BB%BA%E6%97%A5%E5%BF%97.md)。

## 在服务器上设置 Webhook 监听器

在服务器上需要设置一个监听器监听来自 Github 的 push 事件，可以自己用 Flask 写一个，或者直接用现成的 webhook 工具：

```sh
sudo apt install webhook -y
```

创建一个 webhook 文件配置文件 `hooks.json`：

```json
[
  {
    "id": "redeploy-blog",
    "execute-command": "/path/to/your/script.sh",
    "command-working-directory": "/path/to/your/hugo/blog",
    "pass-arguments-to-command": [
      {
        "source": "payload",
        "name": "head_commit.id"
      }
    ],
    "trigger-rule": {
      "and": [
        {
          "match": {
            "type": "payload-hash-sha1",
            "secret": ，"your_webhook_secret",
            "parameter": {
              "source": "header",
              "name": "X-Hub-Signature"
            }
          }
        }
      ]
    }
  }
]
```

`hooks.json` 配置文件中，`execute-command` 指示监听到指定内容后需要执行的脚本，`command-working-directory` 指示了脚本的工作目录，可以设置为博客部署的目录。此外，还有一个 `secret` 字段需要修改为自定义内容，该字段用于验证报文是否来自 Github 发送。

然后创建待执行的脚本 `script.sh`，主要内容是进入指定目录、拉取最新更改、切换到 main 分支、执行构建命令：

```sh
#!/bin/bash
# cd path/to/blog
git pull --all
git switch main
hugo
```

为 `script.sh` 添加可执行权限：

```sh
chmod +x /path/to/your/script.sh
```

开放服务器指定端口（默认 9000），运行 webhook：

```sh
webhook -hooks hooks.json -verbose --port 9000
```

在打印出的状态消息中，可以看到 webhook 正在监听的 url，后面需要填写到 Github 中。

## 设置仓库 Webhook

在你的 Github 对应的 repo 中：

- 转到 "Settings" > "Webhooks" > "Add webhook"
- 在 `Payload url` 中填写服务器 webhook 监听的路径，注意将其中的 `{commits}` 替换为自定义内容；`Content type` 选择 `application/json`；`Secret` 填写与 `hooks.json` 一致的内容
- 添加完成后 Github 会向服务器发送一条 ping 消息，可以在服务器端和 Github Webhook 页面查看接受状态。如果接受失败，请检查：是否开放了服务器指定端口、url 直接使用浏览器访问服务器是否能接收到 get 请求、url 中若为 https 协议需要先配置反向代理。

## 使用 systemd 管理 webhook

首先在服务器上创建 `systemd` 文件：

```sh
sudo vim /etc/systemd/system/webhook.service
```

然后粘贴以下内容，注意修改命令中的端口号：

```ini
[Unit]
Description=GitHub Webhook
After=network.target

[Service]
User=your_username
WorkingDirectory=/path/to/your/hugo/blog
ExecStart=/usr/bin/webhook -hooks /path/to/your/hooks.json -verbose --port xxxx
Restart=always

[Install]
WantedBy=multi-user.target

```

替换 `your_username` 为运行 webhook 的用户，`/path/to/your/hugo/blog` 和 `/path/to/your/hooks.json` 为实际的路径。

启用服务以确保它在每次启动时自动运行，并立即启动服务：

```sh
sudo systemctl enable webhook.service
sudo systemctl start webhook.service
```

可以使用以下命令检查服务状态：

``` sh
sudo systemctl status webhook.service
```
