---
title: 使用ssh远程连接wsl2
tags:
  - wsl2
  - ssh
  - windows
date: 2024-07-17T17:26:00+08:00
lastmod: 2024-09-02T18:42:00+08:00
publish: true
dir: notes
slug: using ssh to connect remotely to wsl2
---

# 概述

wsl2 使得 Windows 用户可以很方便地访问 Linux 环境，微软也在 vscode 中提供了相应的插件支持。但 wsl2 一般都是通过本地访问的，微软似乎没有直接提供远程访问 wsl2 的方式。

经过一番摸索，远程访问 wsl2 主要有以下几个步骤：

- 【非必需】启用 windows 中的 ssh 服务器
- 启用并配置 wsl2 中的 ssh 服务
- 开放防火墙
- 修改 wsl2 网络模式

# 详细步骤

## 【非必需】启用 windows 中的 ssh 服务器

在摸索过程中发现，windows 也是支持通过 ssh 远程连接的，想要 ssh 到 wsl2，自然就有一种曲线救国的方案，即先通过 ssh 连接到 windows 宿主机，然后通过终端进入 wsl2。理论可行，实践如下：

- 启用 ssh 服务器  
windows 中 ssh 服务器启用可参考官方文档 [^1]，写的很详细。以 Windows 11 为例，在 powershell【使用**系统默认版本**，powershell 7.4.3 无法正确执行】中以管理员身份执行以下命令即可启用 ssh 服务器：

```powershell
# 安装OpenSSH客户端
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0

# 安装OpenSSH服务器
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# 启用sshd服务
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'

# 确认防火墙规则被自动配置
if (!(Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue | Select-Object Name, Enabled)) {
    Write-Output "Firewall Rule 'OpenSSH-Server-In-TCP' does not exist, creating it..."
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
} else {
    Write-Output "Firewall rule 'OpenSSH-Server-In-TCP' has been created and exists."
}
```

执行完毕后，可使用 `ssh <username>@127.0.0.1` 测试能否通过 ssh 连接到 windows 终端。注意，对于 Windows OpenSSH，唯一可用的身份验证方法是 `password` 和 `publickey`，即不支持通过 Microsoft 账号验证。

- 修改默认终端为 powershell  
在 windows 中，默认连接的终端为 cmd，可使用命令 `echo %COMSPEC%` 确认。默认使用的终端由注册表中 `HKEY_LOCAL_MACHINE\SOFTWARE\OpenSSH\DefaultShell` 决定，使用如下命令可以将其修改为 powershell：

```powershell
$pwshPath = (Get-Command powershell.exe).Source
$pwshPathQuoted = '"' + $pwshPath + '"'
sudo Set-ItemProperty -Verbose -Path "HKLM:\SOFTWARE\OpenSSH" -Name "DefaultShell" -Value $pwshPathQuoted -Force
```

注意，默认终端修改为 powershell 7（pwsh.exe）有权限不足的报错，这是因为 pwsh.exe 默认安装在 `C\programs files` 路径下，该路径需要管理员权限访问。

- ssh 安全配置  
Windows ssh 服务器的默认配置文件为 `%programdata%\ssh\sshd_config`，各字段含义参考官方文档 [^2][^3]，建议修改默认端口，并使用 `AllowUsers` 指定允许连接的用户，或者使用 `AllowGroups` 指定远程连接用户组连接。在配置文件中追加如下内容：

```text
Port xxxxx # 修改默认端口
AllowGroups "sshUsers" # 仅允许指定组
```

在上述配置文件中，我们仅允许了 sshUsers 组内用户进行连接，接下来我们创建一个 sshUsers 组，并添加相应成员：

```powershell
restart-Service sshd # 修改配置文件后，重启服务才能生效

net localgroup sshUsers /add # 添加sshUsers组
net localgroup sshUsers <username> /add # 将user添加到该组
```

此外，还要在防火墙中开放修改的 ssh 服务端口：

```powershell
New-NetFirewallRule -DisplayName '"Allow SSH on Port xxxxx"' -Direction Inbound -Protocol TCP -LocalPort xxxxx -Action Allow
```

## 启用并配置 wsl2 中的 ssh 服务

- 安装/重装 OpenSSH 服务器  
无论 wsl2 中是否已经安装好 OpenSSH 服务器，都建议卸载后重装，即执行如下命令：

```bash
# 先卸载重装系统自带的sshd
sudo apt-get remove openssh-server
sudo apt-get install openssh-server
```

- ssh 安全配置  
wsl2 ssh 服务器默认配置文件为 `/etc/ssh/sshd_config`，各字段含义参考官方文档 [^2]，建议修改默认端口，并通过密钥认证登录，即在配置中修改如下配置项内容：

```ssh
Port xxxxx # 修改默认端口
PasswordAuthentication no # 禁用密码认证
PubkeyAuthentication yes # 允许公钥认证
AuthenticationMethods publickey # 仅使用公钥认证
```

然后将宿主机和其它需要连接到 wsl2 设备的公钥写入 `~/.ssh/authorized_keys` 文件。

修改 `sshd_config` 配置文件后，需要使用命令 `sudo service sshd restart` 重启服务才会生效。写入公钥后在 windows 宿主机上就可以使用 `ssh <username>@127.0.0.1 -p xxxxx` 测试能否连接到 wsl2。

## 开放防火墙

修改端口后，需要在宿主机的防火墙中开放对应的端口，在宿主机的 powershell 中以管理员权限执行如下命令：

```powershell
New-NetFirewallRule -DisplayName '"Allow SSH on Port xxxxx"' -Direction Inbound -Protocol TCP -LocalPort xxxxx -Action Allow
```

## 修改 wsl2 网络模式

wsl2 的默认网络模式是 NAT[^4]，在此模式下：

- windows 可以使用 localhost 访问 wsl2 网络应用
- wsl2 需要通过获取主机 ip 访问 windows 应用
- 局域网设备需要通过主机端口转发访问 wsl2 应用

在运行 Windows 11 22H2 及更高版本的宿主机上，wsl2 支持镜像网络模式，在此模式下，windows 主机可以使用 localhost 访问 wsl2 网络应用，局域网设备可以直接使用宿主机 ip 访问 wsl2 网络应用。

wsl2 配置文件路径为 `%UserProfile%/.wslconfig`，修改为以下内容：

```wslconfig
[experimental]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
```

上述配置中还启用了自动代理、防火墙和 dns 隧道。修改完成后，重启 wsl 即可应用该配置：

```powershell
wsl --shutdown
wsl
```

至此，我们就可以在局域网内使用 ssh 连接宿主机上的 wsl2，如果想在外网连接，可以使用 zerotier 异地组网，可参考文章 [搭建ZeroTier MOON服务器 | 周鑫的个人博客](https://www.zhouxin.space/notes/setup-zerotier-moon-server/)。

# 参考文档

[^1]: [适用于 Windows 的 OpenSSH 入门 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows-server/administration/openssh/openssh_install_firstuse?tabs=powershell)
[^2]: [sshd\_config(5): OpenSSH SSH daemon config file - Linux man page](https://linux.die.net/man/5/sshd_config)
[^3]: [适用于 Windows 的 OpenSSH 服务器配置 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows-server/administration/openssh/openssh_server_configuration#windows-configurations-in-sshd_config)
[^4]: [使用 WSL 访问网络应用程序 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/networking#default-networking-mode-nat)