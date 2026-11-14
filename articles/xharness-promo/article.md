---
title: 我用 Rust 写了个开源 AI Agent 运行时:XHarness——会话断电可恢复,桌面/网页同一套 Host
topics: []
status: draft
url:
id:
created: 2026-11-14
published:
tags: [rust, ai-agent, open-source, xharness]
---

*(开源项目,Apache-2.0,三平台安装包开箱即用:https://github.com/123123213weqw/x-harness-rs 。国内镜像:https://gitee.com/wangyue2006/x-harness-rs )*

## 为什么要自己写一个 Agent 运行时

用过各种 coding agent 的人大概都被同一件事折磨过:**会话太脆了**。

模型跑一半,进程崩了、电脑断电了、网络断了——上下文全丢,agent 干到哪一步全靠猜,更恐怖的是重启后它可能把已经执行过的写文件、跑命令**再执行一遍**。另一个普遍割裂是:桌面应用和 Web 界面是两个产品,会话、权限、工具行为各玩各的。

所以我用 Rust 写了 **XHarness**:一个跨平台的开源 coding-agent 运行时。核心就赌两件事——**durable(耐久)** 和 **one Host(一套宿主)**。

![XHarness 主界面:模型选择、权限模式、思考等级全部集成在输入区](images/shot1_main_clean.png)

## 卖点一:Durable by design——会话是可以"断电重启"的

这是整个项目最核心的设计。三个机制:

- **Append-only 会话日志**:所有事件只追加、不修改,重启后 Host 从日志完整重建对话
- **持久输入队列 + 检查点**:你发的消息不会因为崩溃而丢失
- **副作用不静默重放**:被中断的写文件/执行命令,恢复时会明确标注而不是偷偷再来一遍——这一点对 coding agent 是安全底线

界面上对应的是每个会话都有的 **Session log(可下载)和 Recover task 按钮**——会话不只是"聊天记录",是一份可审计、可恢复的执行档案。

## 卖点二:一个 Rust Host,桌面和 Web 同源

Tauri 桌面应用打包了 Host + Web UI(隔离的原生 WebView);纯 Web 部署也能跑同一套逻辑。**session、权限、工具行为在两个形态下完全一致**——因为它们本来就是同一个 Host。桌面端还带了内建更新通道,每个 Release 都附带 SHA256 校验和。

## 卖点三:过程可视化——Trajectory 和 Performance 视图

一般的 agent 产品只有聊天流。XHarness 给每个会话配了四个视图,这是我认为最有差异化的部分:

**Trajectory 视图**——agent 的执行轨迹:顶部是甘特式时间线(按耗时分解 Input/Model/Tools 三层),下面是完整轮次流(用户消息 → 推理 → 工具调用 → 结果)。下面这张是真实任务截图,注意中间那次**工具失败后的自我修复**:

![Trajectory 视图:write 工具报错后,agent 自己诊断出 /tmp 软链问题,改用 heredoc 绕过并完成任务](images/shot5_trajectory.png)

**Performance 视图**——token 级别的可观测性:按 1 秒桶绘制的 tok/s 曲线(Text/Reasoning/Tool input 三色),底部阶段条带拆出首 token 等待、工具执行、推理各占多久。做 agent 调优的人一看就知道这有多实用:

![Performance 视图:三色 tok/s 曲线 + 阶段耗时条带,按轮次切换](images/shot4_performance.png)

## 卖点四:真实跑任务的闭环(不是演示壳)

流式输出推理与工具调用、**中途转向(steering)、取消、工具审批门控**、后台监督任务;上下文管理有按模型的窗口限额、请求记账、工具输出截断与压缩(界面直接显示"context 已用 42%"这类真实计量)。模型侧不绑定任何厂商:OpenAI 兼容的 Chat Completions 和 Responses 协议都支持,多 provider 路由,云上或自托管端点都行。

![会话执行:文件、实现说明、测试输出与退出码,一次汇报闭环](images/shot2_working.png)

## 卖点五:插件、MCP 与任务自动化

- 插件系统 + 公共插件目录(xharness-plugin-registry),支持安装校验过的插件包
- MCP(Model Context Protocol)集成,接入现有工具生态
- **Tasks and automations**:给 agent 设目标,它可以多轮自动推进(带暂停/审计)

![任务与自动化面板](images/shot3_tasks_clean.png)

## 工程侧的一些数字

| | |
|---|---|
| 语言 | Rust 为主(约 1087 万行),加 TS/JS 前端 |
| 工作区 | 46 个 crate(core/agent/context/provider/tools/platform/host…) |
| 平台 | macOS(Apple Silicon)/ Linux / Windows 安装包 |
| 发布 | v0.2.66(持续迭代,平均几天一个版本) |
| 许可 | Apache-2.0 |

架构一句话:**模型 provider 与 agent 循环是平台无关的;文件、进程、沙箱行为在平台服务层;Host 拥有会话、准入、恢复与 UI 投影**。

## 最较真的部分:怎么证明"实现"本身的差异

做性能声明容易翻车,所以我专门建了 **xharness-bench** 仓库做受控对比,方法论比结论更值得看:

- XHarness 是 deepseek-harness 的 Rust 重实现,并刻意跟踪上游——两者说**同一套控制平面**(52 个 RPC 方法、10 种 mux/host 帧全兼容,冻结在 upstream 的 tag 上)
- 于是可以做到别家做不到的对照实验:**固定模型、任务、工作区、prompt 和工具面,只变"实现"**(语言、并发模型、上下文管理、传输层)——Tier A 是同构对照,Tier B 才是跨 harness(Claude Code、Codex、OpenHands 等)的弱对照,并明确声明两个 tier 的可比性边界
- 评估跑在标准的 Terminal-Bench 上,只认外部验证的通过/失败;token 记账只记录、不跨家比较(各家口径不同)

这套"诚实对照"的思路,来自我不想自欺:如果你想看数据,仓库里的报告可复现。

## 快速上手

**桌面版**(推荐):去 [Releases](https://github.com/123123213weqw/x-harness-rs/releases) 下载对应平台安装包,开箱即用。

**自己跑 Web Host**:

```bash
XHARNESS_WORKSPACE="$PWD" \
XHARNESS_WEB_DIST="$PWD/ui/dist" \
XHARNESS_BASE_URL="http://127.0.0.1:8000/v1" \
XHARNESS_MODEL="your-model-id" \
cargo run --locked -p xharness-host-app --bin xharness-host
```

打开 http://127.0.0.1:3080/ 即可(默认只监听本机)。想嵌入自己的产品的话,`xharness-sdk` crate 提供进程内 SDK:注入模型与工具、入队输入、观察事件、恢复会话。

## 最后

XHarness 目前是个人维护的活跃项目(46 crates、持续发版),诚实地讲:它还在快速迭代,SDK 未到 1.0 稳定承诺;Web UI 引用了 DeepSeek Harness 的部分组件并保留了对应许可声明。

如果你在做 agent 相关的东西——不管是想要一个**不会丢会话的本地 coding 工具**,还是想**嵌入一个耐久的 agent 运行时**,或者只想研究 agent 架构怎么拆 crate——欢迎来看看,提 issue、提需求,或者单纯点个 star 让我知道这东西有人想要:

**GitHub**:https://github.com/123123213weqw/x-harness-rs
**Gitee 镜像**(国内直连):https://gitee.com/wangyue2006/x-harness-rs
**受控对比实验**:https://github.com/123123213weqw/xharness-bench

写这篇文章的原因很简单:agent 的耐久性和可观测性,值得被当成一等公民来设计,而不是崩了重来。XHarness 是我对这个判断的一次完整实现。

---

*注:文中截图均为本机真实运行画面(Trajectory 那张里的工具失败→自修复也是真实发生的,不是摆拍)。*
