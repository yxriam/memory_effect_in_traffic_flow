# tasks/：交给代理的任务

一个任务一个文件（从 `TEMPLATE.md` 复制）。任务清单和状态在 `../STATUS.md` 第 8 节，规则在 `../AGENTS.md`。

## 一个任务怎样走完

1. 写说明：用户或 Claude Code 按模板写好任务文件，判据先写。
2. 执行：对执行者说“按 `tasks/T01_taxi_heading_shuffle.md` 执行，做完把执行记录写进任务文件”。
3. 审查：对另一个说“审查 T01：只读，对照任务说明检查代码改动、结果文件和实验记录里的数字，关键数字自己从结果文件重算，意见写进任务文件的审查记录”。
4. 修改：执行者按审查意见改，审查者复核，直到审查记录写“通过”。
5. 收尾：用户看过以后运行 `sync_github.bat "说明"`。

同一时间只让一个代理在仓库主文件夹里改文件。要并行，另一个用 git worktree（见 `../AGENTS.md`）。

## 在 Claude Code 里直接调用 Codex（可选）

OpenAI 为 Claude Code 做了一个 Codex 插件，装好后不用开两个窗口。在 Claude Code 里依次输入：

```
/plugin marketplace add openai/codex-plugin-cc
/plugin install codex@openai-codex
/reload-plugins
/codex:setup
```

它调用本机已经登录的 Codex（`codex login`），需要 Node.js 18.18 以上。常用命令：

- `/codex:review`：Codex 审查当前未提交的改动（只读）；
- `/codex:adversarial-review <关注点>`：更挑剔的审查，例如“数据泄漏和比较是否公平”；
- `/codex:rescue <任务>`：把任务交给 Codex 去做；
- `/codex:status`、`/codex:result`、`/codex:cancel`：查看、取结果、取消后台任务。

插件说明没有提到 Windows，装不上就用两个窗口的方式。插件的“审查闸门”（每次回答后自动让 Codex 审查）会很快用完额度，不建议打开。
