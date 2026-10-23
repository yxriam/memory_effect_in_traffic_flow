@AGENTS.md

## 只给 Claude Code 的补充

- 第一行把共用规则 `AGENTS.md` 导入进来。改规则请改 `AGENTS.md`，这里只放 Claude Code 专用的内容。运行 `/context` 可以确认规则已经载入。
- 做独立审查时，开一个只拿到任务说明、代码和结果文件的子代理，不要先把执行者的结论或你自己的判断告诉它。
- 装了 OpenAI 的 Codex 插件（`openai/codex-plugin-cc`）时：`/codex:review` 请 Codex 审查未提交的改动，`/codex:adversarial-review` 做更挑剔的审查，`/codex:rescue` 把任务交给 Codex 执行。审查意见同样写进对应的任务文件。
