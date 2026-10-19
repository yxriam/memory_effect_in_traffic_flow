# 给 Claude 的仓库说明

杨欣然的交通流记忆效应论文代码库（github.com/yxriam/memory_effect_in_traffic_flow）。结构和每个脚本的输入输出见 `README.md`、`controls/README.md`、`prediction/README.md`。

## 每次实验都要记录

在这个仓库里跑了实验、重跑或核对了结果之后，结束前必须完成：

1. 按 `experiment_log/TEMPLATE.md` 新建 `experiment_log/YYYY-MM-DD_<主题>.md`，写清：
   - 数据：位置、规模、版本、筛选；
   - 实验：命令、参数、随机种子、环境；
   - 结果：关键数字；重跑时每个结果文件都要逐个做新旧对比，不能只写“不变”；
   - 结论、问题与改动、下一步。
   同一天的多次实验写在同一个文件里，分小节。
2. 在 `EXPERIMENT_LOG.md` 的总表加一行，并更新其中的“当前的下一步”：做完的打勾并注明日期，新的加上。
3. 重跑过结果时，运行 `python tools/check_paper_numbers.py`，把 summary 行写进记录。

## 约定

- 数据路径只从 `common/paths.py` 读（`local_paths.json` 或 `MEMFX_*` 环境变量），不要写死。
- 随机种子必须固定，并且与机器无关：不要用 Python 的 `hash()`（每次运行结果都不同），改用固定整数或 `zlib.crc32(文件名)`。结果也不能依赖多进程的完成顺序（需要时先排序）。
- 单位：CMI 用 bits；highD、`coarse_model`、`memory_by_mode_speed` 的留出似然增益用 nats；预测交叉熵用 bits/step。
- 推送：在用户电脑上运行 `sync_github.bat "说明"`。做不到的步骤（例如没有推送权限、没有 GPU 或 Infomap）要先告诉用户，再开始做。
