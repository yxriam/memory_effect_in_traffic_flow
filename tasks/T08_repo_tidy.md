# T08 仓库精简

- 执行：Codex　审查：Claude Code
- 状态：待做。不跑实验；做的时候仓库主文件夹里不能有别的代理在改文件（看 `STATUS.md` 第 8 节有没有“进行中”）。
- 关联：`AGENTS.md` 的“目录约定”；`STATUS.md` 第 8 节

## 为什么做

仓库在 10-14 已经删过一轮无用代码，主体不臃肿：75 MB、562 个文件，其中 41 MB 是 `taxi/` 的四个扫描结果表。剩下的是几件小事：

1. `.omc/`（oh-my-claudecode 插件的本机状态）被提交进了公开仓库，里面有本机路径。`.gitignore` 已加上 `.omc/`（10-23），但已提交的文件还要移出版本库。
2. 几个不再用于论文数字的旧文件和正在用的文件混在一起，接手的人分不清。
3. README 的目录图还没有 `STATUS.md` 看板、`tasks/`、`legacy/`。
4. 杭州数据 10-23 迁出为独立小项目，仓库里留下了 `hangzhou/` 和 `tasks/T06_hangzhou_prediction.md` 两个“已迁出”说明，`.gitignore` 里还有两行 `hangzhou/`。

论文第一部分投稿前，`controls/`、`prediction/` 的结构不动。

## 方法

1. 移出 `.omc/`：`git rm -r --cached .omc`，本机文件保留。
2. 归档到 `legacy/<原目录>/`，用 `git mv`，只移不删。每个候选先 `git grep` 文件名，再按下面的规则处理：
   - 被 `tools/check_paper_numbers.py`、任何 `.py` / `.bat` 脚本、或 README 里“论文图表 → 脚本”对照表引用的，**不动**；
   - 只被 README 目录图或旧 notebook 引用的，可以移，同时改 README。

   | 候选 | 目前的引用 | 建议 |
   |---|---|---|
   | `highD/highd_memory_analysis.ipynb` | 只有 README（2026-05 的初探，已被 `controls/` 取代） | 移 |
   | `geolife/06_plot_5s_new_old_comparison.py` | 只有 `geolife/README.md` | 移 |
   | `geolife/90_optional_markov_holdout_eval.py`、`91_optional_rnn_lstm_residual.py` | README、`requirements-full.txt` | 移，并改这两处 |
   | `airports/lstm_rnn_predict.ipynb`、`airports/model/`、`airports/dataloader/` | 滚动返回率已不用，但 README 说图 3 下半的桑基图出自这个 notebook | 等 T04 定了图 3 怎么画再移 |
   | `taxi/memory_backoff_sweep*.csv`、`memory_full_per_file*.csv`（41 MB） | `controls/figures/fig_taxi_gridmode.py`、`controls/multiple_testing.py` 在用 | 不动 |
   | `prediction/keep_awake.py`、`unit_cache_mix2.py`、`ch6_extra.py` | 脚本或 README 在用 | 不动 |

3. 删除 `hangzhou/` 目录和 `tasks/T06_hangzhou_prediction.md`（`git rm`），去掉 `.gitignore` 里的两行 `hangzhou/`。这两处只剩迁出说明，内容已在独立小项目里。
4. `legacy/README.md`：每个文件一行，写清原来的位置、为什么不再用、被什么取代。
5. 更新 README 和 README_zh 的目录图：
   - `STATUS.md` 写作“进度看板”；
   - 加上 `tasks/`、`legacy/`；
   - 删掉已经移走的文件。

## 判据（跑之前写好）

满足以下全部，才算通过：

- `python tools/check_paper_numbers.py` 在改动前后都全部一致；
- `git grep` 移走的每个文件名，除了 `legacy/README.md`、实验记录和 `EXPERIMENT_LOG.md` 以外，没有指向旧位置的引用；
- `git status` 里除了第 3 步的两个迁出说明，没有被删除的文件，只有重命名。

## 产出

- 一次提交（由用户运行 `sync_github.bat "tidy: legacy + .omc"`）；
- `legacy/README.md`；
- 执行记录里列出移动清单和仓库大小的前后对比；
- 更新 `STATUS.md` 第 8 节。

## 审查要点

- 有没有移走还在用的文件，重点看 `check_paper_numbers.py`、`controls/figures/*.py` 和 `.bat` 里的相对路径；
- README 的“论文图表 → 脚本”对照表是否仍然指向存在的文件。

## 执行记录

## 审查记录
