# T03c GeoLife 停留点管线与基线复现（关卡）

- 执行：Codex　审查：Claude Code
- 状态：待做
- 关联：`tasks/T03_prediction_design.md`；依赖 T03a

## 为什么做

GeoLife 的预测改为停留点级的下一位置预测，采用 Hong 等（TR-C 2023）的公开流程，结果才能和文献比较。T03e、T03f、T03g 都建在这条管线上。

## 问题

用 `mie-lab/location-prediction`（提交 `3417149`）在本机数据上，能否复现论文报告的 GeoLife 结果（相差 ≤ 10%）？

## 方法

1. 数据：`common/paths.py` 的 `geolife_raw`（GeoLife 1.3，用户文件夹 000…181）。在外部仓库的 `paths.json` 里指向它（不改仓库里的 `local_paths.json`）。
2. 预处理：`python preprocessing/geolife.py 20`（trackintel 1.2.4：停留点 200 m、30 min，活动阈值 25 min，用户质量筛选，位置聚类 ε = 20 m）。记录停留点数、位置数（配置里 `total_loc_num: 1187`）、用户数（`total_user_num: 46`）是否一致；不一致先查 trackintel 版本。
3. 模型：
   - 论文的 Transformer（MHSA），`config/geolife/transformer.yml`（上下文：用户、时间、时长；`previous_day: 7`）；
   - 仓库自带的非学习基线：持续预测、最常去预测、马尔科夫模型（`baselines/`）。
   - 各 3 个种子，用 T03a 运行器调用外部仓库的训练代码（只加包装，不改外部代码；必须改的地方以补丁形式放在 `prediction/bench/patches/` 并说明）。
4. 指标：Acc@1、Acc@5、Acc@10、F1、MRR、NDCG@10（外部仓库 `utils/train.py`）。
5. 目标数字：从论文（arXiv 2212.01953 / TR-C 2023, 104315）的 GeoLife 结果表中读出 MHSA 和马尔科夫基线的数字，写进执行记录。
6. 记录每个用户的测试样本数（T03f 要用）。

## 判据（跑之前写好）

- 通过：MHSA 的 Acc@1 和 MRR 与论文的相对差 ≤ 10%，马尔科夫基线同样；
- 部分通过：只有 MHSA 超出：检查训练轮数、早停和随机种子，再判；
- 不通过：预处理后的位置数、用户数与论文不一致且原因查不清，或 MHSA 与马尔科夫都超出：不进入 T03e。

## 产出

`external/location-prediction`（submodule）、`prediction/bench/configs/geolife_baseline/`、`prediction/bench/results/geolife_baseline.csv`；`experiment_log/<日期>_geolife_baseline.md`；`EXPERIMENT_LOG.md` 一行；`STATUS.md` 第 8 节。

## 审查要点

- 划分方式（按每个用户的时间顺序切分）是否与论文一致；
- 外部代码有没有被直接修改；
- 目标数字是否出自 GeoLife 表（论文还有另一个数据集）。

## 执行记录

## 审查记录
