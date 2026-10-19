# 2026-10-19 论文一四组数字核对与复现

## 目的

核对论文一（`paper_latex_v2/main.tex`、`main_zh.tex`，2026-10-17 版）中的四组数字，并确认产生它们的代码都能复现：

1. GeoLife 零模型：0.325 / 0.012 bits；
2. highD 状态补全：0.076 → 0.007 nats，以及 AR(1) 替代序列的 0.034 nats；
3. 三层框架：速度持续时间 τ_c、ρ = −0.98、路径记忆、出行起点；
4. LSTM、Transformer 与记忆网络。

## 数据

| 数据集 | 位置（`common/paths.py` 的键） | 规模 | 备注 |
|---|---|---|---|
| GeoLife 清洗缓存 | `geolife_cache`（本机 `geolife_clean_pipeline\cache_clean`） | 181 个 `.npz` | 第 1 组、出行起点 |
| GeoLife 按方式轨迹 | `geolife_trips_by_mode` | 253 个 `.plt` | τ_c、路径记忆、Hurst |
| highD | `highd_tracks` | 60 个 `*_tracks.csv` | 第 2 组 |
| 上海出租车 | `taxi_raw` | 4,316 个文件；出行起点只用 `controls/taxi_pick.json` 的前 100 个 | 第 3 组 |
| prediction 序列 | `prediction/data/*.npz` | taxi_traj20 6,804 段；geolife_bj500m 38,248 段（11,425 个状态）；highd_v20 / v100 110,093 段 | 第 4 组 |
| 对照的旧结果 | `controls/results/`、`prediction/results/`（与论文的 `supplementary_results` 相同） | | |

## 实验

| 内容 | 命令 | 参数与随机种子 | 耗时（云端 2 核） |
|---|---|---|---|
| 第 1 组 | `controls/geolife_robust.py` | 4 种网格 × 是否去抖；50 次置换；每名用户的种子由用户编号决定 | 24 min |
| 第 2 组 | `highd_augment.py` → `highd_fine.py`、`highd_fine3.py`、`coarse_model.py`、`accel_acf.py` | 固定种子（划分 100 + 划分号，替代序列 5，划分 1） | 9 min |
| 第 3 组 | `row_persistence.py` → `row_regression.py`、`row_stops.py`；`memory_by_mode.py`；`origin_memory.py`；`taxi_origin.py 0.002` 和 `0.005` | 30 次置换；出租车 20 次半样本 | 7 min |
| 第 4 组 | 云端没有 GPU、PyTorch 和 Infomap，不能重新训练。改为核对 `prediction/results/` 中的 JSON、日志、`infomap_summary.csv`、`ch6_*.csv` 与论文；并用 `ch6_networks.py highd`、`ch6_networks.py geolife`、`export_surrogates.py` 重新生成 Infomap 的输入文件 | 同原设置 | 5 min |

环境：云端 Linux，Python 3.13、numpy 2.5.3、pandas 3.0.5、scipy 1.18.1。输入就是本机的原始数据文件。第 1–3 组先用论文附带的原脚本（`supplementary_code`，算法与 `controls/` 相同，只是路径写法不同）跑了一遍，又用整理后的复现包从头跑了一遍，两次结果相同。

## 结果

### 结果文件逐个新旧对比

旧：已提交的结果；新：这次重跑。

| 结果文件 | 行数 | 新旧最大差异 | 说明 |
|---|---|---|---|
| `geolife_robust_per_user.csv` | 1,430 | 0 | 相同 |
| `augment_markov.csv` | 30 | 0 | 相同 |
| `fine_resolution.csv` | 30 | 0 | 相同 |
| `fine_third_order.csv` | 12 | 0 | 相同 |
| `coarse_model.csv`、`coarse_model_params.json` | 21 | 0 | 相同 |
| `highd_accel_acf.csv` | 15 | 0 | 相同 |
| `row_persistence.csv` | 7,386 | 0 | 相同。删去未使用的随机数生成器后又重跑一次，内容仍相同，只是行序随多进程完成顺序变化，所以没有提交这次的文件 |
| `row_stops_spearman.csv`、`row_mode_medians.csv` | 8 / 7 | 0 | 相同 |
| `row_regression.csv` | 78 | 系数 0；区间最多 0.057 | 原脚本按行序取用户，修正后按编号排序（见“问题与改动”） |
| `row_regression_split.csv` | 26 | 系数 0；区间最多 0.039 | 同上；论文引用的区间里只有火车下限变化：1.11 → 1.12 |
| `memory_by_mode_path.csv` | 251 | 0 | 相同 |
| `memory_by_mode_speed.csv` | 35 | 4×10⁻¹⁶ | 浮点求和顺序，可忽略 |
| `origin_memory_geolife.csv` | 157 | 0 | 相同 |
| `taxi_origin_0.002.csv`、`taxi_origin_0.005.csv` | 1 / 1 | 0 | 相同（100 辆车，86,413 / 55,763 步） |
| `hurst_control.csv` | 5,147 | 确定性的列（H_speed、增量、DFA、φ）为 0；AR(1) 与打乱两列逐段不同 | 随机种子改为与机器无关后重算，中位数变化不超过 0.007，论文引用的数字不变（见第 6 条） |
| Infomap 输入：highD、GeoLife 的 8 个 `.net` | | 逐字节相同 | 出租车的 4 个要读全部 4,316 个文件，留给本机生成 |
| `ch6_highd.csv`、`ch6_geolife.csv` | 3 / 3 | 0 | 相同（不计运行时间列） |

### 关键数字

| 组 | 论文 | 重算值 | 结果文件 |
|---|---|---|---|
| 1 | 零模型 CMI 中位数 0.325 bits；校正后 0.012 bits（10 m）；66% 用户显著；100 m 校正后 0.109 bits，82% | 0.32452；0.01188；66.3%；0.10850，82.1% | `geolife_robust_per_user.csv` |
| 2 | 粗状态增益 0.076 nats → 加速度和细速度后 0.007 nats | 0.07559 → 0.00719 nats（0.1091 → 0.0104 bits） | `augment_markov.csv`、`fine_resolution.csv` |
| 2 | AR(1) 替代 0.034 nats（约为数据的一半）；AR(2) 0.059（82%） | 0.03375（0.0487 bits，46.9%）；0.05871（81.5%） | `coarse_model.csv` |
| 3 | τ_c 中位数：步行 10 s，自行车 / 公交 20 s，地铁 40 s，小汽车 / 出租车 75 s，火车 175 s | 相同 | `row_mode_medians.csv` |
| 3 | τ_c 与每公里停车次数 ρ = −0.98（p = 8×10⁻⁵）；与速度 ρ = 0.93 | −0.98198（p = 8.29×10⁻⁵）；0.927 | `row_stops_spearman.csv`、`row_mode_medians.csv` |
| 3 | 路径记忆：公交 29 / 35%，自行车 28 / 43%，地铁 20 / 24%，出租车 17%，小汽车 8–9%，步行 9 / 26% | 29.2 / 35.3，28.1 / 42.7，19.9 / 24.3，17.0，8.4 / 9.3，9.0 / 25.8 | `memory_by_mode_path.csv` |
| 3 | 出行起点：GeoLife I_O 0.0028、I_3 0.0072、I_O\|3 0.0013 bits；出租车 0.0148 / 0.0058 / 0.0024 bits；500 m 0.039 / 0.015 / 0.004 | 0.00277 / 0.00717 / 0.00129；0.01481 / 0.00581 / 0.00243；0.0390 / 0.0145 / 0.0044 | `origin_memory_geolife.csv`、`taxi_origin_*.csv` |
| 4 | 出租车 LSTM 2.90、Transformer 3.25 bits；三个种子 2.91 ± 0.01 | 2.9035、3.2464；2.906 ± 0.012 | `prediction/results/neural/` |
| 4 | **GeoLife LSTM 2.37** | **2.3649 → 应写 2.36** | `neural_geolife_bj500m_lstm.json` |
| 4 | **highD 100 箱、只看 k = 2 步的 LSTM 1.45** | **1.4447 → 应写 1.44** | `hist/hist_highd_v100_k2_s0.json` |
| 4 | 出租车记忆网络节省 1.11 bits（替代 0.37），模块 172 → 4,249，70% / 40% 的格子跨模块 | 1.1087（0.3736），172 → 4,249，69.8% / 40.4% | `prediction/results/infomap/infomap_summary.csv` |
| 4 | highD 跨箱后下一步返回 43%，一阶链预测 6%，进入第三箱不到 1% | 42.7%、6.1%、0.86%（新补的脚本） | `prediction/results/ngram/highd_recrossing.json` |

论文数字核对：`python tools/check_paper_numbers.py` → `summary: {'match': 148}`（148 项，含 Hurst 8 项和 highD 跨箱 3 项；期望值已按下面的更正写入）。

## 结论

第 1–3 组的全部结果文件都能用原始数据和现有代码逐位复现，四组里你指定的数字都有出处。第 4 组的数字与保存的 JSON 和 Infomap 结果一致，只有两处舍入错误。神经网络和 Infomap 的重算要在本机做：Infomap 的输入文件已验证逐字节相同；GPU 训练不是逐位确定的，预计相差约 0.01 bits。

## 问题与改动

论文要改（英文版和中文版都改）：

1. GeoLife 完整 LSTM 2.37 → 2.36：表 3 的 LSTM 列、表 4 的“128 steps”列、§8.2 “(2.37 and 2.64 bits …)”。§8.2 “2.37, 2.26 and 2.29 for k = 2, 4 and 8”中的 2.37 是 k = 2 的值（2.3655），不用改。
2. highD 100 箱、k = 2 的 LSTM 1.45 → 1.44：表 4 和 §8.2 “from 1.45 (k = 2)”。
3. §7.2 火车区间 1.11–1.93 → 1.12–1.93（见下面第 5 条）。
4. §7.2 “60% of train and 33% of subway time is linear interpolation”：60% 和 33% 是各分段插值样本比例的中位数，按总时间加权是 42% 和 38%。建议改为 “in the median segment, 60 % (train) and 33 % (subway) of the 5 s samples are linear interpolation”。

代码改动（本次提交）：

5. `controls/row_regression.py`：用户聚类自助抽样时，用户顺序原来取自 `row_persistence.csv` 的行序，而行序由多进程谁先算完决定，所以每次运行的区间会有约 0.01 的变化（`geolife_rerun_comparison.csv` 里已经出现过 1.12 ↔ 1.11、0.71 ↔ 0.72、0.37 ↔ 0.36）。现在先按用户编号排序再抽样。系数不变；论文引用的区间里只有火车下限变为 1.12（与重新下载数据前的值相同）。`row_regression.csv`、`row_regression_split.csv` 已更新。
6. `controls/hurst_control.py`：AR(1) 替代序列和打乱序列的种子原来是 `hash((完整路径, gap))`。Python 的 `hash()` 每次运行都不同，路径又因机器而异，所以这部分结果不可复现（`geolife_rerun_comparison.csv` 里的末位变化就来自这里）。现在改为 `crc32(文件名|gap)`，并重跑了 `hurst_control.csv`。确定性的列不变；随机的两列中位数变化如下（旧 → 新）：30 min 分段的 AR(1) 替代：公交 0.920 → 0.920，小汽车 0.981 → 0.981，步行 0.830 → 0.829；打乱序列 0.596 / 0.593 / 0.586 → 0.592 / 0.593 / 0.586；30 s 分段的 AR(1)：0.938 / 0.983 / 0.766 → 0.934 / 0.986 / 0.770；步行“观测减替代”的中位数 0.0485 → 0.0508。论文写的 0.83–0.98、约 0.59、0.05 都不变。另用不同的 `PYTHONHASHSEED`、在另一个进程里重算了 38 个分段，结果与新文件逐位相同。
7. `controls/origin_memory.py`：出租车逐车部分（论文未用）也用了 `hash()`，同样改为 `crc32(文件名)`。
8. `controls/row_persistence.py`：删去一个用 `hash()` 播种、但从未使用的随机数生成器；重跑后结果不变。
9. `controls/geolife_robust.py`：置换次数默认值 30 → 50（已提交的结果用的是 50，最小 p 值为 1/51）。
10. `controls/taxi_origin.py`：查明已提交的结果用的正是 `taxi_pick.json` 的前 100 个文件，脚本在完整数据上会选到同样的 100 辆车，结果逐位相同。只更新了 `controls/README.md` 的说明。
11. 新增 `prediction/highd_recrossing.py`：论文里“返回 43%、6%、不到 1%”原来没有对应的脚本。
12. `prediction/ch6_networks.py`：`.net` 文件固定用 LF 换行，这样 Windows 上生成的文件也与 Infomap 实际读入的逐字节相同；`prediction/scripts/run_infomap.bat` 固定安装 `infomap==2.8.0`。
13. 新增 `tools/check_paper_numbers.py`、`EXPERIMENT_LOG.md`、`experiment_log/`（含模板）、`CLAUDE.md`。

另外，早先的核对报告把 highD 增益换算成 0.110 bits，这是我换算时的错误，应为 0.109（0.07559 nats ÷ ln 2）。论文用的是 nats，不受影响。

本机另有一个独立的复现包 `D:\desktop\paper\paper_latex_v2\reproduce\`（以论文附带的脚本为基础，含 `compare.py` 和这次的比对表）。

## 下一步

- [ ] 论文按上面 1–4 条修改（英文版和中文版）。
- [ ] 在本机跑一遍 controls 全流程，确认 Windows 下逐位一致：`python tools/check_paper_numbers.py` 全部为 match。
- [ ] 在本机重跑 `prediction/ch6_networks.py taxi` 和 Infomap 2.8.0；神经网络在 GPU 上重跑，预期与已存结果相差约 0.01 bits。
- [ ] 可选：`geolife_robust.py`、`hurst_control.py`、`row_persistence.py`、`memory_by_mode.py`、`origin_memory.py` 用 `imap_unordered` 收集结果，输出 CSV 的行序每次不同（数值相同），git diff 会出现只有行序的变化。可改为 `imap` 或写出前排序；`memory_by_mode_speed.csv` 的 4×10⁻¹⁶ 差异也会随之消失。
