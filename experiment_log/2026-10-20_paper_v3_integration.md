# 2026-10-20 把补充分析的结果写进论文（v3）；论文数字核对扩充到补充检验

## 目的

把补充分析待办清单（P0-1 – P0-4、P1-1 – P1-3、P1-5、P1-6）的结果按各项事先写好的判据写进论文（清单 P2-1 中能做的部分），英文稿 `main.tex` 和中文阅读版 `main_zh.tex` 同步；同时把论文里新增的数字纳入 `tools/check_paper_numbers.py`，以后重跑时能自动核对。

本次没有新的计算实验，只有：重画补充图和图 8、核对论文数字。

## 数据

没有读原始数据。所用结果文件都是 2026-10-19 – 10-20 生成并已提交的（见 [`2026-10-19_paper1_verification.md`](2026-10-19_paper1_verification.md) 第二部分）：

| 来源 | 文件 |
|---|---|
| `controls/results/` | `null_calibration_*.csv/json`、`highd_accel_noise*.csv`、`highd_reaction_delay*.csv`、`stops_*.csv/json`、`nonstationarity_geolife.csv`、`route_*.csv`、`origin_direction_*.csv`、`taxi_dejitter*.csv`、`hurst_power_*.csv`、`three_layers` 用到的已有结果 |
| `prediction/results/ngram/` | `taxi_network_controls.csv` |

## 实验

| 命令 | 说明 | 环境 |
|---|---|---|
| `python controls/figures/fig_supplement_checks.py` | 5 张补充图按论文尺寸重画：宽 16 cm，图中文字 8.5–9.5 pt，长图例移到图注，图例放在图下方不与坐标轴重叠；配色 Nature 色系（红、深蓝、青绿、浅橙、灰紫） | 云端 Linux，Python 3.13.16，matplotlib 3.11.2 |
| `python controls/figures/fig_three_layers.py` | 图 8 从结果表重画：(d) 标题改为 “Trip-level component”，标签 $I_3$、$I_O$、$I_{O\|3}$；原来的青蓝 + 咖啡色改为深蓝 + 浅橙；路权图例移到图下方；新增 `fig_three_layers.png` | 同上 |
| `python tools/check_paper_numbers.py` | 新增第 5 组 “5 checks”，134 项，覆盖论文 v3 新增的数字（§4.1 零模型校准、§4.4 Hurst 功效、§5.4 测量噪声与反应延迟、§6.4 去抖动、§7.2 停站、§7.3 等量样本与分岔度、§7.4 出行层面成分、§8.3 表 5、附录 B） | 同上 |
| 独立核对（子代理） | 不看写作过程，逐条把 `main.tex` 新增的数字和结果文件对照，并检查中英文一致、交叉引用和编译日志 | — |
| LaTeX 预览 | 英文 pdflatex（本地无 elsarticle.cls，用替身类）、中文 xelatex | TeX Live 2023 |

## 结果

论文数字核对：`python tools/check_paper_numbers.py` → `summary: {'match': 282}`（148 项原有 + 134 项新增；改正下面 4 处之前为 278 match、4 MISMATCH）。

核对查出的论文 v3 草稿中的舍入错误（都已在英文、中文两版改正）：

| 位置 | 草稿 | 结果文件 | 改为 |
|---|---|---|---|
| §7.3 等量样本，自行车相对超额 | 21% | 20.4995%（`route_equal_sample.csv`，10 次抽取各自中位数的均值） | 20% |
| 表 5，数据的码长节省 | 1.110 | 6.66059 − 5.55111 = 1.10948（`taxi_network_controls.csv`） | 1.109 |
| 表 5，合并替代的节省 | 0.364 ± 0.017 | 种子 1–3 的均值 0.36349 | 0.363 ± 0.017 |
| §4.4，公交真实片段 DFA 检出率 | 5% | 4.49%（`hurst_power_real.csv`） | 4% |

后三处的错误数字也出现在 2026-10-19 记录第二部分的表里，已在原处注明更正。另外 §4.4 “ARFIMA 检出 0–13%”只对 H = 0.7 成立（H = 0.8 时小汽车的 DFA 为 22%），改为 “with the same $H$”。

独立核对还查出措辞上与数据不符或比数据强的地方（数字本身都对），已在两版中改正：

- highD：未改动的旧文字仍说 AR(2) 拟合表明约三分之一是“真实二阶结构”、讨论中说“三个系统都有真实记忆”，与新的 §5.4（剩余是测量噪声）矛盾。改为：AR 模型是在差分速度上拟合的，其二阶结构包含测量噪声；真实记忆限于 GeoLife 和出租车的路径。
- 零模型“略微低估” → 低估 20–30%（数据越少越多）；“下限”只用于校准过的 10 m 网格。
- 时段切换（P1-1）：27% 是相对未分层超额的比例；相对同样细的随机分层所保留的部分是 54%（打乱标签保留 49.3%、真实标签 22.5%）。两个都报告，结论改为“只有一部分来自时段切换”。
- 反应延迟：“没有反应延迟依赖” → “在反应时间附近没有信息峰”（滞后 Δv 在三个预测尺度上都高于零模型）；前车加速度的 0.006 / 0.018 bits 注明是未减零模型的原始增益。
- §7.2：ρ = 0.93 是 τc 与速度，改为停站频率与平均速度的 ρ = −0.93（`row_persistence.csv`，gap 30，七种方式的中位数）。
- §7.3：自行车、地铁只有 10 名、出租车和火车只有 9 名合格用户（10 次抽取用的是同一批人），已注明。
- §6.4：29% 是每条轨迹删去比例的中位数（按总数为 39%）。
- 记忆网络：去抖动后数据的节省低于两种替代（0.18 对 0.33 / 0.42），不只是“去掉大部分”。
- Hurst：对带短程持续性的长记忆（ARFIMA）几乎检不出，摘要和引言里补上。

改动脚本在论文工作目录的 `edits/fix_numbers.py`（在 `edit_en.py` / `edit_zh.py` 之后运行）。

结果文件：本次没有重跑任何分析，所有结果文件不变。重新生成的只有图（输入数据不变，只改版式）：

| 图 | 变化 |
|---|---|
| `fig_s_null_calibration.pdf/png` | 字号放大；(a) 图例只留 GeoLife / taxi，假阳性率写进图注 |
| `fig_s_accel_noise.pdf/png` | 字号放大；(b) 单位改为 mnat，横轴标签缩短 |
| `fig_s_reaction_delay.pdf/png` | 字号放大；纵轴为“增益 − 零模型均值”（mbit）；图例移到图下方 |
| `fig_s_stops.pdf/png` | 字号放大；图例移到图下方 |
| `fig_s_hurst_power.pdf/png` | 字号放大；真实数据画在横轴 0.9 处并标 “data” |
| `fig_three_layers.pdf`（+ 新 `.png`） | 见上 |

## 结论

论文 v3 已按判据改写；新增数字全部能从已提交的结果文件逐项复算（282 项一致）。核对后论文的主线更清楚：GeoLife 和出租车的路径有小而浅的真实二阶记忆，高速公路速度在 (v, a) 上是一阶的，其余“记忆”是估计偏差、离散化、测量噪声或单元之间的差异。还没有写进论文的只有 P0-5（神经网络阳性对照，要本机 GPU）和 P1-4（地图匹配）。

## 问题与改动

1. `tools/check_paper_numbers.py`：新增第 5 组（134 项）。
2. `controls/figures/fig_supplement_checks.py`、`controls/figures/fig_three_layers.py`：按论文尺寸重画（见上表）。
3. `experiment_log/2026-10-19_paper1_verification.md`：第二部分表中 3 个舍入错误原处注明更正。
4. 论文（不在仓库里，在本机 `D:\desktop\paper\paper_latex_v2`）：`main.tex`、`main_zh.tex` 改写，修改前的版本保存为 `main_v2_before_1020.tex`、`main_zh_v2_before_1020.tex`；改动清单见同目录 `修改说明_v3.md`。

## P0-5 本机运行情况（10-20 晚）

`run_positive_control.bat` 10-20 早上在本机启动（GeForce MX250，2 GB 显存），两条通道并行。到 20:40 的进度：

| 通道 | 完成 | 说明 |
|---|---|---|
| 阳性对照（60 个模型） | 3 个：pc_L16_I2 种子 0 的 LSTM k = 4、LSTM 128 步、Transformer k = 4 | 128 步 Transformer 第 2 轮用了 38,735 − 370 s（约 10.6 h） |
| Transformer 历史长度消融（11 个） | k = 1、2、3、4 | k = 8 的第 2 轮用了约 65 min，11:27 以后无输出 |

原因有两个：白天电脑休眠过（用户确认），休眠时进程冻结，墙钟时间照算；另外两个 Transformer 并行时互相拖慢——两者第 0 轮都用了约 6 min，而单独跑时 128 步 Transformer 每轮约 0.5 min（`log_taxi_traj20_transformer.txt`：15 轮共 421 s）、k = 4 约 1.5 min，并行的总吞吐约为依次运行的三分之一。

已完成的第一组（pc_L16_I2，种子 0；测试交叉熵，bits/步）：LSTM k = 4 为 2.871，LSTM 128 步为 2.881，Transformer k = 4 为 2.874。植入的依赖约 0.20 bits/步，128 步模型若学到应比 k = 4 低约 0.2 bits，实际没有。只是一个数据集一个种子，等全部结果再按判据下结论。

改动：

1. `prediction/scripts/run_positive_control.bat`：两条通道改为在同一窗口依次运行（先阳性对照、再 Transformer 消融），最后自动运行 `positive_control.py summary` 和 `summarise_neural.py`（日志在 `results/neural/`）。已完成的运行仍会跳过，没跑完的会从头重跑。
2. 新增 `prediction/keep_awake.py`；`neural_hist.py`、`neural_ch5.py` 在导入时调用它：在 Windows 上训练期间不进入空闲睡眠（屏幕可以关；合盖仍按电源计划处理）。只影响电源请求，不影响训练和结果。
3. `prediction/README.md`：相应说明，预计总耗时约 12–18 小时。

## 下一步

- [ ] 本机 GPU 跑 P0-5（`prediction\scripts\run_positive_control.bat`），然后替换论文 §8.2 的 `%% TODO P0-5`，决定“可用记忆止于三四步”的说法。
- [ ] 本机用 highD 全部 60 段重跑 `highd_accel_noise.py`、`highd_reaction_delay.py`（及 `--lead-accel`），把论文 §5.4 和附录里“23 段录像”换成全部数据。
- [ ] 摘要压缩（现在约 540 词）；P2-2 – P2-4 的写作。
- [ ] 正文其他图（图 1–7、9–11）仍是旧版式（图中文字约 7 pt），按同样规则从结果表重画。
