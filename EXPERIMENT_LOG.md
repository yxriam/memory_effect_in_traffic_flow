# 实验记录 · Experiment log

这里统一记录本仓库的每一次实验、重跑和结果核对：**数据 → 实验 → 结果 → 下一步**。

每次做完实验：

1. 按 [`experiment_log/TEMPLATE.md`](experiment_log/TEMPLATE.md) 在 `experiment_log/` 新建 `YYYY-MM-DD_<主题>.md`；
2. 在下面的总表加一行；
3. 更新“当前的下一步”；
4. 重跑过结果时，运行 `python tools/check_paper_numbers.py`，把 summary 行写进记录；
5. 用 `sync_github.bat "log: YYYY-MM-DD <主题>"` 推到 GitHub。

单位：CMI 用 bits；highD、`coarse_model`、`memory_by_mode_speed` 的留出似然增益用 nats（论文换算成 bits 时除以 ln 2）；预测交叉熵用 bits/step。

## 总表

2026-10-19 之前的各行是补记，日期取自当时结果文件的修改时间（新西兰时间）。

| 日期 | 主题 | 数据 | 实验 | 主要结果 | 下一步（当时） | 详细记录 |
|---|---|---|---|---|---|---|
| 2026-04-26 | 出租车 CMI 与三阶记忆初探（补记） | 上海出租车 `Taxi_070220` | 本机 `taxi/memory_effect_full.py`、`memory_third_order.py`（未入库） | 结果在本机 `taxi/memory_full_summary.txt`、`memory3rd_summary.txt`；已被 2026-10-09 的 notebook 取代 | 统一平滑方法、调 β | — |
| 2026-05-28 | highD 马尔可夫阶数、似然比检验、VACF/GLE 记忆核（补记） | highD 60 段录像，0.2 s，20 个速度状态 | `highD/highd_memory_analysis.ipynb` | 二阶比一阶留出对数似然 +0.103 nat/step（H(S_{t+1}\|S_t) 的 26%）；LRT G = 1.08×10⁶；“29% 记忆分数”随积分窗口变号，不再报告 | 检验二阶增益是否来自状态不完整（→ 2026-10 controls） | [`highD/highd_experiment_record.md`](highD/highd_experiment_record.md) |
| 2026-10-09 | 出租车：CMI 置换检验、add-α 与回退平滑、β 调参、全局网格（补记） | 出租车 4,307 条轨迹；10×10 / 20×20 / 30×30 网格 | `taxi/taxi_memory_analysis.ipynb` | 20×20：CMI₂ 中位数 0.027 nat，55% 轨迹显著；add-α 下二阶反而更差（−0.076），β 调参后 +0.008 | 用置换零模型、停留压缩把表观记忆和真实记忆分开 | `taxi/memory_full_summary.csv`、`taxi/memory_backoff_tuned_summary.csv` |
| 2026-10-09 – 10-14 | 论文第二版 controls：表观记忆与真实记忆（补记） | GeoLife 181 名用户；highD 60 段；出租车 100 辆 | `controls/*.py` | GeoLife 零模型 0.325、校正后 0.012 bits；highD 0.076 → 0.007 nats；τ_c 与停车频率 ρ = −0.98；路径记忆公交 29%、小汽车 8% | 记忆在预测里能否用上（→ prediction） | [`controls/README.md`](controls/README.md) |
| 2026-10-13 – 10-14 | GeoLife 改用官方 1.3 数据后重跑（补记） | GeoLife 1.3 官方下载 | `geolife/` 流程 + controls | 73 项逐项新旧对比；主要数字不变，Hurst 替代序列和自助区间有末位变化（后来查明是随机种子问题，见 2026-10-19） | — | `controls/results/geolife_rerun_comparison.csv` |
| 2026-10-17 – 10-18 | 预测与记忆网络：n-gram、LSTM / Transformer、历史长度消融、Infomap（补记） | `taxi_traj20`、`geolife_bj500m`、`highd_v20` / `highd_v100` | `prediction/`（GPU：GeForce MX250；Infomap 2.8.0） | 出租车 LSTM 2.90 bits，比 Kneser–Ney 低 0.72；只看 2 步历史的 LSTM 已拿到大部分增益；出租车记忆网络节省 1.11 bits（替代序列 0.37） | 核对论文数字 | [`prediction/README.md`](prediction/README.md) |
| 2026-10-19 | 论文一四组数字核对与复现 | 上面全部原始数据（GeoLife 缓存 181 个、按方式轨迹 253 个、highD 60 个、出租车 100 辆、prediction 的 npz） | 全量重跑 controls（第 1–3 组）；第 4 组核对 JSON、日志，并重新生成 Infomap 输入 | 第 1–3 组结果文件逐位复现；查出 2 处舍入错误和 1 处措辞问题；修了自助抽样的顺序依赖、3 处 `hash()` 种子和置换次数默认值；`check_paper_numbers.py` 148 项全部一致 | 见下 | [`experiment_log/2026-10-19_paper1_verification.md`](experiment_log/2026-10-19_paper1_verification.md) |
| 2026-10-19 – 10-20 | 补充分析待办清单（P0-1 – P0-4、P1-1 – P1-3、P1-5、P1-6；P0-5 的数据与 CPU 部分） | GeoLife 缓存 181 个、按方式轨迹 253 个；出租车全部 4,316 条；highD 原始发布版 23 段录像（新键 `highd_full`）；`taxi_traj20` | `controls/` 新增 10 个脚本、`prediction/positive_control.py`、`prediction/taxi_network_controls.py`；Infomap 2.8.0 | 零模型不多报（假阳性 1.1% / 3.0%）但超额 CMI 偏低（回收 70–79%）；highD 剩余记忆是差分噪声，2 s 内无反应延迟峰；无停站片段上方式差异仍在、控制速度后停站系数为正；出租车记忆网络节省大部分来自车辆差异（逐车替代 0.97 对数据 1.11 bits）和 A→B→A 抖动（去抖后 0.18，低于替代）；起点信息在出租车里是车辆和时段的代理；R/S、DFA 在这些长度上功效低；路线约束无直接支持 | P0-5 在本机 GPU 跑；P1-4 本机做；按判据改论文 | [`experiment_log/2026-10-19_paper1_verification.md`](experiment_log/2026-10-19_paper1_verification.md)（第二部分） |
| 2026-10-20 | 把补充分析结果写进论文 v3；论文数字核对扩充 | 无新数据；用 10-19 – 10-20 的结果文件 | 重画补充图 5 张和图 8；`check_paper_numbers.py` 新增第 5 组（134 项）；独立核对 | 282 项全部一致；查出并改正论文草稿 4 处舍入（自行车 20%、表 5 的 1.109 和 0.363、公交 DFA 4%）和若干过强措辞（highD 无真实记忆、时段切换占 27–54%、零模型低估 20–30%） | P0-5 GPU；highD 60 段重跑；摘要压缩；其他图重画 | [`experiment_log/2026-10-20_paper_v3_integration.md`](experiment_log/2026-10-20_paper_v3_integration.md) |
| 2026-10-20 – 10-22 | P0-5 阳性对照（本机 GPU）；神经网络训练改为可断点续跑 | 5 组合成序列（`taxi_traj20` 的片段与划分，植入 L = 8–32 步前的依赖 0.1–0.3 bits/步） | `run_positive_control.bat`：60 个模型（LSTM / Transformer × k = 4 / 128 步 × 3 种子），GeForce MX250，torch 1.10 | 两种网络都学不到植入的依赖：128 步 LSTM 比 4 步差 0.003–0.011 bits，Transformer 差 0.21–0.23 bits，可得增益 0.10–0.30 bits → 论文删去“可用记忆止于三四步”；`check_paper_numbers.py` 293 项一致 | 续跑 Transformer 消融剩下的 6 个；续跑测试 | [`experiment_log/2026-10-22_p05_positive_control.md`](experiment_log/2026-10-22_p05_positive_control.md) |
| 2026-10-22 | 真实出租车数据上的 Transformer 历史长度消融（P0-5 第二项） | `taxi_traj20` | `lane_transformer_ablation.bat`：k = 1–32、k = 4 三个种子、128 步三个种子；GeForce MX250 | 4.44 / 3.05 / 2.83 / 2.80 / 2.77 / 2.79 / 2.82 bits（k = 1–32），128 步 3.22；k ≥ 4 的差别在种子波动内，与 LSTM 一致 | 论文写作；highD 60 段重跑 | 同上，第二部分 |

## 当前的下一步

- [x] 论文改 4 处（英文版和中文版都改；2026-10-20 在论文 v3 中完成）：GeoLife 完整 LSTM 2.37 → 2.36；highD 100 箱、k = 2 的 LSTM 1.45 → 1.44；§7.2 火车区间 1.11–1.93 → 1.12–1.93；§7.2 “60% / 33% of time”改为分段中位数的说法。详见 2026-10-19 的记录。
- [x] 按补充分析的判据改写论文（2026-10-20 完成论文 v3，P0-5 部分除外；待办清单 P2-1；2026-10-19 记录第二部分“对论文的影响”）：出租车记忆网络（§4.3）、停站（§7.2）、路线约束与“公交最强”（§7.3）、“purposive”（§7.4）、highD 的 0.076 nats 来源与 0.010 bits 表述、Hurst 改为弱表述；补充图用 `controls/figures/output/fig_s_*.pdf`。
- [x] 本机 GPU：P0-5 阳性对照 60 个模型（2026-10-22 完成：检不出植入的长依赖，论文 §8.2 已按判据改写）。
- [x] 本机 GPU：真实数据上的 Transformer 历史长度消融（2026-10-22 02:06 完成，结果与 LSTM 相同：信息集中在前 3–4 步；已补进论文表 4 和 §8.2；`check_paper_numbers.py` 304 项一致）。
- [ ] 本机：`local_paths.json` 已加 `highd_full`；用 highD 全部 60 段重跑 `controls/highd_accel_noise.py`、`controls/highd_reaction_delay.py`（及 `--lead-accel`），确认 23 段上的结论不变；之后把论文 §5.4 和附录 B 的“23 段录像”换成全部数据。
- [ ] 本机：P1-4 地图匹配（需要 OpenStreetMap 路网）。
- [ ] 在本机跑一遍 controls 全流程，确认 Windows 下也逐位一致：`python tools/check_paper_numbers.py` 应全部为 match。
- [ ] 在本机重跑 prediction：`ch6_networks.py taxi`（要读全部 4,316 个出租车文件）和 Infomap 2.8.0；神经网络在 GPU 上重跑，预期与已存结果相差约 0.01 bits。（2026-10-20：云端用 Infomap 2.8.0 命令行版重算出租车网络，码长与已提交结果相差 < 0.002 bits，见 `prediction/results/ngram/taxi_network_controls.csv` 的 data 行。）
- [ ] 可选：多进程脚本的输出行序不固定（数值相同），改为 `imap` 或写出前排序，让 git diff 只显示真实变化。
- [ ] 论文写作：摘要从约 540 词压缩到目标期刊的要求（Elsevier 常见 ≤ 250 词）；P2-2（讨论中的“慢变量记忆”）、P2-3（参考文献核对）、P2-4（后续工作）。
- [ ] 正文其他图（图 1–7、9–11）按同样的版式规则从结果表重画（图中文字与正文同大、无重叠、长图例进图注、Nature 配色）。
- [x] 补充分析待办清单中能在云端做的实验（2026-10-20 完成：P0-1 – P0-4、P1-1 – P1-3、P1-5、P1-6；P0-5 完成数据与 CPU 部分）。
