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

## 当前的下一步

- [ ] 论文改 4 处（英文版和中文版都改）：GeoLife 完整 LSTM 2.37 → 2.36；highD 100 箱、k = 2 的 LSTM 1.45 → 1.44；§7.2 火车区间 1.11–1.93 → 1.12–1.93；§7.2 “60% / 33% of time”改为分段中位数的说法。详见 2026-10-19 的记录。
- [ ] 在本机跑一遍 controls 全流程，确认 Windows 下也逐位一致：`python tools/check_paper_numbers.py` 应全部为 match。
- [ ] 在本机重跑 prediction：`ch6_networks.py taxi`（要读全部 4,316 个出租车文件）和 Infomap 2.8.0；神经网络在 GPU 上重跑，预期与已存结果相差约 0.01 bits。
- [ ] 可选：多进程脚本的输出行序不固定（数值相同），改为 `imap` 或写出前排序，让 git diff 只显示真实变化。
