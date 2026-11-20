# 长 horizon 下的 Semi-OPD：问题诊断与改进（ALFWorld）

## 动机

Semi-OPD 在单轮数学上复现成功（第 60/80/100 步平均 39.3，论文 39.8），但它的偏差上界
`2GT·√(½KL(Pθ‖P0))` 随 horizon T 线性增长：π0 早早失败、走不到后面的状态，缓存里没有这些状态，
而训练后的学生恰恰会访问它们（DAgger 的协变量偏移）。同时，我们在单轮实验里看到"直接用新 rollout 刷新缓存"
会让结果变差（12 个刷新 run 全部低于 Semi，见 `../semiopd/results_agg_4bi.csv`）。
所以改进必须在**尽量少用新 rollout** 的前提下修正状态分布。

## 设置（沿用 sdi 的 ALFWorld 多轮 OPD，保证与已有 OPD 基线可比）

| 项 | 值 |
|---|---|
| 环境 | ALFWorld 文本版，verl-agent 模板（history=2），含 `done` 动作；horizon H ∈ {10, 20, 40} |
| 学生 | Qwen3-1.7B + LoRA r64，lr 1e-5，非 thinking |
| 教师 | 默认 Qwen3-8B；阶段 0 后可能更换（见门槛 G0） |
| 训练 | 每步 8 个任务 × G=8 条轨迹 = 64 局，100 步；目标为逐 token 采样反向 KL（sdi 的估计器 A） |
| 评测 | 64 个 valid_seen 游戏 × 2 次，温度 0.4；成功率、退出率、重复动作率、每轮 token 数、每局步数 |
| OPD 基线 | sdi 已有：`alf-A-H10`、`alf-A-H20`、`alf-A-H40`（vulcan `/scratch/xzhan576/sdi/runs/`） |

## 方法

| 名称 | 说明 | 额外 rollout |
|---|---|---|
| OPD | 每步用当前学生采 64 局，教师打分（sdi 估计器 A，已有） | 每步 64 局 |
| **Semi** | π0 一次性采 6,400 局（= OPD 100 步的局数，任务顺序相同），教师一次打分；每步取同一批 64 局，用当前 θ 重算 A_t | 0 |
| **Semi-IS** | Semi + 轨迹级截断重要性权重：第 k 轮状态的权重 w_k = clip(exp(Σ_{j<k} ℓ_j), [1/c, c])，ℓ_j 为第 j 轮**动作片段**（解析出的 action 文本对应 token）上 log πθ − log π0 之和；按轮次在 batch 内自归一化。只有动作改变环境状态，推理 token 不进入权重 | 0 |
| **Semi-Curr** | 分段 Semi：H 分成长度 k 的段。第 s 段开始时，用当前学生从起点走到第 s·k 轮（roll-in），只把第 (s−1)k…sk 轮的新状态追加进缓存，旧数据保留 | 每段一次，仅前沿段计入训练 |
| Semi-V（可选） | Semi + 教师价值筛选：轨迹进入 V_T=0（教师也救不回）的状态后，其后各轮权重置 0 | 0（需算 V_T） |

## 假设与判定标准（单种子，评测误差约 ±4–6 个百分点）

- **H-A（问题存在）**：Semi 相对 OPD 的差距随 H 变差。判定：成功率差 (Semi − OPD，第 60/80/100 步平均) 在 H=40 比 H=10 至少低 6 个百分点。
  若 H-A 不成立（Semi 在长 horizon 上也不输），则"Semi 不适合长 horizon"不成立，直接报告 Semi ≥ OPD 这一结论。
- **H-B（改进有效）**：在 Semi 输得最多的 H 上，Semi-IS 或 Semi-Curr 比 Semi 高 ≥ 6 个百分点，且不低于 OPD − 6。
- **H-C（没有逃避行为）**：Semi 系方法的退出率、重复动作率、每轮 token 数不出现 sdi 里 B1/B2 式的逃避（退出率 → 100%、重复率 → 1、长度 → 2–16 或 → 512）。
- **H-D（成本）**：Semi 系方法的学生 rollout 数与 GPU 卡时都低于 OPD。

## 门槛

- **G0（教师够强）**：在 H=20 与 H=40 上，教师成功率比 π0 高至少 15 个百分点。否则换更强的教师
  （候选 Qwen3-4B-Instruct-2507、Qwen3-30B-A3B-Instruct-2507），并在新教师下重跑 OPD 基线。
  同时测师生 Overlap@64（π0 每轮回答的 token，定义同论文）。

### G0 结果（2026-11-16，64 个 valid_seen 游戏 × 2，温度 0.4，含 done）

| 策略 | H=20 成功率 | H=40 成功率 | 与 π0 的 Overlap@64 |
|---|---|---|---|
| π0（Qwen3-1.7B） | 9.4% | 11.7% | — |
| Qwen3-8B | 16.4%（+7）❌ | 23.4%（+12）❌ | 54.5% |
| **Qwen3-4B-Instruct-2507** | **27.3%（+18）✅** | **32.0%（+20）✅** | **48.2%** |

决定：教师换成 Qwen3-4B-Instruct-2507（每轮约 360 token，学生约 100）。sdi 里 Qwen3-8B 下的 OPD 基线不再使用，
在新教师下重跑 OPD 与 Semi（H=10/20/40）。Overlap 48% 属于论文里 Semi 占优的低 overlap 区间。
Qwen3-8B 教师下的 Semi-H10（vulcan，已在跑）保留作参考。

## 阶段与算力（vulcan L40S；卡时为估计）

| 阶段 | 内容 | 估计 |
|---|---|---|
| 0 | 教师候选与 π0 在 H=20/40 的成功率；Overlap@64 | ~10 卡时 |
| 1 | Semi 在 H=10/20/40（与已有 OPD 基线比较） | 3 run × ~15 卡时 |
| 2 | 在 Semi 输得最多的 H 上跑 Semi-IS、Semi-Curr（必要时加 Semi-V） | 2–3 run × ~20 卡时 |
| 3（若换教师） | 新教师下的 OPD 基线 | 每个 H ~60 卡时 |

单种子；结果写入 `results.csv`；每个 run 保存配置、日志与 checkpoint 路径。
