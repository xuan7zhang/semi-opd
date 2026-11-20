# Semi-OPD 在 ALFWorld 上的崩溃

Semi-OPD（[arXiv:2610.11291](https://arxiv.org/abs/2610.11291)：只用初始学生 π0 采一次的离线 rollout 做 on-policy 蒸馏）
在多轮 agent 任务 ALFWorld 上**完全崩溃**：三个 horizon 下成功率都在第 10 步降到 0，并且一直保持 0。
同一套代码、同一个教师下的在线 OPD 正常学习。本仓库是复现这一现象的代码、作业脚本和原始结果。

## 结果

学生 Qwen3-1.7B（非 thinking，LoRA r64，lr 1e-5），教师 Qwen3-4B-Instruct-2507；每步 8 个任务 × 8 局，
目标为逐 token 采样反向 KL。评测为 64 个 valid_seen 游戏 × 2 次，温度 0.4，单种子。

评测成功率（%），括号内为每轮平均回答 token 数（上限 512）：

| 步 | 0 | 10 | 20 | 30 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|
| OPD, H=10 | 9.4 (95) | 7.0 (348) | 16.4 (424) | 19.5 (342) | 18.0 (387) | 15.6 (387) | 18.0 (375) | 18.8 (368) |
| **Semi, H=10** | 7.8 (95) | **0.0** (238) | 0.0 (512) | 0.0 (512) | 0.0 (511) | 0.0 (507) | 0.0 (506) | 0.0 (503) |
| OPD, H=20 | 10.9 (100) | 11.7 (367) | 21.1 (455) | 28.9 (365) | 25.0 (404) | 25.0 (404) | | |
| **Semi, H=20** | 10.9 (98) | **0.0** (268) | 0.0 (512) | 0.0 (512) | 0.0 (508) | 0.0 (508) | | |
| OPD, H=40 | 15.6 (100) | 15.6 (387) | 18.8 (469) | | | | | |
| **Semi, H=40** | 12.5 (100) | **0.0** (268) | 0.0 (512) | 0.0 (512) | 0.0 (509) | | | |

H=20 与 H=40 截至上传时仍在训练。教师自身在 H=20 / H=40 的成功率为 27.3% / 32.0%，与学生的 Overlap@64 为 48.2%。
完整数据：`results/eval_summary.csv`、`results/metrics/*.jsonl`。

## 崩溃是什么样子

不是不行动，而是**输出格式坏掉、长度失控**（`results/eval_actions_sample.json`，Semi, H=10）：

| 步 | 解析出的"动作"（出现最多的几种） | 每轮 token |
|---|---|---|
| 0 | `inventory`、`look`、`go to cabinet 1`（正常） | 95 |
| 10 | `$$\boxed{\text{inventory}}$$`、`**action:** inventory`（格式错误，环境不认） | 238 |
| 20 | `urrent observation is "nothing…`（写到 512 被截断，没有动作） | 512 |

OPD 的动作始终合法（`go to drawer 1`、`open cabinet 1`），每轮长度稳定在 370–420，与教师接近。

## 我们的解释

Semi 只在 π0 自己写的短回答（约 100 token）上，按教师的逐 token 偏好调整学生。教师的写法与学生差异很大
（常用 `\boxed`、`**action:**`、长推理），于是学生被推离规定的 `<action>` 格式，也学会"晚点结束"。
但 Semi 从不看到学生改变之后自己写出的文本，没有任何信号告诉它"这个格式不对"或"该停了"，于是格式与长度一起失控。
在线 OPD 每一步都在学生自己的新输出上打分，这类偏离会立刻被纠正。

因此问题是**每一轮内部的输出分布漂移**，而不是长 horizon 下后段状态覆盖不足：Semi 在 H=10 就已完全失败。

对照：合作者在 ALFWorld 上用**同一环境中 RL 训练出的同架构教师**（Qwen3-4B，46%）时，Semi-OPD 没有崩（20.5 → 48.0，
在线 OPD 45.5）。我们的推测是：教师的输出格式与学生一致时 Semi 可用；不一致时 Semi 会崩，而 OPD 不会。
这一点尚待验证（例如换格式一致的教师，或统计教师在 π0 格式 token 上的优势）。

## 代码

```
alfworld/alf_opd.py    多轮 OPD 训练与评测；--semi 即 Semi-OPD（训练 rollout 一律来自 π0）
alfworld/common.py     教师打分（vLLM 服务 prompt_logprobs）与 LoRA 学生
alfworld/alf_probe.py  策略在 ALFWorld 上的成功率，以及师生 Overlap@64
slurm/*_alf_opd.sbatch 3 GPU 作业：GPU0 学生 vLLM + LoRA，GPU1 教师服务，GPU2 训练
slurm/submit_lh.sh     提交 OPD / Semi × H∈{10,20,40}（3 小时一段的续跑链）
PLAN.md                实验方案（假设、门槛、阶段）
```

Semi 的实现只有一处改动：训练时用不挂 LoRA 的 π0 采样，种子与 OPD 相同。π0 固定，所以逐步采样与训练前一次性采好全部
6,400 局等价；优势 A_t = log π_T − log π_θ 每步仍用当前 θ 计算。任务顺序、种子、评测与 OPD 完全一致。

运行示例（3 × GPU，教师选 4B-Instruct）：

```bash
sbatch --export=ALL,TEACH=4bi --job-name=alf-Semi4-H10 slurm/killarney_alf_opd.sbatch --estimator A --semi --H 10
sbatch --export=ALL,TEACH=4bi --job-name=alf-OPD4-H10  slurm/killarney_alf_opd.sbatch --estimator A --H 10
```

依赖：vLLM 0.11.2、torch 2.9、transformers 4.57、peft；ALFWorld 环境封装与 prompt 模板来自
[SDAR](https://github.com/ZJU-REAL/SDAR)（verl-agent 的 `agent_system`）以及一个按动作序列重放游戏的 `alfreplay.Game`
（未包含在本仓库）。作业脚本里的账户与 `/scratch` 路径是 Alliance 集群上的，需要按自己的环境修改。
`alf_opd.py` 里的其他估计器（B1/B2/C/D1）来自另一项多轮 OPD 状态回避实验，本实验只用估计器 A。
