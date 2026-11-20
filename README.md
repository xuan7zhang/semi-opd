# Semi-OPD 在 ALFWorld 上崩溃：设置、现象与分析

> 写给合作者：你在 ALFWorld 上跑的 Semi-OPD 没有崩（20.5 → 48.0，在线 OPD 45.5），我们这边同样的方法在 ALFWorld 上
> **完全崩溃**（成功率第 10 步起一直是 0）。这里把我们的全部设置、崩溃的具体样子、和你那边的差异列出来，方便对照排查。
> 文末有几个想请你确认的问题。

## 1. 一句话结论

在 Qwen3-1.7B ← Qwen3-4B-Instruct-2507 上，**在线 OPD 正常学习，Semi-OPD 在 H = 10 / 20 / 40 全部于第 10 步崩到 0%**。
崩溃形式是**输出格式被改坏、每轮长度涨到 512 上限**，环境解析不出动作。Semi 的训练损失却一直在漂亮地下降。

## 2. 结果

### 2.1 评测成功率（%；括号内为每轮平均回答 token 数，上限 512）

64 个 valid_seen 游戏 × 2 次，温度 0.4，单种子。

| 步 | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| OPD, H=10 | 9.4 (94) | 7.0 (348) | 16.4 (424) | 19.5 (342) | 18.0 (386) | 14.1 (376) | 15.6 (387) | 18.0 (375) | **18.8** (368) |
| **Semi, H=10** | 7.8 (95) | **0.0** (238) | 0.0 (512) | 0.0 (512) | 0.0 (511) | 0.0 (509) | 0.0 (507) | 0.0 (506) | **0.0** (502) |
| OPD, H=20 | 10.9 (100) | 11.7 (367) | 21.1 (455) | **28.9** (364) | 25.0 (404) | 26.6 (403) | 25.0 (404) | 训练中 | |
| **Semi, H=20** | 10.9 (98) | **0.0** (268) | 0.0 (512) | 0.0 (512) | 0.0 (508) | 0.0 (508) | 0.0 (508) | 0.0 | |
| OPD, H=40 | 15.6 (100) | 15.6 (387) | 18.8 (469) | 训练中 | | | | | |
| **Semi, H=40** | 12.5 (100) | **0.0** (268) | 0.0 (512) | 0.0 (512) | 0.0 (508) | 训练中 | | | |

参照：教师 Qwen3-4B-Instruct-2507 自己在 H=20 / H=40 的成功率是 27.3% / 32.0%（每轮约 360 token）；
未训练学生是 9.4% / 11.7%（每轮约 100 token）。OPD 在 H=20 第 30 步已达到教师水平。

### 2.2 训练侧：Semi 的损失看起来完全正常

逐 token 反向 KL（nats/token）与训练 rollout 每轮 token 数。Semi 的 KL 是在**固定的 π0 数据**上算的，OPD 的是在**学生自己的新 rollout** 上算的。

| 步 | 1 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|
| OPD, H=10：KL / 每轮 token | 1.82 / 106 | 1.35 / 145 | 0.95 / 325 | 0.76 / 458 | 0.57 / 361 | 0.48 / 356 |
| Semi, H=10：KL / 每轮 token | 1.84 / 102 | 1.63 / 106 | 1.29 / 106 | 0.60 / 107 | 0.22 / 103 | **0.18** / 103 |

Semi 在 π0 数据上把 KL 压到 0.18，比 OPD 还低，**但同一时间评测成功率是 0**。也就是说，只看训练曲线完全发现不了崩溃：
学生在 π0 的前缀上越来越像教师，而一旦让它自己生成，就走进了训练数据从未覆盖的区域。

### 2.3 崩溃具体长什么样

Semi, H=10 评测时解析出的"动作"（`results/eval_actions_sample.json`）：

| 步 | 出现最多的"动作" | 每轮 token |
|---|---|---|
| 0 | `inventory`、`look`、`go to cabinet 1`、`examine cabinet 1`（正常） | 95 |
| 10 | `$$\boxed{\text{inventory}}$$`、`entory.\n\n**action:** inventory`、`.\n\n**action:** go to cabinet 1` | 238 |
| 20 | `urrent observation is "nothing`、`action could also be useful if`（写到 512 被截断，没有动作） | 512 |
| 80 | `look`、`inventory` 偶尔能解析出来，但 `\boxed{}` 形式仍很多，任务不推进 | 506 |

学生不再用规定的 `<action>…</action>`，而是用教师常见的 `\boxed{}`、`**action:**` 写法，并且越写越长，直到截断。
OPD 的动作始终合法（`go to drawer 1`、`go to cabinet 1`、`open drawer 1`），每轮长度稳定在 360–420，与教师接近。
退出率与重复动作率在 Semi 下都是 0，所以它不是 sdi 里那种"提前退出/原地打转"的逃避行为。

## 3. 完整设置

| 项 | 设置 |
|---|---|
| 学生 | Qwen3-1.7B，非 thinking（`enable_thinking=False`） |
| 训练方式 | **LoRA r64 / α128**，作用于 q,k,v,o,gate,up,down；基座 bf16；AdamW **lr 1e-5**，wd 0，梯度裁剪 1.0 |
| 教师 | **Qwen3-4B-Instruct-2507**（通用 Instruct 模型，零样本，**没有在 ALFWorld 上训练过**） |
| 教师打分 | vLLM 服务（`--max-model-len 4096`），`prompt_logprobs=0` 取学生**同一 token 序列**上的 log π_T。序列用学生的 Qwen3 chat template（非 thinking 时生成提示末尾带空的 `<think></think>`），教师按同一串 token 打分 |
| 目标函数 | 逐 token 采样反向 KL：A_t = log π_T − log π_θ（detach），loss = −Σ A_t · log π_θ / batch 内 token 数（sdi 的估计器 A） |
| 每步数据 | 8 个训练任务 × G=8 局 = 64 局，训练采样温度 1.0，100 步 |
| Semi 的数据 | 每步用**不挂 LoRA 的 π0** 按与 OPD 相同的种子采这一步的 64 局。π0 固定，所以等价于训练前一次性采好全部 6,400 局；A_t 每步用当前 θ 重算。任务顺序、种子、评测与 OPD 完全相同 |
| 环境 | ALFWorld 文本版，verl-agent 的 prompt 模板（history = 2 步），**额外加了 `done` 动作**（"You may choose 'done' at any time to end the episode."，选了即按未完成结束） |
| 长度 | prompt 最多 2,048 token（超出从左截断），**每轮回答最多 512 token** |
| Horizon | H = 10 / 20 / 40（每局最多步数） |
| 评测 | 64 个 valid_seen 游戏（排序后前 64 个）× 2 次，温度 0.4，每 10 步一次 |
| 硬件 | 3 × L40S：GPU0 学生 vLLM（带 LoRA）、GPU1 教师服务、GPU2 LoRA 训练；3 小时一段续跑 |
| 软件 | vLLM 0.11.2、torch 2.9、transformers 4.57.3、peft 0.21 |

## 4. 与你那边（round-2 B 线）的差异

| | 你（没崩） | 我们（崩了） |
|---|---|---|
| **教师** | **同一 ALFWorld 环境里 RL 训练的 Qwen3-4B**（c3-step299，46%） | **通用 Qwen3-4B-Instruct-2507，零样本**（27–32%） |
| 教师与学生的关系 | 同架构，并在同一套 prompt / 动作格式下训练过 | 不同模型，没见过这个环境的格式；Overlap@64 只有 48% |
| 教师的输出风格 | 应与学生一致（短、格式规范）？ | 每轮约 360 token，常写 `\boxed{}`、`**action:**` |
| 学生 | Qwen3-4B，初始 20.5% | Qwen3-1.7B，初始 7.8–12.5% |
| 训练 | 全参数，fp32 主权重，lr 1e-6 | LoRA r64，lr 1e-5 |
| π0 数据 | 4,103 条，复用离线 RL 的 π0 dump | 6,400 局，按 OPD 的任务顺序和种子逐步采 |
| prompt / 动作格式 | AstraFlow 的 alfworld_task_server | verl-agent 模板 + `done` |
| 每轮回答上限 | ? | 512 token |
| Horizon | ? | 10 / 20 / 40 |
| 评测 | valid 50 任务 × 4 | valid_seen 64 × 2 |

## 5. 我们的分析

1. **崩溃来自每一轮内部的输出分布漂移，而不是长 horizon。** H=10 就已完全失败，三个 H 的崩溃曲线几乎一样。
   所以原先担心的"π0 走不到轨迹后段、缓存覆盖不到后段状态"不是主因。
2. **机制推测：教师在推着改格式和长度，而 Semi 没有纠错信号。** Semi 只在 π0 自己写的短回答（约 100 token）上，
   按教师的逐 token 偏好调整学生。教师的写法与学生差别很大，于是在 π0 的格式 token 上给出负优势，把学生推向
   `\boxed{}` / `**action:**` 和更长的推理。学生改变之后写出的新文本从不进入训练，没有信号告诉它"这个格式不对"或"该停了"。
   在线 OPD 每步都在学生自己的新输出上打分，这类偏离会立刻被纠正。
3. **为什么你那边没崩（推测）：** 你的教师是在同一环境、同一格式下 RL 出来的同架构模型，它在 π0 的格式 token 上不会给负分，
   监督只改"选哪个动作"，不改"怎么写"，于是 Semi 与 OPD 一样有效。如果这是对的，结论是：
   **Semi-OPD 是否可用取决于教师的输出格式/风格是否与学生一致**；不一致时 Semi 会崩而 OPD 不会。
   这对论文"低 overlap 时 Semi 更好"是一个补充：论文只评单轮数学，结尾一个 `\boxed{}` 即可；多轮 agent 每轮都要可解析的动作，格式漂移是致命的。
4. **和你 A 线（thinking 数学）的长度失控是同一类现象：** 你观察到教师对"继续想"的 token 几乎不给负优势、对 `</think>` 没有正信号，
   学生在训练上限之外的行为不受约束。我们这里是每轮 512 的上限，再加上格式漂移，后果更严重。
5. **次要因素：** 我们的 lr（1e-5，LoRA）比你大 10 倍，会让漂移更快；但同样的 lr 下我们的 OPD 没崩，所以它不是单独原因。
   每轮 512 的上限对教师和 OPD 来说偏紧，但 Semi 在第 10 步（238 token，远未到上限）格式就已经坏了。

以上第 2、3 点是推测，还没有直接验证。

## 6. 想请你确认的事

1. 你的教师 c3-step299 每轮大约写多少 token？它的输出是不是和学生一样的 `<action>` 格式？
2. 你的每轮回答上限和 horizon 是多少？
3. 你那边 Semi 训练中，学生评测时每轮长度有没有变化？
4. 如果方便，能否用你的设置换一个**格式不一致的通用教师**（比如 Qwen3-4B-Instruct-2507）跑一次 Semi，看会不会复现崩溃？
   反过来我们也可以换成格式一致的教师验证。这样能区分"教师格式不一致"与"lr / LoRA / 学生规模"几种解释。

## 7. 代码与原始数据

```
alfworld/alf_opd.py    多轮 OPD 训练与评测；--semi 即 Semi-OPD
alfworld/common.py     教师打分（vLLM 服务 prompt_logprobs）与 LoRA 学生
alfworld/alf_probe.py  策略成功率与师生 Overlap@64
slurm/*_alf_opd.sbatch 3 GPU 作业脚本（vulcan / rorqual / killarney）
slurm/submit_lh.sh     提交 OPD / Semi × H∈{10,20,40}
results/metrics/*.jsonl          每步训练指标与每 10 步评测（字段见 alf_opd.py 的 rec）
results/eval_summary.csv         评测汇总
results/eval_actions_sample.json 崩溃前后解析出的动作分布（Semi 与 OPD，H=10）
PLAN.md                实验方案
```

Semi 的实现只有一行改动（`alf_opd.py` 训练循环里 `run_eps(..., None if a.semi else lreq, ...)`）。运行示例：

```bash
sbatch --export=ALL,TEACH=4bi --job-name=alf-Semi4-H10 slurm/killarney_alf_opd.sbatch --estimator A --semi --H 10
sbatch --export=ALL,TEACH=4bi --job-name=alf-OPD4-H10  slurm/killarney_alf_opd.sbatch --estimator A --H 10
```

依赖 ALFWorld 环境封装与 prompt 模板（[SDAR](https://github.com/ZJU-REAL/SDAR) 的 `agent_system`）以及按动作序列重放游戏的
`alfreplay.Game`，二者未包含在本仓库。作业脚本里的账户与 `/scratch` 路径需按自己的集群修改。
`alf_opd.py` 里其他估计器（B1/B2/C/D1）来自另一项实验，本实验只用估计器 A。
目前只保存了解析后的动作，没有保存原始回答全文；如需看完整生成文本，可以用 `runs/alf-Semi4-H10/final_adapter` 重新采样。
