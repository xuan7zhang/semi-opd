"""阶段 1：ALFWorld 多轮 OPD，比较估计器 A / B1 / B2 / B2nb（同一批 rollout、同一份 teacher 打分，只 advantage 不同）。

环境：ALFWorld 文本（确定性），加 'done' 动作（始终可选；执行即以"未完成"结束本局），步数上限 H。
prompt：verl-agent 的 ALFWorld 模板（history_length=2）+ 一句中性说明 'done' 可随时结束。
student：Qwen3-1.7B + LoRA r64；teacher：Qwen3-8B（vLLM 服务，prompt_logprobs）。两者 enable_thinking=False、同一 chat template。
每个 turn 是一条 (prompt_ids, resp_ids) 序列；只有 student 生成的 resp token 计入 r_t 与 loss（观测/系统/用户 token 全部在 prompt 段，自然被排除）。

r_t = log π_T(a_t|s_t) − log π_θ(a_t|s_t)（逐 token，detach）
  A    : A_t = r_t；loss 分母 = batch 内 token 数（--norm token）或 episode 数（--norm seq，长度不归一化对照）
  B1   : R = Σ_episode r_t（不做长度归一化），同任务 G 条 rollout 内组归一化 (R−mean)/(std+1e-6)，广播到该 episode 每个 token；分母 = episode 数
  B2   : A_t = Σ_{t'≥t} r_{t'}（跨 turn 的 reward-to-go）− b(turn_index)，b = batch 内各 episode 在该 turn 起点的 reward-to-go 均值；分母 = episode 数
  B2nb : 同 B2，b = 0
评测（每 eval_every 步，固定 held-out 游戏）：成功率、提前退出率（未完成时选 done 且 V_T≥0.5，V_T = teacher 从该状态 K 次续跑的成功比例）、
  首次退出 turn、episode 长度、student token 数、重复动作率、每 turn 反向 KL；第 0 步另测 ε_init 与 p_T(stop)。
"""
import argparse, hashlib, json, os, random, statistics, sys, time
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, "/scratch/xzhan576/sdar/recovery")
from alfreplay import Game, game_files
from agent_system.environments.prompts import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS
from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from common import Teacher, Student

DONE = "done"
DONE_NOTE = "\nYou may choose 'done' at any time to end the episode."
PLACEBO = False   # --placebo：B1 组内打乱回报
USE_DONE = True   # --no_done 时为 False：动作空间与 prompt 里都没有 done
MAX_PROMPT, MAX_RESP, HIST = 2048, 512, 2


class Ep:
    def __init__(self, gf, H, tid, g):
        self.gf, self.H, self.tid, self.g = gf, H, tid, g
        self.game = Game(gf, expert=False)
        self.cur = self.game.reset()
        o = self.cur["obs"]; i = o.find("Your task is to: ")
        self.task = o[i + len("Your task is to: "):].strip()
        self.memory, self.turns, self.actions = [], [], []
        self.done = self.success = self.quit = False

    def admissible(self):
        return [a for a in self.cur["admissible"] if a != "help"] + ([DONE] if USE_DONE else [])

    def prompt_text(self):
        adm = "\n ".join(f"'{s}'" for s in self.admissible())
        if not self.memory:
            t = ALFWORLD_TEMPLATE_NO_HIS.format(current_observation=self.cur["obs"], admissible_actions=adm)
        else:
            rec = self.memory[-HIST:]; st = len(self.memory) - len(rec)
            hist = "\n".join(f"[Observation {st+j+1}: '{r[0]}', Action {st+j+1}: '{r[1]}']" for j, r in enumerate(rec))
            t = ALFWORLD_TEMPLATE.format(task_description=self.task, step_count=len(self.memory), history_length=len(rec),
                                         action_history=hist, current_step=len(self.memory) + 1,
                                         current_observation=self.cur["obs"], admissible_actions=adm)
        return t.rstrip() + (DONE_NOTE if USE_DONE else "") + "\n"

    def apply(self, text, p_ids, r_ids):
        acts, valids = alfworld_projection([text], [self.admissible()])
        a = acts[0]
        self.turns.append({"p": p_ids, "r": r_ids, "action": a, "valid": int(valids[0])})
        self.actions.append(a)
        if a == DONE and USE_DONE:
            self.quit = True; self.done = True
        else:
            prev = self.cur["obs"]
            self.cur = self.game.step(a)
            self.memory.append((prev, a))
            if self.cur["won"]:
                self.success = True; self.done = True
        if len(self.turns) >= self.H:
            self.done = True
        if self.done:
            self.game.close()


def chat_ids(tok, text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True,
                                  enable_thinking=False)
    return ids[-MAX_PROMPT:]


def run_eps(llm, tok, lreq, specs, H, temp, seed):
    """specs: [(gamefile, task_id, group_idx)]。返回结束的 Ep 列表。"""
    eps = [Ep(gf, H, tid, g) for gf, tid, g in specs]
    while True:
        act = [e for e in eps if not e.done]
        if not act:
            break
        prompts, sps = [], []
        for e in act:
            ids = chat_ids(tok, e.prompt_text())
            prompts.append({"prompt_token_ids": ids})
            sd = int(hashlib.md5(f"{seed}|{e.gf}|{e.g}|{len(e.turns)}".encode()).hexdigest()[:8], 16)
            sps.append(SamplingParams(temperature=temp, max_tokens=MAX_RESP, seed=sd))
        outs = llm.generate(prompts, sps, lora_request=lreq, use_tqdm=False)
        for e, o, p in zip(act, outs, prompts):
            e.apply(o.outputs[0].text, p["prompt_token_ids"], list(o.outputs[0].token_ids))
    return eps


def score(eps, teacher, student):
    """每个 turn 填 r（逐 token 反向 KL 奖励，detach）与 kl_turn（−Σr）。"""
    seqs, starts, refs = [], [], []
    for e in eps:
        for t in e.turns:
            if t["r"]:
                seqs.append(t["p"] + t["r"]); starts.append(len(t["p"])); refs.append(t)
    tl = teacher.logprobs(seqs, starts)
    sl = student.logprobs(seqs, starts)
    for t, a, b in zip(refs, tl, sl):
        t["r_tok"] = a - b
    for e in eps:
        for t in e.turns:
            if "r_tok" not in t:
                t["r_tok"] = torch.zeros(0)
            t["kl_turn"] = -float(t["r_tok"].sum())
    return seqs, starts, refs


def advantages(est, eps, G):
    """返回与 eps 中 turn 顺序对应的 advantage 列表（只含非空 response 的 turn）与 loss 分母。"""
    advs = {}
    if est in ("A", "A_seq"):
        for e in eps:
            for t in e.turns:
                advs[id(t)] = t["r_tok"].clone()
    elif est in ("B1", "B1n"):
        if est == "B1":
            R = {id(e): float(sum(t["r_tok"].sum() for t in e.turns)) for e in eps}
        else:   # B1'：整局 mean r（按 token 数归一化），其余同 B1
            R = {id(e): float(sum(t["r_tok"].sum() for t in e.turns)) / max(1, sum(len(t["r_tok"]) for t in e.turns)) for e in eps}
        if PLACEBO:   # 对照：组内打乱回报，advantage 分布与尺度不变，但与本局轨迹无关
            byg = defaultdict(list)
            for e in eps:
                byg[e.tid].append(e)
            for es in byg.values():
                vals = [R[id(e)] for e in es]; random.Random(len(R) + int(sum(vals))).shuffle(vals)
                for e, v in zip(es, vals):
                    R[id(e)] = v
        grp = defaultdict(list)
        for e in eps:
            grp[e.tid].append(R[id(e)])
        for e in eps:
            v = grp[e.tid]; m = statistics.mean(v); s = statistics.pstdev(v)
            a = (R[id(e)] - m) / (s + 1e-6)
            for t in e.turns:
                advs[id(t)] = torch.full_like(t["r_tok"], a)
    elif est in ("B2", "B2nb", "C"):
        rtg_turn_start = defaultdict(list)
        per_ep = []
        for e in eps:
            flat = torch.cat([t["r_tok"] for t in e.turns]) if e.turns else torch.zeros(0)
            rtg = torch.flip(torch.cumsum(torch.flip(flat, [0]), 0), [0])
            per_ep.append((e, rtg))
            pos = 0
            for k, t in enumerate(e.turns):
                if len(t["r_tok"]):
                    rtg_turn_start[k].append(float(rtg[pos]))
                pos += len(t["r_tok"])
        b = {k: (statistics.mean(v) if est in ("B2", "C") else 0.0) for k, v in rtg_turn_start.items()}
        for e, rtg in per_ep:
            pos = 0
            for k, t in enumerate(e.turns):
                n = len(t["r_tok"])
                advs[id(t)] = rtg[pos:pos + n] - b.get(k, 0.0)
                pos += n
    elif est == "D1":
        # turn 粒度 TD：A_t = Σ_{同 turn 内 t'≥t} r_{t'} + V̂(k+1) − V̂(k)
        # V̂(k) = 本批走到第 k 个 turn 的 episode 从该 turn 起的剩余 Σr 均值（按 turn index 的表格，= B2 的基线）。
        # 终止状态：选 done → 取同 turn index 的表格值 V̂(k+1)（"本来还要挨这么多 KL"，终止不是零价值）；
        #           预算用完 / 任务成功 → 0（真实终止）。
        rtg_start = defaultdict(list)
        for e in eps:
            tail = 0.0
            sums = [float(t["r_tok"].sum()) for t in e.turns]
            for k in range(len(sums) - 1, -1, -1):
                tail += sums[k]
                rtg_start[k].append(tail)
        Vtab = {k: statistics.mean(v) for k, v in rtg_start.items()}
        def V(k):
            if k in Vtab:
                return Vtab[k]
            ks = [x for x in Vtab if x < k]
            return Vtab[max(ks)] if ks else 0.0   # 没有轨迹到达该 turn：用最近的已知值
        for e in eps:
            T = len(e.turns)
            for k, t in enumerate(e.turns):
                if k + 1 < T:
                    v_next = V(k + 1)
                elif e.quit:
                    v_next = V(k + 1)        # 反事实剩余代价
                else:
                    v_next = 0.0             # 预算用完或成功：真实终止
                rtg_in = torch.flip(torch.cumsum(torch.flip(t["r_tok"], [0]), 0), [0])
                advs[id(t)] = rtg_in + (v_next - V(k))
    return advs


def v_teacher_batch(teacher, tok, quit_eps, K, temp):
    """V_T：对每个退出的 episode，teacher 从其退出前状态续跑 K 次到结束，返回成功比例列表。
    重放 student 的非 done 动作以恢复相同环境状态与历史；teacher 看与 student 相同的 prompt（含 done 选项）。
    所有续跑按 turn 批量请求 teacher。"""
    conts = []
    for qi, ep in enumerate(quit_eps):
        for k in range(K):
            e = Ep(ep.gf, ep.H, ep.tid, 1000 + k)
            for a in [x for x in ep.actions if x != DONE]:
                prev = e.cur["obs"]; e.cur = e.game.step(a); e.memory.append((prev, a)); e.turns.append(None)
            conts.append((qi, e))
    while True:
        act = [e for _, e in conts if not e.done]
        if not act:
            break
        txts = [tok.apply_chat_template([{"role": "user", "content": e.prompt_text()}], add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False) for e in act]
        outs = []
        for i in range(0, len(txts), 64):
            outs += teacher.generate(txts[i:i + 64], MAX_RESP, temperature=temp)
        for e, o in zip(act, outs):
            e.apply(o, [], [])
    wins = defaultdict(int)
    for qi, e in conts:
        wins[qi] += e.success
    return [wins[i] / K for i in range(len(quit_eps))]


CONT_MAX_TOK = 256   # teacher 续跑时每轮最多生成的 token（控制 V_T 成本）


def v_states(teacher, tok, states, K, temp, cache):
    """V_T(s)：teacher 从状态 s 续跑 K 次到结束的成功比例。state = (gamefile, task_id, H, tuple(非 done 动作))。
    结果按 (task_id, 动作历史) 缓存；所有续跑按 turn 批量请求 teacher。"""
    todo = [st for st in dict.fromkeys(states) if (st[1], st[3]) not in cache]
    conts = []
    for si, (gf, tid_, H, acts) in enumerate(todo):
        for k in range(K):
            e = Ep(gf, H, tid_, 1000 + k)
            for a in acts:
                prev = e.cur["obs"]; e.cur = e.game.step(a); e.memory.append((prev, a)); e.turns.append(None)
                if e.cur["won"]:
                    e.success = True; e.done = True; break
            if len(e.turns) >= H and not e.done:
                e.done = True
            if e.done:
                e.game.close()
            conts.append((si, e))
    while True:
        act = [e for _, e in conts if not e.done]
        if not act:
            break
        txts = [tok.apply_chat_template([{"role": "user", "content": e.prompt_text()}], add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False) for e in act]
        outs = []
        for i in range(0, len(txts), 64):
            outs += teacher.generate(txts[i:i + 64], CONT_MAX_TOK, temperature=temp)
        for e, o in zip(act, outs):
            e.apply(o, [], [])
    wins = defaultdict(int)
    for si, e in conts:
        wins[si] += e.success
    for si, st in enumerate(todo):
        cache[(st[1], st[3])] = wins[si] / K
    return [cache[(st[1], st[3])] for st in states]


def shape(eps, teacher, tok, K, lam_eff, cache, every=1):
    """C：在每个 turn 的最后一个 token 上加 λ_eff·[V_T(s_{k+1}) − V_T(s_k)]（potential-based，V 只依赖状态）。
    终止状态：成功 V=1，done / 预算用完 V=0。返回本批平均 |shaping| 与 V(s_0) 均值，供日志。"""
    states, idx = [], []
    for e in eps:
        acts = [a for a in e.actions if a != DONE]
        for k in range(0, len(e.turns), every):   # 每 every 个 turn 算一次 V_T，其余线性插值
            pre = tuple(acts[:k])
            states.append((e.gf, e.tid, e.H, pre)); idx.append((e, k))
    vals = v_states(teacher, tok, states, K, 0.7, cache)
    V = defaultdict(dict)
    for (e, k), v in zip(idx, vals):
        V[id(e)][k] = v
    tot, n, v0 = 0.0, 0, []
    for e in eps:
        T = len(e.turns)
        V[id(e)][T] = 1.0 if e.success else 0.0
        known = sorted(V[id(e)])
        for k in range(T + 1):
            if k not in V[id(e)]:
                lo = max(x for x in known if x < k); hi = min(x for x in known if x > k)
                V[id(e)][k] = V[id(e)][lo] + (V[id(e)][hi] - V[id(e)][lo]) * (k - lo) / (hi - lo)
        v0.append(V[id(e)].get(0, 0.0))
        for k, t in enumerate(e.turns):
            if len(t["r_tok"]):
                d = lam_eff * (V[id(e)][k + 1] - V[id(e)][k])
                t["r_tok"] = t["r_tok"].clone(); t["r_tok"][-1] += d
                tot += abs(d); n += 1
    return tot / max(1, n), sum(v0) / max(1, len(v0))


def ep_record(e, est, H, seed, step, split):
    acts = e.actions
    rep = sum(1 for i in range(1, len(acts)) if acts[i] == acts[i - 1]) / max(1, len(acts) - 1)
    return {"split": split, "step": step, "task_id": e.tid, "g": e.g, "seed": seed, "H": H, "estimator": est,
            "actions": acts, "success": e.success, "quit": e.quit, "quit_turn": len(e.turns) if e.quit else None,
            "turns": len(e.turns), "student_tokens": sum(len(t["r"]) for t in e.turns), "repeat_rate": rep,
            "kl_per_turn": [round(t.get("kl_turn", 0.0), 4) for t in e.turns],
            "r_tok": [[round(x, 4) for x in t["r_tok"].tolist()] for t in e.turns] if split == "train" and step % 10 == 0 else None,
            "termination": "success" if e.success else ("done" if e.quit else "budget")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", required=True, choices=["A", "A_seq", "B1", "B1n", "B2", "B2nb", "C", "D1"])
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher_url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--eval_games", type=int, default=64)
    ap.add_argument("--eval_n", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--semi", action="store_true",
                    help="Semi-OPD：训练 rollout 一律来自初始学生 π0（不挂 LoRA，种子与 OPD 相同），A_t 仍用当前 θ 计算")
    ap.add_argument("--placebo", action="store_true", help="B1/B1n：组内打乱 episode 回报（尺度对照）")
    ap.add_argument("--no_done", action="store_true", help="去掉 done 动作（动作空间与 prompt 都不含）")
    ap.add_argument("--v_every", type=int, default=1, help="C：每多少个 turn 算一次 V_T（其余插值）")
    ap.add_argument("--vK", type=int, default=4, help="C：每个状态 teacher 续跑次数")
    ap.add_argument("--lam", type=float, default=0.0, help="C 的 shaping 系数，单位 = ε_init（λ_eff = lam × ε_init）")
    a = ap.parse_args()
    global USE_DONE, PLACEBO
    USE_DONE = not a.no_done
    PLACEBO = a.placebo
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    train = game_files("train")
    evalg = sorted(game_files("valid_seen"))[:a.eval_games]
    tok = AutoTokenizer.from_pretrained(a.student)
    llm = LLM(model=a.student, enable_lora=True, max_lora_rank=64, max_loras=1, gpu_memory_utilization=0.85,
              max_model_len=MAX_PROMPT + MAX_RESP + 64, enable_prefix_caching=True)
    student = Student(a.student, "cuda:1", a.lr, micro=4)
    teacher = Teacher(a.teacher_url)
    flog = open(f"{a.out}/metrics.jsonl", "a")
    fep = open(f"{a.out}/episodes.jsonl", "a")
    lora_dir = os.path.join(os.environ.get("SLURM_TMPDIR", "/tmp"), "lora")
    tid = lambda gf: gf.split("json_2.1.1/")[1].rsplit("/", 1)[0]

    def evaluate(lreq, step):
        specs = [(g, tid(g), k) for g in evalg for k in range(a.eval_n)]
        eps = run_eps(llm, tok, lreq, specs, a.H, 0.4, seed=10_000 + step)
        score(eps, teacher, student)
        quits = [e for e in eps if e.quit]
        for e, v in zip(quits, v_teacher_batch(teacher, tok, quits, a.K, 0.7)):
            e.v_T = v
        prem = sum(e.v_T >= 0.5 for e in quits)
        kl_by_turn = defaultdict(list)
        for e in eps:
            for k, t in enumerate(e.turns):
                kl_by_turn[k].append(t["kl_turn"])
            r = ep_record(e, a.estimator, a.H, a.seed, step, "eval"); r["V_T_at_quit"] = getattr(e, "v_T", None)
            fep.write(json.dumps(r) + "\n")
        n = len(eps)
        return {"eval_success": sum(e.success for e in eps) / n, "eval_quit": len(quits) / n,
                "eval_premature": prem / n, "eval_len": sum(len(e.turns) for e in eps) / n,
                "eval_tokens": sum(sum(len(t["r"]) for t in e.turns) for e in eps) / n,
                "eval_repeat": sum(ep_record(e, "", 0, 0, 0, "x")["repeat_rate"] for e in eps) / n,
                "eval_first_quit_turns": sorted(len(e.turns) for e in quits),
                "eval_kl_per_turn": [round(statistics.mean(kl_by_turn[k]), 3) for k in sorted(kl_by_turn)]}

    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    ckpt = f"{a.out}/resume.pt"
    start_step, lreq = 1, None
    if os.path.exists(ckpt):   # 断点续跑：LoRA 权重、优化器、步数、随机数状态
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(student.model, st["lora"])
        student.opt.load_state_dict(st["opt"])
        rng.setstate(st["rng"]); start_step = st["step"] + 1
        lpath = f"{lora_dir}/step{st['step']}"; student.save_adapter(lpath)
        lreq = LoRARequest(f"s{st['step']}", st["step"], lpath)
        print(f"resumed from step {st['step']}", flush=True)

    def save_resume(step):
        torch.save({"lora": get_peft_model_state_dict(student.model), "opt": student.opt.state_dict(),
                    "rng": rng.getstate(), "step": step}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    if start_step == 1:
        # 第 0 步诊断：ε_init、p_T(stop)
        m0 = evaluate(None, 0)
        eps_init = statistics.mean(x for x in m0["eval_kl_per_turn"])
        first = [chat_ids(tok, Ep(g, a.H, tid(g), 0).prompt_text()) for g in evalg]
        txts = [tok.decode(ids) for ids in first]
        stops = 0
        for rep in range(4):
            outs = teacher.generate(txts, MAX_RESP, temperature=1.0)
            acts, _ = alfworld_projection(outs, [[] for _ in outs])
            stops += sum(x == DONE for x in acts)
        p_stop = stops / (4 * len(txts))
        m0.update({"step": 0, "eps_init": eps_init, "p_T_stop": p_stop})
        json.dump({"eps_init": eps_init, "p_T_stop": p_stop}, open(f"{a.out}/step0.json", "w"))
        print(json.dumps({k: v for k, v in m0.items() if k != "eval_first_quit_turns"}), flush=True)
        flog.write(json.dumps(m0) + "\n"); flog.flush()

    eps_init = json.load(open(f"{a.out}/step0.json"))["eps_init"]
    lam_eff = a.lam * eps_init
    vcache = {}
    t0 = time.time()
    for step in range(start_step, a.steps + 1):
        tasks = rng.sample(train, a.tasks)
        specs = [(g, tid(g), k) for g in tasks for k in range(a.G)]
        # Semi：π0 固定，按 OPD 同一种子逐步采样，与训练前一次性采好全部 D0 等价
        eps = run_eps(llm, tok, None if a.semi else lreq, specs, a.H, 1.0, seed=a.seed * 1_000_000 + step)
        seqs, starts, refs = score(eps, teacher, student)
        shp = {}
        if a.estimator == "C":
            raw_kl = -float(sum(t["r_tok"].sum() for t in refs) / max(1, sum(len(t["r_tok"]) for t in refs)))
            mabs, v0 = shape(eps, teacher, tok, a.vK, lam_eff, vcache, every=a.v_every)
            shp = {"shaping_abs_mean": mabs, "V_s0_mean": v0, "lam_eff": lam_eff, "raw_rev_kl_tok": raw_kl,
                   "vcache_size": len(vcache)}
        adv = advantages(a.estimator, eps, a.G)
        advs = [adv[id(t)] for t in refs]
        ntok = sum(len(x) for x in advs)
        denom = ntok if a.estimator == "A" else len(eps)
        loss = student.step(seqs, starts, advs, denom=denom)
        lpath = f"{lora_dir}/step{step}"; student.save_adapter(lpath)
        lreq = LoRARequest(f"s{step}", step, lpath)
        n = len(eps)
        rec = {"step": step, "sec": round(time.time() - t0), "loss": loss,
               "rev_kl_tok": -float(sum(t["r_tok"].sum() for t in refs) / max(1, ntok)),
               "train_success": sum(e.success for e in eps) / n, "train_quit": sum(e.quit for e in eps) / n,
               "train_len": sum(len(e.turns) for e in eps) / n, "train_tokens": ntok / n,
               "obs_token_r_sum": 0.0, "grad_norm": student.last_gnorm, **shp}   # 结构上恒为 0：观测 token 不在任何 r 序列中
        for e in eps:
            fep.write(json.dumps(ep_record(e, a.estimator, a.H, a.seed, step, "train")) + "\n")
        if step % a.eval_every == 0 or step == a.steps:
            rec.update(evaluate(lreq, step))
        print(json.dumps({k: v for k, v in rec.items() if k != "eval_first_quit_turns"}), flush=True)
        flog.write(json.dumps(rec) + "\n"); flog.flush(); fep.flush()
        if step % 5 == 0 or step == a.steps:
            save_resume(step)
    student.save_adapter(f"{a.out}/final_adapter")


if __name__ == "__main__":
    main()
    os._exit(0)
