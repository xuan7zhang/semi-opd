"""长 horizon Semi-OPD，阶段 0：策略在 ALFWorld 上的成功率 + 师生 Overlap@64（复用 alf_opd 的环境与采样）。

  python alf_probe.py play    --model M --tag T --H 20 [--save_turns F --n_turns 600]
        64 个 valid_seen 游戏 × 2 次，温度 0.4（与 sdi 评测相同），含 done 动作 → runs/probe/T-H{H}.json
        --save_turns：另存前 n_turns 个回合的 (prompt_ids, resp_ids)，供 overlap 使用（用 π0 的回合）
  python alf_probe.py topk    --model M --turns F --out P.pkl     每个回答 token 位置的 top-64 token 集合
  python alf_probe.py overlap --student P0.pkl --teacher P1.pkl   token 加权 Overlap@64（同论文）
"""
import argparse, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _wait_free(frac, timeout=600):
    """上一个进程的 vLLM 引擎子进程可能还没退干净：等本卡空闲显存够用再启动。"""
    import time, torch
    t0 = time.time()
    while time.time() - t0 < timeout:
        free, tot = torch.cuda.mem_get_info()
        if free / tot >= frac + 0.02:
            return
        time.sleep(5)


def _finish(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        print("shutdown:", e, flush=True)
    os._exit(0)


def cmd_play(a):
    if os.path.exists(f"runs/probe/{a.tag}-H{a.H}.json") and (not a.save_turns or os.path.exists(a.save_turns)):
        print("skip", a.tag, a.H); os._exit(0)
    _wait_free(0.85)
    from transformers import AutoTokenizer
    from vllm import LLM
    import alf_opd as A
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, gpu_memory_utilization=0.85, max_model_len=A.MAX_PROMPT + A.MAX_RESP + 64,
              enable_prefix_caching=True, tensor_parallel_size=a.tp, seed=0)
    evalg = sorted(A.game_files("valid_seen"))[:64]
    tid = lambda gf: gf.split("json_2.1.1/")[1].rsplit("/", 1)[0]
    specs = [(g, tid(g), k) for g in evalg for k in range(2)]
    eps = A.run_eps(llm, tok, None, specs, a.H, 0.4, seed=10_000)
    n = len(eps)
    res = {"model": a.model, "tag": a.tag, "H": a.H, "n": n,
           "success": sum(e.success for e in eps) / n, "quit": sum(e.quit for e in eps) / n,
           "len": sum(len(e.turns) for e in eps) / n,
           "tok_per_turn": sum(len(t["r"]) for e in eps for t in e.turns) / max(1, sum(len(e.turns) for e in eps)),
           "invalid": sum(1 - t["valid"] for e in eps for t in e.turns) / max(1, sum(len(e.turns) for e in eps))}
    os.makedirs("runs/probe", exist_ok=True)
    json.dump(res, open(f"runs/probe/{a.tag}-H{a.H}.json", "w"))
    print(json.dumps(res), flush=True)
    if a.save_turns:
        turns = [{"p": t["p"], "r": t["r"], "k": k} for e in eps for k, t in enumerate(e.turns) if t["r"]]
        with open(a.save_turns, "w") as f:
            for t in turns[:a.n_turns]:
                f.write(json.dumps(t) + "\n")
    _finish(llm)


def cmd_topk(a):
    from vllm import LLM, SamplingParams
    if os.path.exists(a.out):
        print("skip", a.out); os._exit(0)
    _wait_free(0.6)
    rows = [json.loads(l) for l in open(a.turns)]
    llm = LLM(model=a.model, gpu_memory_utilization=0.6, max_model_len=4096, max_logprobs=64,
              max_num_batched_tokens=4096, tensor_parallel_size=a.tp, seed=0)
    sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=64)
    tops = []
    for i in range(0, len(rows), 16):
        ch = rows[i:i + 16]
        outs = llm.generate([{"prompt_token_ids": r["p"] + r["r"]} for r in ch], sp, use_tqdm=False)
        for r, o in zip(ch, outs):
            st = len(r["p"]); pl = o.prompt_logprobs
            tops.append([frozenset(t for t, v in pl[j].items() if v.rank is not None and v.rank <= 64)
                         for j in range(st, st + len(r["r"]))])
    pickle.dump(tops, open(a.out, "wb"))
    _finish(llm)


def cmd_overlap(a):
    S, T = pickle.load(open(a.student, "rb")), pickle.load(open(a.teacher, "rb"))
    num = den = 0
    for s, t in zip(S, T):
        for x, y in zip(s, t):
            num += len(x & y) / 64; den += 1
    print(json.dumps({"student": a.student, "teacher": a.teacher, "overlap": num / den, "positions": den}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play"); p.add_argument("--model"); p.add_argument("--tag"); p.add_argument("--H", type=int)
    p.add_argument("--tp", type=int, default=1); p.add_argument("--save_turns"); p.add_argument("--n_turns", type=int, default=600)
    p = sub.add_parser("topk"); p.add_argument("--model"); p.add_argument("--turns"); p.add_argument("--out")
    p.add_argument("--tp", type=int, default=1)
    p = sub.add_parser("overlap"); p.add_argument("--student"); p.add_argument("--teacher")
    a = ap.parse_args()
    {"play": cmd_play, "topk": cmd_topk, "overlap": cmd_overlap}[a.cmd](a)
