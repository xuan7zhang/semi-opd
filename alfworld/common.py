"""OPD 共用组件：teacher 打分（vLLM 服务 prompt_logprobs）与 LoRA student（peft）。

约定：所有序列都以 token id 处理；student 与 teacher 共用同一 tokenizer（Qwen3 系列），
teacher 在与 student 完全相同的 token 序列（prompt + student response）上打分，
只取 response 位置的 log π_T —— prompt / 环境观测 token 不进入任何 loss 或 KL 求和。
"""
import json
import urllib.request

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


class Teacher:
    """vLLM OpenAI 兼容服务。logprobs(seqs, starts) → 每条序列 response 段的 log π_T（list[tensor]）。"""

    def __init__(self, url, model="teacher", batch=32):
        self.url, self.model, self.batch = url.rstrip("/"), model, batch

    def _post(self, body):
        req = urllib.request.Request(self.url + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=1800))

    def logprobs(self, seqs, starts):
        out = []
        for i in range(0, len(seqs), self.batch):
            chunk = seqs[i:i + self.batch]
            res = self._post({"model": self.model, "prompt": chunk, "max_tokens": 1, "temperature": 0.0,
                              "prompt_logprobs": 0})
            choices = sorted(res["choices"], key=lambda c: c["index"])
            for ids, st, c in zip(chunk, starts[i:i + self.batch], choices):
                pl = c["prompt_logprobs"]
                vals = []
                for j in range(st, len(ids)):
                    d = pl[j]
                    vals.append(d[str(ids[j])]["logprob"])
                out.append(torch.tensor(vals, dtype=torch.float32))
        return out

    def generate(self, prompts, max_tokens, stop=None, temperature=0.0):
        res = self._post({"model": self.model, "prompt": prompts, "max_tokens": max_tokens,
                          "temperature": temperature, "stop": stop or []})
        return [c["text"] for c in sorted(res["choices"], key=lambda c: c["index"])]


class Student:
    def __init__(self, base_path, device, lr, rank=64, alpha=128, micro=8):
        self.device, self.micro = device, micro
        m = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16).to(device)
        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()
        cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(m, cfg)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr,
                                     betas=(0.9, 0.999), weight_decay=0.0)
        self.pad = 0

    def _batch_logprobs(self, seqs, starts, grad):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        ids, att = ids.to(self.device), att.to(self.device)
        with torch.set_grad_enabled(grad):
            logits = self.model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(2, ids[:, 1:, None])[..., 0]
        return [lp[i, st - 1:len(s) - 1] for i, (s, st) in enumerate(zip(seqs, starts))]

    @torch.no_grad()
    def logprobs(self, seqs, starts):
        out = []
        for i in range(0, len(seqs), self.micro):
            out += [x.float().cpu() for x in self._batch_logprobs(seqs[i:i + self.micro], starts[i:i + self.micro], False)]
        return out

    def step(self, seqs, starts, advs, norm="token", denom=None):
        """loss = −Σ_t A_t · log π_θ(a_t|s_t)。norm="token": 除以 batch 内 response token 总数；
        "seq": 每条序列内求和后对序列取平均（不做长度归一化）。advs: list[tensor]（与 response 等长，已 detach）。
        denom: 显式指定分母（如 episode 数：多轮时一条 episode 有多个 turn 序列），优先于 norm。"""
        self.model.train()
        ntok = sum(len(a) for a in advs)
        nseq = len(advs)
        tot = 0.0
        for i in range(0, len(seqs), self.micro):
            lps = self._batch_logprobs(seqs[i:i + self.micro], starts[i:i + self.micro], True)
            loss = 0.0
            for lp, a in zip(lps, advs[i:i + self.micro]):
                term = -(a.to(lp.device) * lp).sum()
                d = denom if denom is not None else (ntok if norm == "token" else nseq)
                loss = loss + term / d
            loss.backward()
            tot += float(loss)
        self.last_gnorm = float(torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0))
        self.opt.step(); self.opt.zero_grad(set_to_none=True)
        return tot

    def save_adapter(self, path):
        self.model.save_pretrained(path)
