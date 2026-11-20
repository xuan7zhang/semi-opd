#!/bin/bash
# 长 horizon Semi-OPD 阶段 0：sbatch probe.sbatch <stage: student|teacher|overlap> [<tag>]
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/xzhan576/sdi/logs/%x-%j.out
set -eo pipefail
S=/scratch/xzhan576/sdar; D=/scratch/xzhan576/sdi
module load StdEnv/2023 gcc/12.3 python/3.12 arrow/21.0.0 opencv/4.12.0
tar -xf $S/venv_pinned.tar -C $SLURM_TMPDIR
export VIRTUAL_ENV=$SLURM_TMPDIR/venv_pinned PATH=$SLURM_TMPDIR/venv_pinned/bin:$PATH
unset ROCR_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=$SLURM_TMPDIR VLLM_CACHE_ROOT=$SLURM_TMPDIR/vllm
export ALFWORLD_DATA=$S/data/alfworld PYTHONPATH=$S/SDAR:$S/recovery:$D:$PYTHONPATH
snap() { ls -d $1/hub/models--Qwen--$2/snapshots/* | head -1; }
S17=$(snap $S/hf Qwen3-1.7B); T8=$(snap $S/hf Qwen3-8B); T4I=$(snap /scratch/xzhan576/semiopd/hf Qwen3-4B-Instruct-2507)
cd $D; mkdir -p runs/probe; P=runs/probe
case $1 in
  student)
    python alf_probe.py play --model $S17 --tag s17 --H 20 --save_turns $P/turns_s17.jsonl --n_turns 600
    python alf_probe.py play --model $S17 --tag s17 --H 40
    python alf_probe.py topk --model $S17 --turns $P/turns_s17.jsonl --out $P/top_s17.pkl ;;
  teacher)
    M=$T8; [ $2 = t4i ] && M=$T4I
    python alf_probe.py play --model $M --tag $2 --H 20
    python alf_probe.py play --model $M --tag $2 --H 40 ;;
  overlap)
    for t in t8 t4i; do M=$T8; [ $t = t4i ] && M=$T4I
      python alf_probe.py topk --model $M --turns $P/turns_s17.jsonl --out $P/top_$t.pkl
      python alf_probe.py overlap --student $P/top_s17.pkl --teacher $P/top_$t.pkl | tee $P/overlap_$t.json
    done ;;
esac
echo PROBE_OK
