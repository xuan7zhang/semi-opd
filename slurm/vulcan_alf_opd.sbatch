#!/bin/bash
# 阶段 1：ALFWorld 多轮 OPD。GPU0: vLLM student(Qwen3-1.7B)+LoRA；GPU1: teacher 服务(Qwen3-8B)；GPU2: LoRA 训练
# 用法：sbatch --job-name=alf-B1-H40 alf_opd.sbatch --estimator B1 --H 40 [其它参数]
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/xzhan576/sdi/logs/%x-%j.out
set -eo pipefail
S=/scratch/xzhan576/sdar; D=/scratch/xzhan576/sdi
module load StdEnv/2023 gcc/12.3 python/3.12 arrow/21.0.0 opencv/4.12.0
tar -xf $S/venv_pinned.tar -C $SLURM_TMPDIR
export VIRTUAL_ENV=$SLURM_TMPDIR/venv_pinned PATH=$SLURM_TMPDIR/venv_pinned/bin:$PATH
unset ROCR_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1 HF_HOME=$S/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=$SLURM_TMPDIR VLLM_CACHE_ROOT=$SLURM_TMPDIR/vllm
export ALFWORLD_DATA=$S/data/alfworld PYTHONPATH=$S/SDAR:$S/recovery:$D:$PYTHONPATH
snap() { ls -d $S/hf/hub/models--Qwen--$1/snapshots/* | head -1; }
STUDENT=$(snap Qwen3-1.7B)
# 教师可用环境变量 TEACH 指定（默认 Qwen3-8B）；Qwen3-4B-Instruct-2507 在 semiopd/hf 下
case ${TEACH:-8b} in 8b) TEACHER=$(snap Qwen3-8B);; 4bi) TEACHER=$(ls -d /scratch/xzhan576/semiopd/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/* | head -1);; esac
echo "teacher: $TEACHER"
PORT=$((8200 + SLURM_JOB_ID % 500))
CUDA_VISIBLE_DEVICES=1 vllm serve $TEACHER --served-model-name teacher --port $PORT --max-model-len 4096 \
    --gpu-memory-utilization 0.9 --enable-prefix-caching > $SLURM_TMPDIR/teacher.log 2>&1 &
for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models >/dev/null && break; sleep 5; done
cd $D
CUDA_VISIBLE_DEVICES=0,2 python alf_opd.py --student $STUDENT --teacher_url http://127.0.0.1:$PORT \
    --out $D/runs/${SLURM_JOB_NAME} "$@"
