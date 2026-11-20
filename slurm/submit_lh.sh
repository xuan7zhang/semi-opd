#!/bin/bash
# 新教师 Qwen3-4B-Instruct-2507：OPD 与 Semi × H∈{10,20,40}，3 小时一段的同名续跑链。
# 用法：bash submit_lh.sh <sbatch 文件>（vulcan 用 alf_opd.sbatch，rorqual 用 ralf_opd.sbatch）
cd /scratch/xzhan576/sdi
for H in 10 20 40; do for m in OPD Semi; do
  case $H in 10) n=6;; 20) n=10;; 40) n=16;; esac; [ $m = Semi ] && n=$((n*2/3))
  flag=""; [ $m = Semi ] && flag="--semi"
  ids=""; for k in $(seq 1 $n); do
    j=$(sbatch --parsable --export=ALL,TEACH=4bi --job-name=alf-${m}4-H$H --dependency=singleton --time=3:00:00 $1 --estimator A $flag --H $H); ids="$ids $j"
  done
  echo "alf-${m}4-H$H$ids"
done; done
