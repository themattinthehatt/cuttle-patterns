#!/bin/bash
# Tier 2: existing iter-1.0 models (not retrained) predicting on every variant, then scored.
# usage: run_tier2.sh logdir
set -u
LOG=$1; mkdir -p $LOG
ORIG=/media/mattw/CUTTLE/results/.iter-1.0/beast_models
V=/media/mattw/CUTTLE/results/.iter-1.0-photometry/variants
VARS="orig raw final wrong norm"
MODELS="iter-1.1_resnet-18_d16 iter-1.1_msps-vae-mask_d16"
for m in $MODELS; do
  for v in $VARS; do
    d=$V/$v/beast_models/$m; mkdir -p $d
    ln -sfn $ORIG/$m/config.yaml $d/config.yaml
    rm -f "${d:?}/tb_logs"  # drop the earlier dir symlink; rglob does not follow dir symlinks
    mkdir -p "$d/tb_logs/version_0/checkpoints"
    ln -sfn "$(ls $ORIG/$m/tb_logs/version_0/checkpoints/*best.ckpt)" "$d/tb_logs/version_0/checkpoints/best.ckpt"
    cuttle predict --results-dir $V/$v --model-name $m --save-latents > $LOG/predict_${m}_$v.log 2>&1 || echo "PREDICT FAILED $m $v" >> $LOG/progress.txt
  done
done
echo "PREDICTS DONE" >> $LOG/progress.txt
python scratch/photometry/eval_variants.py --model-name iter-1.1_resnet-18_d16:ae --variants $VARS --tag tier2 > $LOG/eval_resnet.txt 2>&1
echo "DONE resnet $(date +%T)" >> $LOG/progress.txt
python scratch/photometry/eval_variants.py --model-name iter-1.1_msps-vae-mask_d16:msps_vae --exclude _z_all --variants $VARS --tag tier2 > $LOG/eval_mspsmask.txt 2>&1
echo "DONE msps-vae-mask $(date +%T)" >> $LOG/progress.txt
