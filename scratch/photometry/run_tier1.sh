#!/bin/bash
# Tier 1: frozen embedders on every variant, then scored per embedder.
# usage: run_tier1.sh logdir
set -u
LOG=$1; mkdir -p $LOG
V=/media/mattw/CUTTLE/results/.iter-1.0-photometry/variants
VARS="orig raw final wrong norm"
# name | cuttle embed args
EMBS=(
"dinov3_vitb16_224_cls|--backbone vitb16 --resolution 224 --readout cls"
"dinov3_vitb16_224_meanpatch_taper|--backbone vitb16 --resolution 224 --readout meanpatch_taper"
"dinov3_vitb16_224_gram_k64_taper|--backbone vitb16 --resolution 224 --readout gram --gram-k 64 --gram-weights taper"
"dinov3_vitb16_448_gram_k64_taper|--backbone vitb16 --resolution 448 --readout gram --gram-k 64 --gram-weights taper"
"vgg19_5_448_gram_k64_taper|--backbone vgg19 --resolution 448 --readout gram --gram-k 64 --gram-weights taper --vgg-layer relu5_1"
"vgg19_1_448_gram_k64_taper|--backbone vgg19 --resolution 448 --readout gram --gram-k 64 --gram-weights taper --vgg-layer relu1_1"
"vgg19_12345_448_gram_k28_taper|--backbone vgg19 --resolution 448 --readout gram --gram-k 28 --gram-weights taper --vgg-layer relu1_1,relu2_1,relu3_1,relu4_1,relu5_1"
)
for entry in "${EMBS[@]}"; do
  name=${entry%%|*}; eargs=${entry#*|}
  for v in $VARS; do
    cuttle embed --results-dir $V/$v $eargs --model-name $name > $LOG/embed_${name}_$v.log 2>&1 || echo "EMBED FAILED $name $v" >> $LOG/progress.txt
  done
  python scratch/photometry/eval_variants.py --model-name $name:ae --variants $VARS > $LOG/eval_$name.txt 2>&1
  echo "DONE $name $(date +%T)" >> $LOG/progress.txt
done
echo "ALL DONE" >> $LOG/progress.txt
