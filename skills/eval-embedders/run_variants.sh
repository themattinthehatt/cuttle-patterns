#!/usr/bin/env bash
# Embed a frame pool with named frozen embedders, then score each with the eval harness.
#
# usage: run_variants.sh [-c CLASSIFIER] [-l LOG_DIR] VARIANT [VARIANT ...]
#        run_variants.sh -L          # list variant names
#
# Run detached for long jobs, e.g. `nohup setsid run_variants.sh vgg19_1_448_gram_k64_taper &`.
# Resumable: an embed is skipped if its embeddings.npy exists. Failures are appended to
# LOG_DIR/failures.log and the next variant still runs. Progress is in LOG_DIR/progress.json.
# The conda env `cuttle` must be installed; this script activates it.
set -u

# name|cuttle embed args (add a line to add a variant)
VARIANTS=(
  "dinov3_vitb16_224_cls|--backbone vitb16 --resolution 224 --readout cls"
  "dinov3_vitb16_224_meanpatch_taper|--backbone vitb16 --resolution 224 --readout meanpatch_taper"
  "dinov3_vitb16_224_gram_k64_taper|--backbone vitb16 --resolution 224 --readout gram --gram-k 64 --gram-weights taper"
  "vgg19_1_448_gram_k64_taper|--backbone vgg19 --resolution 448 --readout gram --gram-k 64 --gram-weights taper --vgg-layer relu1_1"
  "vgg19_5_448_gram_k64_taper|--backbone vgg19 --resolution 448 --readout gram --gram-k 64 --gram-weights taper --vgg-layer relu5_1"
  "vgg19_12345_448_gram_k28_taper|--backbone vgg19 --resolution 448 --readout gram --gram-k 28 --gram-weights taper --vgg-layer relu1_1,relu2_1,relu3_1,relu4_1,relu5_1"
)

CLASSIFIER=""
LOG_DIR=""
while getopts "c:l:L" opt; do
  case $opt in
    c) CLASSIFIER=$OPTARG ;;
    l) LOG_DIR=$OPTARG ;;
    L) for v in "${VARIANTS[@]}"; do echo "${v%%|*}"; done; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { echo "no variants given; see -L" >&2; exit 2; }

eval "$(conda shell.bash hook)" && conda activate cuttle || exit 1
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
RESULTS=$(python -c "from cuttle_patterns.config import load_config; print(load_config().results_dir)") || exit 1
[ -n "$CLASSIFIER" ] || CLASSIFIER=$(ls -1 "$RESULTS/classifications" | grep -v '\.' | tail -1)
[ -n "$LOG_DIR" ] || LOG_DIR="$RESULTS/eval/logs"
mkdir -p "$LOG_DIR"

total=$#; done_n=0
for name in "$@"; do
  args=""
  for v in "${VARIANTS[@]}"; do [ "${v%%|*}" = "$name" ] && args=${v#*|}; done
  if [ -z "$args" ]; then echo "unknown variant: $name" >> "$LOG_DIR/failures.log"; done_n=$((done_n+1)); continue; fi
  echo "{\"done\": $done_n, \"total\": $total, \"current\": \"$name\", \"stage\": \"embed\"}" > "$LOG_DIR/progress.json"
  if [ ! -f "$RESULTS/beast_models/$name/image_predictions/beast_frames/latents/embeddings.npy" ]; then
    # shellcheck disable=SC2086
    cuttle embed $args --model-name "$name" > "$LOG_DIR/$name.embed.log" 2>&1 \
      || { echo "embed failed: $name" >> "$LOG_DIR/failures.log"; done_n=$((done_n+1)); continue; }
  fi
  echo "{\"done\": $done_n, \"total\": $total, \"current\": \"$name\", \"stage\": \"eval\"}" > "$LOG_DIR/progress.json"
  python -m cuttle_patterns.eval.run_core --classifier-name "$CLASSIFIER" --model-name "$name:ae" \
    > "$LOG_DIR/$name.eval.log" 2>&1 || echo "eval failed: $name" >> "$LOG_DIR/failures.log"
  done_n=$((done_n+1))
done
echo "{\"done\": $done_n, \"total\": $total, \"current\": null, \"stage\": \"finished\"}" > "$LOG_DIR/progress.json"
