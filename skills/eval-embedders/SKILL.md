---
name: eval-embedders
description: Embed a frame pool with frozen embedders (DINOv3, VGG-19 Gram) via `cuttle embed` and/or score embeddings (frozen or trained BEAST models) with the eval harness `cuttle_patterns.eval.run_core`, then show the combined scoreboard. Use when asked to run embedders, run the eval harness, or compare embedders.
---

# Embed and score embedders

Metric definitions and how to read them: [`cuttle_patterns/eval/README.md`](../../cuttle_patterns/eval/README.md).
Embedder design: `docs/implementation_notes/embedder.md`. This file is the runbook.

## 1. Check before running

- Env: activate `cuttle` (`conda activate cuttle`); DINOv3 needs a Hugging Face token.
- Pool: `results/beast_frames` holds the frames and `results/manifests/extract.parquet`
  lists them (both written by `cuttle extract-export`). The harness joins on that manifest.
- Classifier: a directory under `results/classifications/` of per-video parquets (from
  `cuttle extract-classify`); pass its name as `--classifier-name`. The old single
  `{name}.parquet` layout also works.
- Trained BEAST models only: `beast_models/{name}/image_predictions/beast_frames/latents`
  must cover the *current* pool (same video count and frames). Latents predicted on an older
  pool fail or score the wrong frames; rerun `cuttle predict --save-latents` first.

## 2. Frozen embedders

`run_variants.sh` holds the named variants (`-L` lists them): DINOv3 ViT-B/16 at 224
(`cls`, `meanpatch_taper`, `gram_k64_taper`) and VGG-19 Gram at 448 (relu1_1 k64,
relu5_1 k64, relu1-5 fused k28). "Layer N" means `reluN_1`. Add variants as new lines in
the `VARIANTS` array.

```bash
# one variant per invocation; detached, logged, resumable
nohup setsid skills/eval-embedders/run_variants.sh vgg19_1_448_gram_k64_taper &
```

It embeds (skipped if `embeddings.npy` exists) then scores each variant, writing logs and
`progress.json` under `results/eval/logs/` (`-l DIR` to change). Failures go to
`failures.log` and the next variant still runs. Read progress without attaching:
`cat results/eval/logs/progress.json`.

Cost on 132k frames: DINOv3 about 4 min each. The GPU is only used by the embed step; eval
runs on the CPU, so a BEAST training run can share the GPU (the VGG relu1_1 and fused runs
are the most memory-hungry).

## 3. Trained models (BEAST / MSPS-VAE)

No embed step; score existing latents. Run one model per command so rows arrive as they finish:

```bash
python -m cuttle_patterns.eval.run_core --classifier-name CLF --model-name resnet-18_d16:ae
python -m cuttle_patterns.eval.run_core --classifier-name CLF --model-name msps-vae-mask_d16:msps_vae
```

`msps_vae` expands to three rows (`_z_u`, `_z_b`, `_z_all`); everything else is `:ae`. Each
call merges into `results/eval/metrics.json`: rerunning a name replaces only its rows. A
model takes a few minutes; run long ones detached, as above.

## 4. Show the table

```bash
python skills/eval-embedders/print_scoreboard.py
```

Prints every row in `metrics.json` as one compact markdown table (also in
`results/eval/scoreboard.md`, with std columns).

## 5. Reading it

- Want high: AMI class (sanity floor, not a goal), effective videos per cluster, cross-video
  kNN, linear probe. Want low: AMI video|class.
- `z_b` of an MSPS-VAE should look identity-heavy (high AMI video, few videos per cluster);
  if not, suspect the harness.
- The reported std is k-means seed spread only and understates real noise; treat gaps near
  0.02 as ties. kNN and probe differences among frozen embedders are usually that small.
- `var ret` is 1.0 for 16-d latents; it is not a quality score.
- Trained models may have seen frames of the pool in training; frozen ones have not.
