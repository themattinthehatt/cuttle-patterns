"""Single source of truth for the on-disk layout of a `results_dir` tree.

Every pipeline stage under `cuttle_patterns/cli/` and QC script under `scripts/` builds its
paths by joining these constants onto `results_dir` at the call site, instead of hardcoding
relative paths, so the tree layout only has to change in one place:

    results_dir/
    ├── beast_frames/                      # cuttle extract -> BEAST training frames
    ├── beast_frames_qc/
    │   ├── clusters/
    │   │   └── {model_name}_{cluster_run}/    # scratch/plot_cluster_frames.py
    │   ├── loss_mask/                     # scratch/plot_loss_weight_profiles.py
    │   └── reconstructions/
    │       └── {model_name}/              # scripts/make_reconstruction_clip.py
    ├── beast_models/
    │   └── {model_name}/                  # cuttle train / cuttle predict
    │       ├── image_predictions/...      # beast predict's own layout, not covered here
    │       ├── reduce/                    # cuttle reduce
    │       │   └── umap_{hparams}.parquet
    │       └── clusters/                  # cuttle cluster
    │           └── {method}_{hparams}.parquet
    ├── classifications/                    # scripts/classify_skin_pattern.py
    │   └── {classifier_name}.parquet       # model-independent, unlike clusters/ above
    ├── eval/                               # cuttle_patterns.eval.run_core
    │   ├── metrics.json
    │   └── scoreboard.md
    ├── manifests/
    │   ├── extract.parquet                # cuttle extract
    │   └── ingest.parquet                 # cuttle ingest
    ├── media/                             # scripts/make_mantle_clip.py
    ├── pose/                              # tail/neck pose predictions (external)
    └── rectangles/                        # cuttle inscribe / cuttle overlay
"""

from pathlib import Path

# manifests, written by `cuttle ingest`/`cuttle extract`
INGEST_MANIFEST_RELPATH = Path('manifests') / 'ingest.parquet'
EXTRACT_MANIFEST_RELPATH = Path('manifests') / 'extract.parquet'

# per-video pipeline stages
RECTANGLES_RELPATH = Path('rectangles')
POSE_RELPATH = Path('pose')

# BEAST training frames, models, and predictions
BEAST_FRAMES_RELPATH = Path('beast_frames')
BEAST_MODELS_RELPATH = Path('beast_models')

# dimensionality reduction output (cuttle reduce), relative to beast_models/{model_name}/
REDUCE_RELPATH = Path('reduce')

# clustering output (cuttle cluster), relative to beast_models/{model_name}/
CLUSTERS_RELPATH = Path('clusters')

# supervised classifier predictions (scripts/classify_skin_pattern.py) - independent of
# any BEAST model, since classification only depends on the frame images
CLASSIFICATIONS_RELPATH = Path('classifications')

# embedding eval harness scoreboard (cuttle_patterns.eval.run_core)
EVAL_RELPATH = Path('eval')

# QC clips (scripts/make_mantle_clip.py, scripts/make_reconstruction_clip.py)
MEDIA_RELPATH = Path('media')
BEAST_FRAMES_QC_RECONSTRUCTIONS_RELPATH = Path('beast_frames_qc') / 'reconstructions'

# per-cluster representative frame grids (scratch/plot_cluster_frames.py); one
# subdirectory per {model_name}_{cluster_run} combination
CLUSTER_FRAME_GRIDS_RELPATH = Path('beast_frames_qc') / 'clusters'

# MSPS-AE loss-weighting mask QC figures (scratch/plot_loss_weight_profiles.py)
LOSS_MASK_QC_RELPATH = Path('beast_frames_qc') / 'loss_mask'
