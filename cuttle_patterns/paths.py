"""Single source of truth for the on-disk layout of a `results_dir` tree.

Every pipeline stage under `cuttle_patterns/cli/` and QC script under `scripts/` builds its
paths by joining these constants onto `results_dir` at the call site, instead of hardcoding
relative paths, so the tree layout only has to change in one place:

    results_dir/
    ├── beast_frames/                      # cuttle extract -> BEAST training frames
    ├── beast_frames_qc/
    │   └── reconstructions/
    │       └── {model_name}/              # scripts/make_reconstruction_clip.py
    ├── beast_models/
    │   └── {model_name}/                  # cuttle train / cuttle predict
    │       ├── image_predictions/...      # beast predict's own layout, not covered here
    │       ├── reduce/                    # cuttle reduce
    │       │   └── umap_{hparams}.parquet
    │       └── clusters/                  # cuttle cluster
    │           └── {method}_{hparams}.parquet
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

# QC clips (scripts/make_mantle_clip.py, scripts/make_reconstruction_clip.py)
MEDIA_RELPATH = Path('media')
BEAST_FRAMES_QC_RECONSTRUCTIONS_RELPATH = Path('beast_frames_qc') / 'reconstructions'
