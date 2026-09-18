"""Load already-trained ("Tier A") embeddings and align them to the eval manifest.

Every embedder this v1 harness scores was already trained through the existing BEAST
pipeline (Phase 4) — nothing here re-runs inference. It reuses
`cuttle_patterns.embeddings.load_latents`/`split_latent_spaces` to read the per-frame
`.npy` latents `cuttle predict --save-latents` wrote, then restricts/reorders them to
match the eval manifest's frame set. A second, pretrained-backbone tier ("Tier B") is
deferred — see `docs/eval_plan.md`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cuttle_patterns.embeddings import load_latents, split_latent_spaces

DEFAULT_PREDICTIONS_NAME = 'beast_frames'


@dataclass
class EmbedderSpec:
    """One embedder to score: a trained model dir plus which latent subspace to read.

    Attributes:
        id: short name for the scoreboard, e.g. `msps_vae_z_u`.
        model_dir: `results_dir/beast_models/{model_name}`.
        subspace: one of `cuttle_patterns.embeddings`'s subspace keys (`all`,
            `unsupervised`, `background`); `all` for every non-`msps_vae` model.
        predictions_name: `cuttle predict`'s `--input-dir` stem; selects which
            `image_predictions/{predictions_name}` directory to read latents from.
    """

    id: str
    model_dir: Path
    subspace: str = 'all'
    predictions_name: str = DEFAULT_PREDICTIONS_NAME


def load_embedder_matrix(spec: EmbedderSpec, manifest: pd.DataFrame) -> np.ndarray:
    """Load one embedder's latents, restricted and row-ordered to match `manifest`.

    Args:
        spec: which model/subspace to load.
        manifest: eval manifest from `eval.build_manifest.build_eval_manifest`, with
            `video_name`/`frame_number` columns.

    Returns:
        float array of shape (len(manifest), latent_dim), row-aligned with manifest.

    Raises:
        ValueError: if any manifest frame has no matching latent vector.
    """
    latents_dir = spec.model_dir / 'image_predictions' / spec.predictions_name / 'latents'
    X, meta = load_latents(latents_dir)
    # 'all' always means the full raw latent vector, regardless of model class --
    # `split_latent_spaces` only exposes that key for non-msps_vae models (an
    # msps_vae config has no combined key, just 'unsupervised'/'background'
    # separately), so it's handled directly here instead.
    X_sub = X if spec.subspace == 'all' else split_latent_spaces(X, spec.model_dir)[spec.subspace]

    row_by_key = {
        (video_name, frame_number): row
        for row, (video_name, frame_number) in enumerate(
            zip(meta['video_name'], meta['frame_number'], strict=True)
        )
    }

    keys = list(
        zip(manifest['video_name'], manifest['frame_number'].astype(int), strict=True)
    )
    missing = [key for key in keys if key not in row_by_key]
    if missing:
        raise ValueError(
            f'{len(missing)} manifest frames have no matching latent vector under '
            f'{latents_dir}, e.g. {missing[0]}'
        )

    rows = np.fromiter((row_by_key[key] for key in keys), dtype=int, count=len(keys))
    return X_sub[rows]
