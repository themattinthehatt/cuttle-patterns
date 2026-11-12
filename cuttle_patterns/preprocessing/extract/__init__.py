"""Frame selection for BEAST training, as a pipeline of `cuttle extract-*` stages.

Run `cuttle extract` for the list of stages. In order:

1. `eligibility` (`cuttle extract-eligibility`): which frames of each aligned video are usable.
2. `classify` (`cuttle extract-classify`): skin-pattern class of every frame (crop videos).
3. `dino_cls` (`cuttle extract-dino-cls`): DINOv3 CLS features of every frame (rectangle videos).
4. `kmeans` (`cuttle extract-kmeans-video`, then `cuttle extract-kmeans-pooled`): class-quota
   selection with per-video-first hierarchical k-means.
5. `report` (`cuttle extract-report`): diagnostics for a selection.
6. `export` (`cuttle extract-export`): write the selected frames as PNGs for BEAST.

`common` holds helpers shared by the stages. `README.md` in this directory is the working guide
(running order, gotchas, what to check, what is not validated).
"""

from cuttle_patterns.preprocessing.extract.eligibility import (
    DEFAULT_SMALL_RECT_RATIO_THRESH,
    compute_small_rectangle_mask,
)

__all__ = ['DEFAULT_SMALL_RECT_RATIO_THRESH', 'compute_small_rectangle_mask']
