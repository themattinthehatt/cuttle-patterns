# Photometric normalization: findings and next steps

**Doc type:** exploratory status note. It records what was measured on the Day 1 full-frame
samples, what is still untested, and what to do next. Nothing here is wired into the
pipeline yet. When a design is chosen, the decision belongs in `docs/DECISIONS.md` and the
mechanism in `docs/PHASES.md`.

Motivation: clustering (and every embedder tried so far) organizes partly by *session /
tank* instead of skin pattern. `scratch/data-analysis/NEXT_STEPS.md` §2 measured the
effect: within a pattern class, video η² is 0.55 for a\*, 0.51 for black level, 0.49 for
saturation, and 0.45 for b\*. This note tests whether a per-tank photometric correction,
applied as far upstream as possible (to the full frame, before cropping), can remove it.

## Data

- **Full-frame samples:** `/media/mattw/CUTTLE/raw_data_samples/Day{n}_Tank{m}/Downsampled_frames/`,
  100 evenly spaced frames per tank, **Day 1 only**. `manifest.csv` maps each file to its
  video frame index (`00001.jpg` is frame 0).
- **Frame sizes differ by tank:** tanks 1 and 3 are 771×434; tanks 2, 4, 5, 6 are
  1156×651.
- **Crops:** Day 1 tanks 2-6 from `{results}/.iter-1.0/beast_frames` (200×100 PNGs), with
  pattern labels from `classifications/iter-1.1_classifier_d512.parquet`. Tank 1 is not in
  that batch. The Day 1 labels are almost all Leopard and Dark aggression.
- **Reference boxes:** `boxes.json`, drawn with `annotate_boxes.py`: tank interior, white
  water spout, black drain, per tank. Pixel coordinates in each frame's own resolution.

## What the full frames show

The six tanks fall into two lighting groups:

| group | tanks | floor | black level (linear) |
|---|---|---|---|
| gray | 1, 3, 5 | neutral, darker | crushed to about 0 |
| blue | 2, 4, 6 | bluish, brighter | lifted to about 0.02 |

Three separate effects, each acting differently:

1. **Offset ("fog").** Additive light that lifts the blacks in the blue tanks, about 2% of
   floor brightness. A gain can't remove it. In the gray tanks the blacks are clipped, so
   the correction is one-way: it can bring the blue tanks down, not recover lost detail.
2. **Gain (tint and brightness).** Multiplicative. In the blue tanks the floor tint also
   *drifts* over a recording (B/R time-sd 0.05-0.09, versus about 0.01 in the gray tanks),
   most strongly in the first thousands of frames. It is uncorrelated with the offset
   (|r| <= 0.3).
3. **Spatial unevenness.** A bright, bluish glow in the middle and dim corners; the same
   pigment looks different depending on where the animal swims.

The recordings **start with a divider in the tank** (confirmed); tanks 4 and 6 show a
different floor tint during that phase. Those frames are not necessarily excluded from
clustering, so the correction has to cope with them.

## Results so far

All numbers are Day 1, linear light, from the scripts in `prototypes/`.

### Offset

| tank | drain-box D (mean RGB) | darkest-background D |
|---|---|---|
| 1, 3, 5 | about 0.0005 | 0.0003-0.004 |
| 2 | 0.022 | 0.014-0.016 |
| 4 | 0.023 | 0.018-0.019 |
| 6 | 0.020 | 0.019-0.022 |

- D is nearly constant within a session; tank 6 steps down by about 0.005 near video frame
  25,000.
- The drain reads 0.005-0.010 higher than the darkest-background estimate in the blue
  tanks (mostly in the blue channel). A glossy surface reflecting extra light is a
  plausible reason; untested.
- The assumption behind both is that the darkest real black should read the same in every
  tank. Subtracting D aligns the tanks with each other; it is not a measurement of true
  black.

### Crops (session-constant corrections), tank η² within pattern class

Lower is better. 1,423 crops, tanks 2-6.

| stat | raw | gain only | offset (darkest bg) + gain | offset (drain) + gain |
|---|---|---|---|---|
| black level | 0.60 | 0.46 | **0.12** | 0.21 |
| b\* | 0.36 | 0.11 | **0.10** | 0.11 |
| saturation | 0.26 | 0.10 | 0.12 | 0.13 |
| L\* | 0.14 | **0.07** | 0.14 | 0.20 |
| a\* | **0.44** | 0.60 | 0.49 | 0.55 |

- The offset is the largest single win (black level 0.60 to 0.12). The darkest-background
  D never used the fish, so that improvement is a real test, not circular.
- **a\* is not fixed.** It stays worse than raw for every offset-plus-gain variant: after
  the floor is neutralized, fish in the blue tanks are still slightly warmer. Hypothesis
  (untested): the two lighting setups have different spectra, which a per-channel gain
  cannot undo.

### Spatial map (flat field)

Held-out floor flatness inside the tank box (p5-p95 spread of the floor, overall scale
removed; fit on other frames, tested on held-out ones):

| map | luminance, blue tanks | B/R, blue tanks | B/R, gray tanks |
|---|---|---|---|
| raw | 0.33-0.36 | 0.60-0.64 | 0.13-0.14 |
| single gain after offset | 0.37-0.41 | 0.42-0.48 | 0.11-0.12 |
| static per-pixel median map (M0) | 0.10-0.12 | 0.13-0.16 | 0.03-0.05 |
| static map, fish-free (M1) | 0.07-0.09 | 0.12-0.16 | 0.03-0.05 |
| 15-sample moving map (W) | 0.06-0.08 | 0.08-0.14 | 0.03-0.04 |

The first two rows come from `boxes_correction.py` (full resolution, even/odd frame split);
the last three from `clean_static_map.py` (4x pooled, leave-one-out). The protocols differ,
so compare within a block, not across it.

- **The plain median map has fish baked in** in tanks 2, 4, 6 (tank 2: 7.8% of the tank;
  a fish resting there is lightened by about 50% on average). Building the map from the
  brightest 60% of frames per pixel (M1) fixes this; an iterative masked version (M2) is
  no better than M1.
- That fix helps luminance (tank 2: 0.124 to 0.084) but **barely touches the blue tanks'
  tint residual**. The tint residual is mostly the lighting *shape* changing in time.
- The moving map is the only thing that reduced it (tanks 2 and 4: about 0.14 to
  0.08-0.09). Tank 6 barely improves (0.157 to 0.144); cause unknown. Its right side has a
  dark region (a person was in frame 0) that was not examined over time.
- The moving map is partly flattered: neighboring frames share non-lighting nuisances
  (shadows, ripples, people).

### White spout

- Clipped in the blue tanks: tanks 2, 4, 6 read exactly 218/218/218, with a histogram
  spike at 218 and no pixel above about 252. So it cannot serve as a white reference there.
- Unclipped in the gray tanks (204-208), but those need little correction, so it cannot
  validate the blue-tank correction either.

## Current best design

Applied once to the **full frame, before cropping**:

> corrected = (pixel - fog) / (map x dial)

- **fog:** one level per tank from the darkest fish-free background pixels, with the drain
  box used only for changes over time (D_t = D_edge + (drain_t - median drain)). The
  combination is proposed, not yet run.
- **map:** a fish-free per-pixel picture of the empty tank (brightest-60% median),
  possibly moving in time.
- **dial:** one brightness/tint scale per channel per frame, estimated from the floor.
  Current prototype: median of frame/map over pixels within 10% of 1. Its flaws: that
  filter limits it to about +/-10% drift; it is noisy frame to frame (smooth it); and the
  evaluation could not test it, because a scalar does not change a relative spread.

Whether a moving map can replace the dial is untested (all comparisons removed the scalar
first).

## What is not established

- Whether the *fish* come out matching across tanks after per-frame or moving-map
  correction. Everything except the crop η² table was measured on the empty floor. This
  needs the crop-to-frame mapping.
- Whether fog, map, and gain are stable across days. Day 1 only.
- The a\* gap, and tank 6's tint residual.
- Whether fog is the same at the frame edge as at the tank center (it was measured only at
  dark spots).

## Corrections to earlier statements in this work

- An early note called the gray tanks "hazy". The measurement says the opposite: the blue
  tanks have the raised black floor; the gray tanks' blacks are crushed.
- An early clipping check used a >= 250 threshold and wrongly reported the blue tanks as
  unclipped; they plateau at 218.
- The fish-in-the-map contamination was first expected to explain the blue tanks' tint
  residual. It explains only part of the luminance residual.

## Next steps

1. **Collaborator request** (sent or about to be): crop-to-frame mapping (frame index
   correspondence, plus crop location/rotation/scale and the resolution it refers to);
   1000 uniformly sampled frames per tank and day (lossless or high-quality, frame index in
   the file name, plus a few short dense bursts of about 30 frames); segmentation masks for
   those frames; original video sizes, frame rate, frame count; camera settings and any color
   processing between the video and the crops or frames; the frame where the divider is
   removed in each session; any empty-tank frames, color card, or black card.
2. **Switch the static map to M1** and keep the fish-masking caveats in mind. Replace the
   brightness-threshold fish mask with the collaborator's segmentation masks once they
   arrive.
3. **Test the moving map on the 1000-frame samples:** sweep the window length, use
   change-point-aware windows around the divider removal and tank 6's step, and check
   whether it can replace the dial.
4. **When the crop mapping arrives,** apply the full correction to crops and recompute the
   η² table, now with per-frame correction. This is the first real test on the fish.
5. **Investigate the unresolved residuals:** tank 6's tint, the a\* gap (try a learned
   per-tank 3x3 color matrix), and which pixels the darkest-background D estimate actually
   uses.
6. **Ask whether the collaborator can re-run cropping on corrected full-frame videos.**
   That would avoid the mapping and put the correction upstream.
7. **Promote to the pipeline** only after the above. A new stage would need the usual
   pieces from `CLAUDE.md`: a `cmd_*.py` wrapper, a business-logic module, `paths.py`
   entries, README and `docs/PHASES.md` entries, and tests.

## Files

| file | purpose |
|---|---|
| `annotate_boxes.py` | matplotlib GUI to draw the tank / white-spout / black-drain boxes; writes `boxes.json` |
| `boxes.json` | the boxes for the six Day 1 sample tanks |
| `prototypes/boxes_correction.py` | drain-based D over time, floor stability, offset + gain, flat field with interior mask, crop η² table |
| `prototypes/window_vs_static_map.py` | contiguous vs random 15-sample windows; moving vs static map on held-out frames |
| `prototypes/clean_static_map.py` | median vs brightness-selected vs iterative-masked vs window maps; fish contamination of the static map |

The prototypes are exploratory, not production code: paths are hardcoded, and they take an
output directory as their one argument. The figures were written to a session scratch
directory and are not preserved; rerun the scripts to regenerate them.
