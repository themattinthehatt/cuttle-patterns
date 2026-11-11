"""Set one global floor target across all fitted sessions.

Each session's params npz stores `floor_ref` (its floor level in linear light after fog removal).
The global target is the median of those, applied as a neutral gray to every session so the
corrected floor is the same in every tank (equalizes brightness and tint across sessions).

Example:
    python scratch/photometry/prototypes/set_global_target.py
"""

from pathlib import Path

import numpy as np

PARAMS = Path('/media/mattw/CUTTLE/results/.iter-1.0-photometry/params')


def main() -> None:
    """Compute the global target from per-session floor levels and write it into each npz."""
    files = sorted(f for f in PARAMS.glob('Day*_Tank?.npz'))
    refs = {f.stem: float(np.load(f)['floor_ref']) for f in files}
    for name, ref in refs.items():
        print(f'{name}: floor_ref {ref:.3f}')
    global_target = float(np.median(list(refs.values())))
    print(f'global target (median): {global_target:.3f}')
    for f in files:
        data = dict(np.load(f))
        data['target'] = np.full(3, global_target, dtype=np.float32)
        data['target_is_global'] = np.array(True)
        np.savez_compressed(f, **data)


if __name__ == '__main__':
    main()
