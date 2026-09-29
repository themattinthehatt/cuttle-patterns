"""Experiment metadata that isn't encoded in video names, shipped with the package.

Video names carry `day`, `tank`, and `role` (Resident/Intruder) but not which animal was
filmed. `assets/identity_mapping.csv` records, for each resident-intruder session
(one per `(day, tank)`), the IDs of the resident (`R1`-`R6`) and intruder (`I1`-`I6`),
converted from `raw_data/identity_mapping.txt`. Residents stay in their tank for the
whole experiment; intruders rotate across tanks day to day.
"""

from pathlib import Path

import pandas as pd

IDENTITY_MAPPING_PATH = Path(__file__).parent / 'assets' / 'identity_mapping.csv'

INDIVIDUAL_COLUMN = 'individual'


def load_identity_mapping(path: Path = IDENTITY_MAPPING_PATH) -> pd.DataFrame:
    """Load the resident/intruder identity mapping, one row per session.

    Args:
        path: path to the mapping CSV; defaults to the copy shipped with the package.

    Returns:
        DataFrame with columns `session`, `date`, `day`, `tank`, `resident`, `intruder`.
    """
    return pd.read_csv(path)


def attach_individual_column(
    meta: pd.DataFrame,
    mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach each frame's individual ID (e.g. `R3`, `I5`) as an `individual` column.

    Args:
        meta: per-frame metadata with columns `day`, `tank`, `role`, where `role` is
            `Resident` or `Intruder`.
        mapping: identity mapping as returned by `load_identity_mapping`; loaded from the
            packaged CSV if None.

    Returns:
        a copy of `meta` with an added `individual` column, row order preserved.

    Raises:
        ValueError: if any `(day, tank)` in `meta` is missing from the mapping, or any
            `role` is neither `Resident` nor `Intruder`.
    """
    if mapping is None:
        mapping = load_identity_mapping()

    roles_unknown = set(meta['role']) - {'Resident', 'Intruder'}
    if roles_unknown:
        raise ValueError(f'unknown role(s) {sorted(roles_unknown)}; expected Resident/Intruder')

    # long format: one row per (day, tank, role)
    long = mapping.melt(
        id_vars=['day', 'tank'],
        value_vars=['resident', 'intruder'],
        var_name='role',
        value_name=INDIVIDUAL_COLUMN,
    )
    long['role'] = long['role'].str.capitalize()

    out = meta.drop(columns=INDIVIDUAL_COLUMN, errors='ignore').merge(
        long,
        on=['day', 'tank', 'role'],
        how='left',
        validate='many_to_one',
    )
    out.index = meta.index

    missing = out.loc[out[INDIVIDUAL_COLUMN].isna(), ['day', 'tank']].drop_duplicates()
    if len(missing) > 0:
        pairs = sorted(map(tuple, missing.values.tolist()))
        raise ValueError(f'no identity mapping for (day, tank) pair(s) {pairs}')

    return out
