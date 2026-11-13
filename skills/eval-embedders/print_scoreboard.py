"""Print the eval scoreboard (`results_dir/eval/metrics.json`) as one compact markdown table.

Status: tool, used by the eval-embedders skill. `DataFrame.to_markdown` is avoided because
`tabulate` is not installed in the cuttle env.
"""

import json

import pandas as pd

from cuttle_patterns.config import load_config
from cuttle_patterns.paths import EVAL_RELPATH

COLUMNS = {
    'embedder': 'embedder',
    'ami_cluster_class_mean': 'AMI class',
    'ami_cluster_video_within_class_mean': 'AMI video|class',
    'effective_videos_per_cluster_mean': 'eff videos/cluster',
    'cross_video_knn_accuracy': 'x-video kNN',
    'linear_probe_accuracy': 'lin probe',
    'variance_retained': 'var ret',
}


def main() -> None:
    """Read metrics.json and print it as a markdown table."""
    rows = json.loads((load_config().results_dir / EVAL_RELPATH / 'metrics.json').read_text())
    df = pd.DataFrame(rows)[list(COLUMNS)].round(3)
    print('| ' + ' | '.join(COLUMNS.values()) + ' |')
    print('|' + '---|' * len(COLUMNS))
    for row in df.itertuples(index=False):
        print('| ' + ' | '.join(str(x) for x in row) + ' |')


if __name__ == '__main__':
    main()
