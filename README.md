# cuttle-patterns

Unsupervised analysis of visual patterns displayed by cuttlefish during social
interaction: egocentric alignment of segmented cuttlefish videos, self-supervised
embedding via BEAST, and interactive tools for exploring the resulting pattern clusters.

## Docs

- [docs/PHASES.md](docs/PHASES.md) — project phases/roadmap
- [docs/DECISIONS.md](docs/DECISIONS.md) — decision log

## Setup

```bash
conda create -n cuttle python=3.12
conda activate cuttle
pip install -e ".[dev]"
```

Each machine needs a local config file at `~/.cuttle-patterns/config.yaml` pointing to
where data and results live, e.g.:

```yaml
data_dir: /media/mattw/poseinterface/cuttle
results_dir: /home/mattw/cuttle-patterns-results
```

## License

MIT
