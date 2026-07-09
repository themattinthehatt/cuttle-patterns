"""Command-line interface for cuttle-patterns.

`main.py` builds the root argparse parser, then auto-discovers every `cmd_*.py` module
in this directory (via `Path.glob('cmd_*.py')`) and calls its `register(subparsers)`
function. There is no central place that lists commands — adding a new subcommand is
just adding a new `cmd_<name>.py` file here; `main.py` never needs to change.

To add a new `cuttle <name>` subcommand, create `cmd_<name>.py` with:

- `register(subparsers: argparse._SubParsersAction) -> None` — adds a subparser via
  `subparsers.add_parser('<name>', help=...)`, defines its arguments with
  `parser.add_argument(...)`, and wires up dispatch with
  `parser.set_defaults(handler=cmd_<name>)`.
- `cmd_<name>(args: argparse.Namespace) -> None` — the handler `main.py` calls with the
  parsed arguments. Keep this thin: it should read config/args, call into real logic
  living in a top-level module (e.g. `cuttle_patterns/ingest.py`), and print/exit. Real
  logic belongs outside `cli/` so it stays testable without argparse in the loop.

See `cmd_ingest.py` (data-processing command backed by `cuttle_patterns/ingest.py`) and
`cmd_setup.py` (interactive prompts, no backing module) for two working examples of this
pattern.
"""
