# Claude Development Guidelines

This file contains project-specific guidelines for Claude Code when working on this project.

## Code Style

### Comments
- Use docstrings for all functions, classes, and modules
- Follow Google-style docstrings
- Add inline comments for complex logic
- Avoid obvious comments
- Comments should start with lowercase letters

### Type Hints
- Use type hints for all function parameters and return values
- Import types from typing module when needed
- use X | Y, list[X], dict[K, V] syntax; avoid typing imports for these

### Imports
- Group imports: standard library, third-party, local
- Use absolute imports (e.g., `from <package_name>.<modlue_name> import func` not `from .<module_name> import func`)
- Sort imports alphabetically within groups
- Avoid wildcard imports

### Code Formatting
- Line length: 99 characters
- Use 4 spaces for indentation
- Follow PEP 8 conventions
- Use meaningful variable names
- Use idx for index
- Use adjectives after nouns, such as idx_train and idx_test rather than train_idx and test_idx
- Include newline at the end of every .py file
- Do not allow trailing whitespace
- Do not include whitespace for blank lines
- Use single quotes for strings
- Add a comma to the end of multi-line function arguments:
```python
foo(
    param1,
    param2,
)
```
not
```python
foo(
    param1,
    param2
)
```

### Misc
- Use `pathlib.Path` instead of `os` for path handling
- Use f-strings for all string interpolation, including `logger` calls and `print` statements — never `%s` formatting or `.format()`

## Testing

### Unit Tests
- Use pytest framework
- Test directory structure must mirror the source package structure exactly:
  strip the `<package_name>/` prefix and prepend `tests/` — the rest of the path is identical.
  `<package_name>/a/b/c.py` → `tests/a/b/test_c.py`, at every level of nesting without exception.
  - Each subdirectory under `tests/` must have an `__init__.py`
- Create test classes for each function
- Use fixtures for common test data; place fixtures in a `conftest.py` in the same directory as the tests that use them
- Test assets (e.g. sample images) live in an `assets/` subdirectory alongside the tests that use them
- Aim for high test coverage
- Test both success and failure cases

### Test Structure
```python
class Test<function_name>:
    """Test the function <function_name>."""

    def test_<function_name>_<scenario>(self):
        # Arrange
        # Act
        # Assert
```

### Mocking
- Use unittest.mock for external dependencies
- Mock at the boundary of your system
- Use dependency injection when possible

### GPU test markers
Two pytest marks gate hardware-dependent tests:

- `@pytest.mark.gpu` — test requires a CUDA GPU (uses a GPU trainer, DALI pipeline, or
  moves tensors to CUDA). The CPU CI workflow runs `pytest -m "not gpu"` and will skip
  these. **Any new test that needs a GPU must carry this mark.**
- `@pytest.mark.multigpu` — test additionally requires two or more GPUs. Always paired
  with `@pytest.mark.gpu` and `@pytest.mark.skipif(torch.cuda.device_count() < 2, ...)`.

For files where every test needs a GPU, use a module-level `pytestmark = pytest.mark.gpu`
instead of decorating each function individually. For parametrized tests with both CPU and
GPU variants, apply the mark only to the GPU parameter via
`pytest.param(..., marks=pytest.mark.gpu)` so the CPU variant still runs in CPU CI.

## Documentation

### Docstrings
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: Description of when this exception is raised
    """
```
If there are too many params for a single line, put one param per line, indented four spaces:
```python
def function_name(
    param1: Type1,
    param2: Type2,
    param3: Type3,
    param4: Type4,
) -> ReturnType:
```

### Module Documentation
- Every module should have a module-level docstring
- Describe the purpose and main functionality

## Error Handling

### Exceptions
- Use specific exception types
- Provide meaningful error messages
- Log errors appropriately
- Use try/except blocks judiciously

### Logging
- Use Python's logging module
- Include appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Log important events and errors
- Include context in log messages

## Project Structure

### Package Organization
- Keep modules focused and cohesive
- Use clear, descriptive module names
- Group related functionality together
- Avoid circular imports

### File Naming
- Use snake_case for Python files
- Use descriptive names
- Group related files in subdirectories

## CLI Guidelines

### Command Structure
- Use subcommands for different operations
- Provide clear help messages
- Validate input parameters
- Handle errors gracefully

### Configuration
- Use sensible defaults
- Validate configuration before execution

### Adding a new pipeline stage / `cuttle` subcommand
Every existing stage (e.g. `cuttle cluster`/`cluster.py`, `cuttle clusterview`/
`visualization/cluster_frames.py`) follows the same shape. Adding a new one touches:

1. `cuttle_patterns/cli/cmd_<name>.py` — thin wrapper only: `register(subparsers)` wires
   argparse (auto-discovered by `main.py`'s `cmd_*.py` glob — no central registry to
   edit), and `cmd_<name>(args)` resolves `results_dir` (config-or-override, with the
   established try/except → `print(f'Error: {e}')` + `sys.exit(1)` pattern) and
   delegates to the real logic.
2. A business-logic module with the real implementation as pure, testable functions —
   either top-level (`cuttle_patterns/<name>.py`, e.g. `cluster.py`) or under an
   existing subpackage (`embedders/`, `visualization/`, `eval/`) if the stage belongs to
   one.
3. `cuttle_patterns/paths.py` — add the new stage's `*_RELPATH` constant(s) and a line
   in the module docstring's tree, with a `# cuttle <name>` comment.
4. `README.md` — a new `### \`cuttle <name>\`` section (unnumbered — order is implied by
   position, so nothing else needs renumbering), matching the existing sections' shape:
   one-paragraph description, example command, then flags/output. Point to
   `cuttle_patterns/paths.py` for the exact output-path template rather than restating
   it, and to `--help` for flag defaults rather than restating them.
5. `docs/PHASES.md` — a bullet on an existing phase, or a new `## Phase N` entry (a
   stable, append-only numbered ID — never renumbered, since other docs reference phases
   by number).
6. Tests in both locations: `tests/cli/test_cmd_<name>.py` (CLI wiring) and a mirrored
   test module for the business-logic module (`tests/test_<name>.py`, or nested to match
   its subpackage).
7. The previous stage's `cmd_<name>.py` gets an updated closing
   `print('next: cuttle <name> ...')` hint, naming the new stage.

## Dependencies

### Adding Dependencies
- Only add dependencies that are truly needed
- Prefer well-maintained packages
- Do not pin versions
- Update pyproject.toml and document changes

## Documentation Map

Where things are documented, and what each doc is *for* — check this before searching
multiple files for the same fact:

- `README.md` — setup instructions + the full `cuttle` CLI usage reference, one section
  per subcommand (flags, example invocation). Points to `--help` for flag defaults and
  to `cuttle_patterns/paths.py` for exact output paths, rather than restating either.
- `docs/PHASES.md` — the living design/status narrative, phase-by-phase (stable,
  append-only numbered IDs — never renumbered). This is where algorithm-level detail
  lives (inscription, smoothing, extraction, embedder readouts, etc.), edited in place
  as the implementation changes.
- `docs/DECISIONS.md` — append-only, reverse-chronological decision log: what was
  decided, why, alternatives considered, trade-offs. Points to `PHASES.md` for mechanism
  rather than restating it — read `PHASES.md` for *what/current state*, this for *why*.
- `docs/pose_estimation.md` — living status doc for the external Lightning Pose
  tail/neck model (training rounds, active-learning loop) consumed by `cuttle inscribe`'s
  pose-informed path.
- `docs/latent_space_confounds.md` — living status doc narrating the AE → MSPS-VAE →
  masked-MSPS-VAE confound-diagnosis timeline; only "Current open issue" is still live,
  the rest is historical narrative.
- `docs/implementation_notes/embedder.md` — living implementation reference for
  `cuttle embed`'s backbone/readout protocol (DINOv3, VGG-19, Gram-matrix math) — the
  canonical source for embedder detail; other docs point here rather than restating it.
- `docs/implementation_notes/eval_plan.md` — historical design doc for the eval harness
  (captures the original plan, including what was deliberately never built). For current
  status, see `cuttle_patterns/eval/README.md` instead of this doc's present-tense claims.
- `docs/implementation_notes/msps_vae.md` — living implementation reference for the
  MSPS-VAE model/sampler/config; its "Implementation gotchas"/"Resolved during
  implementation" sections are a dated log, not rewritten after the fact.
- `cuttle_patterns/eval/README.md` — how-to-run guide for the eval harness (current
  scope, metrics, CLI usage) — the up-to-date counterpart to `eval_plan.md`'s historical
  plan.
