# AGENTS.md

Guidance for AI coding agents working in this repository. This file is meant
to be operational: follow it when planning, editing, testing, and reporting
changes.

## Scope

These instructions apply to the main `distributedKNET` source tree:

- `utils/`
- `experiments/`
- `tests/`
- `models/` when working with code-owned model definitions, checkpoint
  references, or explicitly requested checkpoint management
- `try_models/`
- `skills/` for repo-local Codex skills and their references
- top-level docs and scripts

They do not apply to generated outputs, notebooks, local IDE files, caches, or
virtual environments unless the user explicitly asks to work on those areas.

## Project Summary

This repo implements distributed Kalman filtering and KalmanNet-style graph
models for thesis experiments. The code supports linear and nonlinear system
variants, matched and mismatched process models, classical distributed Kalman
filtering baselines, and learned graph-based KalmanNet models.

The most important code paths are:

- `utils/DistributedKalmanNet.py`: PyTorch/PyTorch-Geometric model components,
  `GraphKalmanProcess`, KalmanNet-style modules, consensus layers, and loss
  helpers.
- `utils/DistributedKalmanData.py`: graph creation, system dynamics,
  measurements, and dataset generation.
- `utils/ClassicDistributedKalman.py`: classical distributed Kalman filtering
  baseline logic.
- `utils/BaselineModels.py`: learned baseline models.
- `utils/reproducibility.py`: seed and deterministic behavior helpers.
- `experiments/graphkalmanprocess_hparams.py`: presets and named sweeps.
- `experiments/run_graphkalmanprocess.py`: main experiment, sweep, dry-run,
  training, and evaluation entry point.
- `tests/`: regression tests. Currently includes focused model behavior tests.

## Out of Scope by Default

Do not modify these unless the user explicitly requests it:

- `notebooks/`
- `lightning_logs/`
- `figures/`
- `experiments/results/`
- `__pycache__/`
- `.pytest_cache/`
- `.idea/`
- `.venv/`
- generated `*.out.txt` and `*.err.txt` files

Treat trained checkpoint files (`*.pt`, `*.pth`) as artifacts. Do not rewrite,
delete, rename, or regenerate them unless that is the task.

## Environment and Dependencies

The project is Python-based and expects:

- Python 3.12
- PyTorch
- PyTorch Lightning
- PyTorch Geometric
- NumPy
- Matplotlib
- tqdm
- pytest for tests

Prefer the existing local environment if present. Do not install or upgrade
dependencies unless the user asks or a required validation step cannot run
without it. If dependencies are missing, report exactly what could not be run.

## Working Principles

- Read the relevant files before editing. Let the existing module boundaries
  and naming style guide the change.
- Keep edits narrowly scoped to the requested behavior.
- Preserve experiment reproducibility: seed handling, config flow, deterministic
  flags, output paths, and run naming matter.
- Avoid large refactors unless the user asked for one or the current change
  cannot be made safely without it.
- Do not convert notebooks into modules or scripts unless requested.
- Do not hand-edit generated results, logs, figures, caches, or checkpoint
  artifacts.
- Prefer adding small, targeted tests for behavioral changes.
- Preserve public experiment CLI behavior unless the user explicitly asks to
  change it.
- Keep new text and code ASCII unless the file already intentionally uses
  Unicode.

## Coding Conventions

- Follow the existing Python style: type hints where useful, simple helper
  functions, explicit imports, and clear tensor-shape handling.
- Keep tensors device-safe. Avoid creating CPU tensors inside model code unless
  they are immediately moved to the relevant device.
- Be careful with dtype. The project generally uses `torch.float`.
- Do not introduce hidden global state that affects experiments across runs.
- Keep config values in `experiments/graphkalmanprocess_hparams.py` and runtime
  expansion logic in `experiments/run_graphkalmanprocess.py`.
- When changing Kalman/data logic, check tensor shapes and batching behavior
  explicitly. Graph batches from PyTorch Geometric are a common source of subtle
  regressions.
- Avoid broad exception handling in training or evaluation code. Prefer clear
  validation errors when config values are invalid.
- If adding a module, place it in the existing package layout and update imports
  explicitly.
- If adding or updating a repo-local skill under `skills/`, keep `SKILL.md`
  concise, include only directly useful resources, and validate the skill
  metadata when possible.

## Experiment and Config Rules

- Use `PRESETS` and `SWEEPS` in `experiments/graphkalmanprocess_hparams.py` for
  named experimental variants.
- Keep one-off runtime changes as CLI overrides when possible:
  `--override dotted.key=value`.
- For noise comparison experiments, train a separate model for each noise level
  instead of evaluating one checkpoint across all `r_scale` values.
- For mismatch comparison experiments, train matched and mismatched conditions
  separately. Do not reuse a matched checkpoint for mismatched evaluation, or a
  mismatched checkpoint for matched evaluation, unless the user explicitly asks
  for cross-condition generalization.
- Use `--dry-run` to validate run planning without training.
- Avoid long training runs as validation unless the user requested them. Prefer
  `fast_debug`, tiny overrides, or dry runs for smoke checks.
- Results are written to `experiments/results/*.csv`.
- Lightning logs are written to `lightning_logs/`.
- When changing output fields or metrics, update code and documentation
  together.

## Validation

Use the smallest validation that covers the change.

For utility or model changes:

```powershell
pytest tests
```

For the existing model regression test:

```powershell
pytest tests/test_distributed_kalman_net.py
```

For experiment config or CLI changes:

```powershell
python -m experiments.run_graphkalmanprocess --preset fast_debug --dry-run
```

For a minimal training smoke test, only run a very small configuration unless
the user has asked for real training:

```powershell
python -m experiments.run_graphkalmanprocess --preset fast_debug --override data.train_sims=8 --override data.val_sims=4 --override trainer.max_epochs=1 --run-name smoke
```

If validation cannot be run because dependencies are missing or the environment
is unavailable, say so clearly and include the attempted command.

## Git and File Safety

- The working tree may contain user changes. Do not revert or overwrite changes
  you did not make.
- Check the current diff before making broad edits.
- Do not use destructive Git operations such as `reset --hard` or checkout-based
  reverts unless the user explicitly asks.
- If Git commands fail because the repository is not marked as a safe directory
  in the current sandbox user, do not change global Git config unless the user
  approves. Continue with filesystem inspection when possible.
- Keep generated files out of commits unless the task is specifically about
  generated outputs.

## Documentation Expectations

- Update `README.md` when user-facing setup, commands, outputs, or experiment
  workflows change.
- Keep comments concise and useful. Add comments for non-obvious math, tensor
  shape expectations, or reproducibility constraints.
- Do not duplicate large blocks of README content in this file. This file should
  tell agents how to work; the README should tell users how to use the project.

## Best Practices for Maintaining This File

- Keep guidance specific to this repository. Avoid generic coding-agent advice
  that could apply anywhere.
- Prefer stable instructions over temporary notes.
- Include exact commands when there is a preferred validation path.
- Document exclusions and artifact policies clearly so future agents do not
  churn generated files.
- Mention known architectural boundaries and invariants, especially around
  reproducibility and experiment configuration.
- Update this file when the repo layout, primary entry points, validation
  commands, or artifact policy changes.
- Keep it short enough to read before making a change.

## Agent Response Expectations

When reporting work back to the user:

- Summarize the actual files changed.
- State which validation command was run and whether it passed.
- If validation was skipped or failed, explain why and what remains risky.
- Call out any changes to experiment outputs, metrics, or saved artifacts.
