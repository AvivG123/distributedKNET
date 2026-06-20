---
name: train-compare-baseline
description: Train a distributedKNET experiment from a requested setup, inspect the resulting metrics/checkpoint, evaluate it against the classical diffusion EKF baseline, and generate a comparison graph like the end of notebooks/non_linear_simulation.ipynb. Use when the user asks to run training, check experiment results, compare GraphKalmanProcess or Distributed KalmanNet to a baseline, sweep noise levels, or recreate the nonlinear/linear comparison plots.
---

# Train Compare Baseline

## Overview

Use this skill to turn a requested distributedKNET setup into a complete
experiment loop: validate the config, train or dry-run it, inspect the saved
results, evaluate the trained model and baseline on matching data, and create a
publication-style comparison plot.

Each noise level must get its own trained model. Matched and mismatched
conditions must also be trained separately. Only reuse a checkpoint across noise
or mismatch conditions when the user explicitly asks for cross-condition
generalization.

## Workflow

1. Identify the setup.
   - Prefer existing presets in `experiments/graphkalmanprocess_hparams.py`.
   - Use `--override key=value` for user-provided changes.
   - Preserve `seed`, graph size, system kind, mismatch angle, time steps,
     `q`, `r_scale`, `x0_scale`, and batch sizes unless the user asks to change
     them.
   - For notebook-style comparisons, default to noise scales
     `0.25,0.5,1,2,4` and x-axis `10 * log10(1 / r_scale**2)`.
   - Treat each noise scale and each mismatch state as a separate training run.

2. Prefer the bundled automation script when the user wants the complete
   train/evaluate/plot loop:

```powershell
python skills/train-compare-baseline/scripts/train_eval_compare.py --preset <preset> --r-scales 0.25,0.5,1,2,4 --run-name <name>
```

   - The script trains each `r_scale` separately.
   - By default, the script also trains both matched and mismatched variants
     separately with `--mismatch-mode both`.
   - Use `--mismatch-mode matched`, `--mismatch-mode mismatched`, or
     `--mismatch-mode config` only when the user asks for a narrower run.
   - Use `--mismatch-angle-deg 20` to control the mismatched true dynamics.
   - Add `--override key=value` for setup changes.
   - Add `--dry-run` before expensive training.
   - The script writes a comparison CSV under `experiments/results/` and a PNG
     under `figures/`.

3. Validate run planning before manual training.
   - Run a dry run first:

```powershell
python -m experiments.run_graphkalmanprocess --preset <preset> --dry-run
```

   - If training would be expensive and the user did not explicitly ask for a
     full run, use `fast_debug` or tiny overrides for a smoke check.

4. Train the requested setup manually only when the script is not flexible
   enough.
   - Use the experiment runner instead of notebook cells:

```powershell
python -m experiments.run_graphkalmanprocess --preset <preset> --run-name <name>
```

   - Add user-specified overrides and sweeps through the runner CLI.
   - For a noise sweep:

```powershell
python -m experiments.run_graphkalmanprocess --preset <preset> --r-scales 0.25,0.5,1,2,4 --run-name <name>
```

   - For matched and 20-degree mismatched variants:

```powershell
python -m experiments.run_graphkalmanprocess --preset <preset> --r-scales 0.25,0.5,1,2,4 --with-without-mismatch --mismatch-angle-deg 20 --run-name <name>
```

   - Confirm the planned run count is `noise_levels * mismatch_states`.
     Example: five noise levels with matched and mismatched conditions should
     plan ten training runs.

5. Inspect results.
   - Check `experiments/results/*.csv` for one row per run.
   - Check `lightning_logs/graphkalmanprocess/<run-name>/metrics.csv` for
     training/validation curves.
   - Record the best validation loss, best checkpoint path, run name, config
     values, and any evaluation loss.
   - If the run generated no CSV row or checkpoint, inspect the terminal/log
     output before plotting and report the issue.

6. Evaluate the trained model against the baseline.
   - Use the same graph, system, `q`, `r_scale`, `x0_scale`, seed, and time
     steps for the learned model and the DEKF baseline.
   - Use `diffusion_extended_kalman_filter_parallel_edge` from
     `utils/ClassicDistributedKalman.py` for the classical baseline.
   - Use `GraphKalmanProcess` from `utils/DistributedKalmanNet.py` for the
     trained model.
   - Convert comparison metrics to dB with `10 * log10(mse)` when matching the
     notebook's final plots.
   - Prefer recomputing metrics from the trained checkpoint and a fresh
     evaluation dataset. Only use hardcoded notebook arrays when the user asks
     to reproduce the notebook figure exactly.

7. Generate the graph.
   - Save figures under `figures/` unless the user asks for another path.
   - Use x-axis label `$\frac{1}{r^2}$ [dB]` for notebook-style noise plots.
   - Use y-axis label `MSE [dB]` for dB comparisons.
   - Include at least `DEKF` and `Distributed KalmanNet`; include mismatch or
     GNN-RNN curves only when evaluated or explicitly requested.
   - Use clear legend labels and grid lines.

## Notebook Comparison Reference

When the user asks to match the end of
`notebooks/non_linear_simulation.ipynb`, read
`references/nonlinear-notebook-comparison.md`. It contains the relevant helper
functions, metric definitions, plotted arrays, and figure conventions extracted
from the notebook tail.

## Reporting

Report the run command, result CSV/log paths, best validation score, checkpoint
path, evaluation metric summary, and generated figure path. If training or
evaluation could not run because dependencies or accelerator support are
missing, state the attempted command and what blocked it.
