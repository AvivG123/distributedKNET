# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Thesis project: a Distributed deep Kalman filter (DKN) for multi-agent target tracking. The core idea is that a graph neural network replaces the hand-crafted Kalman gain in an EKF, letting it learn to correct for model mismatch and nonlinearities that classical filters cannot handle.

## Running training

```bash
# Activate the virtual env first (system site-packages are included)
source .venv/bin/activate

# Train DKN (reads configs/const_vel_scenario_configured.json)
python DKN_training.py
```

The script prompts for a description string, then trains one model per `(measurement_noise, dt_mismatch)` pair. Artifacts land in `models/kfir/experiment_N/` (standard) or `models/kfir/dt_mismatch/experiment_N/` (when `use_dt_mismatch=true`).

## Key config knobs (`configs/const_vel_scenario_configured.json`)

| Key | Effect |
|---|---|
| `use_dt_mismatch` | `true` trains one DKN per ratio in `dt_mismatch_values` |
| `dt_mismatch_values` | list of `dt' / dt` multipliers applied to the model's F matrix |
| `measurement_noise_values` | list of `r` values; one model trained per value |
| `save_root` | relative path under repo root where `experiment_N/` directories are created |
| `max_epochs` | capped by early stopping (patience=5, min_delta=0.001) |

## Architecture

### Filters

Three filters are implemented and compared:

| Class | File | Notes |
|---|---|---|
| `GraphKalmanProcess` | `utils/DistributedKalmanNet.py` | DKN — PyTorch Lightning module, trained end-to-end |
| `centralized_extended_kalman_filter` | `utils/ClassicDistributedKalman.py` | CEKF — all nodes pool measurements, single global estimate |
| `diffusion_extended_kalman_filter_parallel_edge` | `utils/ClassicDistributedKalman.py` | DEKF — each node fuses neighbours' innovation via adjacency diffusion |

All three share the same `f_system` / `h_system` interface, making it straightforward to swap dynamics or swap scenarios.

### The `f_system` / `h_system` protocol

Any system model must implement:
- `__call__(x)` — apply state transition or measurement function
- `jacobian(x, n_expansions)` — return the Jacobian

`ConstantVelocityModel(time_delta)` builds the F matrix once at construction. To introduce dt mismatch, construct two instances: one for data generation (true `dt`) and one for the filter (wrong `dt * ratio`). This is the only place the mismatch lives.

`DistanceAngleObservation` alternates nodes between distance and angle sensors (even index = angle, odd index = distance). It implements `wrap_innovation` / `wrap_measurements` to keep angle residuals in `[-π, π]`.

### DKN internals (`utils/DistributedKalmanNet.py`)

`GraphKalmanProcess` (Lightning) wraps `GraphKalmanFilter`, which runs one predict–update step per time step:

1. **Prediction**: `x̂_{t|t-1} = F · x̂_{t-1|t-1}` via `f_system`
2. **Edge Kalman** (`EdgeKalmanFilter`): computes classical diffusion innovation `Δy` from neighbour measurements
3. **Node GNN-RNN** (`NodeKalmanGnnRnn`): GRU + GCN produces a learned Kalman gain matrix `K` per node
4. **Cross Kalman gain** (`CrossKalmanGain`, `MessagePassing`): optional learned edge correction (enabled when `learn_edge_kalman=True`)
5. **Update**: `x̂_{t|t} = x̂_{t|t-1} + K · Δy`
6. **Diffusion**: `SimpleConv` (graph mean pooling) to propagate estimates across neighbours

Loss is mean position RMSE across nodes and time steps (`loss_function` in `DistributedKalmanNet.py`).

### Data pipeline (`utils/DistributedKalmanData.py`)

`GraphDataset` generates all Monte Carlo trajectories at construction and wraps them as PyG `Data` objects:
- `data.x` — measurements `[num_nodes, time_steps, 1]`
- `data.y` — ground truth states `[time_steps, state_dim, 1]`
- `data.edge_index` — bidirectional edge list (self-loops included)
- `data.adj_matrix` — dense adjacency with self-loops on diagonal

### Experiment directory layout

```
models/kfir/
  experiment_N/
    config.json          # full config snapshot
    DKN/
      r=1.0.pth          # standard run
      r=1.0_dtx0.9.pth   # dt-mismatch run
    plots/
```

`next_experiment_dir(save_root)` / `list_experiments(save_root)` (in `utils/ConstantVelocityScenario.py`) manage the `experiment_N` numbering. The comparison notebook calls `list_experiments` to enumerate experiments — **this means the notebook only works after at least one training run with the current script**.

## Notebooks

`notebooks/const_vel_scenario_comparison.ipynb` is the main evaluation notebook. It:
1. Lists `experiment_*` dirs under `save_root`, you pick one
2. Runs a single-trial visual comparison (CEKF / DEKF / DKN)
3. Runs a Monte Carlo sweep over `measurement_noise_values`
4. (New section at the bottom) Runs a Monte Carlo sweep over `dt_mismatch_values` — set `MISMATCH_EXPERIMENT` and `R_MISMATCH_EVAL` in the setup cell

## GPU / precision

`DKN_training.py` sets `torch.set_float32_matmul_precision("high")` globally (TF32 on Ampere+). BF16 mixed precision (`"bf16-mixed"`) is passed to the Lightning `Trainer` only when `get_trainer_accelerator()` returns `"gpu"` — not on MPS or CPU. `get_trainer_accelerator()` is defined in `utils/ConstantVelocityScenario.py`.
