# distributedKNET
repo of thesis project - Distributed deep kalman filter

## Hyperparameters & Experiments (GraphKalmanProcess)
Use `experiments/run_graphkalmanprocess.py` to run one training or a simple sweep and get a single CSV you can compare.

- Presets + named sweeps live in `experiments/graphkalmanprocess_hparams.py`.
- Results are written to `experiments/results/*.csv` and per-run logs go to `lightning_logs/graphkalmanprocess/*`.

Constant-velocity DKN/GNN-RNN workflow:
- `python -m experiments.run_graphkalmanprocess --localization`
- Skip the prompt: `python -m experiments.run_graphkalmanprocess --localization --description "baseline run"`

Single run:
- `python -m experiments.run_graphkalmanprocess --preset baseline`
- Override any value with dotted keys: `python -m experiments.run_graphkalmanprocess --preset baseline --override model.hidden_dim=128 --override model.lr=1e-4`

Named sweep:
- `python -m experiments.run_graphkalmanprocess --preset baseline --sweep hidden_dim_x_lr`

Ad-hoc grid (cartesian product):
- `python -m experiments.run_graphkalmanprocess --preset baseline --grid model.hidden_dim=32,64,128 --grid model.lr=1e-4,3e-5`

Noise + mismatch sweeps (built-in flags):
- Sweep measurement noise: `python -m experiments.run_graphkalmanprocess --preset baseline --r-scales 0.5,1,2`
- Sweep process + measurement noise: `python -m experiments.run_graphkalmanprocess --preset baseline --q-values 0.1,1 --r-scales 0.5,1,2`
- With/without mismatch (duplicates each run): `python -m experiments.run_graphkalmanprocess --preset baseline --r-scales 0.5,1,2 --with-without-mismatch --mismatch-angle-deg 20`

Evaluation on a graph (post-training):
- `python -m experiments.run_graphkalmanprocess --preset baseline --eval --eval-sims 1024`
