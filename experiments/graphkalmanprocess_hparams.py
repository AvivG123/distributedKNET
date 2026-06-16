"""Hyperparameter presets + sweeps for GraphKalmanProcess experiments.

Edit this file to add new presets or sweeps you want to compare.
"""

from __future__ import annotations

from copy import deepcopy


BASELINE: dict = {
    "seed": 42,
    "system": {
        # "linear" uses FSystemLinear/HSystemLinear, "nonlinear" uses FSystem/HSystem.
        "kind": "linear",
        # Degrees: use different values for mismatch experiments.
        "deg_true": 0.0,
        "deg_model": 0.0,
        "h_alpha": 0.0,
    },
    "graph": {
        "node_num": 50,
        "k_neighbors": 5,
        "rewrite_prob": 0.4,
    },
    "data": {
        "time_steps": 20,
        "q": 1.0,
        # IMPORTANT: the model code currently assumes a scalar R (see EdgeKalmanFilter.repeat()).
        "r_scale": 1.0,
        "x0_scale": 1.0,
        "n_expansions": 1,
        "train_sims": 10_000,
        "val_sims": 256,
        "batch_size": 64,
    },
    "model": {
        "signal_dim": 2,
        "edge_features_dim": 1,
        "node_kalman_dim": 4,
        "edge_kalman_dim": 2,
        "hidden_dim": 64,
        "heads": 1,
        "dropout": 0.0,
        "lr": 5e-5,
        "learn_edge_kalman": False,
        # Diffusion/consensus step at the end of GraphKalmanFilter:
        # "none" (default), "simple" (SimpleConv), "gcn" (GCNConv).
        "consensus_layer": "simple",
        "x0_scale": 1.0,
    },
    "trainer": {
        "max_epochs": 20,
        "log_every_n_steps": 50,
        "early_stop_patience": 5,
        "early_stop_min_delta": 0.01,
        "gradient_clip_val": 1.0,
    },
    "curriculum": {
        "enabled": False,
        "start_time_steps": 10,
        "step_time_steps": 10,
        "max_time_steps": 20,
        "epochs_per_stage": 10,
    },
}


def _with(base: dict, updates: dict) -> dict:
    cfg = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


PRESETS: dict[str, dict] = {
    "baseline": BASELINE,
    "edge_kalman_on": _with(BASELINE, {"model": {"learn_edge_kalman": True}}),
    "bigger_hidden": _with(BASELINE, {"model": {"hidden_dim": 128}}),
    "mismatch_20deg": _with(BASELINE, {"system": {"deg_true": 20.0, "deg_model": 0.0}}),
    "fast_debug": _with(
        BASELINE,
        {
            "data": {"train_sims": 512, "val_sims": 128, "batch_size": 32, "time_steps": 10},
            "trainer": {"max_epochs": 3, "early_stop_patience": 2, "early_stop_min_delta": 0.0},
        },
    ),
}


# Named sweeps: each entry is a dict of dotted keys -> list of values.
SWEEPS: dict[str, dict[str, list]] = {
    "hidden_dim_x_lr": {
        "model.hidden_dim": [32, 64, 128],
        "model.lr": [1e-4, 3e-5, 1e-5],
    },
    "edge_kalman_toggle": {
        "model.learn_edge_kalman": [False, True],
    },
    "dropout": {
        "model.dropout": [0.0, 0.1, 0.2],
    },
}
