"""Hyperparameter presets + sweeps for GraphKalmanProcess experiments.

Edit this file to add new presets or sweeps you want to compare.
"""

from __future__ import annotations

import math
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
        # "none" (default), "simple" (SimpleConv), "gcn" (GCNConv), "adaptive".
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

# Localization baseline experiment. Change only in your local copy.
LOCALIZATION_BASELINE: dict = _with(BASELINE, {
    "seed": 42,
    "save_root": "my_models",
    "system": {
        "time_delta": 0.1,
        "area_size": 100.0,
        "p0_scale": 10.0,
        "use_dt_mismatch": True,
        "dt_mismatch_values": [2.0],
    },
    "graph": {
        "node_num": 50,
        "k_neighbors": 5,
        "graph_seed": 42,
    },
    "data": {
        "time_steps": 20,
        "mu": 1,  # q**2 / r_scale**2
        "rho": (1.0 / math.radians(10.0)) ** 2,
        "r_scale": [1],
        # q           = r_scale * math.sqrt(mu)
        # sigma_r     = r_scale
        # sigma_theta = r_scale / math.sqrt(rho)
        "train_sims": 20_000,
        "val_sims": 256,
        "batch_size": 64,
        "num_trials": 50,
        "preview_trajectories": 4,
    },
    "model": {
        "signal_dim": 4,
        "edge_features_dim": 1,
        "node_kalman_dim": 16,
        "edge_kalman_dim": 2,
        "hidden_dim": 128,
        "lr": 1e-4,
        "learn_edge_kalmanq": True,
        "train_models": ["dkn"],
        "gnn_rnn_hidden_dim": 128,
        "gnn_rnn_learning_rate": 1e-4,
        "consensus_layer": "adaptive",
        "position_only_loss": True,
    },
    "trainer": {
        "max_epochs": 100,
        "early_stop_patience": 5,
        "early_stop_min_delta": 0.001,
        "gradient_clip_val": 1,
    },
    "curriculum": {

        "enabled": True,
        "start_time_steps": 10,
        "step_time_steps": 10,
        "max_time_steps": 20,
        "epochs_per_stage": 10,
    },
})

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
