"""Training, checkpoint, configuration, and plotting helpers for localization."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

from utils.DistributedKalmanData import GraphDataset
from utils.LocalizationScenario import (
    ConstantVelocityModel,
    DistanceAngleObservation,
    build_dkn_model,
    create_distance_based_graph,
    generate_node_positions,
    localization_x0,
)
from utils.reproducibility import seed_everything


def derive_localization_noise(mu, rho, r_scale):
    mu = float(mu)
    rho = float(rho)
    r_scale = float(r_scale)
    if mu < 0 or rho <= 0 or r_scale < 0:
        raise ValueError("mu and r_scale must be nonnegative and rho must be positive.")
    q = r_scale * math.sqrt(mu)
    sigma_r = r_scale
    sigma_theta = r_scale / math.sqrt(rho)
    q_matrix = np.eye(4) * q ** 2
    return {
        "q": q,
        "sigma_r": sigma_r,
        "sigma_theta": sigma_theta,
        "q_matrix": q_matrix,
    }


def measurement_noise_for_nodes(h_system, sigma_r, sigma_theta):
    node_types = h_system.node_classification[:, 0].cpu().numpy().astype(bool)
    return np.where(node_types, float(sigma_r), float(sigma_theta))


def get_trainer_accelerator():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def normalize_localization_train_models(config):
    aliases = {
        "dkn": "dkn",
        "gnn_rnn": "gnn_rnn",
        "gnn-rnn": "gnn_rnn",
        "gnnrnn": "gnn_rnn",
    }
    raw_models = config.get("train_models", ["dkn"])
    if isinstance(raw_models, str):
        raw_models = [raw_models]
    if any(str(model).lower() == "all" for model in raw_models):
        return {"dkn", "gnn_rnn"}
    result = set()
    for model in raw_models:
        key = str(model).lower()
        if key not in aliases:
            raise ValueError(
                f"Unknown train model {model!r}. Expected one of {sorted(aliases)} or 'all'."
            )
        result.add(aliases[key])
    return result


def _as_scale_list(value):
    if np.isscalar(value):
        return [float(value)]
    scales = [float(item) for item in value]
    if not scales:
        raise ValueError("r_scale must contain at least one value.")
    return scales


def normalize_localization_config(config):
    """Flatten a localization preset while accepting legacy saved run configs."""

    if "data" not in config:
        normalized = dict(config)
        normalized.setdefault(
            "time_steps",
            normalized.get("train_time_steps", normalized.get("test_time_steps", 20)),
        )
        normalized.setdefault("area_size", 100.0)
        normalized.setdefault(
            "x0",
            localization_x0(
                normalized["area_size"],
                normalized["time_delta"],
                normalized["time_steps"],
            ),
        )
        if "r_scale" not in normalized:
            normalized["r_scale"] = normalized.get("measurement_noise_values", [1.0])
        normalized["r_scale"] = _as_scale_list(normalized["r_scale"])
        if "mu" not in normalized:
            legacy_q = float(normalized.get("process_noise_std", 1.0))
            reference_r = normalized["r_scale"][0]
            normalized["mu"] = (legacy_q / reference_r) ** 2 if reference_r else 0.0
        normalized.setdefault("rho", 1.0)
        normalized.setdefault("position_only_loss", False)
        normalized.setdefault("curriculum", {})
        return normalized

    system = config["system"]
    graph = config["graph"]
    data = config["data"]
    model = config["model"]
    trainer = config["trainer"]
    time_steps = int(data["time_steps"])
    area_size = float(system["area_size"])
    time_delta = float(system["time_delta"])
    r_scales = _as_scale_list(data.get("r_scale", data.get("measurement_noise_values")))
    if "mu" in data:
        mu = float(data["mu"])
    else:
        reference_r = r_scales[0]
        mu = (float(data["q"]) / reference_r) ** 2 if reference_r else 0.0
    return {
        "seed": int(config["seed"]),
        "save_root": config["save_root"],
        "state_dimension": int(model.get("signal_dim", 4)),
        "area_size": area_size,
        "x0": localization_x0(area_size, time_delta, time_steps),
        "time_delta": time_delta,
        "mu": mu,
        "rho": float(data.get("rho", 1.0)),
        "r_scale": r_scales,
        "p0_scale": float(system["p0_scale"]),
        "use_dt_mismatch": bool(system.get("use_dt_mismatch", False)),
        "dt_mismatch_values": system.get("dt_mismatch_values", [1.0]),
        "num_nodes": int(graph["node_num"]),
        "graph_seed": int(graph.get("graph_seed", config["seed"])),
        "k_neighbors": int(graph["k_neighbors"]),
        "train_models": model.get("train_models", ["dkn"]),
        "hidden_dim": int(model["hidden_dim"]),
        "learn_edge_kalman": bool(model["learn_edge_kalman"]),
        "gnn_rnn_hidden_dim": int(model.get("gnn_rnn_hidden_dim", model["hidden_dim"])),
        "gnn_rnn_learning_rate": float(
            model.get("gnn_rnn_learning_rate", model["lr"])
        ),
        "batch_size": int(data["batch_size"]),
        "learning_rate": float(model["lr"]),
        "max_epochs": int(trainer["max_epochs"]),
        "early_stop_patience": int(trainer.get("early_stop_patience", 5)),
        "early_stop_min_delta": float(trainer.get("early_stop_min_delta", 0.001)),
        "gradient_clip_val": float(trainer.get("gradient_clip_val", 1)),
        "train_sims": int(data["train_sims"]),
        "val_sims": int(data["val_sims"]),
        "time_steps": time_steps,
        "num_trials": int(data["num_trials"]),
        "preview_trajectories": int(data.get("preview_trajectories", 4)),
        "consensus_layer": model.get("consensus_layer", "none"),
        "position_only_loss": bool(model.get("position_only_loss", False)),
        "curriculum": dict(config.get("curriculum", {})),
    }


def build_localization_trainer(
    config,
    trainer_accelerator,
    *,
    checkpoint_dir,
    checkpoint_name,
):
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss_epoch",
        patience=config.get("early_stop_patience", 5),
        verbose=True,
        mode="min",
        min_delta=config.get("early_stop_min_delta", 0.001),
    )
    best_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_name,
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )
    return pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator=trainer_accelerator,
        devices=1,
        **({"precision": "bf16-mixed"} if trainer_accelerator == "gpu" else {}),
        logger=False,
        enable_checkpointing=True,
        callbacks=[early_stopping, best_checkpoint],
        gradient_clip_val=config.get("gradient_clip_val", 1),
    )


def localization_curriculum_schedule(curriculum_cfg, base_time_steps):
    """Time-step schedule for curriculum learning; [base] when disabled."""
    if not bool(curriculum_cfg.get("enabled", False)):
        return [base_time_steps]
    start = int(curriculum_cfg.get("start_time_steps", 10))
    step = int(curriculum_cfg.get("step_time_steps", 10))
    maximum = int(curriculum_cfg.get("max_time_steps", base_time_steps))
    if step <= 0:
        raise ValueError("curriculum.step_time_steps must be positive")
    if maximum < start:
        raise ValueError("curriculum.max_time_steps must be >= curriculum.start_time_steps")
    schedule = list(range(start, maximum + 1, step))
    if base_time_steps not in schedule:
        schedule.append(base_time_steps)
    return sorted(set(schedule))


def train_localization_curriculum(
    config,
    model,
    accelerator,
    *,
    checkpoint_dir,
    checkpoint_name,
    stage_loaders,
    schedule,
):
    """Fit ``model`` across the curriculum ``schedule``.

    Mirrors the non-localization curriculum in ``run_one_experiment``: each stage
    warm-starts from the *last-epoch* weights of the previous stage (the model is
    not reloaded between stages), and the returned model is the *global* best
    validation checkpoint across all stages -- which may be an earlier, shorter
    stage. Returns that global-best checkpoint path; the other stage checkpoints
    are removed."""
    curriculum_cfg = config.get("curriculum", {})
    enabled = bool(curriculum_cfg.get("enabled", False))
    epochs_per_stage = int(curriculum_cfg.get("epochs_per_stage", config["max_epochs"]))
    staged = enabled and len(schedule) > 1
    best_val = None
    best_checkpoint = None
    stage_checkpoints = []
    for stage_time_steps, (train_loader, val_loader) in zip(schedule, stage_loaders):
        stage_name = (
            f"{checkpoint_name}.ts{stage_time_steps}" if staged else checkpoint_name
        )
        stage_max_epochs = epochs_per_stage if enabled else config["max_epochs"]
        trainer = build_localization_trainer(
            {**config, "max_epochs": stage_max_epochs},
            accelerator,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=stage_name,
        )
        # No reload between stages: fit continues from the previous stage's
        # last-epoch weights, matching run_one_experiment.
        trainer.fit(model, train_loader, val_loader)
        checkpoint_cb = next(
            cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)
        )
        stage_best_val = (
            checkpoint_cb.best_model_score.item()
            if checkpoint_cb.best_model_score is not None
            else None
        )
        stage_best_path = (
            Path(checkpoint_cb.best_model_path)
            if checkpoint_cb.best_model_path
            else None
        )
        if stage_best_path is not None:
            stage_checkpoints.append(stage_best_path)
        if stage_best_val is not None and (best_val is None or stage_best_val < best_val):
            best_val = stage_best_val
            best_checkpoint = stage_best_path
    if best_checkpoint is None:
        raise RuntimeError("Training did not produce a best validation checkpoint.")
    # Load the global-best checkpoint (across all stages) into the model.
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    # Remove intermediate stage checkpoints; keep the global best (returned so
    # the caller can delete it after saving the final .pt).
    for path in stage_checkpoints:
        if path != best_checkpoint:
            path.unlink(missing_ok=True)
    return best_checkpoint


def build_gnn_rnn_model(config):
    from utils.BaselineModels import GnnRnnLightning

    return GnnRnnLightning(
        input_dim=1,
        type_num=2,
        hidden_dim=config.get("gnn_rnn_hidden_dim", config["hidden_dim"]),
        output_dim=config["state_dimension"],
        lr=config.get("gnn_rnn_learning_rate", config["learning_rate"]),
        position_only_loss=config.get("position_only_loss", False),
    )


def predict_gnn_rnn(model, graph_data, h_system):
    node_types = h_system.node_classification.to(dtype=torch.long)
    with torch.no_grad():
        return model(
            graph_data.x, node_types, graph_data.edge_index
        ).cpu().numpy().mean(axis=0)


def load_state_dict_checked(model, path):
    state_dict = torch.load(path, map_location="cpu")
    nonfinite_count = sum(
        (~torch.isfinite(value)).sum().item()
        for value in state_dict.values()
        if torch.is_tensor(value) and value.is_floating_point()
    )
    if nonfinite_count:
        print(
            f"Warning: {Path(path).name} contains {nonfinite_count} "
            "non-finite parameters; its results may be NaN."
        )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _filename_value(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def localization_model_name(config, r_scale, dt_ratio=None):
    noise = derive_localization_noise(config["mu"], config["rho"], r_scale)
    parts = [
        "localization",
        f"{int(config['num_nodes'])}n",
        f"r{_filename_value(r_scale)}",
        f"q{_filename_value(noise['q'])}",
    ]
    if dt_ratio is not None:
        parts.append(f"{_filename_value(dt_ratio)}dt")
    return "_".join(parts)


def dkn_run_name(r_scale, use_dt_mismatch=False, dt_ratio=None, default_dt_ratio=None):
    if use_dt_mismatch:
        ratio = default_dt_ratio if dt_ratio is None else dt_ratio
        return f"r={r_scale}_dtx{ratio}"
    return f"r={r_scale}"


def gnn_rnn_run_name(r_scale):
    return f"r={r_scale}"


def _dkn_subdir(use_dt_mismatch=False, dt_ratio=None, default_dt_ratio=None):
    ratio = default_dt_ratio if dt_ratio is None else dt_ratio
    if not use_dt_mismatch or ratio is None or np.isclose(float(ratio), 1.0):
        return "no_mismatch"
    return "with_missmatch"


def _experiment_config_if_exists(experiment_dir):
    try:
        return load_experiment_config(experiment_dir)
    except FileNotFoundError:
        return None


def dkn_model_path(
    experiment_dir,
    r_scale,
    use_dt_mismatch=False,
    dt_ratio=None,
    default_dt_ratio=None,
    config=None,
    for_save=False,
):
    experiment_dir = Path(experiment_dir)
    ratio = default_dt_ratio if dt_ratio is None else dt_ratio
    cfg = config if config is not None else _experiment_config_if_exists(experiment_dir)
    if cfg is not None and "mu" in cfg:
        new_path = (
            experiment_dir
            / _dkn_subdir(use_dt_mismatch, dt_ratio, default_dt_ratio)
            / f"{localization_model_name(cfg, r_scale, ratio if use_dt_mismatch else None)}.pth"
        )
        if for_save or new_path.exists():
            return new_path
    return experiment_dir / "DKN" / (
        f"{dkn_run_name(r_scale, use_dt_mismatch, dt_ratio, default_dt_ratio)}.pth"
    )


def gnn_rnn_model_path(experiment_dir, r_scale, config=None, for_save=False):
    experiment_dir = Path(experiment_dir)
    cfg = config if config is not None else _experiment_config_if_exists(experiment_dir)
    if cfg is not None and "mu" in cfg:
        new_path = (
            experiment_dir
            / "gnn-rnn"
            / f"{localization_model_name(cfg, r_scale)}.pth"
        )
        if for_save or new_path.exists():
            return new_path
    return experiment_dir / "GNN_RNN" / f"{gnn_rnn_run_name(r_scale)}.pth"


def next_experiment_dir(save_root):
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    existing = list_experiments(save_root)
    next_id = experiment_index_from_dir(existing[-1]) + 1 if existing else 1
    return save_root / f"experiment_{next_id}"


def list_experiments(save_root):
    save_root = Path(save_root)
    if not save_root.exists():
        return []
    return sorted(
        (path for path in save_root.iterdir() if path.is_dir() and path.name.startswith("experiment_")),
        key=lambda path: int(path.name.split("_", 1)[1])
        if path.name.split("_", 1)[1].isdigit()
        else 0,
    )


def experiment_index_from_dir(experiment_dir):
    suffix = Path(experiment_dir).name.removeprefix("experiment_")
    if not suffix.isdigit():
        raise ValueError(f"Cannot parse experiment index from: {experiment_dir}")
    return int(suffix)


CANONICAL_CONFIG_ORDER = (
    "seed",
    "save_root",
    "state_dimension",
    "area_size",
    "x0",
    "time_delta",
    "mu",
    "rho",
    "r_scale",
    "p0_scale",
    "use_dt_mismatch",
    "dt_mismatch_values",
    "num_nodes",
    "graph_seed",
    "k_neighbors",
    "node_positions",
    "train_models",
    "hidden_dim",
    "learn_edge_kalman",
    "consensus_layer",
    "position_only_loss",
    "batch_size",
    "learning_rate",
    "max_epochs",
    "train_sims",
    "val_sims",
    "time_steps",
    "num_trials",
    "preview_trajectories",
    "description",
)


def _to_serializable(value):
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def order_config(data, key_order=CANONICAL_CONFIG_ORDER):
    ordered = {key: data[key] for key in key_order if key in data}
    ordered.update((key, value) for key, value in data.items() if key not in ordered)
    return ordered


def _format_log_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return json.dumps(_to_serializable(value))


def _format_run_log(full_config, node_positions):
    lines = ["# Configuration"]
    for key, value in order_config(full_config).items():
        if key != "node_positions":
            lines.append(f"{key}: {_format_log_value(value)}")
    positions = _to_serializable(node_positions)
    lines.extend(["", f"# Node positions ({len(positions)})"])
    lines.extend(f"  {idx}: {json.dumps(position)}" for idx, position in enumerate(positions))
    return "\n".join(lines) + "\n"


def _parse_log_value(value):
    value = value.strip()
    if value == "null":
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_run_log(log_text):
    config = {}
    node_positions = []
    in_positions = False
    for line in log_text.splitlines():
        if not line.strip():
            continue
        if line.startswith("# Node positions"):
            in_positions = True
            continue
        if line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        if in_positions:
            node_positions.append(json.loads(value.strip()))
        else:
            config[key.strip()] = _parse_log_value(value)
    if node_positions:
        config["node_positions"] = node_positions
    return order_config(config)


def save_experiment_config(experiment_dir, full_config, node_positions):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "run.log").write_text(
        _format_run_log(full_config, node_positions), encoding="utf-8"
    )


def load_experiment_config(experiment_dir):
    path = Path(experiment_dir) / "run.log"
    if not path.exists():
        raise FileNotFoundError(f"No run log found at {path}")
    return normalize_localization_config(_parse_run_log(path.read_text(encoding="utf-8")))


def save_config(config, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_serializable(config), indent=2) + "\n", encoding="utf-8")


def load_config(config_path):
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def plot_prediction_sample(
    graph, prediction, node_positions, title, label, save_path=None
):
    x_true = graph.y[:, 0, 0].cpu().numpy()
    y_true = graph.y[:, 2, 0].cpu().numpy()
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.plot(x_true, y_true, "b-", linewidth=2, label="True trajectory")
    axis.plot(prediction[:, 0], prediction[:, 2], "m--", linewidth=2, label=label)
    axis.scatter(node_positions[:, 0], node_positions[:, 1], c="black", marker="s", s=40)
    axis.set(title=title, xlabel="x", ylabel="y")
    axis.legend()
    axis.grid(True, alpha=0.3)
    axis.set_aspect("equal")
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Sensor colors keyed by node classification: type 1 measures range (uses sigma_r),
# type 0 measures bearing/angle (uses sigma_theta). See make_r_std_vector.
RANGE_SENSOR_COLOR = "#e63946"
BEARING_SENSOR_COLOR = "#457b9d"


def sensor_color(node_type):
    return RANGE_SENSOR_COLOR if node_type == 1 else BEARING_SENSOR_COLOR


def sensor_legend_handles():
    return [
        Line2D(
            [], [], marker="s", linestyle="none", markersize=10,
            color=RANGE_SENSOR_COLOR, label="TOA sensor (range, type 1)",
        ),
        Line2D(
            [], [], marker="s", linestyle="none", markersize=10,
            color=BEARING_SENSOR_COLOR, label="DOA sensor (bearing, type 0)",
        ),
    ]


def plot_graph(
    adjacency_matrix, node_positions, node_types=None, title="Sensor Network Graph", save_path=None
):
    fig, axis = plt.subplots(figsize=(6, 6))
    graph = nx.from_numpy_array(adjacency_matrix)
    if node_types is not None:
        node_color = [sensor_color(node_types[idx]) for idx in range(len(node_positions))]
    else:
        node_color = "tomato"
    nx.draw(
        graph,
        {idx: node_positions[idx] for idx in range(len(node_positions))},
        ax=axis,
        with_labels=True,
        node_color=node_color,
        edge_color="gray",
        node_size=500,
        font_size=10,
    )
    if node_types is not None:
        axis.legend(handles=sensor_legend_handles(), loc="best")
    axis.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_and_nodes(
    node_positions, node_types, trajectory, title="Target Trajectory and Sensor Nodes"
):
    values = trajectory.detach().cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory
    plt.figure(figsize=(10, 10))
    trajectory_handle, = plt.plot(
        values[:, 0, 0], values[:, 2, 0], "b-", label="Trajectory"
    )
    for idx, position in enumerate(node_positions):
        plt.plot(*position, marker="s", markersize=12, color=sensor_color(node_types[idx]))
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(handles=[trajectory_handle, *sensor_legend_handles()])
    plt.show()


def plot_tracking_results(
    trajectory,
    x_hat_cekf,
    x_hat_dekf=None,
    x_hat_dkn=None,
    x_hat_gnn_rnn=None,
    node_positions=None,
    node_types=None,
    save_path=None,
    show=True,
):
    truth = trajectory.detach().cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory
    estimates = [("CEKF", x_hat_cekf[:, 0, 0], x_hat_cekf[:, 2, 0], "r--")]
    if x_hat_dekf is not None:
        estimates.append(
            (
                "DEKF",
                x_hat_dekf[:, :, 0, 0].mean(axis=1),
                x_hat_dekf[:, :, 2, 0].mean(axis=1),
                "g:",
            )
        )
    if x_hat_dkn is not None:
        estimates.append(("DKN", x_hat_dkn[:, 0], x_hat_dkn[:, 2], "m-."))
    if x_hat_gnn_rnn is not None:
        estimates.append(("GNN-RNN", x_hat_gnn_rnn[:, 0], x_hat_gnn_rnn[:, 2], "y--"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(truth[:, 0, 0], truth[:, 2, 0], "b-", label="True trajectory")
    for label, x_estimate, y_estimate, style in estimates:
        error = np.sqrt(
            (truth[:, 0, 0] - x_estimate) ** 2
            + (truth[:, 2, 0] - y_estimate) ** 2
        )
        axes[0].plot(x_estimate, y_estimate, style, label=label)
        axes[1].plot(error, label=f"{label} (mean: {error.mean():.4f})")
    if node_positions is not None and node_types is not None:
        for idx, position in enumerate(node_positions):
            axes[0].plot(*position, "s", color=sensor_color(node_types[idx]))
    axes[0].set(title="Trajectory Comparison", xlabel="x", ylabel="y")
    axes[1].set(title="Estimation Error", xlabel="Time step", ylabel="Position error")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    trajectory_handles, trajectory_labels = axes[0].get_legend_handles_labels()
    extra_handles = sensor_legend_handles() if node_types is not None else []
    axes[0].legend(
        handles=trajectory_handles + extra_handles,
        labels=trajectory_labels + [handle.get_label() for handle in extra_handles],
    )
    axes[1].legend()
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_learning_curve(log_dir, r_value, save_dir):
    metrics = pd.read_csv(Path(log_dir) / "metrics.csv")
    fig, axis = plt.subplots(figsize=(7, 5))
    for column, label in (
        ("train_loss_epoch", "Train Loss"),
        ("val_loss_epoch", "Validation Loss"),
    ):
        if column in metrics:
            values = metrics.dropna(subset=[column])
            axis.plot(values["epoch"], values[column], marker="o", label=label)
    axis.legend()
    axis.grid(True)
    save_path = Path(save_dir) / f"learning_curve_r={r_value}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_generated_trajectories(
    node_positions,
    node_types,
    trajectories,
    max_trajectories=4,
    title_prefix="Generated trajectories",
    save_path=None,
):
    if not trajectories:
        return
    count = min(max_trajectories, len(trajectories))
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5), squeeze=False)
    for idx in range(count):
        trajectory = trajectories[idx]
        values = trajectory.detach().cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory
        axis = axes[0, idx]
        trajectory_handle, = axis.plot(values[:, 0, 0], values[:, 2, 0], "b-", label="Trajectory")
        for node_idx, position in enumerate(node_positions):
            axis.plot(*position, "s", color=sensor_color(node_types[node_idx]))
        axis.set_title(f"{title_prefix} #{idx + 1}")
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
        axis.legend(handles=[trajectory_handle, *sensor_legend_handles()])
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_localization_experiment(config, *, root_dir: Path, description: str | None):
    config = normalize_localization_config(config)
    seed_everything(config["seed"])
    state_dimension = config["state_dimension"]
    x0 = np.asarray(config["x0"], dtype=float).reshape(state_dimension, 1)
    node_positions = (
        np.asarray(config["node_positions"], dtype=float)
        if config.get("node_positions") is not None
        else generate_node_positions(
            config["num_nodes"],
            seed=config["graph_seed"],
            area_size=config["area_size"],
        )
    )
    h_system = DistanceAngleObservation(node_positions)
    f_system_data = ConstantVelocityModel(config["time_delta"])
    adjacency = create_distance_based_graph(
        node_positions,
        k_neighbors=config["k_neighbors"],
        seed=config["graph_seed"],
    )
    train_models = normalize_localization_train_models(config)
    accelerator = get_trainer_accelerator()
    experiment_dir = next_experiment_dir(Path(root_dir) / config["save_root"])
    config["description"] = (
        description if description is not None else input("Experiment description: ").strip()
    )
    directories = [experiment_dir / "plots"]
    if "dkn" in train_models:
        directories.extend(
            [experiment_dir / "no_mismatch", experiment_dir / "with_missmatch"]
        )
    if "gnn_rnn" in train_models:
        directories.append(experiment_dir / "gnn-rnn")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    save_experiment_config(experiment_dir, config, node_positions)
    node_types = h_system.node_classification[:, 0].cpu().numpy().astype(int)
    plot_graph(
        adjacency,
        node_positions,
        node_types=node_types,
        save_path=directories[0] / "sensor_graph.png",
    )

    dt_values = (
        config.get("dt_mismatch_values", [1.0])
        if config.get("use_dt_mismatch", False)
        else [1.0]
    )
    node_types = h_system.node_classification[:, 0].cpu().numpy().astype(int)
    for r_scale in config["r_scale"]:
        noise = derive_localization_noise(config["mu"], config["rho"], r_scale)
        r_array = measurement_noise_for_nodes(
            h_system, noise["sigma_r"], noise["sigma_theta"]
        )
        dataset_kwargs = {
            "g": adjacency,
            "f_system": f_system_data,
            "h_system": h_system,
            "q": noise["q_matrix"],
            "r_array": r_array,
            "n_expansions": 1,
            "state_dim": state_dimension,
            "p0": config["p0_scale"],
        }
        # Build one train/val loader per curriculum stage (a single full-length
        # stage when curriculum is disabled). Loaders are shared across models.
        schedule = localization_curriculum_schedule(
            config.get("curriculum", {}), config["time_steps"]
        )
        stage_loaders = []
        train_dataset = val_dataset = None
        for stage_time_steps in schedule:
            stage_x0 = np.asarray(
                localization_x0(
                    config["area_size"], config["time_delta"], stage_time_steps
                ),
                dtype=float,
            ).reshape(state_dimension, 1)
            stage_kwargs = {
                **dataset_kwargs,
                "time_steps": stage_time_steps,
                "x0": stage_x0,
            }
            train_dataset = GraphDataset(
                monte_carlo_simulations=config["train_sims"],
                seed=config["seed"],
                **stage_kwargs,
            )
            val_dataset = GraphDataset(
                monte_carlo_simulations=config["val_sims"],
                seed=config["seed"] + 1,
                **stage_kwargs,
            )
            stage_loaders.append(
                (
                    DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size=config["batch_size"],
                        num_workers=0,
                    ),
                    DataLoader(
                        val_dataset,
                        shuffle=False,
                        batch_size=config["batch_size"],
                        num_workers=0,
                    ),
                )
            )
        # train_dataset / val_dataset now hold the final (full-length) stage,
        # used for the preview plot and the validation prediction samples below.
        preview = [
            train_dataset[idx].y.cpu()
            for idx in range(min(config["preview_trajectories"], len(train_dataset)))
        ]
        plot_generated_trajectories(
            node_positions,
            node_types,
            preview,
            max_trajectories=config["preview_trajectories"],
            title_prefix=f"Training trajectories (r={r_scale})",
            save_path=directories[0] / f"r={r_scale}_trajectories.png",
        )

        if "gnn_rnn" in train_models:
            path = gnn_rnn_model_path(
                experiment_dir, r_scale, config=config, for_save=True
            )
            model = build_gnn_rnn_model(config).to(torch.float32)
            best_checkpoint = train_localization_curriculum(
                config,
                model,
                accelerator,
                checkpoint_dir=path.parent,
                checkpoint_name=f"{path.stem}.best",
                stage_loaders=stage_loaders,
                schedule=schedule,
            )
            torch.save(model.state_dict(), path)
            best_checkpoint.unlink(missing_ok=True)
            sample = val_dataset[0].to(next(model.parameters()).device)
            prediction = predict_gnn_rnn(model, sample, h_system)
            plot_prediction_sample(
                sample,
                prediction,
                node_positions,
                f"GNN-RNN validation example (r={r_scale})",
                "GNN-RNN",
                directories[0] / f"gnn_rnn_{gnn_rnn_run_name(r_scale)}_prediction.png",
            )

        if "dkn" not in train_models:
            continue
        for dt_ratio in dt_values:
            path = dkn_model_path(
                experiment_dir,
                r_scale,
                use_dt_mismatch=config.get("use_dt_mismatch", False),
                dt_ratio=dt_ratio,
                config=config,
                for_save=True,
            )
            f_model = ConstantVelocityModel(config["time_delta"] * dt_ratio)
            model = build_dkn_model(config, f_model, r_array, x0).to(torch.float32)
            best_checkpoint = train_localization_curriculum(
                config,
                model,
                accelerator,
                checkpoint_dir=path.parent,
                checkpoint_name=f"{path.stem}.best",
                stage_loaders=stage_loaders,
                schedule=schedule,
            )
            torch.save(model.state_dict(), path)
            best_checkpoint.unlink(missing_ok=True)
            sample = val_dataset[0].to(next(model.parameters()).device)
            with torch.no_grad():
                prediction = model(sample)[0].mean(dim=1)[..., 0].cpu().numpy()
            run_name = dkn_run_name(
                r_scale,
                use_dt_mismatch=config.get("use_dt_mismatch", False),
                dt_ratio=dt_ratio,
            )
            plot_prediction_sample(
                sample,
                prediction,
                node_positions,
                f"DKN validation example (r={r_scale}, dt x{dt_ratio})",
                "DKN",
                directories[0] / f"{run_name}_prediction.png",
            )
