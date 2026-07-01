import numpy as np
import torch
import networkx as nx
import os
import json
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from utils.DistributedKalmanData import GraphDataset, build_bidirectional_edge_index
from utils.reproducibility import seed_everything


class ConstantVelocityModel:
    """Constant-velocity state model for [x, vx, y, vy]."""

    def __init__(self, time_delta: float):
        self.dt = time_delta
        self.state_dimension = 4
        self.state_transition_matrix = torch.tensor(
            [[1, self.dt, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, self.dt],
             [0, 0, 0, 1]], dtype=torch.float)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return (self.state_transition_matrix @ torch.tensor(x, dtype=torch.float)).numpy()
        transition_matrix = self.state_transition_matrix.to(device=x.device, dtype=x.dtype)
        return transition_matrix @ x

    def jacobian(self, x):
        F = self.state_transition_matrix.numpy()
        if x is None:
            return F[np.newaxis, ...]
        if x.ndim == 1:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        return np.tile(F[np.newaxis, ...], (batch_size, 1, 1))


class DistanceAngleObservation:
    """Distance/angle observation model for fixed sensor nodes."""

    def __init__(self, node_positions):
        self.num_nodes = len(node_positions)
        self.node_positions = np.array(node_positions, dtype=float)
        self.state_dimension = 4
        self.measurement_dimension = 1
        node_classification = np.zeros((self.num_nodes, 1))
        for i in range(self.num_nodes):
            node_classification[i, 0] = i % 2
        self.node_classification = torch.tensor(node_classification, dtype=torch.float)
        self.angle_node_mask_np = (node_classification[:, 0] == 0)
        self.angle_node_mask = torch.tensor(self.angle_node_mask_np, dtype=torch.bool)
        self.has_wrapped_angles = True

    @staticmethod
    def _wrap_to_pi(values):
        if isinstance(values, torch.Tensor):
            return torch.atan2(torch.sin(values), torch.cos(values))
        return np.arctan2(np.sin(values), np.cos(values))

    def _angle_mask_for(self, values, sensor_axis):
        axis = sensor_axis if sensor_axis >= 0 else values.ndim + sensor_axis
        if axis < 0 or axis >= values.ndim:
            raise ValueError(f"sensor_axis={sensor_axis} is out of bounds for ndim={values.ndim}")
        shape = [1] * values.ndim
        shape[axis] = self.num_nodes
        if isinstance(values, torch.Tensor):
            return self.angle_node_mask.to(device=values.device).view(shape)
        return self.angle_node_mask_np.reshape(shape)

    def wrap_innovation(self, innovation, sensor_axis):
        mask = self._angle_mask_for(innovation, sensor_axis)
        wrapped = self._wrap_to_pi(innovation)
        if isinstance(innovation, torch.Tensor):
            return torch.where(mask, wrapped, innovation)
        return np.where(mask, wrapped, innovation)

    def wrap_measurements(self, measurements, sensor_axis):
        mask = self._angle_mask_for(measurements, sensor_axis)
        wrapped = self._wrap_to_pi(measurements)
        if isinstance(measurements, torch.Tensor):
            return torch.where(mask, wrapped, measurements)
        return np.where(mask, wrapped, measurements)

    def _obs_single(self, x_pos, y_pos, node_idx):
        node_type = self.node_classification[node_idx, 0].item()
        dx = x_pos - self.node_positions[node_idx, 0]
        dy = y_pos - self.node_positions[node_idx, 1]
        if node_type == 1:
            return np.sqrt(dx ** 2 + dy ** 2)
        else:
            return np.arctan2(dy, dx)

    def _jac_single(self, x_pos, y_pos, node_idx):
        node_type = self.node_classification[node_idx, 0].item()
        dx = x_pos - self.node_positions[node_idx, 0]
        dy = y_pos - self.node_positions[node_idx, 1]
        jac = np.zeros(self.state_dimension)
        if node_type == 1:
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > 1e-10:
                jac[0], jac[2] = dx / dist, dy / dist
        else:
            denom = dx ** 2 + dy ** 2
            if denom > 1e-10:
                jac[0], jac[2] = -dy / denom, dx / denom
        return jac

    def func(self, state_vector, n_expansions=0):
        sv = state_vector.numpy() if hasattr(state_vector, 'numpy') else np.asarray(state_vector)

        if n_expansions == 0:

            if sv.ndim == 3:
                sv = sv[0]
            x_pos, y_pos = sv[0, 0], sv[2, 0]
            result = np.array([self._obs_single(x_pos, y_pos, i) for i in range(self.num_nodes)])
            return result[:, np.newaxis]
        else:

            if sv.ndim == 2:
                sv = sv[np.newaxis, ...]
            batch = sv.shape[0]
            result = np.zeros((self.num_nodes, batch, 1))
            for b in range(batch):
                x_pos, y_pos = sv[b, 0, 0], sv[b, 2, 0]
                for i in range(self.num_nodes):
                    result[i, b, 0] = self._obs_single(x_pos, y_pos, i)
            return result

    def __call__(self, state_vector, n_expansions=0):
        is_tensor = isinstance(state_vector, torch.Tensor)
        if is_tensor:
            sv = state_vector
            device = state_vector.device
            dtype = state_vector.dtype
            node_positions = torch.as_tensor(self.node_positions, dtype=dtype, device=device)
            node_classification = self.node_classification.to(device=device, dtype=dtype)
        else:
            sv = np.asarray(state_vector)


        if sv.ndim == 2 and sv.shape[1] > 1:
            if is_tensor:
                x_pos = sv[0, :]
                y_pos = sv[2, :]
                dx = x_pos.unsqueeze(0) - node_positions[:, 0].unsqueeze(1)
                dy = y_pos.unsqueeze(0) - node_positions[:, 1].unsqueeze(1)
                distance = torch.sqrt(dx ** 2 + dy ** 2)
                angle = torch.atan2(dy, dx)
                result = node_classification * distance + (1 - node_classification) * angle
                return result.unsqueeze(1)
            N = sv.shape[1]
            x_pos = sv[0, :]
            y_pos = sv[2, :]
            result = np.zeros((self.num_nodes, 1, N))
            for i in range(self.num_nodes):
                dx = x_pos - self.node_positions[i, 0]
                dy = y_pos - self.node_positions[i, 1]
                if self.node_classification[i, 0].item() == 1:
                    result[i, 0, :] = np.sqrt(dx ** 2 + dy ** 2)
                else:
                    result[i, 0, :] = np.arctan2(dy, dx)
            return result



        if sv.ndim == 4:
            if is_tensor:
                x_pos = sv[:, :, 0, 0].unsqueeze(-1)
                y_pos = sv[:, :, 2, 0].unsqueeze(-1)
                pos_x = node_positions[:, 0].view(1, 1, -1)
                pos_y = node_positions[:, 1].view(1, 1, -1)
                node_types = node_classification[:, 0].view(1, 1, -1)
                dx = x_pos - pos_x
                dy = y_pos - pos_y
                distance = torch.sqrt(dx ** 2 + dy ** 2)
                angle = torch.atan2(dy, dx)
                result = node_types * distance + (1 - node_types) * angle
                return result.unsqueeze(-1)
            batch_size = sv.shape[0]
            num_nodes_in = sv.shape[1]
            result = np.zeros((batch_size, num_nodes_in, self.num_nodes, 1))
            for b in range(batch_size):
                for src in range(num_nodes_in):
                    x_pos = sv[b, src, 0, 0]
                    y_pos = sv[b, src, 2, 0]
                    for sensor in range(self.num_nodes):
                        result[b, src, sensor, 0] = self._obs_single(x_pos, y_pos, sensor)
            return result


        result = self.func(sv, n_expansions)
        if is_tensor:
            return torch.as_tensor(result, dtype=dtype, device=device)
        return result

    def jacobian(self, state_vector, n_expansions=0):
        is_tensor = isinstance(state_vector, torch.Tensor)
        if is_tensor:
            sv = state_vector
            device = state_vector.device
            dtype = state_vector.dtype
            node_positions = torch.as_tensor(self.node_positions, dtype=dtype, device=device)
            node_classification = self.node_classification.to(device=device, dtype=dtype)[:, 0].unsqueeze(0)
        else:
            sv = np.asarray(state_vector)

        if sv.ndim == 2:
            sv = sv.unsqueeze(0) if is_tensor else sv[np.newaxis, ...]
        batch = sv.shape[0]

        if is_tensor:
            x_pos = sv[:, 0, 0].unsqueeze(1)
            y_pos = sv[:, 2, 0].unsqueeze(1)
            dx = x_pos - node_positions[:, 0].unsqueeze(0)
            dy = y_pos - node_positions[:, 1].unsqueeze(0)
            dist = torch.sqrt(dx ** 2 + dy ** 2)
            denom = dx ** 2 + dy ** 2

            jacs = torch.zeros((batch, self.num_nodes, self.state_dimension), dtype=dtype, device=device)
            distance_dx = torch.where(dist > 1e-10, dx / dist, torch.zeros_like(dx))
            distance_dy = torch.where(dist > 1e-10, dy / dist, torch.zeros_like(dy))
            angle_dx = torch.where(denom > 1e-10, -dy / denom, torch.zeros_like(dx))
            angle_dy = torch.where(denom > 1e-10, dx / denom, torch.zeros_like(dy))

            jacs[:, :, 0] = node_classification * distance_dx + (1 - node_classification) * angle_dx
            jacs[:, :, 2] = node_classification * distance_dy + (1 - node_classification) * angle_dy

            if n_expansions > 0:
                return jacs[:, None, :, :, None]
            return jacs.transpose(0, 1)[:, :, :, None]

        jacs = np.zeros((batch, self.num_nodes, self.state_dimension))
        for b in range(batch):
            x_pos, y_pos = sv[b, 0, 0], sv[b, 2, 0]
            for i in range(self.num_nodes):
                jacs[b, i, :] = self._jac_single(x_pos, y_pos, i)

        if n_expansions > 0:
            return jacs[:, np.newaxis, :, :, np.newaxis]
        return jacs.transpose(1, 0, 2)[:, :, :, np.newaxis]


def generate_node_positions(num_nodes, seed=None, area_size=100.0):
    """Generate random 2-D node positions from a seed for reproducibility."""
    rng = np.random.default_rng(seed)
    return rng.random((num_nodes, 2)) * area_size


def create_distance_based_graph(node_positions, k_neighbors=3, seed=None):
    num_nodes = len(node_positions)
    rng = np.random.default_rng(seed)

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))


    for i in range(num_nodes):
        dists = np.linalg.norm(node_positions - node_positions[i], axis=1)
        dists[i] = np.inf
        local_k = max(2, k_neighbors + rng.integers(-1, 2))
        neighbor_idx = np.argsort(dists)[:local_k]
        for j in neighbor_idx:
            g.add_edge(i, j)


    if not nx.is_connected(g):
        for cc in list(nx.connected_components(g))[1:]:
            min_i, min_j, min_dist = None, None, np.inf
            for u in list(cc):
                for v in range(num_nodes):
                    if v not in cc:
                        d = np.linalg.norm(node_positions[u] - node_positions[v])
                        if d < min_dist:
                            min_dist = d
                            min_i, min_j = u, v
            g.add_edge(min_i, min_j)

    g.add_edges_from((i, i) for i in range(num_nodes))
    return nx.to_numpy_array(g)


def get_trainer_accelerator():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def generate_trajectory(f_system, initial_state, num_time_steps, process_noise_std):
    state_dim = initial_state.shape[0]
    trajectory = torch.zeros((num_time_steps, state_dim, 1), dtype=torch.float32)
    state_noise = torch.randn(num_time_steps, state_dim, 1) * process_noise_std

    x_current = torch.tensor(initial_state, dtype=torch.float32)
    for k in range(num_time_steps):
        x_next = f_system(x_current) + state_noise[k]
        trajectory[k] = x_next
        x_current = x_next
    return trajectory


def generate_measurements(h_system, trajectory, measurement_noise_std):
    num_time_steps = trajectory.shape[0]
    num_nodes = h_system.num_nodes

    measurements = np.zeros((num_nodes, 1, num_time_steps))
    observation_noise = np.random.randn(num_nodes, num_time_steps) * measurement_noise_std

    for k in range(num_time_steps):
        x_k = trajectory[k].numpy() if isinstance(trajectory[k], torch.Tensor) else trajectory[k]
        obs = h_system.func(x_k)
        measurements[:, 0, k] = obs[:, 0] + observation_noise[:, k]

    if hasattr(h_system, "wrap_measurements"):
        measurements = h_system.wrap_measurements(measurements, sensor_axis=0)

    return measurements


def build_graph_data_for_dkn(adjacency_matrix, h_system, trajectory, measurements):
    adjacency_with_self = np.array(adjacency_matrix, dtype=float, copy=True)
    np.fill_diagonal(adjacency_with_self, 1.0)

    graph = nx.from_numpy_array(adjacency_with_self)
    edge_index = build_bidirectional_edge_index(graph)
    measurement_tensor = torch.tensor(measurements.transpose(0, 2, 1), dtype=torch.float32)
    trajectory_tensor = torch.tensor(trajectory, dtype=torch.float32)

    return Data(
        x=measurement_tensor,
        edge_index=edge_index,
        y=trajectory_tensor,
        adj_matrix=torch.tensor(adjacency_with_self, dtype=torch.float32),
        h_system=h_system,
    )


def format_dt_ratio(ratio):
    ratio_float = float(ratio)
    if np.isclose(ratio_float, round(ratio_float)):
        return f"{ratio_float:.1f}"
    return f"{ratio_float:g}"


def nearest_nominal_dt_ratio(dt_values):
    ratio = min(dt_values, key=lambda value: abs(float(value) - 1.0))
    if not np.isclose(float(ratio), 1.0):
        raise RuntimeError(f"Need a no-mismatch dt ratio near 1.0; got {dt_values}.")
    return ratio


def farthest_mismatch_dt_ratio(dt_values, nominal_ratio):
    candidates = [ratio for ratio in dt_values if not np.isclose(float(ratio), float(nominal_ratio))]
    if not candidates:
        raise RuntimeError(f"Need at least one dt mismatch ratio besides {nominal_ratio}; got {dt_values}.")
    return max(candidates, key=lambda ratio: abs(float(ratio) - 1.0))


def _filename_value(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def localization_model_name(config, r_noise, dt_ratio=None):
    parts = [
        "localization",
        f"{int(config['num_nodes'])}n",
        f"r{_filename_value(r_noise)}",
        f"q{_filename_value(config['process_noise_std'])}",
    ]
    if dt_ratio is not None:
        parts.append(f"{_filename_value(dt_ratio)}dt")
    return "_".join(parts)


def dkn_run_name(r_noise, use_dt_mismatch=False, dt_ratio=None, default_dt_ratio=None):
    if use_dt_mismatch:
        ratio = default_dt_ratio if dt_ratio is None else dt_ratio
        return f"r={r_noise}_dtx{ratio}"
    return f"r={r_noise}"


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


def dkn_model_path(experiment_dir, r_noise, use_dt_mismatch=False, dt_ratio=None, default_dt_ratio=None, config=None, for_save=False):
    experiment_dir = Path(experiment_dir)
    ratio = default_dt_ratio if dt_ratio is None else dt_ratio
    cfg = config if config is not None else _experiment_config_if_exists(experiment_dir)
    if cfg is not None:
        new_path = (
            experiment_dir
            / _dkn_subdir(use_dt_mismatch=use_dt_mismatch, dt_ratio=dt_ratio, default_dt_ratio=default_dt_ratio)
            / f"{localization_model_name(cfg, r_noise, ratio if use_dt_mismatch else None)}.pth"
        )
        if for_save or new_path.exists():
            return new_path

    legacy_run_name = dkn_run_name(
        r_noise,
        use_dt_mismatch=use_dt_mismatch,
        dt_ratio=dt_ratio,
        default_dt_ratio=default_dt_ratio,
    )
    return experiment_dir / "DKN" / f"{legacy_run_name}.pth"


def gnn_rnn_run_name(r_noise):
    return f"r={r_noise}"


def gnn_rnn_model_path(experiment_dir, r_noise, config=None, for_save=False):
    experiment_dir = Path(experiment_dir)
    cfg = config if config is not None else _experiment_config_if_exists(experiment_dir)
    if cfg is not None:
        new_path = experiment_dir / "gnn-rnn" / f"{localization_model_name(cfg, r_noise)}.pth"
        if for_save or new_path.exists():
            return new_path
    return experiment_dir / "GNN_RNN" / f"{gnn_rnn_run_name(r_noise)}.pth"


def load_state_dict_checked(model, path):
    state_dict = torch.load(path, map_location="cpu")
    nonfinite_count = sum(
        (~torch.isfinite(value)).sum().item()
        for value in state_dict.values()
        if torch.is_tensor(value) and value.is_floating_point()
    )
    if nonfinite_count:
        print(f"Warning: {Path(path).name} contains {nonfinite_count} non-finite parameters; its results may be NaN.")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_dkn_model(config, f_model, r_noise, x0):
    from utils.DistributedKalmanNet import GraphKalmanProcess

    state_dimension = int(config["state_dimension"])
    return GraphKalmanProcess(
        f_model,
        signal_dim=state_dimension,
        edge_features_dim=1,
        node_kalman_dim=state_dimension ** 2,
        edge_kalman_dim=2,
        hidden_dim=config["hidden_dim"],
        lr=config["learning_rate"],
        r_array=r_noise,
        learn_edge_kalman=config["learn_edge_kalman"],
        x0_scale=x0,
        consensus_layer=config.get("consensus_layer", "none"),
    )


def build_gnn_rnn_model(config):
    from utils.BaselineModels import GnnRnnLightning

    return GnnRnnLightning(
        input_dim=1,
        type_num=2,
        hidden_dim=config.get("gnn_rnn_hidden_dim", config["hidden_dim"]),
        output_dim=config["state_dimension"],
        lr=config.get("gnn_rnn_learning_rate", config["learning_rate"]),
    )


def predict_gnn_rnn(model, graph_data, h_system):
    node_types = h_system.node_classification.to(dtype=torch.long)
    with torch.no_grad():
        return model(graph_data.x, node_types, graph_data.edge_index).cpu().numpy().mean(axis=0)


def clean_measurements_for_trajectory(h_system, trajectory):
    clean = np.zeros((h_system.num_nodes, 1, trajectory.shape[0]))
    for k in range(trajectory.shape[0]):
        x_k = trajectory[k].numpy() if isinstance(trajectory[k], torch.Tensor) else trajectory[k]
        clean[:, 0, k] = h_system.func(x_k)[:, 0]
    if hasattr(h_system, "wrap_measurements"):
        clean = h_system.wrap_measurements(clean, sensor_axis=0)
    return clean


def measurement_snr_db(h_system, trajectory, measurements, eps=1e-12):
    clean = clean_measurements_for_trajectory(h_system, trajectory)
    noise = measurements - clean
    if hasattr(h_system, "wrap_innovation"):
        noise = h_system.wrap_innovation(noise, sensor_axis=0)
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10((signal_power + eps) / (noise_power + eps))


def position_error_from_state_sequence(trajectory, estimate):
    x_true = trajectory[:, 0, 0].numpy()
    y_true = trajectory[:, 2, 0].numpy()
    return np.sqrt((x_true - estimate[:, 0]) ** 2 + (y_true - estimate[:, 2]) ** 2).mean()


def position_error_from_dekf(trajectory, estimate):
    x_true = trajectory[:, 0, 0].numpy()
    y_true = trajectory[:, 2, 0].numpy()
    x_est = estimate[:, :, 0, 0].mean(axis=1)
    y_est = estimate[:, :, 2, 0].mean(axis=1)
    return np.sqrt((x_true - x_est) ** 2 + (y_true - y_est) ** 2).mean()


def generate_trial_data(f_system, h_system, x0, time_steps, process_noise_std, r_noise):
    trajectory = generate_trajectory(f_system, x0, time_steps, process_noise_std)
    measurements = generate_measurements(h_system, trajectory, r_noise)
    return trajectory, measurements


def experiment_time_steps(config):
    time_steps = config.get("time_steps")
    if time_steps is None:
        time_steps = config.get("test_time_steps")
    if time_steps is None:
        time_steps = config.get("train_time_steps")
    if time_steps is None:
        raise KeyError("Experiment config must include time_steps.")
    return time_steps


def sync_torch_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def normalize_localization_train_models(config_val):
    aliases = {
        "dkn": "dkn",
        "gnn_rnn": "gnn_rnn",
        "gnn-rnn": "gnn_rnn",
        "gnnrnn": "gnn_rnn",
    }
    raw_models = config_val.get("train_models", ["dkn"])
    if isinstance(raw_models, str):
        raw_models = [raw_models]
    if any(str(model).lower() == "all" for model in raw_models):
        return {"dkn", "gnn_rnn"}

    train_models = set()
    for model in raw_models:
        key = str(model).lower()
        if key not in aliases:
            raise ValueError(f"Unknown train model '{model}'. Expected one of: {sorted(aliases)} or 'all'.")
        train_models.add(aliases[key])
    return train_models


def build_localization_trainer(config_val, trainer_accelerator):
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss_epoch",
        patience=5,
        verbose=True,
        mode="min",
        min_delta=0.001,
    )
    return pl.Trainer(
        max_epochs=config_val["max_epochs"],
        accelerator=trainer_accelerator,
        devices=1,
        **({"precision": "bf16-mixed"} if trainer_accelerator == "gpu" else {}),
        logger=False,
        enable_checkpointing=False,
        callbacks=[early_stopping],
        gradient_clip_val=1,
    )


def plot_prediction_sample(graph, prediction, node_positions, title, label, save_path=None):
    x_true = graph.y[:, 0, 0].cpu().numpy()
    y_true = graph.y[:, 2, 0].cpu().numpy()
    x_pred = prediction[:, 0]
    y_pred = prediction[:, 2]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_true, y_true, "b-", linewidth=2, label="True trajectory")
    ax.plot(x_pred, y_pred, "m--", linewidth=2, label=f"{label} prediction")
    ax.scatter(node_positions[:, 0], node_positions[:, 1], c="black", marker="s", s=40, label="Nodes")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved prediction plot: {save_path}")
    plt.close(fig)


def localization_x0(area_size, time_delta, time_steps):
    speed = float(area_size) / (float(time_delta) * int(time_steps))
    return [0.0, speed, 0.0, speed]


def normalize_localization_config(config_val):
    if "data" not in config_val:
        cfg = dict(config_val)
        cfg.setdefault("time_steps", cfg.get("train_time_steps", cfg.get("test_time_steps", 20)))
        cfg.setdefault("area_size", 100.0)
        cfg.setdefault("x0", localization_x0(cfg["area_size"], cfg["time_delta"], cfg["time_steps"]))
        return cfg

    cfg = dict(config_val)
    system = cfg["system"]
    graph = cfg["graph"]
    data = cfg["data"]
    model = cfg["model"]
    trainer = cfg["trainer"]
    time_steps = data["time_steps"]
    area_size = system["area_size"]
    time_delta = system["time_delta"]

    return {
        "seed": cfg["seed"],
        "save_root": cfg["save_root"],
        "state_dimension": model.get("signal_dim", 4),
        "area_size": area_size,
        "x0": localization_x0(area_size, time_delta, time_steps),
        "time_delta": time_delta,
        "process_noise_std": data["q"],
        "p0_scale": system["p0_scale"],
        "measurement_noise_values": data["measurement_noise_values"],
        "use_dt_mismatch": system.get("use_dt_mismatch", False),
        "dt_mismatch_values": system.get("dt_mismatch_values", [1.0]),
        "num_nodes": graph["node_num"],
        "graph_seed": graph.get("graph_seed", cfg["seed"]),
        "k_neighbors": graph["k_neighbors"],
        "train_models": model.get("train_models", ["dkn"]),
        "hidden_dim": model["hidden_dim"],
        "learn_edge_kalman": model["learn_edge_kalman"],
        "gnn_rnn_hidden_dim": model.get("gnn_rnn_hidden_dim", model["hidden_dim"]),
        "gnn_rnn_learning_rate": model.get("gnn_rnn_learning_rate", model["lr"]),
        "batch_size": data["batch_size"],
        "learning_rate": model["lr"],
        "max_epochs": trainer["max_epochs"],
        "train_sims": data["train_sims"],
        "val_sims": data["val_sims"],
        "time_steps": time_steps,
        "num_trials": data["num_trials"],
        "preview_trajectories": data.get("preview_trajectories", 4),
        "consensus_layer": model.get("consensus_layer", "none"),
    }


def run_localization_experiment(config_val: dict, *, root_dir: Path, description: str | None) -> None:
    config_val = normalize_localization_config(config_val)
    seed_everything(config_val["seed"])

    num_nodes = config_val["num_nodes"]
    state_dimension = config_val["state_dimension"]
    time_steps = config_val["time_steps"]
    time_delta = config_val["time_delta"]
    process_noise_std = config_val["process_noise_std"]
    measurement_noise_values = config_val["measurement_noise_values"]
    x0 = np.array(config_val["x0"], dtype=float).reshape(state_dimension, 1)
    if config_val.get("node_positions") is not None:
        node_positions = np.array(config_val["node_positions"], dtype=float)
    else:
        node_positions = generate_node_positions(num_nodes, seed=config_val["graph_seed"], area_size=config_val["area_size"])

    trainer_accelerator = get_trainer_accelerator()
    use_dt_mismatch = config_val.get("use_dt_mismatch", False)
    train_models = normalize_localization_train_models(config_val)
    save_root = root_dir / config_val["save_root"]
    experiment_dir = next_experiment_dir(save_root)
    config_val["description"] = description if description is not None else input("Experiment description: ").strip()

    print("Loaded config from: experiments/graphkalmanprocess_hparams.py::LOCALIZATION_BASELINE")
    print(json.dumps(config_val, indent=2))
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Trainer accelerator: {trainer_accelerator}")
    print(f"Training models: {', '.join(sorted(train_models))}")
    print(f"Initial state vector x0:\n{x0}")

    f_system_data = ConstantVelocityModel(time_delta)
    h_system = DistanceAngleObservation(node_positions)
    adjacency_matrix = create_distance_based_graph(
        node_positions,
        k_neighbors=config_val["k_neighbors"],
        seed=config_val["graph_seed"],
    )
    node_types = h_system.node_classification[:, 0].cpu().numpy().astype(int)
    dt_mismatch_values = config_val.get("dt_mismatch_values", [1.0]) if use_dt_mismatch else [1.0]

    plot_dir = experiment_dir / "plots"
    model_paths = [plot_dir]
    if "dkn" in train_models:
        model_paths.extend([experiment_dir / "no_mismatch", experiment_dir / "with_missmatch"])
    if "gnn_rnn" in train_models:
        model_paths.append(experiment_dir / "gnn-rnn")
    for path in model_paths:
        path.mkdir(parents=True, exist_ok=True)

    save_experiment_config(experiment_dir, full_config=config_val, node_positions=node_positions)
    plot_graph(adjacency_matrix, node_positions, save_path=plot_dir / "sensor_graph.png")

    print("Training node positions:")
    print(node_positions)
    if use_dt_mismatch:
        print(f"dt mismatch enabled; model dt multipliers: {dt_mismatch_values}")

    for r_noise in measurement_noise_values:
        r_array = r_noise * np.ones(num_nodes)
        train_dataset = GraphDataset(
            adjacency_matrix,
            f_system_data,
            h_system,
            process_noise_std,
            r_array,
            monte_carlo_simulations=config_val["train_sims"],
            time_steps=time_steps,
            n_expansions=1,
            x0=x0,
            state_dim=state_dimension,
        )
        val_dataset = GraphDataset(
            adjacency_matrix,
            f_system_data,
            h_system,
            process_noise_std,
            r_array,
            monte_carlo_simulations=config_val["val_sims"],
            time_steps=time_steps,
            n_expansions=1,
            x0=x0,
            state_dim=state_dimension,
        )

        n_preview = config_val.get("preview_trajectories", 4)
        sample_trajectories = [
            train_dataset[idx].y.cpu()
            for idx in range(min(n_preview, len(train_dataset)))
        ]
        plot_generated_trajectories(
            node_positions,
            node_types,
            sample_trajectories,
            max_trajectories=n_preview,
            title_prefix=f"Training trajectories (r={r_noise})",
            save_path=plot_dir / f"r={r_noise}_trajectories.png",
        )

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=config_val["batch_size"],
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=config_val["batch_size"],
            num_workers=0,
            pin_memory=False,
        )

        if "gnn_rnn" in train_models:
            print(f"\nTraining GNN-RNN baseline: r={r_noise}")
            gnn_rnn_process = build_gnn_rnn_model(config_val).to(torch.float32)

            trainer = build_localization_trainer(config_val, trainer_accelerator)
            trainer.fit(gnn_rnn_process, train_loader, val_loader)

            model_path = gnn_rnn_model_path(experiment_dir, r_noise, config=config_val, for_save=True)
            torch.save(gnn_rnn_process.state_dict(), model_path)
            print(f"Saved model: {model_path}")

            sample_device = next(gnn_rnn_process.parameters()).device
            sample_graph = val_dataset[0].to(sample_device)
            with torch.no_grad():
                sample_prediction = predict_gnn_rnn(gnn_rnn_process, sample_graph, h_system)
            plot_prediction_sample(
                sample_graph,
                sample_prediction,
                node_positions,
                title=f"GNN-RNN validation example (r={r_noise})",
                label="GNN-RNN",
                save_path=plot_dir / f"gnn_rnn_{gnn_rnn_run_name(r_noise)}_prediction.png",
            )

        if "dkn" not in train_models:
            continue

        for dt_mismatch in dt_mismatch_values:
            print(f"\nTraining DKN model: r={r_noise}" + (f", dt_mismatch={dt_mismatch}" if use_dt_mismatch else ""))
            f_system_model = ConstantVelocityModel(time_delta * dt_mismatch)
            kalman_process = build_dkn_model(config_val, f_system_model, r_noise, x0).to(torch.float32)

            trainer = build_localization_trainer(config_val, trainer_accelerator)
            trainer.fit(kalman_process, train_loader, val_loader)

            model_path = dkn_model_path(
                experiment_dir,
                r_noise,
                use_dt_mismatch=use_dt_mismatch,
                dt_ratio=dt_mismatch,
                config=config_val,
                for_save=True,
            )
            torch.save(kalman_process.state_dict(), model_path)
            print(f"Saved model: {model_path}")

            sample_device = next(kalman_process.parameters()).device
            sample_graph = val_dataset[0].to(sample_device)
            with torch.no_grad():
                sample_prediction = kalman_process(sample_graph).squeeze().cpu().numpy().mean(axis=1)
            run_name = dkn_run_name(r_noise, use_dt_mismatch=use_dt_mismatch, dt_ratio=dt_mismatch)
            title = f"DKN validation example (r={r_noise}" + (f", dt x{dt_mismatch})" if use_dt_mismatch else ")")
            plot_prediction_sample(
                sample_graph,
                sample_prediction,
                node_positions,
                title=title,
                label="DKN",
                save_path=plot_dir / f"{run_name}_prediction.png",
            )


def plot_graph(adjacency_matrix, node_positions, title="Sensor Network Graph", save_path=None):
    """Visualize the sensor graph."""
    plt.figure(figsize=(6, 6))
    graph = nx.from_numpy_array(adjacency_matrix)
    pos = {i: node_positions[i] for i in range(len(node_positions))}
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="tomato",
        edge_color="gray",
        node_size=500,
        font_size=10,
    )
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved graph plot: {save_path}")
    plt.show()
    plt.close()


def plot_trajectory_and_nodes(node_positions, node_types, trajectory, title="Target Trajectory and Sensor Nodes"):
    """Plot the target trajectory and color sensor nodes by measurement type."""
    plt.figure(figsize=(10, 10))

    x_vals = trajectory[:, 0, 0].numpy() if isinstance(trajectory, torch.Tensor) else trajectory[:, 0, 0]
    y_vals = trajectory[:, 2, 0].numpy() if isinstance(trajectory, torch.Tensor) else trajectory[:, 2, 0]
    plt.plot(x_vals, y_vals, "b-", linewidth=2, marker="o", markersize=2, label="Trajectory", alpha=0.7)

    for i in range(len(node_positions)):
        x_pos, y_pos = node_positions[i]
        node_type = node_types[i]
        color = "#e63946" if node_type == 1 else "#457b9d"
        label = "Distance nodes" if node_type == 1 else "Angle nodes"
        plt.plot(x_pos, y_pos, marker="s", markersize=12, color=color, label=label if i < 2 else None)
        plt.text(x_pos + 1, y_pos + 1, f"{i}", fontsize=10)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
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
    """Plot trajectory estimates and position errors for CEKF, DEKF, and DKN."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x_true = trajectory[:, 0, 0].numpy() if isinstance(trajectory, torch.Tensor) else trajectory[:, 0, 0]
    y_true = trajectory[:, 2, 0].numpy() if isinstance(trajectory, torch.Tensor) else trajectory[:, 2, 0]

    x_cekf = x_hat_cekf[:, 0, 0]
    y_cekf = x_hat_cekf[:, 2, 0]

    ax1 = axes[0]
    ax1.plot(x_true, y_true, "b-", linewidth=2, label="True trajectory", alpha=0.7)
    ax1.plot(x_cekf, y_cekf, "r--", linewidth=2, label="CEKF estimate", alpha=0.7)

    x_dekf = y_dekf = None
    if x_hat_dekf is not None:
        x_dekf = x_hat_dekf[:, :, 0, 0].mean(axis=1)
        y_dekf = x_hat_dekf[:, :, 2, 0].mean(axis=1)
        ax1.plot(x_dekf, y_dekf, "g:", linewidth=2, label="DEKF estimate (avg)", alpha=0.8)

    x_dkn = y_dkn = None
    if x_hat_dkn is not None:
        x_dkn = x_hat_dkn[:, 0]
        y_dkn = x_hat_dkn[:, 2]
        ax1.plot(x_dkn, y_dkn, color="#9b5de5", linewidth=2, linestyle="-.", label="DKN estimate", alpha=0.8)

    x_gnn_rnn = y_gnn_rnn = None
    if x_hat_gnn_rnn is not None:
        x_gnn_rnn = x_hat_gnn_rnn[:, 0]
        y_gnn_rnn = x_hat_gnn_rnn[:, 2]
        ax1.plot(x_gnn_rnn, y_gnn_rnn, color="#f4a261", linewidth=2, linestyle="--", label="GNN-RNN estimate", alpha=0.8)

    if node_positions is not None and node_types is not None:
        for i in range(len(node_positions)):
            color = "#e63946" if node_types[i] == 1 else "#457b9d"
            ax1.plot(node_positions[i, 0], node_positions[i, 1], "s", markersize=8, color=color)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Trajectory Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    ax2 = axes[1]
    error_cekf = np.sqrt((x_true - x_cekf) ** 2 + (y_true - y_cekf) ** 2)
    ax2.plot(error_cekf, "r-", linewidth=1, label=f"CEKF (mean: {error_cekf.mean():.4f})")

    if x_dekf is not None and y_dekf is not None:
        error_dekf = np.sqrt((x_true - x_dekf) ** 2 + (y_true - y_dekf) ** 2)
        ax2.plot(error_dekf, "g-", linewidth=1, label=f"DEKF (mean: {error_dekf.mean():.4f})")

    if x_dkn is not None and y_dkn is not None:
        error_dkn = np.sqrt((x_true - x_dkn) ** 2 + (y_true - y_dkn) ** 2)
        ax2.plot(error_dkn, color="#9b5de5", linewidth=1, label=f"DKN (mean: {error_dkn.mean():.4f})")

    if x_gnn_rnn is not None and y_gnn_rnn is not None:
        error_gnn_rnn = np.sqrt((x_true - x_gnn_rnn) ** 2 + (y_true - y_gnn_rnn) ** 2)
        ax2.plot(error_gnn_rnn, color="#f4a261", linewidth=1, label=f"GNN-RNN (mean: {error_gnn_rnn.mean():.4f})")

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Position error")
    ax2.set_title("Estimation Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        save_path_str = os.fspath(save_path)
        save_dir = os.path.dirname(save_path_str)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path_str, dpi=200, bbox_inches="tight")
        print(f"Saved tracking plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_learning_curve(log_dir, r_value, save_dir):
    """Plot epoch-level train and validation loss from Lightning CSV logs."""
    metrics_path = os.path.join(log_dir, "metrics.csv")
    df = pd.read_csv(metrics_path)

    plt.figure(figsize=(7, 5))
    if "train_loss_epoch" in df.columns:
        train_df = df.dropna(subset=["train_loss_epoch"])
        plt.plot(train_df["epoch"], train_df["train_loss_epoch"], marker="o", label="Train Loss")
    if "val_loss_epoch" in df.columns:
        val_df = df.dropna(subset=["val_loss_epoch"])
        plt.plot(val_df["epoch"], val_df["val_loss_epoch"], marker="o", label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss per Epoch (r = {r_value})")
    plt.grid(True)
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"learning_curve_r={r_value}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved learning curve: {save_path}")


def _to_serializable(value):
    """Recursively convert config values into JSON-serializable objects."""
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value







_PRIMITIVE_ARRAY_RE = re.compile(
    r"\[\s*\n"
    r"(?:\s*[^][{}\n]+\s*,\s*\n)*"
    r"\s*[^][{}\n]+\s*\n"
    r"\s*\]"
)


def _split_top_level_commas(text):
    """Split a comma-separated JSON value list, respecting quoted strings."""
    parts = []
    buf = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if in_string and ch == "\\":
            buf.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            buf.append(ch)
            continue
        if ch == "," and not in_string:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _compact_scalar_arrays(json_text, max_inline_length=120):
    """Collapse JSON arrays of primitives onto a single line."""

    def collapse(match):
        text = match.group(0)
        parts = _split_top_level_commas(text[1:-1])
        compact = "[" + ", ".join(parts) + "]"
        return compact if len(compact) <= max_inline_length else text

    previous = None
    current = json_text
    while previous != current:
        previous = current
        current = _PRIMITIVE_ARRAY_RE.sub(collapse, current)
    return current


def save_config(config_val, save_path):
    """Save a configuration dictionary as a JSON file, preserving key order
    and inlining short arrays of primitives for readability."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json_text = json.dumps(_to_serializable(config_val), indent=2, sort_keys=False)
    json_text = _compact_scalar_arrays(json_text)
    with open(save_path, "w", encoding="utf-8") as fp:
        fp.write(json_text)
        fp.write("\n")


def load_config(config_path):
    """Load a JSON configuration dictionary from disk."""
    with open(config_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def next_experiment_dir(save_root):
    """Return ``save_root / experiment_N`` with the next available integer N."""
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        (d for d in save_root.iterdir() if d.is_dir() and d.name.startswith("experiment_")),
        key=lambda d: int(d.name.split("_", 1)[1]) if d.name.split("_", 1)[1].isdigit() else -1,
    )
    next_id = int(existing[-1].name.split("_", 1)[1]) + 1 if existing else 1
    return save_root / f"experiment_{next_id}"


def list_experiments(save_root):
    """Return a sorted list of ``experiment_*`` directories under *save_root*."""
    save_root = Path(save_root)
    if not save_root.exists():
        return []
    return sorted(
        (d for d in save_root.iterdir() if d.is_dir() and d.name.startswith("experiment_")),
        key=lambda d: int(d.name.split("_", 1)[1]) if d.name.split("_", 1)[1].isdigit() else 0,
    )







CANONICAL_CONFIG_ORDER = (
    "seed",
    "save_root",
    "state_dimension",
    "area_size",
    "x0",
    "time_delta",
    "process_noise_std",
    "p0_scale",
    "measurement_noise_values",
    "use_dt_mismatch",
    "dt_mismatch_values",
    "num_nodes",
    "graph_seed",
    "k_neighbors",
    "node_positions",
    "hidden_dim",
    "learn_edge_kalman",
    "batch_size",
    "learning_rate",
    "max_epochs",
    "train_sims",
    "val_sims",
    "time_steps",
    "train_time_steps",
    "test_time_steps",
    "num_trials",
    "preview_trajectories",
    "description",
    "run_metadata",
)





def experiment_index_from_dir(experiment_dir):
    """Parse the trailing integer ``n`` from an ``experiment_<n>`` directory name."""
    name = Path(experiment_dir).name
    if not name.startswith("experiment_"):
        raise ValueError(f"Not an experiment directory: {experiment_dir}")
    suffix = name.split("_", 1)[1]
    if not suffix.isdigit():
        raise ValueError(f"Cannot parse experiment index from: {name}")
    return int(suffix)


def order_config(data, key_order=CANONICAL_CONFIG_ORDER):
    """Return a new dict with ``data``'s keys reordered per ``key_order``.

    Keys present in ``data`` but missing from ``key_order`` are appended last
    in their original insertion order, so unknown / future keys are preserved.
    """
    ordered = {key: data[key] for key in key_order if key in data}
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _format_log_value(value):
    """Render a single value for the plain-text run log."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(_to_serializable(value))
    if isinstance(value, str):
        return value
    return json.dumps(_to_serializable(value))


def _format_run_log(full_config, node_positions):
    """Render the run log as a plain-text document for human inspection."""
    lines = ["# Configuration"]
    skip_keys = {"node_positions"}
    seen = set()
    for key in CANONICAL_CONFIG_ORDER:
        if key in skip_keys or key not in full_config:
            continue
        lines.append(f"{key}: {_format_log_value(full_config[key])}")
        seen.add(key)
    for key, value in full_config.items():
        if key in skip_keys or key in seen:
            continue
        lines.append(f"{key}: {_format_log_value(value)}")
    lines.append("")

    positions_serialized = _to_serializable(node_positions)
    lines.append(f"# Node positions ({len(positions_serialized)})")
    for idx, pos in enumerate(positions_serialized):
        lines.append(f"  {idx}: {json.dumps(pos)}")

    return "\n".join(lines) + "\n"


def _parse_log_value(value):
    value = value.strip()
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_run_log(log_text):
    config = {}
    node_positions = []
    in_node_positions = False
    for line in log_text.splitlines():
        if not line.strip():
            continue
        if line.startswith("# Node positions"):
            in_node_positions = True
            continue
        if line.startswith("#"):
            continue
        if in_node_positions:
            _, value = line.split(":", 1)
            node_positions.append(json.loads(value.strip()))
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = _parse_log_value(value)
    if node_positions:
        config["node_positions"] = node_positions
    return order_config(config)


def save_experiment_config(experiment_dir, full_config, node_positions):
    """Write the full run log for one experiment."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    log_path = experiment_dir / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_text = _format_run_log(full_config, node_positions)
    log_path.write_text(log_text, encoding="utf-8")


def load_experiment_config(experiment_dir):
    """Load experiment config from run.log."""
    experiment_dir = Path(experiment_dir)
    log_path = experiment_dir / "run.log"
    if not log_path.exists():
        raise FileNotFoundError(f"No run log found at {log_path}")
    return _parse_run_log(log_path.read_text(encoding="utf-8"))


def plot_generated_trajectories(node_positions, node_types, trajectories, max_trajectories=4, title_prefix="Generated trajectories", save_path=None):
    """Plot a few generated trajectories to visualize the training data distribution."""
    if len(trajectories) == 0:
        return

    max_trajectories = min(max_trajectories, len(trajectories))
    fig, axes = plt.subplots(1, max_trajectories, figsize=(5 * max_trajectories, 5), squeeze=False)

    for idx in range(max_trajectories):
        ax = axes[0, idx]
        trajectory = trajectories[idx]
        x_vals = trajectory[:, 0, 0]
        y_vals = trajectory[:, 2, 0]
        if isinstance(trajectory, torch.Tensor):
            x_vals = x_vals.cpu().numpy()
            y_vals = y_vals.cpu().numpy()

        ax.plot(x_vals, y_vals, "b-", linewidth=2, marker="o", markersize=2, alpha=0.8)
        for node_idx, (x_pos, y_pos) in enumerate(node_positions):
            color = "#e63946" if node_types[node_idx] == 1 else "#457b9d"
            ax.plot(x_pos, y_pos, "s", markersize=8, color=color)
        ax.set_title(f"{title_prefix} #{idx + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved trajectory preview: {save_path}")
    plt.show()
    plt.close(fig)
