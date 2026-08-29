"""Localization models, simulation helpers, and scenario metrics."""

from __future__ import annotations

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from utils.DistributedKalmanData import build_bidirectional_edge_index


class ConstantVelocityModel:
    """Constant-velocity state model for [x, vx, y, vy]."""

    def __init__(self, time_delta: float):
        self.dt = time_delta
        self.state_dimension = 4
        self.state_transition_matrix = torch.tensor(
            [
                [1, self.dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, self.dt],
                [0, 0, 0, 1],
            ],
            dtype=torch.float,
        )

    def __call__(self, x):
        transition = self.state_transition_matrix
        if isinstance(x, np.ndarray):
            return (transition @ torch.as_tensor(x, dtype=torch.float)).numpy()
        return transition.to(device=x.device, dtype=x.dtype) @ x

    def jacobian(self, x):
        transition = self.state_transition_matrix.numpy()
        batch_size = 1 if x is None or x.ndim == 1 else x.shape[0]
        return np.tile(transition[None, ...], (batch_size, 1, 1))


class DistanceAngleObservation:
    """Alternating range/DOA observations from fixed sensor nodes.

    Even node indices measure DOA using the navigation convention
    ``atan2(dx, dy)``; odd node indices measure range.
    """

    def __init__(self, node_positions):
        self.num_nodes = len(node_positions)
        self.node_positions = np.asarray(node_positions, dtype=float)
        self.state_dimension = 4
        self.measurement_dimension = 1
        node_classification = (np.arange(self.num_nodes) % 2).reshape(-1, 1)
        self.node_classification = torch.tensor(node_classification, dtype=torch.float)
        self.angle_node_mask_np = node_classification[:, 0] == 0
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
        dx = x_pos - self.node_positions[node_idx, 0]
        dy = y_pos - self.node_positions[node_idx, 1]
        if self.node_classification[node_idx, 0].item() == 1:
            return np.sqrt(dx ** 2 + dy ** 2)
        return np.arctan2(dx, dy)

    def _jac_single(self, x_pos, y_pos, node_idx):
        dx = x_pos - self.node_positions[node_idx, 0]
        dy = y_pos - self.node_positions[node_idx, 1]
        jacobian = np.zeros(self.state_dimension)
        if self.node_classification[node_idx, 0].item() == 1:
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance > 1e-10:
                jacobian[0], jacobian[2] = dx / distance, dy / distance
        else:
            denominator = dx ** 2 + dy ** 2
            if denominator > 1e-10:
                jacobian[0], jacobian[2] = dy / denominator, -dx / denominator
        return jacobian

    def func(self, state_vector, n_expansions=0):
        state = state_vector.numpy() if hasattr(state_vector, "numpy") else np.asarray(state_vector)
        if n_expansions == 0:
            if state.ndim == 3:
                state = state[0]
            values = [self._obs_single(state[0, 0], state[2, 0], i) for i in range(self.num_nodes)]
            return np.asarray(values)[:, None]

        if state.ndim == 2:
            state = state[None, ...]
        result = np.zeros((self.num_nodes, state.shape[0], 1))
        for batch_idx in range(state.shape[0]):
            for node_idx in range(self.num_nodes):
                result[node_idx, batch_idx, 0] = self._obs_single(
                    state[batch_idx, 0, 0],
                    state[batch_idx, 2, 0],
                    node_idx,
                )
        return result

    def __call__(self, state_vector, n_expansions=0):
        is_tensor = isinstance(state_vector, torch.Tensor)
        state = state_vector if is_tensor else np.asarray(state_vector)
        if is_tensor:
            node_positions = torch.as_tensor(
                self.node_positions, dtype=state.dtype, device=state.device
            )
            node_types = self.node_classification.to(device=state.device, dtype=state.dtype)

        if state.ndim == 2 and state.shape[1] > 1:
            if is_tensor:
                dx = state[0].unsqueeze(0) - node_positions[:, 0].unsqueeze(1)
                dy = state[2].unsqueeze(0) - node_positions[:, 1].unsqueeze(1)
                distance = torch.sqrt(dx ** 2 + dy ** 2)
                angle = torch.atan2(dx, dy)
                return (node_types * distance + (1 - node_types) * angle).unsqueeze(1)
            result = np.zeros((self.num_nodes, 1, state.shape[1]))
            for node_idx in range(self.num_nodes):
                dx = state[0] - self.node_positions[node_idx, 0]
                dy = state[2] - self.node_positions[node_idx, 1]
                if self.node_classification[node_idx, 0].item() == 1:
                    result[node_idx, 0] = np.sqrt(dx ** 2 + dy ** 2)
                else:
                    result[node_idx, 0] = np.arctan2(dx, dy)
            return result

        if state.ndim == 4:
            if is_tensor:
                x_pos = state[:, :, 0, 0].unsqueeze(-1)
                y_pos = state[:, :, 2, 0].unsqueeze(-1)
                dx = x_pos - node_positions[:, 0].view(1, 1, -1)
                dy = y_pos - node_positions[:, 1].view(1, 1, -1)
                distance = torch.sqrt(dx ** 2 + dy ** 2)
                angle = torch.atan2(dx, dy)
                types = node_types[:, 0].view(1, 1, -1)
                return (types * distance + (1 - types) * angle).unsqueeze(-1)
            result = np.zeros((state.shape[0], state.shape[1], self.num_nodes, 1))
            for batch_idx in range(state.shape[0]):
                for source_idx in range(state.shape[1]):
                    for sensor_idx in range(self.num_nodes):
                        result[batch_idx, source_idx, sensor_idx, 0] = self._obs_single(
                            state[batch_idx, source_idx, 0, 0],
                            state[batch_idx, source_idx, 2, 0],
                            sensor_idx,
                        )
            return result

        result = self.func(state, n_expansions)
        if is_tensor:
            return torch.as_tensor(result, dtype=state.dtype, device=state.device)
        return result

    def jacobian(self, state_vector, n_expansions=0):
        is_tensor = isinstance(state_vector, torch.Tensor)
        state = state_vector if is_tensor else np.asarray(state_vector)
        if state.ndim == 2:
            state = state.unsqueeze(0) if is_tensor else state[None, ...]

        if is_tensor:
            node_positions = torch.as_tensor(
                self.node_positions, dtype=state.dtype, device=state.device
            )
            node_types = self.node_classification.to(
                device=state.device, dtype=state.dtype
            )[:, 0].unsqueeze(0)
            dx = state[:, 0, 0].unsqueeze(1) - node_positions[:, 0].unsqueeze(0)
            dy = state[:, 2, 0].unsqueeze(1) - node_positions[:, 1].unsqueeze(0)
            distance = torch.sqrt(dx ** 2 + dy ** 2)
            denominator = dx ** 2 + dy ** 2
            zero_x = torch.zeros_like(dx)
            zero_y = torch.zeros_like(dy)
            distance_dx = torch.where(distance > 1e-10, dx / distance, zero_x)
            distance_dy = torch.where(distance > 1e-10, dy / distance, zero_y)
            angle_dx = torch.where(denominator > 1e-10, dy / denominator, zero_x)
            angle_dy = torch.where(denominator > 1e-10, -dx / denominator, zero_y)
            jacobians = torch.zeros(
                (state.shape[0], self.num_nodes, self.state_dimension),
                dtype=state.dtype,
                device=state.device,
            )
            jacobians[:, :, 0] = node_types * distance_dx + (1 - node_types) * angle_dx
            jacobians[:, :, 2] = node_types * distance_dy + (1 - node_types) * angle_dy
            if n_expansions > 0:
                return jacobians[:, None, :, :, None]
            return jacobians.transpose(0, 1)[:, :, :, None]

        jacobians = np.zeros((state.shape[0], self.num_nodes, self.state_dimension))
        for batch_idx in range(state.shape[0]):
            for node_idx in range(self.num_nodes):
                jacobians[batch_idx, node_idx] = self._jac_single(
                    state[batch_idx, 0, 0],
                    state[batch_idx, 2, 0],
                    node_idx,
                )
        if n_expansions > 0:
            return jacobians[:, None, :, :, None]
        return jacobians.transpose(1, 0, 2)[:, :, :, None]


def generate_node_positions(num_nodes, seed=None, area_size=100.0):
    rng = np.random.default_rng(seed)
    return rng.random((num_nodes, 2)) * area_size


def create_distance_based_graph(node_positions, k_neighbors=3, seed=None):
    num_nodes = len(node_positions)
    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for node_idx in range(num_nodes):
        distances = np.linalg.norm(node_positions - node_positions[node_idx], axis=1)
        distances[node_idx] = np.inf
        local_k = max(2, k_neighbors + rng.integers(-1, 2))
        for neighbor_idx in np.argsort(distances)[:local_k]:
            graph.add_edge(node_idx, neighbor_idx)
    if not nx.is_connected(graph):
        for component in list(nx.connected_components(graph))[1:]:
            best = min(
                (
                    (np.linalg.norm(node_positions[u] - node_positions[v]), u, v)
                    for u in component
                    for v in range(num_nodes)
                    if v not in component
                ),
                default=None,
            )
            if best is not None:
                graph.add_edge(best[1], best[2])
    graph.add_edges_from((i, i) for i in range(num_nodes))
    return nx.to_numpy_array(graph)


def _process_covariance(process_noise, state_dim):
    if np.isscalar(process_noise):
        return np.eye(state_dim) * float(process_noise) ** 2
    noise = np.asarray(process_noise, dtype=float)
    if noise.shape == (state_dim,):
        return np.diag(noise ** 2)
    if noise.shape != (state_dim, state_dim):
        raise ValueError(
            f"process_noise must be scalar, shape ({state_dim},), or "
            f"shape ({state_dim}, {state_dim}); got {noise.shape}."
        )
    return noise # returns Q = diag(0, q**2, 0, q**2)


def generate_trajectory(f_system, initial_state, num_time_steps, process_noise):
    state_dim = initial_state.shape[0]
    covariance = torch.as_tensor(
        _process_covariance(process_noise, state_dim), dtype=torch.float32
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if torch.min(eigenvalues) < -1e-7:
        raise ValueError("process covariance must be positive semidefinite.")
    square_root = (eigenvectors * torch.sqrt(torch.clamp(eigenvalues, min=0))) @ eigenvectors.T
    standard_noise = torch.randn(num_time_steps, state_dim, 1)
    state_noise = square_root @ standard_noise
    trajectory = torch.zeros((num_time_steps, state_dim, 1), dtype=torch.float32)
    x_current = torch.as_tensor(initial_state, dtype=torch.float32)
    for time_idx in range(num_time_steps):
        x_current = f_system(x_current) + state_noise[time_idx]
        trajectory[time_idx] = x_current
    return trajectory


def generate_measurements(h_system, trajectory, measurement_noise_std):
    num_time_steps = trajectory.shape[0]
    noise_std = np.asarray(measurement_noise_std, dtype=float)
    if noise_std.ndim == 0:
        noise_std = np.full(h_system.num_nodes, float(noise_std))
    else:
        noise_std = noise_std.reshape(-1)
    if noise_std.shape != (h_system.num_nodes,):
        raise ValueError(
            f"measurement_noise_std must be scalar or have one value per node "
            f"({h_system.num_nodes}); got {noise_std.shape}."
        )
    measurements = np.zeros((h_system.num_nodes, 1, num_time_steps))
    observation_noise = np.random.randn(h_system.num_nodes, num_time_steps) * noise_std[:, None]
    for time_idx in range(num_time_steps):
        state = trajectory[time_idx]
        state = state.numpy() if isinstance(state, torch.Tensor) else state
        measurements[:, 0, time_idx] = h_system.func(state)[:, 0] + observation_noise[:, time_idx]
    return h_system.wrap_measurements(measurements, sensor_axis=0)


def build_graph_data_for_dkn(adjacency_matrix, h_system, trajectory, measurements):
    adjacency = np.array(adjacency_matrix, dtype=float, copy=True)
    np.fill_diagonal(adjacency, 1.0)
    edge_index = build_bidirectional_edge_index(nx.from_numpy_array(adjacency))
    return Data(
        x=torch.tensor(measurements.transpose(0, 2, 1), dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(trajectory, dtype=torch.float32),
        adj_matrix=torch.tensor(adjacency, dtype=torch.float32),
        h_system=h_system,
    )


def build_dkn_model(config, f_model, r_array, x0):
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
        r_array=r_array,
        learn_edge_kalman=config["learn_edge_kalman"],
        x0_scale=x0,
        consensus_layer=config.get("consensus_layer", "none"),
        position_only_loss=config.get("position_only_loss", False),
    )


def position_error_from_state_sequence(trajectory, estimate):
    truth = trajectory.detach().cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory
    return np.sqrt(
        (truth[:, 0, 0] - estimate[:, 0]) ** 2
        + (truth[:, 2, 0] - estimate[:, 2]) ** 2
    ).mean()


def position_error_from_dekf(trajectory, estimate):
    truth = trajectory.detach().cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory
    x_estimate = estimate[:, :, 0, 0].mean(axis=1)
    y_estimate = estimate[:, :, 2, 0].mean(axis=1)
    return np.sqrt(
        (truth[:, 0, 0] - x_estimate) ** 2
        + (truth[:, 2, 0] - y_estimate) ** 2
    ).mean()


def generate_trial_data(f_system, h_system, x0, time_steps, process_noise, measurement_noise_std):
    trajectory = generate_trajectory(f_system, x0, time_steps, process_noise)
    measurements = generate_measurements(h_system, trajectory, measurement_noise_std)
    return trajectory, measurements


def localization_x0(area_size, time_delta, time_steps):
    speed = float(area_size) / (float(time_delta) * int(time_steps))
    return [0.0, speed, 0.0, speed]


def format_dt_ratio(ratio):
    ratio = float(ratio)
    return f"{ratio:.1f}" if np.isclose(ratio, round(ratio)) else f"{ratio:g}"


def nearest_nominal_dt_ratio(dt_values):
    ratio = min(dt_values, key=lambda value: abs(float(value) - 1.0))
    if not np.isclose(float(ratio), 1.0):
        raise RuntimeError(f"Need a no-mismatch dt ratio near 1.0; got {dt_values}.")
    return ratio


def farthest_mismatch_dt_ratio(dt_values, nominal_ratio):
    candidates = [
        ratio for ratio in dt_values
        if not np.isclose(float(ratio), float(nominal_ratio))
    ]
    if not candidates:
        raise RuntimeError(
            f"Need at least one dt mismatch ratio besides {nominal_ratio}; got {dt_values}."
        )
    return max(candidates, key=lambda ratio: abs(float(ratio) - 1.0))


def experiment_time_steps(config):
    for key in ("time_steps", "test_time_steps", "train_time_steps"):
        if config.get(key) is not None:
            return config[key]
    raise KeyError("Experiment config must include time_steps.")


def sync_torch_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
