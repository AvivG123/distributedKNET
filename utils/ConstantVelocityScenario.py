import numpy as np
import torch
import networkx as nx
import random
import os
import json
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data


class ConstantVelocityModel:
    """
    Linear state evolution for constant velocity target tracking.
    State: [x, vx, y, vy]^T (position and velocity in 2D)

    Compatible with ClassicDistributedKalman.py interface:
    - __call__(x): returns next state
    - jacobian(x): returns Jacobian matrix
    """

    def __init__(self, time_delta: float):
        self.dt = time_delta
        self.state_dimension = 4
        # F matrix for constant velocity model
        self.state_transition_matrix = torch.tensor(
            [[1, self.dt, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, self.dt],
             [0, 0, 0, 1]], dtype=torch.float)

    def __call__(self, x):
        """Apply state evolution: x_{k+1} = F * x_k"""
        if isinstance(x, np.ndarray):
            return (self.state_transition_matrix @ torch.tensor(x, dtype=torch.float)).numpy()
        transition_matrix = self.state_transition_matrix.to(device=x.device, dtype=x.dtype)
        return transition_matrix @ x

    def jacobian(self, x):
        """
        For linear system, Jacobian F is constant.

        Args:
            x: State array, shape (N, state_dim) or (state_dim,)
        Returns:
            Jacobians, shape (N, state_dim, state_dim)
        """
        F = self.state_transition_matrix.numpy()
        if x is None:
            return F[np.newaxis, ...]
        # Determine batch size from input
        if x.ndim == 1:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        return np.tile(F[np.newaxis, ...], (batch_size, 1, 1))


def build_bidirectional_edge_index(graph):
    """
    Build a directed PyG edge_index from a NetworkX graph.
    For each undirected edge (u, v), include both (u, v) and (v, u).
    Self loops (u, u) are kept once.
    """
    edge_pairs = np.asarray(list(graph.edges()), dtype=np.int64)
    if edge_pairs.size == 0:
        return torch.empty((2, 0), dtype=torch.int64)

    non_self_mask = edge_pairs[:, 0] != edge_pairs[:, 1]
    reverse_pairs = edge_pairs[non_self_mask][:, [1, 0]]
    directed_pairs = np.concatenate([edge_pairs, reverse_pairs], axis=0)
    # Guard against accidental duplicates if input already contains both directions.
    directed_pairs = np.unique(directed_pairs, axis=0)
    return torch.tensor(directed_pairs.T, dtype=torch.int64)


class DistanceAngleObservation:
    """
    Nonlinear observation function using distance/angle measurements.

    Type-0 (angle): h(x) = atan2(y - y_i, x - x_i)
    Type-1 (distance): h(x) = sqrt((x - x_i)^2 + (y - y_i)^2)

    Compatible with ClassicDistributedKalman.py interface:
    - __call__(x, n_expansions): returns measurements
    - func(x, n_expansions): same as __call__
    - jacobian(x, n_expansions): returns Jacobian matrices

    Note: node_classification uses 1=distance, 0=angle to match HSystem convention
    """

    def __init__(self, node_positions):
        self.num_nodes = len(node_positions)
        self.node_positions = np.array(node_positions, dtype=float)
        self.state_dimension = 4  # [x, vx, y, vy]
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
        """
        Wrap innovations for angle sensors into [-pi, pi], keep distance residuals linear.
        """
        mask = self._angle_mask_for(innovation, sensor_axis)
        wrapped = self._wrap_to_pi(innovation)
        if isinstance(innovation, torch.Tensor):
            return torch.where(mask, wrapped, innovation)
        return np.where(mask, wrapped, innovation)

    def wrap_measurements(self, measurements, sensor_axis):
        """
        Wrap absolute angle measurements into [-pi, pi], keep distance measurements unchanged.
        """
        mask = self._angle_mask_for(measurements, sensor_axis)
        wrapped = self._wrap_to_pi(measurements)
        if isinstance(measurements, torch.Tensor):
            return torch.where(mask, wrapped, measurements)
        return np.where(mask, wrapped, measurements)

    def _obs_single(self, x_pos, y_pos, node_idx):
        """Compute observation for single node."""
        node_type = self.node_classification[node_idx, 0].item()
        dx = x_pos - self.node_positions[node_idx, 0]
        dy = y_pos - self.node_positions[node_idx, 1]
        if node_type == 1:  # distance
            return np.sqrt(dx ** 2 + dy ** 2)
        else:  # angle
            return np.arctan2(dy, dx)

    def _jac_single(self, x_pos, y_pos, node_idx):
        """Compute Jacobian for single node (analytical). Returns shape (state_dim,)"""
        node_type = self.node_classification[node_idx, 0].item()
        dx = x_pos - self.node_positions[node_idx, 0]
        dy = y_pos - self.node_positions[node_idx, 1]
        jac = np.zeros(self.state_dimension)
        if node_type == 1:  # distance: d/dx sqrt((x-xi)^2 + (y-yi)^2)
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > 1e-10:
                jac[0], jac[2] = dx / dist, dy / dist  # d/dx, d/dy (vx, vy have 0 derivative)
        else:  # angle: d/dx atan2(y-yi, x-xi)
            denom = dx ** 2 + dy ** 2
            if denom > 1e-10:
                jac[0], jac[2] = -dy / denom, dx / denom
        return jac

    def func(self, state_vector, n_expansions=0):
        """
        Compute h^{(i)}(x) for all nodes.

        For n_expansions=0: Single state input
        For n_expansions=1: Per-node state input (N, state_dim, 1)

        Returns shape compatible with ClassicDistributedKalman.py
        """
        sv = state_vector.numpy() if hasattr(state_vector, 'numpy') else np.asarray(state_vector)

        if n_expansions == 0:
            # Single state: shape (state_dim, 1) or (1, state_dim, 1)
            if sv.ndim == 3:
                sv = sv[0]
            x_pos, y_pos = sv[0, 0], sv[2, 0]
            result = np.array([self._obs_single(x_pos, y_pos, i) for i in range(self.num_nodes)])
            return result[:, np.newaxis]  # (N, 1)
        else:
            # n_expansions=1: Per-node states, shape (N, state_dim, 1)
            if sv.ndim == 2:
                sv = sv[np.newaxis, ...]
            batch = sv.shape[0]
            result = np.zeros((self.num_nodes, batch, 1))
            for b in range(batch):
                x_pos, y_pos = sv[b, 0, 0], sv[b, 2, 0]
                for i in range(self.num_nodes):
                    result[i, b, 0] = self._obs_single(x_pos, y_pos, i)
            return result  # (N, batch, 1)

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

        # Case 1: Batched flattened input from generate_measurements: shape (state_dim, N)
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
            return result  # (num_nodes, 1, N)

        # Case 2: Training forward pass: shape (batch, num_nodes, state_dim, 1)
        # Return full cross-sensor predictions: (batch, source_node, sensor_node, 1)
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

        # Case 3: Single state or small input
        result = self.func(sv, n_expansions)
        if is_tensor:
            return torch.as_tensor(result, dtype=dtype, device=device)
        return result

    def jacobian(self, state_vector, n_expansions=0):
        """
        Compute Jacobians for EKF linearization.

        For n_expansions=1 (used by ClassicDistributedKalman.py):
            Input: (N, state_dim, 1) per-node states
            Output: (N, 1, N, state_dim, 1)
                    After [:, 0, ...] -> (N, N, state_dim, 1)
                    Then H_T[j, ...] gives (N, state_dim, 1) for node j

        During training (calculate_h_mat):
            Input: (batch*num_nodes, state_dim, 1) with n_expansions=1
            Output: (batch*num_nodes, 1, num_nodes, state_dim, 1)
                    After [:, 0, ...] -> (batch*num_nodes, num_nodes, state_dim, 1)
        """
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
    """
    Create a connected graph based on node distances.
    Each node connects to its k nearest neighbors.
    """
    num_nodes = len(node_positions)
    rng = np.random.default_rng(seed)

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))

    # Connect each node to k nearest neighbors
    for i in range(num_nodes):
        dists = np.linalg.norm(node_positions - node_positions[i], axis=1)
        dists[i] = np.inf  # Exclude self
        local_k = max(2, k_neighbors + rng.integers(-1, 2))
        neighbor_idx = np.argsort(dists)[:local_k]
        for j in neighbor_idx:
            g.add_edge(i, j)

    # Ensure connectivity
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

    # Keep self loops for parity with DistributedKalmanData/CreateGraph and DEKF's (A + I) usage.
    g.add_edges_from((i, i) for i in range(num_nodes))
    return nx.to_numpy_array(g)


def seed_everything(seed=42):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_trainer_accelerator():
    """Return the preferred Lightning accelerator for the current machine."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def generate_trajectory(f_system, initial_state, num_time_steps, process_noise_std):
    """
    Generate a state trajectory using the supplied dynamics model.

    Returns:
        torch.Tensor of shape (num_time_steps, state_dim, 1)
    """
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
    """
    Generate noisy node measurements from a trajectory.

    Returns:
        np.ndarray of shape (num_nodes, 1, num_time_steps)
    """
    num_time_steps = trajectory.shape[0]
    num_nodes = h_system.num_nodes

    measurements = np.zeros((num_nodes, 1, num_time_steps))
    observation_noise = np.random.randn(num_nodes, num_time_steps) * measurement_noise_std

    for k in range(num_time_steps):
        x_k = trajectory[k].numpy() if isinstance(trajectory[k], torch.Tensor) else trajectory[k]
        obs = h_system.func(x_k)
        measurements[:, 0, k] = obs[:, 0] + observation_noise[:, k]

    # Angle channels are circular variables and must stay in [-pi, pi] after adding noise.
    if hasattr(h_system, "wrap_measurements"):
        measurements = h_system.wrap_measurements(measurements, sensor_axis=0)

    return measurements


def build_graph_data_for_dkn(adjacency_matrix, h_system, trajectory, measurements):
    """
    Convert one simulated scenario into a PyG Data object for GraphKalmanProcess.

    Args:
        adjacency_matrix: np.ndarray of shape (num_nodes, num_nodes)
        trajectory: torch.Tensor or np.ndarray of shape (time_steps, state_dim, 1)
        measurements: np.ndarray of shape (num_nodes, 1, time_steps)

    Returns:
        torch_geometric.data.Data
    """
    # Idempotent A + I: ensures the diagonal is 1 even if caller forgot to include self loops.
    adjacency_with_self = np.array(adjacency_matrix, dtype=float, copy=True)
    np.fill_diagonal(adjacency_with_self, 1.0)

    # edge_index and adj_matrix must stay consistent for message passing vs Kalman aggregation.
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
    if "train_loss:_epoch" in df.columns:
        train_df = df.dropna(subset=["train_loss:_epoch"])
        plt.plot(train_df["epoch"], train_df["train_loss:_epoch"], marker="o", label="Train Loss")
    if "val_loss:_epoch" in df.columns:
        val_df = df.dropna(subset=["val_loss:_epoch"])
        plt.plot(val_df["epoch"], val_df["val_loss:_epoch"], marker="o", label="Val Loss")

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


# Match a JSON array whose elements all sit on their own line and contain no
# nested arrays / objects. Used to collapse arrays of primitives onto a single
# line for readability (e.g. ``"x0": [0, 50, 0, 50]``). Nested structures like
# ``node_positions`` (a list of lists) only get their innermost arrays
# inlined; the outer array stays expanded.
_PRIMITIVE_ARRAY_RE = re.compile(
    r"\[\s*\n"                            # opening bracket + newline
    r"(?:\s*[^][{}\n]+\s*,\s*\n)*"       # element + comma + newline (repeated)
    r"\s*[^][{}\n]+\s*\n"                # last element + newline
    r"\s*\]"                              # closing bracket
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


# Canonical key order. Mirrors ``configs/const_vel_scenario_configured.json``
# so that saved configs read top-to-bottom in the same order as the input.
# Keys not in the input config (``node_positions``, ``description``,
# ``run_metadata``) are slotted in at logical positions and at the end.
CANONICAL_CONFIG_ORDER = (
    "seed",
    "save_root",
    "state_dimension",
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
    "train_time_steps",
    "test_time_steps",
    "num_trials",
    "preview_trajectories",
    "description",
    "run_metadata",
)

# Keys that go into the small per-experiment config consumed by the comparison
# notebook. Anything outside this set lives in ``run_log.json`` only.
# The order here mirrors ``CANONICAL_CONFIG_ORDER`` so saved files read in the
# same order as the input config.
EXPERIMENT_CONFIG_KEYS = (
    "seed",
    "state_dimension",
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
    "learning_rate",
    "train_time_steps",
    "test_time_steps",
    "num_trials",
    "description",
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


def _build_experiment_config(full_config, node_positions):
    """Project a full training config down to the comparison-only fields."""
    subset = {key: full_config[key] for key in EXPERIMENT_CONFIG_KEYS if key in full_config}
    subset["node_positions"] = _to_serializable(node_positions)
    return order_config(subset, key_order=EXPERIMENT_CONFIG_KEYS)


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


def save_experiment_config(experiment_dir, full_config, node_positions):
    """
    Persist the per-experiment config artifacts.

    Writes into ``experiment_dir``:

    * ``experiment_config_<n>.json`` — small, frozen, comparison-only JSON
      subset including the resolved ``node_positions``. This is the only file
      the comparison notebook reads.
    * ``run.log`` — plain-text snapshot of the full training-time config and
      ``node_positions``. Intended for human inspection / reproducibility
      only, never read at inference.

    The JSON file uses ``CANONICAL_CONFIG_ORDER`` so it reads in the same order
    as the source ``configs/const_vel_scenario_configured.json``.
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    n = experiment_index_from_dir(experiment_dir)

    experiment_config = _build_experiment_config(full_config, node_positions)
    save_config(experiment_config, experiment_dir / f"experiment_config_{n}.json")

    log_path = experiment_dir / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_text = _format_run_log(full_config, node_positions)
    log_path.write_text(log_text, encoding="utf-8")


def load_experiment_config(experiment_dir):
    """Load the comparison-only ``experiment_config_<n>.json`` for an experiment."""
    experiment_dir = Path(experiment_dir)
    n = experiment_index_from_dir(experiment_dir)
    config_path = experiment_dir / f"experiment_config_{n}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No experiment config found at {config_path}")
    return load_config(config_path)


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
