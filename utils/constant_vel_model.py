import numpy as np
import torch
import networkx as nx
import random


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
        return self.state_transition_matrix @ x

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
        """Same as func() for compatibility."""
        return self.func(state_vector, n_expansions)

    def jacobian(self, state_vector, n_expansions=0):
        """
        Compute Jacobians for EKF linearization.

        For n_expansions=1 (used by ClassicDistributedKalman.py):
            Input: (N, state_dim, 1) per-node states
            Output: (N, 1, N, state_dim, 1)
                    After [:, 0, ...] -> (N, N, state_dim, 1)
                    Then H_T[j, ...] gives (N, state_dim, 1) for node j
        """
        sv = state_vector.numpy() if hasattr(state_vector, 'numpy') else np.asarray(state_vector)

        if sv.ndim == 2:
            sv = sv[np.newaxis, ...]
        batch = sv.shape[0]  # N nodes for n_expansions=1

        # Compute Jacobians: (batch, N, state_dim)
        jacs = np.zeros((batch, self.num_nodes, self.state_dimension))
        for b in range(batch):
            x_pos, y_pos = sv[b, 0, 0], sv[b, 2, 0]
            for i in range(self.num_nodes):
                jacs[b, i, :] = self._jac_single(x_pos, y_pos, i)

        if n_expansions > 0:
            # Shape: (batch, 1, N, state_dim, 1) for ClassicDistributedKalman.py
            return jacs[:, np.newaxis, :, :, np.newaxis]
        else:
            # Shape: (N, batch, state_dim, 1)
            return jacs.transpose(1, 0, 2)[:, :, :, np.newaxis]


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

    return nx.to_numpy_array(g)


