"""Data-generation utilities for the distributed KalmanNet experiments.

This module defines:
- Graph/topology construction (`CreateGraph`).
- Simple nonlinear + linear state/measurement systems (`FSystem*`, `HSystem*`).
- A torch-geometric `Dataset` that generates Monte-Carlo simulations (`GraphDataset`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import networkx as nx
import torch
from torch.func import jacfwd
from torch_geometric.data import Data, Dataset

from utils.reproducibility import seed_everything as _seed_everything


def seed_everything(seed: int = 42) -> None:
    """Backward-compatible wrapper for older imports."""

    _seed_everything(seed)


def _randn(shape: tuple[int, ...], *, seed: int | None) -> np.ndarray:
    if seed is None:
        return np.random.randn(*shape)
    # Match legacy `np.random.seed(seed); np.random.randn(...)` behavior.
    return np.random.RandomState(int(seed)).randn(*shape)


def generate_data_points(
    f: Any,
    q: float,
    x0: np.ndarray,
    time_steps: int,
    *,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate state trajectories x[t] = f(x[t-1]) + w[t].

    Default `seed=42` preserves legacy behavior; pass `seed=None` to use the
    global NumPy RNG (e.g., when you already seeded globally).
    """

    noise_shape = x0.shape + (time_steps,)
    w = q * _randn(noise_shape, seed=seed)
    data_points = np.zeros(shape=noise_shape, dtype=np.float32)
    x = x0
    for t in range(time_steps):
        x = f(x) + w[..., t]
        data_points[..., t] = x
    return data_points


def generate_measurements(
    h: Any,
    data_points: np.ndarray,
    r_array: np.ndarray,
    *,
    n_expansions: int = 0,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate measurements z = h(x) + v."""

    z = h(data_points, n_expansions=n_expansions)
    v = r_array[:, None, None] * _randn(z.shape, seed=seed)
    return z + v


class CreateGraph:
    def __init__(self, node_num, k_neighbors=5, rewrite_prob=0.4, seed=42):
        self.node_num = node_num
        self.graph = nx.connected_watts_strogatz_graph(node_num, k_neighbors, rewrite_prob, seed=seed)
        self.graph.add_edges_from([(i, i) for i in range(node_num)])
        self.adj_matrix = np.asarray(nx.adjacency_matrix(self.graph).todense())
        self.edges = list(self.graph.edges())
        self.edge_index = torch.tensor(np.array(self.edges).T, dtype=torch.int64)


class FSystem:
    def __init__(self, alpha=1, beta=1, sigma=0, delta=0, deg=0):
        self.f = lambda x: x + alpha * torch.sin(beta * x + sigma * np.pi) + delta
        self.rotation_matrix = torch.tensor(
            [[np.cos(deg), -1*np.sin(deg)], [np.sin(deg), np.cos(deg)]], dtype=torch.float)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            return (self.rotation_matrix @ self.f(x)).numpy()
        return self.rotation_matrix.to(x.device) @ self.f(x)

    def jacobian(self, x):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            flag = 1
        f_jac = torch.vmap(jacfwd(self.f))
        if flag:
            return f_jac(x).numpy()
        return f_jac(x)


class FSystemLinear:
    def __init__(self, eps=0.015, deg=0):
        A0 = 2 * torch.tensor([[0, -1], [1, 0]])
        self.A = torch.eye(2) + eps * A0 + (((eps * A0) ** 2) / 2) + (((eps * A0) ** 3) / 6)
        self.rotation_matrix = torch.tensor(
            [[np.cos(deg), -1*np.sin(deg)], [np.sin(deg), np.cos(deg)]], dtype=torch.float)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            return (self.rotation_matrix @ self.A @ x).numpy()
        return self.rotation_matrix.to(x.device) @ self.A.to(x.device) @ x

    def jacobian(self, x):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            flag = 1
        A = self.rotation_matrix @ self.A
        if flag:
            return A.numpy()[None, ...].repeat(x.shape[0], axis=0)
        return A[None, ...].repeat_interleave(x.shape[0], dim=0)


class HSystem:
    def __init__(self, node_num, alpha=0):
        # self.h2 = lambda x: torch.tensor([1., 0.]) @ ((x ** 2) ** 0.6)
        self.rotation_matrix = torch.tensor(
            [[np.cos(alpha), -1*np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]], dtype=torch.float)
        self.node_classification = torch.tensor(np.random.binomial(1, 0.5, (node_num, 1)), dtype=torch.float)

    def h1(self, x):
        # x = x.to(torch.float)
        return torch.tensor([[0., 1.],], dtype=torch.float, device=x.device) @  self.rotation_matrix.to(device=x.device) @ (x * (x**2+1e-6)**0.1)

    def h2(self, x):
        # x = x.to(torch.float)
        return torch.tensor([[1., 0.],], dtype=torch.float, device=x.device) @  self.rotation_matrix.to(device=x.device) @ (x + torch.arctan(x))

    def func(self, x, n_expansions=0):
        return self(x, n_expansions=n_expansions)

    def __call__(self, x, n_expansions=0):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            flag = 1
        node_classification = self.node_classification.to(device=x.device)
        for i in range(n_expansions):
            node_classification = node_classification[:, None]
        result = node_classification * self.h1(x) + (1 - node_classification) * self.h2(x)
        if flag:
            return result.numpy()
        return result

    def jacobian(self, x, n_expansions=0):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            flag = 1
        h1_jac = torch.vmap(jacfwd(self.h1))
        h2_jac = torch.vmap(jacfwd(self.h2))
        node_classification = self.node_classification.to(device=x.device)
        for i in range(n_expansions):
            node_classification = node_classification[:, None]
        result = node_classification * h1_jac(x) + (1 - node_classification) * h2_jac(x)
        if flag:
            return result.numpy()
        return result


class HSystemLinear:
    def __init__(self, node_num, p=0.5, h1=None, h2=None):
        if h1 is None:
            h1 = [[0., 1.],]
        if h2 is None:
            h2 = [[1., 0.],]
        self.node_classification = torch.tensor(np.random.binomial(1, p, (node_num, 1)), dtype=torch.float)
        self.h1_matrix = torch.tensor(h1, dtype=torch.float)
        self.h2_matrix = torch.tensor(h2, dtype=torch.float)
        # self.H = self.node_classification * h1_matrix + (1 - self.node_classification) * h2_matrix

    def __call__(self, x, n_expansions=0):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            flag = 1
        node_classification = self.node_classification
        for i in range(n_expansions):
            node_classification = node_classification[..., None]
        H = node_classification * self.h1_matrix + (1 - node_classification) * self.h2_matrix
        result = H.to(device=x.device) @ x
        if flag:
            return result.numpy()
        return result

    def jacobian(self, x, n_expansions=0):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            flag = 1
        node_classification = self.node_classification
        for i in range(n_expansions):
            node_classification = node_classification[..., None]
        H = node_classification * self.h1_matrix + (1 - node_classification) * self.h2_matrix
        H = H.transpose(1, 2)[None, None, ...]
        if flag:
            return H.numpy().repeat(x.shape[0], axis=0)
        return H.repeat_interleave(x.shape[0], dim=0).to(device=x.device)


class GraphDataset(Dataset):
    def __init__(
            self, g, f_system, h_system, q, r_array, monte_carlo_simulations=1000,
            time_steps=100, n_expansions=0, x0=10, *, seed: int | None = 42
    ):
        super(GraphDataset, self).__init__()
        self.g = g
        self.r_array = r_array
        self.monte_carlo_simulations = monte_carlo_simulations
        self.time_steps = time_steps
        self.seed = seed
        self.x0 = (x0 * np.ones((self.monte_carlo_simulations, 2, 1), dtype=np.float32) +
                   np.random.randn(self.monte_carlo_simulations, 2, 1))
        self.data_points = generate_data_points(f_system, q, self.x0, self.time_steps, seed=seed)
        self.measurements = self.generate_measurements(h_system, n_expansions, seed=seed)
        self.h_system = h_system
        self._edge_index = g.edge_index if hasattr(g, "edge_index") else torch.tensor(np.array(g.edges).T, dtype=torch.int64)
        self._adj_matrix = torch.tensor(np.asarray(g.adj_matrix), dtype=torch.float)

    def generate_measurements(self, h_func, n_expansions, *, seed: int | None):
        # data_points: (mc, state_dim=2, 1, time) -> (2, mc*time)
        data_to_pass = self.data_points.transpose(1, 2, -1, 0).reshape(2, -1)
        measurements = generate_measurements(h_func, data_to_pass, self.r_array, n_expansions=n_expansions, seed=seed)
        measurements = measurements.reshape(
            self.g.graph.number_of_nodes(), 1, self.time_steps, self.monte_carlo_simulations
        )
        measurements = measurements.transpose(3, 0, 2, 1)
        return measurements

    def len(self):
        return self.monte_carlo_simulations

    def get(self, idx):
        return Data(
            x=torch.tensor(self.measurements[idx, ...], dtype=torch.float),
            edge_index=self._edge_index,
            y=torch.tensor(self.data_points[idx, ...].transpose(-1, 0, 1), dtype=torch.float),
            edge_attr=torch.zeros(self._edge_index.shape[1], 1, dtype=torch.float),
            adj_matrix=self._adj_matrix,
            h_system=self.h_system,
        )
