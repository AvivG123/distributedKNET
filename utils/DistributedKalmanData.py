import os
import torch
import random
import numpy as np
import networkx as nx
from torch.func import jacfwd
from torch_geometric.data import Data, Dataset


def seed_everything(seed=42):
    """
    Set the random seed for reproducibility.
    :param seed: the random seed to set
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_data_points(f, q, x0, time_steps):
    """
    Generate data points using the function f and noise q.
    :param f: the time propagation function
    :param q: the noise level
    :param x0: starting point for the data points
    :param time_steps: number of time steps to generate for each data point
    :return: the generated data points that we want to measure then predict
    """
    seed_everything(42)
    noise_shape = x0.shape + (time_steps,)
    w = q * np.random.randn(*noise_shape)
    data_points = np.zeros(shape=noise_shape)
    x = x0
    for i in range(time_steps):
        x = f(x) + w[..., i]
        data_points[..., i] = x
    return data_points  # sdsdsdfsdf


def generate_measurements(h, data_points, r_array, n_expansions=0):
    """
    Generate measurements using the function h and noise r_array.
    :param h: the measurement function
    :param data_points: the data points to be measured by the nodes
    :param r_array: the noise level for each node
    :param n_expansions: number of expansions for the measurement function
    :return:
    """
    seed_everything(42)
    z = h(data_points, n_expansions=n_expansions)
    noise_shape = z.shape
    # print(z)
    v = r_array[:, None, None] * np.random.randn(*noise_shape)
    measurements = z + v
    return measurements


class CreateGraph:
    def __init__(self, node_num, k_neighbors=5, rewrite_prob=0.4, seed=42):
        self.node_num = node_num
        self.graph = nx.connected_watts_strogatz_graph(node_num, k_neighbors, rewrite_prob, seed=seed)
        self.graph.add_edges_from([(i, i) for i in range(node_num)])
        self.adj_matrix = nx.adjacency_matrix(self.graph).todense()
        self.edges = self.graph.edges()


class FSystem:
    def __init__(self):
        self.f = lambda x: torch.sin(x) + x

    def func(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
            return self.f(x).numpy()
        return self.f(x)

    def jacobian(self, x):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            flag = 1
        f_jac = torch.vmap(jacfwd(self.f))
        if flag:
            return f_jac(x).numpy()
        return f_jac(x)


class HSystem:
    def __init__(self, node_num):
        # self.h2 = lambda x: torch.tensor([1., 0.]) @ ((x ** 2) ** 0.6)
        self.node_classification = torch.tensor(np.random.binomial(1, 0.5, (node_num, 1)), dtype=torch.float)

    def h1(self, x):
        x = x.to(torch.float)
        return torch.tensor([[0., 1.],], dtype=torch.float) @ (x ** 2) ** 0.6

    def h2(self, x):
        x = x.to(torch.float)
        return torch.tensor([[1., 0.],], dtype=torch.float) @ (x * torch.tanh(x))

    def func(self, x, n_expansions=0):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            flag = 1
        node_classification = self.node_classification
        for i in range(n_expansions):
            node_classification = node_classification[:, None]
        result = node_classification * self.h1(x) + (1 - node_classification) * self.h2(x)
        if flag:
            return result.numpy()
        return result

    def jacobian(self, x, n_expansions=0):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
            flag = 1
        h1_jac = torch.vmap(jacfwd(self.h1))
        h2_jac = torch.vmap(jacfwd(self.h2))
        node_classification = self.node_classification
        for i in range(n_expansions):
            node_classification = node_classification[:, None]
        result = node_classification * h1_jac(x) + (1 - node_classification) * h2_jac(x)
        if flag:
            return result.numpy()
        return result


class GraphDataset(Dataset):
    def __init__(
            self, g, f_system, h_system, q, r_array, monte_carlo_simulations=1000,
            time_steps=100, n_expansions=0, mean=None, std=None
    ):
        super(GraphDataset, self).__init__()
        self.g = g
        self.f = f_system.func
        self.h = h_system.func
        self.q = q
        self.r_array = r_array
        self.monte_carlo_simulations = monte_carlo_simulations
        self.time_steps = time_steps
        self.x0 = (10 * np.ones((self.monte_carlo_simulations, 2, 1), dtype=np.float32) +
                   np.random.randn(self.monte_carlo_simulations, 2, 1))
        self.data_points = generate_data_points(self.f, self.q, self.x0, self.time_steps)
        self.mean = mean
        self.std = std
        self.measurements = self.generate_measurements(n_expansions)
        self.data = self.create_dataset()

    def generate_measurements(self, n_expansions):
        data_to_pass = self.data_points.transpose(1, 2, -1, 0).reshape(2, -1)
        measurements = generate_measurements(self.h, data_to_pass, self.r_array, n_expansions)
        measurements = measurements.reshape(self.g.graph.number_of_nodes(), 1, self.time_steps, self.monte_carlo_simulations)
        measurements = measurements.transpose(3, 0, 2, 1)
        return measurements

    def normalize_data(self, data):
        if self.mean is None and self.std is None:
            self.mean = np.mean(data, axis=(0, 2))[None, :, None, ...]
            self.std = np.std(data, axis=(0, 2))[None, :, None, ...]
        return (data - self.mean) / self.std

    def create_dataset(self):
        data_list = []
        for idx in range(self.monte_carlo_simulations):
            data = Data(x=torch.tensor(self.measurements[idx, ...], dtype=torch.float64),
                        edge_index=torch.tensor(np.array(self.g.edges).T, dtype=torch.int64),
                        y=torch.tensor(self.data_points[idx, ...].transpose(-1, 0, 1), dtype=torch.float64),
                        edge_attr=torch.randn(self.g.graph.number_of_edges(), 1, dtype=torch.float64),
                        adj_matrix=torch.Tensor(self.g.adj_matrix))
            data_list.append(data)
        return data_list

    def len(self):
        return self.monte_carlo_simulations

    def get(self, idx):
        return self.data[idx]
