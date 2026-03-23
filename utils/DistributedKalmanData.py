import os
import torch
import random
import numpy as np
import networkx as nx
from torch.func import jacfwd, jacrev
from torch.autograd.functional import jacobian
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


def generate_data_point_const_vel(f_system, initial_state, num_time_steps, process_noise_std):
    """
    Generate ground truth state trajectory.
    x_{k+1} = f(x_k) + w_k
    """
    state_dim = initial_state.shape[0]
    trajectory = torch.zeros((num_time_steps, state_dim, 1))
    state_noise = torch.randn(num_time_steps, state_dim, 1) * process_noise_std

    x_current = torch.tensor(initial_state, dtype=torch.float)
    for k in range(num_time_steps):
        x_next = f_system(x_current) + state_noise[k]
        trajectory[k] = x_next
        x_current = x_next

    return trajectory


def generate_measurements_const_vel(h_system, trajectory, measurement_noise_std):
    """
    Generate noisy measurements from trajectory.
    y_k^{(i)} = h^{(i)}(x_k) + v_k^{(i)}

    ClassicDistributedKalman.py expects: [num_nodes, batch=1, time_steps]

    :return: measurements [num_nodes, batch=1, time_steps]
    """
    num_time_steps = trajectory.shape[0]
    num_nodes = h_system.num_nodes

    # Shape: [num_nodes, batch=1, time_steps] for ClassicDistributedKalman.py
    measurements = np.zeros((num_nodes, 1, num_time_steps))
    observation_noise = np.random.randn(num_nodes, num_time_steps) * measurement_noise_std

    for k in range(num_time_steps):
        x_k = trajectory[k].numpy() if isinstance(trajectory[k], torch.Tensor) else trajectory[k]
        obs = h_system.func(x_k)  # [num_nodes, 1] - use func() method
        measurements[:, 0, k] = obs[:, 0] + observation_noise[:, k]

    return measurements

class CreateGraph:
    def __init__(self, node_num, k_neighbors=5, rewrite_prob=0.4, seed=42):
        self.node_num = node_num
        self.graph = nx.connected_watts_strogatz_graph(node_num, k_neighbors, rewrite_prob, seed=seed)
        self.graph.add_edges_from([(i, i) for i in range(node_num)])
        self.adj_matrix = nx.adjacency_matrix(self.graph).todense()
        self.edges = self.graph.edges()


class FSystem:
    def __init__(self, alpha=1, beta=1, sigma=0, delta=0, deg=0):
        self.f = lambda x: x + alpha * torch.sin(beta * x + sigma * np.pi) + delta
        self.rotation_matrix = torch.tensor(
            [[np.cos(deg), -1*np.sin(deg)], [np.sin(deg), np.cos(deg)]], dtype=torch.float)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            return (self.rotation_matrix @ self.f(x)).numpy()
        return self.rotation_matrix @ self.f(x)

    def jacobian(self, x):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            flag = 1
        f_jac = torch.vmap(jacrev(self.f))
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
        return self.rotation_matrix @ self.A @ x

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
        x = x.to(torch.float)
        return torch.tensor([[0., 1.],], dtype=torch.float) @  self.rotation_matrix @ (torch.sign(x) * (x ** 2) ** 0.6)

    def h2(self, x):
        x = x.to(torch.float)
        return torch.tensor([[1., 0.],], dtype=torch.float) @ self.rotation_matrix @ (x + torch.arctan(x))

    def func(self, x, n_expansions=0):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            flag = 1
        node_classification = self.node_classification
        for i in range(n_expansions):
            node_classification = node_classification[:, None]
        result = node_classification * self.h1(x) + (1 - node_classification) * self.h2(x)
        if flag:
            return result.numpy()
        return result

    def __call__(self, x, n_expansions=0):
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
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
            x = torch.tensor(x, dtype=torch.float)
            flag = 1
        h1_jac = torch.vmap(jacrev(self.h1))
        h2_jac = torch.vmap(jacrev(self.h2))
        node_classification = self.node_classification
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
        result = H @ x
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
        return H.repeat_interleave(x.shape[0], dim=0)


class GraphDataset(Dataset):
    def __init__(
            self, g, f_system, h_system, q, r_array, monte_carlo_simulations=1000,
            time_steps=100, n_expansions=0, x0=10,state_dim=2
    ):
        super(GraphDataset, self).__init__()
        self.state_dim = state_dim
        if isinstance(g, np.ndarray):
            self.nx_graph = nx.from_numpy_array(g)
            self.adj_matrix = g
        else:
            self.nx_graph = g.graph
            self.adj_matrix = np.array(g.adj_matrix)
        self.g = g
        self.r_array = r_array
        self.monte_carlo_simulations = monte_carlo_simulations
        self.time_steps = time_steps
        self.x0 = self._build_initial_state_batch(x0)
        self.data_points = generate_data_points(f_system, q, self.x0, self.time_steps)
        self.measurements = self.generate_measurements(h_system, n_expansions)
        self.h_system = h_system
        self.data = self.create_dataset()

    def _build_initial_state_batch(self, x0):
        """
        Build a batch of initial states.

        Supported inputs:
        - scalar: same center value for every state component
        - vector of shape (state_dim,) or (state_dim, 1): one initial-state vector
        - batch of shape (monte_carlo_simulations, state_dim, 1): full batch supplied directly
        """
        noise = np.random.randn(self.monte_carlo_simulations, self.state_dim, 1).astype(np.float32)

        if np.isscalar(x0):
            base_state = np.full((self.state_dim, 1), x0, dtype=np.float32)
            return base_state[None, ...] + noise

        x0_array = np.asarray(x0, dtype=np.float32)

        if x0_array.shape == (self.state_dim,):
            x0_array = x0_array[:, None]

        if x0_array.shape == (self.state_dim, 1):
            return x0_array[None, ...] + noise

        expected_batch_shape = (self.monte_carlo_simulations, self.state_dim, 1)
        if x0_array.shape == expected_batch_shape:
            return x0_array

        raise ValueError(
            f"x0 must be a scalar, shape ({self.state_dim},), "
            f"shape ({self.state_dim}, 1), or shape {expected_batch_shape}, "
            f"but got shape {x0_array.shape}."
        )


    def generate_measurements(self, h_func, n_expansions):
        data_to_pass = self.data_points.transpose(1, 2, -1, 0).reshape(self.state_dim, -1)
        measurements = generate_measurements(h_func, data_to_pass, self.r_array, n_expansions)
        measurements = measurements.reshape(self.nx_graph.number_of_nodes(), 1, self.time_steps, self.monte_carlo_simulations)
        measurements = measurements.transpose(3, 0, 2, 1)
        return measurements

    def create_dataset(self):
        data_list = []
        for idx in range(self.monte_carlo_simulations):
            data = Data(x=torch.tensor(self.measurements[idx, ...], dtype=torch.float),
                        edge_index=torch.tensor(np.array(self.nx_graph.edges).T, dtype=torch.int64),
                        y=torch.tensor(self.data_points[idx, ...].transpose(-1, 0, 1), dtype=torch.float),
                        edge_attr=torch.randn(self.nx_graph.number_of_edges(), 1, dtype=torch.float),
                        adj_matrix=torch.Tensor(self.adj_matrix), h_system=self.h_system)
            data_list.append(data)
        return data_list

    def len(self):
        return self.monte_carlo_simulations

    def get(self, idx):
        return self.data[idx]
