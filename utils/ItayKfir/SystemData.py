import torch
import numpy as np
from torch.func import jacrev
import matplotlib.pyplot as plt

class FSystemLinear:
    def __init__(self, time_delta: float):
        self.dt = time_delta
        self.F_matrix = torch.tensor([[1, self.dt, 0,       0], 
                                      [0,       1,       0,       0], 
                                      [0,       0,       1, self.dt], 
                                      [0,       0,       0,       1]], dtype=torch.float)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(x, np.ndarray):
            return (self.F_matrix @ torch.tensor(x, dtype=torch.float)).numpy()
        return self.F_matrix @ x
    
    def jacobian(self, x: np.ndarray | torch.Tensor = None) -> np.ndarray:
        return self.F_matrix.numpy()

class Node:
    def __init__(self, node_num: int, node_position: np.ndarray):
        self.node_num = node_num
        self.node_position = node_position
        self.node_classification = node_num % 2
        self.h = HSystem(node_num=self.node_num, alpha=0, node_position=self.node_position)
    
    def observation_model(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self.h(x)


class HSystem:
    def __init__(self, node_num: int, alpha: float, node_position: np.ndarray):
        # Rotation matrix for potential mismatch scenarios
        self.rotation_matrix = torch.tensor([[np.cos(alpha), -1*np.sin(alpha)], 
                                             [np.sin(alpha), np.cos(alpha)]], dtype=torch.float)
        self.node_classification = node_num % 2
        self.node_position = node_position

    def h_dist(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float)
        target_x_location = x[0]
        target_y_location = x[2]
        # Use pure PyTorch operations for autograd compatibility
        dx = target_x_location - self.node_position[0].item()
        dy = target_y_location - self.node_position[1].item()
        return torch.sqrt(dx ** 2 + dy ** 2)

    def h_angle(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float)
        target_x_location = x[0]
        target_y_location = x[2]
        # Use pure PyTorch operations for autograd compatibility
        dx = target_x_location - self.node_position[0].item()
        dy = target_y_location - self.node_position[1].item()
        return torch.atan2(dy, dx)

    def __call__(self, x: np.ndarray | torch.Tensor, n_expansions: int = 0) -> np.ndarray | torch.Tensor:
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            flag = 1

        result = self.node_classification * self.h_dist(x) + (1 - self.node_classification) * self.h_angle(x)
        if flag:
            return result.numpy()
        return result

    def jacobian(self, x: np.ndarray | torch.Tensor, n_expansions: int = 0) -> np.ndarray | torch.Tensor:
        flag = 0
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
            flag = 1
        h_dist_jac = torch.vmap(jacrev(self.h_dist))
        h_angle_jac = torch.vmap(jacrev(self.h_angle))
        node_classification = self.node_classification
        for i in range(n_expansions):
            node_classification = node_classification[:, None]
        result = node_classification * h_dist_jac(x) + (1 - node_classification) * h_angle_jac(x)
        if flag:
            return result.detach().numpy()
        return result.detach()


def plot_trajectory_and_nodes(nodes: np.ndarray, data_points: torch.Tensor, figsize: tuple = (10, 10), 
                               title: str = 'Target Trajectory and Sensor Nodes') -> plt.Figure:
    """
    Plot the target trajectory and sensor nodes on the same figure.
    
    Args:
        nodes: Array of Node objects, shape (1, node_num) or (node_num,)
        data_points: Trajectory data, shape (time_steps, state_dim, 1)
        figsize: Figure size tuple
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    # Handle different node array shapes
    if nodes.ndim == 2:
        node_list = nodes[0]
    else:
        node_list = nodes
    node_num = len(node_list)
    
    # Extract trajectory x, y coordinates
    x_vals = data_points[:, 0, 0].numpy() if isinstance(data_points, torch.Tensor) else data_points[:, 0, 0]
    y_vals = data_points[:, 2, 0].numpy() if isinstance(data_points, torch.Tensor) else data_points[:, 2, 0]
    
    fig = plt.figure(figsize=figsize)
    plt.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1, label='Trajectory')
    
    # Plot nodes with colors based on node type
    for i in range(node_num):
        node = node_list[i]
        node_pos = node.node_position
        node_type = node.node_classification  # 0 = distance, 1 = angle
        x_pos, y_pos = node_pos[0, 0], node_pos[1, 0]
        color = '#e63946' if node_type == 0 else '#457b9d'  # red for distance, blue for angle
        label = 'Distance nodes' if node_type == 0 else 'Angle nodes'
        # Only add label for first occurrence of each type
        plt.plot(x_pos, y_pos, marker='s', markersize=12, color=color, 
                 label=label if i < 2 else None)
        plt.text(x_pos + 0.2, y_pos + 0.2, f'{i}', fontsize=10, ha='left')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # return fig

def generate_data_points(F_system: FSystemLinear, x0: np.ndarray, time_steps: int, state_noise_level: float) -> torch.Tensor:
    state_noise = torch.randn(time_steps, x0.shape[0], x0.shape[1]) * state_noise_level
    data_points = torch.zeros((time_steps, x0.shape[0], x0.shape[1]))
    x_current = torch.tensor(x0, dtype=torch.float)
    for i in range(time_steps):
        x_next = F_system(x_current) + state_noise[i, ...]
        data_points[i, ...] = x_next
        x_current = x_next
    return data_points

def generate_measurements(observation_model: np.ndarray, node_num: int, data_points: torch.Tensor, measurement_noise_level_r: float) -> np.ndarray:
    time_steps = data_points.shape[0]
    observation_noise = np.random.randn(time_steps, node_num) * measurement_noise_level_r
    y_vector = np.zeros((time_steps, node_num))
    h_vector = observation_model
    for i in range(time_steps):
        for j in range(node_num):
            result = h_vector[j](data_points[i, ...])
            # Convert torch tensor to numpy scalar
            y_vector[i, j] = result.item() if isinstance(result, torch.Tensor) else float(result)
        y_vector[i, :] += observation_noise[i, :]
    return y_vector
