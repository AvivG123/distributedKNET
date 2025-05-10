# implement ST_GAT Model using  pytorch_lightning
import torch
import numpy as np
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, SimpleConv, GATConv
from torch_geometric.data import Data
from torch.nn.functional import normalize
from utils.DistributedKalmanData import HSystem, FSystem, CreateGraph
torch.set_default_dtype(torch.float64)


class NodeBatchNorm(torch.nn.Module):
    def __init__(self, node_number, feature_dim):
        super(NodeBatchNorm, self).__init__()
        # Initialize learnable scale (gamma) and shift (beta) parameters for each feature
        self.node_number = node_number
        self.gamma = torch.nn.Parameter(torch.ones(node_number, feature_dim))  # Shape: (node_number, feature_dim)
        self.beta = torch.nn.Parameter(torch.zeros(node_number, feature_dim))  # Shape: (node_number, feature_dim)

    def forward(self, x):
        # x shape: (batch_size * node_number, feature_dim) reshape to (node_number, batch_size, feature_dim)
        x = x.reshape(-1, self.node_number, x.shape[-1])
        # Compute mean and variance for each node across the batch dimension
        mean = x.mean(dim=0)  # Shape: (node_number, feature_dim)
        var = x.var(dim=0, unbiased=False)  # Shape: (node_number, feature_dim)
        # Normalize the tensor (apply batch normalization per node)
        x_norm = (x - mean) / torch.sqrt(var + 1e-10)  # Avoid division by zero
        # Scale and shift the normalized tensor using gamma and beta
        x_scaled = self.gamma * x_norm + self.beta
        x_scaled = x_scaled.reshape(-1, x.shape[-1])
        return x_scaled.float()


def edge_kalman_gain(h_system, r_array, j_matrix, x_pred, measurements):
    H_T = h_system.jacobian(x_pred, 1)[:, 0, ...]
    r_inv = 1 / (r_array ** 2)
    y_node = H_T @ r_inv[None, :, None, None] @ measurements[None, ..., None]
    y_pred = H_T @ r_inv[None, :, None, None] @ h_system.func(x_pred)[..., None]
    y_diff = y_node - y_pred
    y_local_delta_unnorm = (j_matrix[..., None, None] * y_diff).sum(1)
    y_local_delta = y_local_delta_unnorm / j_matrix.sum(0)[..., None, None]
    return y_local_delta


class EdgeKalmanFilter:
    def __init__(self, h_system, node_num, r_array, signal_dim, measurement_dim=1):
        super(EdgeKalmanFilter, self).__init__()
        self.h_system = h_system
        self.r_inv = torch.tensor(1 / (r_array ** 2), dtype=torch.float64)
        self.node_num = node_num
        self.signal_dim = signal_dim
        self.measurement_dim = measurement_dim

    def __call__(self, x_pred, measurements, adj_matrix):
        x_pred_reshaped = x_pred.reshape(-1, self.signal_dim, 1)
        adj_matrix_reshaped = adj_matrix.reshape(-1, self.node_num, self.node_num)
        measurements = measurements.reshape(-1, self.node_num, self.measurement_dim, 1)
        h_transpose_mat = self.h_system.jacobian(x_pred_reshaped, 1)[:, 0, ...]
        h_transpose_mat = h_transpose_mat.reshape(-1, self.node_num, self.node_num, self.signal_dim, 1)
        y_node = h_transpose_mat @ self.r_inv[None, None, :, None, None].float() @ measurements[:, None, ...].float()
        y_pred = h_transpose_mat @ self.r_inv[None, None, :, None, None].float() @ self.h_system.func(x_pred)[..., None]
        y_diff = y_node - y_pred
        y_local_delta_unnorm = (adj_matrix_reshaped[..., None, None] * y_diff).sum(2)
        y_local_delta = y_local_delta_unnorm / adj_matrix_reshaped.sum(1)[..., None, None]
        return y_local_delta, h_transpose_mat


class GlobalKalmanFilter:
    def __init__(self, h_system, node_num, r_array, signal_dim):
        super(GlobalKalmanFilter, self).__init__()
        self.h_system = h_system
        self.r_inv = torch.tensor(1 / (r_array ** 2), dtype=torch.float64)
        self.node_num = node_num
        self.signal_dim = signal_dim

    def __call__(self, x_pred, p_mat, adj_matrix):
        x_pred_reshaped = x_pred.reshape(-1, self.signal_dim, 1)
        adj_matrix_reshaped = adj_matrix.reshape(-1, self.node_num, self.node_num)
        h_transpose_mat = self.h_system.jacobian(x_pred_reshaped, 1)[:, 0, ...]
        h_transpose_mat = h_transpose_mat.reshape(-1, self.node_num, self.node_num, self.signal_dim, 1)
        node_s_all = h_transpose_mat @ self.r_inv[None, None, :, None, None] @ h_transpose_mat.transpose(-1, -2)
        s_local_all = (adj_matrix_reshaped[..., None, None] * node_s_all).sum(2) / adj_matrix_reshaped.sum(1)[..., None, None]
        m_mat = torch.linalg.inv(torch.linalg.inv(p_mat) + s_local_all)
        return m_mat


class NodeKalmanGnnRnn(torch.nn.Module):
    # Todo: Dig into the hidden_dim for each layer
    # Todo: add gat layer instead of gcn with edge features
    # Todo: add h function jacobian as feature
    def __init__(self, signal_dim, measurement_dim, output_dim, hidden_dim=16, heads=4, dropout=0.0):
        super(NodeKalmanGnnRnn, self).__init__()
        # input for the gnn + GCN layer(gat?)
        self.fc_delta_y_innov = torch.nn.Sequential(
            torch.nn.Linear(measurement_dim + signal_dim * measurement_dim, hidden_dim, dtype=torch.float64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
            torch.nn.LeakyReLU(),
        )
        self.fc_r = torch.nn.Sequential(
            torch.nn.Linear(measurement_dim, hidden_dim, dtype=torch.float64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
            torch.nn.LeakyReLU(),
        )
        self.r_input_gru = torch.nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.gcn_nodes = GCNConv(
            in_channels=2 * hidden_dim, out_channels=hidden_dim, aggr='mean', normalize=True, add_self_loops=True
        )
        # self.gcn_nodes = GATConv(
        #     in_channels=2 * hidden_dim, out_channels=hidden_dim, heads=heads, dropout=dropout,
        #     concat=True, edge_dim=measurement_dim + signal_dim * measurement_dim)
        self.fc_signal_features = torch.nn.Sequential(
            torch.nn.Linear(2 * signal_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )
        self.sigma_gru = torch.nn.GRU(input_size=hidden_dim, hidden_size=output_dim, num_layers=1, batch_first=True)

        self.fc_node_output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + output_dim, 2 * hidden_dim, dtype=torch.float64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * hidden_dim, hidden_dim, dtype=torch.float64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, output_dim, dtype=torch.float64),
        )

    def forward(self, delta_x_features, delta_y_i, y_innov_features, edge_features, edge_index, hidden_r, pred_sigma):
        delta_y_innov_i = self.fc_delta_y_innov(y_innov_features)
        r_gru_input = self.fc_r(delta_y_i)
        r_gru_output, hidden_r = self.r_input_gru(r_gru_input.unsqueeze(1), hidden_r)
        r_gru_output = r_gru_output[:, 0, ...]
        gnn_features = torch.cat([delta_y_innov_i, r_gru_output], dim=-1)
        node_output_features = self.gcn_nodes(gnn_features, edge_index)
        node_kalman_input = torch.cat([node_output_features, pred_sigma.float()], dim=-1)
        node_kalman_output = self.fc_node_output(node_kalman_input)
        delta_x_features = self.fc_signal_features(delta_x_features)
        pred_sigma, _ = self.sigma_gru(delta_x_features.unsqueeze(1), node_kalman_output.unsqueeze(0))
        return node_kalman_output, hidden_r, pred_sigma[:, 0, ...], edge_index


class GraphKalmanFilter(torch.nn.Module):
    def __init__(self, node_num, h_system, f_system, signal_dim, node_kalman_dim, edge_features_dim, r_array, hidden_dim,
                 heads=1, dropout=0.0):
        super(GraphKalmanFilter, self).__init__()
        self.node_num = node_num
        self.signal_dim = signal_dim
        self.measurement_dim = edge_features_dim
        self.hidden_dim = hidden_dim
        self.gat_rnn = NodeKalmanGnnRnn(
            signal_dim=signal_dim, measurement_dim=edge_features_dim, output_dim=node_kalman_dim,
            hidden_dim=hidden_dim, heads=heads, dropout=dropout
        )
        self.norm_x_features = NodeBatchNorm(node_num, 2 * signal_dim)
        self.norm_y_innov = NodeBatchNorm(node_num, edge_features_dim + signal_dim * edge_features_dim)
        self.norm_y_i_features = NodeBatchNorm(node_num, edge_features_dim)
        self.edge_kalman = EdgeKalmanFilter(h_system, node_num, r_array, signal_dim)
        self.global_kalman = GlobalKalmanFilter(h_system, node_num, r_array, signal_dim)
        self.gcn = SimpleConv(aggr='mean')
        self.h = h_system.func
        self.f = f_system.func
        self.f_jac = f_system.jacobian
        self.q = 1

    def forward(self, measurements, x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, p_mat,
                delta_y_innov_i, edge_index, adj_matrix, hidden_r, pred_sigma):
        # Todo: Check the dims of the edge aggregations
        # Todo: normalize features
        hidden_r = hidden_r.float()
        x_pred_t_t_1, y_pred_t_t_1 = self.prediction_step(x_pred_t_1_t_1)
        delta_x_hat_t_1, delta_x_wave_t_1, delta_y_t, delta_y_t_i, edge_features = self.extract_kalman_features(
            edge_index, measurements, x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, y_pred_t_t_1
        )
        # concat delta_x_hat_t_1, delta_x_wave_t_1
        x_features = torch.cat([delta_x_hat_t_1[..., 0], delta_x_wave_t_1[..., 0]], dim=-1)
        x_features = self.norm_x_features(x_features)
        x_features = x_features.reshape(-1, 2 * self.signal_dim)
        delta_y_t_i = delta_y_t_i.reshape(-1, self.measurement_dim)
        delta_y_t_i = self.norm_y_i_features(delta_y_t_i)
        delta_y_innov_i = delta_y_innov_i.reshape(-1, self.measurement_dim)
        delta_y_innov_i = normalize(delta_y_innov_i, p=2, dim=-1, out=None)
        edge_kalman_filter_summed, h_mat = self.edge_kalman(x_pred_t_t_1, measurements, adj_matrix)
        h_mat_i, h_mat_edges = self.calculate_h_mat_features(h_mat, edge_index)
        all_edge_features = torch.cat([edge_features, h_mat_edges], dim=-1)
        y_innov_features = torch.cat([delta_y_innov_i, h_mat_i], dim=-1)
        y_innov_features = self.norm_y_innov(y_innov_features)
        node_kalman, hidden_r, pred_sigma, edge_index = self.gat_rnn(
            x_features, delta_y_t_i, y_innov_features, all_edge_features, edge_index, hidden_r, pred_sigma
        )
        # global_kalman_gain = self.global_kalman(x_pred_t_t_1, p_mat, adj_matrix)
        node_kalman_reshaped = node_kalman.reshape(-1, self.node_num, self.signal_dim, self.signal_dim)
        phi_pred_t_t = x_pred_t_t_1.float() + node_kalman_reshaped.float() @ edge_kalman_filter_summed.float()
        # diffusion consensus step
        phi_pred_t_t = phi_pred_t_t.reshape(-1, self.signal_dim, 1)
        # print(phi_pred_t_t.shape)
        x_pred_t_t = self.gcn(phi_pred_t_t[..., 0], edge_index)
        # p_mat = self.calculate_p_mat(x_pred_t_t, global_kalman_gain, adj_matrix)
        return x_pred_t_t.reshape(x_pred_t_t_1.shape), x_pred_t_t_1, p_mat, edge_index, hidden_r, pred_sigma

    def calculate_kalman_gains(self, delta_y_t, edge_index, edge_kalman, node_kalman, batch_size):
        edge_kalman_delta_y = torch.zeros((batch_size, self.node_num, self.node_num, self.signal_dim, 1))
        edge_features = self.calculate_edge_features(delta_y_t, edge_index)
        edge_mul = edge_kalman[..., None] @ edge_features[..., None]
        edge_kalman_delta_y[
            edge_index[0, :] // self.node_num, edge_index[0, :] % self.node_num,
            edge_index[1, :] % self.node_num, ...
        ] = edge_mul
        node_kalman_reshaped = node_kalman.reshape(-1, self.node_num, self.signal_dim, self.signal_dim)
        return edge_kalman_delta_y, node_kalman_reshaped

    def extract_kalman_features(self, edge_index, measurements, x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2,
                                y_pred_t_t_1):
        delta_y_t = (measurements.unsqueeze(0) - y_pred_t_t_1[..., 0]).transpose(1, 0)
        delta_x_hat_t_1 = x_pred_t_1_t_1 - x_pred_t_1_t_2
        delta_x_wave_t_1 = x_pred_t_1_t_1 - x_pred_t_2_t_2
        delta_y_t = delta_y_t.reshape(-1, self.node_num, self.node_num, delta_y_t.shape[-1])
        node_list = [i for i in range(self.node_num)]
        delta_y_t_i = delta_y_t[:, node_list, node_list, ...]
        # only good if same topology is used for all the graphs in the batch
        edge_features = self.calculate_edge_features(delta_y_t, edge_index)
        return delta_x_hat_t_1, delta_x_wave_t_1, delta_y_t, delta_y_t_i, edge_features

    def calculate_edge_features(self, delta_y_t, edge_index):
        graph_index = edge_index[0, :] // self.node_num
        source_index = edge_index[0, :] % self.node_num
        target_index = edge_index[1, :] % self.node_num
        edge_features = delta_y_t[graph_index, source_index, target_index, ...]
        return edge_features

    def calculate_h_mat_features(self, h_mat, edge_index):
        node_list = [i for i in range(self.node_num)]
        h_mat_edges = self.calculate_edge_features(h_mat, edge_index)[..., 0]
        h_mat_edges = normalize(h_mat_edges, p=2, dim=-1, out=None)
        h_mat_i = h_mat[:, node_list, node_list, ...]
        h_mat_i = h_mat_i.reshape(-1, self.signal_dim * self.measurement_dim)
        # h_mat_i = normalize(h_mat_i, p=2, dim=-1, out=None)
        return h_mat_i, h_mat_edges

    def prediction_step(self, x_pred_t_1_t_1):
        x_pred_t_t_1 = self.f(x_pred_t_1_t_1)
        y_pred_t_t_1 = torch.Tensor(self.h(x_pred_t_t_1, 3))
        x_pred_t_t_1 = torch.Tensor(x_pred_t_t_1)
        return x_pred_t_t_1, y_pred_t_t_1

    def calculate_p_mat(self, x_pred_t_t, m_mat, adj_matrix):
        f_mat = self.f_jac(x_pred_t_t.reshape(-1, self.signal_dim))
        f_mat = f_mat.reshape(-1, self.node_num, self.signal_dim, self.signal_dim)
        adj_matrix = adj_matrix.reshape(-1, self.node_num, self.node_num)
        m_mat = m_mat.reshape(-1, self.node_num, self.signal_dim, self.signal_dim)
        p_mat = f_mat @ m_mat @ f_mat.transpose(3, 2) + adj_matrix.sum(axis=1)[..., None, None] * self.q ** 2 * torch.eye(2)
        return p_mat


def loss_function(x_pred, x_true):
    diff_x = x_pred - x_true[..., None, :, :]
    loss = torch.linalg.norm(diff_x, ord=2, dim=(-1, -2)).mean()
    return loss


class GraphKalmanProcess(pl.LightningModule):
    # This class take the GraphKalmanFilter module and run the process of it for a given number of time steps.
    # then calculate the loss of the mse error between the predicted values and the ground truth values over
    # the entire trajectory.
    def __init__(self, node_num, h_system, f_system, signal_dim, node_feature_dim, node_output_dim,
                 edge_features_dim, node_kalman_dim, edge_kalman_dim, r_array, hidden_dim=32, heads=1, dropout=0.0,
                 lr=5e-3):
        super(GraphKalmanProcess, self).__init__()
        self.node_num = node_num
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim
        self.node_kalman_dim = node_kalman_dim
        self.edge_kalman_dim = edge_kalman_dim
        self.r_array = r_array
        self.gkf = GraphKalmanFilter(
            node_num, h_system, f_system, signal_dim, node_kalman_dim,
            edge_features_dim, r_array, hidden_dim, heads, dropout
        )
        self.loss = torch.nn.MSELoss()
        self.lr = lr

    def forward(self, data, x_0: torch.Tensor = None):
        measurements_shape = (len(data), self.node_num,) + data.x.shape[1:]  # (BATCH, NODE_NUM, Time_steps, 1)
        x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, delta_y_innov_i, edge_index, hidden_r, pred_sigma = self.initiate_graph_kalman_parameters(
            x_0, data, measurements_shape)
        measurements = data.x.reshape(*measurements_shape)
        time_steps_number = measurements.shape[-2]
        x_pred_t = torch.zeros(data.num_graphs, time_steps_number, self.node_num, self.signal_dim, 1)
        p_mat = torch.eye(self.signal_dim, dtype=torch.float)
        p_mat = p_mat[None, None, ...].repeat(len(data), self.node_num, 1, 1)
        for i in range(measurements.shape[2]):
            if i > 0:
                delta_y_innov_i = measurements[:, :, i, ...] - measurements[:, :, i - 1, ...]
            x_pred_t_t, x_pred_t_t_1, p_mat, edge_index, hidden_r, pred_sigma = self.gkf(
                measurements[:, :, i, ...], x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, p_mat, delta_y_innov_i,
                edge_index, data.adj_matrix, hidden_r, pred_sigma)
            x_pred_t[:, i, ...] = x_pred_t_t
            x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2 = x_pred_t_t, x_pred_t_t_1, x_pred_t_1_t_1
        return x_pred_t

    def initiate_graph_kalman_parameters(self, x_0: torch.Tensor, data: Data, measurement_shape: tuple):
        # todo: check the dims of the hidden_r and pred_sigma
        if x_0 is None:
            x_0 = torch.ones(len(data), self.node_num, self.signal_dim, 1, dtype=torch.float64)  # (batch, node_number, 2, 1)
        edge_index = data.edge_index
        delta_y_innov_i = torch.zeros(measurement_shape, dtype=torch.float64)[:, :, 0, ...]
        hidden_r = torch.randn((1, data.num_nodes, self.hidden_dim), dtype=torch.float64)  # (1, node_number, hidden_dim)
        pred_sigma = torch.randn((data.num_nodes, self.signal_dim * self.signal_dim), dtype=torch.float64)  # (node_number, output_dim)
        x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2 = x_0, x_0, x_0  # (batch, node_number, 2, 1)
        return x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, delta_y_innov_i, edge_index, hidden_r, pred_sigma

    def _shared_step(self, batch, batch_idx, mode='train'):
        x_true = batch.y.reshape(batch.num_graphs, -1, self.signal_dim, 1)
        x_pred = self(batch)
        loss = loss_function(x_pred, x_true)
        self.log(f'{mode}_loss:', loss, batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [scheduler]


def main():
    node_kalman_net = NodeKalmanGnnRnn(signal_dim=2, measurement_dim=1, output_dim=16)
    delta_y_innov_i = torch.randn(8, 1)
    delta_y_i = torch.randn(8, 1)
    delta_x_features = torch.randn(8, 4)
    # edge index for 8 nodes
    graph = CreateGraph(8)
    graph_edges = torch.tensor(np.array(graph.graph.edges()).T, dtype=torch.int64)
    print(graph_edges)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])
    hidden_r = torch.randn(1, 8, 16)
    hidden_sigma = torch.randn(8, 16)
    node_kalman_output, hidden_r, pred_sigma, edge_index = node_kalman_net(
        delta_x_features, delta_y_i, delta_y_innov_i, edge_index, hidden_r, hidden_sigma)
    print("forward pass successful for NodeKalmanGnnRnn")
    h_sys = HSystem(node_num=8)
    f_sys = FSystem()
    r_array = torch.randn(8)
    # j matrix is the adjacency matrix of the graph defined by the edge index
    j_matrix = torch.tensor(graph.adj_matrix)
    x_pred = torch.randn(8, 2, 1)
    edge_gain = edge_kalman_gain(h_sys, r_array, j_matrix, x_pred, delta_y_innov_i)
    print("edge kalman gain successful")
    x_pred_filter = torch.randn(2, 8, 2, 1)
    pred_sigma = torch.randn(16, 16)
    hidden_r = torch.randn(1, 16, 16)
    meas = torch.randn(2, 8, 1)
    node_kalman_filter = GraphKalmanFilter(
        node_num=8, h_system=h_sys, f_func=f_sys.func, signal_dim=2, node_kalman_dim=2, edge_features_dim=1,
        r_array=r_array, hidden_dim=16)
    node_kalman_filter(meas, x_pred_filter, x_pred_filter, x_pred_filter, meas, edge_index, j_matrix, hidden_r,
                       pred_sigma)
    kalman_process = GraphKalmanProcess(8, h_sys, f_sys.func, signal_dim=2, node_feature_dim=4, node_output_dim=16,
                                        edge_features_dim=1, node_kalman_dim=4, edge_kalman_dim=2, hidden_dim=16, heads=1, lr=0.0005, r_array=r_array)
    print("hyde")


if __name__ == '__main__':
    main()