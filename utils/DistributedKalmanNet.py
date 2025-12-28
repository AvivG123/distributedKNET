import torch
import pytorch_lightning as pl
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.nn import GCNConv, SimpleConv, MessagePassing, SAGEConv, GATConv
from torch_geometric.data import Data, Batch

torch.set_default_dtype(torch.float)


def loss_function(x_pred, x_true):
    diff_x = x_pred - x_true[..., None, :, :]
    loss = torch.linalg.norm(diff_x, ord=2, dim=(-1, -2)).mean()
    return loss


class StateKnowledge:
    def __init__(self, f_system, signal_dim, q, r_array, x0):
        self.f_system = f_system
        self.signal_dim = signal_dim
        self.q = q
        self.r_array = r_array
        self.x0 = x0


class ModelHyperparameters:
    def __init__(self, hidden_dim, learn_edge_kalman, gcn_layer=None, learning_rate=1e-3):
        self.hidden_dim = hidden_dim
        self.learn_edge_kalman = learn_edge_kalman
        self.gcn_layer = gcn_layer
        self.learning_rate = learning_rate


class DataCharacteristics:
    def __init__(self, graph_number, node_number, time_steps_number, batch_size):
        self.graph_number = graph_number
        self.node_number = node_number
        self.time_steps_number = time_steps_number
        self.batch_size = batch_size


class EdgeKalmanFilter:
    def __init__(self, r_array, signal_dim, measurement_dim=1):
        super(EdgeKalmanFilter, self).__init__()
        self.r_inv = torch.tensor(1 / (r_array ** 2), dtype=torch.float)
        self.signal_dim = signal_dim
        self.measurement_dim = measurement_dim

    def __call__(self, x_pred, measurements, h_system, adj_matrix, node_number):
        r_inv = self.r_inv.repeat(node_number)
        adj_matrix_reshaped = adj_matrix.reshape(-1, node_number, node_number)
        measurements = measurements.reshape(-1, node_number, self.measurement_dim, 1)
        h_transpose_mat = self.calculate_h_mat(h_system, node_number, x_pred)
        y_diff = h_transpose_mat @ r_inv[None, None, :, None, None].float() @ (
                    measurements[:, None, ...].float() - h_system(x_pred)[..., None])
        y_local_delta_unnorm = (adj_matrix_reshaped[..., None, None] * y_diff).sum(2)
        y_local_delta = y_local_delta_unnorm / adj_matrix_reshaped.sum(1)[..., None, None]
        return y_local_delta, h_transpose_mat

    def calculate_h_mat(self, h_system, node_number, x_pred) -> torch.Tensor:
        x_pred_reshaped = x_pred.reshape(-1, self.signal_dim, 1)
        h_transpose_mat = h_system.jacobian(x_pred_reshaped, 1)[:, 0, ...]
        h_transpose_mat = h_transpose_mat.reshape(-1, node_number, node_number, self.signal_dim, 1)
        return h_transpose_mat


class CrossKalmanGain(MessagePassing):
    def __init__(self, node_noise_dim: int, h_mat_dim: int, delta_y_dim: int,
                 hidden_dim: int, out_dim: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.r_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_noise_dim, h_mat_dim),
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * h_mat_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, h_mat_edge, delta_y):
        x = self.r_mlp(x)
        return self.propagate(edge_index, x=x, h_mat_edge=h_mat_edge, delta_y=delta_y)

    def message(self, x_i: Tensor, h_mat_edge: Tensor, delta_y: Tensor) -> Tensor:
        # [E, hidden_dim]
        z = torch.cat([x_i, h_mat_edge], dim=-1)
        m = self.mlp(z)
        m = (m.unsqueeze(-1) @ delta_y.float().unsqueeze(-1)).reshape(x_i.shape[0], -1)
        # m = (h_mat_edge.unsqueeze(-1) @ x_i.unsqueeze(-1) @ delta_y.float().unsqueeze(-1)).reshape(x_i.shape[0], -1)
        return m

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


class NodeKalmanGnnRnn(torch.nn.Module):

    def __init__(self, signal_dim, measurement_dim, output_dim, hidden_dim=16):
        super(NodeKalmanGnnRnn, self).__init__()
        self.fc_delta_y_innov = torch.nn.Sequential(
            torch.nn.Linear(measurement_dim + signal_dim * measurement_dim, hidden_dim, dtype=torch.float),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.float),
            torch.nn.LeakyReLU(),
        )
        self.fc_r = torch.nn.Sequential(
            torch.nn.Linear(measurement_dim, hidden_dim, dtype=torch.float),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.float),
            torch.nn.LeakyReLU(),
        )
        self.r_input_gru = torch.nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.gcn_nodes = GCNConv(
            in_channels=2 * hidden_dim, out_channels=hidden_dim, aggr='mean', normalize=True, add_self_loops=True
        )
        self.fc_signal_features = torch.nn.Sequential(
            torch.nn.Linear(2 * signal_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
        )
        self.sigma_gru = torch.nn.GRU(input_size=hidden_dim, hidden_size=output_dim, num_layers=1, batch_first=True)

        self.fc_node_output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + output_dim, 2 * hidden_dim, dtype=torch.float),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * hidden_dim, hidden_dim, dtype=torch.float),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, output_dim, dtype=torch.float),
        )

    def forward(self, delta_x_features, delta_y_i, y_innov_features, edge_index, hidden_r, pred_sigma):
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
        return node_kalman_output, r_gru_output, hidden_r, pred_sigma[:, 0, ...], edge_index


def calculate_edge_features(delta_y_t, edge_index, node_number):
    graph_index = edge_index[0, :] // node_number
    source_index = edge_index[0, :] % node_number
    target_index = edge_index[1, :] % node_number
    edge_features = delta_y_t[graph_index, source_index, target_index, ...]
    return edge_features


def extract_kalman_features(edge_index, measurements, x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2,
                            y_pred_t_t_1, node_number):
    delta_y_t = measurements.unsqueeze(-2) - y_pred_t_t_1.transpose(2, 1)
    delta_x_hat_t_1 = x_pred_t_1_t_1 - x_pred_t_1_t_2
    delta_x_wave_t_1 = x_pred_t_1_t_1 - x_pred_t_2_t_2
    delta_y_t = delta_y_t.reshape(-1, node_number, node_number, delta_y_t.shape[-1])
    node_list = [i for i in range(node_number)]
    delta_y_t_i = delta_y_t[:, node_list, node_list, ...]
    # only good if same topology is used for all the graphs in the batch
    edge_features = calculate_edge_features(delta_y_t, edge_index, node_number=node_number)
    return delta_x_hat_t_1, delta_x_wave_t_1, delta_y_t, delta_y_t_i, edge_features


class GraphKalmanFilter(torch.nn.Module):

    def __init__(self, f_system, signal_dim, node_kalman_dim, edge_features_dim, r_array,
                 hidden_dim, heads=1, dropout=0.0, learn_edge_kalman=True):
        super(GraphKalmanFilter, self).__init__()
        self.signal_dim = signal_dim
        self.measurement_dim = edge_features_dim
        self.hidden_dim = hidden_dim
        self.node_gnn_rnn = NodeKalmanGnnRnn(
            signal_dim=signal_dim, measurement_dim=edge_features_dim, output_dim=node_kalman_dim,
            hidden_dim=hidden_dim
        )
        self.edge_kalman = EdgeKalmanFilter(r_array, signal_dim)
        self.learn_edge_kalman = learn_edge_kalman
        if learn_edge_kalman:
            self.cross_kalman_gain = CrossKalmanGain(
                node_noise_dim=hidden_dim, h_mat_dim=edge_features_dim * signal_dim, delta_y_dim=edge_features_dim,
                hidden_dim=hidden_dim, out_dim=signal_dim
            )
        self.gcn = SimpleConv(aggr='mean')
        # self.gcn = None
        # self.gcn = GCNConv(in_channels=2*signal_dim, out_channels=signal_dim, aggr='mean')
        self.f = f_system
        self.q = 1

    def forward(self, h_system, measurements, x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2,
                delta_y_innov_i, edge_index, adj_matrix, hidden_r, pred_sigma):
        node_number = measurements.shape[1]
        x_pred_t_t_1, y_pred_t_t_1 = self.prediction_step(x_pred_t_1_t_1, h_system)
        edge_kalman_filter_summed, h_mat = self.edge_kalman(x_pred_t_t_1, measurements, h_system, adj_matrix,
                                                            node_number=node_number)
        h_mat_i, h_mat_edges = self.calculate_h_mat_features(h_mat.transpose(1, 2), edge_index, node_number)
        delta_y_t_i, edge_features, x_features, y_innov_features = self.calculate_features_for_gnn(
            delta_y_innov_i, edge_index, h_mat_i, measurements, node_number, x_pred_t_1_t_1,
            x_pred_t_1_t_2, x_pred_t_2_t_2, y_pred_t_t_1)
        node_kalman, r_gru_output, hidden_r, pred_sigma, edge_index = self.node_gnn_rnn(
            x_features, delta_y_t_i, y_innov_features,
            edge_index, hidden_r.float(), pred_sigma
        )
        node_kalman_reshaped = node_kalman.reshape(-1, node_number, self.signal_dim, self.signal_dim)
        if self.learn_edge_kalman:
            aggregated_cross_kalman = self.cross_kalman_gain(
                r_gru_output, edge_index, h_mat_edge=h_mat_edges, delta_y=edge_features)
            aggregated_cross_kalman_reshaped = aggregated_cross_kalman.reshape(-1, node_number, self.signal_dim, 1)
            phi_pred_t_t = x_pred_t_t_1.float() + node_kalman_reshaped.float() @ aggregated_cross_kalman_reshaped.float()
        else:
            phi_pred_t_t = x_pred_t_t_1.float() + node_kalman_reshaped.float() @ edge_kalman_filter_summed.float()
        # diffusion consensus step
        phi_pred_t_t = phi_pred_t_t.reshape(-1, self.signal_dim, 1)
        if self.gcn is None:
            # no graph convolution: use phi_pred_t_t directly
            x_pred_t_t = phi_pred_t_t
        else:
            x_pred_t_t = self.gcn(phi_pred_t_t[..., 0], edge_index)
            # x_pred_t_t = self.gcn(torch.cat((phi_pred_t_t[..., 0], h_mat_i), dim=-1), edge_index)
        return x_pred_t_t.reshape(x_pred_t_t_1.shape), x_pred_t_t_1, edge_index, hidden_r, pred_sigma

    def calculate_features_for_gnn(self, delta_y_innov_i, edge_index, h_mat_i, measurements, node_number,
                                   x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, y_pred_t_t_1: Tensor) -> tuple[
        Tensor, Tensor, Tensor, Tensor]:
        delta_x_hat_t_1, delta_x_wave_t_1, _, delta_y_t_i, edge_features = extract_kalman_features(
            edge_index, measurements, x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, y_pred_t_t_1, node_number
        )
        x_features = torch.cat([delta_x_hat_t_1[..., 0], delta_x_wave_t_1[..., 0]], dim=-1)
        x_features = x_features.reshape(-1, 2 * self.signal_dim).float()
        delta_y_t_i = delta_y_t_i.reshape(-1, self.measurement_dim).float()
        delta_y_innov_i = delta_y_innov_i.reshape(-1, self.measurement_dim).float()
        # all_edge_features = torch.cat([edge_features, h_mat_edges], dim=-1)
        y_innov_features = torch.cat([delta_y_innov_i, h_mat_i], dim=-1).float()
        return delta_y_t_i, edge_features, x_features, y_innov_features

    def calculate_h_mat_features(self, h_mat, edge_index, node_number):
        node_list = [i for i in range(node_number)]
        h_mat_edges = calculate_edge_features(h_mat, edge_index, node_number)[..., 0]
        h_mat_i = h_mat[:, node_list, node_list, ...]
        h_mat_i = h_mat_i.reshape(-1, self.signal_dim * self.measurement_dim)
        return h_mat_i, h_mat_edges

    def prediction_step(self, x_pred_t_1_t_1, h_system):
        x_pred_t_t_1 = self.f(x_pred_t_1_t_1)
        y_pred_t_t_1 = torch.Tensor(h_system(x_pred_t_t_1))
        x_pred_t_t_1 = torch.Tensor(x_pred_t_t_1)
        return x_pred_t_t_1, y_pred_t_t_1


class GraphKalmanProcess(pl.LightningModule):
    def __init__(self, f_system, signal_dim, edge_features_dim, node_kalman_dim, edge_kalman_dim, r_array,
                 hidden_dim=32, heads=1, dropout=0.0,
                 lr=1e-3, learn_edge_kalman=True, x0_scale=10):
        super(GraphKalmanProcess, self).__init__()
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim
        self.node_kalman_dim = node_kalman_dim
        self.edge_kalman_dim = edge_kalman_dim
        self.r_array = r_array
        self.x0_scale = x0_scale
        self.gkf = GraphKalmanFilter(
            f_system, signal_dim, node_kalman_dim,
            edge_features_dim, r_array, hidden_dim, heads, dropout, learn_edge_kalman
        )
        self.loss = torch.nn.MSELoss()
        self.lr = lr

    def forward(self, data, x_0: torch.Tensor = None):
        if isinstance(data, Batch):
            graph_number = len(data)
            h_system = data[0].h_system
        else:
            graph_number = 1
            h_system = data.h_system
        node_number = data.num_nodes // graph_number
        measurements_shape = (graph_number, node_number,) + data.x.shape[1:]  # (BATCH, NODE_NUM, Time_steps, 1)
        x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, delta_y_innov_i, edge_index, hidden_r, pred_sigma = self.initiate_graph_kalman_parameters(
            x_0, data, measurements_shape)
        measurements = data.x.reshape(*measurements_shape)
        time_steps_number = measurements.shape[-2]
        x_pred_t = torch.zeros(graph_number, time_steps_number, node_number, self.signal_dim, 1)
        for i in range(measurements.shape[2]):
            if i > 0:
                delta_y_innov_i = measurements[:, :, i, ...] - measurements[:, :, i - 1, ...]
            x_pred_t_t, x_pred_t_t_1, edge_index, hidden_r, pred_sigma = self.gkf(
                h_system, measurements[:, :, i, ...], x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, delta_y_innov_i,
                edge_index, data.adj_matrix, hidden_r, pred_sigma)
            x_pred_t[:, i, ...] = x_pred_t_t
            x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2 = x_pred_t_t, x_pred_t_t_1, x_pred_t_1_t_1
        return x_pred_t

    def initiate_graph_kalman_parameters(self, x_0: torch.Tensor, data: Data, measurement_shape: tuple):
        batch_size, node_number = measurement_shape[0], measurement_shape[1]
        if x_0 is None:
            x_0 = self.x0_scale * torch.ones(
                batch_size, node_number, self.signal_dim, 1, dtype=torch.float)  # (batch, node_number, 2, 1)
        edge_index = data.edge_index
        delta_y_innov_i = torch.zeros(measurement_shape, dtype=torch.float)[:, :, 0, ...]
        # todo: check whether the init of the hidden state are necessary
        hidden_r = torch.randn((1, data.num_nodes, self.hidden_dim),
                               dtype=torch.float)  # (1, node_number, hidden_dim)
        pred_sigma = torch.randn((data.num_nodes, self.signal_dim * self.signal_dim),
                                 dtype=torch.float)  # (node_number, output_dim)
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        with torch.no_grad():
            for param_group in self.parameters():
                if param_group.grad is not None:
                    param_group.data = torch.nan_to_num(param_group.data, nan=0.0)
