"""Utilities: Graph Kalman network models and helpers.

This module contains PyTorch / PyTorch-Geometric model components used by
the distributed KalmanNet experiments.
"""

from dataclasses import dataclass

import torch
import pytorch_lightning as pl
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.nn import GCNConv, SimpleConv, MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax

torch.set_default_dtype(torch.float)

ADAPTIVE_CONSENSUS_FEATURES = ("delta_y_innov", "delta_y", "h_mat_i", "x_hat", "delta_x_hat_t_1")


def loss_function(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Compute batched L2 norm loss between predictions and truth.

    Shapes expected: x_pred (B, T, N, D, 1), x_true (B, T, N, D, 1) or
    compatible broadcastable shapes used by the project.
    """
    diff_x = x_pred - x_true[..., None, :, :]
    loss = torch.linalg.norm(diff_x, ord=2, dim=(-1, -2)).mean()
    return loss


@dataclass
class StateKnowledge:
    f_system: object
    signal_dim: int
    q: float
    r_array: object
    x0: object


@dataclass
class ModelHyperparameters:
    hidden_dim: int
    learn_edge_kalman: bool
    gcn_layer: str | None = None
    learning_rate: float = 1e-3


@dataclass
class DataCharacteristics:
    graph_number: int
    node_number: int
    time_steps_number: int
    batch_size: int


class EdgeKalmanFilter:
    def __init__(self, r_array, signal_dim, measurement_dim=1):
        # keep r_inv as a torch tensor for device-safe use
        self.r_inv = torch.as_tensor(1 / (r_array ** 2), dtype=torch.float)
        self.signal_dim = signal_dim
        self.measurement_dim = measurement_dim

    def __call__(self, x_pred, measurements, h_system, adj_matrix, node_number):
        r_inv = self.r_inv.repeat(node_number).to(device=x_pred.device)
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
        # Ensure jacobian output is a torch tensor on the correct device
        h_transpose_mat = h_system.jacobian(x_pred_reshaped, 1)[:, 0, ...]
        if not isinstance(h_transpose_mat, torch.Tensor):
            h_transpose_mat = torch.tensor(h_transpose_mat, dtype=torch.float, device=x_pred.device)
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
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, h_mat_edge, delta_y):
        x = self.r_mlp(x)
        return self.propagate(edge_index, x=x, h_mat_edge=h_mat_edge, delta_y=delta_y)

    def message(self, x_i: Tensor, h_mat_edge: Tensor, delta_y: Tensor) -> Tensor:
        # [E, hidden_dim]
        z = torch.cat([x_i, h_mat_edge], dim=-1)
        m = self.mlp(z)
        # Robust multiplication: compute outer-product-like interaction and then flatten
        dy = delta_y.float().unsqueeze(-1) if delta_y.dim() == 1 else delta_y.float()
        # m: [E, out_dim], dy: [E, delta_dim] -> create [E, out_dim, delta_dim] then flatten
        m = (m.unsqueeze(-1) * dy.unsqueeze(1)).reshape(x_i.shape[0], -1)
        return m

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


class AdaptiveMeanConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # we implement weighting ourselves

        # edge scoring network (from features)
        self.att_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

        # 🔥 initialize to zero => uniform attention
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Non-degenerate init is required; zero init freezes the whole MLP
        # because ReLU blocks gradient flow through the attention stack.
        for module in self.att_mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, feat=None):
        if feat is None:
            feat = x
        return self.propagate(edge_index, x=x, feat=feat)

    def message(self, x_j, feat_i, feat_j, index):
        # edge features
        edge_feat = torch.cat([feat_i, feat_j], dim=-1)
        # attention logits
        alpha = self.att_mlp(edge_feat).squeeze(-1)
        # normalize per destination node
        alpha = softmax(alpha, index)
        # transform messages
        return alpha.unsqueeze(-1) * x_j

    def update(self, aggr_out):
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

    def _flatten_recurrent_weights(self) -> None:
        # cuDNN expects GRU weights to be in a compact layout; flattening keeps
        # the optimized path active after device moves or checkpoint restores.
        self.r_input_gru.flatten_parameters()
        self.sigma_gru.flatten_parameters()

    def forward(self, delta_x_features, delta_y_i, y_innov_features, edge_index, hidden_r, pred_sigma):
        self._flatten_recurrent_weights()
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
    graph_idx = edge_index[0, :] // node_number
    src_idx = edge_index[0, :] % node_number
    dst_idx = edge_index[1, :] % node_number
    return delta_y_t[graph_idx, src_idx, dst_idx, ...]


def extract_kalman_features(edge_index, measurements, x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2,
                            y_pred_t_t_1, node_number):
    delta_y_t = measurements.unsqueeze(-2) - y_pred_t_t_1.transpose(2, 1)
    delta_x_hat_t_1 = x_pred_t_1_t_1 - x_pred_t_1_t_2
    delta_x_wave_t_1 = x_pred_t_1_t_1 - x_pred_t_2_t_2
    delta_y_t = delta_y_t.reshape(-1, node_number, node_number, delta_y_t.shape[-1])
    delta_y_t_i = delta_y_t[:, range(node_number), range(node_number), ...]
    # only good if same topology is used for all the graphs in the batch
    edge_features = calculate_edge_features(delta_y_t, edge_index, node_number=node_number)
    return delta_x_hat_t_1, delta_x_wave_t_1, delta_y_t, delta_y_t_i, edge_features


def _normalize_consensus_layer(consensus_layer: str | None) -> str | None:
    return None if consensus_layer is None else str(consensus_layer).strip().lower()


def _build_consensus_layer(consensus_layer: str | None, signal_dim, measurement_dim):
    _consensus_layer = _normalize_consensus_layer(consensus_layer)
    # no-op / identity
    if _consensus_layer in {None, "", "none", "identity"}:
        return None
    # simple mean aggregation
    elif _consensus_layer in {"simple", "simpleconv"}:
        return SimpleConv(aggr="mean")
    # graph convolutional consensus (keeps signal dim)
    elif _consensus_layer in {"gcn", "gcnconv"}:
        return GCNConv(
            in_channels=signal_dim,
            out_channels=signal_dim,
            aggr="mean",
            normalize=True,
            add_self_loops=True,
        )
    # adaptive mean attention-based consensus requires an input feature dimension
    elif _consensus_layer in {"adaptive", "adaptive_mean"}:
        return AdaptiveMeanConv(
            in_channels=2 * measurement_dim + signal_dim * measurement_dim + 2 * signal_dim,
            out_channels=signal_dim,
        )
    else:
        raise ValueError(
            f"Unsupported consensus_layer={consensus_layer!r}; use 'none', 'simple', 'gcn' or 'adaptive'."
        )


def _flatten_node_signal(tensor: Tensor, signal_dim: int) -> Tensor:
    return tensor.reshape(-1, signal_dim).float()


def _flatten_measurement(tensor: Tensor, measurement_dim: int) -> Tensor:
    return tensor.reshape(-1, measurement_dim).float()


class GraphKalmanFilter(torch.nn.Module):

    def __init__(self, f_system, signal_dim, node_kalman_dim, edge_features_dim, r_array,
                 hidden_dim, heads=1, dropout=0.0, learn_edge_kalman=True, consensus_layer: str | None = "none"):
        super(GraphKalmanFilter, self).__init__()
        self.signal_dim = signal_dim
        self.measurement_dim = edge_features_dim
        self.hidden_dim = hidden_dim
        self.consensus_layer = _normalize_consensus_layer(consensus_layer)
        self.node_gnn_rnn = NodeKalmanGnnRnn(
            signal_dim=signal_dim, measurement_dim=edge_features_dim, output_dim=node_kalman_dim, hidden_dim=hidden_dim
        )
        self.edge_kalman = EdgeKalmanFilter(r_array, signal_dim)
        self.learn_edge_kalman = learn_edge_kalman
        if learn_edge_kalman:
            self.cross_kalman_gain = CrossKalmanGain(
                node_noise_dim=hidden_dim, h_mat_dim=edge_features_dim * signal_dim, delta_y_dim=edge_features_dim,
                hidden_dim=hidden_dim, out_dim=signal_dim
            )
        self.gcn = _build_consensus_layer(self.consensus_layer, signal_dim, edge_features_dim)
        self.f = f_system
        self.q = 1

    def forward(self, h_system, measurements, x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2,
                delta_y_innov_i, edge_index, adj_matrix, hidden_r, pred_sigma):
        node_number = measurements.shape[1]
        x_pred_t_t_1, y_pred_t_t_1 = self.prediction_step(x_pred_t_1_t_1, h_system)

        edge_kalman_filter_summed, h_mat = self.edge_kalman(x_pred_t_t_1, measurements, h_system, adj_matrix,
                                                            node_number=node_number)
        h_mat_i, h_mat_edges = self.calculate_h_mat_features(h_mat.transpose(1, 2), edge_index, node_number)

        delta_y_t_i, edge_features, x_features, y_innov_features, delta_x_hat_t_1 = self.calculate_features_for_gnn(
            delta_y_innov_i, edge_index, h_mat_i, measurements, node_number, x_pred_t_1_t_1,
            x_pred_t_1_t_2, x_pred_t_2_t_2, y_pred_t_t_1)

        node_kalman, r_gru_output, hidden_r, pred_sigma, edge_index = self.node_gnn_rnn(
            x_features, delta_y_t_i, y_innov_features, edge_index, hidden_r.float(), pred_sigma
        )
        node_kalman_reshaped = node_kalman.reshape(-1, node_number, self.signal_dim, self.signal_dim)
        if self.learn_edge_kalman:
            cross_kalman = self.cross_kalman_gain(
                r_gru_output, edge_index, h_mat_edge=h_mat_edges, delta_y=edge_features)
            cross_kalman_reshaped = cross_kalman.reshape(-1, node_number, self.signal_dim, 1)
            phi_pred_t_t = x_pred_t_t_1.float() + node_kalman_reshaped.float() @ cross_kalman_reshaped.float()
        else:
            phi_pred_t_t = x_pred_t_t_1.float() + node_kalman_reshaped.float() @ edge_kalman_filter_summed.float()
        x_pred_t_t = self._apply_consensus(
            phi_pred_t_t,
            edge_index,
            delta_y_innov_i,
            delta_y_t_i,
            h_mat_i,
            delta_x_hat_t_1,
        )
        return x_pred_t_t.reshape(x_pred_t_t_1.shape), x_pred_t_t_1, edge_index, hidden_r, pred_sigma

    def calculate_features_for_gnn(self, delta_y_innov_i, edge_index, h_mat_i, measurements, node_number,
                                   x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, y_pred_t_t_1: Tensor) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor]:
        delta_x_hat_t_1, delta_x_wave_t_1, _, delta_y_t_i, edge_features = extract_kalman_features(
            edge_index, measurements, x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, y_pred_t_t_1, node_number
        )

        delta_x_hat_t_1 = _flatten_node_signal(delta_x_hat_t_1[..., 0], self.signal_dim)
        delta_x_wave_t_1 = _flatten_node_signal(delta_x_wave_t_1[..., 0], self.signal_dim)
        x_features = torch.cat([
            delta_x_hat_t_1,
            delta_x_wave_t_1,
        ], dim=-1)
        delta_y_t_i = _flatten_measurement(delta_y_t_i, self.measurement_dim)
        delta_y_innov_i = _flatten_measurement(delta_y_innov_i, self.measurement_dim)
        y_innov_features = torch.cat([delta_y_innov_i, h_mat_i.float()], dim=-1)

        return delta_y_t_i, edge_features, x_features, y_innov_features, delta_x_hat_t_1

    def calculate_h_mat_features(self, h_mat, edge_index, node_number):
        h_mat_edges = calculate_edge_features(h_mat, edge_index, node_number)[..., 0]
        h_mat_i = h_mat[:, range(node_number), range(node_number), ...]
        h_mat_i = h_mat_i.reshape(-1, self.signal_dim * self.measurement_dim)
        return h_mat_i, h_mat_edges

    def _build_adaptive_features(
        self,
        delta_y_innov_i: Tensor,
        delta_y_t_i: Tensor,
        h_mat_i: Tensor,
        x_hat: Tensor,
        delta_x_hat_t_1: Tensor,
    ) -> Tensor:
        feature_map = {
            "delta_y_innov": _flatten_measurement(delta_y_innov_i, self.measurement_dim),
            "delta_y": _flatten_measurement(delta_y_t_i, self.measurement_dim),
            "h_mat_i": h_mat_i.float(),
            "x_hat": _flatten_node_signal(x_hat, self.signal_dim),
            "delta_x_hat_t_1": _flatten_node_signal(delta_x_hat_t_1, self.signal_dim),
        }
        return torch.cat([feature_map[name] for name in ADAPTIVE_CONSENSUS_FEATURES], dim=-1)

    def _apply_consensus(self, phi_pred_t_t: Tensor, edge_index, delta_y_innov_i: Tensor,
                         delta_y_t_i: Tensor, h_mat_i: Tensor, delta_x_hat_t_1: Tensor) -> Tensor:
        phi_pred_t_t = phi_pred_t_t.reshape(-1, self.signal_dim, 1)
        if self.gcn is None:
            return phi_pred_t_t
        if isinstance(self.gcn, AdaptiveMeanConv):
            adaptive_features = self._build_adaptive_features(
                delta_y_innov_i,
                delta_y_t_i,
                h_mat_i,
                phi_pred_t_t[..., 0],
                delta_x_hat_t_1,
            )
            return self.gcn(phi_pred_t_t[..., 0], edge_index, adaptive_features)
        return self.gcn(phi_pred_t_t[..., 0], edge_index)

    def prediction_step(self, x_pred_t_1_t_1, h_system):
        # Call system function; it may return numpy arrays or tensors; normalize to torch tensor.
        x_pred_t_t_1_raw = self.f(x_pred_t_1_t_1)
        x_pred_t_t_1 = self._to_tensor(value=x_pred_t_t_1_raw, ref=x_pred_t_1_t_1)

        y_pred_raw = h_system(x_pred_t_t_1)
        y_pred_t_t_1 = self._to_tensor(value=y_pred_raw, ref=x_pred_t_t_1)

        return x_pred_t_t_1, y_pred_t_t_1

    @staticmethod
    def _to_tensor(value, ref):
        if isinstance(value, torch.Tensor):
            return value.to(device=ref.device, dtype=ref.dtype)
        return torch.tensor(value, dtype=ref.dtype, device=ref.device)


class GraphKalmanProcess(pl.LightningModule):
    def __init__(self, f_system, signal_dim, edge_features_dim, node_kalman_dim, edge_kalman_dim, r_array,
                 hidden_dim=32, heads=1, dropout=0.0,
                 lr: float | None = None, learning_rate: float | None = None,
                 learn_edge_kalman=True, x0_scale=10, consensus_layer: str | None = "none"):
        super(GraphKalmanProcess, self).__init__()
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim
        self.node_kalman_dim = node_kalman_dim
        self.edge_kalman_dim = edge_kalman_dim
        self.r_array = r_array
        self.x0_scale = x0_scale
        self.gkf = GraphKalmanFilter(
            f_system, signal_dim, node_kalman_dim,
            edge_features_dim, r_array, hidden_dim, heads, dropout, learn_edge_kalman, consensus_layer=consensus_layer
        )
        self.loss = torch.nn.MSELoss()
        if learning_rate is None:
            learning_rate = lr if lr is not None else 1e-3
        self.lr = float(learning_rate)
        self.learning_rate = self.lr

    def forward(self, data, x_0: torch.Tensor | None = None):
        if isinstance(data, Batch):
            graph_number = len(data)
            h_system = data[0].h_system
        else:
            graph_number = 1
            h_system = data.h_system

        node_number = data.num_nodes // graph_number
        measurements_shape = (graph_number, node_number,) + data.x.shape[1:]  # (BATCH, NODE_NUM, Time_steps, 1)
        x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, delta_y_innov_i, edge_index, hidden_r, pred_sigma = self.init_graph_kalman_params(
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

    def init_graph_kalman_params(self, x_0: torch.Tensor | None, data: Data | Batch, measurement_shape: tuple):
        batch_size, node_number = measurement_shape[0], measurement_shape[1]
        if x_0 is None:
            x_0 = self.x0_scale * torch.ones(
                batch_size, node_number, self.signal_dim, 1, dtype=torch.float,
                device=self.device)  # (batch, node_number, 2, 1)
        edge_index = data.edge_index
        delta_y_innov_i = torch.zeros(measurement_shape, dtype=torch.float, device=self.device)[:, :, 0, ...]
        hidden_r = torch.zeros(
            (1, batch_size * node_number, self.hidden_dim), dtype=torch.float, device=self.device)  # (1, node_number, hidden_dim)
        pred_sigma = torch.zeros(
            (batch_size * node_number, self.signal_dim * self.signal_dim), dtype=torch.float, device=self.device)  # (node_number, output_dim)
        x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2 = x_0, x_0, x_0  # (batch, node_number, 2, 1)
        return x_pred_t_1_t_1, x_pred_t_1_t_2, x_pred_t_2_t_2, delta_y_innov_i, edge_index, hidden_r, pred_sigma

    def _shared_step(self, batch, batch_idx, mode='train'):
        x_true = batch.y.reshape(batch.num_graphs, -1, self.signal_dim, 1)
        x_pred = self(batch).to(device=x_true.device)
        loss = loss_function(x_pred, x_true)
        self.log(f"{mode}_loss", loss, batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=True)
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
        # return optimizer

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        with torch.no_grad():
            for param_group in self.parameters():
                if param_group.grad is not None:
                    param_group.data = torch.nan_to_num(param_group.data, nan=0.0)
