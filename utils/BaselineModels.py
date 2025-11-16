import pytorch_lightning as pl
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv

from utils.DistributedKalmanNet import loss_function


class GnnRnnLightning(pl.LightningModule):
    def __init__(self, input_dim=1, type_num=2, hidden_dim=32, output_dim=2, lr=5e-5):
        super(GnnRnnLightning, self).__init__()
        self.fc_measurement = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU()
        )
        self.type_embedding = torch.nn.Embedding(type_num, hidden_dim)

        self.gcn1 = GCNConv(
            in_channels=2 * hidden_dim, out_channels=2 * hidden_dim, aggr='mean', normalize=True, add_self_loops=True
        )
        self.activation = torch.nn.LeakyReLU()

        self.gcn2 = GCNConv(
            in_channels=2 * hidden_dim, out_channels=2 * hidden_dim, aggr='mean', normalize=True,
            add_self_loops=True
        )
        self.gru = torch.nn.GRU(
            input_size=2 * hidden_dim, hidden_size=2 * hidden_dim, num_layers=1, batch_first=True)

        self.fc_output = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        self.lr = lr
        self.loss = torch.nn.MSELoss()

    def forward(self, measurements, node_types, edge_index, hidden):
        all_node_num = measurements.shape[0]
        time_steps = measurements.shape[1]
        x_pred_t = torch.zeros(all_node_num, time_steps, self.fc_output[-1].out_features)
        for i in range(time_steps):
            measurement = measurements[:, i, ...]
            measurement_features = self.fc_measurement(measurement)
            type_features = self.type_embedding(node_types)[:, 0, :]
            x_features = torch.cat([measurement_features, type_features], dim=-1)
            x_gnn_output = self.gnn_pass(edge_index, x_features)
            node_output, hidden = self.gru(x_gnn_output.unsqueeze(1), hidden)
            node_output = self.fc_output(node_output[:, 0, :])
            x_pred_t[:, i, ...] = node_output
        return x_pred_t

    def gnn_pass(self, edge_index, x_features: Tensor) -> Tensor:
        x_gnn = self.gcn1(x_features, edge_index)
        x_gnn = self.activation(x_gnn)
        x_gnn = self.gcn2(x_gnn, edge_index)
        x_gnn = self.activation(x_gnn)
        return x_gnn

    def _shared_step(self, batch, batch_idx, mode='train'):
        node_classification = batch.h_system[0].node_classification.repeat_interleave(
            batch.num_graphs, dim=0).int()
        node_number = batch.x.shape[0] // batch.num_graphs
        x_true = batch.y.reshape(batch.num_graphs, batch.x.shape[1], -1)
        x_pred = self(batch.x, node_classification, batch.edge_index, None)
        x_pred = x_pred.reshape(batch.num_graphs, node_number, batch.x.shape[1], -1)

        loss = loss_function(x_pred, x_true)
        self.log(f'{mode}_loss:', loss, batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
