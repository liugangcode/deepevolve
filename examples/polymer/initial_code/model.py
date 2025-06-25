import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import reset

from conv import GNN_node, GNN_node_Virtualnode
from utils import scatter_add

nn_act = torch.nn.ReLU()
F_act = F.relu


class GraphEnvAug(torch.nn.Module):
    def __init__(
        self,
        num_tasks,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        drop_ratio=0.5,
        gamma=0.4,
        use_linear_predictor=False,
    ):
        """
        num_tasks (int): number of labels to be predicted
        """

        super(GraphEnvAug, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.gamma = gamma

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        gnn_name = gnn_type.split("-")[0]
        emb_dim_rat = emb_dim
        if "virtual" in gnn_type:
            rationale_gnn_node = GNN_node_Virtualnode(
                2,
                emb_dim_rat,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
            )
            self.graph_encoder = GNN_node_Virtualnode(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
            )
        else:
            rationale_gnn_node = GNN_node(
                2,
                emb_dim_rat,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
            )
            self.graph_encoder = GNN_node(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
            )
        self.separator = Separator(
            rationale_gnn_node=rationale_gnn_node,
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim_rat, 2 * emb_dim_rat),
                torch.nn.BatchNorm1d(2 * emb_dim_rat),
                nn_act,
                torch.nn.Dropout(),
                torch.nn.Linear(2 * emb_dim_rat, 1),
            ),
            nn=None,
        )
        rep_dim = emb_dim
        if use_linear_predictor:
            self.predictor = torch.nn.Linear(rep_dim, self.num_tasks)
        else:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(rep_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                nn_act,
                torch.nn.Dropout(),
                torch.nn.Linear(2 * emb_dim, self.num_tasks),
            )

    def forward(self, batched_data):
        h_node = self.graph_encoder(batched_data)
        h_r, h_env, r_node_num, env_node_num = self.separator(batched_data, h_node)
        h_rep = (h_r.unsqueeze(1) + h_env.unsqueeze(0)).view(-1, self.emb_dim)
        pred_rem = self.predictor(h_r)
        pred_rep = self.predictor(h_rep)
        loss_reg = torch.abs(
            r_node_num / (r_node_num + env_node_num)
            - self.gamma * torch.ones_like(r_node_num)
        ).mean()
        output = {"pred_rep": pred_rep, "pred_rem": pred_rem, "loss_reg": loss_reg}
        return output

    def eval_forward(self, batched_data):
        h_node = self.graph_encoder(batched_data)
        h_r, _, _, _ = self.separator(batched_data, h_node)
        pred_rem = self.predictor(h_r)
        return pred_rem


class Separator(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, nn=None):
        super(Separator, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.rationale_gnn_node)
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, batched_data, h_node, size=None):
        x = self.rationale_gnn_node(batched_data)
        batch = batched_data.batch
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        h_node = self.nn(h_node) if self.nn is not None else h_node
        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)
        gate = torch.sigmoid(gate)

        h_out = scatter_add(gate * h_node, batch, dim=0, dim_size=size)
        c_out = scatter_add((1 - gate) * h_node, batch, dim=0, dim_size=size)

        r_node_num = scatter_add(gate, batch, dim=0, dim_size=size)
        env_node_num = scatter_add((1 - gate), batch, dim=0, dim_size=size)

        return h_out, c_out, r_node_num + 1e-8, env_node_num + 1e-8
