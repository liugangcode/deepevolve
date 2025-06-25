### >>> DEEPEVOLVE-BLOCK-START: Import global pooling functions for adaptive pooling
import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import global_mean_pool, global_add_pool

### <<< DEEPEVOLVE-BLOCK-END

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
        ### >>> DEEPEVOLVE-BLOCK-START: Remove unused separator to simplify pooling as per research idea
        # Removed the separator module since we now use adaptive global pooling with invariance injection.
        ### <<< DEEPEVOLVE-BLOCK-END
        rep_dim = emb_dim
        ### >>> DEEPEVOLVE-BLOCK-START: Ensure pooling_alpha is set regardless of predictor type
        ### >>> DEEPEVOLVE-BLOCK-START: Incorporate repetition invariant projection in GraphEnvAug __init__
        if use_linear_predictor:
            self.predictor = torch.nn.Linear(rep_dim, self.num_tasks)
        else:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(rep_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                nn_act,
                torch.nn.Dropout(drop_ratio),
                torch.nn.Linear(2 * emb_dim, self.num_tasks),
            )
        self.meta_pooling = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            nn_act,
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim, 1),
            torch.nn.Sigmoid(),
        )
        self.invariance_proj = torch.nn.Linear(1, emb_dim)
        ### >>> DEEPEVOLVE-BLOCK-START: Add DP-aware physics-informed loss parameters
        self.lambda_phys = torch.nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32), requires_grad=True
        )
        self.Tg_inf = torch.nn.Parameter(
            torch.tensor(500.0, dtype=torch.float32), requires_grad=True
        )
        self.K_Tg = torch.nn.Parameter(
            torch.tensor(100.0, dtype=torch.float32), requires_grad=True
        )
        ### <<< DEEPEVOLVE-BLOCK-END

    ### <<< DEEPEVOLVE-BLOCK-END

    ### <<< DEEPEVOLVE-BLOCK-END

    ### >>> DEEPEVOLVE-BLOCK-START: Replace separator-based pooling with adaptive global pooling using mean and sum
    ### >>> DEEPEVOLVE-BLOCK-START: Replace separator-based pooling with adaptive global pooling using mean and sum and invariant injection
    def forward(self, batched_data):
        h_node = self.graph_encoder(batched_data)
        if hasattr(batched_data, "repeat_unit"):
            # Perform hierarchical pooling using repeat unit segmentation with dynamic scaling
            max_repeat = batched_data.repeat_unit.max() + 1
            group = batched_data.batch * max_repeat + batched_data.repeat_unit
            local_features = global_mean_pool(h_node, group)
            # Aggregate local segment features to graph-level by averaging over segments
            unique_groups = torch.unique(group)
            # DEBUG: select only existing segment features and map to graph ids using max_repeat
            seg_features = local_features[unique_groups]
            graph_ids = unique_groups // max_repeat
            h_pool = global_mean_pool(seg_features, graph_ids)
        else:
            batch = batched_data.batch
            h_mean = global_mean_pool(h_node, batch)
            h_sum = global_add_pool(h_node, batch)
            meta_alpha = self.meta_pooling(h_mean)
            h_pool = meta_alpha * h_sum + (1 - meta_alpha) * h_mean
        if hasattr(batched_data, "invariant"):
            invar = batched_data.invariant.float().view(-1, 1)
            invar_emb = self.invariance_proj(invar)
            h_pool = h_pool + invar_emb
        pred = self.predictor(h_pool)
        # Compute physics-informed loss for Tg using dp_est if available
        if hasattr(batched_data, "dp_est"):
            dp = batched_data.dp_est.float().view(-1, 1)
            dp = torch.clamp(dp, min=1e-6)
            physics_target = self.Tg_inf - self.K_Tg / dp
            physics_loss = torch.abs(pred[:, 0:1] - physics_target)
            physics_loss = physics_loss.mean()
        else:
            physics_loss = torch.tensor(0.0, device=pred.device)
        output = {
            "pred_rem": pred,
            "pred_rep": pred,
            "physics_loss": physics_loss,
        }
        return output

    ### <<< DEEPEVOLVE-BLOCK-END

    ### <<< DEEPEVOLVE-BLOCK-END

    ### >>> DEEPEVOLVE-BLOCK-START: Update eval_forward with adaptive global pooling
    ### >>> DEEPEVOLVE-BLOCK-START: Update eval_forward with adaptive global pooling and invariant injection
    def eval_forward(self, batched_data):
        h_node = self.graph_encoder(batched_data)
        if hasattr(batched_data, "repeat_unit"):
            max_repeat = batched_data.repeat_unit.max() + 1
            group = batched_data.batch * max_repeat + batched_data.repeat_unit
            local_features = global_mean_pool(h_node, group)
            unique_groups = torch.unique(group)
            # DEBUG: select only existing segment features and map to graph ids using max_repeat
            seg_features = local_features[unique_groups]
            graph_ids = unique_groups // max_repeat
            h_pool = global_mean_pool(seg_features, graph_ids)
        else:
            batch = batched_data.batch
            h_mean = global_mean_pool(h_node, batch)
            h_sum = global_add_pool(h_node, batch)
            meta_alpha = self.meta_pooling(h_mean)
            h_pool = meta_alpha * h_sum + (1 - meta_alpha) * h_mean
        if hasattr(batched_data, "invariant"):
            invar = batched_data.invariant.float().view(-1, 1)
            invar_emb = self.invariance_proj(invar)
            h_pool = h_pool + invar_emb
        pred = self.predictor(h_pool)
        return pred


### <<< DEEPEVOLVE-BLOCK-END


### <<< DEEPEVOLVE-BLOCK-END


class Separator(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, nn=None):
        super(Separator, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    ### >>> DEEPEVOLVE-BLOCK-START: Fix reset_parameters to avoid error when nn is None
    def reset_parameters(self):
        reset(self.rationale_gnn_node)
        reset(self.gate_nn)
        if self.nn is not None:
            reset(self.nn)

    ### <<< DEEPEVOLVE-BLOCK-END

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


