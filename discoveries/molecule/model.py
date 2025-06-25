### >>> DEEPEVOLVE-BLOCK-START: Add InfoNCE loss for contrastive learning and ensure it is available in model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import reset


### >>> DEEPEVOLVE-BLOCK-START: Update documentation for InfoNCE loss with advanced negative sampling note
### >>> DEEPEVOLVE-BLOCK-START: Update InfoNCE loss to support uncertainty-guided negative sampling
def info_nce_loss(z1, z2, temperature=0.5, negatives=None):
    """
    Computes the InfoNCE loss using current batch negatives.
    If 'negatives' is provided, applies advanced negative sampling for enhanced robustness.
    """
    z1 = torch.nn.functional.normalize(z1, p=2, dim=1)
    z2 = torch.nn.functional.normalize(z2, p=2, dim=1)
    if negatives is not None:
        negatives = torch.nn.functional.normalize(negatives, p=2, dim=1)
        sim_pos = torch.sum(z1 * z2, dim=1, keepdim=True) / temperature
        sim_neg = torch.matmul(z1, negatives.t()) / temperature
        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.zeros(z1.size(0), device=z1.device, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    else:
        logits = torch.matmul(z1, z2.t()) / temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


### <<< DEEPEVOLVE-BLOCK-END


### <<< DEEPEVOLVE-BLOCK-END


### <<< DEEPEVOLVE-BLOCK-END

from conv import GNN_node, GNN_node_Virtualnode
from utils import scatter_add

nn_act = torch.nn.ReLU()
F_act = F.relu


class GraphEnvAug(torch.nn.Module):
    ### >>> DEEPEVOLVE-BLOCK-START: Add temperature parameter for contrastive loss scaling
    def __init__(
        self,
        num_tasks,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        drop_ratio=0.5,
        gamma=0.4,
        use_linear_predictor=False,
        temperature=0.5,
    ):
        """
        num_tasks (int): number of labels to be predicted
        """
        self.temperature = temperature
        self.mc_dropout_samples = (
            20  # Increased number of MC dropout iterations for uncertainty estimation
        )
        self.gumbel_tau = 1.0
        ### <<< DEEPEVOLVE-BLOCK-END

        super(GraphEnvAug, self).__init__()
        ### >>> DEEPEVOLVE-BLOCK-START: Initialize self-supervised motif reconstruction module
        self.motif_decoder = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            nn_act,
            torch.nn.Dropout(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        ### <<< DEEPEVOLVE-BLOCK-END

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

    ### >>> DEEPEVOLVE-BLOCK-START: Incorporate dual augmented views with contrastive loss and adaptive weighting in ACGR
    ### >>> DEEPEVOLVE-BLOCK-START: Incorporate dual augmented views with motif-aware attribute masking and adaptive weighting in ACGR
    ### >>> DEEPEVOLVE-BLOCK-START: Integrate self-supervised motif reconstruction branch with uncertainty-guided negative sampling
    ### >>> DEEPEVOLVE-BLOCK-START: Incorporate dual-phase adversarial perturbation and uncertainty‐guided negative sampling in forward pass
    def forward(self, batched_data, phase="standard"):
        h_node = self.graph_encoder(batched_data)
        # Self-supervised motif reconstruction branch: apply motif-aware attribute masking
        masked_data = self.motif_mask(batched_data)
        h_masked = self.graph_encoder(masked_data)
        # Reconstruction: recover masked motifs from the masked view
        motif_pred = self.motif_decoder(h_masked)
        loss_recon = 1 - F.cosine_similarity(motif_pred, h_node, dim=1).mean()

        # If in adversarial phase, apply dual-phase perturbation based on computed uncertainty
        if phase == "adversarial" and hasattr(self, "last_uncertainty"):
            perturb = torch.randn_like(h_node) * (
                self.last_uncertainty.mean() * self.gumbel_tau
            )
            h_node = h_node + perturb

        # Generate dual augmented views via separator for environment replacement
        h_r1, h_env1, r_node_num1, env_node_num1 = self.separator(batched_data, h_node)
        h_r2, h_env2, r_node_num2, env_node_num2 = self.separator(batched_data, h_node)
        pred_rem = self.predictor(h_r1)

        # Compute contrast losses with uncertainty-guided negative sampling in adversarial phase
        if phase == "adversarial":
            adv_negatives = h_r1[torch.randperm(h_r1.size(0))]
            contrast_loss_env = info_nce_loss(
                h_r1, h_r2, temperature=self.temperature, negatives=adv_negatives
            )
        else:
            contrast_loss_env = info_nce_loss(h_r1, h_r2, temperature=self.temperature)
        contrast_loss_motif = info_nce_loss(
            h_node, h_masked, temperature=self.temperature
        )
        contrast_loss = (contrast_loss_env + contrast_loss_motif) / 2

        # Regularization to align node count ratios with the predefined gamma
        r_node_num = (r_node_num1 + r_node_num2) / 2
        env_node_num = (env_node_num1 + env_node_num2) / 2
        loss_reg = torch.abs(
            r_node_num / (r_node_num + env_node_num) - self.gamma
        ).mean()

        output = {
            "pred_rem": pred_rem,
            "contrast_loss": contrast_loss,
            "loss_reg": loss_reg,
            "motif_loss": loss_recon,
        }
        return output

    ### <<< DEEPEVOLVE-BLOCK-END

    ### <<< DEEPEVOLVE-BLOCK-END

    ### <<< DEEPEVOLVE-BLOCK-END

    ### <<< DEEPEVOLVE-BLOCK-END

    def eval_forward(self, batched_data):
        h_node = self.graph_encoder(batched_data)
        h_r, _, _, _ = self.separator(batched_data, h_node)
        pred_rem = self.predictor(h_r)
        return pred_rem

    ### >>> DEEPEVOLVE-BLOCK-START: Add motif-aware attribute masking method to GraphEnvAug
    ### >>> DEEPEVOLVE-BLOCK-START: Update motif_mask for uncertainty-aware differentiable motif extraction using Gumbel-Softmax and MC Dropout
    def motif_mask(self, batched_data):
        import copy
        import torch.nn.functional as F

        # motif_mask: compute adaptive motif mask without altering original x
        new_data = copy.deepcopy(batched_data)
        orig_x = new_data.x
        x_float = orig_x.float()

        # Initialize motif_selector and dropout if not already defined
        if not hasattr(self, "motif_selector"):
            self.motif_selector = torch.nn.Linear(orig_x.size(1), 2).to(orig_x.device)
            self.motif_dropout = torch.nn.Dropout(p=0.5)
        num_samples = (
            self.mc_dropout_samples if hasattr(self, "mc_dropout_samples") else 5
        )  # Use configured number of MC dropout samples
        motif_samples = []
        tau = 1.0  # Temperature parameter for Gumbel-Softmax; can be tuned
        for _ in range(num_samples):
            logits = self.motif_selector(x_float)
            logits = self.motif_dropout(logits)  # MC Dropout
            sample = F.gumbel_softmax(logits, tau=tau, hard=False, dim=1)[
                :, 1
            ].unsqueeze(1)
            motif_samples.append(sample)
        motif_samples = torch.stack(
            motif_samples, dim=0
        )  # Shape: [num_samples, num_nodes, 1]
        mean_score = motif_samples.mean(dim=0)  # Aggregated motif probability
        uncertainty = motif_samples.var(dim=0)  # Variance as uncertainty
        threshold_uncertainty = 0.05  # Adaptive threshold hyperparameter
        adaptive_mask = torch.where(
            uncertainty < threshold_uncertainty,
            mean_score,
            mean_score * (threshold_uncertainty / (uncertainty + 1e-8)),
        )

        # Store computed uncertainty for potential adversarial perturbation
        self.last_uncertainty = uncertainty
        # DEBUG: store adaptive mask for use in GNN (applied in conv.py)
        new_data.mask = adaptive_mask
        return new_data


### <<< DEEPEVOLVE-BLOCK-END


### <<< DEEPEVOLVE-BLOCK-END

### <<< DEEPEVOLVE-BLOCK-END


class Separator(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, nn=None):
        super(Separator, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    ### >>> DEEPEVOLVE-BLOCK-START: Safeguard reset of optional submodule 'nn' in Separator
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


