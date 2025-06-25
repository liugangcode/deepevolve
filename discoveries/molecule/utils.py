import torch
from sklearn.metrics import r2_score


class Args:
    def __init__(self):
        # device
        self.device = 0

        # model
        self.gnn = "gin-virtual"
        self.drop_ratio = 0.5
        self.num_layer = 5
        self.emb_dim = 128
        self.use_linear_predictor = False
        self.gamma = 0.4

        # training
        self.batch_size = 256
        self.epochs = 200
        self.patience = 50
        self.lr = 1e-2
        self.l2reg = 5e-6
        self.use_lr_scheduler = False
        self.use_clip_norm = False
        self.path_list = [1, 4]
        self.initw_name = "default"

        # dataset
        self.dataset = "ogbg-molbbbp"
        self.trials = 5
        self.by_default = False


def get_args():
    return Args()


cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(args, model, device, loader, optimizers, task_type, optimizer_name):
    optimizer = optimizers[optimizer_name]
    model.train()
    if optimizer_name == "predictor":
        set_requires_grad([model.graph_encoder, model.predictor], requires_grad=True)
        set_requires_grad([model.separator], requires_grad=False)
    if optimizer_name == "separator":
        set_requires_grad([model.separator], requires_grad=True)
        set_requires_grad([model.graph_encoder, model.predictor], requires_grad=False)

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        ### >>> DEEPEVOLVE-BLOCK-START: Replace pass with continue to skip bad batches in train loop
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            continue
        ### <<< DEEPEVOLVE-BLOCK-END
        else:
            optimizer.zero_grad()
            pred = model(batch)
            if "classification" in task_type:
                criterion = cls_criterion
            else:
                criterion = reg_criterion

            if args.dataset.startswith("plym"):
                if args.plym_prop == "density":
                    batch.y = torch.log(batch[args.plym_prop])
                else:
                    batch.y = batch[args.plym_prop]
            target = batch.y.to(torch.float32)
            is_labeled = batch.y == batch.y
            ### >>> DEEPEVOLVE-BLOCK-START: Replace dual prediction loss with ACGR loss incorporating contrastive loss, motif reconstruction loss, and adaptive weighting
            pred_loss = criterion(
                pred["pred_rem"].to(torch.float32)[is_labeled], target[is_labeled]
            )
            contrast_loss = pred["contrast_loss"]
            adaptive_lambda = torch.sigmoid(contrast_loss - pred_loss)
            loss = pred_loss + adaptive_lambda * contrast_loss + pred["motif_loss"]
            ### <<< DEEPEVOLVE-BLOCK-END

            if optimizer_name == "separator":
                loss += pred["loss_reg"]

            loss.backward()
            if args.use_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def train_with_loss(args, model, device, loader, optimizers, task_type, optimizer_name):
    optimizer = optimizers[optimizer_name]
    model.train()
    if optimizer_name == "predictor":
        set_requires_grad([model.graph_encoder, model.predictor], requires_grad=True)
        set_requires_grad([model.separator], requires_grad=False)
    if optimizer_name == "separator":
        set_requires_grad([model.separator], requires_grad=True)
        set_requires_grad([model.graph_encoder, model.predictor], requires_grad=False)

    total_loss = 0
    num_batches = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        ### >>> DEEPEVOLVE-BLOCK-START: Replace pass with continue to skip bad batches in train_with_loss loop
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            continue
        ### <<< DEEPEVOLVE-BLOCK-END
        else:
            optimizer.zero_grad()
            pred = model(batch)
            if "classification" in task_type:
                criterion = cls_criterion
            else:
                criterion = reg_criterion

            if args.dataset.startswith("plym"):
                if args.plym_prop == "density":
                    batch.y = torch.log(batch[args.plym_prop])
                else:
                    batch.y = batch[args.plym_prop]
            target = batch.y.to(torch.float32)
            is_labeled = batch.y == batch.y
            ### >>> DEEPEVOLVE-BLOCK-START: Replace dual prediction loss with ACGR loss incorporating contrastive loss, motif reconstruction loss, and adaptive weighting
            pred_loss = criterion(
                pred["pred_rem"].to(torch.float32)[is_labeled], target[is_labeled]
            )
            contrast_loss = pred["contrast_loss"]
            adaptive_lambda = torch.sigmoid(contrast_loss - pred_loss)
            loss = pred_loss + adaptive_lambda * contrast_loss + pred["motif_loss"]
            ### <<< DEEPEVOLVE-BLOCK-END

            if optimizer_name == "separator":
                loss += pred["loss_reg"]

            total_loss += loss.item()
            num_batches += 1

            loss.backward()
            if args.use_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return total_loss / num_batches if num_batches > 0 else 0


def eval(args, model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        ### >>> DEEPEVOLVE-BLOCK-START: Replace pass with continue to avoid processing empty batches in eval loop
        if batch.x.shape[0] == 1:
            continue
        ### <<< DEEPEVOLVE-BLOCK-END
        else:
            with torch.no_grad():
                pred = model.eval_forward(batch)

            if args.dataset.startswith("plym"):
                if args.plym_prop == "density":
                    batch.y = torch.log(batch[args.plym_prop])
                else:
                    batch.y = batch[args.plym_prop]
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    if args.dataset.startswith("plym"):
        return [evaluator.eval(input_dict)["rmse"], r2_score(y_true, y_pred)]
    elif args.dataset.startswith("ogbg"):
        return [evaluator.eval(input_dict)["rocauc"]]


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == "default":
                pass
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# The code is from torch_scatter: https://github.com/rusty1s/pytorch_scatter/blob/1.3.0/torch_scatter/add.py
from itertools import repeat


def maybe_dim_size(index, dim_size=None):
    if dim_size is not None:
        return dim_size
    return index.max().item() + 1 if index.numel() > 0 else 0


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        out_size = list(src.size())
        dim_size = maybe_dim_size(index, dim_size)
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Sums all values from the :attr:`src` tensor into :attr:`out` at the indices
    specified in the :attr:`index` tensor along a given axis :attr:`dim`. For
    each value in :attr:`src`, its output index is specified by its index in
    :attr:`input` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`. If
    multiple indices reference the same location, their **contributions add**.

    Formally, if :attr:`src` and :attr:`index` are n-dimensional tensors with
    size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})` and
    :attr:`dim` = `i`, then :attr:`out` must be an n-dimensional tensor with
    size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`. Moreover, the
    values of :attr:`index` must be between `0` and `out.size(dim) - 1`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j \mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        out (Tensor, optional): The destination tensor. (default: :obj:`None`)
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor is
            returned. (default: :obj:`None`)
        fill_value (int, optional): If :attr:`out` is not given, automatically
            fill output tensor with :attr:`fill_value`. (default: :obj:`0`)

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_add

        src = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        out = src.new_zeros((2, 6))

        out = scatter_add(src, index, out=out)

        print(out)

    .. testoutput::

       tensor([[0., 0., 4., 3., 3., 0.],
               [2., 4., 4., 0., 0., 0.]])
    """
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)
