from types import SimpleNamespace

# Data configuration
DATA_DIR = ""
TARGET_HORIZONS = [0, 6, 12, 24]
TEST_VMONTHS = [0, 6, 12, 18, 24, 36, 48, 60, 72, 84]

# LightGBM parameters
LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_class": 87,
    "n_estimators": 300,
    "learning_rate": 0.019673004699536346,
    "num_leaves": 208,
    "max_depth": 14,
    "min_data_in_leaf": 850,
    "feature_fraction": 0.5190632906197453,
    "lambda_l1": 7.405660751699475e-08,
    "lambda_l2": 0.14583961675675494,
    "max_bin": 240,
    "verbose": -1,
    "force_col_wise": True,
    "n_jobs": -1,
}


# Neural Network configuration
def get_nn_config():
    cfg = SimpleNamespace(**{})
    cfg.tr_collate_fn = None
    cfg.val_collate_fn = None
    cfg.target_column = "target_norm"
    cfg.output_dir = "results/nn_temp"
    cfg.seed = -1
    cfg.eval_epochs = 1
    cfg.mixed_precision = False
    ### >>> DEEPEVOLVE-BLOCK-START: Set device dynamically based on CUDA availability
    import torch

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    ### >>> DEEPEVOLVE-BLOCK-START: Enable cuDNN benchmark for performance if using CUDA
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
    ### <<< DEEPEVOLVE-BLOCK-END
    ### <<< DEEPEVOLVE-BLOCK-END
    cfg.pretrained_transformer = (
        None  # path to pre-trained transformer encoder weights (if available)
    )
    cfg.n_classes = 1
    cfg.batch_size = 128
    cfg.batch_size_val = 256
    cfg.n_hidden = 64
    cfg.n_layers = 2
    cfg.num_workers = 0
    cfg.drop_last = False
    cfg.gradient_clip = 1.0
    cfg.bag_size = 1
    cfg.bag_agg_function = "mean"
    cfg.lr = 2e-3
    cfg.warmup = 0
    cfg.epochs = 10
    # Added parameters for hybrid model enhancements
    cfg.use_cat = False  # set to True to enable categorical covariate embedding
    cfg.use_protein = False  # set to True to enable protein sequence embeddings
    cfg.use_transformer = (
        True  # enable transformer encoder for adaptive feature extraction
    )
    cfg.use_transformer = (
        True  # enable transformer encoder for adaptive feature extraction
    )
    cfg.interp_steps = 5
    cfg.cat_vocab_size = 10
    cfg.cat_embed_dim = 8
    cfg.protein_vocab_size = 1000
    cfg.protein_embed_dim = 32
    # Enable meta‐learning and MC dropout uncertainty estimation for rapid per‐patient adaptation.
    cfg.meta_learning = True
    cfg.mc_dropout = True
    cfg.mc_dropout_samples = 10
    cfg.mc_dropout_prob = 0.1
    cfg.calib_factor = 1.0
    cfg.inner_lr = 1e-3
    return cfg


# Feature configuration
def get_lgb_features():
    features = [
        "target_i",
        "target_month",
        "horizon",
        "visit_month",
        "visit_6m",
        "blood_taken",
    ]
    features += ["visit_18m", "is_suppl"]
    features += ["count_non12_visits"]
    features += ["visit_48m"]
    return features


def get_nn_features(sample_df):
    features = ["visit_6m"]
    features += [c for c in sample_df.columns if c.startswith("t_month_eq_")]
    features += [c for c in sample_df.columns if c.startswith("v_month_eq_")]
    features += [c for c in sample_df.columns if c.startswith("hor_eq_")]
    features += [c for c in sample_df.columns if c.startswith("target_n_")]
    features += ["visit_18m"]
    features += ["visit_48m"]
    features += ["is_suppl"]
    features += ["horizon_scaled"]
    return features


