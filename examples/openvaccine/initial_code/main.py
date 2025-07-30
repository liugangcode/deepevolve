import pandas as pd
import numpy as np
import os
import random
import sys
from sklearn.model_selection import KFold
from tqdm import tqdm
import ast

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F

# Check if running in interactive terminal
is_tty = sys.stdout.isatty()

class config:
    learning_rate = 0.001
    batch_size = 256
    n_epoch = 100
    k_folds = 5
    weight_decay = 0
    K = 1  # number of aggregation loop (also means number of GCN layers)
    gcn_agg = 'mean'  # aggregator function: mean, conv, lstm, pooling
    filter_noise = True
    seed = 1234


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=3):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, :, i], y[:, :, i]) / self.num_scored
        return score


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GCN(nn.Module):
    """Implementation of one layer of GraphSAGE with Batch Normalization"""
    def __init__(self, input_dim, output_dim, aggregator='mean'):
        super(GCN, self).__init__()
        self.aggregator = aggregator
        
        if aggregator == 'mean':
            linear_input_dim = input_dim * 2
        elif aggregator == 'conv':
            linear_input_dim = input_dim
        elif aggregator == 'pooling':
            linear_input_dim = input_dim * 2
            self.linear_pooling = nn.Linear(input_dim, input_dim)
            self.bn_pooling = nn.BatchNorm1d(input_dim)
        elif aggregator == 'lstm':
            self.lstm_hidden = 128
            linear_input_dim = input_dim + self.lstm_hidden
            self.lstm_agg = nn.LSTM(input_dim, self.lstm_hidden, num_layers=1, batch_first=True)
            self.bn_lstm = nn.BatchNorm1d(self.lstm_hidden)
        
        self.linear_gcn = nn.Linear(in_features=linear_input_dim, out_features=output_dim)
        self.bn_gcn = nn.BatchNorm1d(output_dim)
        
    def forward(self, input_, adj_matrix):
        if self.aggregator == 'conv':
            # set elements in diagonal of adj matrix to 1 with conv aggregator
            idx = torch.arange(0, adj_matrix.shape[-1], out=torch.LongTensor())
            adj_matrix[:, idx, idx] = 1
            
        adj_matrix = adj_matrix.type(torch.float32)
        sum_adj = torch.sum(adj_matrix, axis=2)
        sum_adj[sum_adj==0] = 1
        
        if self.aggregator == 'mean' or self.aggregator == 'conv':
            feature_agg = torch.bmm(adj_matrix, input_)
            feature_agg = feature_agg / sum_adj.unsqueeze(dim=2)
            
        elif self.aggregator == 'pooling':
            feature_pooling = self.linear_pooling(input_)
            # Apply batch norm to pooling features
            batch_size, seq_len, feature_dim = feature_pooling.shape
            feature_pooling = feature_pooling.transpose(1, 2)  # (batch, feature, seq)
            feature_pooling = self.bn_pooling(feature_pooling)
            feature_pooling = feature_pooling.transpose(1, 2)  # (batch, seq, feature)
            
            feature_agg = torch.sigmoid(feature_pooling)
            feature_agg = torch.bmm(adj_matrix, feature_agg)
            feature_agg = feature_agg / sum_adj.unsqueeze(dim=2)

        elif self.aggregator == 'lstm':
            feature_agg = torch.zeros(input_.shape[0], input_.shape[1], self.lstm_hidden).cuda()
            for i in range(adj_matrix.shape[1]):
                neighbors = adj_matrix[:, i, :].unsqueeze(2) * input_
                _, hn = self.lstm_agg(neighbors)
                feature_agg[:, i, :] = torch.squeeze(hn[0], 0)
            
            # Apply batch norm to LSTM features
            batch_size, seq_len, feature_dim = feature_agg.shape
            feature_agg = feature_agg.transpose(1, 2)  # (batch, feature, seq)
            feature_agg = self.bn_lstm(feature_agg)
            feature_agg = feature_agg.transpose(1, 2)  # (batch, seq, feature)
                
        if self.aggregator != 'conv':
            feature_cat = torch.cat((input_, feature_agg), axis=2)
        else:
            feature_cat = feature_agg
        
        # Apply linear transformation
        feature = self.linear_gcn(feature_cat)
        
        # Apply batch normalization
        batch_size, seq_len, feature_dim = feature.shape
        feature = feature.transpose(1, 2)  # (batch, feature, seq)
        feature = self.bn_gcn(feature)
        feature = feature.transpose(1, 2)  # (batch, seq, feature)
        
        # Apply activation and normalization
        feature = torch.sigmoid(feature)
        feature = feature / torch.norm(feature, p=2, dim=2).unsqueeze(dim=2)
        
        return feature


class Net(nn.Module):
    def __init__(self, num_embedding=14, seq_len=107, pred_len=68, dropout=0.5, 
                 embed_dim=100, hidden_dim=128, K=1, aggregator='mean'):
        """
        K: number of GCN layers
        aggregator: type of aggregator function
        """
        super(Net, self).__init__()
        
        self.pred_len = pred_len
        self.embedding_layer = nn.Embedding(num_embeddings=num_embedding, 
                                      embedding_dim=embed_dim)
        
        # Batch normalization for embedding
        self.bn_embedding = nn.BatchNorm1d(3 * embed_dim)
        
        self.gcn = nn.ModuleList([GCN(3 * embed_dim, 3 * embed_dim, aggregator=aggregator) for i in range(K)])
        
        self.gru_layer = nn.GRU(input_size=3 * embed_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=3, 
                          batch_first=True, 
                          dropout=dropout, 
                          bidirectional=True)
        
        # Batch normalization for GRU output
        self.bn_gru = nn.BatchNorm1d(2 * hidden_dim)
        
        self.linear_layer = nn.Linear(in_features=2 * hidden_dim, 
                                out_features=3)  # Only 3 outputs now
        
    def forward(self, input_, adj_matrix):
        # embedding
        embedding = self.embedding_layer(input_)
        embedding = torch.reshape(embedding, (-1, embedding.shape[1], embedding.shape[2] * embedding.shape[3]))
        
        # Apply batch normalization to embedding
        batch_size, seq_len, feature_dim = embedding.shape
        embedding = embedding.transpose(1, 2)  # (batch, feature, seq)
        embedding = self.bn_embedding(embedding)
        embedding = embedding.transpose(1, 2)  # (batch, seq, feature)
        
        # gcn
        gcn_feature = embedding
        for gcn_layer in self.gcn:
            gcn_feature = gcn_layer(gcn_feature, adj_matrix)
        
        # gru
        gru_output, gru_hidden = self.gru_layer(gcn_feature)
        truncated = gru_output[:, :self.pred_len]
        
        # Apply batch normalization to GRU output
        batch_size, seq_len, feature_dim = truncated.shape
        truncated = truncated.transpose(1, 2)  # (batch, feature, seq)
        truncated = self.bn_gru(truncated)
        truncated = truncated.transpose(1, 2)  # (batch, seq, feature)
        
        output = self.linear_layer(truncated)
        
        return output


# Only use the first 3 prediction columns as specified, PREDICTION COLUMNS ARE FIXED and DON'T CHANGE!
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'] # FIXED and DON'T CHANGE!
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')} # FIXED and DON'T CHANGE!


def get_couples(structure):
    """
    For each closing parenthesis, find the matching opening one and store their index in the couples list.
    """
    opened = [idx for idx, i in enumerate(structure) if i == '(']
    closed = [idx for idx, i in enumerate(structure) if i == ')']

    assert len(opened) == len(closed)
    assigned = []
    couples = []

    for close_idx in closed:
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        assigned.append(candidate)
        couples.append([candidate, close_idx])
        
    assert len(couples) == len(opened)
    
    return couples


def build_matrix(couples, size):
    mat = np.zeros((size, size))
    
    for i in range(size):  # neighboring bases are linked as well
        if i < size - 1:
            mat[i, i + 1] = 1
        if i > 0:
            mat[i, i - 1] = 1
    
    for i, j in couples:
        mat[i, j] = 1
        mat[j, i] = 1
        
    return mat


def convert_to_adj(structure):
    couples = get_couples(structure)
    mat = build_matrix(couples, len(structure))
    return mat


def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    inputs = np.transpose(
        np.array(
            df[cols]
            .map(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    
    adj_matrix = np.array(df['structure'].apply(convert_to_adj).values.tolist())
    
    return inputs, adj_matrix


def prepare_labels_from_csv(df, sn_filter_mask):
    """
    Prepare labels from CSV data format
    """
    # Extract label columns and apply SN filter
    labels = []
    for col in pred_cols:
        # Parse the string representation of the list and convert to list of floats
        col_data = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        # Convert each element to a list if it's not already
        col_data = col_data.apply(lambda x: x if isinstance(x, list) else [x])
        labels.append(list(col_data.values))
    
    # Convert to numpy array - labels is now [n_targets, n_samples, seq_len]
    # We need to transpose to get [n_samples, seq_len, n_targets]
    max_len = max(len(seq) for target_data in labels for seq in target_data)
    
    # Pad sequences to same length and convert to proper format
    processed_labels = []
    for i in range(len(labels[0])):  # for each sample
        sample_labels = []
        for j in range(len(labels)):  # for each target
            seq = labels[j][i]
            # Pad if necessary
            if len(seq) < max_len:
                seq = seq + [0.0] * (max_len - len(seq))
            sample_labels.append(seq)
        processed_labels.append(np.array(sample_labels).T)  # Transpose to get [seq_len, n_targets]
    
    labels = np.array(processed_labels)  # Shape: (n_samples, seq_len, n_targets)
    
    # Apply SN filter mask
    labels = labels[sn_filter_mask]
    
    return labels


def train_fn(model, train_loader, criterion, optimizer, epoch_desc="Training"):
    model.train()
    model.zero_grad()
    train_loss = AverageMeter()
    
    # Remove tqdm for batch iterations
    for batch_idx, (input_, adj, label, sn_mask) in enumerate(train_loader):
        input_ = input_.cuda()
        adj = adj.cuda()
        label = label.cuda()
        sn_mask = sn_mask.cuda()
        
        preds = model(input_, adj)
        
        # Apply SN filter mask to both predictions and labels
        valid_preds = preds[sn_mask]
        valid_labels = label[sn_mask]
        
        if valid_preds.size(0) > 0:  # Only compute loss if we have valid samples
            loss = criterion(valid_preds, valid_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item())
    
    return train_loss.avg


def eval_fn(model, valid_loader, criterion, epoch_desc="Validation"):
    model.eval()
    eval_loss = AverageMeter()
    
    # Remove tqdm for batch iterations
    with torch.no_grad():
        for batch_idx, (input_, adj, label, sn_mask) in enumerate(valid_loader):
            input_ = input_.cuda()
            adj = adj.cuda()
            label = label.cuda()
            sn_mask = sn_mask.cuda()
            
            preds = model(input_, adj)
            
            # Apply SN filter mask to both predictions and labels
            valid_preds = preds[sn_mask]
            valid_labels = label[sn_mask]
            
            if valid_preds.size(0) > 0:  # Only compute loss if we have valid samples
                loss = criterion(valid_preds, valid_labels)
                eval_loss.update(loss.item())
    
    return eval_loss.avg


def run_fold(train_loader, valid_loader, test_loader, cfg, fold_idx):
    """Train model for one fold"""
    model = Net(K=cfg.K, aggregator=cfg.gcn_agg, pred_len=68)  # For validation
    model.cuda()
    criterion = MCRMSELoss(num_scored=3)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    train_losses = []
    eval_losses = []
    test_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    if is_tty:
        print(f"\nFold {fold_idx + 1}/{cfg.k_folds} Training:")
    
    # Only show tqdm for epochs, not for batch iterations
    epoch_bar = tqdm(range(cfg.n_epoch), desc=f"Fold {fold_idx + 1}", disable=not is_tty)
    
    for epoch in epoch_bar:
        train_loss = train_fn(model, train_loader, criterion, optimizer, 
                            f"Fold {fold_idx+1} Epoch {epoch+1}/{cfg.n_epoch} - Train")
        eval_loss = eval_fn(model, valid_loader, criterion, 
                          f"Fold {fold_idx+1} Epoch {epoch+1}/{cfg.n_epoch} - Valid")
        
        # Evaluate on test set with 91 positions
        model_test = Net(K=cfg.K, aggregator=cfg.gcn_agg, pred_len=91)  # For test evaluation
        model_test.load_state_dict(model.state_dict(), strict=False)
        model_test.cuda()
        test_loss = eval_fn(model_test, test_loader, criterion,
                          f"Fold {fold_idx+1} Epoch {epoch+1}/{cfg.n_epoch} - Test")
        
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        test_losses.append(test_loss)
        
        # Save best model state
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            best_model_state = model.state_dict().copy()
        
        if is_tty:
            epoch_bar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{eval_loss:.6f}',
                'test_loss': f'{test_loss:.6f}',
                'best_val': f'{best_val_loss:.6f}'
            })
    
    return best_model_state, train_losses, eval_losses, test_losses, best_val_loss


def predict_with_model(model_state, test_loader, cfg):
    """Make predictions with a single model"""
    model = Net(K=cfg.K, aggregator=cfg.gcn_agg, pred_len=91)  # For test evaluation
    model.load_state_dict(model_state)
    model.cuda()
    model.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for batch_idx, (input_, adj, label, sn_mask) in enumerate(test_loader):
            input_ = input_.cuda()
            adj = adj.cuda()
            
            preds = model(input_, adj)
            all_preds.append(preds.cpu().numpy())
    
    return np.concatenate(all_preds, axis=0)


def get_bpps_features(base_dir, seq_id):
    """
    BPPS features are pre-calculated by the competition organizers and stored in the base_dir/bpps/ directory as {seq_id}.npy
    It is a matrix of base pair probabilities. Biophysically speaking, this matrix gives the probability that each pair of nucleotides in the RNA forms a base pair (given a particular model of RNA folding.
    It is a symmetric square matrix with the same length as the sequence.
    Each column and each row should sum to one (up to rounding error), but more than one entry in each column/row will be nonzero -- usually somewhere between 1-5 entries.

    It is not used in the initial idea, but we keep it for future idea evolution.
    """
    matrix = np.load(f'{base_dir}/bpps/{seq_id}.npy')
    return matrix

def main(base_dir='../../../data_cache/openvaccine'):
    """
    Main function to run the GraphSAGE GRU model with K-Fold cross-validation
    
    Args:
        base_dir (str): Path to the data directory containing train.json and private_test_labels.csv
    """
    
    # Update config with the provided base_dir
    config.train_file = os.path.join(base_dir, 'train.json')
    config.test_file = os.path.join(base_dir, 'post_deadline_files', 'private_test_labels.csv')
    
    # Set random seed
    seed_everything(config.seed)
    train = pd.read_json(config.train_file, lines=True)

    if config.filter_noise:
        train = train[train.signal_to_noise > 1]
        
    # Load test data from CSV
    test = pd.read_csv(config.test_file)

    # Un-comment to load BPPS features for each sequence if needed, this code is not used in the initial idea
    # train['bpps'] = train['id'].apply(lambda x: get_bpps_features(base_dir, x))
    # test['bpps'] = test['id'].apply(lambda x: get_bpps_features(base_dir, x))

    # Filter test data by SN_filter == 1
    test_filtered = test[test['S/N filter'] == 1].copy()
    
    # Preprocess training data - use only first 68 positions for validation
    train_inputs, train_adj = preprocess_inputs(train)
    train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
    
    # Truncate to first 68 positions for validation
    train_inputs_val = train_inputs[:, :68, :]
    train_adj_val = train_adj[:, :68, :68]
    train_labels_val = train_labels[:, :68, :]
    
    # Create SN filter mask for training data (all 1s since we already filtered)
    train_sn_mask = np.ones(len(train), dtype=bool)
    
    # Convert to tensors
    train_inputs_tensor = torch.tensor(train_inputs_val, dtype=torch.long)
    train_adj_tensor = torch.tensor(train_adj_val, dtype=torch.long)
    train_labels_tensor = torch.tensor(train_labels_val, dtype=torch.float32)
    train_sn_mask_tensor = torch.tensor(train_sn_mask, dtype=torch.bool)
    
    # Preprocess test data - use first 91 positions for test evaluation
    test_inputs, test_adj = preprocess_inputs(test_filtered)
    test_labels = prepare_labels_from_csv(test_filtered, np.ones(len(test_filtered), dtype=bool))
    
    # Truncate to first 91 positions for test
    test_inputs_91 = test_inputs[:, :91, :]
    test_adj_91 = test_adj[:, :91, :91]
    test_labels_91 = test_labels[:, :91, :]
    
    # Create SN filter mask for test data (all 1s since we already filtered)
    test_sn_mask = np.ones(len(test_filtered), dtype=bool)
    
    # Convert test data to tensors
    test_inputs_tensor = torch.tensor(test_inputs_91, dtype=torch.long)
    test_adj_tensor = torch.tensor(test_adj_91, dtype=torch.long)
    test_labels_tensor = torch.tensor(test_labels_91, dtype=torch.float32)
    test_sn_mask_tensor = torch.tensor(test_sn_mask, dtype=torch.bool)
    
    test_dataset = TensorDataset(
        test_inputs_tensor, 
        test_adj_tensor, 
        test_labels_tensor,
        test_sn_mask_tensor
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # K-Fold cross-validation
    kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
    
    fold_results = []
    model_states = []
    all_train_losses = []
    all_eval_losses = []
    all_test_losses = []
    
    if is_tty:
        print(f"\nStarting {config.k_folds}-Fold Cross-Validation...")
        print(f"Total training samples: {len(train)}")
        print(f"Test samples: {len(test_filtered)}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(train)))):
        if is_tty:
            print(f"\nFold {fold_idx + 1}/{config.k_folds}")
            print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        
        # Create datasets for this fold
        train_dataset = TensorDataset(
            train_inputs_tensor[train_idx], 
            train_adj_tensor[train_idx], 
            train_labels_tensor[train_idx],
            train_sn_mask_tensor[train_idx]
        )
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
        
        valid_dataset = TensorDataset(
            train_inputs_tensor[val_idx], 
            train_adj_tensor[val_idx], 
            train_labels_tensor[val_idx],
            train_sn_mask_tensor[val_idx]
        )
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
        
        # Train model for this fold
        model_state, train_losses, eval_losses, test_losses, best_val_loss = run_fold(
            train_loader, valid_loader, test_loader, config, fold_idx
        )
        
        # Store results
        model_states.append(model_state)
        all_train_losses.append(np.mean(train_losses))
        all_eval_losses.append(np.mean(eval_losses))
        all_test_losses.append(np.mean(test_losses))
        
        fold_result = {
            'fold': fold_idx + 1,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
        fold_results.append(fold_result)
        
        if is_tty:
            print(f"Fold {fold_idx + 1} completed:")
            print(f"  Best validation loss: {best_val_loss:.6f}")
            print(f"  Final train loss: {train_losses[-1]:.6f}")
            print(f"  Final test loss: {test_losses[-1]:.6f}")
    
    # Ensemble prediction on test set
    if is_tty:
        print(f"\nGenerating ensemble predictions from {len(model_states)} models...")
    test_predictions = []
    
    for i, model_state in enumerate(model_states):
        if is_tty:
            print(f"Getting predictions from model {i+1}/{len(model_states)}...")
        preds = predict_with_model(model_state, test_loader, config)
        test_predictions.append(preds)
    
    # Average predictions
    ensemble_predictions = np.mean(test_predictions, axis=0)
    
    # Calculate ensemble test loss
    criterion = MCRMSELoss(num_scored=3)
    ensemble_test_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (input_, adj, label, sn_mask) in enumerate(test_loader):
            batch_start = batch_idx * config.batch_size
            batch_end = min(batch_start + config.batch_size, len(ensemble_predictions))
            
            batch_preds = torch.tensor(ensemble_predictions[batch_start:batch_end], dtype=torch.float32).cuda()
            label = label.cuda()
            sn_mask = sn_mask.cuda()
            
            valid_preds = batch_preds[sn_mask]
            valid_labels = label[sn_mask]
            
            if valid_preds.size(0) > 0:
                loss = criterion(valid_preds, valid_labels)
                ensemble_test_loss += loss.item()
                num_batches += 1
    
    ensemble_test_loss /= num_batches if num_batches > 0 else 1
    
    # Print final results
    if is_tty:
        print(f"\n{'='*80}")
        print("K-Fold Cross-Validation Results:")
        print(f"{'='*80}")
        
        val_losses = [result['best_val_loss'] for result in fold_results]
        train_losses_final = [result['final_train_loss'] for result in fold_results]
        test_losses_final = [result['final_test_loss'] for result in fold_results]
        
        print(f"Validation Loss - Mean: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
        print(f"Train Loss - Mean: {np.mean(train_losses_final):.6f} ± {np.std(train_losses_final):.6f}")
        print(f"Test Loss - Mean: {np.mean(test_losses_final):.6f} ± {np.std(test_losses_final):.6f}")
        print(f"Ensemble Test Loss: {ensemble_test_loss:.6f}")
        
        print("\nPer-fold results:")
        for result in fold_results:
            print(f"Fold {result['fold']}: Val={result['best_val_loss']:.6f}, "
                  f"Train={result['final_train_loss']:.6f}, Test={result['final_test_loss']:.6f}")
    
    return {
        'train_mean_loss_across_folds': float(np.mean(all_train_losses)),
        'test_MCRMSE': float(ensemble_test_loss),
    }


if __name__ == "__main__":
    results = main('../../../data_cache/openvaccine')
    if is_tty:
        print("\nTraining completed successfully!")
        print(results)
