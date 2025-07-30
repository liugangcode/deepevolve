import os
import time
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rapidfuzz.distance import Levenshtein

import cv2

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

import timm
import warnings
warnings.filterwarnings('ignore')

import sys
is_tty = sys.stdout.isatty()

class Config:
    """Simple configuration class for model hyperparameters and settings"""
    
    def __init__(self, base_dir='data_cache/molecular_translation'):
        # Training settings
        self.debug = False
        self.epochs = 10
        self.batch_size = 512
        
        # Data settings
        self.base_dir = base_dir              # Base data directory
        
        # Model architecture
        self.model_name = 'resnet34'          # Backbone model name
        self.size = 224                       # Input image size
        self.max_len = 275                    # Maximum sequence length
        self.attention_dim = 256              # Attention dimension
        self.embed_dim = 256                  # Embedding dimension
        self.decoder_dim = 512                # Decoder hidden dimension
        self.dropout = 0.5                    # Dropout rate
        
        # Training hyperparameters
        self.encoder_lr = 1e-4                # Encoder learning rate
        self.decoder_lr = 1e-4                # Decoder learning rate
        self.min_lr = 1e-6                    # Minimum learning rate
        self.weight_decay = 1e-6              # Weight decay
        self.max_grad_norm = 5                # Gradient clipping norm
        
        # Scheduler settings
        self.scheduler = 'CosineAnnealingLR'  # Learning rate scheduler
        self.T_max = 4                        # T_max for CosineAnnealingLR
        
        # Other settings
        self.seed = 42                        # Random seed
        self.print_freq = 1000                # Print frequency
        self.gradient_accumulation_steps = 1  # Gradient accumulation steps
        self.train = True                     # Whether to train the model
        
        # Detect if running in tmp environment
        if '/tmp/' in os.getcwd():
            self.num_workers = 0  # No multiprocessing in tmp, FIXED it to 0 and do NOT change it
            self.pin_memory = False  # Reduce memory overhead, FIXED it to False and do NOT change it
        else:
            self.num_workers = 4
            self.pin_memory = True


class Tokenizer:
    """Tokenizer for converting text to sequences and vice versa"""
    
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)
    
    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(list(text))  # Character-wise tokenization
        vocab = sorted(vocab)
        vocab.extend(['<sos>', '<eos>', '<pad>'])
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        
    def text_to_sequence(self, text):
        sequence = [self.stoi['<sos>']]
        for char in text:  # Iterate through characters
            if char in self.stoi:
                sequence.append(self.stoi[char])
        sequence.append(self.stoi['<eos>'])
        return sequence
    
    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption
    
    def predict_captions(self, sequences):
        return [self.predict_caption(sequence) for sequence in sequences]


class TrainDataset(Dataset):
    """Dataset class for training data"""
    
    def __init__(self, df, tokenizer, base_dir, transform=None):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.base_dir = base_dir
        self.image_ids = df['image_id'].values
        self.labels = df['InChI_text'].values
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_path = os.path.join(self.base_dir, 'images', f'{image_id}.png')
        
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)
        label_length = torch.LongTensor([len(label)])
        
        return image, torch.LongTensor(label), label_length


class TestDataset(Dataset):
    """Dataset class for test/validation data"""
    
    def __init__(self, df, base_dir, transform=None):
        super().__init__()
        self.df = df
        self.base_dir = base_dir
        self.image_ids = df['image_id'].values
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_path = os.path.join(self.base_dir, 'images', f'{image_id}.png')
        
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image


class Encoder(nn.Module):
    """CNN Encoder using timm models"""
    
    def __init__(self, model_name='resnet34', pretrained=True):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)
        self.n_features = self.cnn.fc.in_features
        self.cnn.global_pool = nn.Identity()
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        features = self.cnn(x)
        # Global average pooling to get a single feature vector per image
        features = features.mean(dim=[2, 3])  # Average over spatial dimensions
        return features


class GRUDecoder(nn.Module):
    """Simple GRU Decoder without attention mechanism"""

    def __init__(self, embed_dim, decoder_dim, vocab_size, device, encoder_dim=512, dropout=0.5, num_layers=2):
        super(GRUDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.device = device
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # GRU that takes embedded tokens as input
        self.gru = nn.GRU(embed_dim, decoder_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Linear layer to initialize hidden state from encoder features
        self.init_hidden = nn.Linear(encoder_dim, decoder_dim * num_layers)
        
        # Output projection layer
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """Initialize hidden state from encoder output"""
        batch_size = encoder_out.size(0)
        # Project encoder output to hidden state dimensions
        hidden = self.init_hidden(encoder_out)  # [batch_size, decoder_dim * num_layers]
        hidden = hidden.view(batch_size, self.num_layers, self.decoder_dim)  # [batch_size, num_layers, decoder_dim]
        hidden = hidden.transpose(0, 1).contiguous()  # [num_layers, batch_size, decoder_dim]
        return hidden

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size
        
        # Sort by caption length for efficient packing
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embed the captions
        embeddings = self.embedding(encoded_captions)
        
        # Initialize hidden state
        hidden = self.init_hidden_state(encoder_out)
        
        # Pack padded sequences for efficient RNN processing
        decode_lengths = (caption_lengths - 1).tolist()
        embeddings_packed = pack_padded_sequence(embeddings[:, :-1, :], decode_lengths, batch_first=True)
        
        # Forward through GRU
        gru_out, _ = self.gru(embeddings_packed, hidden)
        
        # Apply dropout and output projection
        gru_out_data = self.dropout_layer(gru_out.data)
        predictions = self.fc(gru_out_data)
        
        return predictions, encoded_captions, decode_lengths, None, sort_ind
    
    def predict(self, encoder_out, decode_lengths, tokenizer):
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size
        
        # Initialize hidden state
        hidden = self.init_hidden_state(encoder_out)
        
        # Start with <sos> tokens
        input_token = torch.ones(batch_size, dtype=torch.long).to(self.device) * tokenizer.stoi["<sos>"]
        
        predictions = []
        
        for t in range(decode_lengths):
            # Embed current input - make sure it's the right shape
            embedded = self.embedding(input_token)  # [batch_size, embed_dim]
            embedded = embedded.unsqueeze(1)  # [batch_size, 1, embed_dim] for GRU input
            
            # Forward through GRU
            gru_out, hidden = self.gru(embedded, hidden)
            
            # Apply dropout and get predictions
            gru_out = self.dropout_layer(gru_out)
            pred = self.fc(gru_out.squeeze(1))  # [batch_size, vocab_size]
            predictions.append(pred)
            
            # Get next input token (greedy decoding)
            next_token = torch.argmax(pred, dim=-1)  # [batch_size]
            
            # Check if all sequences have generated <eos>
            if (next_token == tokenizer.stoi["<eos>"]).all():
                break
                
            input_token = next_token  # [batch_size] for next iteration
            
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # [batch_size, seq_len, vocab_size]
        
        return predictions


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


def bms_collate(batch, tokenizer):
    """Custom collate function for DataLoader"""
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)


def get_transforms(cfg, data_type):
    """Get image transforms for training/validation"""
    return Compose([
        Resize(cfg.size, cfg.size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_score(y_true, y_pred):
    """Calculate normalized Levenshtein distance score (0-1 scale)"""
    scores = []
    for true, pred in zip(y_true, y_pred):
        distance = Levenshtein.distance(true, pred)
        max_length = max(len(true), len(pred))
        if max_length == 0:
            normalized_score = 0.0
        else:
            normalized_score = distance / max_length
        scores.append(normalized_score)
    return np.mean(scores)


def seed_torch(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_fn(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, cfg, device):
    """Training function for one epoch"""
    losses = AverageMeter()
    encoder.train()
    decoder.train()
    
    for step, (images, labels, label_lengths) in enumerate(tqdm(train_loader, desc="Training", disable=not is_tty)):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        features = encoder(images)
        predictions, caps_sorted, decode_lengths, _, sort_ind = decoder(features, labels, label_lengths)
        
        targets = caps_sorted[:, 1:]
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        
        loss = criterion(predictions, targets_packed)
        losses.update(loss.item(), images.size(0))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg.max_grad_norm)
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
    return losses.avg


def valid_fn(valid_loader, encoder, decoder, tokenizer, cfg, device):
    """Validation function"""
    encoder.eval()
    decoder.eval()
    text_preds = []
    
    with torch.no_grad():
        for images in tqdm(valid_loader, desc="Validation", disable=not is_tty):
            images = images.to(device)
            features = encoder(images)
            predictions = decoder.predict(features, cfg.max_len, tokenizer)
            predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
            _text_preds = tokenizer.predict_captions(predicted_sequence)
            text_preds.extend(_text_preds)
    
    return text_preds


def load_data(cfg):
    """Load and prepare data from CSV files"""
    print("Loading data...")
    
    # Load CSV files
    train_csv_path = os.path.join(cfg.base_dir, 'train.csv')
    valid_csv_path = os.path.join(cfg.base_dir, 'valid.csv')
    test_csv_path = os.path.join(cfg.base_dir, 'test.csv')
    
    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    print(f'Train data shape: {train_df.shape}')
    print(f'Valid data shape: {valid_df.shape}')
    print(f'Test data shape: {test_df.shape}')
    
    # Extract InChI text (remove "InChI=1S/" prefix for tokenization)
    train_df['InChI_text'] = train_df['InChI'].str.replace('InChI=1S/', '', regex=False)
    valid_df['InChI_text'] = valid_df['InChI'].str.replace('InChI=1S/', '', regex=False)
    
    # Create tokenizer from training data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['InChI_text'].values)
    
    print(f'Vocabulary size: {len(tokenizer)}')
    # print('valid_df', valid_df['InChI_text'].values)
    # print('tokenizer', tokenizer.stoi)
    # raise Exception('Stop here')
    
    return train_df, valid_df, test_df, tokenizer


def train_loop(train_df, valid_df, test_df, tokenizer, cfg, device):
    """Main training loop with early stopping on validation set"""
    print("========== Starting training ==========")
    
    # Datasets and dataloaders
    train_dataset = TrainDataset(train_df, tokenizer, cfg.base_dir, transform=get_transforms(cfg, 'train'))
    valid_dataset = TestDataset(valid_df, cfg.base_dir, transform=get_transforms(cfg, 'valid'))
    test_dataset = TestDataset(test_df, cfg.base_dir, transform=get_transforms(cfg, 'valid'))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory,
        drop_last=True, 
        collate_fn=lambda batch: bms_collate(batch, tokenizer)
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False
    )
    
    # Model
    encoder = Encoder(cfg.model_name, pretrained=True).to(device)
    decoder = GRUDecoder(
        embed_dim=cfg.embed_dim,
        decoder_dim=cfg.decoder_dim,
        vocab_size=len(tokenizer),
        dropout=cfg.dropout,
        device=device
    ).to(device)
    
    # Optimizers and scheduler
    encoder_optimizer = Adam(encoder.parameters(), lr=cfg.encoder_lr, weight_decay=cfg.weight_decay)
    decoder_optimizer = Adam(decoder.parameters(), lr=cfg.decoder_lr, weight_decay=cfg.weight_decay)
    
    encoder_scheduler = CosineAnnealingLR(encoder_optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr)
    decoder_scheduler = CosineAnnealingLR(decoder_optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])
    
    best_valid_score = np.inf
    best_encoder_state = None
    best_decoder_state = None
    valid_labels = valid_df['InChI'].values
    test_labels = test_df['InChI'].values
    
    for epoch in range(cfg.epochs):
        print(f'Epoch {epoch+1}/{cfg.epochs}')
        start_time = time.time()
        
        # Train
        avg_loss = train_fn(train_loader, encoder, decoder, criterion, 
                           encoder_optimizer, decoder_optimizer, cfg, device)
        
        # Validation
        valid_preds = valid_fn(valid_loader, encoder, decoder, tokenizer, cfg, device)
        valid_preds = [f"InChI=1S/{text}" for text in valid_preds]
        
        # Scoring on validation set
        valid_score = get_score(valid_labels, valid_preds)
        
        encoder_scheduler.step()
        decoder_scheduler.step()
        
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f} - Valid Score: {valid_score:.4f} - time: {elapsed:.0f}s')
        
        # Early stopping: save best model based on validation score
        if valid_score < best_valid_score:
            best_valid_score = valid_score
            best_encoder_state = encoder.state_dict().copy()
            best_decoder_state = decoder.state_dict().copy()
            print(f'Epoch {epoch+1} - New Best Validation Score: {best_valid_score:.4f}')
    
    # Load best model and evaluate on test set
    print("\n" + "="*30)
    print("Loading best model and evaluating on test set...")
    encoder.load_state_dict(best_encoder_state)
    decoder.load_state_dict(best_decoder_state)
    
    # Test evaluation
    test_preds = valid_fn(test_loader, encoder, decoder, tokenizer, cfg, device)
    test_preds = [f"InChI=1S/{text}" for text in test_preds]
    
    # Final scoring on test set
    test_score = get_score(test_labels, test_preds)
    
    print(f"Best Validation Score: {best_valid_score:.4f}")
    print(f"Final Test Score: {test_score:.4f}")
    print("="*30)
    
    return test_score


def main(cfg):
    """Main function to run the training and evaluation"""
    print("Starting Molecular Translation Model Training")
    
    # Setup
    seed_torch(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_df, valid_df, test_df, tokenizer = load_data(cfg)
    
    if cfg.debug:
        train_df = train_df.sample(n=min(1000, len(train_df)), random_state=cfg.seed).reset_index(drop=True)
        valid_df = valid_df.sample(n=min(200, len(valid_df)), random_state=cfg.seed).reset_index(drop=True)
        test_df = test_df.sample(n=min(200, len(test_df)), random_state=cfg.seed).reset_index(drop=True)
        print(f'Debug mode: reduced to {len(train_df)} train, {len(valid_df)} valid, and {len(test_df)} test samples')
    
    # Training
    final_test_score = train_loop(train_df, valid_df, test_df, tokenizer, cfg, device)

    return final_test_score

if __name__ == '__main__':
    # Set the base directory path and create config
    base_dir = '../../../data_cache/molecular_translation'
    cfg = Config(base_dir=base_dir)
    
    print("Configuration Settings:")
    print(f"Base directory: {cfg.base_dir}")
    print(f"Debug mode: {cfg.debug}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print("-" * 50)
    
    results = main(cfg)
    print(f"Final Test Levenshtein Distance: {results:.4f}")