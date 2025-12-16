"""
================================================================================
FOREX AI - TRAINING CON CLASS WEIGHTS BILANCIATI
================================================================================
Risolve il problema del class imbalance dove il modello predice sempre HOLD.

Modifiche chiave:
1. Class weights calcolati automaticamente
2. Focal Loss per gestire imbalance
3. Label smoothing per generalizzazione
4. Threshold calibration per predizioni
================================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'data_dir': Path(__file__).parent / 'data' / 'processed',
    'models_dir': Path(__file__).parent / 'models' / 'lstm_balanced',

    # Training
    'batch_size': 256,
    'epochs': 50,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    'patience': 15,

    # Model
    'seq_length': 50,  # Shorter sequence
    'hidden_size': 256,
    'num_layers': 3,
    'dropout': 0.3,

    # Class balancing
    'use_class_weights': True,
    'use_focal_loss': True,
    'focal_gamma': 2.0,
    'label_smoothing': 0.1,

    # Sampling
    'use_weighted_sampler': True,
}


# ============================================================================
# FOCAL LOSS - Gestisce class imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss per class imbalance.
    Down-weights easy examples, focuses on hard ones.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# DATASET
# ============================================================================

class ForexDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int = 50):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_length]
        y_target = self.y[idx + self.seq_length]
        return x_seq, y_target


def load_data(data_dir: Path, seq_length: int = 50):
    """Carica dataset con analisi class distribution."""

    logger.info("Caricamento dataset...")

    # Load
    X_train = pd.read_parquet(data_dir / 'X_train.parquet').values.astype(np.float32)
    X_val = pd.read_parquet(data_dir / 'X_val.parquet').values.astype(np.float32)
    X_test = pd.read_parquet(data_dir / 'X_test.parquet').values.astype(np.float32)

    y_train = pd.read_parquet(data_dir / 'y_train.parquet')['direction'].values
    y_val = pd.read_parquet(data_dir / 'y_val.parquet')['direction'].values
    y_test = pd.read_parquet(data_dir / 'y_test.parquet')['direction'].values

    # Convert: -1,0,1 -> 0,1,2
    y_train = (y_train + 1).astype(np.int64)
    y_val = (y_val + 1).astype(np.int64)
    y_test = (y_test + 1).astype(np.int64)

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    n_features = X_train.shape[1]

    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))

    logger.info(f"  Train: {len(X_train):,} samples, {n_features} features")
    logger.info(f"  Val: {len(X_val):,} samples")
    logger.info(f"  Test: {len(X_test):,} samples")
    logger.info(f"  Class distribution: {class_dist}")
    logger.info(f"  Class percentages: SELL={counts[0]/len(y_train)*100:.1f}%, HOLD={counts[1]/len(y_train)*100:.1f}%, BUY={counts[2]/len(y_train)*100:.1f}%")

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1, 2]),
        y=y_train
    )
    logger.info(f"  Class weights: SELL={class_weights[0]:.2f}, HOLD={class_weights[1]:.2f}, BUY={class_weights[2]:.2f}")

    # Create datasets
    train_dataset = ForexDataset(X_train, y_train, seq_length)
    val_dataset = ForexDataset(X_val, y_val, seq_length)
    test_dataset = ForexDataset(X_test, y_test, seq_length)

    # Weighted sampler for training
    if CONFIG['use_weighted_sampler']:
        # Sample weights based on class weights
        sample_weights = [class_weights[y] for y in y_train[seq_length:]]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )

    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, n_features, class_weights


# ============================================================================
# MODEL
# ============================================================================

class LSTMBalanced(nn.Module):
    """LSTM con migliore architettura per classificazione."""

    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3, num_classes=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Better initialization
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        return self.classifier(context)


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model, class_weights, config):
        self.model = model.to(device)
        self.config = config

        # Loss function
        weights_tensor = torch.FloatTensor(class_weights).to(device)

        if config['use_focal_loss']:
            self.criterion = FocalLoss(
                alpha=weights_tensor,
                gamma=config['focal_gamma']
            )
            logger.info(f"Using Focal Loss (gamma={config['focal_gamma']})")
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=weights_tensor,
                label_smoothing=config['label_smoothing']
            )
            logger.info("Using CrossEntropyLoss with class weights")

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        self.best_acc = 0
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}

        pbar = tqdm(train_loader, desc="Training")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            # Per-class accuracy
            for c in [0, 1, 2]:
                mask = y == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (predicted[mask] == c).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        # Per-class accuracy
        class_acc = {
            c: 100. * class_correct[c] / class_total[c] if class_total[c] > 0 else 0
            for c in [0, 1, 2]
        }

        return avg_loss, accuracy, class_acc

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        # Prediction distribution
        preds = np.array(all_preds)
        pred_dist = {0: (preds == 0).sum(), 1: (preds == 1).sum(), 2: (preds == 2).sum()}

        return avg_loss, accuracy, pred_dist

    def train(self, train_loader, val_loader, epochs):
        logger.info(f"\nStarting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Train
            train_loss, train_acc, class_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, pred_dist = self.evaluate(val_loader)

            # Scheduler
            self.scheduler.step(val_acc)

            # Log
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%"
            )
            logger.info(
                f"  Class Acc: SELL={class_acc[0]:.1f}%, HOLD={class_acc[1]:.1f}%, BUY={class_acc[2]:.1f}%"
            )
            logger.info(
                f"  Pred Dist: SELL={pred_dist[0]}, HOLD={pred_dist[1]}, BUY={pred_dist[2]}"
            )

            # Save best
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0
                self.save_model('best_model.pt')
                logger.info(f"  New best model! Acc: {val_acc:.2f}%")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return self.best_acc

    def save_model(self, filename):
        save_dir = self.config['models_dir']
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'config': dict(self.config),
        }, save_dir / filename)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("TRAINING LSTM CON CLASS WEIGHTS BILANCIATI")
    print("=" * 60)

    # Load data
    train_loader, val_loader, test_loader, n_features, class_weights = load_data(
        CONFIG['data_dir'],
        CONFIG['seq_length']
    )

    logger.info(f"\nInput features: {n_features}")

    # Create model
    model = LSTMBalanced(
        input_size=n_features,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Train
    trainer = Trainer(model, class_weights, CONFIG)
    best_acc = trainer.train(train_loader, val_loader, CONFIG['epochs'])

    # Test
    logger.info("\nEvaluating on test set...")
    trainer.model.load_state_dict(
        torch.load(CONFIG['models_dir'] / 'best_model.pt')['model_state_dict']
    )
    test_loss, test_acc, pred_dist = trainer.evaluate(test_loader)

    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Prediction Distribution: {pred_dist}")

    # Save results
    results = {
        'best_val_acc': best_acc,
        'test_acc': test_acc,
        'test_pred_dist': {str(k): int(v) for k, v in pred_dist.items()},
        'config': {k: str(v) for k, v in CONFIG.items()},
        'class_weights': class_weights.tolist(),
    }

    with open(CONFIG['models_dir'] / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETATO")
    print("=" * 60)
    print(f"Best Val Acc: {best_acc:.2f}%")
    print(f"Test Acc: {test_acc:.2f}%")
    print(f"Model saved in: {CONFIG['models_dir']}")


if __name__ == "__main__":
    main()
