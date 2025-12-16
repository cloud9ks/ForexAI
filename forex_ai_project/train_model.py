#!/usr/bin/env python3
"""
================================================================================
FOREX AI - MODEL TRAINING (GPU Optimized)
================================================================================
Training ottimizzato per GPU NVIDIA RTX 4060.

Modelli disponibili:
1. LSTM - Baseline veloce
2. Transformer - Attenzione temporale
3. TFT-like - Temporal Fusion per interpretabilita

Uso:
    python train_model.py                    # Default: LSTM
    python train_model.py --model transformer
    python train_model.py --model tft
================================================================================
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

CONFIG = {
    'data_dir': Path(__file__).parent / 'data' / 'processed',
    'models_dir': Path(__file__).parent / 'models',

    # Training
    'batch_size': 512,          # Ottimizzato per RTX 4060 8GB
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 10,             # Early stopping

    # Model
    'seq_length': 100,          # Finestra temporale
    'hidden_size': 256,
    'num_layers': 3,
    'dropout': 0.2,

    # GPU
    'num_workers': 4,           # DataLoader workers
    'pin_memory': True,         # Faster GPU transfer
}


# ============================================================================
# DATASET
# ============================================================================

class ForexDataset(Dataset):
    """Dataset per time series forex."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_length: int = 100
    ):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        # Sequenza di input
        x_seq = self.X[idx:idx + self.seq_length]
        # Target: direzione al tempo t+seq_length
        y_target = self.y[idx + self.seq_length]
        return x_seq, y_target


def load_data(data_dir: Path, seq_length: int = 100) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Carica i dataset e crea DataLoader."""

    logger.info("Caricamento dataset...")

    # Load data
    X_train = pd.read_parquet(data_dir / 'X_train.parquet').values
    X_val = pd.read_parquet(data_dir / 'X_val.parquet').values
    X_test = pd.read_parquet(data_dir / 'X_test.parquet').values

    y_train = pd.read_parquet(data_dir / 'y_train.parquet')['direction'].values
    y_val = pd.read_parquet(data_dir / 'y_val.parquet')['direction'].values
    y_test = pd.read_parquet(data_dir / 'y_test.parquet')['direction'].values

    # Converti labels: -1,0,1 -> 0,1,2 per CrossEntropyLoss
    y_train = y_train + 1
    y_val = y_val + 1
    y_test = y_test + 1

    n_features = X_train.shape[1]

    logger.info(f"  Train: {len(X_train):,} samples, {n_features} features")
    logger.info(f"  Val: {len(X_val):,} samples")
    logger.info(f"  Test: {len(X_test):,} samples")

    # Create datasets
    train_dataset = ForexDataset(X_train, y_train, seq_length)
    val_dataset = ForexDataset(X_val, y_val, seq_length)
    test_dataset = ForexDataset(X_test, y_test, seq_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )

    return train_loader, val_loader, test_loader, n_features


# ============================================================================
# MODELLI
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM per trading - Baseline veloce."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden*2]

        # Attention
        attn_weights = self.attention(lstm_out)  # [batch, seq, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden*2]

        # Classification
        out = self.classifier(context)
        return out


class TransformerModel(nn.Module):
    """Transformer per trading - Cattura pattern complessi."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        num_classes: int = 3
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_size) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)  # [batch, seq, hidden]

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer
        x = self.transformer(x)  # [batch, seq, hidden]

        # Use last token for classification
        x = x[:, -1, :]  # [batch, hidden]

        # Classification
        out = self.classifier(x)
        return out


class TFTModel(nn.Module):
    """
    Simplified Temporal Fusion Transformer.
    Combina LSTM + Attention per interpretabilita.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 3
    ):
        super().__init__()

        # Variable selection network (simplified)
        self.var_selection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1)
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Gated residual network
        self.grn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Variable selection
        var_weights = self.var_selection(x)  # [batch, seq, input]
        x = x * var_weights

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden]

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Gated residual
        grn_out = self.grn(attn_out)
        gate = self.gate(attn_out)
        out = self.layer_norm(lstm_out + gate * grn_out)

        # Use last timestep
        out = out[:, -1, :]

        # Classification
        out = self.classifier(out)
        return out


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Training manager con early stopping e checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: Path,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Mixed precision for faster training
        self.scaler = torch.amp.GradScaler('cuda')

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train per una epoca."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc='Training')
        for X, y in pbar:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad()

            # Mixed precision forward
            with torch.amp.autocast('cuda'):
                outputs = self.model(X)
                loss = self.criterion(outputs, y)

            # Backward with scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validazione."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for X, y in val_loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).long()

            with torch.amp.autocast('cuda'):
                outputs = self.model(X)
                loss = self.criterion(outputs, y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10
    ) -> Dict:
        """Training completo con early stopping."""

        logger.info(f"Training per {epochs} epoche (patience={patience})")
        logger.info(f"Device: {self.device}")

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                logger.info(f"  -> Nuovo best model salvato!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.load_checkpoint('best_model.pt')

        return self.history

    def save_checkpoint(self, filename: str):
        """Salva checkpoint."""
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, path)

    def load_checkpoint(self, filename: str):
        """Carica checkpoint."""
        path = self.save_dir / filename
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """Valutazione finale del modello."""

    model.eval()
    all_preds = []
    all_labels = []

    for X, y in tqdm(test_loader, desc='Evaluating'):
        X = X.to(device)
        y = y.to(device).long()

        outputs = model(X)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    accuracy = (all_preds == all_labels).mean() * 100

    # Per-class accuracy
    class_names = ['SELL', 'HOLD', 'BUY']
    class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc[name] = (all_preds[mask] == i).mean() * 100
        else:
            class_acc[name] = 0

    results = {
        'accuracy': accuracy,
        'class_accuracy': class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Forex AI Model')
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'tft'],
                       help='Model type')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--batch-size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=CONFIG['learning_rate'])
    parser.add_argument('--seq-length', type=int, default=CONFIG['seq_length'])

    args = parser.parse_args()

    # Update config
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    CONFIG['learning_rate'] = args.lr
    CONFIG['seq_length'] = args.seq_length

    logger.info("=" * 60)
    logger.info("FOREX AI - MODEL TRAINING")
    logger.info("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        logger.info("WARNING: GPU non disponibile, usando CPU")

    # Load data
    train_loader, val_loader, test_loader, n_features = load_data(
        CONFIG['data_dir'],
        CONFIG['seq_length']
    )

    # Create model
    logger.info(f"\nCreazione modello: {args.model.upper()}")

    if args.model == 'lstm':
        model = LSTMModel(
            input_size=n_features,
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout']
        )
    elif args.model == 'transformer':
        model = TransformerModel(
            input_size=n_features,
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout']
        )
    elif args.model == 'tft':
        model = TFTModel(
            input_size=n_features,
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout']
        )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parametri: {n_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        save_dir=CONFIG['models_dir'] / args.model,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Train
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)

    start_time = datetime.now()
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=CONFIG['epochs'],
        patience=CONFIG['patience']
    )
    training_time = datetime.now() - start_time

    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    results = evaluate_model(model, test_loader, device)

    logger.info(f"\nTest Accuracy: {results['accuracy']:.2f}%")
    logger.info("Per-class accuracy:")
    for name, acc in results['class_accuracy'].items():
        logger.info(f"  {name}: {acc:.2f}%")

    # Save results
    results_path = CONFIG['models_dir'] / args.model / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'model': args.model,
            'accuracy': results['accuracy'],
            'class_accuracy': results['class_accuracy'],
            'training_time': str(training_time),
            'config': {k: str(v) for k, v in CONFIG.items()},
            'n_params': n_params
        }, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETATO")
    logger.info(f"Training time: {training_time}")
    logger.info(f"Model salvato in: {CONFIG['models_dir'] / args.model}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
