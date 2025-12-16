"""
================================================================================
PREPARE TRAINING DATA - NexNow LTD
================================================================================
Combina feature e labels da tutti i pair e crea train/val/test split.
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurazione
PROCESSED_DIR = Path(__file__).parent / 'data' / 'processed'
OUTPUT_DIR = Path(__file__).parent / 'data' / 'processed'

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "CADJPY", "CADCHF", "CHFJPY", "NZDJPY", "NZDCHF", "NZDCAD"
]


def main():
    logger.info("=" * 60)
    logger.info("PREPARAZIONE DATI TRAINING")
    logger.info("=" * 60)

    all_X = []
    all_y = []

    # Carica e combina tutti i pair
    for pair in tqdm(FOREX_PAIRS, desc="Loading pairs"):
        features_path = PROCESSED_DIR / f"{pair}_features.parquet"
        labels_path = PROCESSED_DIR / f"{pair}_labels.parquet"

        if not features_path.exists() or not labels_path.exists():
            logger.warning(f"Skip {pair}: file mancanti")
            continue

        features = pd.read_parquet(features_path)
        labels = pd.read_parquet(labels_path)

        # Allinea indici
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        # Rimuovi righe con NaN
        valid_mask = ~(features.isna().any(axis=1) | labels['direction'].isna())
        features = features[valid_mask]
        labels = labels[valid_mask]

        # Rimuovi colonne non numeriche
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features = features[numeric_cols]

        all_X.append(features.values)
        all_y.append(labels['direction'].values)

        logger.info(f"  {pair}: {len(features):,} samples, {features.shape[1]} features")

    # Combina
    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    logger.info(f"\nTotale: {len(X):,} samples, {X.shape[1]} features")

    # Gestisci NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Salva nomi colonne
    feature_names = features.columns.tolist()
    pd.DataFrame({'feature': feature_names}).to_parquet(OUTPUT_DIR / 'feature_names.parquet')

    # Split: 70% train, 15% val, 15% test
    # Split temporale: usa indici ordinati
    n_samples = len(X)
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    logger.info(f"\nSplit:")
    logger.info(f"  Train: {len(X_train):,} ({len(X_train)/n_samples*100:.1f}%)")
    logger.info(f"  Val: {len(X_val):,} ({len(X_val)/n_samples*100:.1f}%)")
    logger.info(f"  Test: {len(X_test):,} ({len(X_test)/n_samples*100:.1f}%)")

    # Distribuzione classi
    for name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        dist = dict(zip(unique, counts))
        total = sum(counts)
        logger.info(f"  {name} class dist: SELL={dist.get(-1, 0)} ({dist.get(-1, 0)/total*100:.1f}%), " +
                   f"HOLD={dist.get(0, 0)} ({dist.get(0, 0)/total*100:.1f}%), " +
                   f"BUY={dist.get(1, 0)} ({dist.get(1, 0)/total*100:.1f}%)")

    # Salva come DataFrame per compatibilita' con train_balanced.py
    pd.DataFrame(X_train).to_parquet(OUTPUT_DIR / 'X_train.parquet')
    pd.DataFrame(X_val).to_parquet(OUTPUT_DIR / 'X_val.parquet')
    pd.DataFrame(X_test).to_parquet(OUTPUT_DIR / 'X_test.parquet')

    # Labels devono avere colonna 'direction'
    pd.DataFrame({'direction': y_train}).to_parquet(OUTPUT_DIR / 'y_train.parquet')
    pd.DataFrame({'direction': y_val}).to_parquet(OUTPUT_DIR / 'y_val.parquet')
    pd.DataFrame({'direction': y_test}).to_parquet(OUTPUT_DIR / 'y_test.parquet')

    logger.info(f"\nFile salvati in: {OUTPUT_DIR}")
    logger.info("=" * 60)
    logger.info("COMPLETATO")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
