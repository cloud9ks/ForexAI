#!/usr/bin/env python3
"""
================================================================================
FOREX AI - OVERNIGHT RUN
================================================================================
Script ottimizzato per esecuzione notturna.
Usa tutti i core CPU per velocizzare il processing.

Esegue:
1. Feature Engineering (parallelizzato)
2. Creazione Labels (parallelizzato)
3. Creazione Dataset Training

Uso:
    python run_overnight.py
================================================================================
"""

import os
import sys
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import logging
import json

# Setup logging con file
log_file = f"overnight_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

PROJECT_CONFIG = {
    'name': 'Forex AI Trading Agent',
    'version': '1.0.0',
    'base_dir': Path(__file__).parent,
    'data_dir': Path(__file__).parent / 'data',
    'raw_dir': Path(__file__).parent / 'data' / 'raw',
    'processed_dir': Path(__file__).parent / 'data' / 'processed',
    'models_dir': Path(__file__).parent / 'models',

    # Training settings
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'seq_length': 100,
    'horizon': 24,
}

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "CADJPY", "CADCHF", "CHFJPY", "NZDJPY", "NZDCHF", "NZDCAD"
]


def run_feature_engineering():
    """Esegue feature engineering parallelizzato."""
    logger.info("=" * 60)
    logger.info("FASE 1: FEATURE ENGINEERING (PARALLELIZZATO)")
    logger.info(f"CPU cores: {mp.cpu_count()}")
    logger.info("=" * 60)

    from data_collection.compute_features_parallel import (
        compute_all_features_parallel,
        create_labels_parallel
    )

    # Compute features
    compute_all_features_parallel(
        data_dir=str(PROJECT_CONFIG['raw_dir']),
        output_dir=str(PROJECT_CONFIG['processed_dir']),
        timeframe='H1',
        include_seasonal=True,
        n_workers=mp.cpu_count()
    )

    # Create labels
    create_labels_parallel(
        data_dir=str(PROJECT_CONFIG['raw_dir']),
        features_dir=str(PROJECT_CONFIG['processed_dir']),
        output_dir=str(PROJECT_CONFIG['processed_dir']),
        n_workers=mp.cpu_count()
    )

    logger.info("Feature engineering completato!")


def create_training_dataset():
    """Crea dataset finale per training."""
    logger.info("=" * 60)
    logger.info("FASE 2: CREAZIONE DATASET TRAINING")
    logger.info("=" * 60)

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import joblib

    processed_dir = PROJECT_CONFIG['processed_dir']

    # Carica tutti i features e labels
    all_features = []
    all_labels = []

    for pair in FOREX_PAIRS:
        features_path = processed_dir / f"{pair}_features.parquet"
        labels_path = processed_dir / f"{pair}_labels.parquet"

        if not features_path.exists() or not labels_path.exists():
            logger.warning(f"Dati mancanti per {pair}")
            continue

        features = pd.read_parquet(features_path)
        labels = pd.read_parquet(labels_path)

        # Allinea
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        all_features.append(features)
        all_labels.append(labels)
        logger.info(f"  ✓ {pair}: {len(features):,} samples")

    # Concatena
    X = pd.concat(all_features)
    y = pd.concat(all_labels)

    logger.info(f"Dataset totale: {len(X):,} samples, {X.shape[1]} features")

    # Remove infinities and fill NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns with too many NaN
    nan_pct = X.isnull().sum() / len(X)
    cols_to_drop = nan_pct[nan_pct > 0.5].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >50% NaN")
        X = X.drop(columns=cols_to_drop)

    # Fill remaining NaN
    X = X.ffill().bfill().fillna(0)

    # Temporal split
    train_end = X.index[int(len(X) * PROJECT_CONFIG['train_split'])]
    val_end = X.index[int(len(X) * (PROJECT_CONFIG['train_split'] + PROJECT_CONFIG['val_split']))]

    X_train = X[X.index < train_end]
    y_train = y[y.index < train_end]

    X_val = X[(X.index >= train_end) & (X.index < val_end)]
    y_val = y[(y.index >= train_end) & (y.index < val_end)]

    X_test = X[X.index >= val_end]
    y_test = y[y.index >= val_end]

    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Normalize features
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[numeric_cols]),
        index=X_train.index,
        columns=numeric_cols
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val[numeric_cols]),
        index=X_val.index,
        columns=numeric_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[numeric_cols]),
        index=X_test.index,
        columns=numeric_cols
    )

    # Save
    output_dir = PROJECT_CONFIG['processed_dir']

    X_train_scaled.to_parquet(output_dir / 'X_train.parquet')
    X_val_scaled.to_parquet(output_dir / 'X_val.parquet')
    X_test_scaled.to_parquet(output_dir / 'X_test.parquet')

    y_train.to_parquet(output_dir / 'y_train.parquet')
    y_val.to_parquet(output_dir / 'y_val.parquet')
    y_test.to_parquet(output_dir / 'y_test.parquet')

    # Save scaler
    joblib.dump(scaler, output_dir / 'scaler.joblib')

    # Save feature names
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(numeric_cols, f)

    # Save config
    config = {
        'n_features': len(numeric_cols),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'train_end': str(train_end),
        'val_end': str(val_end),
        'created_at': str(datetime.now()),
    }
    with open(output_dir / 'dataset_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"\n✅ Dataset salvato in {output_dir}")
    logger.info(f"   Features: {len(numeric_cols)}")

    return config


def main():
    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("FOREX AI - OVERNIGHT RUN")
    logger.info(f"Start: {start_time}")
    logger.info(f"CPU cores: {mp.cpu_count()}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)

    try:
        # Fase 1: Feature Engineering
        run_feature_engineering()

        # Fase 2: Dataset Creation
        create_training_dataset()

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "=" * 60)
        logger.info("✅ OVERNIGHT RUN COMPLETATO CON SUCCESSO!")
        logger.info(f"Durata totale: {duration}")
        logger.info("=" * 60)
        logger.info("\nProssimi passi:")
        logger.info("  1. Verifica i dati in ./data/processed/")
        logger.info("  2. Esegui training: python train_model.py")

    except Exception as e:
        logger.error(f"ERRORE: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    mp.freeze_support()  # Necessario per Windows
    sys.exit(main())
