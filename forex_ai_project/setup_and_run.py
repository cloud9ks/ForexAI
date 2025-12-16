#!/usr/bin/env python3
"""
================================================================================
FOREX AI TRADING AGENT - SETUP & RUN
================================================================================
NexNow LTD

Script principale che coordina tutto il processo:
1. Verifica ambiente
2. Scarica dati
3. Calcola features
4. Crea dataset per training
5. (Opzionale) Lancia training

Uso:
    python setup_and_run.py --all          # Esegue tutto
    python setup_and_run.py --download     # Solo download dati
    python setup_and_run.py --features     # Solo feature engineering
    python setup_and_run.py --dataset      # Solo creazione dataset
================================================================================
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAZIONE PROGETTO
# ============================================================================

PROJECT_CONFIG = {
    'name': 'Forex AI Trading Agent',
    'version': '1.0.0',
    'author': 'NexNow LTD',
    
    # Directory
    'base_dir': Path(__file__).parent,
    'data_dir': Path(__file__).parent / 'data',
    'raw_dir': Path(__file__).parent / 'data' / 'raw',
    'processed_dir': Path(__file__).parent / 'data' / 'processed',
    'external_dir': Path(__file__).parent / 'data' / 'external',
    'models_dir': Path(__file__).parent / 'models',
    'logs_dir': Path(__file__).parent / 'logs',
    
    # Data settings
    'years_history': 10,
    'timeframes': ['H1', 'H4', 'D1'],
    'main_timeframe': 'H1',
    
    # Training settings
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'seq_length': 100,
    'horizon': 24,
}


def check_environment():
    """Verifica che l'ambiente sia configurato correttamente"""
    logger.info("=" * 60)
    logger.info("VERIFICA AMBIENTE")
    logger.info("=" * 60)
    
    issues = []
    
    # Python version
    py_version = sys.version_info
    logger.info(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or py_version.minor < 9:
        issues.append("Python 3.9+ richiesto")
    
    # Check packages
    required_packages = [
        'pandas', 'numpy', 'torch', 'sklearn', 
        'tqdm', 'pyarrow'
    ]
    
    for pkg in required_packages:
        try:
            __import__(pkg)
            logger.info(f"  ✓ {pkg}")
        except ImportError:
            logger.warning(f"  ✗ {pkg} - NON INSTALLATO")
            issues.append(f"Package mancante: {pkg}")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  ✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("  ✗ GPU: Non disponibile (training sarà lento)")
    except:
        logger.warning("  ✗ PyTorch non installato o errore GPU")
    
    # Check MT5 (solo su Windows)
    if sys.platform == 'win32':
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                account = mt5.account_info()
                logger.info(f"  ✓ MT5: Connesso (Account: {account.login})")
                mt5.shutdown()
            else:
                issues.append("MT5 non può inizializzarsi")
        except ImportError:
            issues.append("MetaTrader5 package non installato")
    else:
        logger.info("  ⚠ MT5 disponibile solo su Windows")
    
    # Check directories
    for name, path in [
        ('Data', PROJECT_CONFIG['data_dir']),
        ('Raw', PROJECT_CONFIG['raw_dir']),
        ('Processed', PROJECT_CONFIG['processed_dir']),
        ('Models', PROJECT_CONFIG['models_dir']),
    ]:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✓ Directory {name}: {path}")
    
    if issues:
        logger.error("\n⚠️  PROBLEMI RILEVATI:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("\n✅ Ambiente OK!")
    return True


def download_data(years: int = 10):
    """Scarica tutti i dati forex"""
    logger.info("=" * 60)
    logger.info("DOWNLOAD DATI")
    logger.info("=" * 60)
    
    # Check if on Windows (required for MT5)
    if sys.platform != 'win32':
        logger.error("MT5 richiede Windows. Usa dati alternativi o esegui su Windows.")
        logger.info("Alternativa: usa download_alternative_data() per dati da yfinance")
        return download_alternative_data(years)
    
    from data_collection.download_forex_data import ForexDataDownloader
    
    downloader = ForexDataDownloader(output_dir=str(PROJECT_CONFIG['raw_dir']))
    
    report = downloader.download_all(
        timeframes=PROJECT_CONFIG['timeframes'],
        years=years
    )
    
    return report


def download_alternative_data(years: int = 10):
    """Download dati da fonti alternative (non MT5)"""
    logger.info("Download da fonti alternative...")
    
    import yfinance as yf
    import pandas as pd
    
    # Mapping forex pairs a Yahoo Finance tickers
    yf_pairs = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'USDCHF': 'USDCHF=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
        'NZDUSD': 'NZDUSD=X',
        'EURGBP': 'EURGBP=X',
        'EURJPY': 'EURJPY=X',
        # ... altri pairs
    }
    
    output_dir = PROJECT_CONFIG['raw_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pair, ticker in yf_pairs.items():
        logger.info(f"Downloading {pair}...")
        try:
            data = yf.download(
                ticker,
                period=f"{years}y",
                interval="1h",
                progress=False
            )
            
            if len(data) > 0:
                data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                data.to_parquet(output_dir / f"{pair}_H1.parquet")
                logger.info(f"  ✓ {pair}: {len(data)} barre")
        except Exception as e:
            logger.error(f"  ✗ {pair}: {e}")
    
    logger.info("Download alternativo completato")


def compute_features():
    """Calcola tutte le features"""
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    from data_collection.compute_features import compute_all_features, create_labels
    
    # Compute features
    features = compute_all_features(
        data_dir=str(PROJECT_CONFIG['raw_dir']),
        output_dir=str(PROJECT_CONFIG['processed_dir']),
        timeframe=PROJECT_CONFIG['main_timeframe'],
        include_seasonal=True
    )
    
    # Create labels
    create_labels(
        data_dir=str(PROJECT_CONFIG['raw_dir']),
        features_dir=str(PROJECT_CONFIG['processed_dir']),
        output_dir=str(PROJECT_CONFIG['processed_dir']),
        config={
            'horizon': PROJECT_CONFIG['horizon'],
            'direction_threshold': 0.002,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 1.5,
        }
    )
    
    return features


def create_training_dataset():
    """Crea dataset finale per training"""
    logger.info("=" * 60)
    logger.info("CREAZIONE DATASET TRAINING")
    logger.info("=" * 60)
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    processed_dir = PROJECT_CONFIG['processed_dir']
    
    # Carica tutti i features e labels
    all_features = []
    all_labels = []
    
    from data_collection.compute_features import FOREX_PAIRS
    
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
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
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
    # Seleziona solo colonne numeriche
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
    logger.info(f"   Scaler: scaler.joblib")
    
    return config


def run_all():
    """Esegue l'intera pipeline"""
    logger.info("=" * 60)
    logger.info("FOREX AI TRADING AGENT - SETUP COMPLETO")
    logger.info(f"Versione: {PROJECT_CONFIG['version']}")
    logger.info(f"Data: {datetime.now()}")
    logger.info("=" * 60)
    
    # 1. Check environment
    if not check_environment():
        logger.error("Ambiente non configurato correttamente. Risolvi i problemi e riprova.")
        return False
    
    # 2. Download data
    try:
        download_data(years=PROJECT_CONFIG['years_history'])
    except Exception as e:
        logger.error(f"Errore download dati: {e}")
        return False
    
    # 3. Compute features
    try:
        compute_features()
    except Exception as e:
        logger.error(f"Errore feature engineering: {e}")
        return False
    
    # 4. Create dataset
    try:
        create_training_dataset()
    except Exception as e:
        logger.error(f"Errore creazione dataset: {e}")
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ SETUP COMPLETATO CON SUCCESSO!")
    logger.info("=" * 60)
    logger.info("\nProssimi passi:")
    logger.info("  1. Verifica i dati in ./data/processed/")
    logger.info("  2. Esegui training: python train_model.py")
    logger.info("  3. Testa su demo account prima di andare live")
    
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Forex AI Trading Agent - Setup')
    parser.add_argument('--all', action='store_true', help='Esegue tutto il setup')
    parser.add_argument('--check', action='store_true', help='Solo verifica ambiente')
    parser.add_argument('--download', action='store_true', help='Solo download dati')
    parser.add_argument('--features', action='store_true', help='Solo feature engineering')
    parser.add_argument('--dataset', action='store_true', help='Solo creazione dataset')
    parser.add_argument('--years', type=int, default=10, help='Anni di storico')
    
    args = parser.parse_args()
    
    # Update config
    PROJECT_CONFIG['years_history'] = args.years
    
    if args.all:
        run_all()
    elif args.check:
        check_environment()
    elif args.download:
        check_environment()
        download_data(args.years)
    elif args.features:
        compute_features()
    elif args.dataset:
        create_training_dataset()
    else:
        parser.print_help()
        print("\nEsempi:")
        print("  python setup_and_run.py --all           # Setup completo")
        print("  python setup_and_run.py --check         # Solo verifica")
        print("  python setup_and_run.py --download      # Solo download")
        print("  python setup_and_run.py --years 5       # 5 anni di storico")


if __name__ == "__main__":
    main()
