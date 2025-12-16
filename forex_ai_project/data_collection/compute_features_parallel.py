"""
================================================================================
FEATURE ENGINEERING PARALLELIZZATO - NexNow LTD
================================================================================
Versione ottimizzata con multiprocessing per sfruttare tutti i core CPU.

Uso:
    python compute_features_parallel.py --input ./data/raw --output ./data/processed
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
import multiprocessing as mp
from functools import partial
import os

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "CADJPY", "CADCHF", "CHFJPY", "NZDJPY", "NZDCHF", "NZDCAD"
]

CURRENCIES = ['USD', 'EUR', 'GBP', 'CHF', 'CAD', 'JPY', 'AUD', 'NZD']

def get_currencies(pair: str) -> Tuple[str, str]:
    return pair[:3], pair[3:6]


# ============================================================================
# INDICATORI TECNICI (stessi del file originale)
# ============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    lowest_low = df['Low'].rolling(window=k_period).min()
    highest_high = df['High'].rolling(window=k_period).max()
    k = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


# ============================================================================
# CURRENCY STRENGTH (deve rimanere sequenziale - richiede tutti i dati)
# ============================================================================

def compute_currency_strength(
    all_pairs_data: Dict[str, pd.DataFrame],
    smoothing1: int = 20,
    smoothing2: int = 15,
    normalize: bool = True
) -> pd.DataFrame:
    """Calcola la forza relativa delle 8 valute principali."""

    common_index = None
    for pair, df in all_pairs_data.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    if len(common_index) == 0:
        raise ValueError("Nessun indice comune tra i pair")

    returns = {}
    for pair, df in all_pairs_data.items():
        df_aligned = df.loc[common_index]
        returns[pair] = df_aligned['Close'].pct_change()

    returns_df = pd.DataFrame(returns)
    strength = pd.DataFrame(index=common_index)

    for currency in CURRENCIES:
        currency_returns = []

        for pair in FOREX_PAIRS:
            if pair not in returns_df.columns:
                continue
            base, quote = get_currencies(pair)

            if base == currency:
                currency_returns.append(returns_df[pair])
            elif quote == currency:
                currency_returns.append(-returns_df[pair])

        if currency_returns:
            avg_return = pd.concat(currency_returns, axis=1).mean(axis=1)
            smoothed1 = avg_return.ewm(span=smoothing1, adjust=False).mean()
            smoothed2 = smoothed1.ewm(span=smoothing2, adjust=False).mean()
            strength[currency] = smoothed2

    if normalize:
        for col in strength.columns:
            rolling_min = strength[col].rolling(500).min()
            rolling_max = strength[col].rolling(500).max()
            range_val = rolling_max - rolling_min
            range_val = range_val.replace(0, 1)
            strength[col] = 200 * (strength[col] - rolling_min) / range_val - 100

    return strength


# ============================================================================
# FEATURE COMPUTATION PER SINGOLO PAIR (parallelizzabile)
# ============================================================================

def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola tutte le feature tecniche base per un singolo pair."""
    features = pd.DataFrame(index=df.index)

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df.get('Volume', pd.Series(0, index=df.index))

    # RETURNS
    for period in [1, 5, 10, 20, 50, 100]:
        features[f'return_{period}'] = close.pct_change(period)
    features['return_cum_20'] = close.pct_change(20).rolling(20).sum()

    # VOLATILITY
    for period in [10, 20, 50]:
        features[f'volatility_{period}'] = close.pct_change().rolling(period).std()
    features['volatility_ratio'] = features['volatility_10'] / features['volatility_50']

    # MOVING AVERAGES
    for period in [10, 20, 50, 100, 200]:
        sma = close.rolling(period).mean()
        ema = close.ewm(span=period, adjust=False).mean()
        features[f'sma_{period}'] = sma
        features[f'ema_{period}'] = ema
        features[f'dist_sma_{period}'] = (close - sma) / sma * 100
        features[f'dist_ema_{period}'] = (close - ema) / ema * 100

    features['sma_10_50_cross'] = (features['sma_10'] > features['sma_50']).astype(int)
    features['sma_20_100_cross'] = (features['sma_20'] > features['sma_100']).astype(int)

    # MOMENTUM - RSI
    for period in [7, 14, 21]:
        features[f'rsi_{period}'] = compute_rsi(close, period)

    # MACD
    macd, signal, hist = compute_macd(close)
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_hist'] = hist
    features['macd_cross'] = (macd > signal).astype(int)

    # Stochastic
    k, d = compute_stochastic(df)
    features['stoch_k'] = k
    features['stoch_d'] = d
    features['stoch_cross'] = (k > d).astype(int)

    # ATR
    for period in [7, 14, 21]:
        atr = compute_atr(df, period)
        features[f'atr_{period}'] = atr
        features[f'atr_pct_{period}'] = atr / close * 100

    # BOLLINGER BANDS
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close, 20, 2)
    features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
    features['bb_width'] = (bb_upper - bb_lower) / bb_mid * 100

    # PRICE ACTION
    features['body'] = close - df['Open']
    features['body_pct'] = features['body'] / df['Open'] * 100
    features['upper_wick'] = high - np.maximum(close, df['Open'])
    features['lower_wick'] = np.minimum(close, df['Open']) - low
    features['range'] = high - low
    features['body_to_range'] = features['body'].abs() / features['range']

    features['hh'] = (high > high.shift(1)).astype(int)
    features['ll'] = (low < low.shift(1)).astype(int)
    features['hh_streak'] = features['hh'].groupby((features['hh'] != features['hh'].shift()).cumsum()).cumsum()
    features['ll_streak'] = features['ll'].groupby((features['ll'] != features['ll'].shift()).cumsum()).cumsum()

    # VOLUME
    if volume.sum() > 0:
        features['volume_sma_20'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_sma_20']
        features['volume_trend'] = volume.rolling(10).mean() / volume.rolling(50).mean()

    # TIME FEATURES
    features['hour'] = df.index.hour
    features['day_of_week'] = df.index.dayofweek
    features['day_of_month'] = df.index.day
    features['month'] = df.index.month
    features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
    features['is_overlap'] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(int)

    return features


def compute_atr_dashboard(df: pd.DataFrame, ema_period: int = 21, atr_period: int = 14, history_bars: int = 500) -> pd.DataFrame:
    """ATR Dashboard - versione ottimizzata."""
    result = pd.DataFrame(index=df.index)

    ema = df['Close'].ewm(span=ema_period, adjust=False).mean()
    result['ema'] = ema

    atr = compute_atr(df, atr_period)
    result['atr'] = atr

    distance = df['Close'] - ema
    distance_atr = distance / atr
    result['ema_distance_atr'] = distance_atr
    result['ema_distance_atr_abs'] = distance_atr.abs()

    # Percentile ottimizzato con numpy
    abs_dist = result['ema_distance_atr_abs'].values
    percentiles = np.full(len(abs_dist), 50.0)

    for i in range(history_bars, len(abs_dist)):
        window = abs_dist[i-history_bars:i]
        percentiles[i] = (window < abs_dist[i]).sum() / history_bars * 100

    result['atr_percentile'] = percentiles
    result['excess_direction'] = np.sign(distance_atr)

    return result


def compute_ar_lines(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """AR Lines - versione ottimizzata."""
    result = pd.DataFrame(index=df.index)

    high_24 = df['High'].rolling(24).max()
    low_24 = df['Low'].rolling(24).min()
    daily_range = high_24 - low_24
    result['adr'] = daily_range.rolling(period * 24).mean()

    high_week = df['High'].rolling(120).max()
    low_week = df['Low'].rolling(120).min()
    weekly_range = high_week - low_week
    result['awr'] = weekly_range.rolling(period).mean()

    result['adr_high'] = df['Close'] + result['adr'] / 2
    result['adr_low'] = df['Close'] - result['adr'] / 2
    result['dist_to_adr_high_pct'] = (result['adr_high'] - df['Close']) / df['Close'] * 100
    result['dist_to_adr_low_pct'] = (df['Close'] - result['adr_low']) / df['Close'] * 100

    return result


def compute_seasonal_pattern_fast(df: pd.DataFrame, correlation_window: int = 60) -> pd.DataFrame:
    """
    Seasonal Pattern - versione MOLTO più veloce.
    Usa vectorized operations invece di loop.
    """
    result = pd.DataFrame(index=df.index)

    df = df.copy()
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear

    years = sorted(df['year'].unique())

    if len(years) < 3:
        result['seasonal_correlation'] = 0.0
        result['seasonal_direction'] = 0
        return result

    # Pre-calcola returns normalizzati per anno
    yearly_returns = {}
    for year in years:
        year_mask = df['year'] == year
        year_close = df.loc[year_mask, 'Close']
        if len(year_close) > 100:
            first_price = year_close.iloc[0]
            yearly_returns[year] = (year_close / first_price * 100).values

    # Calcola correlazione media con anni precedenti (solo per ultimo anno)
    current_year = years[-1]
    result['seasonal_correlation'] = 0.0
    result['seasonal_direction'] = 0

    # Semplificazione: calcola solo correlazione rolling con media storica
    historical_mean = None
    for year in years[:-1]:
        if year in yearly_returns:
            if historical_mean is None:
                historical_mean = yearly_returns[year].copy()
            else:
                min_len = min(len(historical_mean), len(yearly_returns[year]))
                historical_mean = (historical_mean[:min_len] + yearly_returns[year][:min_len]) / 2

    if historical_mean is not None:
        current_mask = df['year'] == current_year
        current_data = df.loc[current_mask, 'Close'].values
        first_price = current_data[0] if len(current_data) > 0 else 1
        current_norm = current_data / first_price * 100

        # Correlazione rolling semplificata
        corrs = np.zeros(len(df))
        min_len = min(len(current_norm), len(historical_mean))

        for i in range(correlation_window, min_len):
            curr_window = current_norm[i-correlation_window:i]
            hist_window = historical_mean[i-correlation_window:i]
            if len(curr_window) == len(hist_window):
                corr = np.corrcoef(curr_window, hist_window)[0, 1]
                if not np.isnan(corr):
                    corrs[df[current_mask].index.get_indexer_for(df[current_mask].index)[i] if i < sum(current_mask) else -1] = corr

        result['seasonal_correlation'] = corrs
        result['seasonal_direction'] = np.sign(corrs)

    return result


def process_single_pair(args):
    """
    Processa un singolo pair - funzione per multiprocessing.
    """
    pair, data_dir, output_dir, currency_strength_path, include_seasonal = args

    try:
        # Carica dati
        data_path = Path(data_dir) / f"{pair}_H1.parquet"
        if not data_path.exists():
            return pair, False, "File non trovato"

        df = pd.read_parquet(data_path)

        # Carica currency strength
        currency_strength = pd.read_parquet(currency_strength_path)

        features = pd.DataFrame(index=df.index)

        # Feature tecniche base
        base_features = compute_base_features(df)
        features = features.join(base_features)

        # Forza Relativa
        base, quote = get_currencies(pair)
        strength_aligned = currency_strength.reindex(df.index, method='ffill')

        if base in strength_aligned.columns and quote in strength_aligned.columns:
            fr_spread = strength_aligned[base] - strength_aligned[quote]
            features['fr_spread'] = fr_spread
            features['fr_spread_abs'] = fr_spread.abs()
            features[f'strength_{base}'] = strength_aligned[base]
            features[f'strength_{quote}'] = strength_aligned[quote]

        # ATR Dashboard
        atr_dash = compute_atr_dashboard(df, ema_period=21, atr_period=14, history_bars=500)
        features = features.join(atr_dash, rsuffix='_atrdash')

        # AR Lines
        ar_lines = compute_ar_lines(df, period=10)
        features = features.join(ar_lines, rsuffix='_ar')

        # Seasonal Pattern (versione veloce)
        if include_seasonal:
            try:
                seasonal = compute_seasonal_pattern_fast(df, correlation_window=60)
                features = features.join(seasonal, rsuffix='_seasonal')
            except Exception as e:
                pass

        # Pair ID
        features['pair_id'] = FOREX_PAIRS.index(pair)

        # Cleanup
        features = features.replace([np.inf, -np.inf], np.nan)

        # Salva
        output_path = Path(output_dir) / f"{pair}_features.parquet"
        features.to_parquet(output_path)

        return pair, True, features.shape[1]

    except Exception as e:
        return pair, False, str(e)


def process_single_label(args):
    """
    Processa labels per un singolo pair - funzione per multiprocessing.
    """
    pair, data_dir, features_dir, output_dir, config = args

    try:
        data_path = Path(data_dir) / f"{pair}_H1.parquet"
        features_path = Path(features_dir) / f"{pair}_features.parquet"

        if not data_path.exists() or not features_path.exists():
            return pair, False, "File non trovato"

        df = pd.read_parquet(data_path)
        features = pd.read_parquet(features_path)

        labels = pd.DataFrame(index=df.index)

        # Direction Label
        future_return = df['Close'].shift(-config['horizon']).pct_change(config['horizon'])

        labels['direction'] = 0
        labels.loc[future_return > config['direction_threshold'], 'direction'] = 1
        labels.loc[future_return < -config['direction_threshold'], 'direction'] = -1

        # Target Return
        labels['target_return'] = future_return

        # MFE/MAE
        future_high = df['High'].rolling(config['horizon']).max().shift(-config['horizon'])
        future_low = df['Low'].rolling(config['horizon']).min().shift(-config['horizon'])

        labels['mfe_long'] = (future_high - df['Close']) / df['Close']
        labels['mfe_short'] = (df['Close'] - future_low) / df['Close']
        labels['mae_long'] = (df['Close'] - future_low) / df['Close']
        labels['mae_short'] = (future_high - df['Close']) / df['Close']

        # Win Probability
        atr = features.get('atr_14', compute_atr(df, 14))
        sl = config['sl_atr_mult'] * atr
        tp = config['tp_atr_mult'] * atr

        long_tp_hit = labels['mfe_long'] * df['Close'] >= tp
        long_sl_hit = labels['mae_long'] * df['Close'] >= sl
        labels['long_win'] = (long_tp_hit & ~long_sl_hit).astype(int)

        short_tp_hit = labels['mfe_short'] * df['Close'] >= tp
        short_sl_hit = labels['mae_short'] * df['Close'] >= sl
        labels['short_win'] = (short_tp_hit & ~short_sl_hit).astype(int)

        # Optimal Action
        labels['optimal_action'] = 0
        labels.loc[labels['long_win'] == 1, 'optimal_action'] = 1
        labels.loc[labels['short_win'] == 1, 'optimal_action'] = -1

        # Salva
        labels.to_parquet(Path(output_dir) / f"{pair}_labels.parquet")

        return pair, True, len(labels)

    except Exception as e:
        return pair, False, str(e)


# ============================================================================
# MAIN PARALLEL FUNCTIONS
# ============================================================================

def compute_all_features_parallel(
    data_dir: str,
    output_dir: str,
    timeframe: str = "H1",
    include_seasonal: bool = True,
    n_workers: int = None
) -> Dict[str, pd.DataFrame]:
    """
    Calcola tutte le feature in parallelo usando tutti i core CPU.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = mp.cpu_count()

    logger.info(f"Usando {n_workers} workers (CPU cores)")

    # STEP 1: Carica tutti i dati (sequenziale, necessario per currency strength)
    logger.info("Caricamento dati...")
    all_data = {}
    for pair in FOREX_PAIRS:
        filepath = data_dir / f"{pair}_{timeframe}.parquet"
        if filepath.exists():
            all_data[pair] = pd.read_parquet(filepath)
            logger.info(f"  ✓ {pair}: {len(all_data[pair]):,} barre")

    if len(all_data) == 0:
        raise ValueError("Nessun dato trovato!")

    # STEP 2: Calcola Currency Strength (sequenziale, richiede tutti i dati)
    logger.info("Calcolo Currency Strength...")
    currency_strength = compute_currency_strength(all_data, smoothing1=20, smoothing2=15)
    currency_strength_path = output_dir / 'currency_strength.parquet'
    currency_strength.to_parquet(currency_strength_path)
    logger.info(f"  ✓ Currency Strength salvato")

    # Libera memoria
    del all_data

    # STEP 3: Calcola features in parallelo
    logger.info(f"Calcolo features in parallelo ({n_workers} workers)...")

    pairs_to_process = []
    for pair in FOREX_PAIRS:
        filepath = data_dir / f"{pair}_{timeframe}.parquet"
        if filepath.exists():
            pairs_to_process.append(pair)

    # Prepara argomenti per multiprocessing
    args_list = [
        (pair, str(data_dir), str(output_dir), str(currency_strength_path), include_seasonal)
        for pair in pairs_to_process
    ]

    # Esegui in parallelo
    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_pair, args_list),
            total=len(args_list),
            desc="Computing features"
        ))

    # Report risultati
    success_count = 0
    for pair, success, info in results:
        if success:
            logger.info(f"  ✓ {pair}: {info} features")
            success_count += 1
        else:
            logger.warning(f"  ✗ {pair}: {info}")

    logger.info(f"Completati {success_count}/{len(pairs_to_process)} pair")

    return {}


def create_labels_parallel(
    data_dir: str,
    features_dir: str,
    output_dir: str,
    config: dict = None,
    n_workers: int = None
) -> None:
    """
    Crea labels in parallelo.
    """
    if config is None:
        config = {
            'horizon': 24,
            'direction_threshold': 0.002,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 1.5,
        }

    if n_workers is None:
        n_workers = mp.cpu_count()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creazione labels in parallelo ({n_workers} workers)...")

    # Prepara argomenti
    args_list = [
        (pair, data_dir, features_dir, str(output_dir), config)
        for pair in FOREX_PAIRS
    ]

    # Esegui in parallelo
    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_label, args_list),
            total=len(args_list),
            desc="Creating labels"
        ))

    # Report
    success_count = sum(1 for _, success, _ in results if success)
    logger.info(f"Labels create per {success_count}/{len(FOREX_PAIRS)} pair")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute Features PARALLEL')
    parser.add_argument('--input', type=str, default='./data/raw', help='Directory dati raw')
    parser.add_argument('--output', type=str, default='./data/processed', help='Directory output')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe principale')
    parser.add_argument('--no-seasonal', action='store_true', help='Skip seasonal pattern')
    parser.add_argument('--labels', action='store_true', help='Crea anche le labels')
    parser.add_argument('--workers', type=int, default=None, help='Numero di workers (default: tutti i core)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PARALLELIZZATO")
    logger.info(f"CPU cores disponibili: {mp.cpu_count()}")
    logger.info("=" * 60)

    # Compute features
    compute_all_features_parallel(
        data_dir=args.input,
        output_dir=args.output,
        timeframe=args.timeframe,
        include_seasonal=not args.no_seasonal,
        n_workers=args.workers
    )

    # Create labels
    if args.labels:
        logger.info("\nCreazione labels...")
        create_labels_parallel(
            data_dir=args.input,
            features_dir=args.output,
            output_dir=args.output,
            n_workers=args.workers
        )

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETATO")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Windows richiede questo per multiprocessing
    mp.freeze_support()
    main()
