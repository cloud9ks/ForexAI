"""
================================================================================
FEATURE ENGINEERING - NexNow LTD
================================================================================
Calcola tutte le feature per il training del modello AI, inclusi:
- Indicatori tecnici standard
- Indicatori custom (Forza Relativa, ATR Dashboard, Seasonal Oracle, AR Lines)
- Feature multi-timeframe
- Labels per training

Requisiti:
    pip install pandas numpy ta scikit-learn pyarrow tqdm

Uso:
    python compute_features.py --input ./data/raw --output ./data/processed
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
warnings.filterwarnings('ignore')

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

# Mappa pair -> (base, quote)
def get_currencies(pair: str) -> Tuple[str, str]:
    return pair[:3], pair[3:6]


# ============================================================================
# INDICATORI TECNICI STANDARD
# ============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator"""
    lowest_low = df['Low'].rolling(window=k_period).min()
    highest_high = df['High'].rolling(window=k_period).max()
    
    k = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    
    return k, d


# ============================================================================
# INDICATORI CUSTOM - SISTEMA FORZA RELATIVA
# ============================================================================

def compute_currency_strength(
    all_pairs_data: Dict[str, pd.DataFrame],
    smoothing1: int = 20,
    smoothing2: int = 15,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Calcola la forza relativa delle 8 valute principali.
    
    Logica:
    - Per ogni valuta, calcola la media dei movimenti % di tutti i pair dove appare
    - Applica doppio smoothing EMA
    - Normalizza tra -100 e +100
    
    Returns:
        DataFrame con colonne per ogni valuta (USD, EUR, GBP, etc.)
    """
    # Allinea tutti i dati su un indice comune
    common_index = None
    for pair, df in all_pairs_data.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    if len(common_index) == 0:
        raise ValueError("Nessun indice comune tra i pair")
    
    # Calcola returns per ogni pair
    returns = {}
    for pair, df in all_pairs_data.items():
        df_aligned = df.loc[common_index]
        returns[pair] = df_aligned['Close'].pct_change()
    
    returns_df = pd.DataFrame(returns)
    
    # Calcola forza per ogni valuta
    strength = pd.DataFrame(index=common_index)
    
    for currency in CURRENCIES:
        # Trova tutti i pair che contengono questa valuta
        currency_returns = []
        
        for pair in FOREX_PAIRS:
            if pair not in returns_df.columns:
                continue
                
            base, quote = get_currencies(pair)
            
            if base == currency:
                # Se è la base currency, return positivo = valuta forte
                currency_returns.append(returns_df[pair])
            elif quote == currency:
                # Se è la quote currency, return positivo = valuta debole
                currency_returns.append(-returns_df[pair])
        
        if currency_returns:
            # Media dei returns
            avg_return = pd.concat(currency_returns, axis=1).mean(axis=1)
            
            # Doppio smoothing EMA
            smoothed1 = avg_return.ewm(span=smoothing1, adjust=False).mean()
            smoothed2 = smoothed1.ewm(span=smoothing2, adjust=False).mean()
            
            strength[currency] = smoothed2
    
    # Normalizza tra -100 e +100
    if normalize:
        for col in strength.columns:
            # Rolling min/max per normalizzazione dinamica
            rolling_min = strength[col].rolling(500).min()
            rolling_max = strength[col].rolling(500).max()
            
            range_val = rolling_max - rolling_min
            range_val = range_val.replace(0, 1)  # Evita divisione per zero
            
            strength[col] = 200 * (strength[col] - rolling_min) / range_val - 100
    
    return strength


def compute_forza_relativa_spread(
    df: pd.DataFrame,
    pair: str,
    currency_strength: pd.DataFrame
) -> pd.Series:
    """
    Calcola lo spread di forza relativa tra le due valute del pair.
    
    Spread = Forza(Base) - Forza(Quote)
    
    - Spread > 0: Base currency più forte -> favorisce LONG
    - Spread < 0: Quote currency più forte -> favorisce SHORT
    """
    base, quote = get_currencies(pair)
    
    # Allinea con l'indice del DataFrame
    strength_aligned = currency_strength.reindex(df.index, method='ffill')
    
    spread = strength_aligned[base] - strength_aligned[quote]
    
    return spread


def compute_atr_dashboard(
    df: pd.DataFrame,
    ema_period: int = 21,
    atr_period: int = 14,
    history_bars: int = 500
) -> pd.DataFrame:
    """
    Calcola le metriche del ATR Dashboard:
    - Distanza del prezzo dalla EMA in multipli di ATR
    - Percentile storico di questa distanza
    
    Logica:
    - Se il prezzo è molto lontano dalla EMA (alto percentile), è in eccesso
    - Eccessi tendono a mean-revert
    """
    result = pd.DataFrame(index=df.index)
    
    # EMA
    ema = df['Close'].ewm(span=ema_period, adjust=False).mean()
    result['ema'] = ema
    
    # ATR
    atr = compute_atr(df, atr_period)
    result['atr'] = atr
    
    # Distanza dalla EMA in multipli di ATR
    distance = df['Close'] - ema
    distance_atr = distance / atr
    result['ema_distance_atr'] = distance_atr
    result['ema_distance_atr_abs'] = distance_atr.abs()
    
    # Percentile storico (rolling)
    def rolling_percentile(series, window):
        def percentile_func(x):
            if len(x) < 2:
                return 50
            return (x.values < x.values[-1]).sum() / len(x) * 100
        return series.rolling(window).apply(percentile_func, raw=False)
    
    result['atr_percentile'] = rolling_percentile(result['ema_distance_atr_abs'], history_bars)
    
    # Direzione dell'eccesso (1 = sopra EMA, -1 = sotto EMA)
    result['excess_direction'] = np.sign(distance_atr)
    
    return result


def compute_ar_lines(
    df: pd.DataFrame,
    period: int = 10
) -> pd.DataFrame:
    """
    Calcola i livelli Average Range (ADR/AWR/AMR).
    
    - ADR: Average Daily Range
    - AWR: Average Weekly Range
    - AMR: Average Monthly Range
    
    Questi livelli fungono da supporti/resistenze dinamici.
    """
    result = pd.DataFrame(index=df.index)
    
    # Per H1 data, dobbiamo aggregare a D1
    # Assumiamo che df sia già su timeframe appropriato o facciamo rolling
    
    # Range giornaliero approssimato (ultime 24 barre per H1)
    high_24 = df['High'].rolling(24).max()
    low_24 = df['Low'].rolling(24).min()
    daily_range = high_24 - low_24
    result['adr'] = daily_range.rolling(period * 24).mean()  # Media degli ultimi N giorni
    
    # Range settimanale (ultime 120 barre per H1)
    high_week = df['High'].rolling(120).max()
    low_week = df['Low'].rolling(120).min()
    weekly_range = high_week - low_week
    result['awr'] = weekly_range.rolling(period).mean()
    
    # Livelli basati su open giornaliero + range
    # Per semplicità, usiamo il prezzo corrente come riferimento
    result['adr_high'] = df['Close'] + result['adr'] / 2
    result['adr_low'] = df['Close'] - result['adr'] / 2
    
    # Distanza percentuale dal livello AR
    result['dist_to_adr_high_pct'] = (result['adr_high'] - df['Close']) / df['Close'] * 100
    result['dist_to_adr_low_pct'] = (df['Close'] - result['adr_low']) / df['Close'] * 100
    
    return result


def compute_seasonal_pattern(
    df: pd.DataFrame,
    years: int = 10,
    correlation_window: int = 60
) -> pd.DataFrame:
    """
    Calcola il pattern stagionale e la correlazione con anni passati.
    
    Logica:
    - Confronta il pattern dell'anno corrente con gli anni passati
    - Calcola correlazione rolling
    - Genera bias direzionale basato su pattern storici
    """
    result = pd.DataFrame(index=df.index)
    
    # Normalizza prezzi per ogni anno (returns cumulativi)
    df = df.copy()
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear
    
    # Calcola returns cumulativi per anno
    yearly_data = {}
    for year in df['year'].unique():
        year_df = df[df['year'] == year].copy()
        if len(year_df) > 100:  # Almeno 100 barre
            # Normalizza: primo prezzo dell'anno = 100
            first_price = year_df['Close'].iloc[0]
            year_df['normalized'] = year_df['Close'] / first_price * 100
            yearly_data[year] = year_df[['day_of_year', 'normalized']].set_index('day_of_year')['normalized']
    
    if len(yearly_data) < 2:
        result['seasonal_correlation'] = 0
        result['seasonal_direction'] = 0
        return result
    
    # Per ogni punto, calcola correlazione con anni passati
    current_year = df['year'].max()
    
    correlations = []
    directions = []
    
    for idx in df.index:
        current_day = idx.dayofyear
        year = idx.year
        
        if year not in yearly_data or current_day < correlation_window:
            correlations.append(0)
            directions.append(0)
            continue
        
        # Prendi finestra corrente
        current_window = yearly_data[year].loc[:current_day].tail(correlation_window)
        
        if len(current_window) < correlation_window // 2:
            correlations.append(0)
            directions.append(0)
            continue
        
        # Confronta con anni passati
        corr_scores = []
        future_directions = []
        
        for past_year in yearly_data:
            if past_year >= year:
                continue
            
            past_data = yearly_data[past_year]
            
            # Finestra corrispondente
            try:
                past_window = past_data.loc[:current_day].tail(correlation_window)
                
                if len(past_window) == len(current_window):
                    corr = current_window.corr(past_window)
                    if not np.isnan(corr):
                        corr_scores.append(corr)
                        
                        # Direzione futura in quell'anno (prossimi 20 giorni)
                        future_days = list(range(current_day + 1, min(current_day + 21, 366)))
                        if any(d in past_data.index for d in future_days):
                            future_vals = [past_data.get(d, np.nan) for d in future_days]
                            future_vals = [v for v in future_vals if not np.isnan(v)]
                            if future_vals:
                                direction = 1 if np.mean(future_vals) > past_data.loc[current_day] else -1
                                future_directions.append(direction * corr)  # Peso per correlazione
            except:
                continue
        
        # Media pesata
        if corr_scores:
            correlations.append(np.mean(corr_scores))
            directions.append(np.sign(np.sum(future_directions)) if future_directions else 0)
        else:
            correlations.append(0)
            directions.append(0)
    
    result['seasonal_correlation'] = correlations
    result['seasonal_direction'] = directions
    
    return result


# ============================================================================
# FEATURE ENGINEERING PRINCIPALE
# ============================================================================

def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola tutte le feature tecniche base per un singolo pair.
    """
    features = pd.DataFrame(index=df.index)
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df.get('Volume', pd.Series(0, index=df.index))
    
    # ========== RETURNS ==========
    for period in [1, 5, 10, 20, 50, 100]:
        features[f'return_{period}'] = close.pct_change(period)
    
    # Returns cumulativi
    features['return_cum_20'] = close.pct_change(20).rolling(20).sum()
    
    # ========== VOLATILITY ==========
    for period in [10, 20, 50]:
        features[f'volatility_{period}'] = close.pct_change().rolling(period).std()
    
    # Volatility ratio
    features['volatility_ratio'] = features['volatility_10'] / features['volatility_50']
    
    # ========== MOVING AVERAGES ==========
    for period in [10, 20, 50, 100, 200]:
        sma = close.rolling(period).mean()
        ema = close.ewm(span=period, adjust=False).mean()
        
        features[f'sma_{period}'] = sma
        features[f'ema_{period}'] = ema
        features[f'dist_sma_{period}'] = (close - sma) / sma * 100
        features[f'dist_ema_{period}'] = (close - ema) / ema * 100
    
    # MA crossovers
    features['sma_10_50_cross'] = (features['sma_10'] > features['sma_50']).astype(int)
    features['sma_20_100_cross'] = (features['sma_20'] > features['sma_100']).astype(int)
    
    # ========== MOMENTUM ==========
    # RSI
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
    
    # ========== ATR ==========
    for period in [7, 14, 21]:
        atr = compute_atr(df, period)
        features[f'atr_{period}'] = atr
        features[f'atr_pct_{period}'] = atr / close * 100
    
    # ========== BOLLINGER BANDS ==========
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close, 20, 2)
    features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
    features['bb_width'] = (bb_upper - bb_lower) / bb_mid * 100
    
    # ========== PRICE ACTION ==========
    # Candlestick features
    features['body'] = close - df['Open']
    features['body_pct'] = features['body'] / df['Open'] * 100
    features['upper_wick'] = high - np.maximum(close, df['Open'])
    features['lower_wick'] = np.minimum(close, df['Open']) - low
    features['range'] = high - low
    features['body_to_range'] = features['body'].abs() / features['range']
    
    # Higher highs / Lower lows
    features['hh'] = (high > high.shift(1)).astype(int)
    features['ll'] = (low < low.shift(1)).astype(int)
    features['hh_streak'] = features['hh'].groupby((features['hh'] != features['hh'].shift()).cumsum()).cumsum()
    features['ll_streak'] = features['ll'].groupby((features['ll'] != features['ll'].shift()).cumsum()).cumsum()
    
    # ========== VOLUME ==========
    if volume.sum() > 0:
        features['volume_sma_20'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_sma_20']
        features['volume_trend'] = volume.rolling(10).mean() / volume.rolling(50).mean()
    
    # ========== TIME FEATURES ==========
    features['hour'] = df.index.hour
    features['day_of_week'] = df.index.dayofweek
    features['day_of_month'] = df.index.day
    features['month'] = df.index.month
    features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
    features['is_overlap'] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(int)
    
    return features


def compute_all_features(
    data_dir: str,
    output_dir: str,
    timeframe: str = "H1",
    include_seasonal: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Calcola tutte le feature per tutti i pair.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carica tutti i dati
    logger.info("Caricamento dati...")
    all_data = {}
    for pair in FOREX_PAIRS:
        filepath = data_dir / f"{pair}_{timeframe}.parquet"
        if filepath.exists():
            all_data[pair] = pd.read_parquet(filepath)
            logger.info(f"  ✓ {pair}: {len(all_data[pair]):,} barre")
        else:
            logger.warning(f"  ✗ {pair}: file non trovato")
    
    if len(all_data) == 0:
        raise ValueError("Nessun dato trovato!")
    
    # Calcola forza relativa delle valute
    logger.info("Calcolo Currency Strength...")
    currency_strength = compute_currency_strength(all_data, smoothing1=20, smoothing2=15)
    
    # Salva currency strength
    currency_strength.to_parquet(output_dir / 'currency_strength.parquet')
    
    # Calcola feature per ogni pair
    all_features = {}
    
    for pair in tqdm(all_data, desc="Computing features"):
        df = all_data[pair]
        features = pd.DataFrame(index=df.index)
        
        # Feature tecniche base
        base_features = compute_base_features(df)
        features = features.join(base_features)
        
        # Forza Relativa Spread
        fr_spread = compute_forza_relativa_spread(df, pair, currency_strength)
        features['fr_spread'] = fr_spread
        features['fr_spread_abs'] = fr_spread.abs()
        
        # Aggiungi forza delle singole valute
        base, quote = get_currencies(pair)
        strength_aligned = currency_strength.reindex(df.index, method='ffill')
        features[f'strength_{base}'] = strength_aligned[base]
        features[f'strength_{quote}'] = strength_aligned[quote]
        
        # ATR Dashboard
        atr_dash = compute_atr_dashboard(df, ema_period=21, atr_period=14, history_bars=500)
        features = features.join(atr_dash, rsuffix='_atrdash')
        
        # AR Lines
        ar_lines = compute_ar_lines(df, period=10)
        features = features.join(ar_lines, rsuffix='_ar')
        
        # Seasonal Pattern (opzionale, richiede più tempo)
        if include_seasonal:
            try:
                seasonal = compute_seasonal_pattern(df, years=10, correlation_window=60)
                features = features.join(seasonal, rsuffix='_seasonal')
            except Exception as e:
                logger.warning(f"Seasonal pattern failed for {pair}: {e}")
        
        # Aggiungi pair ID (per training multi-pair)
        features['pair_id'] = FOREX_PAIRS.index(pair)
        
        # Rimuovi colonne non numeriche e inf
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Salva
        filepath = output_dir / f"{pair}_features.parquet"
        features.to_parquet(filepath)
        
        all_features[pair] = features
        logger.info(f"  {pair}: {features.shape[1]} features")
    
    return all_features


def create_labels(
    data_dir: str,
    features_dir: str,
    output_dir: str,
    config: dict = None
) -> None:
    """
    Crea le labels per il training.
    """
    if config is None:
        config = {
            'horizon': 24,  # Predici 24 ore avanti
            'direction_threshold': 0.002,  # 0.2% movimento minimo
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 1.5,
        }
    
    data_dir = Path(data_dir)
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pair in tqdm(FOREX_PAIRS, desc="Creating labels"):
        # Carica dati originali
        data_path = data_dir / f"{pair}_H1.parquet"
        features_path = features_dir / f"{pair}_features.parquet"
        
        if not data_path.exists() or not features_path.exists():
            continue
        
        df = pd.read_parquet(data_path)
        features = pd.read_parquet(features_path)
        
        labels = pd.DataFrame(index=df.index)
        
        # ===== Direction Label =====
        future_return = df['Close'].shift(-config['horizon']).pct_change(config['horizon'])
        
        labels['direction'] = 0  # Hold
        labels.loc[future_return > config['direction_threshold'], 'direction'] = 1   # Buy
        labels.loc[future_return < -config['direction_threshold'], 'direction'] = -1  # Sell
        
        # ===== Target Return =====
        labels['target_return'] = future_return
        
        # ===== Max Favorable Excursion (MFE) =====
        # Massimo movimento favorevole nelle prossime N ore
        future_high = df['High'].rolling(config['horizon']).max().shift(-config['horizon'])
        future_low = df['Low'].rolling(config['horizon']).min().shift(-config['horizon'])
        
        labels['mfe_long'] = (future_high - df['Close']) / df['Close']
        labels['mfe_short'] = (df['Close'] - future_low) / df['Close']
        
        # ===== Max Adverse Excursion (MAE) =====
        labels['mae_long'] = (df['Close'] - future_low) / df['Close']
        labels['mae_short'] = (future_high - df['Close']) / df['Close']
        
        # ===== Win Probability (simulazione trade) =====
        atr = features.get('atr_14', compute_atr(df, 14))
        sl = config['sl_atr_mult'] * atr
        tp = config['tp_atr_mult'] * atr
        
        # Simula long trade
        long_tp_hit = labels['mfe_long'] * df['Close'] >= tp
        long_sl_hit = labels['mae_long'] * df['Close'] >= sl
        labels['long_win'] = (long_tp_hit & ~long_sl_hit).astype(int)
        
        # Simula short trade
        short_tp_hit = labels['mfe_short'] * df['Close'] >= tp
        short_sl_hit = labels['mae_short'] * df['Close'] >= sl
        labels['short_win'] = (short_tp_hit & ~short_sl_hit).astype(int)
        
        # ===== Optimal Action =====
        labels['optimal_action'] = 0
        labels.loc[labels['long_win'] == 1, 'optimal_action'] = 1
        labels.loc[labels['short_win'] == 1, 'optimal_action'] = -1
        
        # Salva
        labels.to_parquet(output_dir / f"{pair}_labels.parquet")
    
    logger.info("Labels create per tutti i pair")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute Features for AI Training')
    parser.add_argument('--input', type=str, default='./data/raw', help='Directory dati raw')
    parser.add_argument('--output', type=str, default='./data/processed', help='Directory output')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe principale')
    parser.add_argument('--no-seasonal', action='store_true', help='Skip seasonal pattern (più veloce)')
    parser.add_argument('--labels', action='store_true', help='Crea anche le labels')
    
    args = parser.parse_args()
    
    # Compute features
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    features = compute_all_features(
        data_dir=args.input,
        output_dir=args.output,
        timeframe=args.timeframe,
        include_seasonal=not args.no_seasonal
    )
    
    # Create labels
    if args.labels:
        logger.info("\nCreazione labels...")
        create_labels(
            data_dir=args.input,
            features_dir=args.output,
            output_dir=args.output
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETATO")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
