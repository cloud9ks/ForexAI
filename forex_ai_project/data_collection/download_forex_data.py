"""
================================================================================
FOREX DATA COLLECTOR - NexNow LTD
================================================================================
Script completo per scaricare tutti i dati necessari per il training del modello AI.

Scarica:
- OHLCV per tutti i 28 pair forex (M1, M5, H1, H4, D1, W1)
- 10+ anni di storico
- Salva in formato Parquet (efficiente)

Requisiti:
- MetaTrader 5 installato e connesso a un broker
- pip install MetaTrader5 pandas pyarrow tqdm

Uso:
    python download_forex_data.py --years 10 --output ./data/raw
================================================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# Tutti i 28 major forex pairs
FOREX_PAIRS = [
    # Major pairs (USD)
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    # EUR crosses
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    # GBP crosses
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    # AUD crosses
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    # Other crosses
    "CADJPY", "CADCHF", "CHFJPY", "NZDJPY", "NZDCHF", "NZDCAD"
]

# Timeframes da scaricare
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,    # Per dati tick-level
    "M5": mt5.TIMEFRAME_M5,    # Per pattern intraday
    "M15": mt5.TIMEFRAME_M15,  # Per scalping
    "H1": mt5.TIMEFRAME_H1,    # PRINCIPALE per training
    "H4": mt5.TIMEFRAME_H4,    # Per swing trading
    "D1": mt5.TIMEFRAME_D1,    # Per trend analysis
    "W1": mt5.TIMEFRAME_W1,    # Per macro trend
}

# Stima barre per anno per ogni timeframe
BARS_PER_YEAR = {
    "M1": 525600,   # 60 * 24 * 365
    "M5": 105120,   # 12 * 24 * 365
    "M15": 35040,   # 4 * 24 * 365
    "H1": 8760,     # 24 * 365
    "H4": 2190,     # 6 * 365
    "D1": 365,
    "W1": 52,
}


class ForexDataDownloader:
    """
    Classe per scaricare dati forex da MetaTrader 5
    """
    
    def __init__(self, output_dir: str = "./data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inizializza MT5
        logger.info("Inizializzazione MetaTrader 5...")
        if not mt5.initialize():
            error = mt5.last_error()
            raise Exception(f"MT5 initialization failed: {error}")
        
        # Info account
        account = mt5.account_info()
        if account:
            logger.info(f"Connesso all'account: {account.login}")
            logger.info(f"Server: {account.server}")
            logger.info(f"Balance: ${account.balance:,.2f}")
        else:
            logger.warning("Impossibile ottenere info account")
    
    def get_symbol_info(self, symbol: str) -> dict:
        """Ottiene informazioni su un simbolo"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        
        return {
            'symbol': symbol,
            'digits': info.digits,
            'point': info.point,
            'trade_tick_size': info.trade_tick_size,
            'trade_tick_value': info.trade_tick_value,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'spread': info.spread,
        }
    
    def download_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        years: int = 10,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Scarica dati OHLCV per un simbolo
        
        Args:
            symbol: Nome del pair (es. "EURUSD")
            timeframe: Timeframe (es. "H1")
            years: Anni di storico da scaricare
            end_date: Data finale (default: oggi)
        
        Returns:
            DataFrame con colonne: Open, High, Low, Close, Volume
        """
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"Timeframe {timeframe} non supportato")
        
        tf = TIMEFRAMES[timeframe]
        
        # Calcola numero di barre
        bars = BARS_PER_YEAR.get(timeframe, 8760) * years
        # MT5 ha un limite di 99999 barre per chiamata
        bars = min(bars, 500000)
        
        # Data finale
        if end_date is None:
            end_date = datetime.now()
        
        # Abilita simbolo se necessario
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Impossibile selezionare {symbol}")
            return pd.DataFrame()
        
        # Scarica dati
        rates = mt5.copy_rates_from(symbol, tf, end_date, bars)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"Nessun dato per {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Converti in DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Rinomina colonne
        df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume',
            'spread': 'Spread',
            'real_volume': 'RealVolume'
        }, inplace=True)
        
        # Seleziona colonne utili
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Spread' in df.columns:
            cols.append('Spread')
        
        return df[cols]
    
    def download_pair(
        self, 
        symbol: str, 
        timeframes: list = None,
        years: int = 10
    ) -> dict:
        """
        Scarica tutti i timeframe per un pair
        
        Returns:
            Dict con {timeframe: DataFrame}
        """
        if timeframes is None:
            timeframes = list(TIMEFRAMES.keys())
        
        results = {}
        
        for tf in timeframes:
            logger.info(f"  Scaricando {symbol} {tf}...")
            
            try:
                df = self.download_ohlcv(symbol, tf, years)
                
                if not df.empty:
                    # Salva in parquet
                    filepath = self.output_dir / f"{symbol}_{tf}.parquet"
                    df.to_parquet(filepath, compression='snappy')
                    
                    results[tf] = {
                        'bars': len(df),
                        'start': df.index.min(),
                        'end': df.index.max(),
                        'file': str(filepath)
                    }
                    
                    logger.info(f"    ✓ {len(df):,} barre salvate ({df.index.min().date()} - {df.index.max().date()})")
                else:
                    logger.warning(f"    ✗ Nessun dato")
                    
            except Exception as e:
                logger.error(f"    ✗ Errore: {e}")
        
        return results
    
    def download_all(
        self,
        pairs: list = None,
        timeframes: list = None,
        years: int = 10
    ) -> dict:
        """
        Scarica tutti i pair e timeframe
        
        Args:
            pairs: Lista di pair (default: tutti i 28)
            timeframes: Lista di timeframe (default: tutti)
            years: Anni di storico
        
        Returns:
            Report completo del download
        """
        if pairs is None:
            pairs = FOREX_PAIRS
        
        if timeframes is None:
            # Per training AI usiamo principalmente H1 e D1
            timeframes = ["H1", "H4", "D1"]
        
        report = {
            'start_time': datetime.now(),
            'pairs': {},
            'total_bars': 0,
            'total_files': 0,
        }
        
        logger.info(f"=" * 60)
        logger.info(f"DOWNLOAD DATI FOREX")
        logger.info(f"Pairs: {len(pairs)}")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Anni: {years}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"=" * 60)
        
        for pair in tqdm(pairs, desc="Downloading pairs"):
            logger.info(f"\n[{pair}]")
            
            try:
                results = self.download_pair(pair, timeframes, years)
                report['pairs'][pair] = results
                
                for tf, info in results.items():
                    report['total_bars'] += info['bars']
                    report['total_files'] += 1
                    
            except Exception as e:
                logger.error(f"Errore per {pair}: {e}")
                report['pairs'][pair] = {'error': str(e)}
            
            # Piccola pausa per non sovraccaricare
            time.sleep(0.1)
        
        report['end_time'] = datetime.now()
        report['duration'] = report['end_time'] - report['start_time']
        
        # Salva report
        self._save_report(report)
        
        logger.info(f"\n" + "=" * 60)
        logger.info(f"DOWNLOAD COMPLETATO")
        logger.info(f"Totale barre: {report['total_bars']:,}")
        logger.info(f"Totale file: {report['total_files']}")
        logger.info(f"Durata: {report['duration']}")
        logger.info(f"=" * 60)
        
        return report
    
    def _save_report(self, report: dict):
        """Salva report del download"""
        import json
        
        # Converti datetime per JSON
        report_json = {
            'start_time': str(report['start_time']),
            'end_time': str(report['end_time']),
            'duration_seconds': report['duration'].total_seconds(),
            'total_bars': report['total_bars'],
            'total_files': report['total_files'],
            'pairs': {}
        }
        
        for pair, data in report['pairs'].items():
            report_json['pairs'][pair] = {}
            for tf, info in data.items():
                if isinstance(info, dict) and 'bars' in info:
                    report_json['pairs'][pair][tf] = {
                        'bars': info['bars'],
                        'start': str(info['start']),
                        'end': str(info['end']),
                        'file': info['file']
                    }
        
        filepath = self.output_dir / 'download_report.json'
        with open(filepath, 'w') as f:
            json.dump(report_json, f, indent=2)
        
        logger.info(f"Report salvato: {filepath}")
    
    def verify_data(self) -> pd.DataFrame:
        """Verifica i dati scaricati"""
        files = list(self.output_dir.glob("*.parquet"))
        
        data = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                parts = f.stem.split('_')
                pair = parts[0]
                tf = parts[1] if len(parts) > 1 else 'unknown'
                
                data.append({
                    'pair': pair,
                    'timeframe': tf,
                    'bars': len(df),
                    'start': df.index.min(),
                    'end': df.index.max(),
                    'missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
                    'file_size_mb': f.stat().st_size / 1024 / 1024
                })
            except Exception as e:
                data.append({
                    'pair': f.stem,
                    'error': str(e)
                })
        
        return pd.DataFrame(data)
    
    def __del__(self):
        """Chiude connessione MT5"""
        mt5.shutdown()
        logger.info("MT5 disconnesso")


def download_economic_calendar(output_dir: str = "./data/external"):
    """
    Scarica calendario economico da fonti pubbliche
    
    Nota: Richiede API key per dati completi (es. ForexFactory, Investing.com)
    Questa è una versione base che usa dati FRED
    """
    from fredapi import Fred
    import os
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inizializza FRED (richiede API key gratuita)
    fred_api_key = os.environ.get('FRED_API_KEY')
    if not fred_api_key:
        logger.warning("FRED_API_KEY non impostata. Salta download dati economici.")
        logger.info("Ottieni una chiave gratuita su: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    
    fred = Fred(api_key=fred_api_key)
    
    # Serie importanti per Forex
    series = {
        # US
        'DFF': 'Fed Funds Rate',
        'T10Y2Y': 'Yield Curve 10Y-2Y',
        'DTWEXBGS': 'USD Index (Trade Weighted)',
        'UMCSENT': 'Consumer Sentiment',
        'PAYEMS': 'Non-Farm Payrolls',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'CPI All Items',
        'PCEPI': 'PCE Price Index',
        'GDPC1': 'Real GDP',
        'WALCL': 'Fed Balance Sheet',
        
        # Interest Rate Differentials
        'IR3TIB01USM156N': 'US 3M Rate',
        'IR3TIB01EZM156N': 'EU 3M Rate',
        'IR3TIB01GBM156N': 'UK 3M Rate',
        'IR3TIB01JPM156N': 'JP 3M Rate',
        
        # Volatility
        'VIXCLS': 'VIX',
    }
    
    data = {}
    for code, name in tqdm(series.items(), desc="Downloading FRED data"):
        try:
            s = fred.get_series(code, observation_start='2010-01-01')
            data[code] = s
            logger.info(f"  ✓ {name}: {len(s)} observations")
        except Exception as e:
            logger.warning(f"  ✗ {name}: {e}")
    
    # Combina in DataFrame
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    
    # Forward fill per dati mensili/trimestrali
    df = df.ffill()
    
    # Salva
    filepath = output_dir / 'fred_economic_data.parquet'
    df.to_parquet(filepath)
    logger.info(f"Dati economici salvati: {filepath}")
    
    return df


def download_cot_data(output_dir: str = "./data/external"):
    """
    Scarica dati COT (Commitment of Traders) dalla CFTC
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # URL per dati COT storici
    # Forex futures codes
    cot_codes = {
        '099741': 'EUR',
        '096742': 'GBP', 
        '097741': 'JPY',
        '092741': 'CHF',
        '090741': 'CAD',
        '232741': 'AUD',
        '112741': 'NZD',
    }
    
    try:
        # Scarica file COT completo (legacy format)
        url = "https://www.cftc.gov/dea/newcot/deafut.txt"
        logger.info(f"Scaricando dati COT da CFTC...")
        
        # Questo è un file grande, potrebbe richiedere tempo
        df = pd.read_csv(url, low_memory=False)
        
        # Filtra solo forex
        forex_df = df[df['CFTC_Contract_Market_Code'].astype(str).isin(cot_codes.keys())]
        
        # Salva
        filepath = output_dir / 'cot_forex.parquet'
        forex_df.to_parquet(filepath)
        logger.info(f"Dati COT salvati: {filepath} ({len(forex_df)} records)")
        
        return forex_df
        
    except Exception as e:
        logger.error(f"Errore download COT: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Download Forex Data for AI Training')
    parser.add_argument('--years', type=int, default=10, help='Anni di storico (default: 10)')
    parser.add_argument('--output', type=str, default='./data/raw', help='Directory output')
    parser.add_argument('--timeframes', nargs='+', default=['H1', 'H4', 'D1'], 
                        help='Timeframes da scaricare (default: H1 H4 D1)')
    parser.add_argument('--pairs', nargs='+', default=None,
                        help='Pairs specifici (default: tutti i 28)')
    parser.add_argument('--economic', action='store_true', help='Scarica anche dati economici')
    parser.add_argument('--cot', action='store_true', help='Scarica anche dati COT')
    parser.add_argument('--verify', action='store_true', help='Solo verifica dati esistenti')
    
    args = parser.parse_args()
    
    # Inizializza downloader
    downloader = ForexDataDownloader(output_dir=args.output)
    
    if args.verify:
        # Solo verifica
        logger.info("Verifica dati esistenti...")
        report = downloader.verify_data()
        print(report.to_string())
        return
    
    # Download principale
    downloader.download_all(
        pairs=args.pairs,
        timeframes=args.timeframes,
        years=args.years
    )
    
    # Download dati aggiuntivi
    if args.economic:
        download_economic_calendar(output_dir=str(Path(args.output).parent / 'external'))
    
    if args.cot:
        download_cot_data(output_dir=str(Path(args.output).parent / 'external'))
    
    # Verifica finale
    logger.info("\nVerifica dati scaricati:")
    report = downloader.verify_data()
    print(report.to_string())


if __name__ == "__main__":
    main()
