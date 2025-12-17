"""
================================================================================
AI TRADING AGENT - DXY MODEL (Enrico's Macro Model)
================================================================================
Modello macro basato su bilanci FED vs ECB per prevedere direzione USD.

Logica:
- Se FED contrae più di ECB → USD forte → DXY bullish
- Se FED contrae meno di ECB → USD debole → DXY bearish

Integrazione: Bonus +5% confidence se segnale allineato con trade.
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DXYModel:
    """
    Modello DXY basato su spread bilanci FED-ECB.
    Fornisce bias USD per le decisioni di trading.
    """

    def __init__(self):
        self.data = None
        self.last_signal = None
        self.last_update = None
        self.confidence_bonus = 0.05  # +5% bonus se allineato

        # Carica dati dal file Excel
        self._load_data()

    def _load_data(self):
        """Carica i dati dal file Excel di Enrico."""
        try:
            # Trova il file
            base_dir = Path(__file__).parent.parent
            excel_path = base_dir / "FIle modello dxy.xlsx"

            if not excel_path.exists():
                logger.warning(f"DXY model file not found: {excel_path}")
                return

            # Leggi Excel (skip header rows)
            df = pd.read_excel(excel_path, skiprows=5)
            df.columns = ['date', 'FED_balance', 'ECB_balance', 'EURUSD',
                         'col4', 'col5', 'YoY_US', 'pct_ECB', 'col8']

            # Pulisci dati
            df = df[['date', 'FED_balance', 'ECB_balance', 'EURUSD', 'YoY_US', 'pct_ECB']]
            df = df.dropna(subset=['date'])

            # Converti a numerico
            for col in ['FED_balance', 'ECB_balance', 'EURUSD', 'YoY_US', 'pct_ECB']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Calcola spread
            df['spread'] = df['YoY_US'] - df['pct_ECB']

            self.data = df
            self.last_update = datetime.now()

            logger.info(f"DXY model loaded: {len(df)} months of data")

        except Exception as e:
            logger.error(f"Failed to load DXY model: {e}")
            self.data = None

    def get_usd_bias(self) -> dict:
        """
        Ottiene il bias USD corrente basato sul modello.

        Returns:
            dict con:
                - bias: 'bullish', 'bearish', o 'neutral'
                - spread: valore spread FED-ECB
                - fed_yoy: variazione YoY Fed
                - ecb_yoy: variazione YoY ECB
                - confidence: quanto è forte il segnale (0-1)
        """
        result = {
            'bias': 'neutral',
            'spread': 0.0,
            'fed_yoy': 0.0,
            'ecb_yoy': 0.0,
            'confidence': 0.0,
            'date': None,
        }

        if self.data is None or len(self.data) == 0:
            return result

        # Prendi ultimo dato disponibile
        last = self.data.iloc[-1]

        result['date'] = last['date']
        result['fed_yoy'] = float(last['YoY_US'])
        result['ecb_yoy'] = float(last['pct_ECB'])
        result['spread'] = float(last['spread'])

        # Determina bias
        spread = result['spread']

        if spread > 0.03:  # FED espande più di ECB (+3%)
            result['bias'] = 'bearish'  # USD bearish
            result['confidence'] = min(1.0, abs(spread) / 0.10)
        elif spread < -0.03:  # FED contrae più di ECB (-3%)
            result['bias'] = 'bullish'  # USD bullish
            result['confidence'] = min(1.0, abs(spread) / 0.10)
        else:
            result['bias'] = 'neutral'
            result['confidence'] = 0.3

        self.last_signal = result
        return result

    def get_pair_bias(self, pair: str) -> dict:
        """
        Ottiene il bias per una specifica coppia forex.

        Args:
            pair: Coppia forex (es. 'EURUSD')

        Returns:
            dict con bias e raccomandazione per la coppia
        """
        usd_bias = self.get_usd_bias()

        result = {
            'pair': pair,
            'usd_bias': usd_bias['bias'],
            'pair_bias': 'neutral',
            'aligned_with': None,
            'confidence_bonus': 0.0,
        }

        # Determina come USD bias influenza la coppia
        base = pair[:3]
        quote = pair[3:6]

        if quote == 'USD':
            # Coppie come EURUSD, GBPUSD, AUDUSD
            # USD bearish → coppia sale → favorisce BUY
            # USD bullish → coppia scende → favorisce SELL
            if usd_bias['bias'] == 'bearish':
                result['pair_bias'] = 'bullish'
                result['aligned_with'] = 'BUY'
            elif usd_bias['bias'] == 'bullish':
                result['pair_bias'] = 'bearish'
                result['aligned_with'] = 'SELL'

        elif base == 'USD':
            # Coppie come USDJPY, USDCHF, USDCAD
            # USD bearish → coppia scende → favorisce SELL
            # USD bullish → coppia sale → favorisce BUY
            if usd_bias['bias'] == 'bearish':
                result['pair_bias'] = 'bearish'
                result['aligned_with'] = 'SELL'
            elif usd_bias['bias'] == 'bullish':
                result['pair_bias'] = 'bullish'
                result['aligned_with'] = 'BUY'

        return result

    def get_confidence_bonus(self, pair: str, signal: str) -> float:
        """
        Calcola il bonus di confidenza se il segnale è allineato con DXY.

        Args:
            pair: Coppia forex
            signal: Segnale LSTM ('BUY' o 'SELL')

        Returns:
            Bonus da aggiungere alla confidenza (0.0 o 0.05)
        """
        if signal not in ['BUY', 'SELL']:
            return 0.0

        pair_bias = self.get_pair_bias(pair)

        # Se il segnale è allineato con il bias DXY, dai bonus
        if pair_bias['aligned_with'] == signal:
            logger.info(f"DXY bonus applied: {pair} {signal} aligned with USD {pair_bias['usd_bias']}")
            return self.confidence_bonus

        # Nessuna penalità se non allineato
        return 0.0

    def format_for_agent(self, pair: str) -> str:
        """
        Formatta il contesto DXY per il prompt GPT.

        Args:
            pair: Coppia forex

        Returns:
            Stringa formattata
        """
        usd_bias = self.get_usd_bias()
        pair_bias = self.get_pair_bias(pair)

        lines = [
            f"=== DXY MACRO MODEL ===",
            f"Data: {usd_bias['date']}",
            f"",
            f"FED Balance YoY: {usd_bias['fed_yoy']*100:+.1f}%",
            f"ECB Balance YoY: {usd_bias['ecb_yoy']*100:+.1f}%",
            f"Spread (FED-ECB): {usd_bias['spread']*100:+.1f}%",
            f"",
            f"USD Bias: {usd_bias['bias'].upper()}",
            f"{pair} Bias: {pair_bias['pair_bias'].upper()}",
            f"Favors: {pair_bias['aligned_with'] or 'NEUTRAL'}",
        ]

        return "\n".join(lines)


# Singleton instance
_dxy_model = None


def get_dxy_model() -> DXYModel:
    """Get singleton DXYModel instance."""
    global _dxy_model
    if _dxy_model is None:
        _dxy_model = DXYModel()
    return _dxy_model
