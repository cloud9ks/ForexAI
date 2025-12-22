"""
================================================================================
FED SENTIMENT MODEL - Monthly FOMC Analysis
================================================================================
Modello basato su NLP dei verbali FOMC per determinare la stance della Fed.
Metodologia: Morgan Stanley MNLPFEDS + Loughran-McDonald Dictionary

Timeframe: MENSILE (aggiornato dopo ogni FOMC meeting)
Output: Hawkish (+), Dovish (-), Neutral

Integrazione con DXY Model:
- DXY = direzione STRATEGICA (trimestrale, dove andare)
- Fed Sentiment = conferma TATTICA (mensile, timing)

Combinazione:
- DXY bullish + Fed hawkish = FORTE USD bullish (+15% boost)
- DXY bullish + Fed dovish = CONFLITTO (-15% penalty)
- DXY bullish + Fed neutral = MODERATO USD bullish
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Path al CSV mensile generato da fed_sentiment_pro.py
FED_SENTIMENT_CSV = Path(__file__).parent.parent.parent / "modello_tassi_interesse_dollaro" / "fed_sentiment_monthly.csv"

# Soglie per regime classification
FED_THRESHOLDS = {
    'hawkish': 0.15,      # Sopra 0.15 = Hawkish (USD bullish)
    'dovish': -0.15,      # Sotto -0.15 = Dovish (USD bearish)
    'strong_hawkish': 0.30,  # Sopra 0.30 = Molto Hawkish
    'strong_dovish': -0.30,  # Sotto -0.30 = Molto Dovish
}

# Bonus/Penalty per alignment con DXY
ALIGNMENT_BONUSES = {
    'strong_align': 0.15,     # DXY e Fed fortemente allineati
    'align': 0.10,            # DXY e Fed allineati
    'neutral': 0.0,           # Uno dei due neutral
    'conflict': -0.10,        # DXY e Fed in conflitto
    'strong_conflict': -0.15, # DXY e Fed fortemente in conflitto
}


class FedSentimentModel:
    """
    Modello di sentiment Fed basato su NLP dei verbali FOMC.
    Fornisce bias USD su base mensile.
    """

    def __init__(self):
        self.data = None
        self.last_update = None
        self._load_data()

    def _load_data(self):
        """Carica i dati del Fed Sentiment Index."""
        if not FED_SENTIMENT_CSV.exists():
            logger.warning(f"Fed Sentiment CSV not found: {FED_SENTIMENT_CSV}")
            self.data = None
            return

        try:
            self.data = pd.read_csv(FED_SENTIMENT_CSV)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date')
            self.last_update = datetime.now()

            latest = self.data.iloc[-1]
            logger.info(f"Fed Sentiment loaded: {len(self.data)} months")
            logger.info(f"Latest: {latest['date'].strftime('%Y-%m')} = {latest['sentiment_ewma']:.3f}")

        except Exception as e:
            logger.error(f"Error loading Fed Sentiment: {e}")
            self.data = None

    def get_current_sentiment(self) -> Dict:
        """
        Ottiene il sentiment Fed corrente.

        Returns:
            Dict con sentiment, regime, momentum, etc.
        """
        result = {
            'sentiment': 0.0,
            'regime': 'NEUTRAL',
            'momentum': 0.0,
            'ma_30': 0.0,
            'ma_90': 0.0,
            'date': None,
            'usd_bias': 'NEUTRAL',
            'confidence': 0.5,
            'available': False,
        }

        if self.data is None or len(self.data) == 0:
            return result

        # Prendi l'ultimo mese disponibile
        latest = self.data.iloc[-1]

        sentiment = latest['sentiment_ewma']
        momentum = latest.get('momentum', 0) or 0

        result['sentiment'] = round(sentiment, 4)
        result['momentum'] = round(momentum, 4)
        result['ma_30'] = round(latest.get('ma_30', sentiment) or sentiment, 4)
        result['ma_90'] = round(latest.get('ma_90', sentiment) or sentiment, 4)
        result['date'] = latest['date'].strftime('%Y-%m')
        result['available'] = True

        # Determina regime
        if sentiment >= FED_THRESHOLDS['strong_hawkish']:
            result['regime'] = 'STRONG_HAWKISH'
            result['usd_bias'] = 'STRONG_BULLISH'
            result['confidence'] = 0.9
        elif sentiment >= FED_THRESHOLDS['hawkish']:
            result['regime'] = 'HAWKISH'
            result['usd_bias'] = 'BULLISH'
            result['confidence'] = 0.75
        elif sentiment <= FED_THRESHOLDS['strong_dovish']:
            result['regime'] = 'STRONG_DOVISH'
            result['usd_bias'] = 'STRONG_BEARISH'
            result['confidence'] = 0.9
        elif sentiment <= FED_THRESHOLDS['dovish']:
            result['regime'] = 'DOVISH'
            result['usd_bias'] = 'BEARISH'
            result['confidence'] = 0.75
        else:
            result['regime'] = 'NEUTRAL'
            result['usd_bias'] = 'NEUTRAL'
            result['confidence'] = 0.5

        # Aggiusta confidence in base al momentum
        if abs(momentum) > 0.05:
            # Momentum forte nella stessa direzione del sentiment
            if (momentum > 0 and sentiment > 0) or (momentum < 0 and sentiment < 0):
                result['confidence'] = min(result['confidence'] + 0.1, 0.95)

        return result

    def get_pair_bias(self, pair: str) -> Dict:
        """
        Ottiene il bias per una specifica coppia forex.

        Args:
            pair: Coppia forex (es. 'EURUSD')

        Returns:
            Dict con bias per la coppia
        """
        fed = self.get_current_sentiment()

        result = {
            'pair': pair,
            'fed_sentiment': fed['sentiment'],
            'fed_regime': fed['regime'],
            'fed_usd_bias': fed['usd_bias'],
            'pair_bias': 'NEUTRAL',
            'aligned_with': None,
            'confidence': fed['confidence'],
            'date': fed['date'],
        }

        if not fed['available']:
            return result

        # Determina bias per la coppia
        base_currency = pair[:3]
        quote_currency = pair[3:6]

        # USD è base o quote?
        if quote_currency == 'USD':
            # Es: EURUSD - se USD bullish, pair dovrebbe scendere (SELL)
            if fed['usd_bias'] in ['BULLISH', 'STRONG_BULLISH']:
                result['pair_bias'] = 'BEARISH'  # Sell the pair
                result['aligned_with'] = 'SELL'
            elif fed['usd_bias'] in ['BEARISH', 'STRONG_BEARISH']:
                result['pair_bias'] = 'BULLISH'  # Buy the pair
                result['aligned_with'] = 'BUY'
        elif base_currency == 'USD':
            # Es: USDJPY - se USD bullish, pair dovrebbe salire (BUY)
            if fed['usd_bias'] in ['BULLISH', 'STRONG_BULLISH']:
                result['pair_bias'] = 'BULLISH'  # Buy the pair
                result['aligned_with'] = 'BUY'
            elif fed['usd_bias'] in ['BEARISH', 'STRONG_BEARISH']:
                result['pair_bias'] = 'BEARISH'  # Sell the pair
                result['aligned_with'] = 'SELL'

        return result

    def get_alignment_with_dxy(self, dxy_usd_bias: str) -> Tuple[str, float]:
        """
        Calcola l'allineamento tra Fed Sentiment e DXY Model.

        Args:
            dxy_usd_bias: Bias USD dal DXY Model ('BULLISH', 'BEARISH', 'NEUTRAL')

        Returns:
            Tuple(alignment_type, bonus)
        """
        fed = self.get_current_sentiment()

        if not fed['available']:
            return 'neutral', 0.0

        fed_bias = fed['usd_bias']

        # Mapping per confronto
        bullish_set = {'BULLISH', 'STRONG_BULLISH'}
        bearish_set = {'BEARISH', 'STRONG_BEARISH'}

        dxy_bullish = dxy_usd_bias.upper() in bullish_set or dxy_usd_bias.upper() == 'BULLISH'
        dxy_bearish = dxy_usd_bias.upper() in bearish_set or dxy_usd_bias.upper() == 'BEARISH'
        dxy_neutral = dxy_usd_bias.upper() == 'NEUTRAL'

        fed_bullish = fed_bias in bullish_set
        fed_bearish = fed_bias in bearish_set
        fed_neutral = fed_bias == 'NEUTRAL'

        # Forte allineamento
        if (dxy_bullish and fed_bias == 'STRONG_BULLISH') or \
           (dxy_bearish and fed_bias == 'STRONG_BEARISH'):
            return 'strong_align', ALIGNMENT_BONUSES['strong_align']

        # Allineamento normale
        if (dxy_bullish and fed_bullish) or (dxy_bearish and fed_bearish):
            return 'align', ALIGNMENT_BONUSES['align']

        # Uno neutral
        if dxy_neutral or fed_neutral:
            return 'neutral', ALIGNMENT_BONUSES['neutral']

        # Forte conflitto
        if (dxy_bullish and fed_bias == 'STRONG_BEARISH') or \
           (dxy_bearish and fed_bias == 'STRONG_BULLISH'):
            return 'strong_conflict', ALIGNMENT_BONUSES['strong_conflict']

        # Conflitto normale
        if (dxy_bullish and fed_bearish) or (dxy_bearish and fed_bullish):
            return 'conflict', ALIGNMENT_BONUSES['conflict']

        return 'neutral', 0.0

    def get_combined_macro_signal(self, pair: str, dxy_analysis: Dict) -> Dict:
        """
        Combina Fed Sentiment e DXY Model per un segnale macro unificato.

        Args:
            pair: Coppia forex
            dxy_analysis: Output dal DXY Model

        Returns:
            Dict con segnale macro combinato
        """
        fed_bias = self.get_pair_bias(pair)
        fed_sentiment = self.get_current_sentiment()

        dxy_usd_bias = dxy_analysis.get('usd_bias', 'NEUTRAL')
        alignment_type, alignment_bonus = self.get_alignment_with_dxy(dxy_usd_bias)

        result = {
            'pair': pair,
            # DXY (trimestrale)
            'dxy_usd_bias': dxy_usd_bias,
            'dxy_pair_bias': dxy_analysis.get('pair_bias', 'NEUTRAL'),
            # Fed Sentiment (mensile)
            'fed_sentiment': fed_sentiment['sentiment'],
            'fed_regime': fed_sentiment['regime'],
            'fed_usd_bias': fed_sentiment['usd_bias'],
            'fed_pair_bias': fed_bias['pair_bias'],
            # Combinazione
            'alignment': alignment_type,
            'alignment_bonus': alignment_bonus,
            'combined_bias': 'NEUTRAL',
            'combined_signal': None,
            'confidence': 0.5,
            'reasoning': '',
        }

        # Logica di combinazione
        dxy_signal = dxy_analysis.get('aligned_with')
        fed_signal = fed_bias.get('aligned_with')

        if alignment_type in ['strong_align', 'align']:
            # Entrambi concordano
            result['combined_bias'] = fed_bias['pair_bias']
            result['combined_signal'] = fed_signal or dxy_signal
            result['confidence'] = min(0.95, fed_bias['confidence'] + abs(alignment_bonus))
            result['reasoning'] = f"DXY ({dxy_usd_bias}) + Fed ({fed_sentiment['regime']}) ALIGNED → {result['combined_signal']}"

        elif alignment_type in ['conflict', 'strong_conflict']:
            # In conflitto - usa DXY come direzione principale (trimestrale) ma riduci confidence
            result['combined_bias'] = dxy_analysis.get('pair_bias', 'NEUTRAL')
            result['combined_signal'] = dxy_signal
            result['confidence'] = max(0.4, fed_bias['confidence'] + alignment_bonus)  # alignment_bonus è negativo
            result['reasoning'] = f"DXY ({dxy_usd_bias}) vs Fed ({fed_sentiment['regime']}) CONFLICT → Use DXY with penalty"

        else:
            # Uno neutral - usa quello che ha segnale
            if fed_signal and not dxy_signal:
                result['combined_bias'] = fed_bias['pair_bias']
                result['combined_signal'] = fed_signal
                result['confidence'] = fed_bias['confidence']
                result['reasoning'] = f"DXY neutral, Fed ({fed_sentiment['regime']}) → {fed_signal}"
            elif dxy_signal and not fed_signal:
                result['combined_bias'] = dxy_analysis.get('pair_bias', 'NEUTRAL')
                result['combined_signal'] = dxy_signal
                result['confidence'] = 0.6
                result['reasoning'] = f"Fed neutral, DXY ({dxy_usd_bias}) → {dxy_signal}"
            else:
                result['reasoning'] = "Both neutral → No macro signal"

        return result

    def get_summary(self) -> str:
        """Ritorna un riepilogo dello stato Fed Sentiment."""
        fed = self.get_current_sentiment()

        if not fed['available']:
            return "Fed Sentiment: DATA NOT AVAILABLE"

        lines = [
            "=" * 50,
            "FED SENTIMENT MODEL (Monthly)",
            "=" * 50,
            f"  Date: {fed['date']}",
            f"  Sentiment: {fed['sentiment']:+.3f}",
            f"  Regime: {fed['regime']}",
            f"  USD Bias: {fed['usd_bias']}",
            f"  Momentum: {fed['momentum']:+.3f}",
            f"  MA(30): {fed['ma_30']:+.3f}",
            f"  MA(90): {fed['ma_90']:+.3f}",
        ]

        return "\n".join(lines)


# Singleton instance
_fed_sentiment_model = None


def get_fed_sentiment_model() -> FedSentimentModel:
    """Get singleton FedSentimentModel instance."""
    global _fed_sentiment_model
    if _fed_sentiment_model is None:
        _fed_sentiment_model = FedSentimentModel()
    return _fed_sentiment_model
