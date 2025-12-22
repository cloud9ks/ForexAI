"""
================================================================================
CVOL MONITOR - Currency Volatility Warning System
================================================================================
Monitora la volatilità delle coppie forex e genera warning quando è troppo alta.
Quando CVOL è in WARNING, riduce il risk sizing o blocca i trade.

Logica:
- CVOL = volatility_20 * sqrt(252) * 100 * 2 (annualizzata)
- Soglia WARNING = media storica + 1.5 * std
- Se CVOL > soglia → WARNING attivo

Integrazione:
- CVOL OK → Trade normale (risk 0.65%)
- CVOL WARNING → Risk ridotto al 50% (0.325%)
- CVOL EXTREME (>2 std) → Skip trade
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Soglie CVOL pre-calcolate per ogni pair (basate su dati storici 2019-2024)
# Queste vengono aggiornate dinamicamente se disponibili dati freschi
CVOL_THRESHOLDS = {
    'EURUSD': {'warning': 12.0, 'extreme': 18.0, 'normal_range': (6.0, 10.0)},
    'GBPUSD': {'warning': 14.0, 'extreme': 20.0, 'normal_range': (7.0, 12.0)},
    'USDJPY': {'warning': 13.0, 'extreme': 19.0, 'normal_range': (6.5, 11.0)},
    'AUDUSD': {'warning': 13.5, 'extreme': 19.5, 'normal_range': (7.0, 11.5)},
}

# Risk adjustment factors
CVOL_RISK_FACTORS = {
    'OK': 1.0,           # Normal risk
    'WARNING': 0.5,      # 50% risk reduction
    'EXTREME': 0.0,      # No trade
}


class CVOLMonitor:
    """
    Monitora la volatilità delle coppie forex e determina lo stato CVOL.
    """

    def __init__(self):
        self.cvol_cache = {}  # Cache per evitare ricalcoli frequenti
        self.cache_duration = timedelta(hours=1)  # Ricalcola ogni ora
        self.thresholds = CVOL_THRESHOLDS.copy()

    def calculate_cvol(self, df: pd.DataFrame) -> float:
        """
        Calcola CVOL (Currency Volatility Index) da un DataFrame OHLC.

        Args:
            df: DataFrame con colonne 'close' (minimo 20 righe)

        Returns:
            CVOL value (annualizzato, percentuale)
        """
        if len(df) < 20:
            return 0.0

        # Calcola returns
        returns = df['close'].pct_change().dropna()

        # Volatilità rolling 20 periodi
        volatility_20 = returns.rolling(20).std().iloc[-1]

        # Annualizza (H1 = 252*24 ore/anno, ma usiamo 252 per semplicità)
        # Moltiplica per 100 per avere percentuale, *2 per scaling
        cvol = volatility_20 * np.sqrt(252) * 100 * 2

        return cvol if not np.isnan(cvol) else 0.0

    def get_cvol_status(self, pair: str, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Ottiene lo stato CVOL per una coppia.

        Args:
            pair: Coppia forex (es. 'EURUSD')
            df: DataFrame OHLC opzionale (se non fornito, usa cache)

        Returns:
            Dict con cvol, status, risk_factor, message
        """
        result = {
            'pair': pair,
            'cvol': 0.0,
            'status': 'OK',
            'risk_factor': 1.0,
            'warning_level': 0.0,
            'extreme_level': 0.0,
            'message': '',
            'timestamp': datetime.now().isoformat(),
        }

        # Get thresholds for this pair
        thresholds = self.thresholds.get(pair, {
            'warning': 13.0,
            'extreme': 19.0,
            'normal_range': (7.0, 11.0)
        })

        result['warning_level'] = thresholds['warning']
        result['extreme_level'] = thresholds['extreme']

        # Check cache
        cache_key = pair
        if cache_key in self.cvol_cache:
            cached = self.cvol_cache[cache_key]
            if datetime.now() - cached['time'] < self.cache_duration:
                return cached['result']

        # Calculate CVOL if df provided
        if df is not None and len(df) >= 20:
            cvol = self.calculate_cvol(df)
            result['cvol'] = round(cvol, 2)

            # Determine status
            if cvol >= thresholds['extreme']:
                result['status'] = 'EXTREME'
                result['risk_factor'] = CVOL_RISK_FACTORS['EXTREME']
                result['message'] = f"EXTREME volatility ({cvol:.1f} > {thresholds['extreme']}). SKIP TRADE!"
            elif cvol >= thresholds['warning']:
                result['status'] = 'WARNING'
                result['risk_factor'] = CVOL_RISK_FACTORS['WARNING']
                result['message'] = f"High volatility ({cvol:.1f} > {thresholds['warning']}). Risk reduced 50%."
            else:
                result['status'] = 'OK'
                result['risk_factor'] = CVOL_RISK_FACTORS['OK']
                result['message'] = f"Normal volatility ({cvol:.1f}). Trading OK."

            # Update cache
            self.cvol_cache[cache_key] = {
                'time': datetime.now(),
                'result': result
            }
        else:
            result['message'] = "Insufficient data for CVOL calculation"

        return result

    def get_adjusted_risk(self, pair: str, base_risk: float, df: Optional[pd.DataFrame] = None) -> Tuple[float, Dict]:
        """
        Calcola il risk aggiustato in base al CVOL.

        Args:
            pair: Coppia forex
            base_risk: Risk base (es. 0.0065 = 0.65%)
            df: DataFrame OHLC opzionale

        Returns:
            Tuple(adjusted_risk, cvol_status)
        """
        cvol_status = self.get_cvol_status(pair, df)
        adjusted_risk = base_risk * cvol_status['risk_factor']

        logger.info(f"CVOL {pair}: {cvol_status['cvol']:.1f} ({cvol_status['status']}) → Risk: {base_risk:.2%} → {adjusted_risk:.2%}")

        return adjusted_risk, cvol_status

    def should_trade(self, pair: str, df: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
        """
        Verifica se è possibile tradare in base al CVOL.

        Returns:
            Tuple(can_trade, reason)
        """
        cvol_status = self.get_cvol_status(pair, df)

        if cvol_status['status'] == 'EXTREME':
            return False, cvol_status['message']
        else:
            return True, cvol_status['message']

    def update_thresholds_from_data(self, pair: str, historical_df: pd.DataFrame):
        """
        Aggiorna le soglie CVOL basandosi sui dati storici.

        Args:
            pair: Coppia forex
            historical_df: DataFrame con almeno 1 anno di dati
        """
        if len(historical_df) < 252 * 24:  # Almeno 1 anno di dati H1
            logger.warning(f"Insufficient data to update {pair} thresholds")
            return

        # Calcola CVOL rolling per tutto lo storico
        returns = historical_df['close'].pct_change().dropna()
        volatility_20 = returns.rolling(20).std()
        cvol_series = volatility_20 * np.sqrt(252) * 100 * 2
        cvol_series = cvol_series.dropna()

        # Calcola statistiche
        mean_cvol = cvol_series.mean()
        std_cvol = cvol_series.std()

        # Nuove soglie
        warning_level = round(mean_cvol + 1.5 * std_cvol, 1)
        extreme_level = round(mean_cvol + 2.0 * std_cvol, 1)

        self.thresholds[pair] = {
            'warning': warning_level,
            'extreme': extreme_level,
            'normal_range': (round(mean_cvol - std_cvol, 1), round(mean_cvol + std_cvol, 1))
        }

        logger.info(f"Updated {pair} thresholds: WARNING={warning_level}, EXTREME={extreme_level}")

    def get_all_status(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Ottiene lo stato CVOL per tutte le coppie.

        Args:
            pairs_data: Dict con {pair: DataFrame}

        Returns:
            Dict con stato per ogni pair
        """
        result = {}
        for pair, df in pairs_data.items():
            result[pair] = self.get_cvol_status(pair, df)
        return result

    def get_summary(self) -> str:
        """Ritorna un riepilogo dello stato CVOL."""
        lines = [
            "=" * 50,
            "CVOL MONITOR STATUS",
            "=" * 50,
        ]

        for pair, cached in self.cvol_cache.items():
            status = cached['result']
            emoji = "🟢" if status['status'] == 'OK' else "🟡" if status['status'] == 'WARNING' else "🔴"
            lines.append(f"  {pair}: {emoji} {status['cvol']:.1f} ({status['status']})")

        if not self.cvol_cache:
            lines.append("  No data available")

        return "\n".join(lines)


# Singleton instance
_cvol_monitor = None


def get_cvol_monitor() -> CVOLMonitor:
    """Get singleton CVOLMonitor instance."""
    global _cvol_monitor
    if _cvol_monitor is None:
        _cvol_monitor = CVOLMonitor()
    return _cvol_monitor
