"""
================================================================================
STRATEGIA FORZA RELATIVA - NexNow LTD
================================================================================
Sistema di trading basato sugli indicatori Forza Relativa:
- Currency Strength (forza delle 8 valute)
- FR Spread (differenza di forza tra base e quote)
- ATR Dashboard (eccesso dalla media)
- Filtri di sessione e volatilità

Regole chiare e interpretabili per il trading.
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURAZIONE STRATEGIA
# ============================================================================

STRATEGY_CONFIG = {
    # === SEGNALI ENTRATA (PIU' SELETTIVI) ===
    # FR Spread thresholds - richiedi segnali più forti
    'fr_spread_entry_strong': 40,      # > 40 = segnale molto forte
    'fr_spread_entry_min': 30,         # Minimo 30 per considerare trade

    # ATR Dashboard - preferisci mercati in range, non in eccesso
    'atr_percentile_max': 60,          # Non entrare se > 60% (già in movimento)
    'atr_percentile_min': 30,          # Non entrare se < 30% (troppo flat)

    # RSI - cerca divergenze o livelli estremi
    'rsi_overbought': 65,              # Più conservativo
    'rsi_oversold': 35,

    # Bollinger Bands - conferma
    'bb_upper_threshold': 0.85,        # Long solo se < 85% (spazio per salire)
    'bb_lower_threshold': 0.15,        # Short solo se > 15% (spazio per scendere)

    # MACD - conferma trend
    'require_macd_confirmation': True,

    # === FILTRI ===
    # Sessioni (solo overlap London-NY)
    'require_overlap_session': 1,       # Solo durante overlap

    # Volatilità
    'min_volatility_ratio': 0.9,
    'max_volatility_ratio': 1.3,       # Non entrare in volatilità eccessiva

    # === GESTIONE POSIZIONE ===
    'max_position_size': 0.1,
    'sl_atr_multiplier': 1.2,          # SL più stretto
    'tp_atr_multiplier': 2.5,          # TP più ambizioso (1:2 ratio)
    'trailing_stop': True,
    'trailing_activation': 1.0,        # Attiva trailing dopo 1 ATR di profitto

    # === COSTI ===
    'spread_pips': 1.5,
    'commission_per_lot': 7.0,

    # === CAPITAL ===
    'initial_balance': 10000,
    'risk_per_trade': 0.01,            # 1% rischio per trade (più conservativo)

    # === COOLDOWN ===
    'min_bars_between_trades': 4,      # Almeno 4 ore tra trade
}


# ============================================================================
# CLASSE STRATEGIA FR
# ============================================================================

class ForzaRelativaStrategy:
    """
    Implementazione della strategia Forza Relativa.

    Logica:
    1. FR Spread > soglia → segnale LONG
    2. FR Spread < -soglia → segnale SHORT
    3. Filtri: ATR%, RSI, Sessione, Volatilità
    4. Position sizing basato su rischio e ATR
    5. SL/TP dinamici basati su ATR
    """

    def __init__(self, config: dict = None):
        self.config = config or STRATEGY_CONFIG
        self.reset()

    def reset(self):
        """Reset stato strategia"""
        self.position = 0  # -1 = short, 0 = flat, 1 = long
        self.entry_price = 0
        self.entry_time = None
        self.stop_loss = 0
        self.take_profit = 0
        self.position_size = 0
        self.balance = self.config['initial_balance']
        self.trades = []
        self.equity_curve = []

    def check_entry_signal(self, row: pd.Series, pair: str, current_step: int = 0) -> int:
        """
        Controlla segnale di entrata con filtri di confluenza.

        Returns:
            1 = LONG, -1 = SHORT, 0 = NESSUN SEGNALE
        """
        # Estrai indicatori
        fr_spread = row.get('fr_spread', 0)
        atr_pct = row.get('atr_percentile', 50)
        rsi = row.get('rsi_14', 50)
        is_overlap = row.get('is_overlap', 0)
        vol_ratio = row.get('volatility_ratio', 1.0)
        bb_position = row.get('bb_position', 0.5)
        macd_hist = row.get('macd_hist', 0)

        # === FILTRI BASE ===

        # Filtro cooldown (almeno N bars dall'ultimo trade)
        if hasattr(self, 'last_trade_bar'):
            if current_step - self.last_trade_bar < self.config.get('min_bars_between_trades', 4):
                return 0

        # Filtro sessione overlap (migliore liquidità)
        if self.config.get('require_overlap_session', 0) and not is_overlap:
            return 0

        # Filtro ATR percentile
        if atr_pct > self.config['atr_percentile_max']:
            return 0
        if atr_pct < self.config['atr_percentile_min']:
            return 0

        # Filtro volatilità (non troppo bassa, non troppo alta)
        if vol_ratio < self.config.get('min_volatility_ratio', 0.9):
            return 0
        if vol_ratio > self.config.get('max_volatility_ratio', 1.5):
            return 0

        # === SEGNALI CON CONFLUENZA ===

        # LONG: FR Spread positivo + RSI non ipercomprato + BB basso + MACD positivo
        if fr_spread >= self.config['fr_spread_entry_min']:
            # Controllo RSI
            if rsi >= self.config['rsi_overbought']:
                return 0  # Troppo ipercomprato

            # Controllo Bollinger (spazio per salire)
            if bb_position > self.config.get('bb_upper_threshold', 0.85):
                return 0  # Già vicino alla banda superiore

            # Controllo MACD (conferma momentum)
            if self.config.get('require_macd_confirmation', True):
                if macd_hist < 0:
                    return 0  # MACD non conferma

            # Segnale valido
            return 1

        # SHORT: FR Spread negativo + RSI non ipervenduto + BB alto + MACD negativo
        if fr_spread <= -self.config['fr_spread_entry_min']:
            # Controllo RSI
            if rsi <= self.config['rsi_oversold']:
                return 0  # Troppo ipervenduto

            # Controllo Bollinger (spazio per scendere)
            if bb_position < self.config.get('bb_lower_threshold', 0.15):
                return 0  # Già vicino alla banda inferiore

            # Controllo MACD (conferma momentum)
            if self.config.get('require_macd_confirmation', True):
                if macd_hist > 0:
                    return 0  # MACD non conferma

            # Segnale valido
            return -1

        return 0

    def calculate_position_size(self, atr: float, price: float) -> float:
        """Calcola size posizione basata su rischio e ATR"""
        # Rischio in dollari
        risk_amount = self.balance * self.config['risk_per_trade']

        # Stop loss in pips
        sl_pips = atr * self.config['sl_atr_multiplier'] / 0.0001

        # Position size (approssimato)
        # Per un mini lot (0.1), 1 pip ~ $1
        if sl_pips > 0:
            size = risk_amount / (sl_pips * 10)  # ~$10 per pip per 0.1 lot
            size = min(size, self.config['max_position_size'])
            size = max(size, 0.01)  # Minimo 0.01 lot
        else:
            size = 0.01

        return round(size, 2)

    def calculate_sl_tp(self, entry_price: float, direction: int, atr: float) -> tuple:
        """Calcola Stop Loss e Take Profit"""
        sl_distance = atr * self.config['sl_atr_multiplier']
        tp_distance = atr * self.config['tp_atr_multiplier']

        if direction == 1:  # Long
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # Short
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        return sl, tp

    def check_exit_conditions(self, row: pd.Series, current_price: float) -> bool:
        """Controlla condizioni di uscita"""
        if self.position == 0:
            return False

        # Stop Loss hit
        if self.position == 1 and current_price <= self.stop_loss:
            return True
        if self.position == -1 and current_price >= self.stop_loss:
            return True

        # Take Profit hit
        if self.position == 1 and current_price >= self.take_profit:
            return True
        if self.position == -1 and current_price <= self.take_profit:
            return True

        # FR Spread reversal (exit se FR va contro posizione)
        fr_spread = row.get('fr_spread', 0)
        if self.position == 1 and fr_spread < -5:  # Long ma FR negativo
            return True
        if self.position == -1 and fr_spread > 5:  # Short ma FR positivo
            return True

        return False

    def update_trailing_stop(self, current_price: float, atr: float):
        """Aggiorna trailing stop"""
        if not self.config['trailing_stop'] or self.position == 0:
            return

        trail_distance = atr * self.config['sl_atr_multiplier']

        if self.position == 1:  # Long
            new_sl = current_price - trail_distance
            if new_sl > self.stop_loss:
                self.stop_loss = new_sl

        elif self.position == -1:  # Short
            new_sl = current_price + trail_distance
            if new_sl < self.stop_loss:
                self.stop_loss = new_sl

    def execute_trade(self, direction: int, price: float, time, atr: float, pair: str):
        """Esegue un trade"""
        # Calcola size
        self.position_size = self.calculate_position_size(atr, price)

        # Calcola SL/TP
        self.stop_loss, self.take_profit = self.calculate_sl_tp(price, direction, atr)

        # Set position
        self.position = direction
        self.entry_price = price
        self.entry_time = time

    def close_trade(self, price: float, time, pair: str, reason: str = ''):
        """Chiude un trade e calcola PnL"""
        if self.position == 0:
            return

        # Calcola PnL
        pip_value = 0.0001 if 'JPY' not in pair else 0.01

        if self.position == 1:  # Long
            pips = (price - self.entry_price) / pip_value
        else:  # Short
            pips = (self.entry_price - price) / pip_value

        # Costi
        spread_pips = self.config['spread_pips']
        commission = self.config['commission_per_lot'] * self.position_size

        # PnL netto
        pnl = (pips - spread_pips) * self.position_size * 10 - commission

        # Aggiorna balance
        self.balance += pnl

        # Registra trade
        self.trades.append({
            'pair': pair,
            'direction': 'LONG' if self.position == 1 else 'SHORT',
            'entry_time': self.entry_time,
            'exit_time': time,
            'entry_price': self.entry_price,
            'exit_price': price,
            'size': self.position_size,
            'pips': pips - spread_pips,
            'pnl': pnl,
            'reason': reason,
        })

        # Reset position
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0

    def run_backtest(self, features_df: pd.DataFrame, raw_df: pd.DataFrame, pair: str):
        """Esegue backtest su un pair"""
        self.reset()
        self.last_trade_bar = 0  # Per cooldown

        # Allinea dati
        common_idx = features_df.index.intersection(raw_df.index)
        features_df = features_df.loc[common_idx]
        raw_df = raw_df.loc[common_idx]

        print(f"\nBacktest {pair}: {len(features_df):,} bars")

        for i, (idx, row) in enumerate(tqdm(features_df.iterrows(), total=len(features_df), desc=pair, leave=False)):
            current_price = raw_df.loc[idx, 'Close']
            atr = row.get('atr_14', row.get('atr', 0.001))

            # Registra equity ogni giorno
            if i % 24 == 0:
                self.equity_curve.append({
                    'date': idx,
                    'balance': self.balance,
                })

            # Se abbiamo posizione, controlla uscita
            if self.position != 0:
                # Trailing stop
                self.update_trailing_stop(current_price, atr)

                # Check exit
                if self.check_exit_conditions(row, current_price):
                    reason = 'SL/TP' if (
                        (self.position == 1 and (current_price <= self.stop_loss or current_price >= self.take_profit)) or
                        (self.position == -1 and (current_price >= self.stop_loss or current_price <= self.take_profit))
                    ) else 'FR Reversal'
                    self.close_trade(current_price, idx, pair, reason)

            # Se flat, cerca entrata
            if self.position == 0:
                signal = self.check_entry_signal(row, pair, current_step=i)

                if signal != 0:
                    self.execute_trade(signal, current_price, idx, atr, pair)
                    self.last_trade_bar = i  # Aggiorna per cooldown

        # Chiudi posizione aperta a fine backtest
        if self.position != 0:
            last_price = raw_df.iloc[-1]['Close']
            self.close_trade(last_price, features_df.index[-1], pair, 'End of data')

        return self.calculate_metrics(pair)

    def calculate_metrics(self, pair: str) -> dict:
        """Calcola metriche di performance"""
        if len(self.trades) == 0:
            return {'pair': pair, 'error': 'Nessun trade'}

        pnls = [t['pnl'] for t in self.trades]
        winning = [t for t in self.trades if t['pnl'] > 0]
        losing = [t for t in self.trades if t['pnl'] <= 0]

        gross_profit = sum(t['pnl'] for t in winning)
        gross_loss = abs(sum(t['pnl'] for t in losing))

        # Max Drawdown
        equity = [self.config['initial_balance']]
        for t in self.trades:
            equity.append(equity[-1] + t['pnl'])

        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            max_dd = max(max_dd, dd)

        return {
            'pair': pair,
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(self.trades) * 100,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': sum(pnls),
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'avg_win': np.mean([t['pnl'] for t in winning]) if winning else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing]) if losing else 0,
            'max_drawdown': max_dd,
            'final_balance': self.balance,
            'return_pct': (self.balance - self.config['initial_balance']) / self.config['initial_balance'] * 100,
            'avg_pips': np.mean([t['pips'] for t in self.trades]),
            'avg_holding_time': np.mean([(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in self.trades]),
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("STRATEGIA FORZA RELATIVA - BACKTEST")
    print("=" * 60)

    # Pairs da testare
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
             "EURJPY", "GBPJPY", "EURGBP"]

    processed_dir = Path('./data/processed')
    raw_dir = Path('./data/raw')

    all_results = []
    all_trades = []

    for pair in pairs:
        features_path = processed_dir / f"{pair}_features.parquet"
        raw_path = raw_dir / f"{pair}_H1.parquet"

        if not features_path.exists() or not raw_path.exists():
            print(f"{pair}: dati non trovati")
            continue

        # Carica dati
        features_df = pd.read_parquet(features_path)
        raw_df = pd.read_parquet(raw_path)

        # Usa ultimi 20% per test
        test_start = int(len(features_df) * 0.8)
        features_test = features_df.iloc[test_start:]
        raw_test = raw_df.iloc[test_start:]

        # Run backtest
        strategy = ForzaRelativaStrategy(STRATEGY_CONFIG)
        metrics = strategy.run_backtest(features_test, raw_test, pair)

        if 'error' in metrics:
            print(f"\n{pair}: {metrics['error']}")
            continue

        all_results.append(metrics)
        all_trades.extend(strategy.trades)

        # Stampa risultati
        print(f"\n{pair}:")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Net Profit: ${metrics['net_profit']:.2f}")
        print(f"  Return: {metrics['return_pct']:.2f}%")
        print(f"  Max DD: {metrics['max_drawdown']:.2f}%")
        print(f"  Avg Pips: {metrics['avg_pips']:.1f}")

    # === RIEPILOGO ===
    print("\n" + "=" * 60)
    print("RIEPILOGO TOTALE")
    print("=" * 60)

    if all_results:
        total_trades = sum(r['total_trades'] for r in all_results)
        total_profit = sum(r['net_profit'] for r in all_results)
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        avg_pf = np.mean([r['profit_factor'] for r in all_results if r['profit_factor'] != float('inf')])

        print(f"Pairs testati: {len(all_results)}")
        print(f"Trades totali: {total_trades}")
        print(f"Profit totale: ${total_profit:,.2f}")
        print(f"Win Rate medio: {avg_win_rate:.1f}%")
        print(f"Profit Factor medio: {avg_pf:.2f}")

        # Salva risultati
        output_dir = Path('models/fr_strategy')
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'backtest_results.json', 'w') as f:
            json.dump({
                'timestamp': str(datetime.now()),
                'config': STRATEGY_CONFIG,
                'pairs': all_results,
                'summary': {
                    'total_trades': total_trades,
                    'total_profit': total_profit,
                    'avg_win_rate': avg_win_rate,
                    'avg_profit_factor': avg_pf,
                }
            }, f, indent=2, default=str)

        # Salva trades
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(output_dir / 'all_trades.csv', index=False)

        print(f"\nRisultati salvati in: {output_dir}")

    else:
        print("Nessun risultato disponibile")


if __name__ == "__main__":
    main()
