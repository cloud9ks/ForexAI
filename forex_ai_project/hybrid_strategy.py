"""
================================================================================
STRATEGIA IBRIDA - LSTM + FORZA RELATIVA - NexNow LTD
================================================================================
Combina:
1. LSTM Model: Predice direzione del mercato (LONG/SHORT/HOLD)
2. FR Indicators: Conferma e timing dell'entrata

Logica:
- LSTM dà il segnale direzionale
- FR Spread conferma la forza relativa
- ATR Dashboard evita eccessi
- Position sizing basato su confidenza LSTM
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================================================
# LSTM MODEL (stesso di train_model.py)
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2, num_classes=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.classifier(context)
        return output


# ============================================================================
# STRATEGIA IBRIDA
# ============================================================================

class HybridStrategy:
    """
    Strategia che combina LSTM predictor + FR indicators.

    Entry Rules:
    1. LSTM predice LONG con confidenza > threshold
    2. FR Spread > min_spread (conferma forza valuta)
    3. ATR percentile in range (non in eccesso)
    4. RSI non in zona estrema

    Exit Rules:
    1. SL/TP basati su ATR
    2. FR Spread reversal
    3. LSTM cambia previsione
    """

    def __init__(self, model_path: str, config: dict = None):
        self.config = config or {
            # LSTM
            'lstm_confidence_threshold': 0.45,  # Min confidenza per entrare

            # FR Filters
            'fr_spread_min': 10,           # FR deve essere almeno 10
            'atr_percentile_max': 70,
            'atr_percentile_min': 20,

            # Trading
            'initial_balance': 10000,
            'risk_per_trade': 0.015,       # 1.5% per trade
            'max_position_size': 0.1,
            'sl_atr_multiplier': 1.5,
            'tp_atr_multiplier': 2.0,

            # Costs
            'spread_pips': 1.5,
            'commission_per_lot': 7.0,

            # Filters
            'min_bars_between_trades': 6,
        }

        # Carica modello LSTM
        self.load_model(model_path)
        self.reset()

    def load_model(self, model_path: str):
        """Carica il modello LSTM"""
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)

        # Determina input_size dal checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        input_size = state_dict['lstm.weight_ih_l0'].shape[1]
        print(f"LSTM input_size: {input_size}")

        self.model = LSTMModel(input_size=input_size).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.input_size = input_size

    def reset(self):
        """Reset state"""
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.position_size = 0
        self.entry_confidence = 0
        self.entry_time = None
        self.last_trade_bar = 0

        self.balance = self.config['initial_balance']
        self.trades = []
        self.equity_curve = []

    def get_lstm_prediction(self, features: np.ndarray) -> tuple:
        """
        Ottiene predizione LSTM.

        Returns:
            (direction, confidence): -1/0/1 e probabilità
        """
        with torch.no_grad():
            # Prepara input (batch=1, seq=1, features)
            x = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)[0]

            # Classi: 0=SELL, 1=HOLD, 2=BUY
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()

            # Mappa a direzione
            if pred_class == 2:  # BUY
                direction = 1
            elif pred_class == 0:  # SELL
                direction = -1
            else:  # HOLD
                direction = 0

            return direction, confidence

    def check_entry(self, features: np.ndarray, fr_data: dict, current_bar: int) -> tuple:
        """
        Controlla condizioni di entrata.

        Returns:
            (direction, confidence) o (0, 0) se no entry
        """
        # Cooldown
        if current_bar - self.last_trade_bar < self.config['min_bars_between_trades']:
            return 0, 0

        # LSTM prediction
        direction, confidence = self.get_lstm_prediction(features)

        # Check confidence
        if confidence < self.config['lstm_confidence_threshold']:
            return 0, 0

        if direction == 0:  # HOLD
            return 0, 0

        # FR Spread confirmation
        fr_spread = fr_data.get('fr_spread', 0)

        # Per LONG: FR deve essere positivo
        if direction == 1:
            if fr_spread < self.config['fr_spread_min']:
                return 0, 0

        # Per SHORT: FR deve essere negativo
        if direction == -1:
            if fr_spread > -self.config['fr_spread_min']:
                return 0, 0

        # ATR percentile filter
        atr_pct = fr_data.get('atr_percentile', 50)
        if atr_pct > self.config['atr_percentile_max']:
            return 0, 0
        if atr_pct < self.config['atr_percentile_min']:
            return 0, 0

        return direction, confidence

    def calculate_position_size(self, atr: float, confidence: float) -> float:
        """Position size basato su rischio e confidenza"""
        risk_amount = self.balance * self.config['risk_per_trade']

        # Aggiusta per confidenza (più confidenza = più size)
        confidence_mult = 0.5 + confidence  # Range 0.5 - 1.5

        sl_pips = atr * self.config['sl_atr_multiplier'] / 0.0001
        if sl_pips > 0:
            size = (risk_amount / (sl_pips * 10)) * confidence_mult
            size = min(size, self.config['max_position_size'])
            size = max(size, 0.01)
        else:
            size = 0.01

        return round(size, 2)

    def open_position(self, direction: int, price: float, atr: float,
                     confidence: float, time):
        """Apre posizione"""
        self.position = direction
        self.entry_price = price
        self.entry_confidence = confidence
        self.entry_time = time

        # Position size
        self.position_size = self.calculate_position_size(atr, confidence)

        # SL/TP
        sl_distance = atr * self.config['sl_atr_multiplier']
        tp_distance = atr * self.config['tp_atr_multiplier']

        if direction == 1:
            self.stop_loss = price - sl_distance
            self.take_profit = price + tp_distance
        else:
            self.stop_loss = price + sl_distance
            self.take_profit = price - tp_distance

    def check_exit(self, current_price: float, fr_data: dict) -> tuple:
        """
        Controlla condizioni di uscita.

        Returns:
            (should_exit, reason)
        """
        if self.position == 0:
            return False, ''

        # SL hit
        if self.position == 1 and current_price <= self.stop_loss:
            return True, 'SL'
        if self.position == -1 and current_price >= self.stop_loss:
            return True, 'SL'

        # TP hit
        if self.position == 1 and current_price >= self.take_profit:
            return True, 'TP'
        if self.position == -1 and current_price <= self.take_profit:
            return True, 'TP'

        # FR reversal (forte)
        fr_spread = fr_data.get('fr_spread', 0)
        if self.position == 1 and fr_spread < -20:
            return True, 'FR Reversal'
        if self.position == -1 and fr_spread > 20:
            return True, 'FR Reversal'

        return False, ''

    def close_position(self, price: float, time, pair: str, reason: str):
        """Chiude posizione e registra trade"""
        pip_value = 0.0001 if 'JPY' not in pair else 0.01

        if self.position == 1:
            pips = (price - self.entry_price) / pip_value
        else:
            pips = (self.entry_price - price) / pip_value

        # Costi
        spread_pips = self.config['spread_pips']
        commission = self.config['commission_per_lot'] * self.position_size

        pnl = (pips - spread_pips) * self.position_size * 10 - commission
        self.balance += pnl

        self.trades.append({
            'pair': pair,
            'direction': 'LONG' if self.position == 1 else 'SHORT',
            'entry_time': self.entry_time,
            'exit_time': time,
            'entry_price': self.entry_price,
            'exit_price': price,
            'size': self.position_size,
            'confidence': self.entry_confidence,
            'pips': pips - spread_pips,
            'pnl': pnl,
            'reason': reason,
        })

        # Reset
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0

    def run_backtest(self, features_df: pd.DataFrame, raw_df: pd.DataFrame, pair: str):
        """Esegue backtest"""
        self.reset()

        # Allinea
        common_idx = features_df.index.intersection(raw_df.index)
        features_df = features_df.loc[common_idx]
        raw_df = raw_df.loc[common_idx]

        # Prepara features per LSTM
        feature_cols = [c for c in features_df.columns if features_df[c].dtype in ['float64', 'float32', 'int64']]

        # Se abbiamo meno features, paddiamo con zeri
        num_features = len(feature_cols)
        need_padding = self.input_size - num_features if num_features < self.input_size else 0

        if need_padding > 0:
            print(f"Info: {pair} has {num_features} features, padding {need_padding} zeros")

        print(f"\nBacktest {pair}: {len(features_df):,} bars, {len(feature_cols)} features")

        for i, (idx, row) in enumerate(tqdm(features_df.iterrows(), total=len(features_df), desc=pair, leave=False)):
            if i < 100:  # Skip warmup
                continue

            current_price = raw_df.loc[idx, 'Close']
            atr = row.get('atr_14', 0.001)

            # Features per LSTM
            lstm_features = row[feature_cols].values.astype(np.float32)
            lstm_features = np.nan_to_num(lstm_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Padding se necessario
            if need_padding > 0:
                lstm_features = np.concatenate([lstm_features, np.zeros(need_padding, dtype=np.float32)])

            # FR data
            fr_data = {
                'fr_spread': row.get('fr_spread', 0),
                'atr_percentile': row.get('atr_percentile', 50),
            }

            # Equity curve (giornaliero)
            if i % 24 == 0:
                self.equity_curve.append({'date': idx, 'balance': self.balance})

            # Check exit
            if self.position != 0:
                should_exit, reason = self.check_exit(current_price, fr_data)
                if should_exit:
                    self.close_position(current_price, idx, pair, reason)

            # Check entry
            if self.position == 0:
                direction, confidence = self.check_entry(lstm_features, fr_data, i)

                if direction != 0:
                    self.open_position(direction, current_price, atr, confidence, idx)
                    self.last_trade_bar = i

        # Chiudi posizione finale
        if self.position != 0:
            self.close_position(raw_df.iloc[-1]['Close'], features_df.index[-1], pair, 'EOD')

        return self.calculate_metrics(pair)

    def calculate_metrics(self, pair: str) -> dict:
        """Calcola metriche"""
        if len(self.trades) == 0:
            return {'pair': pair, 'error': 'No trades'}

        pnls = [t['pnl'] for t in self.trades]
        winning = [t for t in self.trades if t['pnl'] > 0]
        losing = [t for t in self.trades if t['pnl'] <= 0]

        gross_profit = sum(t['pnl'] for t in winning)
        gross_loss = abs(sum(t['pnl'] for t in losing))

        # Analisi per confidenza
        high_conf = [t for t in self.trades if t['confidence'] > 0.6]
        high_conf_wins = [t for t in high_conf if t['pnl'] > 0]

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
            'avg_confidence': np.mean([t['confidence'] for t in self.trades]),
            'high_conf_trades': len(high_conf),
            'high_conf_win_rate': len(high_conf_wins) / len(high_conf) * 100 if high_conf else 0,
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("STRATEGIA IBRIDA - LSTM + FORZA RELATIVA")
    print("=" * 60)

    # Trova modello LSTM
    model_path = Path('models/lstm/best_model.pt')
    if not model_path.exists():
        model_path = Path('models/lstm_model.pt')
    if not model_path.exists():
        # Cerca qualsiasi .pt
        model_files = list(Path('models').glob('**/*.pt'))
        if model_files:
            model_path = model_files[0]
        else:
            print("Errore: Nessun modello LSTM trovato!")
            return

    print(f"Usando modello: {model_path}")

    # Inizializza strategia
    strategy = HybridStrategy(str(model_path))

    # Pairs
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY"]

    processed_dir = Path('./data/processed')
    raw_dir = Path('./data/raw')

    all_results = []
    all_trades = []

    for pair in pairs:
        features_path = processed_dir / f"{pair}_features.parquet"
        raw_path = raw_dir / f"{pair}_H1.parquet"

        if not features_path.exists() or not raw_path.exists():
            continue

        features_df = pd.read_parquet(features_path)
        raw_df = pd.read_parquet(raw_path)

        # Test: ultimi 20%
        test_start = int(len(features_df) * 0.8)
        features_test = features_df.iloc[test_start:]
        raw_test = raw_df.iloc[test_start:]

        # Run
        metrics = strategy.run_backtest(features_test, raw_test, pair)

        if 'error' in metrics:
            print(f"\n{pair}: {metrics['error']}")
            continue

        all_results.append(metrics)
        all_trades.extend(strategy.trades)

        print(f"\n{pair}:")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Net Profit: ${metrics['net_profit']:.2f}")
        print(f"  Return: {metrics['return_pct']:.2f}%")
        print(f"  Max DD: {metrics['max_drawdown']:.2f}%")
        print(f"  Avg Confidence: {metrics['avg_confidence']:.2f}")
        print(f"  High Conf Win Rate: {metrics['high_conf_win_rate']:.1f}%")

    # Riepilogo
    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)

    if all_results:
        total_trades = sum(r['total_trades'] for r in all_results)
        total_profit = sum(r['net_profit'] for r in all_results)
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        avg_pf = np.mean([r['profit_factor'] for r in all_results if r['profit_factor'] != float('inf')])

        print(f"Pairs: {len(all_results)}")
        print(f"Trades: {total_trades}")
        print(f"Profit: ${total_profit:,.2f}")
        print(f"Win Rate: {avg_win_rate:.1f}%")
        print(f"Profit Factor: {avg_pf:.2f}")

        # Salva
        output_dir = Path('models/hybrid_strategy')
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'backtest_results.json', 'w') as f:
            json.dump({
                'timestamp': str(datetime.now()),
                'model_path': str(model_path),
                'pairs': all_results,
                'summary': {
                    'total_trades': total_trades,
                    'total_profit': total_profit,
                    'avg_win_rate': avg_win_rate,
                }
            }, f, indent=2, default=str)

        print(f"\nSalvato in: {output_dir}")


if __name__ == "__main__":
    main()
