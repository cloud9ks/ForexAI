"""
Backtest 2025 - Forex AI Trading
Esegue backtest solo sui dati 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURAZIONE BACKTEST 2025
# ============================================================================
CONFIG = {
    'initial_capital': 100000,
    'risk_per_trade': 0.0065,     # 0.65% rischio per trade
    'spread_pips': 1.5,
    'slippage_pips': 0.5,
    'commission_per_lot': 7,
    'leverage': 30,
    'min_confidence': 0.70,
    'take_profit_atr': 3.0,
    'stop_loss_atr': 1.5,
    'min_bars_between_trades': 72,
    'max_trades_per_day': 2,
    'max_lot_size': 10.0,
    'year_filter': 2025,  # Solo dati 2025
}


# ============================================================================
# MODELLO LSTM
# ============================================================================
class LSTMBalanced(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.classifier(context)
        return out


# ============================================================================
# BACKTEST ENGINE
# ============================================================================
class BacktestEngine:
    def __init__(self, config=CONFIG):
        self.config = config
        self.capital = config['initial_capital']
        self.initial_capital = config['initial_capital']
        self.positions = {}
        self.trades = []
        self.equity_curve = [config['initial_capital']]
        self.timestamps = []
        self.last_trade_time = {}

    def calculate_position_size(self, price, atr):
        base_capital = self.config['initial_capital']
        risk_amount = base_capital * self.config['risk_per_trade']
        sl_pips = atr * self.config['stop_loss_atr'] * 10000
        if sl_pips < 1:
            sl_pips = 10
        pip_value = 10
        lots = risk_amount / (sl_pips * pip_value)
        max_lots = self.config.get('max_lot_size', 5.0)
        lots = min(lots, max_lots)
        lots = max(lots, 0.01)
        return round(lots, 2)

    def open_position(self, pair, signal, price, atr, timestamp):
        if pair in self.positions:
            return False

        lots = self.calculate_position_size(price, atr)
        spread_cost = self.config['spread_pips'] * 10 * lots
        slippage_cost = self.config['slippage_pips'] * 10 * lots
        commission = self.config['commission_per_lot'] * lots

        if signal == 'BUY':
            entry_price = price + (self.config['spread_pips'] + self.config['slippage_pips']) * 0.0001
            sl = entry_price - atr * self.config['stop_loss_atr']
            tp = entry_price + atr * self.config['take_profit_atr']
        else:
            entry_price = price - self.config['slippage_pips'] * 0.0001
            sl = entry_price + atr * self.config['stop_loss_atr']
            tp = entry_price - atr * self.config['take_profit_atr']

        self.positions[pair] = {
            'signal': signal,
            'entry_price': entry_price,
            'lots': lots,
            'sl': sl,
            'tp': tp,
            'entry_time': timestamp,
            'costs': spread_cost + slippage_cost + commission
        }

        self.last_trade_time[pair] = timestamp
        return True

    def check_exits(self, high, low, close, timestamp):
        closed = []
        for pair, pos in list(self.positions.items()):
            exit_price = None
            exit_reason = None

            if pos['signal'] == 'BUY':
                if low <= pos['sl']:
                    exit_price = pos['sl']
                    exit_reason = 'SL'
                elif high >= pos['tp']:
                    exit_price = pos['tp']
                    exit_reason = 'TP'
            else:
                if high >= pos['sl']:
                    exit_price = pos['sl']
                    exit_reason = 'SL'
                elif low <= pos['tp']:
                    exit_price = pos['tp']
                    exit_reason = 'TP'

            if exit_price:
                self._close_position(pair, exit_price, exit_reason, timestamp)
                closed.append(pair)

        return closed

    def _close_position(self, pair, exit_price, reason, timestamp):
        pos = self.positions[pair]

        if pos['signal'] == 'BUY':
            pnl_pips = (exit_price - pos['entry_price']) * 10000
        else:
            pnl_pips = (pos['entry_price'] - exit_price) * 10000

        pnl_money = pnl_pips * 10 * pos['lots'] - pos['costs']
        self.capital += pnl_money

        self.trades.append({
            'pair': pair,
            'signal': pos['signal'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'lots': pos['lots'],
            'pnl_pips': pnl_pips,
            'pnl_money': pnl_money,
            'reason': reason,
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
        })

        del self.positions[pair]

    def update_equity(self, timestamp):
        self.equity_curve.append(self.capital)
        self.timestamps.append(timestamp)

    def get_stats(self):
        if not self.trades:
            return {'n_trades': 0}

        profits = [t['pnl_money'] for t in self.trades if t['pnl_money'] > 0]
        losses = [t['pnl_money'] for t in self.trades if t['pnl_money'] <= 0]

        n_trades = len(self.trades)
        win_rate = len(profits) / n_trades if n_trades > 0 else 0

        gross_profit = sum(profits) if profits else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = drawdown.max()

        returns = np.diff(equity) / equity[:-1]
        sharpe = np.sqrt(252 * 24) * returns.mean() / returns.std() if returns.std() > 0 else 0
        avg_trade = np.mean([t['pnl_money'] for t in self.trades])

        return {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'avg_trade': avg_trade,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }


# ============================================================================
# BACKTEST RUNNER
# ============================================================================
def run_backtest(model, pairs_data, device='cuda'):
    print("=" * 60)
    print("BACKTEST 2025 - FOREX AI TRADING")
    print("=" * 60)

    engine = BacktestEngine()
    print("\nPreparazione dati...")

    # Crea timeline unificata
    timeline = []
    for pair, (features, prices_df, labels) in pairs_data.items():
        for i in range(len(prices_df)):
            timeline.append({
                'timestamp': prices_df.index[i],
                'pair': pair,
                'idx': i,
                'features': features[i] if i < len(features) else features[-1],
                'open': prices_df['Open'].iloc[i],
                'high': prices_df['High'].iloc[i],
                'low': prices_df['Low'].iloc[i],
                'close': prices_df['Close'].iloc[i],
            })

    timeline.sort(key=lambda x: x['timestamp'])

    # FILTRA SOLO 2025
    year_filter = CONFIG['year_filter']
    test_timeline = [e for e in timeline if e['timestamp'].year == year_filter]

    print(f"Timeline totale: {len(timeline):,} eventi")
    print(f"Filtrato {year_filter}: {len(test_timeline):,} eventi")

    if not test_timeline:
        print(f"ERRORE: Nessun dato per {year_filter}")
        return engine

    print(f"Periodo test: {test_timeline[0]['timestamp']} - {test_timeline[-1]['timestamp']}")

    # Calcola ATR per ogni pair
    atr_cache = {}
    for pair, (features, prices_df, _) in pairs_data.items():
        high = prices_df['High'].values
        low = prices_df['Low'].values
        close = prices_df['Close'].values
        tr = np.maximum(high[1:] - low[1:],
                       np.abs(high[1:] - close[:-1]),
                       np.abs(low[1:] - close[:-1]))
        atr = pd.Series(tr).rolling(14).mean().values
        atr_cache[pair] = np.concatenate([[atr[0]], atr])

    # Run backtest
    print("\nEsecuzione backtest...")
    model.eval()

    current_prices = {}
    last_signal = {}
    daily_trades = {}

    for i, event in enumerate(test_timeline):
        pair = event['pair']
        timestamp = event['timestamp']
        current_prices[pair] = event['close']

        engine.check_exits(event['high'], event['low'], event['close'], timestamp)

        features = torch.FloatTensor(event['features']).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        signal = ['SELL', 'HOLD', 'BUY'][pred]

        if signal != 'HOLD' and confidence >= CONFIG['min_confidence']:
            if pair not in engine.positions:
                last_time = engine.last_trade_time.get(pair)
                if last_time is None or (timestamp - last_time).total_seconds() / 3600 >= CONFIG['min_bars_between_trades']:
                    date_key = timestamp.date()
                    if date_key not in daily_trades:
                        daily_trades[date_key] = 0

                    if daily_trades[date_key] < CONFIG['max_trades_per_day']:
                        atr = atr_cache[pair][event['idx']] if event['idx'] < len(atr_cache[pair]) else atr_cache[pair][-1]
                        if atr > 0:
                            if engine.open_position(pair, signal, event['close'], atr, timestamp):
                                daily_trades[date_key] += 1

        engine.update_equity(timestamp)

        if i % 5000 == 0:
            print(f"  Progress: {i:,}/{len(test_timeline):,} | Capital: ${engine.capital:,.2f}")

    print(f"  Progress: {len(test_timeline):,}/{len(test_timeline):,} | Capital: ${engine.capital:,.2f}")
    return engine


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"FOREX AI - BACKTEST {CONFIG['year_filter']}")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "lstm_balanced" / "best_model.pt"
    data_dir = base_dir / "data"

    # Load model
    print("\nCaricamento modello LSTM...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_config' in checkpoint:
        input_size = checkpoint['model_config'].get('input_size', 88)
    else:
        input_size = 88

    print(f"Input size: {input_size}")
    model = LSTMBalanced(input_size=input_size)

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Modello caricato!")

    # Load data
    print("\nCaricamento dati...")
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    pairs_data = {}

    for pair in pairs:
        try:
            features = pd.read_parquet(data_dir / "processed" / f"{pair}_features.parquet")
            prices_df = None
            for tf in ['H1', 'H4', 'D1']:
                raw_file = data_dir / "raw" / f"{pair}_{tf}.parquet"
                if raw_file.exists():
                    prices_df = pd.read_parquet(raw_file)
                    break

            if prices_df is not None:
                min_len = min(len(features), len(prices_df))
                features = features.iloc[:min_len].values
                prices_df = prices_df.iloc[:min_len]
                pairs_data[pair] = (features, prices_df, None)
                print(f"  {pair}: {len(prices_df):,} samples")
        except Exception as e:
            print(f"  {pair}: ERRORE - {e}")

    if not pairs_data:
        print("Nessun dato disponibile!")
        exit(1)

    # Run backtest
    engine = run_backtest(model, pairs_data, device)

    # Results
    stats = engine.get_stats()
    print("\n" + "=" * 60)
    print(f"RISULTATI BACKTEST {CONFIG['year_filter']}")
    print("=" * 60)
    print(f"Capitale iniziale:  ${CONFIG['initial_capital']:,.2f}")
    print(f"Capitale finale:    ${engine.capital:,.2f}")
    print(f"Return totale:      {(engine.capital / CONFIG['initial_capital'] - 1) * 100:+.2f}%")
    print()
    print(f"Trades totali:      {stats.get('n_trades', 0)}")
    print(f"Win rate:           {stats.get('win_rate', 0) * 100:.1f}%")
    print(f"Profit factor:      {stats.get('profit_factor', 0):.2f}")
    print()
    print(f"Max drawdown:       {stats.get('max_drawdown', 0) * 100:.2f}%")
    print(f"Sharpe ratio:       {stats.get('sharpe', 0):.2f}")

    # Save chart
    if engine.equity_curve:
        plt.figure(figsize=(14, 6))
        plt.plot(engine.equity_curve, 'b-', linewidth=1)
        plt.title(f'Equity Curve - Backtest {CONFIG["year_filter"]}')
        plt.xlabel('Trade #')
        plt.ylabel('Capital ($)')
        plt.grid(True, alpha=0.3)

        report_dir = base_dir / "reports" / "backtest"
        report_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(report_dir / f"backtest_{CONFIG['year_filter']}_results.png", dpi=150, bbox_inches='tight')
        print(f"\nGrafico salvato: {report_dir / f'backtest_{CONFIG['year_filter']}_results.png'}")
