"""
Backtesting System - Forex AI Trading
Simula trading realistico con il modello LSTM trainato
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
# CONFIGURAZIONE BACKTEST
# ============================================================================
CONFIG = {
    'initial_capital': 100000,    # Capitale iniziale USD (100k)
    'risk_per_trade': 0.0065,     # 0.65% rischio per trade (ridotto per DD < 15%)
    'spread_pips': 1.5,           # Spread medio
    'slippage_pips': 0.5,         # Slippage
    'commission_per_lot': 7,      # Commissione per lotto standard
    'leverage': 30,               # Leva
    'min_confidence': 0.66,       # Confidenza minima 66% (simula AI Agent)
    'take_profit_atr': 3.0,       # TP in multipli di ATR (3x)
    'stop_loss_atr': 1.5,         # SL in multipli di ATR (ratio 2:1)
    'min_bars_between_trades': 72, # Min 72 ore (3 giorni) tra trades per coppia
    'max_trades_per_day': 2,      # Max 2 trades al giorno totali
    'max_lot_size': 5.0,          # Max lotti per trade (allineato con live trading)
}


# ============================================================================
# MODELLO LSTM (stesso del training - LSTMBalanced)
# ============================================================================
class LSTMBalanced(nn.Module):
    """LSTM con LayerNorm per classificazione bilanciata."""

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
            x = x.unsqueeze(1)  # Add sequence dimension

        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden*2]

        # Attention
        attn_weights = self.attention(lstm_out)  # [batch, seq, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden*2]

        # Classification
        out = self.classifier(context)
        return out


# ============================================================================
# BACKTEST ENGINE
# ============================================================================
class BacktestEngine:
    def __init__(self, config=CONFIG):
        self.config = config
        self.reset()

    def reset(self):
        self.capital = self.config['initial_capital']
        self.equity_curve = [self.capital]
        self.trades = []
        self.positions = []  # Posizioni aperte
        self.daily_returns = []

    def calculate_position_size(self, price, atr):
        """Calcola size posizione basata sul rischio - FIXED SIZING"""
        # Usa capitale INIZIALE per evitare over-leveraging
        # Questo mantiene risk costante e previene esplosione DD
        base_capital = self.config['initial_capital']

        risk_amount = base_capital * self.config['risk_per_trade']
        sl_pips = atr * self.config['stop_loss_atr'] * 10000  # Converti in pips
        if sl_pips < 1:
            sl_pips = 10  # Minimo 10 pips SL

        # Valore pip per lotto standard (circa $10 per major pairs)
        pip_value = 10

        # Lotti - sizing fisso basato su capitale iniziale
        lots = risk_amount / (sl_pips * pip_value)
        max_lots = self.config.get('max_lot_size', 5.0)
        lots = min(lots, max_lots)  # Max lotti configurabile
        lots = max(lots, 0.01)  # Minimo 0.01 lotti

        return round(lots, 2)

    def open_trade(self, direction, price, atr, timestamp, pair):
        """Apre un trade"""
        # Applica spread e slippage
        spread_cost = self.config['spread_pips'] * 0.0001
        slippage = self.config['slippage_pips'] * 0.0001

        if direction == 'BUY':
            entry_price = price * (1 + spread_cost + slippage)
            tp = entry_price + atr * self.config['take_profit_atr']
            sl = entry_price - atr * self.config['stop_loss_atr']
        else:  # SELL
            entry_price = price * (1 - spread_cost - slippage)
            tp = entry_price - atr * self.config['take_profit_atr']
            sl = entry_price + atr * self.config['stop_loss_atr']

        size = self.calculate_position_size(price, atr)

        trade = {
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'size': size,
            'tp': tp,
            'sl': sl,
            'atr': atr,
            'status': 'OPEN'
        }

        self.positions.append(trade)
        return trade

    def check_exits(self, high, low, close, timestamp):
        """Controlla se posizioni devono essere chiuse"""
        closed = []

        for pos in self.positions[:]:
            exit_price = None
            exit_reason = None

            if pos['direction'] == 'BUY':
                if high >= pos['tp']:
                    exit_price = pos['tp']
                    exit_reason = 'TP'
                elif low <= pos['sl']:
                    exit_price = pos['sl']
                    exit_reason = 'SL'
            else:  # SELL
                if low <= pos['tp']:
                    exit_price = pos['tp']
                    exit_reason = 'TP'
                elif high >= pos['sl']:
                    exit_price = pos['sl']
                    exit_reason = 'SL'

            if exit_price:
                # Calcola PnL
                if pos['direction'] == 'BUY':
                    pnl_pips = (exit_price - pos['entry_price']) * 10000
                else:
                    pnl_pips = (pos['entry_price'] - exit_price) * 10000

                # PnL in USD (approssimativo)
                pnl_usd = pnl_pips * pos['size'] * 10  # $10 per pip per lotto

                # Commissione
                commission = self.config['commission_per_lot'] * pos['size']
                pnl_usd -= commission

                # Aggiorna capitale
                self.capital += pnl_usd

                # Salva trade
                pos['exit_price'] = exit_price
                pos['exit_time'] = timestamp
                pos['exit_reason'] = exit_reason
                pos['pnl_pips'] = pnl_pips
                pos['pnl_usd'] = pnl_usd
                pos['status'] = 'CLOSED'

                self.trades.append(pos)
                self.positions.remove(pos)
                closed.append(pos)

        return closed

    def update_equity(self, current_prices):
        """Aggiorna equity curve con mark-to-market"""
        mtm = self.capital

        for pos in self.positions:
            price = current_prices.get(pos['pair'], pos['entry_price'])
            if pos['direction'] == 'BUY':
                unrealized = (price - pos['entry_price']) * 10000 * pos['size'] * 10
            else:
                unrealized = (pos['entry_price'] - price) * 10000 * pos['size'] * 10
            mtm += unrealized

        self.equity_curve.append(mtm)

    def get_metrics(self):
        """Calcola metriche di performance"""
        if not self.trades:
            return {
                'total_return': 0,
                'n_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe': 0,
                'avg_trade': 0
            }

        # Trades
        n_trades = len(self.trades)
        wins = [t for t in self.trades if t['pnl_usd'] > 0]
        losses = [t for t in self.trades if t['pnl_usd'] <= 0]

        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t['pnl_usd'] for t in wins)
        gross_loss = abs(sum(t['pnl_usd'] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Return
        total_return = (self.capital - self.config['initial_capital']) / self.config['initial_capital'] * 100

        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_dd = drawdown.max()

        # Sharpe (semplificato)
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)

        # Avg trade
        avg_trade = sum(t['pnl_usd'] for t in self.trades) / n_trades

        return {
            'total_return': total_return,
            'final_capital': self.capital,
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
    """
    Esegue backtest su tutti i dati

    pairs_data: dict con {pair: (features, prices_df, labels)}
    """
    print("=" * 60)
    print("BACKTEST - FOREX AI TRADING")
    print("=" * 60)

    engine = BacktestEngine()
    all_predictions = []

    # Ordina dati per timestamp (simulazione realistica)
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
                'open': prices_df['open'].iloc[i],
                'high': prices_df['high'].iloc[i],
                'low': prices_df['low'].iloc[i],
                'close': prices_df['close'].iloc[i],
            })

    # Ordina per timestamp
    timeline.sort(key=lambda x: x['timestamp'])
    print(f"Timeline: {len(timeline):,} eventi")

    # Usa TUTTO lo storico (10 anni)
    test_timeline = timeline
    print(f"Test period: {len(test_timeline):,} eventi ({test_timeline[0]['timestamp']} - {test_timeline[-1]['timestamp']})")

    # Calcola ATR per ogni pair (media mobile)
    atr_cache = {}
    for pair, (features, prices_df, _) in pairs_data.items():
        high = prices_df['high'].values
        low = prices_df['low'].values
        close = prices_df['close'].values

        tr = np.maximum(high[1:] - low[1:],
                       np.abs(high[1:] - close[:-1]),
                       np.abs(low[1:] - close[:-1]))
        atr = pd.Series(tr).rolling(14).mean().values
        atr_cache[pair] = np.concatenate([[atr[0]], atr])

    # Run backtest
    print("\nEsecuzione backtest...")
    model.eval()

    current_prices = {}
    last_signal = {}  # Evita over-trading
    daily_trades = {}  # Conta trades per giorno

    for i, event in enumerate(test_timeline):
        pair = event['pair']
        timestamp = event['timestamp']

        # Update prezzi correnti
        current_prices[pair] = event['close']

        # Check exits per posizioni aperte
        engine.check_exits(event['high'], event['low'], event['close'], timestamp)

        # Predizione modello
        features = torch.FloatTensor(event['features']).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        # Signal: 0=SELL, 1=HOLD, 2=BUY
        signal = ['SELL', 'HOLD', 'BUY'][pred]

        # Entra solo se:
        # 1. Confidenza sufficiente
        # 2. Non abbiamo già posizione su questa coppia
        # 3. Non abbiamo appena tradato questa coppia

        has_position = any(p['pair'] == pair for p in engine.positions)
        recently_traded = last_signal.get(pair, 0) > i - CONFIG['min_bars_between_trades']

        # Filtro max trades per giorno
        current_day = timestamp.date() if hasattr(timestamp, 'date') else str(timestamp)[:10]
        if current_day not in daily_trades:
            daily_trades[current_day] = 0
        max_trades_reached = daily_trades[current_day] >= CONFIG['max_trades_per_day']

        # FILTRO MACRO: Solo sessioni attive (Londra 8-16 UTC, NY 13-21 UTC)
        hour = timestamp.hour if hasattr(timestamp, 'hour') else 12
        in_active_session = 8 <= hour <= 20  # Sessioni Londra/NY overlap

        # FILTRO TREND: Verifica trend su periodo più lungo (semplice: close > media 50 periodi)
        # Questo viene calcolato dalla feature già presente nei dati

        if (signal != 'HOLD' and
            confidence >= CONFIG['min_confidence'] and
            not has_position and
            not recently_traded and
            not max_trades_reached and
            in_active_session):

            # Recupera ATR
            idx = min(event['idx'], len(atr_cache[pair]) - 1)
            atr = atr_cache[pair][idx]

            if atr > 0:
                engine.open_trade(signal, event['close'], atr, timestamp, pair)
                last_signal[pair] = i
                daily_trades[current_day] += 1

        # Update equity ogni 100 eventi
        if i % 100 == 0:
            engine.update_equity(current_prices)

        # Progress
        if i % 10000 == 0:
            print(f"  Progress: {i:,}/{len(test_timeline):,} | Capital: ${engine.capital:,.2f}")

    # Chiudi posizioni rimaste
    for pos in engine.positions[:]:
        price = current_prices.get(pos['pair'], pos['entry_price'])
        if pos['direction'] == 'BUY':
            pnl_pips = (price - pos['entry_price']) * 10000
        else:
            pnl_pips = (pos['entry_price'] - price) * 10000

        pnl_usd = pnl_pips * pos['size'] * 10
        engine.capital += pnl_usd

        pos['exit_price'] = price
        pos['exit_time'] = test_timeline[-1]['timestamp']
        pos['exit_reason'] = 'END'
        pos['pnl_pips'] = pnl_pips
        pos['pnl_usd'] = pnl_usd
        pos['status'] = 'CLOSED'
        engine.trades.append(pos)

    engine.positions = []
    engine.update_equity(current_prices)

    return engine


def plot_results(engine, output_path):
    """Genera grafici risultati"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Backtest Results - Forex AI Trading', fontsize=14, fontweight='bold')

    # 1. Equity curve
    ax1 = axes[0, 0]
    equity = engine.equity_curve
    ax1.plot(equity, color='cyan', linewidth=1)
    ax1.axhline(y=CONFIG['initial_capital'], color='white', linestyle='--', alpha=0.5)
    ax1.fill_between(range(len(equity)), CONFIG['initial_capital'], equity,
                     where=np.array(equity) >= CONFIG['initial_capital'],
                     color='green', alpha=0.3)
    ax1.fill_between(range(len(equity)), CONFIG['initial_capital'], equity,
                     where=np.array(equity) < CONFIG['initial_capital'],
                     color='red', alpha=0.3)
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Capital ($)')
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[0, 1]
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / peak * 100
    ax2.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.5)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    # 3. Trade distribution
    ax3 = axes[1, 0]
    if engine.trades:
        pnls = [t['pnl_usd'] for t in engine.trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='white', linewidth=0.5)
        ax3.set_title(f'Trade PnL Distribution (n={len(pnls)})')
        ax3.set_ylabel('PnL ($)')
    ax3.grid(True, alpha=0.3)

    # 4. Stats
    ax4 = axes[1, 1]
    ax4.axis('off')

    metrics = engine.get_metrics()

    stats_text = f"""
    PERFORMANCE SUMMARY
    {'='*40}

    Initial Capital:    ${CONFIG['initial_capital']:,.2f}
    Final Capital:      ${metrics['final_capital']:,.2f}
    Total Return:       {metrics['total_return']:+.2f}%

    Total Trades:       {metrics['n_trades']}
    Win Rate:           {metrics['win_rate']:.1f}%
    Profit Factor:      {metrics['profit_factor']:.2f}

    Gross Profit:       ${metrics['gross_profit']:,.2f}
    Gross Loss:         ${metrics['gross_loss']:,.2f}
    Avg Trade:          ${metrics['avg_trade']:,.2f}

    Max Drawdown:       {metrics['max_drawdown']:.2f}%
    Sharpe Ratio:       {metrics['sharpe']:.2f}
    """

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print(f"\nGrafico salvato: {output_path}")


def main():
    print("=" * 60)
    print("FOREX AI - BACKTEST SYSTEM")
    print("=" * 60)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    model_path = base_dir / "models" / "lstm_balanced" / "best_model.pt"
    output_dir = base_dir / "reports" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carica modello
    print("\nCaricamento modello LSTM...")
    if not model_path.exists():
        print(f"ERRORE: Modello non trovato: {model_path}")
        return

    # Carica checkpoint per determinare input_size
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Determina input size dal modello salvato
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Input size dalla prima layer LSTM
    input_size = state_dict['lstm.weight_ih_l0'].shape[1]
    print(f"Input size dal modello: {input_size}")

    model = LSTMBalanced(input_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Modello caricato!")

    # Carica dati
    print("\nCaricamento dati...")
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    pairs_data = {}

    for pair in pairs:
        try:
            # Features
            features = pd.read_parquet(data_dir / "processed" / f"{pair}_features.parquet")
            features = features.values.astype(np.float32)
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
            features = np.clip(features, -10, 10)

            # Adatta al numero di features del modello
            if features.shape[1] > input_size:
                features = features[:, :input_size]
            elif features.shape[1] < input_size:
                # Padding con zeri
                padding = np.zeros((features.shape[0], input_size - features.shape[1]), dtype=np.float32)
                features = np.concatenate([features, padding], axis=1)

            # Prezzi
            for tf in ['H1', 'H4', 'D1']:
                raw_file = data_dir / "raw" / f"{pair}_{tf}.parquet"
                if raw_file.exists():
                    prices_df = pd.read_parquet(raw_file)
                    prices_df.columns = [c.lower() for c in prices_df.columns]
                    break

            # Labels (opzionale)
            labels_file = data_dir / "processed" / f"{pair}_labels.parquet"
            labels = pd.read_parquet(labels_file) if labels_file.exists() else None

            # Allinea
            min_len = min(len(features), len(prices_df))
            features = features[:min_len]
            prices_df = prices_df.iloc[:min_len]

            pairs_data[pair] = (features, prices_df, labels)
            print(f"  {pair}: {len(features):,} samples")

        except Exception as e:
            print(f"  {pair}: SKIP - {e}")

    if not pairs_data:
        print("ERRORE: Nessun dato caricato!")
        return

    # Run backtest
    engine = run_backtest(model, pairs_data, device)

    # Risultati
    metrics = engine.get_metrics()

    print("\n" + "=" * 60)
    print("RISULTATI BACKTEST")
    print("=" * 60)
    print(f"Capitale iniziale:  ${CONFIG['initial_capital']:,.2f}")
    print(f"Capitale finale:    ${metrics['final_capital']:,.2f}")
    print(f"Return totale:      {metrics['total_return']:+.2f}%")
    print(f"")
    print(f"Trades totali:      {metrics['n_trades']}")
    print(f"Win rate:           {metrics['win_rate']:.1f}%")
    print(f"Profit factor:      {metrics['profit_factor']:.2f}")
    print(f"")
    print(f"Max drawdown:       {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe ratio:       {metrics['sharpe']:.2f}")

    # Salva grafici
    plot_results(engine, output_dir / "backtest_results.png")

    # Salva report JSON
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'metrics': metrics,
        'n_pairs': len(pairs_data)
    }

    with open(output_dir / "backtest_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport salvato: {output_dir / 'backtest_report.json'}")


if __name__ == "__main__":
    main()
