"""
================================================================================
LSTM TRADER - NexNow LTD
================================================================================
Trading basato direttamente sulle predizioni LSTM.
Versione semplificata per testare l'efficacia del modello.
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


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2, num_classes=3):
        super().__init__()
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
        return self.classifier(context)


class LSTMTrader:
    def __init__(self, model_path: str):
        # Carica modello
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.input_size = state_dict['lstm.weight_ih_l0'].shape[1]

        self.model = LSTMModel(input_size=self.input_size).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"LSTM input_size: {self.input_size}")

        # Config
        self.initial_balance = 10000
        self.spread_pips = 1.5
        self.commission = 7.0
        self.position_size = 0.1

    def predict(self, features: np.ndarray) -> tuple:
        """Predice direzione e confidenza"""
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            # 0=SELL, 1=HOLD, 2=BUY
            pred = np.argmax(probs)
            conf = probs[pred]

            direction = {0: -1, 1: 0, 2: 1}.get(pred, 0)
            return direction, conf, probs

    def backtest(self, features_df: pd.DataFrame, raw_df: pd.DataFrame, pair: str):
        """Backtest semplice"""
        # Allinea
        common_idx = features_df.index.intersection(raw_df.index)
        features_df = features_df.loc[common_idx]
        raw_df = raw_df.loc[common_idx]

        # Prepara features
        feature_cols = [c for c in features_df.columns if features_df[c].dtype in ['float64', 'float32', 'int64']]
        num_features = len(feature_cols)
        need_padding = max(0, self.input_size - num_features)

        print(f"\n{pair}: {len(features_df):,} bars, {num_features} features (pad: {need_padding})")

        # Stats
        balance = self.initial_balance
        position = 0
        entry_price = 0
        trades = []

        # Conta predizioni
        pred_counts = {-1: 0, 0: 0, 1: 0}

        for i, (idx, row) in enumerate(tqdm(features_df.iterrows(), total=len(features_df), desc=pair, leave=False)):
            if i < 100:
                continue

            price = raw_df.loc[idx, 'Close']

            # Features
            feats = row[feature_cols].values.astype(np.float32)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            if need_padding > 0:
                feats = np.concatenate([feats, np.zeros(need_padding, dtype=np.float32)])

            # Predict
            direction, conf, probs = self.predict(feats)
            pred_counts[direction] += 1

            # Trading logic: entra con confidenza > 0.4
            if position == 0 and conf > 0.4 and direction != 0:
                position = direction
                entry_price = price

            # Esci dopo 24 ore o se direzione cambia
            elif position != 0:
                holding_time = i - trades[-1]['entry_bar'] if trades else 0

                # Exit conditions
                should_exit = False
                if direction == -position and conf > 0.5:  # Segnale opposto
                    should_exit = True
                elif holding_time >= 24:  # Timeout
                    should_exit = True

                if should_exit:
                    pip_value = 0.0001 if 'JPY' not in pair else 0.01

                    if position == 1:
                        pips = (price - entry_price) / pip_value
                    else:
                        pips = (entry_price - price) / pip_value

                    pips -= self.spread_pips
                    pnl = pips * self.position_size * 10 - self.commission * self.position_size
                    balance += pnl

                    trades.append({
                        'pair': pair,
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'entry_bar': i - holding_time,
                        'exit_bar': i,
                        'pips': pips,
                        'pnl': pnl,
                        'confidence': conf,
                    })

                    position = 0
                    entry_price = 0

        print(f"  Predictions: BUY={pred_counts[1]}, HOLD={pred_counts[0]}, SELL={pred_counts[-1]}")

        if not trades:
            return {'pair': pair, 'error': 'No trades'}

        # Metrics
        winning = [t for t in trades if t['pnl'] > 0]
        losing = [t for t in trades if t['pnl'] <= 0]
        gross_profit = sum(t['pnl'] for t in winning)
        gross_loss = abs(sum(t['pnl'] for t in losing))

        return {
            'pair': pair,
            'total_trades': len(trades),
            'winning': len(winning),
            'losing': len(losing),
            'win_rate': len(winning) / len(trades) * 100,
            'net_profit': sum(t['pnl'] for t in trades),
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'final_balance': balance,
            'return_pct': (balance - self.initial_balance) / self.initial_balance * 100,
            'avg_pips': np.mean([t['pips'] for t in trades]),
        }


def main():
    print("=" * 60)
    print("LSTM TRADER")
    print("=" * 60)

    model_path = Path('models/lstm/best_model.pt')
    if not model_path.exists():
        model_path = Path('models/lstm_model.pt')

    trader = LSTMTrader(str(model_path))

    pairs = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY"]
    results = []

    for pair in pairs:
        features_path = Path(f'./data/processed/{pair}_features.parquet')
        raw_path = Path(f'./data/raw/{pair}_H1.parquet')

        if not features_path.exists() or not raw_path.exists():
            continue

        features_df = pd.read_parquet(features_path)
        raw_df = pd.read_parquet(raw_path)

        # Ultimi 20%
        test_start = int(len(features_df) * 0.8)
        features_test = features_df.iloc[test_start:]
        raw_test = raw_df.iloc[test_start:]

        metrics = trader.backtest(features_test, raw_test, pair)

        if 'error' in metrics:
            print(f"{pair}: {metrics['error']}")
            continue

        results.append(metrics)

        print(f"\n{pair}:")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Return: {metrics['return_pct']:.2f}%")

    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)

    if results:
        print(f"Pairs: {len(results)}")
        print(f"Trades: {sum(r['total_trades'] for r in results)}")
        print(f"Profit: ${sum(r['net_profit'] for r in results):,.2f}")
        print(f"Win Rate: {np.mean([r['win_rate'] for r in results]):.1f}%")


if __name__ == "__main__":
    main()
