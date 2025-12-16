"""
================================================================================
BACKTEST FR AGENT - NexNow LTD
================================================================================
Backtest dell'agente Forza Relativa con metriche realistiche.
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# CARICA MODELLO (stessa architettura di train_fr_agent.py)
# ============================================================================

class FRActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_size: int = 256, n_layers: int = 3):
        super().__init__()

        layers = [nn.Linear(obs_dim, hidden_size), nn.ReLU(), nn.LayerNorm(hidden_size)]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1),
            ])

        self.encoder = nn.Sequential(*layers)

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, obs):
        return self.encoder(obs)

    def get_action(self, obs):
        features = self.forward(obs)
        logits = self.actor(features)
        probs = torch.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)


def extract_fr_features(features_df: pd.DataFrame) -> tuple:
    """Estrae le feature FR dal DataFrame"""
    fr_columns = [
        'fr_spread', 'fr_spread_abs', 'ema_distance_atr', 'atr_percentile',
        'excess_direction', 'dist_to_adr_high_pct', 'dist_to_adr_low_pct',
        'seasonal_correlation', 'seasonal_direction', 'rsi_14', 'macd_hist',
        'bb_position', 'volatility_ratio', 'hour', 'day_of_week',
        'is_london_session', 'is_ny_session', 'is_overlap',
    ]

    strength_cols = [c for c in features_df.columns if c.startswith('strength_')]
    fr_columns.extend(strength_cols[:2])

    available_cols = [c for c in fr_columns if c in features_df.columns]
    data = features_df[available_cols].values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data.astype(np.float32), available_cols


# ============================================================================
# BACKTESTER
# ============================================================================

class FRBacktester:
    def __init__(self, model_path: str, config: dict = None):
        # Carica modello
        checkpoint = torch.load(model_path, map_location=device)
        saved_config = checkpoint.get('config', {})

        self.config = config or {
            'initial_balance': 10000,
            'leverage': 30,
            'spread_pips': 1.5,
            'commission_per_lot': 7.0,
            'max_position_size': 0.1,
            'hidden_size': saved_config.get('hidden_size', 256),
            'n_layers': saved_config.get('n_layers', 3),
        }

        # Obs dim = features + position state (3)
        feature_names = checkpoint.get('feature_names', [])
        self.obs_dim = len(feature_names) + 3 if feature_names else 23
        self.action_dim = 4

        self.model = FRActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_size=self.config['hidden_size'],
            n_layers=self.config['n_layers'],
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Modello caricato: {model_path}")
        print(f"Obs dim: {self.obs_dim}, Action dim: {self.action_dim}")

    def run_backtest(self, pair: str, features: np.ndarray, prices: np.ndarray,
                     dates: pd.DatetimeIndex, feature_names: list):
        """Esegue backtest su un singolo pair"""

        results = {
            'pair': pair,
            'trades': [],
            'equity_curve': [],
            'dates': [],
        }

        # Stato
        balance = self.config['initial_balance']
        position = 0  # -1, 0, +1
        position_price = 0
        position_size = 0
        last_trade_step = 0

        # Indici feature
        fr_spread_idx = feature_names.index('fr_spread') if 'fr_spread' in feature_names else -1

        print(f"\nBacktest {pair}...")

        for step in tqdm(range(100, len(features) - 1), desc=pair, leave=False):
            current_price = prices[step]
            current_date = dates[step]

            # Costruisci osservazione
            fr_features = features[step]
            position_state = np.array([
                position,
                (current_price - position_price) / position_price if position != 0 else 0,
                min((step - last_trade_step) / 100, 1.0),
            ], dtype=np.float32)

            obs = np.concatenate([fr_features, position_state])
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

            # Predici azione
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = self.model.get_action(obs_t).item()

            # Esegui azione
            if action == 1 and position == 0:  # BUY
                position = 1
                position_price = current_price
                position_size = self.config['max_position_size']
                last_trade_step = step

            elif action == 2 and position == 0:  # SELL
                position = -1
                position_price = current_price
                position_size = self.config['max_position_size']
                last_trade_step = step

            elif action == 3 and position != 0:  # CLOSE
                # Calcola PnL
                pip_value = 0.0001 if 'JPY' not in pair else 0.01
                spread_cost = self.config['spread_pips'] * pip_value * position_size * 100000
                commission = self.config['commission_per_lot'] * position_size

                if position == 1:
                    pnl = (current_price - position_price) * position_size * 100000
                else:
                    pnl = (position_price - current_price) * position_size * 100000

                pnl -= (spread_cost + commission)
                balance += pnl

                # Registra trade
                fr_spread = features[last_trade_step, fr_spread_idx] if fr_spread_idx >= 0 else 0
                results['trades'].append({
                    'entry_date': dates[last_trade_step],
                    'exit_date': current_date,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': position_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'fr_spread': fr_spread,
                    'holding_bars': step - last_trade_step,
                })

                position = 0
                position_price = 0
                position_size = 0

            # Registra equity
            if step % 24 == 0:  # Ogni giorno
                results['equity_curve'].append(balance)
                results['dates'].append(current_date)

        # Chiudi posizione finale
        if position != 0:
            current_price = prices[-1]
            pip_value = 0.0001 if 'JPY' not in pair else 0.01
            spread_cost = self.config['spread_pips'] * pip_value * position_size * 100000
            commission = self.config['commission_per_lot'] * position_size

            if position == 1:
                pnl = (current_price - position_price) * position_size * 100000
            else:
                pnl = (position_price - current_price) * position_size * 100000

            pnl -= (spread_cost + commission)
            balance += pnl

        results['final_balance'] = balance
        results['total_return'] = (balance - self.config['initial_balance']) / self.config['initial_balance'] * 100

        return results

    def calculate_metrics(self, results: dict) -> dict:
        """Calcola metriche di performance"""
        trades = results['trades']

        if len(trades) == 0:
            return {'error': 'Nessun trade eseguito'}

        # PnL
        pnls = [t['pnl'] for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        metrics = {
            'pair': results['pair'],
            'total_trades': len(trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(trades) * 100 if trades else 0,
            'total_pnl': sum(pnls),
            'avg_win': np.mean(winning) if winning else 0,
            'avg_loss': np.mean(losing) if losing else 0,
            'profit_factor': abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else float('inf'),
            'max_drawdown': self._calculate_max_drawdown(results['equity_curve']),
            'final_balance': results['final_balance'],
            'total_return_pct': results['total_return'],
        }

        # Analisi FR Spread
        fr_spreads = [t['fr_spread'] for t in trades]
        long_with_positive_fr = sum(1 for t in trades if t['direction'] == 'LONG' and t['fr_spread'] > 15)
        short_with_negative_fr = sum(1 for t in trades if t['direction'] == 'SHORT' and t['fr_spread'] < -15)

        metrics['trades_with_fr_signal'] = long_with_positive_fr + short_with_negative_fr
        metrics['fr_signal_rate'] = (long_with_positive_fr + short_with_negative_fr) / len(trades) * 100 if trades else 0

        # Win rate quando FR > 15 o < -15
        aligned_trades = [t for t in trades if
                         (t['direction'] == 'LONG' and t['fr_spread'] > 15) or
                         (t['direction'] == 'SHORT' and t['fr_spread'] < -15)]
        if aligned_trades:
            aligned_wins = sum(1 for t in aligned_trades if t['pnl'] > 0)
            metrics['win_rate_with_fr_signal'] = aligned_wins / len(aligned_trades) * 100
        else:
            metrics['win_rate_with_fr_signal'] = 0

        return metrics

    def _calculate_max_drawdown(self, equity_curve: list) -> float:
        if not equity_curve:
            return 0

        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd


def main():
    print("=" * 60)
    print("BACKTEST FR AGENT")
    print("=" * 60)

    # Carica modello
    model_path = Path('models/fr_agent/fr_agent_best.pt')
    if not model_path.exists():
        model_path = Path('models/fr_agent/fr_agent_final.pt')

    if not model_path.exists():
        print("Errore: Modello non trovato!")
        return

    backtester = FRBacktester(str(model_path))

    # Carica dati di test (ultimi 20% dei dati)
    processed_dir = Path('./data/processed')
    raw_dir = Path('./data/raw')

    pairs = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY"]
    all_results = []

    for pair in pairs:
        features_path = processed_dir / f"{pair}_features.parquet"
        raw_path = raw_dir / f"{pair}_H1.parquet"

        if not features_path.exists() or not raw_path.exists():
            continue

        features_df = pd.read_parquet(features_path)
        raw_df = pd.read_parquet(raw_path)

        # Estrai feature FR
        fr_features, feature_names = extract_fr_features(features_df)

        # Allinea
        common_idx = features_df.index.intersection(raw_df.index)
        features_df = features_df.loc[common_idx]
        raw_df = raw_df.loc[common_idx]
        fr_features = fr_features[-len(common_idx):]

        # Split: ultimi 20% per test
        test_start = int(len(fr_features) * 0.8)
        test_features = fr_features[test_start:]
        test_prices = raw_df['Close'].values[test_start:]
        test_dates = features_df.index[test_start:]

        # Esegui backtest
        results = backtester.run_backtest(
            pair=pair,
            features=test_features,
            prices=test_prices,
            dates=test_dates,
            feature_names=feature_names,
        )

        metrics = backtester.calculate_metrics(results)

        if 'error' in metrics:
            print(f"\n{pair}: {metrics['error']}")
            continue

        all_results.append(metrics)

        # Stampa risultati
        print(f"\n{pair}:")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"  FR Signal Rate: {metrics['fr_signal_rate']:.1f}%")
        print(f"  Win Rate con FR: {metrics['win_rate_with_fr_signal']:.1f}%")

    # Riepilogo
    print("\n" + "=" * 60)
    print("RIEPILOGO TOTALE")
    print("=" * 60)

    total_trades = sum(r['total_trades'] for r in all_results)
    total_pnl = sum(r['total_pnl'] for r in all_results)
    avg_win_rate = np.mean([r['win_rate'] for r in all_results if r['total_trades'] > 0])
    avg_return = np.mean([r['total_return_pct'] for r in all_results])

    print(f"Pairs testati: {len(all_results)}")
    print(f"Trades totali: {total_trades}")
    print(f"PnL totale: ${total_pnl:,.2f}")
    print(f"Win Rate medio: {avg_win_rate:.1f}%")
    print(f"Return medio: {avg_return:.2f}%")

    # Salva risultati
    output_dir = Path('models/fr_agent')
    with open(output_dir / 'backtest_results.json', 'w') as f:
        json.dump({
            'timestamp': str(datetime.now()),
            'pairs': [r for r in all_results],
            'summary': {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'avg_win_rate': avg_win_rate,
                'avg_return': avg_return,
            }
        }, f, indent=2, default=str)

    print(f"\nRisultati salvati in: {output_dir / 'backtest_results.json'}")


if __name__ == "__main__":
    main()
