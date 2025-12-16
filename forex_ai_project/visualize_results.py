"""
Visualizzazione risultati trading con regimi di mercato e momentum
"""

import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurazione
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "reports" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stile grafici
plt.style.use('dark_background')


def detect_regimes(prices, window=50):
    """
    Rileva regimi di mercato:
    - BULL: trend rialzista
    - BEAR: trend ribassista
    - RANGE: laterale
    """
    returns = prices.pct_change()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    regimes = pd.Series('RANGE', index=prices.index)

    # Bull: media positiva e significativa
    bull_mask = rolling_mean > rolling_std * 0.3
    regimes[bull_mask] = 'BULL'

    # Bear: media negativa e significativa
    bear_mask = rolling_mean < -rolling_std * 0.3
    regimes[bear_mask] = 'BEAR'

    return regimes


def calculate_momentum(prices, fast=12, slow=26):
    """Calcola momentum (MACD-style)"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    momentum = ema_fast - ema_slow
    signal = momentum.ewm(span=9).mean()
    histogram = momentum - signal
    return momentum, signal, histogram


def simulate_trading(prices, labels, initial_capital=10000):
    """
    Simula trading basato sui label
    labels: 0=SELL, 1=HOLD, 2=BUY
    """
    capital = initial_capital
    position = 0  # 0=flat, 1=long, -1=short
    equity = [capital]
    entry_price = 0
    trades = []

    for i in range(1, len(prices)):
        price = prices.iloc[i]
        label = labels.iloc[i] if i < len(labels) else 1

        # Chiudi posizione se segnale opposto
        if position == 1 and label == 0:  # Long -> Sell
            pnl = (price - entry_price) / entry_price * capital * 0.5  # 50% del capitale per trade
            capital += pnl
            trades.append({'type': 'CLOSE_LONG', 'pnl': pnl})
            position = 0
        elif position == -1 and label == 2:  # Short -> Buy
            pnl = (entry_price - price) / entry_price * capital * 0.5
            capital += pnl
            trades.append({'type': 'CLOSE_SHORT', 'pnl': pnl})
            position = 0

        # Apri nuova posizione
        if position == 0:
            if label == 2:  # BUY
                position = 1
                entry_price = price
                trades.append({'type': 'OPEN_LONG', 'pnl': 0})
            elif label == 0:  # SELL
                position = -1
                entry_price = price
                trades.append({'type': 'OPEN_SHORT', 'pnl': 0})

        # Calcola equity mark-to-market
        if position == 1:
            mtm = capital + (price - entry_price) / entry_price * capital * 0.5
        elif position == -1:
            mtm = capital + (entry_price - price) / entry_price * capital * 0.5
        else:
            mtm = capital

        equity.append(mtm)

    return pd.Series(equity, index=prices.index[:len(equity)]), trades


def plot_pair_analysis(raw_df, features_df, labels_df, pair_name, output_path):
    """
    Crea grafico completo per una coppia:
    - Prezzo con segnali BUY/SELL
    - Regimi di mercato
    - Momentum histogram
    - Equity curve simulata
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1, 2]})
    fig.suptitle(f'{pair_name} - Analisi Trading AI', fontsize=16, fontweight='bold', color='white')

    # Usa ultimi N punti
    n_points = min(len(raw_df), 2000)
    raw_df = raw_df.tail(n_points).copy()

    prices = raw_df['close']
    dates = raw_df.index

    # Allinea labels se disponibili
    if labels_df is not None and len(labels_df) > 0:
        # Prendi ultimi label allineati con i dati
        # Colonna può essere 'label' o 'optimal_action'
        label_col = 'optimal_action' if 'optimal_action' in labels_df.columns else 'label'
        n_labels = min(len(labels_df), n_points)
        labels = labels_df[label_col].tail(n_labels).reset_index(drop=True)
        labels.index = dates[-len(labels):]
    else:
        labels = None

    # Rileva regimi
    regimes = detect_regimes(prices)

    # Calcola momentum
    momentum, signal_line, histogram = calculate_momentum(prices)

    # === GRAFICO 1: Prezzo + Segnali + Regimi ===
    ax1 = axes[0]

    # Sfondo regimi
    for i in range(len(regimes) - 1):
        if i >= len(dates) - 1:
            break
        if regimes.iloc[i] == 'BULL':
            ax1.axvspan(dates[i], dates[i+1], alpha=0.15, color='green')
        elif regimes.iloc[i] == 'BEAR':
            ax1.axvspan(dates[i], dates[i+1], alpha=0.15, color='red')

    # Prezzo
    ax1.plot(dates, prices, color='#00D4FF', linewidth=1, label='Price')

    # Segnali BUY/SELL dai labels
    if labels is not None:
        buy_mask = labels == 2
        sell_mask = labels == 0

        label_dates = labels.index
        label_prices = prices.loc[label_dates]

        if buy_mask.any():
            ax1.scatter(label_dates[buy_mask], label_prices[buy_mask],
                       color='#00FF88', marker='^', s=30, label='BUY', alpha=0.7, zorder=5)

        if sell_mask.any():
            ax1.scatter(label_dates[sell_mask], label_prices[sell_mask],
                       color='#FF4444', marker='v', s=30, label='SELL', alpha=0.7, zorder=5)

    ax1.set_ylabel('Price', color='white')
    ax1.legend(loc='upper left')
    ax1.set_title('Prezzo con Segnali e Regimi di Mercato (verde=BULL, rosso=BEAR)', color='white')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white')

    # === GRAFICO 2: Distribuzione Regimi ===
    ax2 = axes[1]
    regime_values = regimes.map({'BULL': 1, 'BEAR': -1, 'RANGE': 0})
    ax2.fill_between(dates, regime_values, 0, where=regime_values > 0, color='green', alpha=0.5, label='BULL')
    ax2.fill_between(dates, regime_values, 0, where=regime_values < 0, color='red', alpha=0.5, label='BEAR')
    ax2.axhline(y=0, color='white', linewidth=0.5)
    ax2.set_ylabel('Regime', color='white')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['BEAR', 'RANGE', 'BULL'])
    ax2.legend(loc='upper left')
    ax2.set_title('Regime di Mercato', color='white')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')

    # === GRAFICO 3: Momentum ===
    ax3 = axes[2]
    colors = ['green' if h >= 0 else 'red' for h in histogram.fillna(0)]
    ax3.bar(dates, histogram.fillna(0), color=colors, alpha=0.7, width=0.8)
    ax3.plot(dates, momentum, color='cyan', linewidth=1, label='Momentum')
    ax3.plot(dates, signal_line, color='orange', linewidth=1, label='Signal')
    ax3.axhline(y=0, color='white', linewidth=0.5)
    ax3.set_ylabel('Momentum', color='white')
    ax3.legend(loc='upper left')
    ax3.set_title('Momentum (MACD-style)', color='white')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(colors='white')

    # === GRAFICO 4: Equity Curve ===
    ax4 = axes[3]
    total_return = 0

    if labels is not None and len(labels) > 10:
        # Simula trading
        sim_prices = prices.loc[labels.index]
        equity, trades = simulate_trading(sim_prices, labels)

        # Colora equity in base a performance
        equity_color = 'green' if equity.iloc[-1] > equity.iloc[0] else 'red'
        ax4.fill_between(equity.index, equity, equity.iloc[0], alpha=0.3, color=equity_color)
        ax4.plot(equity.index, equity, color=equity_color, linewidth=2)

        # Statistiche
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        max_equity = equity.cummax()
        drawdown = (max_equity - equity) / max_equity * 100
        max_dd = drawdown.max()
        n_trades = len([t for t in trades if 'OPEN' in t['type']])

        stats_text = f'Return: {total_return:+.2f}% | Max DD: {max_dd:.2f}% | Trades: {n_trades}'
        ax4.text(0.02, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for simulation', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14, color='white')

    ax4.set_ylabel('Equity ($)', color='white')
    ax4.set_xlabel('Date', color='white')
    ax4.set_title('Simulazione Trading (capitale iniziale $10,000)', color='white')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=10000, color='white', linewidth=0.5, linestyle='--')
    ax4.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    return total_return


def create_summary_dashboard(results, output_path):
    """Crea dashboard riepilogativo di tutte le coppie"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dashboard Riepilogativo Trading AI', fontsize=18, fontweight='bold', color='white')

    pairs = list(results.keys())
    returns = [results[p]['return'] for p in pairs]

    # 1. Returns per coppia
    ax1 = axes[0, 0]
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax1.barh(pairs, returns, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='white', linewidth=1)
    ax1.set_xlabel('Return %', color='white')
    ax1.set_title('Performance per Coppia', color='white')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white')

    # 2. Distribuzione returns
    ax2 = axes[0, 1]
    ax2.hist(returns, bins=15, color='cyan', alpha=0.7, edgecolor='white')
    mean_ret = np.mean(returns)
    ax2.axvline(x=mean_ret, color='yellow', linewidth=2, linestyle='--', label=f'Mean: {mean_ret:.2f}%')
    ax2.set_xlabel('Return %', color='white')
    ax2.set_ylabel('Frequency', color='white')
    ax2.set_title('Distribuzione Returns', color='white')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')

    # 3. Top 5 / Bottom 5
    ax3 = axes[1, 0]
    sorted_pairs = sorted(results.items(), key=lambda x: x[1]['return'], reverse=True)
    top5 = sorted_pairs[:5]
    bottom5 = sorted_pairs[-5:]

    labels = [p[0] for p in top5 + bottom5]
    values = [p[1]['return'] for p in top5 + bottom5]
    colors = ['green'] * 5 + ['red'] * 5

    ax3.barh(labels, values, color=colors, alpha=0.7)
    ax3.axvline(x=0, color='white', linewidth=1)
    ax3.set_xlabel('Return %', color='white')
    ax3.set_title('Top 5 e Bottom 5 Coppie', color='white')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(colors='white')

    # 4. Statistiche generali
    ax4 = axes[1, 1]
    ax4.axis('off')

    profitable = sum(1 for r in returns if r > 0)
    total_return = sum(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = avg_return / std_return if std_return > 0 else 0

    stats = f"""
    STATISTICHE GENERALI
    {'='*40}

    Coppie analizzate:     {len(pairs)}
    Coppie profittevoli:   {profitable} ({profitable/len(pairs)*100:.1f}%)
    Coppie in perdita:     {len(pairs)-profitable} ({(len(pairs)-profitable)/len(pairs)*100:.1f}%)

    Return totale:         {total_return:+.2f}%
    Return medio:          {avg_return:+.2f}%
    Deviazione std:        {std_return:.2f}%
    Return max:            {max(returns):+.2f}%
    Return min:            {min(returns):+.2f}%

    Sharpe Ratio:          {sharpe:.2f}
    """

    ax4.text(0.1, 0.9, stats, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace', color='white',
            bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()


def main():
    print("=" * 60)
    print("FOREX AI - VISUALIZZAZIONE RISULTATI")
    print("=" * 60)

    # Lista coppie uniche
    feature_files = list(DATA_DIR.glob("*_features.parquet"))
    pairs = sorted(set(f.stem.replace('_features', '') for f in feature_files))

    print(f"\n[1/3] Trovate {len(pairs)} coppie")

    results = {}

    print("\n[2/3] Generazione grafici per coppia...")

    for i, pair in enumerate(pairs):
        print(f"   [{i+1}/{len(pairs)}] {pair}...", end=" ", flush=True)

        try:
            # Carica dati raw (cerca il timeframe H1 per default)
            raw_file = RAW_DIR / f"{pair}_H1.parquet"
            if not raw_file.exists():
                # Prova altri timeframe
                for tf in ['H4', 'D1']:
                    raw_file = RAW_DIR / f"{pair}_{tf}.parquet"
                    if raw_file.exists():
                        break

            if not raw_file.exists():
                print("SKIP (no raw data)")
                continue

            raw_df = pd.read_parquet(raw_file)
            raw_df.columns = [c.lower() for c in raw_df.columns]
            if raw_df.index.name != 'time' and 'time' in raw_df.columns:
                raw_df.set_index('time', inplace=True)

            # Carica features e labels
            features_file = DATA_DIR / f"{pair}_features.parquet"
            labels_file = DATA_DIR / f"{pair}_labels.parquet"

            features_df = pd.read_parquet(features_file) if features_file.exists() else None
            labels_df = pd.read_parquet(labels_file) if labels_file.exists() else None

            # Genera grafico
            output_path = OUTPUT_DIR / f"{pair}_analysis.png"
            ret = plot_pair_analysis(raw_df, features_df, labels_df, pair, output_path)

            results[pair] = {'return': ret, 'samples': len(raw_df)}
            print(f"OK (Return: {ret:+.2f}%)")

        except Exception as e:
            print(f"ERRORE: {e}")
            continue

    # Dashboard riepilogativo
    if results:
        print("\n[3/3] Generazione dashboard...")
        summary_path = OUTPUT_DIR / "00_DASHBOARD_SUMMARY.png"
        create_summary_dashboard(results, summary_path)
        print(f"   Dashboard salvato!")

    # Salva report JSON
    report = {
        'timestamp': datetime.now().isoformat(),
        'pairs_analyzed': len(results),
        'results': results,
        'charts_dir': str(OUTPUT_DIR)
    }

    report_path = OUTPUT_DIR / "analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETATO!")
    print("=" * 60)
    print(f"\nGrafici salvati in: {OUTPUT_DIR}")

    # Statistiche finali
    if results:
        returns = [r['return'] for r in results.values()]
        profitable = sum(1 for r in returns if r > 0)
        print(f"\nRiepilogo:")
        print(f"  - Coppie analizzate: {len(results)}")
        print(f"  - Return medio: {np.mean(returns):+.2f}%")
        print(f"  - Coppie profittevoli: {profitable}/{len(results)}")


if __name__ == "__main__":
    main()
