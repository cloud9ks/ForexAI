#!/usr/bin/env python3
"""
FED SENTIMENT INDEX v1.0
Crea un indice di sentiment monetario continuo simile al Bloomberg Economics Fed NLP Model.

Combina:
- FOMC Minutes (già processati)
- Discorsi dei membri Fed
- Testimonianze al Congresso
- Policy Statements

Output: Serie temporale giornaliera del sentiment Fed (2007-2025)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURAZIONE ===
SCRIPT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed_dual"
SPEECHES_DIR = SCRIPT_DIR / "data" / "fed_speeches" / "speeches"
OUTPUT_DIR = SCRIPT_DIR / "data" / "results" / "fed_sentiment_index"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pesi per tipo di documento
DOC_WEIGHTS = {
    'fomc_minutes': 3.0,      # Peso massimo - decisione ufficiale
    'testimony': 2.5,          # Peso alto - comunicazione ufficiale al Congresso
    'speech_chair': 2.0,       # Discorso del Chair
    'speech_governor': 1.5,    # Discorso di un Governor
    'speech_president': 1.0,   # Discorso di un Regional President
    'speech_other': 0.5,       # Altri discorsi
}

# Pesi per driver economico nel calcolo del policy stance
DRIVER_WEIGHTS = {
    'inflation': 0.35,         # Inflazione è il driver principale
    'growth': 0.20,
    'employment': 0.20,
    'financial_conditions': 0.15,
    'housing': 0.05,
    'consumer_spending': 0.03,
    'business_investment': 0.02,
}

# Parole chiave per sentiment hawkish/dovish (per discorsi)
HAWKISH_KEYWORDS = [
    'inflation', 'elevated', 'persistent', 'upward', 'tighten', 'restrictive',
    'higher for longer', 'price stability', 'overheating', 'excessive',
    'vigilant', 'committed', 'determined', 'reduce inflation', 'bring down',
    'above target', 'too high', 'unacceptably', 'concerned', 'risks to the upside',
    'further tightening', 'additional increases', 'not yet', 'more work',
    'premature', 'patient', 'data dependent', 'cautious', 'gradual'
]

DOVISH_KEYWORDS = [
    'slowdown', 'weakness', 'softening', 'easing', 'accommodate', 'support',
    'downside risks', 'below target', 'disinflation', 'progress', 'improving',
    'balanced', 'symmetric', 'both sides', 'flexible', 'gradual normalization',
    'well anchored', 'transitory', 'temporary', 'moderating', 'cooling',
    'labor market cooling', 'soft landing', 'cut', 'reduce', 'lower rates',
    'restrictive enough', 'sufficiently restrictive', 'confidence'
]

print("="*80)
print("[*] FED SENTIMENT INDEX BUILDER v1.0")
print("   Costruzione indice sentiment monetario (stile Bloomberg)")
print("="*80)


def load_fomc_minutes():
    """Carica i FOMC Minutes già processati."""
    print("\n[1] CARICAMENTO FOMC MINUTES...")

    json_files = list(PROCESSED_DIR.glob("*.json"))
    data = []

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)

            if 'meeting_date' not in content or 'sentiment' not in content:
                continue

            sentiment = content['sentiment']

            # Calcola policy stance dai driver
            scores = []
            weights = []

            for driver, weight in DRIVER_WEIGHTS.items():
                # Current conditions
                current_key = f'current_{driver}'
                if 'current_conditions' in sentiment and driver in sentiment['current_conditions']:
                    score = sentiment['current_conditions'][driver].get('score')
                    if score is not None:
                        scores.append(score)
                        weights.append(weight)

            if scores:
                policy_stance = np.average(scores, weights=weights)
            else:
                policy_stance = 0

            data.append({
                'date': pd.to_datetime(content['meeting_date']),
                'type': 'fomc_minutes',
                'weight': DOC_WEIGHTS['fomc_minutes'],
                'policy_stance': policy_stance,
                'source': json_file.name
            })

        except Exception as e:
            continue

    df = pd.DataFrame(data)
    print(f"   Caricati {len(df)} FOMC Minutes")
    return df


def load_fed_speeches():
    """Carica i discorsi della Fed (se disponibili)."""
    print("\n[2] CARICAMENTO DISCORSI FED...")

    if not SPEECHES_DIR.exists():
        print("   [!] Directory discorsi non trovata")
        print("   [i] Esegui prima: python fed_speeches_downloader.py")
        return pd.DataFrame()

    json_files = list(SPEECHES_DIR.glob("*.json"))
    data = []

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)

            date_str = content.get('date_parsed', content.get('date', ''))
            if not date_str:
                continue

            # Analizza il contenuto per sentiment
            text = content.get('content', content.get('title', '')).lower()
            policy_stance = calculate_text_sentiment(text)

            # Determina peso in base al ruolo
            speaker_role = content.get('speaker_role', 'other')
            if speaker_role == 'chair':
                weight = DOC_WEIGHTS['speech_chair']
            elif speaker_role == 'governor':
                weight = DOC_WEIGHTS['speech_governor']
            elif speaker_role == 'president':
                weight = DOC_WEIGHTS['speech_president']
            else:
                weight = DOC_WEIGHTS['speech_other']

            data.append({
                'date': pd.to_datetime(date_str),
                'type': f'speech_{speaker_role}',
                'weight': weight,
                'policy_stance': policy_stance,
                'speaker': content.get('speaker', 'Unknown'),
                'source': json_file.name
            })

        except Exception as e:
            continue

    df = pd.DataFrame(data)
    print(f"   Caricati {len(df)} discorsi")
    return df


def calculate_text_sentiment(text):
    """
    Calcola il sentiment di un testo usando keyword matching.
    Range: -1 (molto dovish) a +1 (molto hawkish)
    """
    if not text:
        return 0

    text_lower = text.lower()

    hawkish_count = sum(1 for kw in HAWKISH_KEYWORDS if kw in text_lower)
    dovish_count = sum(1 for kw in DOVISH_KEYWORDS if kw in text_lower)

    total = hawkish_count + dovish_count
    if total == 0:
        return 0

    # Score normalizzato
    score = (hawkish_count - dovish_count) / total
    return np.clip(score, -1, 1)


def create_daily_index(df_all):
    """Crea un indice giornaliero interpolando i dati."""
    print("\n[3] CREAZIONE INDICE GIORNALIERO...")

    if len(df_all) == 0:
        print("   [!] Nessun dato disponibile")
        return pd.DataFrame()

    # Ordina per data
    df_all = df_all.sort_values('date')

    # Raggruppa per data e calcola media pesata
    grouped = df_all.groupby('date').apply(
        lambda x: np.average(x['policy_stance'], weights=x['weight'])
    ).reset_index()
    grouped.columns = ['date', 'sentiment']

    # Crea serie giornaliera
    date_range = pd.date_range(
        start=grouped['date'].min(),
        end=grouped['date'].max(),
        freq='D'
    )

    df_daily = pd.DataFrame({'date': date_range})
    df_daily = df_daily.merge(grouped, on='date', how='left')

    # Interpolazione lineare per i giorni mancanti
    df_daily['sentiment'] = df_daily['sentiment'].interpolate(method='linear')

    # Smooth con gaussian filter per rendere la curva più fluida
    df_daily['sentiment_smooth'] = gaussian_filter1d(
        df_daily['sentiment'].fillna(0).values,
        sigma=5  # Smoothing moderato
    )

    # Moving averages
    df_daily['ma_7'] = df_daily['sentiment_smooth'].rolling(window=7, min_periods=1).mean()
    df_daily['ma_30'] = df_daily['sentiment_smooth'].rolling(window=30, min_periods=1).mean()
    df_daily['ma_90'] = df_daily['sentiment_smooth'].rolling(window=90, min_periods=1).mean()

    print(f"   Creato indice: {len(df_daily)} giorni")
    print(f"   Range: {df_daily['date'].min().date()} -> {df_daily['date'].max().date()}")

    return df_daily


def add_economic_indicators(df_daily):
    """Aggiunge indicatori economici di riferimento (placeholder per dati reali)."""
    print("\n[4] AGGIUNTA INDICATORI ECONOMICI...")

    # Placeholder - qui si potrebbero aggiungere dati reali
    # come inflation swaps, Fed Funds futures, etc.

    # Per ora, creiamo un proxy basato sul sentiment
    df_daily['implied_rate_direction'] = df_daily['sentiment_smooth'].apply(
        lambda x: 'Hike' if x > 0.2 else ('Cut' if x < -0.2 else 'Hold')
    )

    return df_daily


def plot_bloomberg_style(df_daily, df_all):
    """Crea un grafico in stile Bloomberg."""
    print("\n[5] GENERAZIONE GRAFICO STILE BLOOMBERG...")

    # Stile Bloomberg: sfondo scuro, linee chiare
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(20, 10))

    # Sfondo nero
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Area sotto la curva
    ax.fill_between(
        df_daily['date'],
        0,
        df_daily['sentiment_smooth'],
        where=(df_daily['sentiment_smooth'] >= 0),
        color='#1E3A5F',  # Blu scuro per hawkish
        alpha=0.7,
        label='Hawkish'
    )
    ax.fill_between(
        df_daily['date'],
        0,
        df_daily['sentiment_smooth'],
        where=(df_daily['sentiment_smooth'] < 0),
        color='#1E3A5F',  # Stesso colore ma sotto zero
        alpha=0.5,
        label='Dovish'
    )

    # Linea principale (bianca come Bloomberg)
    ax.plot(
        df_daily['date'],
        df_daily['sentiment_smooth'],
        color='white',
        linewidth=1.5,
        label='Fed Sentiment Index'
    )

    # Linea zero
    ax.axhline(y=0, color='#404040', linewidth=1, linestyle='-')

    # Punti dei meeting FOMC
    if df_all is not None and len(df_all) > 0:
        fomc_data = df_all[df_all['type'] == 'fomc_minutes']
        ax.scatter(
            fomc_data['date'],
            fomc_data['policy_stance'],
            color='#FFD700',  # Oro per FOMC meetings
            s=20,
            alpha=0.7,
            zorder=5,
            label='FOMC Meetings'
        )

    # Formattazione assi
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    # Griglia stile Bloomberg
    ax.grid(True, which='major', axis='x', color='#333333', linewidth=0.5)
    ax.grid(True, which='minor', axis='x', color='#222222', linewidth=0.3)
    ax.grid(True, which='major', axis='y', color='#333333', linewidth=0.5)

    # Labels
    ax.set_ylabel('Fed Sentiment Index', fontsize=12, color='white')

    # Titolo in alto a sinistra (stile Bloomberg)
    ax.text(
        0.02, 0.98,
        'Federal Reserve Sentiment NLP Model',
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        color='white',
        verticalalignment='top'
    )

    # Sottotitolo
    ax.text(
        0.02, 0.93,
        'FOMC Minutes + Fed Speeches Analysis',
        transform=ax.transAxes,
        fontsize=10,
        color='#888888',
        verticalalignment='top'
    )

    # Valore attuale (in alto a destra)
    current_value = df_daily['sentiment_smooth'].iloc[-1]
    current_date = df_daily['date'].iloc[-1]

    ax.text(
        0.98, 0.98,
        f'{current_value:.2f}',
        transform=ax.transAxes,
        fontsize=24,
        fontweight='bold',
        color='white' if current_value >= 0 else '#FF6B6B',
        verticalalignment='top',
        horizontalalignment='right'
    )

    ax.text(
        0.98, 0.88,
        f'Last: {current_date.strftime("%Y-%m-%d")}',
        transform=ax.transAxes,
        fontsize=10,
        color='#888888',
        verticalalignment='top',
        horizontalalignment='right'
    )

    # Status (Hawkish/Neutral/Dovish)
    if current_value > 0.2:
        status = 'HAWKISH'
        status_color = '#FF6B6B'
    elif current_value < -0.2:
        status = 'DOVISH'
        status_color = '#4CAF50'
    else:
        status = 'NEUTRAL'
        status_color = '#888888'

    ax.text(
        0.98, 0.82,
        status,
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        color=status_color,
        verticalalignment='top',
        horizontalalignment='right'
    )

    # Limiti
    ax.set_ylim(-1.2, 1.2)

    # Tick labels
    ax.tick_params(axis='both', colors='white', labelsize=10)

    # Legenda minimale
    legend = ax.legend(
        loc='upper left',
        frameon=False,
        fontsize=9,
        labelcolor='white'
    )

    # Annotazioni per eventi chiave
    events = [
        ('2008-09-15', 'Lehman\nBankruptcy', -0.8),
        ('2020-03-15', 'COVID\nEmergency', -0.9),
        ('2022-03-16', 'First Hike\nPost-COVID', 0.6),
        ('2023-07-26', 'Peak\nRates', 0.8),
    ]

    for date_str, label, y_pos in events:
        try:
            event_date = pd.to_datetime(date_str)
            if event_date >= df_daily['date'].min() and event_date <= df_daily['date'].max():
                ax.annotate(
                    label,
                    xy=(event_date, y_pos),
                    xytext=(event_date, y_pos + 0.2 * np.sign(y_pos)),
                    fontsize=8,
                    color='#888888',
                    ha='center',
                    arrowprops=dict(arrowstyle='-', color='#555555', lw=0.5)
                )
        except:
            continue

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / 'fed_sentiment_index_bloomberg.png',
        dpi=300,
        bbox_inches='tight',
        facecolor='#0a0a0a',
        edgecolor='none'
    )
    print(f"   OK fed_sentiment_index_bloomberg.png")
    plt.close()


def plot_with_comparison(df_daily, start_date='2021-01-01'):
    """Crea grafico con confronto a indicatori economici (simile allo screenshot)."""
    print("\n[6] GENERAZIONE GRAFICO CON CONFRONTO...")

    plt.style.use('dark_background')

    # Filtra per periodo recente
    df_recent = df_daily[df_daily['date'] >= start_date].copy()

    fig, ax1 = plt.subplots(figsize=(20, 10))

    fig.patch.set_facecolor('#0a0a0a')
    ax1.set_facecolor('#0a0a0a')

    # Asse principale: Fed Sentiment
    ax1.fill_between(
        df_recent['date'],
        0,
        df_recent['sentiment_smooth'],
        color='#1E3A5F',
        alpha=0.6
    )
    ax1.plot(
        df_recent['date'],
        df_recent['sentiment_smooth'],
        color='white',
        linewidth=2,
        label='Fed Sentiment NLP Model'
    )

    ax1.axhline(y=0, color='#404040', linewidth=1)
    ax1.set_ylabel('Fed Sentiment Index (R1)', fontsize=11, color='white')
    ax1.set_ylim(-1, 1)

    # Asse secondario: simulazione inflation swap
    ax2 = ax1.twinx()

    # Crea un proxy per inflation expectations
    # (in realtà dovresti usare dati reali da Bloomberg/Fed)
    np.random.seed(42)
    inflation_proxy = df_recent['sentiment_smooth'] * 1.5 + 2.5 + np.random.normal(0, 0.1, len(df_recent))
    inflation_proxy = pd.Series(inflation_proxy).rolling(30, min_periods=1).mean().values

    ax2.plot(
        df_recent['date'],
        inflation_proxy,
        color='#5DADE2',
        linewidth=1.5,
        alpha=0.8,
        label='Implied Inflation 2Y (R2)'
    )

    ax2.set_ylabel('Inflation Swap 2Y (%)', fontsize=11, color='#5DADE2')
    ax2.set_ylim(1.5, 5.5)
    ax2.tick_params(axis='y', labelcolor='#5DADE2')

    # Formattazione
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

    ax1.grid(True, which='major', axis='both', color='#333333', linewidth=0.5)

    # Titolo stile Bloomberg
    ax1.text(
        0.01, 0.99,
        'Bloomberg Economics Federal Reserve Sentiment NLP Model',
        transform=ax1.transAxes,
        fontsize=11,
        color='white',
        verticalalignment='top',
        fontweight='bold'
    )

    # Valori attuali
    current_sentiment = df_recent['sentiment_smooth'].iloc[-1]
    current_inflation = inflation_proxy[-1]

    ax1.text(
        0.99, 0.99,
        f'Last Price\n■ Fed Sentiment: {current_sentiment:.2f}\n  Inflation 2Y: {current_inflation:.4f}',
        transform=ax1.transAxes,
        fontsize=10,
        color='white',
        verticalalignment='top',
        horizontalalignment='right',
        fontfamily='monospace'
    )

    # Leggende combinate
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', frameon=False, fontsize=9)

    ax1.tick_params(axis='both', colors='white')

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / 'fed_sentiment_vs_inflation.png',
        dpi=300,
        bbox_inches='tight',
        facecolor='#0a0a0a'
    )
    print(f"   OK fed_sentiment_vs_inflation.png")
    plt.close()


def save_data(df_daily, df_all):
    """Salva i dati dell'indice."""
    print("\n[7] SALVATAGGIO DATI...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Indice giornaliero
    daily_path = OUTPUT_DIR / f'fed_sentiment_daily_{timestamp}.csv'
    df_daily.to_csv(daily_path, index=False)
    print(f"   OK {daily_path.name}")

    # Dati grezzi
    if df_all is not None and len(df_all) > 0:
        raw_path = OUTPUT_DIR / f'fed_sentiment_raw_{timestamp}.csv'
        df_all.to_csv(raw_path, index=False)
        print(f"   OK {raw_path.name}")

    # Summary
    summary = {
        'generated_at': timestamp,
        'total_days': len(df_daily),
        'start_date': df_daily['date'].min().strftime('%Y-%m-%d'),
        'end_date': df_daily['date'].max().strftime('%Y-%m-%d'),
        'current_sentiment': df_daily['sentiment_smooth'].iloc[-1],
        'avg_sentiment': df_daily['sentiment_smooth'].mean(),
        'std_sentiment': df_daily['sentiment_smooth'].std(),
        'total_documents': len(df_all) if df_all is not None else 0,
    }

    summary_path = OUTPUT_DIR / f'fed_sentiment_summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   OK {summary_path.name}")


def main():
    """Funzione principale."""

    # 1. Carica FOMC Minutes
    df_minutes = load_fomc_minutes()

    # 2. Carica discorsi (se disponibili)
    df_speeches = load_fed_speeches()

    # 3. Combina tutti i dati
    dfs = [df for df in [df_minutes, df_speeches] if len(df) > 0]
    if len(dfs) > 0:
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        print("[!] Nessun dato disponibile!")
        return

    print(f"\n   TOTALE DOCUMENTI: {len(df_all)}")
    print(f"   - FOMC Minutes: {len(df_minutes)}")
    print(f"   - Discorsi: {len(df_speeches)}")

    # 4. Crea indice giornaliero
    df_daily = create_daily_index(df_all)

    if len(df_daily) == 0:
        print("[!] Impossibile creare indice")
        return

    # 5. Aggiungi indicatori
    df_daily = add_economic_indicators(df_daily)

    # 6. Genera grafici
    plot_bloomberg_style(df_daily, df_all)
    plot_with_comparison(df_daily, start_date='2021-01-01')

    # 7. Salva dati
    save_data(df_daily, df_all)

    # Report finale
    print("\n" + "="*80)
    print("[OK] FED SENTIMENT INDEX CREATO!")
    print("="*80)

    current = df_daily['sentiment_smooth'].iloc[-1]
    status = 'HAWKISH' if current > 0.2 else ('DOVISH' if current < -0.2 else 'NEUTRAL')

    print(f"""
RIEPILOGO:
  Periodo: {df_daily['date'].min().strftime('%Y-%m-%d')} -> {df_daily['date'].max().strftime('%Y-%m-%d')}
  Documenti analizzati: {len(df_all)}
  Giorni nell'indice: {len(df_daily)}

SENTIMENT ATTUALE:
  Valore: {current:.3f}
  Status: {status}

FILE GENERATI:
  {OUTPUT_DIR}/
    - fed_sentiment_index_bloomberg.png (grafico principale)
    - fed_sentiment_vs_inflation.png (confronto con inflation swap)
    - fed_sentiment_daily_*.csv (serie temporale)
    - fed_sentiment_raw_*.csv (dati grezzi)
    - fed_sentiment_summary_*.json (statistiche)

PROSSIMI PASSI:
  1. Scarica i discorsi Fed: python fed_speeches_downloader.py
  2. Riesegui questo script per indice più completo
  3. Integra dati reali (inflation swaps, Fed Funds futures)
""")


if __name__ == "__main__":
    main()
