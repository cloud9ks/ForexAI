#!/usr/bin/env python3
"""
FED SENTIMENT INDEX - PROFESSIONAL EDITION v2.0
Replica della metodologia Bloomberg/Morgan Stanley per analisi NLP Fed

Basato su:
- Morgan Stanley MNLPFEDS Index methodology
- Loughran-McDonald Financial Sentiment Dictionary
- Federal Reserve Research Papers (2025)
- BIS Working Papers on Central Bank Communication

Features:
- Dizionario hawkish/dovish specifico per politica monetaria
- Gestione negazioni (3-word window)
- Pesatura per speaker importance
- Smoothing professionale (Kalman filter style)
- Output giornaliero interpolato

Sources:
- https://sraf.nd.edu/loughranmcdonald-master-dictionary/
- https://www.morganstanley.com/articles/mnlpfeds-sentiment-index-federal-reserve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import re
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURAZIONE ===
SCRIPT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed_dual"
SPEECHES_DIR = SCRIPT_DIR / "data" / "fed_speeches" / "speeches"
OUTPUT_DIR = SCRIPT_DIR / "data" / "results" / "fed_sentiment_pro"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DIZIONARIO HAWKISH/DOVISH - MONETARY POLICY SPECIFIC
# Basato su ricerca accademica e prassi industria (BIS, Fed, Morgan Stanley)
# =============================================================================

# HAWKISH: Parole/frasi che indicano stance restrittiva (tightening)
HAWKISH_TERMS = {
    # Inflazione - preoccupazioni
    'inflation': 1.0,
    'inflationary': 1.2,
    'inflationary pressures': 1.5,
    'price pressures': 1.3,
    'price stability': 0.8,  # Enfasi su stabilita = preoccupazione
    'elevated inflation': 1.8,
    'persistent inflation': 1.8,
    'sticky inflation': 1.6,
    'core inflation': 0.9,
    'above target': 1.5,
    'above 2 percent': 1.4,
    'unacceptably high': 2.0,
    'too high': 1.6,
    'upside risks to inflation': 1.7,

    # Policy stance - restrittiva
    'tighten': 1.5,
    'tightening': 1.5,
    'restrictive': 1.3,
    'sufficiently restrictive': 1.4,
    'more restrictive': 1.6,
    'further tightening': 1.8,
    'additional increases': 1.6,
    'rate increases': 1.4,
    'raise rates': 1.5,
    'higher rates': 1.3,
    'higher for longer': 1.7,
    'maintain restrictive': 1.4,
    'keep rates elevated': 1.5,

    # Mercato lavoro - forte
    'tight labor market': 1.2,
    'labor market tightness': 1.2,
    'strong labor market': 1.0,
    'robust employment': 1.0,
    'wage pressures': 1.4,
    'wage growth': 1.1,
    'labor costs': 1.2,

    # Economia - forte
    'overheating': 1.8,
    'strong growth': 0.9,
    'robust growth': 0.9,
    'solid growth': 0.8,
    'economic strength': 0.8,
    'resilient': 0.7,
    'momentum': 0.6,

    # Cautela/Vigilanza
    'vigilant': 1.3,
    'committed': 1.2,
    'determined': 1.3,
    'resolve': 1.2,
    'firmly committed': 1.5,
    'whatever it takes': 1.6,
    'not done': 1.4,
    'more work to do': 1.5,
    'premature': 1.4,
    'too soon': 1.3,
    'patient': 0.8,
    'cautious': 0.7,

    # Rischi upside
    'upside risks': 1.3,
    'risks to the upside': 1.3,
    'could increase': 0.9,
    'may rise': 0.8,
}

# DOVISH: Parole/frasi che indicano stance accomodante (easing)
DOVISH_TERMS = {
    # Inflazione - miglioramento
    'disinflation': -1.3,
    'inflation falling': -1.4,
    'inflation declining': -1.4,
    'inflation moderating': -1.2,
    'lower inflation': -1.3,
    'toward 2 percent': -1.0,
    'returning to target': -1.2,
    'progress on inflation': -1.3,
    'inflation progress': -1.3,
    'good progress': -1.1,
    'significant progress': -1.4,
    'welcome decline': -1.3,

    # Policy stance - accomodante
    'ease': -1.3,
    'easing': -1.3,
    'accommodative': -1.4,
    'accommodation': -1.3,
    'support': -0.8,
    'supportive': -0.9,
    'cut rates': -1.5,
    'rate cuts': -1.5,
    'lower rates': -1.3,
    'reduce rates': -1.4,
    'policy easing': -1.5,
    'less restrictive': -1.2,
    'normalize': -1.0,
    'normalization': -1.0,
    'recalibrate': -1.1,
    'recalibration': -1.1,
    'dial back': -1.2,

    # Mercato lavoro - debole
    'labor market cooling': -1.2,
    'cooling labor market': -1.2,
    'softening': -1.1,
    'soft landing': -1.0,
    'slowing': -0.9,
    'slower growth': -1.0,
    'weaker': -1.1,
    'weakness': -1.2,
    'slack': -1.0,
    'unemployment rising': -1.3,

    # Economia - debole
    'slowdown': -1.2,
    'recession': -1.8,
    'contraction': -1.6,
    'downturn': -1.5,
    'headwinds': -1.0,
    'downside risks': -1.3,
    'risks to the downside': -1.3,
    'fragile': -1.1,
    'vulnerable': -1.0,
    'uncertainty': -0.8,
    'concerns': -0.7,

    # Fiducia/Comfort
    'confidence': -0.9,
    'confident': -0.9,
    'greater confidence': -1.2,
    'more confident': -1.1,
    'comfortable': -0.8,
    'well anchored': -1.0,
    'anchored expectations': -1.1,
    'balanced': -0.6,
    'symmetric': -0.5,
    'both sides': -0.5,
    'flexible': -0.7,
    'data dependent': -0.5,
}

# NEGAZIONI - invertono il sentiment (window di 3 parole prima)
NEGATIONS = {
    'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
    'hardly', 'scarcely', 'barely', "n't", 'cannot', "can't", "won't",
    "wouldn't", "shouldn't", "couldn't", "didn't", "doesn't", "isn't",
    "aren't", "wasn't", "weren't", 'without', 'lack', 'lacking',
    'unlikely', 'fail', 'failed', 'failing'
}

# INTENSIFICATORI - amplificano il sentiment
INTENSIFIERS = {
    'very': 1.3,
    'extremely': 1.5,
    'highly': 1.3,
    'significantly': 1.4,
    'substantially': 1.4,
    'considerably': 1.3,
    'particularly': 1.2,
    'especially': 1.2,
    'notably': 1.2,
    'strongly': 1.3,
    'firmly': 1.3,
    'clearly': 1.1,
    'certainly': 1.2,
    'absolutely': 1.4,
    'quite': 1.1,
    'rather': 1.1,
    'somewhat': 0.8,
    'slightly': 0.7,
    'marginally': 0.7,
    'modestly': 0.8,
    'gradually': 0.8,
}

# PESI PER TIPO DI SPEAKER (Morgan Stanley approach)
SPEAKER_WEIGHTS = {
    'chair': 3.0,           # Powell, Yellen, Bernanke
    'vice chair': 2.5,      # Jefferson, Clarida
    'governor': 2.0,        # Board members
    'president': 1.5,       # Regional Fed presidents
    'vice president': 1.2,
    'director': 1.0,
    'other': 0.8,
}

# PESI PER TIPO DI DOCUMENTO
DOCUMENT_WEIGHTS = {
    'fomc_statement': 4.0,      # Massima importanza
    'fomc_minutes': 3.5,        # Molto importante
    'testimony': 3.0,           # Testimonianze al Congresso
    'speech_chair': 2.5,        # Discorsi del Chair
    'speech_governor': 2.0,     # Discorsi Governors
    'speech_president': 1.5,    # Discorsi Regional Presidents
    'press_conference': 2.5,    # Press conference
    'other': 1.0,
}

print("="*80)
print("[*] FED SENTIMENT INDEX - PROFESSIONAL EDITION v2.0")
print("   Metodologia: Morgan Stanley MNLPFEDS + Loughran-McDonald")
print("="*80)


def preprocess_text(text):
    """Preprocessa il testo per l'analisi."""
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Rimuovi caratteri speciali ma mantieni apostrofi
    text = re.sub(r'[^\w\s\'-]', ' ', text)

    # Normalizza spazi
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_sentences(text):
    """Divide il testo in frasi."""
    # Split su punto, punto esclamativo, punto interrogativo
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def analyze_sentence(sentence, hawkish_dict, dovish_dict):
    """
    Analizza una singola frase per sentiment hawkish/dovish.
    Gestisce negazioni e intensificatori.

    Returns: (score, matched_terms)
    """
    words = sentence.lower().split()
    score = 0.0
    matched_terms = []

    # Prima cerca frasi multi-parola (più specifiche)
    sentence_lower = sentence.lower()

    # Check hawkish phrases
    for phrase, weight in sorted(hawkish_dict.items(), key=lambda x: -len(x[0].split())):
        if phrase in sentence_lower:
            # Controlla negazione
            phrase_idx = sentence_lower.find(phrase)
            words_before = sentence_lower[:phrase_idx].split()[-3:]  # 3-word window

            is_negated = any(neg in words_before for neg in NEGATIONS)

            # Controlla intensificatori
            intensity = 1.0
            for word in words_before:
                if word in INTENSIFIERS:
                    intensity = INTENSIFIERS[word]
                    break

            if is_negated:
                score -= weight * intensity  # Negazione inverte
                matched_terms.append(f"NOT {phrase}")
            else:
                score += weight * intensity
                matched_terms.append(phrase)

    # Check dovish phrases
    for phrase, weight in sorted(dovish_dict.items(), key=lambda x: -len(x[0].split())):
        if phrase in sentence_lower:
            phrase_idx = sentence_lower.find(phrase)
            words_before = sentence_lower[:phrase_idx].split()[-3:]

            is_negated = any(neg in words_before for neg in NEGATIONS)

            intensity = 1.0
            for word in words_before:
                if word in INTENSIFIERS:
                    intensity = INTENSIFIERS[word]
                    break

            if is_negated:
                score -= weight * intensity  # Negazione inverte (rende hawkish)
                matched_terms.append(f"NOT {phrase}")
            else:
                score += weight * intensity  # Dovish terms hanno gia peso negativo
                matched_terms.append(phrase)

    return score, matched_terms


def analyze_document(text, doc_type='other', speaker='unknown'):
    """
    Analizza un documento completo.

    Returns: dict con score, details, matched_terms
    """
    if not text:
        return {'score': 0, 'sentence_scores': [], 'matched_terms': [], 'num_sentences': 0}

    text_clean = preprocess_text(text)
    sentences = get_sentences(text_clean)

    sentence_scores = []
    all_matched = []

    for sentence in sentences:
        if len(sentence.split()) < 5:  # Skip frasi troppo corte
            continue

        score, matched = analyze_sentence(sentence, HAWKISH_TERMS, DOVISH_TERMS)
        if score != 0:
            sentence_scores.append(score)
            all_matched.extend(matched)

    if not sentence_scores:
        return {'score': 0, 'sentence_scores': [], 'matched_terms': [], 'num_sentences': len(sentences)}

    # Score medio normalizzato
    raw_score = np.mean(sentence_scores)

    # Normalizza in range [-1, 1]
    normalized_score = np.tanh(raw_score / 2)  # tanh per smoothing naturale

    # Applica peso documento
    doc_weight = DOCUMENT_WEIGHTS.get(doc_type, 1.0)

    # Applica peso speaker
    speaker_lower = speaker.lower()
    speaker_weight = 1.0
    for role, weight in SPEAKER_WEIGHTS.items():
        if role in speaker_lower:
            speaker_weight = weight
            break

    return {
        'score': normalized_score,
        'raw_score': raw_score,
        'doc_weight': doc_weight,
        'speaker_weight': speaker_weight,
        'weighted_score': normalized_score * doc_weight * speaker_weight,
        'sentence_scores': sentence_scores,
        'matched_terms': all_matched,
        'num_sentences': len(sentences),
        'num_scored': len(sentence_scores)
    }


def load_fomc_minutes():
    """Carica e analizza i FOMC Minutes usando FORWARD EXPECTATIONS per stance policy."""
    print("\n[1] ANALISI FOMC MINUTES...")

    json_files = list(PROCESSED_DIR.glob("*.json"))
    data = []

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)

            if 'meeting_date' not in content or 'sentiment' not in content:
                continue

            sentiment = content['sentiment']

            # =================================================================
            # USA FORWARD_EXPECTATIONS per catturare la STANCE DI POLICY
            # (non current_conditions che riflette solo l'economia attuale)
            # =================================================================

            # Pesi per driver - LOGICA POLICY STANCE:
            # - inflation forward: se attesa in calo -> Fed puo tagliare (dovish)
            # - growth forward: se attesa debole -> Fed deve supportare (dovish)
            # - employment forward: se atteso debole -> Fed deve supportare (dovish)
            # - financial_conditions forward: se attese strette -> Fed deve allentare (dovish)

            driver_weights = {
                'inflation': 0.30,        # Aspettative inflazione
                'growth': 0.25,           # Outlook crescita
                'employment': 0.20,       # Outlook occupazione
                'financial_conditions': 0.15,  # Condizioni finanziarie attese
                'housing': 0.05,
                'consumer_spending': 0.03,
                'business_investment': 0.02,
            }

            forward_scores = []
            current_scores = []

            # FORWARD EXPECTATIONS (peso 70%)
            for driver, weight in driver_weights.items():
                if 'forward_expectations' in sentiment and driver in sentiment['forward_expectations']:
                    score = sentiment['forward_expectations'][driver].get('score')
                    if score is not None:
                        forward_scores.append((score, weight))

            # CURRENT CONDITIONS (peso 30%) - per contesto
            for driver, weight in driver_weights.items():
                if 'current_conditions' in sentiment and driver in sentiment['current_conditions']:
                    score = sentiment['current_conditions'][driver].get('score')
                    if score is not None:
                        current_scores.append((score, weight))

            # Calcola score combinato: 70% forward + 30% current
            forward_weighted = 0
            current_weighted = 0

            if forward_scores:
                forward_weighted = sum(s * w for s, w in forward_scores) / sum(w for _, w in forward_scores)
            if current_scores:
                current_weighted = sum(s * w for s, w in current_scores) / sum(w for _, w in current_scores)

            # Combina: forward expectations pesano di piu per la stance
            if forward_scores and current_scores:
                weighted_score = forward_weighted * 0.7 + current_weighted * 0.3
            elif forward_scores:
                weighted_score = forward_weighted
            elif current_scores:
                weighted_score = current_weighted
            else:
                weighted_score = 0

            data.append({
                'date': pd.to_datetime(content['meeting_date']),
                'type': 'fomc_minutes',
                'speaker': 'FOMC',
                'score': weighted_score,
                'doc_weight': DOCUMENT_WEIGHTS['fomc_minutes'],
                'speaker_weight': SPEAKER_WEIGHTS['chair'],
                'weighted_score': weighted_score * DOCUMENT_WEIGHTS['fomc_minutes'] * SPEAKER_WEIGHTS['chair'],
                'source': json_file.name,
                'forward_score': forward_weighted,
                'current_score': current_weighted
            })

        except Exception as e:
            continue

    df = pd.DataFrame(data)
    print(f"   Caricati {len(df)} FOMC Minutes")
    return df


def load_speeches():
    """Carica e analizza i discorsi Fed."""
    print("\n[2] ANALISI DISCORSI FED...")

    if not SPEECHES_DIR.exists():
        print("   [!] Directory discorsi non trovata")
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

            speaker = content.get('speaker', 'Unknown')
            text = content.get('content', content.get('title', ''))

            # Determina tipo documento
            speaker_lower = speaker.lower()
            if 'chair' in speaker_lower and 'vice' not in speaker_lower:
                doc_type = 'speech_chair'
            elif 'governor' in speaker_lower:
                doc_type = 'speech_governor'
            elif 'president' in speaker_lower:
                doc_type = 'speech_president'
            else:
                doc_type = 'other'

            # Analizza il testo
            analysis = analyze_document(text, doc_type, speaker)

            data.append({
                'date': pd.to_datetime(date_str),
                'type': doc_type,
                'speaker': speaker,
                'score': analysis['score'],
                'doc_weight': analysis['doc_weight'],
                'speaker_weight': analysis['speaker_weight'],
                'weighted_score': analysis['weighted_score'],
                'num_terms': len(analysis['matched_terms']),
                'source': json_file.name
            })

        except Exception as e:
            continue

    df = pd.DataFrame(data)
    print(f"   Analizzati {len(df)} discorsi")
    return df


def create_daily_index(df_all):
    """
    Crea indice giornaliero con smoothing professionale.
    Usa interpolazione + Exponential Weighted Moving Average (come Bloomberg).
    """
    print("\n[3] CREAZIONE INDICE GIORNALIERO...")

    if len(df_all) == 0:
        return pd.DataFrame()

    # Aggrega per data con media pesata
    df_all['contribution'] = df_all['score'] * df_all['doc_weight'] * df_all['speaker_weight']
    df_all['total_weight'] = df_all['doc_weight'] * df_all['speaker_weight']

    daily_agg = df_all.groupby('date').agg({
        'contribution': 'sum',
        'total_weight': 'sum',
        'score': 'count'
    }).reset_index()

    daily_agg.columns = ['date', 'contribution', 'total_weight', 'doc_count']
    daily_agg['weighted_sentiment'] = daily_agg['contribution'] / daily_agg['total_weight']

    # Crea serie giornaliera completa
    date_range = pd.date_range(
        start=daily_agg['date'].min(),
        end=max(daily_agg['date'].max(), datetime.now()),
        freq='D'
    )

    df_daily = pd.DataFrame({'date': date_range})
    df_daily = df_daily.merge(
        daily_agg[['date', 'weighted_sentiment', 'doc_count']],
        on='date',
        how='left'
    )

    # INTERPOLAZIONE PROFESSIONALE
    # 1. Interpolazione lineare per gap piccoli (<30 giorni)
    df_daily['sentiment_interp'] = df_daily['weighted_sentiment'].interpolate(
        method='linear',
        limit=30,
        limit_direction='both'
    )

    # 2. Per gap piu grandi, usa il valore precedente (forward fill)
    df_daily['sentiment_interp'] = df_daily['sentiment_interp'].ffill().bfill()

    # SMOOTHING PROFESSIONALE (Bloomberg-style EWMA)
    # Halflife maggiore per curva piu smooth
    df_daily['sentiment_ewma'] = df_daily['sentiment_interp'].ewm(
        halflife=60,  # Aumentato per smoothing piu forte
        min_periods=1
    ).mean()

    # Secondo passaggio di smoothing per eliminare rumore residuo
    df_daily['sentiment_ewma'] = df_daily['sentiment_ewma'].ewm(
        halflife=30,
        min_periods=1
    ).mean()

    # NORMALIZZAZIONE: centra i dati attorno allo zero
    # Il sentiment medio storico dovrebbe essere ~0 (neutrale)
    historical_mean = df_daily['sentiment_ewma'].mean()
    historical_std = df_daily['sentiment_ewma'].std()

    # Z-score normalization: (x - mean) / std, poi riscala per range ragionevole
    df_daily['sentiment_ewma'] = (df_daily['sentiment_ewma'] - historical_mean) / historical_std
    df_daily['sentiment_ewma'] = df_daily['sentiment_ewma'] * 0.3  # Scala per range [-1, 1]

    # Clip per evitare outlier estremi
    df_daily['sentiment_ewma'] = df_daily['sentiment_ewma'].clip(-0.8, 0.8)

    # Moving averages per analisi
    df_daily['ma_7'] = df_daily['sentiment_ewma'].rolling(window=7, min_periods=1).mean()
    df_daily['ma_30'] = df_daily['sentiment_ewma'].rolling(window=30, min_periods=1).mean()
    df_daily['ma_90'] = df_daily['sentiment_ewma'].rolling(window=90, min_periods=1).mean()

    # Volatilita (rolling std)
    df_daily['volatility'] = df_daily['sentiment_ewma'].rolling(window=30, min_periods=7).std()

    # Momentum (rate of change)
    df_daily['momentum'] = df_daily['sentiment_ewma'].diff(periods=30)

    # Regime classification
    df_daily['regime'] = 'Neutral'
    df_daily.loc[df_daily['sentiment_ewma'] > 0.15, 'regime'] = 'Hawkish'
    df_daily.loc[df_daily['sentiment_ewma'] < -0.15, 'regime'] = 'Dovish'

    print(f"   Creato indice: {len(df_daily)} giorni")
    print(f"   Range: {df_daily['date'].min().date()} -> {df_daily['date'].max().date()}")

    return df_daily


def plot_bloomberg_style(df_daily, df_all, output_path):
    """Genera grafico stile Bloomberg Terminal."""
    print("\n[4] GENERAZIONE GRAFICO BLOOMBERG STYLE...")

    # Stile Bloomberg
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(24, 12))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')

    # Area chart principale
    dates = df_daily['date']
    sentiment = df_daily['sentiment_ewma']

    # Colore area basato su sentiment
    ax.fill_between(
        dates, 0, sentiment,
        where=(sentiment >= 0),
        color='#1a4a6e',  # Blu scuro per hawkish
        alpha=0.8,
        linewidth=0
    )
    ax.fill_between(
        dates, 0, sentiment,
        where=(sentiment < 0),
        color='#1a4a6e',  # Stesso colore
        alpha=0.5,
        linewidth=0
    )

    # Linea principale (bianca)
    ax.plot(dates, sentiment, color='white', linewidth=1.2, zorder=5)

    # Linea zero
    ax.axhline(y=0, color='#333333', linewidth=1.5, zorder=2)

    # Soglie hawkish/dovish
    ax.axhline(y=0.15, color='#8B0000', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=-0.15, color='#006400', linewidth=0.8, linestyle='--', alpha=0.5)

    # Punti FOMC meetings
    if df_all is not None:
        fomc = df_all[df_all['type'] == 'fomc_minutes']
        if len(fomc) > 0:
            # Merge con daily per ottenere sentiment smooth alla data
            for _, row in fomc.iterrows():
                date = row['date']
                daily_row = df_daily[df_daily['date'] == date]
                if len(daily_row) > 0:
                    y_val = daily_row['sentiment_ewma'].values[0]
                    ax.scatter(date, y_val, color='#FFD700', s=25, zorder=10, alpha=0.9)

    # Formattazione assi
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    # Griglia Bloomberg
    ax.grid(True, which='major', axis='x', color='#222222', linewidth=0.5)
    ax.grid(True, which='minor', axis='x', color='#181818', linewidth=0.3)
    ax.grid(True, which='major', axis='y', color='#222222', linewidth=0.5)

    # Tick labels
    ax.tick_params(axis='both', colors='#888888', labelsize=10)
    ax.set_ylabel('Fed Sentiment Index', fontsize=11, color='#888888')

    # Titolo (stile Bloomberg - in alto a sinistra)
    ax.text(0.01, 0.98, 'Federal Reserve Sentiment NLP Model',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            color='white', verticalalignment='top', fontfamily='sans-serif')

    ax.text(0.01, 0.93, 'FOMC Minutes + Fed Speeches | Methodology: MNLPFEDS',
            transform=ax.transAxes, fontsize=9, color='#666666',
            verticalalignment='top')

    # Valore attuale (in alto a destra)
    current_value = df_daily['sentiment_ewma'].iloc[-1]
    current_date = df_daily['date'].iloc[-1]

    value_color = '#FF6B6B' if current_value > 0 else '#4CAF50' if current_value < 0 else 'white'

    ax.text(0.99, 0.98, f'{current_value:.2f}',
            transform=ax.transAxes, fontsize=28, fontweight='bold',
            color=value_color, verticalalignment='top',
            horizontalalignment='right', fontfamily='monospace')

    ax.text(0.99, 0.88, f'Last: {current_date.strftime("%Y-%m-%d")}',
            transform=ax.transAxes, fontsize=10, color='#666666',
            verticalalignment='top', horizontalalignment='right')

    # Status label
    if current_value > 0.15:
        status = 'HAWKISH'
        status_color = '#FF6B6B'
    elif current_value < -0.15:
        status = 'DOVISH'
        status_color = '#4CAF50'
    else:
        status = 'NEUTRAL'
        status_color = '#888888'

    ax.text(0.99, 0.82, status,
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=status_color, verticalalignment='top',
            horizontalalignment='right')

    # Limiti
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlim(df_daily['date'].min(), df_daily['date'].max())

    # Eventi chiave
    events = [
        ('2008-09-15', 'Lehman', -0.55),
        ('2020-03-15', 'COVID', -0.55),
        ('2022-03-16', 'Hike\nCycle', 0.45),
        ('2024-09-18', 'First\nCut', 0.35),
    ]

    for date_str, label, y_offset in events:
        try:
            event_date = pd.to_datetime(date_str)
            if event_date >= df_daily['date'].min() and event_date <= df_daily['date'].max():
                ax.axvline(x=event_date, color='#444444', linewidth=0.5, linestyle=':')
                ax.text(event_date, y_offset, label,
                       fontsize=8, color='#555555', ha='center', va='center')
        except:
            continue

    # Legenda minimale
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='white', linewidth=1.5, label='Sentiment Index'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700',
               markersize=6, linestyle='None', label='FOMC Meetings'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False,
              fontsize=9, labelcolor='#888888')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='#000000', edgecolor='none')
    print(f"   OK {output_path.name}")
    plt.close()


def plot_detailed_view(df_daily, output_path, start_date='2020-01-01'):
    """Genera vista dettagliata periodo recente (stile Bloomberg 2 panels)."""
    print("\n[5] GENERAZIONE VISTA DETTAGLIATA...")

    plt.style.use('dark_background')

    # Filtra periodo recente
    df_recent = df_daily[df_daily['date'] >= start_date].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)

    fig.patch.set_facecolor('#000000')
    ax1.set_facecolor('#000000')
    ax2.set_facecolor('#000000')

    # Panel 1: Sentiment con MA
    dates = df_recent['date']
    sentiment = df_recent['sentiment_ewma']

    ax1.fill_between(dates, 0, sentiment,
                     where=(sentiment >= 0), color='#1a4a6e', alpha=0.7)
    ax1.fill_between(dates, 0, sentiment,
                     where=(sentiment < 0), color='#1a4a6e', alpha=0.4)

    ax1.plot(dates, sentiment, color='white', linewidth=1.5, label='Sentiment')
    ax1.plot(dates, df_recent['ma_30'], color='#FF9800', linewidth=1,
             linestyle='--', alpha=0.7, label='MA(30)')
    ax1.plot(dates, df_recent['ma_90'], color='#E91E63', linewidth=1,
             linestyle=':', alpha=0.5, label='MA(90)')

    ax1.axhline(y=0, color='#333333', linewidth=1.5)
    ax1.axhline(y=0.15, color='#8B0000', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.axhline(y=-0.15, color='#006400', linewidth=0.8, linestyle='--', alpha=0.5)

    ax1.set_ylabel('Sentiment Index', fontsize=11, color='#888888')
    ax1.legend(loc='upper left', frameon=False, fontsize=9, labelcolor='#888888')
    ax1.grid(True, color='#222222', linewidth=0.5)
    ax1.tick_params(axis='both', colors='#888888')
    ax1.set_ylim(-0.6, 0.6)

    # Titolo
    ax1.text(0.01, 0.98, 'Fed Sentiment Index - Detailed View',
             transform=ax1.transAxes, fontsize=14, fontweight='bold',
             color='white', verticalalignment='top')

    current_value = df_recent['sentiment_ewma'].iloc[-1]
    ax1.text(0.99, 0.98, f'{current_value:.3f}',
             transform=ax1.transAxes, fontsize=20, fontweight='bold',
             color='white', verticalalignment='top', horizontalalignment='right')

    # Panel 2: Momentum
    momentum = df_recent['momentum']
    colors_mom = ['#4CAF50' if m > 0 else '#F44336' for m in momentum]

    ax2.bar(dates, momentum, color=colors_mom, alpha=0.7, width=1)
    ax2.axhline(y=0, color='#333333', linewidth=1)
    ax2.set_ylabel('30-Day Momentum', fontsize=10, color='#888888')
    ax2.set_xlabel('Date', fontsize=10, color='#888888')
    ax2.grid(True, color='#222222', linewidth=0.5)
    ax2.tick_params(axis='both', colors='#888888')

    # Formattazione date
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='#000000', edgecolor='none')
    print(f"   OK {output_path.name}")
    plt.close()


def plot_monthly_view(df_daily, output_path):
    """Genera grafico con cadenza mensile - stile Bloomberg."""
    print("\n[5b] GENERAZIONE GRAFICO MENSILE...")

    plt.style.use('dark_background')

    # Aggrega a mensile
    df_monthly = df_daily.copy()
    df_monthly['year_month'] = df_monthly['date'].dt.to_period('M')

    df_agg = df_monthly.groupby('year_month').agg({
        'sentiment_ewma': 'mean',
        'ma_30': 'mean',
        'ma_90': 'mean',
        'volatility': 'mean',
        'momentum': 'mean'
    }).reset_index()

    # Converti period a datetime (primo del mese)
    df_agg['date'] = df_agg['year_month'].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(24, 12))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')

    dates = df_agg['date']
    sentiment = df_agg['sentiment_ewma']

    # Bar chart mensile (stile Bloomberg)
    colors = ['#FF6B6B' if s > 0.05 else '#4CAF50' if s < -0.05 else '#666666' for s in sentiment]

    ax.bar(dates, sentiment, width=25, color=colors, alpha=0.8, edgecolor='none')

    # Linea trend (MA3 mesi)
    ma3 = sentiment.rolling(window=3, min_periods=1).mean()
    ax.plot(dates, ma3, color='white', linewidth=2, label='Trend (MA3)')

    # Linea zero
    ax.axhline(y=0, color='#444444', linewidth=2, zorder=1)

    # Soglie
    ax.axhline(y=0.15, color='#8B0000', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(y=-0.15, color='#006400', linewidth=1, linestyle='--', alpha=0.5)

    # Formattazione assi
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    # Griglia
    ax.grid(True, which='major', axis='x', color='#222222', linewidth=0.5)
    ax.grid(True, which='major', axis='y', color='#222222', linewidth=0.5)

    ax.tick_params(axis='both', colors='#888888', labelsize=11)
    ax.set_ylabel('Fed Sentiment Index (Monthly Avg)', fontsize=12, color='#888888')
    ax.set_xlabel('', fontsize=10)

    # Titolo
    ax.text(0.01, 0.98, 'Federal Reserve Sentiment NLP Model - MONTHLY',
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            color='white', verticalalignment='top')

    ax.text(0.01, 0.93, f'Cadenza: Mensile | Totale mesi: {len(df_agg)} | Metodologia: MNLPFEDS',
            transform=ax.transAxes, fontsize=10, color='#666666',
            verticalalignment='top')

    # Valore attuale
    current_value = df_agg['sentiment_ewma'].iloc[-1]
    current_month = df_agg['date'].iloc[-1].strftime('%b %Y')

    value_color = '#FF6B6B' if current_value > 0.05 else '#4CAF50' if current_value < -0.05 else 'white'

    ax.text(0.99, 0.98, f'{current_value:.3f}',
            transform=ax.transAxes, fontsize=28, fontweight='bold',
            color=value_color, verticalalignment='top',
            horizontalalignment='right', fontfamily='monospace')

    ax.text(0.99, 0.88, f'{current_month}',
            transform=ax.transAxes, fontsize=12, color='#666666',
            verticalalignment='top', horizontalalignment='right')

    # Status
    if current_value > 0.15:
        status = 'HAWKISH'
        status_color = '#FF6B6B'
    elif current_value < -0.15:
        status = 'DOVISH'
        status_color = '#4CAF50'
    else:
        status = 'NEUTRAL'
        status_color = '#888888'

    ax.text(0.99, 0.82, status,
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            color=status_color, verticalalignment='top',
            horizontalalignment='right')

    # Limiti
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlim(df_agg['date'].min() - pd.Timedelta(days=30),
                df_agg['date'].max() + pd.Timedelta(days=30))

    # Legenda
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Hawkish (>0.05)'),
        Patch(facecolor='#4CAF50', label='Dovish (<-0.05)'),
        Patch(facecolor='#666666', label='Neutral'),
        Line2D([0], [0], color='white', linewidth=2, label='Trend MA(3)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False,
              fontsize=10, labelcolor='#888888')

    # Eventi chiave
    events = [
        ('2008-09-01', 'GFC'),
        ('2020-03-01', 'COVID'),
        ('2022-03-01', 'Hikes'),
        ('2024-09-01', 'Cuts'),
    ]

    for date_str, label in events:
        try:
            event_date = pd.to_datetime(date_str)
            if event_date >= df_agg['date'].min() and event_date <= df_agg['date'].max():
                ax.axvline(x=event_date, color='#555555', linewidth=1, linestyle=':')
                ax.text(event_date, 0.52, label, fontsize=9, color='#777777',
                       ha='center', fontweight='bold')
        except:
            continue

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='#000000', edgecolor='none')
    print(f"   OK {output_path.name}")
    plt.close()

    # Salva anche CSV mensile
    monthly_csv = output_path.parent / 'fed_sentiment_monthly.csv'
    df_agg.to_csv(monthly_csv, index=False)
    print(f"   OK fed_sentiment_monthly.csv ({len(df_agg)} mesi)")

    return df_agg


def save_data(df_daily, df_all):
    """Salva tutti i dati."""
    print("\n[6] SALVATAGGIO DATI...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Indice giornaliero
    daily_path = OUTPUT_DIR / f'fed_sentiment_pro_daily_{timestamp}.csv'
    df_daily.to_csv(daily_path, index=False)
    print(f"   OK {daily_path.name}")

    # Dati grezzi
    if df_all is not None and len(df_all) > 0:
        raw_path = OUTPUT_DIR / f'fed_sentiment_pro_raw_{timestamp}.csv'
        df_all.to_csv(raw_path, index=False)
        print(f"   OK {raw_path.name}")

    # JSON summary
    current = df_daily['sentiment_ewma'].iloc[-1]
    summary = {
        'generated_at': timestamp,
        'methodology': 'MNLPFEDS + Loughran-McDonald',
        'total_days': len(df_daily),
        'start_date': df_daily['date'].min().strftime('%Y-%m-%d'),
        'end_date': df_daily['date'].max().strftime('%Y-%m-%d'),
        'current_sentiment': round(current, 4),
        'current_regime': 'Hawkish' if current > 0.15 else ('Dovish' if current < -0.15 else 'Neutral'),
        'ma_30': round(df_daily['ma_30'].iloc[-1], 4),
        'ma_90': round(df_daily['ma_90'].iloc[-1], 4),
        'momentum_30d': round(df_daily['momentum'].iloc[-1], 4) if pd.notna(df_daily['momentum'].iloc[-1]) else 0,
        'volatility': round(df_daily['volatility'].iloc[-1], 4) if pd.notna(df_daily['volatility'].iloc[-1]) else 0,
        'total_documents': len(df_all) if df_all is not None else 0,
        'hawkish_terms_count': len(HAWKISH_TERMS),
        'dovish_terms_count': len(DOVISH_TERMS),
    }

    summary_path = OUTPUT_DIR / f'fed_sentiment_pro_summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   OK {summary_path.name}")

    return summary


def main():
    """Funzione principale."""

    # 1. Carica FOMC Minutes
    df_minutes = load_fomc_minutes()

    # 2. Carica e analizza discorsi
    df_speeches = load_speeches()

    # 3. Combina
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

    # 5. Genera grafici
    plot_bloomberg_style(df_daily, df_all, OUTPUT_DIR / 'fed_sentiment_bloomberg.png')
    plot_detailed_view(df_daily, OUTPUT_DIR / 'fed_sentiment_detailed.png', start_date='2020-01-01')
    plot_monthly_view(df_daily, OUTPUT_DIR / 'fed_sentiment_monthly.png')

    # 6. Salva dati
    summary = save_data(df_daily, df_all)

    # Report finale
    print("\n" + "="*80)
    print("[OK] FED SENTIMENT INDEX PROFESSIONAL - COMPLETATO!")
    print("="*80)

    print(f"""
RIEPILOGO:
  Periodo: {summary['start_date']} -> {summary['end_date']}
  Documenti analizzati: {summary['total_documents']}
  Giorni nell'indice: {summary['total_days']}

SENTIMENT ATTUALE:
  Valore: {summary['current_sentiment']:.4f}
  Regime: {summary['current_regime']}
  MA(30): {summary['ma_30']:.4f}
  MA(90): {summary['ma_90']:.4f}
  Momentum: {summary['momentum_30d']:.4f}

METODOLOGIA:
  - Dizionario: {summary['hawkish_terms_count']} hawkish + {summary['dovish_terms_count']} dovish terms
  - Negation handling: 3-word window
  - Smoothing: EWMA (halflife=30)
  - Weighting: Document type + Speaker importance

FILE GENERATI:
  {OUTPUT_DIR}/
    - fed_sentiment_bloomberg.png (grafico principale)
    - fed_sentiment_detailed.png (vista dettagliata)
    - fed_sentiment_pro_daily_*.csv
    - fed_sentiment_pro_raw_*.csv
    - fed_sentiment_pro_summary_*.json

FONTI METODOLOGIA:
  - Morgan Stanley MNLPFEDS Index
  - Loughran-McDonald Dictionary (Notre Dame)
  - BIS Working Papers
  - Federal Reserve Research
""")


if __name__ == "__main__":
    main()
