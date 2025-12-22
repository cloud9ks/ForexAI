"""
================================================================================
FOREX AI TRADING DASHBOARD v2.0
================================================================================
Dashboard Streamlit per monitorare l'AI Trading Agent in tempo reale.

Uso:
    streamlit run dashboard.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import MetaTrader5 as mt5
import json
import re
import os
from dotenv import load_dotenv

# Load API keys
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# MT5 Configuration
MT5_LOGIN = 3000138
MT5_SERVER = "MetaQuotes-Demo"

# Page config
st.set_page_config(
    page_title="Forex AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark theme professionale ULTRA
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    * { font-family: 'Inter', sans-serif; }
    code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

    /* Main background - deeper dark with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0e14 0%, #0d1117 50%, #161b22 100%);
    }

    /* Hide default header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Metric cards - glassmorphism effect */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(33,38,45,0.9) 0%, rgba(22,27,34,0.9) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(48,54,61,0.8);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.08);
    }

    div[data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar - sleeker design */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #0a0e14 100%);
        border-right: 1px solid rgba(48,54,61,0.5);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9;
    }

    /* Headers with gradient text */
    h1 {
        background: linear-gradient(135deg, #58a6ff 0%, #a371f7 50%, #f778ba 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
    }
    h2, h3 { color: #e6edf3 !important; font-weight: 600 !important; }

    /* Tabs - modern pill style */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(33,38,45,0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 6px;
        gap: 4px;
        border: 1px solid rgba(48,54,61,0.5);
    }

    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(88,166,255,0.1);
        color: #c9d1d9;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(88,166,255,0.2) 0%, rgba(163,113,247,0.2) 100%) !important;
        color: #58a6ff !important;
        border: 1px solid rgba(88,166,255,0.3);
        box-shadow: 0 0 20px rgba(88,166,255,0.2);
    }

    /* Dataframes */
    .stDataFrame {
        border: 1px solid rgba(48,54,61,0.5);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Info/Alert boxes */
    .stAlert {
        background: rgba(33,38,45,0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(48,54,61,0.5);
        border-radius: 12px;
        color: #c9d1d9;
    }

    /* Buttons - gradient with glow */
    .stButton button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 50%, #3fb950 100%);
        color: #fff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 12px 24px;
        box-shadow: 0 4px 15px rgba(46,160,67,0.3);
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 50%, #56d364 100%);
        box-shadow: 0 6px 25px rgba(46,160,67,0.4);
        transform: translateY(-1px);
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(33,38,45,0.8);
        border: 1px solid rgba(48,54,61,0.5);
        border-radius: 10px;
        color: #c9d1d9;
    }

    /* Text colors */
    p, span, label { color: #c9d1d9; }
    hr { border-color: rgba(48,54,61,0.5); margin: 1.5rem 0; }

    /* Custom header banner */
    .header-banner {
        background: linear-gradient(135deg, rgba(88,166,255,0.1) 0%, rgba(163,113,247,0.1) 50%, rgba(247,120,186,0.1) 100%);
        border: 1px solid rgba(88,166,255,0.2);
        border-radius: 20px;
        padding: 25px 30px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .header-title {
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .header-title h1 {
        margin: 0;
        font-size: 2rem;
    }

    .header-logo {
        font-size: 2.5rem;
        filter: drop-shadow(0 0 10px rgba(88,166,255,0.5));
    }

    /* Live indicator - enhanced */
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(63,185,80,0.2) 0%, rgba(63,185,80,0.1) 100%);
        border: 1px solid rgba(63,185,80,0.3);
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #3fb950;
    }

    .live-dot {
        width: 10px;
        height: 10px;
        background: #3fb950;
        border-radius: 50%;
        box-shadow: 0 0 10px #3fb950, 0 0 20px rgba(63,185,80,0.5);
        animation: pulse-glow 2s infinite;
    }

    @keyframes pulse-glow {
        0%, 100% { opacity: 1; box-shadow: 0 0 10px #3fb950, 0 0 20px rgba(63,185,80,0.5); }
        50% { opacity: 0.7; box-shadow: 0 0 5px #3fb950, 0 0 10px rgba(63,185,80,0.3); }
    }

    /* Signal cards - enhanced */
    .signal-card {
        background: linear-gradient(135deg, rgba(33,38,45,0.9) 0%, rgba(22,27,34,0.9) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border-left: 5px solid;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }

    .signal-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }

    .signal-card.buy {
        border-left-color: #3fb950;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 0 30px rgba(63,185,80,0.05);
    }

    .signal-card.sell {
        border-left-color: #f85149;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 0 30px rgba(248,81,73,0.05);
    }

    .signal-card.hold {
        border-left-color: #8b949e;
    }

    .signal-pair {
        color: #8b949e;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .signal-direction {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .signal-confidence {
        font-size: 1.1rem;
        color: #c9d1d9;
        margin-bottom: 10px;
    }

    .signal-decision {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .decision-trade {
        background: linear-gradient(135deg, rgba(63,185,80,0.2) 0%, rgba(63,185,80,0.1) 100%);
        color: #3fb950;
        border: 1px solid rgba(63,185,80,0.3);
    }

    .decision-hold {
        background: rgba(139,148,158,0.1);
        color: #8b949e;
        border: 1px solid rgba(139,148,158,0.3);
    }

    /* Agent status card */
    .agent-status {
        background: linear-gradient(135deg, rgba(33,38,45,0.9) 0%, rgba(22,27,34,0.9) 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(48,54,61,0.5);
        margin: 15px 0;
    }

    .agent-online {
        border-left: 4px solid #3fb950;
    }

    .agent-offline {
        border-left: 4px solid #f85149;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0d1117;
    }

    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #484f58;
        font-size: 0.8rem;
        padding: 20px;
        margin-top: 30px;
        border-top: 1px solid rgba(48,54,61,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MT5 CONNECTION
# ============================================================================
@st.cache_resource
def init_mt5():
    # First try to initialize (will use currently logged in account)
    if not mt5.initialize():
        return None

    # Check if we're on the correct account
    account = mt5.account_info()
    if account and account.login != MT5_LOGIN:
        # Different account, try to switch (note: may need password)
        mt5.shutdown()
        if not mt5.initialize():
            return None
        account = mt5.account_info()

    return account


def get_account_info():
    account = mt5.account_info()
    if account:
        return {
            'login': account.login,
            'server': account.server,
            'balance': account.balance,
            'equity': account.equity,
            'profit': account.profit,
            'margin': account.margin,
            'margin_free': account.margin_free,
            'margin_level': account.margin_level if account.margin > 0 else 0,
        }
    return None


def get_open_positions():
    positions = mt5.positions_get()
    if positions:
        return pd.DataFrame([{
            'Ticket': p.ticket,
            'Symbol': p.symbol,
            'Type': '🟢 BUY' if p.type == 0 else '🔴 SELL',
            'Volume': p.volume,
            'Open Price': p.price_open,
            'Current': p.price_current,
            'SL': p.sl,
            'TP': p.tp,
            'Profit': p.profit,
            'Time': datetime.fromtimestamp(p.time).strftime('%H:%M:%S'),
        } for p in positions])
    return pd.DataFrame()


def get_trade_history(days=30):
    from_date = datetime.now() - timedelta(days=days)
    to_date = datetime.now()
    deals = mt5.history_deals_get(from_date, to_date)
    if deals:
        df = pd.DataFrame([{
            'Time': datetime.fromtimestamp(d.time),
            'Symbol': d.symbol,
            'Type': 'BUY' if d.type == 0 else 'SELL',
            'Volume': d.volume,
            'Price': d.price,
            'Profit': d.profit,
        } for d in deals if d.symbol])
        return df
    return pd.DataFrame()


def get_market_data(symbol, timeframe=mt5.TIMEFRAME_H1, bars=200):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is not None:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    return None


def calculate_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_middle'] = sma20

    # Moving Averages
    df['sma_20'] = sma20
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    return df


def read_agent_log():
    log_path = Path(__file__).parent / "agent_trading.log"
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.readlines()[-200:]
    return []


def parse_last_analysis(log_lines):
    """Estrae l'ultima analisi per ogni coppia dal log."""
    analyses = {}
    current_pair = None

    for line in log_lines:
        if 'ANALYZING:' in line:
            match = re.search(r'ANALYZING:\s+(\w+)', line)
            if match:
                current_pair = match.group(1)
        elif 'LSTM:' in line and current_pair:
            # LSTM: BUY (42.1%) | ATR: 0.00175
            match = re.search(r'LSTM:\s+(\w+)\s+\((\d+\.?\d*)%\)', line)
            if match:
                analyses[current_pair] = {
                    'signal': match.group(1),  # BUY/SELL
                    'confidence': float(match.group(2)),  # 42.1
                    'decision': 'HOLD',  # default
                }
        elif 'DECISION:' in line and current_pair and current_pair in analyses:
            match = re.search(r'DECISION:\s+(\w+)', line)
            if match:
                analyses[current_pair]['decision'] = match.group(1)

    return analyses


# ============================================================================
# AI CHAT FUNCTIONS
# ============================================================================
def get_multi_timeframe_analysis(symbol):
    """Analizza una coppia su più timeframe."""
    timeframes = {
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }

    analysis = {}
    for tf_name, tf in timeframes.items():
        df = get_market_data(symbol, tf, 100)
        if df is not None:
            df = calculate_indicators(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]

            # Trend
            trend = "UP" if last['sma_20'] > last['sma_50'] else "DOWN"

            # RSI
            rsi = last['rsi']
            rsi_signal = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"

            # MACD
            macd_signal = "BULLISH" if last['macd'] > last['macd_signal'] else "BEARISH"

            # BB position
            bb_pos = (last['close'] - last['bb_lower']) / (last['bb_upper'] - last['bb_lower']) * 100

            # Price change
            change = (last['close'] - prev['close']) / prev['close'] * 100

            analysis[tf_name] = {
                'price': last['close'],
                'change': change,
                'trend': trend,
                'rsi': rsi,
                'rsi_signal': rsi_signal,
                'macd_signal': macd_signal,
                'atr': last['atr'],
                'bb_position': bb_pos,
                'sma_20': last['sma_20'],
                'sma_50': last['sma_50'],
            }

    return analysis


def get_economic_calendar():
    """Ottiene il calendario economico da Investing.com via scraping."""
    try:
        import requests
        from bs4 import BeautifulSoup

        # Prova a ottenere eventi da ForexFactory (più semplice)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # Usa un'API gratuita alternativa
        today = datetime.now().strftime('%Y-%m-%d')
        events = []

        # Eventi importanti hardcoded per le prossime ore (fallback)
        # In produzione si userebbe un'API come FXStreet, ForexFactory, etc.

        # Prova Forex Factory calendar
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                today_str = datetime.now().strftime('%Y-%m-%d')
                tomorrow_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

                for event in data:
                    event_date = event.get('date', '')[:10]
                    if event_date in [today_str, tomorrow_str]:
                        impact = event.get('impact', 'Low')
                        if impact in ['High', 'Medium']:
                            events.append({
                                'time': event.get('date', '')[11:16],
                                'currency': event.get('country', ''),
                                'event': event.get('title', ''),
                                'impact': impact,
                                'forecast': event.get('forecast', ''),
                                'previous': event.get('previous', ''),
                            })
        except:
            pass

        return events[:10]  # Max 10 eventi

    except Exception as e:
        return []


def get_currency_news(currency):
    """Ottiene news recenti per una valuta."""
    try:
        import requests

        # Usa un feed RSS gratuito o API
        news = []

        # Mappa valuta -> keywords
        currency_map = {
            'USD': ['dollar', 'fed', 'us economy', 'fomc'],
            'EUR': ['euro', 'ecb', 'eurozone'],
            'GBP': ['pound', 'sterling', 'boe', 'uk economy'],
            'JPY': ['yen', 'boj', 'japan economy'],
            'AUD': ['aussie', 'rba', 'australia'],
            'CHF': ['swiss', 'snb'],
            'CAD': ['loonie', 'boc', 'canada'],
            'NZD': ['kiwi', 'rbnz', 'new zealand'],
        }

        # Prova Google News RSS (gratuito)
        keywords = currency_map.get(currency, [currency.lower()])
        query = '+'.join(keywords[:2]) + '+forex'

        try:
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')[:5]

                for item in items:
                    title = item.find('title')
                    pub_date = item.find('pubDate')
                    if title:
                        news.append({
                            'title': title.text[:100],
                            'date': pub_date.text[:16] if pub_date else '',
                        })
        except:
            pass

        return news

    except Exception as e:
        return []


def ask_ai_agent(question, symbol, analysis_data):
    """Chiedi all'agente AI la sua opinione."""
    if not OPENAI_API_KEY:
        return "❌ OpenAI API key non configurata. Aggiungi OPENAI_API_KEY nel file .env"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Estrai le valute dalla coppia
        base_currency = symbol[:3]  # es. EUR da EURUSD
        quote_currency = symbol[3:]  # es. USD da EURUSD

        # Prepara il contesto con i dati di mercato
        context = f"""Sei un esperto analista forex AI. Analizza {symbol} e rispondi alla domanda dell'utente.
Data e ora corrente: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATI DI MERCATO ATTUALI PER {symbol}:
"""
        for tf, data in analysis_data.items():
            context += f"""
[{tf}]
- Prezzo: {data['price']:.5f} ({data['change']:+.2f}%)
- Trend: {data['trend']} (SMA20: {data['sma_20']:.5f}, SMA50: {data['sma_50']:.5f})
- RSI: {data['rsi']:.1f} ({data['rsi_signal']})
- MACD: {data['macd_signal']}
- ATR: {data['atr']:.5f}
- BB Position: {data['bb_position']:.1f}%
"""

        # Aggiungi ultimo segnale LSTM se disponibile
        log_lines = read_agent_log()
        analyses = parse_last_analysis(log_lines)
        if symbol in analyses:
            lstm_data = analyses[symbol]
            context += f"""
ULTIMO SEGNALE LSTM (dal modello AI):
- Direzione: {lstm_data.get('signal', 'N/A')}
- Confidenza: {lstm_data.get('confidence', 0):.1f}%
- Decisione: {lstm_data.get('decision', 'N/A')}
"""

        # Aggiungi calendario economico
        calendar = get_economic_calendar()
        if calendar:
            context += f"""
CALENDARIO ECONOMICO (Eventi oggi/domani ad alto impatto):
"""
            for event in calendar:
                relevant = event['currency'] in [base_currency, quote_currency, 'USD', 'EUR']
                if relevant or event['impact'] == 'High':
                    context += f"- [{event['time']}] {event['currency']}: {event['event']} (Impatto: {event['impact']})"
                    if event['forecast']:
                        context += f" | Prev: {event['forecast']}"
                    if event['previous']:
                        context += f" | Prec: {event['previous']}"
                    context += "\n"
        else:
            context += """
CALENDARIO ECONOMICO: Nessun evento ad alto impatto rilevato per oggi/domani.
"""

        # Aggiungi news recenti
        news_base = get_currency_news(base_currency)
        news_quote = get_currency_news(quote_currency)

        if news_base or news_quote:
            context += f"""
NEWS RECENTI:
"""
            if news_base:
                context += f"[{base_currency}]\n"
                for n in news_base[:3]:
                    context += f"- {n['title']}\n"
            if news_quote:
                context += f"[{quote_currency}]\n"
                for n in news_quote[:3]:
                    context += f"- {n['title']}\n"

        context += """
REGOLE DI TRADING:
- Min confidenza LSTM: 70%
- TP: 3.0 ATR, SL: 1.5 ATR (ratio 2:1)
- Risk per trade: 0.65%
- Sessioni attive: London/NY (8-20 UTC)

HAI ACCESSO A DATI IN TEMPO REALE. Usa questi dati per rispondere. Rispondi in modo conciso ma completo.
Se dai un'opinione di trading, specifica sempre il livello di rischio.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Errore: {str(e)}"


# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================
def main():
    # Header banner moderno
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">
            <span class="header-logo">🤖</span>
            <div>
                <h1 style="margin:0; font-size:1.8rem;">FOREX AI TRADING</h1>
                <p style="margin:0; color:#8b949e; font-size:0.9rem;">Powered by LSTM + GPT-5.1 + DXY Model</p>
            </div>
        </div>
        <div class="live-badge">
            <span class="live-dot"></span>
            LIVE TRADING
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize MT5
    account = init_mt5()
    if account is None:
        st.error("❌ MetaTrader 5 non connesso. Apri MT5 e riprova.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Impostazioni")

        # Manual refresh button
        if st.button("🔄 Aggiorna Dati", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

        st.markdown("")
        selected_pair = st.selectbox(
            "Coppia da analizzare",
            ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD', 'EURGBP']
        )

        timeframe_options = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        selected_tf = st.selectbox("Timeframe", list(timeframe_options.keys()), index=3)

        st.markdown("---")

        # Agent Status - Enhanced
        st.markdown("### 🤖 Trading Agent")
        log_lines = read_agent_log()
        if log_lines:
            last_cycle = None
            for line in reversed(log_lines):
                if 'TRADING CYCLE' in line:
                    match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if match:
                        last_cycle = match.group(1)
                    break

            st.markdown(f"""
            <div class="agent-status agent-online">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                    <div style="width:12px; height:12px; background:#3fb950; border-radius:50%; box-shadow:0 0 10px #3fb950;"></div>
                    <span style="color:#3fb950; font-weight:600; font-size:1rem;">ONLINE</span>
                </div>
                <div style="color:#8b949e; font-size:0.8rem;">
                    <div style="margin-bottom:5px;">📅 Last: {last_cycle or 'N/A'}</div>
                    <div style="margin-bottom:5px;">🎯 Mode: <span style="color:#58a6ff;">DEMO</span></div>
                    <div style="margin-bottom:5px;">📊 Pairs: 4</div>
                    <div>🔒 Min Conf: <span style="color:#f0883e;">66%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="agent-status agent-offline">
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:12px; height:12px; background:#f85149; border-radius:50%;"></div>
                    <span style="color:#f85149; font-weight:600;">OFFLINE</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Next cycle countdown
        st.markdown("### ⏱️ Prossimo Ciclo")
        now = datetime.now()
        next_hour = now.replace(minute=5, second=0, microsecond=0)
        if now.minute >= 5:
            next_hour += timedelta(hours=1)
        time_left = next_hour - now
        mins_left = int(time_left.total_seconds() // 60)
        st.info(f"**{next_hour.strftime('%H:%M')}** ({mins_left} min)")

    # Main content
    acc_info = get_account_info()

    if acc_info:
        # Account metrics row
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("💰 Balance", f"${acc_info['balance']:,.0f}")
        with col2:
            st.metric("📊 Equity", f"${acc_info['equity']:,.0f}")
        with col3:
            profit = acc_info['profit']
            st.metric("💵 P/L", f"${profit:,.2f}",
                     delta=f"{profit:+.2f}" if profit != 0 else None,
                     delta_color="normal" if profit >= 0 else "inverse")
        with col4:
            st.metric("🔒 Margin", f"${acc_info['margin']:,.0f}")
        with col5:
            st.metric("💳 Free", f"${acc_info['margin_free']:,.0f}")
        with col6:
            ml = acc_info['margin_level']
            st.metric("📏 Level", f"{ml:.0f}%" if ml > 0 else "∞")

    st.markdown("")

    # Last AI Signals
    log_lines = read_agent_log()
    analyses = parse_last_analysis(log_lines)

    if analyses:
        st.markdown("### 🎯 AI Signals Dashboard")
        cols = st.columns(4)
        for i, (pair, data) in enumerate(analyses.items()):
            with cols[i % 4]:
                signal = data.get('signal', 'N/A')
                conf = data.get('confidence', 0)
                decision = data.get('decision', 'HOLD')

                # Determina colori e classe
                if signal == 'BUY':
                    color = '#3fb950'
                    card_class = 'buy'
                    icon = '📈'
                elif signal == 'SELL':
                    color = '#f85149'
                    card_class = 'sell'
                    icon = '📉'
                else:
                    color = '#8b949e'
                    card_class = 'hold'
                    icon = '⏸️'

                decision_class = 'decision-trade' if decision in ['BUY', 'SELL'] else 'decision-hold'
                conf_bar_width = min(conf, 100)
                conf_color = '#3fb950' if conf >= 66 else '#f0883e' if conf >= 50 else '#f85149'

                st.markdown(f"""
                <div class="signal-card {card_class}">
                    <div class="signal-pair">{icon} {pair}</div>
                    <div class="signal-direction" style="color:{color};">{signal}</div>
                    <div class="signal-confidence">
                        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                            <span>Confidence</span>
                            <span style="color:{conf_color}; font-weight:600;">{conf:.1f}%</span>
                        </div>
                        <div style="background:rgba(48,54,61,0.5); border-radius:10px; height:8px; overflow:hidden;">
                            <div style="background:linear-gradient(90deg, {conf_color}, {conf_color}aa); width:{conf_bar_width}%; height:100%; border-radius:10px;"></div>
                        </div>
                    </div>
                    <div style="margin-top:12px;">
                        <span class="signal-decision {decision_class}">→ {decision}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Chart", "📊 Positions", "📜 History", "📝 Log", "⚡ Multi-Chart", "🤖 AI Chat"])

    with tab1:
        st.markdown(f"### {selected_pair} - {selected_tf}")

        df = get_market_data(selected_pair, timeframe_options[selected_tf], 200)

        if df is not None:
            df = calculate_indicators(df)

            # Main chart
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=['', 'Volume', 'RSI', 'MACD']
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df['time'], open=df['open'], high=df['high'],
                low=df['low'], close=df['close'], name='Price',
                increasing_line_color='#3fb950', decreasing_line_color='#f85149',
                increasing_fillcolor='#3fb950', decreasing_fillcolor='#f85149',
            ), row=1, col=1)

            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df['time'], y=df['bb_upper'], name='BB',
                line=dict(color='rgba(88,166,255,0.3)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['bb_lower'], showlegend=False,
                line=dict(color='rgba(88,166,255,0.3)', width=1),
                fill='tonexty', fillcolor='rgba(88,166,255,0.05)'), row=1, col=1)

            # Moving Averages
            fig.add_trace(go.Scatter(x=df['time'], y=df['sma_20'], name='SMA20',
                line=dict(color='#f0883e', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['sma_50'], name='SMA50',
                line=dict(color='#a371f7', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['ema_9'], name='EMA9',
                line=dict(color='#39d353', width=1)), row=1, col=1)

            # Volume
            colors = ['#3fb950' if c >= o else '#f85149' for c, o in zip(df['close'], df['open'])]
            fig.add_trace(go.Bar(x=df['time'], y=df['tick_volume'], name='Volume',
                marker_color=colors, opacity=0.7), row=2, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], name='RSI',
                line=dict(color='#58a6ff', width=1.5)), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#f85149", line_width=1, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#3fb950", line_width=1, row=3, col=1)
            fig.add_hrect(y0=30, y1=70, fillcolor="rgba(88,166,255,0.05)", line_width=0, row=3, col=1)

            # MACD
            fig.add_trace(go.Scatter(x=df['time'], y=df['macd'], name='MACD',
                line=dict(color='#58a6ff', width=1.5)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df['time'], y=df['macd_signal'], name='Signal',
                line=dict(color='#f0883e', width=1.5)), row=4, col=1)
            hist_colors = ['#3fb950' if v >= 0 else '#f85149' for v in df['macd_hist']]
            fig.add_trace(go.Bar(x=df['time'], y=df['macd_hist'], name='Hist',
                marker_color=hist_colors, opacity=0.7), row=4, col=1)

            fig.update_layout(
                height=800,
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117',
                font=dict(color='#c9d1d9'),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e', size=10)),
                xaxis_rangeslider_visible=False,
                margin=dict(l=50, r=20, t=30, b=20),
            )

            for i in range(1, 5):
                fig.update_xaxes(gridcolor='#21262d', zerolinecolor='#30363d', row=i, col=1)
                fig.update_yaxes(gridcolor='#21262d', zerolinecolor='#30363d', row=i, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Current indicators
            st.markdown("### 📊 Indicatori Attuali")
            c1, c2, c3, c4, c5, c6 = st.columns(6)

            with c1:
                price = df['close'].iloc[-1]
                change = (price - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
                st.metric("Price", f"{price:.5f}", f"{change:+.2f}%")
            with c2:
                rsi = df['rsi'].iloc[-1]
                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                st.metric("RSI", f"{rsi:.1f}", rsi_status)
            with c3:
                macd = df['macd'].iloc[-1]
                macd_sig = df['macd_signal'].iloc[-1]
                st.metric("MACD", f"{macd:.5f}", "Bullish" if macd > macd_sig else "Bearish")
            with c4:
                atr = df['atr'].iloc[-1]
                st.metric("ATR", f"{atr:.5f}")
            with c5:
                trend = "Uptrend" if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] else "Downtrend"
                st.metric("Trend", trend)
            with c6:
                bb_pos = (price - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) * 100
                st.metric("BB %", f"{bb_pos:.0f}%")

    with tab2:
        st.markdown("### 📊 Posizioni Aperte")
        positions = get_open_positions()

        if not positions.empty:
            total_profit = positions['Profit'].sum()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Posizioni", len(positions))
            with c2:
                st.metric("Profit Totale", f"${total_profit:,.2f}",
                         delta="Profit" if total_profit > 0 else "Loss")
            with c3:
                st.metric("Volume Totale", f"{positions['Volume'].sum():.2f} lots")
            with c4:
                buy_count = len(positions[positions['Type'].str.contains('BUY')])
                sell_count = len(positions) - buy_count
                st.metric("Buy/Sell", f"{buy_count}/{sell_count}")

            st.dataframe(positions, use_container_width=True, hide_index=True)
        else:
            st.info("📭 Nessuna posizione aperta")

    with tab3:
        st.markdown("### 📜 Trade History")
        history = get_trade_history(30)

        if not history.empty:
            total = len(history)
            wins = len(history[history['Profit'] > 0])
            losses = len(history[history['Profit'] < 0])
            total_pl = history['Profit'].sum()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Trades", total)
            with c2:
                st.metric("Win Rate", f"{wins/total*100:.1f}%" if total > 0 else "N/A")
            with c3:
                st.metric("W/L", f"{wins}/{losses}")
            with c4:
                st.metric("Total P/L", f"${total_pl:,.2f}")

            # Equity curve
            history_sorted = history.sort_values('Time')
            history_sorted['Cumulative'] = history_sorted['Profit'].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_sorted['Time'], y=history_sorted['Cumulative'],
                fill='tozeroy', fillcolor='rgba(63,185,80,0.2)',
                line=dict(color='#3fb950', width=2), name='Equity'
            ))
            fig.update_layout(
                title="Equity Curve", height=300,
                plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
                font=dict(color='#c9d1d9'),
            )
            fig.update_xaxes(gridcolor='#21262d')
            fig.update_yaxes(gridcolor='#21262d')
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(history.sort_values('Time', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("📭 No trades in history")

    with tab4:
        st.markdown("### 📝 Agent Log")
        log_lines = read_agent_log()

        if log_lines:
            c1, c2 = st.columns([3, 1])
            with c1:
                filter_type = st.selectbox("Filter", ["All", "INFO", "WARNING", "ERROR", "ANALYZING", "DECISION"])
            with c2:
                st.download_button("📥 Download", "".join(log_lines),
                    file_name=f"agent_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")

            if filter_type != "All":
                log_lines = [l for l in log_lines if filter_type in l]

            log_text = "".join(log_lines[-100:])
            st.code(log_text, language="log")
        else:
            st.info("📭 No log available")

    with tab5:
        st.markdown("### ⚡ Multi-Chart View")
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

        col1, col2 = st.columns(2)

        for i, pair in enumerate(pairs):
            with col1 if i % 2 == 0 else col2:
                df = get_market_data(pair, mt5.TIMEFRAME_H1, 50)
                if df is not None:
                    fig = go.Figure(go.Candlestick(
                        x=df['time'], open=df['open'], high=df['high'],
                        low=df['low'], close=df['close'],
                        increasing_line_color='#3fb950', decreasing_line_color='#f85149',
                        increasing_fillcolor='#3fb950', decreasing_fillcolor='#f85149',
                    ))

                    change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
                    title_color = '#3fb950' if change >= 0 else '#f85149'

                    fig.update_layout(
                        title=dict(text=f"{pair} ({change:+.2f}%)", font=dict(color=title_color)),
                        height=250,
                        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
                        font=dict(color='#c9d1d9'),
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=40, r=20, t=40, b=20),
                        showlegend=False
                    )
                    fig.update_xaxes(gridcolor='#21262d', showticklabels=False)
                    fig.update_yaxes(gridcolor='#21262d')
                    st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.markdown("### 🤖 AI Trading Assistant")
        st.markdown("Chiedi all'agente AI la sua opinione su qualsiasi coppia e timeframe.")

        # Selezione coppia per chat
        col_pair, col_btn = st.columns([3, 1])
        with col_pair:
            chat_pair = st.selectbox(
                "Seleziona coppia da analizzare",
                ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD', 'EURGBP'],
                key="chat_pair"
            )

        # Quick analysis buttons
        st.markdown("#### ⚡ Analisi Rapida")
        quick_cols = st.columns(4)

        with quick_cols[0]:
            if st.button("📊 Analisi Completa", use_container_width=True):
                st.session_state['quick_question'] = f"Fammi un'analisi completa di {chat_pair} su tutti i timeframe. Qual è la tua view e perché?"

        with quick_cols[1]:
            if st.button("🎯 Segnale Trading", use_container_width=True):
                st.session_state['quick_question'] = f"Dovrei entrare long o short su {chat_pair} adesso? Dammi entry, SL e TP."

        with quick_cols[2]:
            if st.button("📈 Trend Analysis", use_container_width=True):
                st.session_state['quick_question'] = f"Qual è il trend principale su {chat_pair}? È in fase di inversione o continuazione?"

        with quick_cols[3]:
            if st.button("⚠️ Risk Check", use_container_width=True):
                st.session_state['quick_question'] = f"Ci sono rischi particolari nel tradare {chat_pair} in questo momento? News, eventi, livelli chiave?"

        st.markdown("---")

        # Chat input
        st.markdown("#### 💬 Chiedi all'Agente")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Get quick question if set
        default_question = st.session_state.pop('quick_question', '')

        user_question = st.text_area(
            "La tua domanda:",
            value=default_question,
            height=80,
            placeholder=f"Es: Cosa ne pensi di {chat_pair}? Dovrei comprare o vendere?"
        )

        col_send, col_clear = st.columns([1, 1])
        with col_send:
            send_btn = st.button("📤 Invia", use_container_width=True, type="primary")
        with col_clear:
            if st.button("🗑️ Pulisci Chat", use_container_width=True):
                st.session_state['chat_history'] = []
                st.rerun()

        if send_btn and user_question:
            with st.spinner(f"🔄 Analizzo {chat_pair}..."):
                # Get multi-timeframe analysis
                analysis_data = get_multi_timeframe_analysis(chat_pair)

                if analysis_data:
                    # Ask AI
                    response = ask_ai_agent(user_question, chat_pair, analysis_data)

                    # Add to history
                    st.session_state['chat_history'].append({
                        'pair': chat_pair,
                        'question': user_question,
                        'answer': response,
                        'time': datetime.now().strftime('%H:%M:%S')
                    })
                else:
                    st.error(f"❌ Impossibile ottenere dati per {chat_pair}")

        # Display chat history
        if st.session_state.get('chat_history'):
            st.markdown("---")
            st.markdown("#### 📜 Conversazione")

            for i, msg in enumerate(reversed(st.session_state['chat_history'])):
                with st.container():
                    # User message
                    st.markdown(f"""
                    <div style="background:#21262d; border-radius:10px; padding:12px; margin-bottom:10px; border-left:3px solid #58a6ff;">
                        <div style="color:#58a6ff; font-size:0.8rem; margin-bottom:5px;">
                            🧑 Tu • {msg['pair']} • {msg['time']}
                        </div>
                        <div style="color:#c9d1d9;">{msg['question']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # AI response
                    st.markdown(f"""
                    <div style="background:#161b22; border-radius:10px; padding:12px; margin-bottom:20px; border-left:3px solid #3fb950;">
                        <div style="color:#3fb950; font-size:0.8rem; margin-bottom:5px;">
                            🤖 AI Agent
                        </div>
                        <div style="color:#c9d1d9; white-space:pre-wrap;">{msg['answer']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Multi-TF summary
        st.markdown("---")
        st.markdown("#### 📊 Riepilogo Multi-Timeframe")

        mtf_data = get_multi_timeframe_analysis(chat_pair)
        if mtf_data:
            tf_cols = st.columns(4)
            for i, (tf, data) in enumerate(mtf_data.items()):
                with tf_cols[i]:
                    trend_color = '#3fb950' if data['trend'] == 'UP' else '#f85149'
                    rsi_color = '#f85149' if data['rsi_signal'] == 'OVERBOUGHT' else '#3fb950' if data['rsi_signal'] == 'OVERSOLD' else '#8b949e'
                    macd_color = '#3fb950' if data['macd_signal'] == 'BULLISH' else '#f85149'

                    st.markdown(f"""
                    <div style="background:#21262d; border-radius:10px; padding:15px; text-align:center;">
                        <div style="color:#8b949e; font-size:0.9rem; font-weight:bold;">{tf}</div>
                        <div style="color:{trend_color}; font-size:1.2rem; font-weight:bold; margin:8px 0;">{data['trend']}</div>
                        <div style="color:#c9d1d9; font-size:0.8rem;">RSI: <span style="color:{rsi_color}">{data['rsi']:.0f}</span></div>
                        <div style="color:#c9d1d9; font-size:0.8rem;">MACD: <span style="color:{macd_color}">{data['macd_signal']}</span></div>
                        <div style="color:#c9d1d9; font-size:0.8rem;">BB: {data['bb_position']:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    if acc_info:
        st.markdown(f"""
        <div class="footer">
            <div style="display:flex; justify-content:center; align-items:center; gap:20px; flex-wrap:wrap;">
                <span>🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                <span>|</span>
                <span>👤 Account: <strong>{acc_info['login']}</strong></span>
                <span>|</span>
                <span>🖥️ {acc_info['server']}</span>
                <span>|</span>
                <span style="color:#3fb950;">● Connected</span>
            </div>
            <div style="margin-top:10px; color:#30363d;">
                Forex AI Trading System v2.0 • LSTM + GPT-5.1 + DXY Model
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="footer">
            <div style="display:flex; justify-content:center; align-items:center; gap:20px;">
                <span>🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                <span>|</span>
                <span style="color:#f85149;">● MT5 Disconnected</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Auto refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
