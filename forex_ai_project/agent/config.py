"""
================================================================================
AI TRADING AGENT - CONFIGURATION
================================================================================
"""

# API Keys (da settare come environment variables)
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # newsapi.org
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Optional fallback

# Trading Configuration
TRADING_CONFIG = {
    # Pairs da tradare
    'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],

    # Risk Management
    'risk_per_trade': 0.0065,       # 0.65% per trade (ottimizzato per DD < 15%)
    'max_daily_risk': 0.03,         # 3% max rischio giornaliero
    'max_trades_per_day': 2,
    'max_open_positions': 3,
    'max_lot_size': 5.0,            # Max lotti per trade (allineato con backtest)

    # Entry Rules
    'min_lstm_confidence': 0.66,    # Confidenza LSTM minima (66% per più trade)
    'min_agent_confidence': 0.60,   # Confidenza Agent minima

    # Position Sizing
    'take_profit_atr': 3.0,
    'stop_loss_atr': 1.5,

    # Session Filter (UTC)
    'active_hours': (8, 20),        # London/NY sessions

    # Cooldown
    'min_hours_between_trades': 72,  # 3 giorni per coppia
}

# CVOL Configuration (Currency Volatility Warning System)
CVOL_CONFIG = {
    # Soglie di default (verranno aggiornate dinamicamente se disponibili dati)
    'default_warning_level': 13.0,   # CVOL sopra questo → WARNING (risk -50%)
    'default_extreme_level': 19.0,   # CVOL sopra questo → SKIP trade

    # Risk adjustment
    'warning_risk_factor': 0.5,      # 50% del risk normale in WARNING
    'extreme_risk_factor': 0.0,      # 0% = no trade in EXTREME

    # Cache
    'cache_duration_hours': 1,       # Ricalcola CVOL ogni ora
}

# Agent Configuration
AGENT_CONFIG = {
    'model': 'gpt-5.1',              # Modello GPT-5.1
    'reasoning_effort': 'medium',    # low | medium | high
}

# Macro Events Configuration
MACRO_CONFIG = {
    # Eventi ad alto impatto da evitare
    'high_impact_events': [
        'Non-Farm Payrolls',
        'FOMC',
        'Interest Rate Decision',
        'CPI',
        'GDP',
        'ECB Press Conference',
        'BOE Interest Rate',
        'BOJ Policy Rate',
    ],

    # Buffer prima/dopo eventi (minuti)
    'pre_event_buffer': 60,
    'post_event_buffer': 30,

    # Valute monitorate
    'currencies': ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CHF', 'CAD', 'NZD'],
}

# News Sentiment Configuration
NEWS_CONFIG = {
    # Keywords per forex news
    'forex_keywords': [
        'forex', 'currency', 'dollar', 'euro', 'pound', 'yen',
        'fed', 'ecb', 'boe', 'boj', 'central bank',
        'interest rate', 'inflation', 'gdp', 'employment',
    ],

    # Sentiment thresholds
    'bullish_threshold': 0.3,
    'bearish_threshold': -0.3,

    # News freshness (ore)
    'max_news_age_hours': 24,
}

# MetaTrader 5 Configuration
MT5_CONFIG = {
    'login': 3000138,
    'password': os.getenv("MT5_PASSWORD", ""),
    'server': "MetaQuotes-Demo",
}

# Logging
LOGGING_CONFIG = {
    'log_file': 'agent_trading.log',
    'log_level': 'INFO',
    'log_trades': True,
    'log_decisions': True,
}
