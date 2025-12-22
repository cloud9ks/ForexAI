# AI FOREX TRADING AGENT - COMPLETE SPECIFICATIONS

**Version:** 3.0
**Last Updated:** December 2025
**Author:** Alessandro + Claude AI

---

## OVERVIEW

Sistema di trading forex automatizzato che combina:
- **Machine Learning** (LSTM neural network)
- **Macro Analysis** (Multi-timeframe: trimestrale + mensile)
- **Volatility Monitoring** (CVOL Warning System)
- **AI Reasoning** (GPT-5.1 decision making)

---

## ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI TRADING AGENT v3.0 - SYSTEM ARCHITECTURE              │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   MT5 API    │
                              │  (Broker)    │
                              └──────┬───────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Market Data (H1)    │
                         │   OHLCV + Features    │
                         └───────────┬───────────┘
                                     │
    ┌────────────────────────────────┼────────────────────────────────┐
    │                                │                                │
    ▼                                ▼                                ▼
┌──────────────┐            ┌──────────────┐             ┌──────────────┐
│   CVOL       │            │   LSTM       │             │   News       │
│   Monitor    │            │   Model      │             │   Sentiment  │
│              │            │   (88 feat)  │             │   (NewsAPI)  │
│  Volatility  │            │  BUY/SELL/   │             │  bullish/    │
│  Warning     │            │  HOLD + Conf │             │  bearish     │
└──────┬───────┘            └──────┬───────┘             └──────┬───────┘
       │                           │                            │
       │    ┌──────────────────────┴────────────────────────────┘
       │    │
       │    │     ┌─────────────────────────────────────────────────────┐
       │    │     │         MACRO MODELS (Multi-Timeframe)              │
       │    │     ├─────────────────────────────────────────────────────┤
       │    │     │                                                     │
       │    │     │   DXY Balance Sheet        Fed Sentiment NLP        │
       │    │     │   (QUARTERLY)              (MONTHLY)                │
       │    │     │   ┌─────────────┐         ┌─────────────┐          │
       │    │     │   │ FED vs ECB  │         │ FOMC Minutes│          │
       │    │     │   │ QT/QE Diff  │         │ + Speeches  │          │
       │    │     │   │             │         │             │          │
       │    │     │   │ Strategic   │         │ Tactical    │          │
       │    │     │   │ Direction   │         │ Confirmation│          │
       │    │     │   └──────┬──────┘         └──────┬──────┘          │
       │    │     │          │                       │                 │
       │    │     │          └───────────┬───────────┘                 │
       │    │     │                      ▼                             │
       │    │     │           ┌───────────────────┐                    │
       │    │     │           │ COMBINED SIGNAL   │                    │
       │    │     │           │ Align → +15%      │                    │
       │    │     │           │ Conflict → -15%   │                    │
       │    │     │           └───────────────────┘                    │
       │    │     └─────────────────────────────────────────────────────┘
       │    │                            │
       ▼    ▼                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         DECISION ENGINE                                   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        GPT-5.1 REASONING                           │ │
│  │  - Analizza tutti i fattori                                        │ │
│  │  - Applica regole di trading                                       │ │
│  │  - Calcola confidenza finale                                       │ │
│  │  - Genera reasoning testuale                                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Decision Flow:                                                          │
│  1. LSTM filter (conf >= 66%)                                           │
│  2. CVOL check (EXTREME → block, WARNING → -50% risk)                   │
│  3. Risk checks (max trades, positions, hours)                          │
│  4. Macro check (no high-impact events)                                 │
│  5. DXY + Fed Sentiment → Combined Macro Signal                         │
│  6. GPT-5.1 final decision with reasoning                               │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   POSITION SIZING     │
                         │                       │
                         │ lots = risk$ / (SL×10)│
                         │ risk$ = bal × 0.65%   │
                         │       × CVOL_adj      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   MT5 ORDER           │
                         │   EXECUTION           │
                         └───────────────────────┘
```

---

## COMPONENTS DETAIL

### 1. LSTM MODEL (Primary Signal)

**File:** `models/lstm_balanced/best_model.pt`

| Specification | Value |
|---------------|-------|
| Architecture | Bidirectional LSTM + Attention |
| Input Size | 88 features |
| Hidden Size | 256 |
| Num Layers | 3 |
| Dropout | 0.3 |
| Output | 3 classes (SELL=0, HOLD=1, BUY=2) |

**Features (88 total):**
- Price: returns, log_returns, high_low_range
- Moving Averages: SMA(5,10,20,50), EMA(12,26)
- Oscillators: RSI(14), Stochastic K/D
- Volatility: ATR(14), Bollinger Bands
- Trend: MACD, MACD Signal, MACD Histogram
- Time: hour_sin, hour_cos (cyclical encoding)

---

### 2. CVOL MONITOR (Volatility Warning System)

**File:** `agent/cvol_monitor.py`

| Status | CVOL Range | Risk Adjustment | Action |
|--------|------------|-----------------|--------|
| **OK** | < 13.0 | 100% | Normal trading |
| **WARNING** | 13.0 - 19.0 | 50% | Trade with reduced size |
| **EXTREME** | > 19.0 | 0% | Block trade |

**Formula:**
```python
cvol = volatility_20 * sqrt(252) * 100 * 2  # Annualized
warning_level = mean(cvol_history) + 1.5 * std
extreme_level = mean(cvol_history) + 2.0 * std
```

---

### 3. DXY MODEL (Quarterly - Strategic)

**File:** `agent/dxy_model.py`

Analizza il differenziale di bilancio FED vs ECB per determinare la direzione strategica del dollaro.

| Spread (FED-ECB YoY) | USD Bias | Logic |
|---------------------|----------|-------|
| < -3% | BULLISH | FED contracting more → USD stronger |
| > +3% | BEARISH | ECB contracting more → EUR stronger |
| -3% to +3% | NEUTRAL | No clear direction |

**Timeframe:** Trimestrale (aggiornato ogni 3 mesi)

---

### 4. FED SENTIMENT MODEL (Monthly - Tactical)

**File:** `agent/fed_sentiment_model.py`

Analisi NLP dei verbali FOMC e discorsi Fed per determinare la stance di politica monetaria.

**Methodology:** Morgan Stanley MNLPFEDS + Loughran-McDonald Dictionary

| Sentiment | Range | USD Bias | Regime |
|-----------|-------|----------|--------|
| > +0.30 | Strong Hawkish | STRONG BULLISH | Rate hikes expected |
| +0.15 to +0.30 | Hawkish | BULLISH | Tightening bias |
| -0.15 to +0.15 | Neutral | NEUTRAL | Data dependent |
| -0.30 to -0.15 | Dovish | BEARISH | Easing bias |
| < -0.30 | Strong Dovish | STRONG BEARISH | Rate cuts expected |

**Current (Dec 2025):** +0.089 (NEUTRAL, trend → Dovish)

---

### 5. MACRO COMBINATION LOGIC

| DXY (Quarterly) | Fed (Monthly) | Alignment | Bonus/Penalty |
|-----------------|---------------|-----------|---------------|
| BULLISH | HAWKISH | **STRONG ALIGN** | +15% |
| BULLISH | NEUTRAL | PARTIAL | +5% |
| BULLISH | DOVISH | **CONFLICT** | -10% to -15% |
| BEARISH | DOVISH | **STRONG ALIGN** | +15% |
| BEARISH | NEUTRAL | PARTIAL | +5% |
| BEARISH | HAWKISH | **CONFLICT** | -10% to -15% |
| NEUTRAL | * | NEUTRAL | 0% |

---

### 6. NEWS SENTIMENT

**File:** `agent/news_sentiment.py`

- **Source:** NewsAPI.org
- **Analysis:** Keyword-based sentiment scoring
- **Currencies:** USD, EUR, GBP, JPY, AUD, CHF, CAD, NZD
- **Freshness:** Max 24 hours

---

### 7. DECISION ENGINE

**File:** `agent/decision_engine.py`

**AI Model:** GPT-5.1 (Responses API)
- Reasoning Effort: Medium
- Input: All analysis data in structured prompt
- Output: JSON with should_trade, signal, confidence, reasoning

**Fallback:** Rule-based decision when AI unavailable

---

## TRADING RULES

### Entry Conditions

| Rule | Value | Description |
|------|-------|-------------|
| LSTM Confidence | >= 66% | Minimum model confidence |
| Active Hours | 08:00 - 20:00 UTC | London + NY sessions |
| Max Daily Trades | 2 | Per day total |
| Max Open Positions | 3 | Across all pairs |
| Cooldown | 72 hours | Per pair between trades |

### Position Sizing

```
risk_per_trade = 0.65%  (of account balance)
adjusted_risk = risk_per_trade × CVOL_adjustment

risk_amount = balance × adjusted_risk
sl_pips = ATR × 1.5 × 10000

lot_size = risk_amount / (sl_pips × $10)
lot_size = min(max(lot_size, 0.01), 5.0)
```

### Stop Loss & Take Profit

| Parameter | Value | R:R Ratio |
|-----------|-------|-----------|
| Stop Loss | ATR × 1.5 | 1.0 |
| Take Profit | ATR × 3.0 | 2.0 |

---

## PAIRS TRADED

| Pair | Type | Pip Value | Notes |
|------|------|-----------|-------|
| EURUSD | Major | $10/lot | Most liquid |
| GBPUSD | Major | $10/lot | Higher volatility |
| USDJPY | Major | ~$9/lot | USD is base |
| AUDUSD | Major | $10/lot | Commodity-linked |

---

## RISK MANAGEMENT

### Per-Trade Risk
- **Base Risk:** 0.65% per trade
- **CVOL Warning:** 0.325% (50% reduction)
- **Max Lot Size:** 5.0 lots

### Daily Risk
- **Max Daily Risk:** 3.0%
- **Max Trades:** 2 per day
- **Max Open:** 3 positions

### Drawdown Target
- **Max Drawdown:** < 15% (optimized via backtest)

---

## FILE STRUCTURE

```
forex_ai_project/
├── agent/
│   ├── __init__.py
│   ├── config.py              # All configuration
│   ├── trading_agent.py       # Main orchestrator
│   ├── decision_engine.py     # AI decision making
│   ├── macro_analyzer.py      # Economic calendar
│   ├── news_sentiment.py      # News analysis
│   ├── dxy_model.py           # Quarterly macro (FED/ECB)
│   ├── fed_sentiment_model.py # Monthly FOMC NLP
│   └── cvol_monitor.py        # Volatility warning
│
├── models/
│   └── lstm_balanced/
│       └── best_model.pt      # Trained LSTM
│
├── data/
│   ├── raw/                   # Raw OHLC data
│   └── processed/             # Feature data
│
├── backtest.py                # Backtesting engine
├── live_trading.py            # Simple live trader
├── dashboard.py               # Monitoring UI
├── train_lstm.py              # Model training
│
└── AGENT_SPECIFICATIONS.md    # This file
```

---

## CONFIGURATION REFERENCE

### TRADING_CONFIG
```python
{
    'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
    'risk_per_trade': 0.0065,      # 0.65%
    'max_daily_risk': 0.03,        # 3%
    'max_trades_per_day': 2,
    'max_open_positions': 3,
    'max_lot_size': 5.0,
    'min_lstm_confidence': 0.66,   # 66%
    'take_profit_atr': 3.0,
    'stop_loss_atr': 1.5,
    'active_hours': (8, 20),       # UTC
    'min_hours_between_trades': 72,
}
```

### CVOL_CONFIG
```python
{
    'default_warning_level': 13.0,
    'default_extreme_level': 19.0,
    'warning_risk_factor': 0.5,
    'extreme_risk_factor': 0.0,
    'cache_duration_hours': 1,
}
```

### AGENT_CONFIG
```python
{
    'model': 'gpt-5.1',
    'reasoning_effort': 'medium',
}
```

---

## SIGNAL FLOW EXAMPLE

```
1. MT5 provides EURUSD H1 data
   └─> Close: 1.08500, ATR: 0.00450

2. LSTM Model predicts
   └─> Signal: BUY, Confidence: 72%

3. CVOL Monitor checks
   └─> CVOL: 10.5 (OK), Risk adj: 100%

4. DXY Model (Quarterly)
   └─> FED -8.9%, ECB -3.1%, Spread: -5.8%
   └─> USD Bias: BULLISH → EURUSD should SELL

5. Fed Sentiment (Monthly)
   └─> Sentiment: +0.089 (NEUTRAL)
   └─> No strong bias

6. Combined Macro
   └─> DXY BULLISH + Fed NEUTRAL = Partial alignment
   └─> LSTM says BUY, DXY says SELL → CONFLICT
   └─> Penalty: -10%

7. GPT-5.1 Decision
   └─> LSTM 72% is strong, macro conflict is mild
   └─> Final: BUY with 62% confidence

8. Position Sizing
   └─> Account: $100,000
   └─> Risk: $650 (0.65%)
   └─> SL: 67 pips (ATR × 1.5)
   └─> Lots: $650 / (67 × $10) = 0.97 lots

9. Order Execution
   └─> BUY 0.97 EURUSD @ 1.08500
   └─> SL: 1.07825, TP: 1.09850
```

---

## PERFORMANCE METRICS (Backtest 2019-2024)

| Metric | Value |
|--------|-------|
| Total Return | ~45% |
| Max Drawdown | < 15% |
| Win Rate | ~55% |
| Profit Factor | ~1.5 |
| Sharpe Ratio | ~1.2 |
| Total Trades | ~180 |

*Results based on backtesting with 66% confidence threshold and 0.65% risk per trade.*

---

## RUNNING THE AGENT

### Demo Mode (Paper Trading)
```bash
cd forex_ai_project
python -m agent.trading_agent --mode demo
```

### Live Mode (Real Money)
```bash
python -m agent.trading_agent --mode live --confirm
```

### Dashboard
```bash
python dashboard.py
```

---

## DEPENDENCIES

```
MetaTrader5
numpy
pandas
torch
openai
requests
python-dotenv
matplotlib
flask (dashboard)
```

---

## ENVIRONMENT VARIABLES

```
OPENAI_API_KEY=sk-...
NEWS_API_KEY=...
MT5_PASSWORD=...
```

---

## CHANGELOG

### v3.0 (December 2025)
- Added Fed Sentiment Model (Monthly FOMC NLP)
- Multi-timeframe macro combination (DXY + Fed)
- CVOL Warning System with risk adjustment
- Aligned risk sizing across backtest/live
- max_lot_size: 10 → 5 (more conservative)

### v2.0 (November 2025)
- Added DXY Model (FED vs ECB)
- Changed DXY from veto to weight system
- Confidence threshold: 70% → 66%
- Risk per trade: 1% → 0.65%

### v1.0 (October 2025)
- Initial release
- LSTM model + GPT-5.1 reasoning
- Basic macro and news analysis

---

## CONTACT

For issues and improvements, see the GitHub repository.

**Disclaimer:** This is an experimental trading system. Past performance does not guarantee future results. Trade at your own risk.
