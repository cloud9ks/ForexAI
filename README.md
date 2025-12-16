# AI Forex Trading Agent

Autonomous forex trading agent combining LSTM deep learning with GPT-5.1 reasoning.

## Features
- **LSTM Model**: 4.1M parameters, 88 features, trained on 10 years of data
- **GPT-5.1 Integration**: Advanced reasoning for trade decisions
- **Macro Analysis**: Economic calendar filtering
- **News Sentiment**: Real-time news analysis
- **Risk Management**: 1% per trade, 2:1 R:R ratio

## Backtest Results (10 years)
- Return: +6,381%
- Win Rate: 56.1%
- Profit Factor: 1.96
- Max Drawdown: 25.6%

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key"

# Run in demo mode
python -m agent.trading_agent --mode demo
```

## Modes
- `demo`: Paper trading on demo account
- `simulation`: Log-only, no trades
- `live`: Real money (requires confirmation)
