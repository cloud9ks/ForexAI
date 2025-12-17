"""
================================================================================
AI TRADING AGENT - MAIN ORCHESTRATOR
================================================================================
Agente di trading AI che combina:
- LSTM Model (segnali tecnici)
- Macro Analysis (calendario economico)
- News Sentiment (analisi news)
- Claude AI (ragionamento avanzato)

Uso:
    python -m agent.trading_agent --mode demo
================================================================================
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import time
import logging
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.config import TRADING_CONFIG, LOGGING_CONFIG
from agent.decision_engine import get_decision_engine
from agent.macro_analyzer import get_macro_analyzer
from agent.news_sentiment import get_news_analyzer

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['log_level']),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# LSTM MODEL (same as training)
# ============================================================================
class LSTMBalanced(nn.Module):
    """LSTM model for forex prediction."""

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
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context)


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================
def compute_features_realtime(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for real-time prediction."""
    features = pd.DataFrame(index=df.index)

    # Price features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    features['high_low_range'] = (df['high'] - df['low']) / df['close']

    # Moving averages
    for period in [5, 10, 20, 50]:
        sma = df['close'].rolling(period).mean()
        features[f'sma_{period}_dist'] = (df['close'] - sma) / sma

    # EMA
    for period in [12, 26]:
        ema = df['close'].ewm(span=period).mean()
        features[f'ema_{period}_dist'] = (df['close'] - ema) / ema

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    features['macd'] = macd / df['close']
    features['macd_signal'] = signal / df['close']
    features['macd_hist'] = (macd - signal) / df['close']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    features['rsi_norm'] = (features['rsi'] - 50) / 50

    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    features['bb_position'] = (df['close'] - sma20) / (2 * std20 + 1e-10)
    features['bb_width'] = (4 * std20) / sma20

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr'] = tr.rolling(14).mean() / df['close']
    features['atr_raw'] = tr.rolling(14).mean()

    # Stochastic
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    features['stoch_k'] = ((df['close'] - low14) / (high14 - low14 + 1e-10) - 0.5) * 2
    features['stoch_d'] = features['stoch_k'].rolling(3).mean()

    # Time features
    if hasattr(df.index, 'hour'):
        features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    else:
        features['hour_sin'] = 0
        features['hour_cos'] = 0

    # Clean
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)

    return features


# ============================================================================
# AI TRADING AGENT
# ============================================================================
class AITradingAgent:
    """
    Agente di trading AI che combina LSTM + Macro + News + Claude reasoning.
    """

    def __init__(self, mode='demo'):
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.input_size = 88

        # Components
        self.decision_engine = get_decision_engine()
        self.macro_analyzer = get_macro_analyzer()
        self.news_analyzer = get_news_analyzer()

        # Load LSTM model
        self._load_model()

        logger.info(f"AI Trading Agent initialized (mode: {mode})")

    def _load_model(self):
        """Load trained LSTM model."""
        model_path = Path(__file__).parent.parent / "models" / "lstm_balanced" / "best_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.input_size = state_dict['lstm.weight_ih_l0'].shape[1]

        self.model = LSTMBalanced(self.input_size)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"LSTM model loaded (input_size: {self.input_size})")

    def connect_broker(self):
        """Connect to MetaTrader 5."""
        if not mt5.initialize():
            raise Exception(f"MT5 initialization failed: {mt5.last_error()}")

        account = mt5.account_info()
        if account:
            logger.info(f"Connected to MT5: {account.login} ({account.server})")
            logger.info(f"Balance: ${account.balance:.2f}")

            # Safety check for live mode
            if self.mode == 'live' and account.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO:
                logger.warning("Running LIVE mode on DEMO account")
            elif self.mode == 'demo' and account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
                logger.warning("DEMO mode requested but account is LIVE - switching to simulation")
                self.mode = 'simulation'

    def get_market_data(self, pair: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Get market data from MT5."""
        rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_H1, 0, bars)

        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.columns = [c.lower() for c in df.columns]

        return df

    def get_lstm_prediction(self, pair: str) -> tuple:
        """Get LSTM prediction for a pair."""
        df = self.get_market_data(pair)
        if df is None:
            return 'HOLD', 0.0, 0.0

        # Compute features
        features = compute_features_realtime(df)
        feature_values = features.iloc[-1].values.astype(np.float32)

        # Adjust dimensions
        if len(feature_values) > self.input_size:
            feature_values = feature_values[:self.input_size]
        elif len(feature_values) < self.input_size:
            padding = np.zeros(self.input_size - len(feature_values), dtype=np.float32)
            feature_values = np.concatenate([feature_values, padding])

        # Clean
        feature_values = np.clip(feature_values, -10, 10)
        feature_values = np.nan_to_num(feature_values, nan=0)

        # Predict
        X = torch.FloatTensor(feature_values).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(X)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        # Get ATR
        atr = features['atr_raw'].iloc[-1] if 'atr_raw' in features.columns else 0.001

        signal = ['SELL', 'HOLD', 'BUY'][pred]
        return signal, confidence, atr

    def execute_trade(self, pair: str, signal: str, sl: float, tp: float) -> bool:
        """Execute trade on MT5."""
        if self.mode == 'simulation':
            logger.info(f"[SIMULATION] Would execute: {signal} {pair} | SL: {sl:.5f} | TP: {tp:.5f}")
            return True

        symbol_info = mt5.symbol_info(pair)
        if symbol_info is None:
            logger.error(f"Symbol {pair} not found")
            return False

        if not symbol_info.visible:
            mt5.symbol_select(pair, True)

        tick = mt5.symbol_info_tick(pair)
        if tick is None:
            return False

        # Calculate lot size
        account = mt5.account_info()
        risk_amount = account.balance * TRADING_CONFIG['risk_per_trade']
        sl_pips = abs(tick.ask - sl) * 10000 if signal == 'BUY' else abs(tick.bid - sl) * 10000
        lot_size = min(max(risk_amount / (sl_pips * 10), 0.01), 1.0)

        # Order parameters
        if signal == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pair,
            "volume": round(lot_size, 2),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": f"AI Agent {signal}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode}")
            return False

        logger.info(f"ORDER EXECUTED: {signal} {pair} @ {price:.5f} | Ticket: {result.order}")

        # Record in decision engine
        self.decision_engine.record_trade(pair, signal, price, sl, tp)

        return True

    def analyze_pair(self, pair: str) -> Dict:
        """Full analysis of a trading pair."""
        logger.info(f"\n{'='*50}")
        logger.info(f"ANALYZING: {pair}")
        logger.info(f"{'='*50}")

        # Get LSTM prediction
        lstm_signal, lstm_confidence, atr = self.get_lstm_prediction(pair)
        logger.info(f"LSTM: {lstm_signal} ({lstm_confidence:.1%}) | ATR: {atr:.5f}")

        # Get current price
        tick = mt5.symbol_info_tick(pair)
        current_price = tick.ask if tick else 0

        # Get decision from engine (combines LSTM + Macro + News + AI)
        decision = self.decision_engine.evaluate_trade(
            pair=pair,
            lstm_signal=lstm_signal,
            lstm_confidence=lstm_confidence,
            current_price=current_price,
            atr=atr
        )

        # Log decision
        logger.info(f"DECISION: {decision['final_signal']} (conf: {decision['confidence']:.1%})")
        logger.info(f"REASONING: {decision['reasoning']}")

        return decision

    def run_cycle(self):
        """Run one trading cycle."""
        logger.info("\n" + "=" * 60)
        logger.info(f"TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        for pair in TRADING_CONFIG['pairs']:
            try:
                # Full analysis
                decision = self.analyze_pair(pair)

                # Execute if approved
                if decision['should_trade']:
                    logger.info(f">>> EXECUTING: {decision['final_signal']} {pair}")

                    success = self.execute_trade(
                        pair=pair,
                        signal=decision['final_signal'],
                        sl=decision['stop_loss'],
                        tp=decision['take_profit']
                    )

                    if success:
                        logger.info(f"Trade executed successfully!")
                    else:
                        logger.error(f"Trade execution failed!")

            except Exception as e:
                logger.error(f"Error analyzing {pair}: {e}")

        # Print status
        logger.info("\n" + self.decision_engine.get_summary())

    def run(self, interval_minutes: int = 60):
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("AI TRADING AGENT - STARTING")
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Pairs: {TRADING_CONFIG['pairs']}")
        logger.info("=" * 60)

        self.connect_broker()

        while True:
            try:
                self.run_cycle()

                # Wait for next hour
                now = datetime.now()
                next_hour = now.replace(minute=5, second=0) + timedelta(hours=1)
                wait_seconds = (next_hour - now).total_seconds()

                logger.info(f"\nNext cycle: {next_hour.strftime('%H:%M')} (in {wait_seconds/60:.1f} min)")
                time.sleep(min(wait_seconds, interval_minutes * 60))

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)

        mt5.shutdown()
        logger.info("Agent stopped")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='AI Trading Agent')
    parser.add_argument('--mode', choices=['demo', 'live', 'simulation'], default='demo')
    parser.add_argument('--confirm', action='store_true', help='Skip live mode confirmation')
    args = parser.parse_args()

    if args.mode == 'live' and not args.confirm:
        confirm = input("WARNING: LIVE mode with real money! Confirm? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled")
            return

    agent = AITradingAgent(mode=args.mode)
    agent.run()


if __name__ == "__main__":
    main()
