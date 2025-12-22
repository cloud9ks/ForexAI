"""
================================================================================
FOREX AI - LIVE TRADING SYSTEM
================================================================================
Paper Trading / Live Trading con MT5

Esegue:
1. Connessione a MetaTrader 5
2. Download dati real-time H1
3. Calcolo features
4. Predizione LSTM
5. Apertura/chiusura ordini

Uso:
    python live_trading.py --mode demo    # Paper trading su demo
    python live_trading.py --mode live    # Live trading (ATTENZIONE!)
================================================================================
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
import json
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
CONFIG = {
    'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
    'timeframe': mt5.TIMEFRAME_H1,
    'lookback_bars': 100,           # Barre per calcolo features
    'min_confidence': 0.66,         # Confidenza minima per trade (allineata con agent)
    'risk_per_trade': 0.0065,       # 0.65% rischio per trade (allineato con agent)
    'take_profit_atr': 3.0,         # TP in multipli ATR
    'stop_loss_atr': 1.5,           # SL in multipli ATR
    'max_trades_per_day': 2,        # Max trades giornalieri
    'min_hours_between_trades': 72, # Min ore tra trades per coppia
    'active_hours': (8, 20),        # Sessioni attive (UTC)
    'magic_number': 123456,         # ID per identificare ordini
    'max_lot_size': 5.0,            # Max lotti per trade (allineato con agent)
}

# ============================================================================
# MODELLO LSTM
# ============================================================================
class LSTMBalanced(nn.Module):
    """LSTM con LayerNorm per classificazione bilanciata."""

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
        out = self.classifier(context)
        return out


# ============================================================================
# FEATURE ENGINEERING (versione real-time)
# ============================================================================
def compute_features_realtime(df):
    """Calcola features per una singola coppia in real-time."""
    features = pd.DataFrame(index=df.index)

    # Price features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    features['high_low_range'] = (df['high'] - df['low']) / df['close']
    features['close_open_range'] = (df['close'] - df['open']) / df['open']

    # Moving averages
    for period in [5, 10, 20, 50]:
        sma = df['close'].rolling(period).mean()
        features[f'sma_{period}_dist'] = (df['close'] - sma) / sma
        features[f'sma_{period}_slope'] = sma.pct_change(5)

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
    features['bb_upper_dist'] = (df['close'] - (sma20 + 2*std20)) / df['close']
    features['bb_lower_dist'] = (df['close'] - (sma20 - 2*std20)) / df['close']
    features['bb_width'] = (4 * std20) / sma20
    features['bb_position'] = (df['close'] - sma20) / (2 * std20 + 1e-10)

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr'] = tr.rolling(14).mean() / df['close']
    features['atr_raw'] = tr.rolling(14).mean()  # Per calcolo SL/TP

    # Stochastic
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    features['stoch_k'] = ((df['close'] - low14) / (high14 - low14 + 1e-10)) * 100
    features['stoch_d'] = features['stoch_k'].rolling(3).mean()
    features['stoch_k_norm'] = (features['stoch_k'] - 50) / 50
    features['stoch_d_norm'] = (features['stoch_d'] - 50) / 50

    # Volume proxy (usando range)
    features['volume_proxy'] = high_low / high_low.rolling(20).mean()

    # Momentum
    for period in [5, 10, 20]:
        features[f'momentum_{period}'] = df['close'].pct_change(period)

    # Volatility
    features['volatility_5'] = features['returns'].rolling(5).std()
    features['volatility_20'] = features['returns'].rolling(20).std()
    features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-10)

    # Hour/Day features
    if hasattr(df.index, 'hour'):
        features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    else:
        features['hour_sin'] = 0
        features['hour_cos'] = 0
        features['day_sin'] = 0
        features['day_cos'] = 0

    # Lagged features
    for lag in [1, 2, 3]:
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        features[f'rsi_lag_{lag}'] = features['rsi_norm'].shift(lag)

    # Pattern features
    features['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10) < 0.1).astype(float)
    features['bullish_engulf'] = ((df['close'] > df['open']) &
                                   (df['close'].shift(1) < df['open'].shift(1)) &
                                   (df['close'] > df['open'].shift(1)) &
                                   (df['open'] < df['close'].shift(1))).astype(float)

    # Currency Strength proxy (basato su returns)
    features['strength_proxy'] = features['returns'].rolling(20).mean() / (features['returns'].rolling(20).std() + 1e-10)

    # Fill NaN
    features = features.fillna(0)

    # Rimuovi colonne non numeriche o infinite
    features = features.replace([np.inf, -np.inf], 0)

    return features


# ============================================================================
# TRADING ENGINE
# ============================================================================
class LiveTrader:
    def __init__(self, mode='demo'):
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.input_size = 88  # Default, sarà aggiornato dal modello
        self.last_trade_time = {}
        self.daily_trades = 0
        self.last_trade_date = None
        self.positions = {}

        # Carica modello
        self.load_model()

    def load_model(self):
        """Carica il modello LSTM trainato."""
        model_path = Path(__file__).parent / "models" / "lstm_balanced" / "best_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.input_size = state_dict['lstm.weight_ih_l0'].shape[1]
        logger.info(f"Input size dal modello: {self.input_size}")

        self.model = LSTMBalanced(self.input_size)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Modello LSTM caricato!")

    def connect_mt5(self):
        """Connette a MetaTrader 5."""
        if not mt5.initialize():
            error = mt5.last_error()
            raise Exception(f"MT5 initialization failed: {error}")

        account = mt5.account_info()
        if account:
            logger.info(f"Connesso a MT5: {account.login} ({account.server})")
            logger.info(f"Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f}")

            if self.mode == 'demo' and account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
                logger.warning("ATTENZIONE: Account NON DEMO! Passando a modalità simulazione.")
                self.mode = 'simulation'
        else:
            raise Exception("Impossibile ottenere info account")

    def get_market_data(self, pair):
        """Scarica dati recenti per una coppia."""
        rates = mt5.copy_rates_from_pos(pair, CONFIG['timeframe'], 0, CONFIG['lookback_bars'])

        if rates is None or len(rates) == 0:
            logger.warning(f"Nessun dato per {pair}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.columns = [c.lower() for c in df.columns]

        return df

    def get_prediction(self, pair):
        """Ottiene predizione per una coppia."""
        # Scarica dati
        df = self.get_market_data(pair)
        if df is None:
            return None, 0, 0

        # Calcola features
        features = compute_features_realtime(df)

        # Prendi ultima riga
        feature_values = features.iloc[-1].values.astype(np.float32)

        # Adatta dimensioni
        if len(feature_values) > self.input_size:
            feature_values = feature_values[:self.input_size]
        elif len(feature_values) < self.input_size:
            padding = np.zeros(self.input_size - len(feature_values), dtype=np.float32)
            feature_values = np.concatenate([feature_values, padding])

        # Normalizza
        feature_values = np.clip(feature_values, -10, 10)
        feature_values = np.nan_to_num(feature_values, nan=0, posinf=0, neginf=0)

        # Predizione
        X = torch.FloatTensor(feature_values).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(X)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        # ATR per SL/TP
        atr = features['atr_raw'].iloc[-1] if 'atr_raw' in features.columns else 0.001

        signal = ['SELL', 'HOLD', 'BUY'][pred]
        return signal, confidence, atr

    def can_trade(self, pair):
        """Verifica se possiamo aprire un trade."""
        now = datetime.now()

        # Reset contatore giornaliero
        today = now.date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today

        # Max trades giornalieri
        if self.daily_trades >= CONFIG['max_trades_per_day']:
            return False, "Max trades giornalieri raggiunto"

        # Ore attive
        if not (CONFIG['active_hours'][0] <= now.hour <= CONFIG['active_hours'][1]):
            return False, f"Fuori orario attivo ({CONFIG['active_hours']})"

        # Tempo minimo tra trades per coppia
        if pair in self.last_trade_time:
            hours_since = (now - self.last_trade_time[pair]).total_seconds() / 3600
            if hours_since < CONFIG['min_hours_between_trades']:
                return False, f"Trade recente su {pair} ({hours_since:.1f}h fa)"

        # Posizione già aperta
        if pair in self.positions:
            return False, f"Posizione già aperta su {pair}"

        return True, "OK"

    def calculate_lot_size(self, pair, atr):
        """Calcola dimensione posizione con proper risk sizing."""
        account = mt5.account_info()
        if not account:
            return 0.01

        risk_amount = account.balance * CONFIG['risk_per_trade']
        sl_pips = atr * CONFIG['stop_loss_atr'] * 10000

        if sl_pips < 10:
            sl_pips = 10

        # Valore pip approssimativo
        pip_value = 10  # Per major pairs

        lots = risk_amount / (sl_pips * pip_value)

        # Apply max lot limit from config
        max_lots = CONFIG.get('max_lot_size', 5.0)
        lots = min(lots, max_lots)
        lots = max(lots, 0.01)

        logger.info(f"Position sizing: Risk ${risk_amount:.0f} → {lots:.2f} lots (SL: {sl_pips:.0f} pips)")

        return round(lots, 2)

    def open_trade(self, pair, signal, atr):
        """Apre un trade su MT5."""
        symbol_info = mt5.symbol_info(pair)
        if symbol_info is None:
            logger.error(f"Simbolo {pair} non trovato")
            return False

        if not symbol_info.visible:
            mt5.symbol_select(pair, True)

        # Prezzo corrente
        tick = mt5.symbol_info_tick(pair)
        if tick is None:
            logger.error(f"Tick non disponibile per {pair}")
            return False

        # Parametri ordine
        lot_size = self.calculate_lot_size(pair, atr)
        point = symbol_info.point

        if signal == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = price - atr * CONFIG['stop_loss_atr']
            tp = price + atr * CONFIG['take_profit_atr']
        else:  # SELL
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = price + atr * CONFIG['stop_loss_atr']
            tp = price - atr * CONFIG['take_profit_atr']

        # Request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pair,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": CONFIG['magic_number'],
            "comment": f"ForexAI {signal}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if self.mode == 'simulation':
            logger.info(f"[SIMULAZIONE] {signal} {pair} @ {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Lotti: {lot_size}")
            return True

        # Esegui ordine
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Ordine fallito: {result.retcode} - {result.comment}")
            return False

        logger.info(f"ORDINE ESEGUITO: {signal} {pair} @ {price:.5f} | Ticket: {result.order}")

        # Aggiorna stato
        self.positions[pair] = {
            'ticket': result.order,
            'direction': signal,
            'entry_price': price,
            'sl': sl,
            'tp': tp,
            'lot_size': lot_size,
            'time': datetime.now()
        }
        self.last_trade_time[pair] = datetime.now()
        self.daily_trades += 1

        return True

    def check_positions(self):
        """Controlla posizioni aperte."""
        positions = mt5.positions_get(magic=CONFIG['magic_number'])

        if positions:
            logger.info(f"Posizioni aperte: {len(positions)}")
            for pos in positions:
                profit = pos.profit
                logger.info(f"  {pos.symbol}: {pos.type} | Profit: ${profit:.2f}")

    def run_once(self):
        """Esegue un ciclo di trading."""
        logger.info("=" * 50)
        logger.info(f"Ciclo trading - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for pair in CONFIG['pairs']:
            # Verifica se possiamo tradare
            can, reason = self.can_trade(pair)

            # Ottieni predizione
            signal, confidence, atr = self.get_prediction(pair)

            if signal is None:
                logger.warning(f"{pair}: Nessuna predizione")
                continue

            logger.info(f"{pair}: {signal} (conf: {confidence:.1%}) | ATR: {atr:.5f}")

            # Verifica condizioni
            if signal == 'HOLD':
                continue

            if confidence < CONFIG['min_confidence']:
                logger.info(f"  -> Skip: confidenza insufficiente ({confidence:.1%} < {CONFIG['min_confidence']:.1%})")
                continue

            if not can:
                logger.info(f"  -> Skip: {reason}")
                continue

            # Apri trade
            logger.info(f"  -> SEGNALE VALIDO! Apertura {signal}...")
            self.open_trade(pair, signal, atr)

        # Check posizioni
        self.check_positions()

    def run(self, interval_minutes=60):
        """Loop principale di trading."""
        logger.info("=" * 60)
        logger.info("FOREX AI - LIVE TRADING SYSTEM")
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info("=" * 60)

        self.connect_mt5()

        while True:
            try:
                self.run_once()

                # Aspetta prossima candela H1
                now = datetime.now()
                next_hour = now.replace(minute=5, second=0, microsecond=0) + timedelta(hours=1)
                wait_seconds = (next_hour - now).total_seconds()

                logger.info(f"Prossimo check: {next_hour.strftime('%H:%M')} (tra {wait_seconds/60:.1f} min)")
                time.sleep(min(wait_seconds, interval_minutes * 60))

            except KeyboardInterrupt:
                logger.info("Interruzione manuale")
                break
            except Exception as e:
                logger.error(f"Errore: {e}")
                time.sleep(60)

        mt5.shutdown()
        logger.info("Trading terminato")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Forex AI Live Trading')
    parser.add_argument('--mode', choices=['demo', 'live', 'simulation'], default='demo',
                       help='Modalità: demo (paper trading), live (real money), simulation (no ordini)')
    args = parser.parse_args()

    if args.mode == 'live':
        confirm = input("ATTENZIONE: Modalità LIVE con soldi reali! Confermi? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Annullato")
            return

    trader = LiveTrader(mode=args.mode)
    trader.run()


if __name__ == "__main__":
    main()
