"""
================================================================================
AI TRADING AGENT - DECISION ENGINE
================================================================================
Combina tutti i segnali (LSTM, Macro, News) e prende decisioni di trading.
Usa OpenAI GPT API per ragionamento avanzato.
================================================================================
"""

from openai import OpenAI
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
import json

from .config import TRADING_CONFIG, AGENT_CONFIG, OPENAI_API_KEY
from .macro_analyzer import get_macro_analyzer
from .news_sentiment import get_news_analyzer

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Motore decisionale che combina:
    - Segnali LSTM (tecnici)
    - Analisi Macro (calendario economico)
    - News Sentiment
    - Ragionamento AI (GPT-4)
    """

    def __init__(self):
        self.macro_analyzer = get_macro_analyzer()
        self.news_analyzer = get_news_analyzer()

        # Initialize OpenAI client
        if OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.openai_client = None
            logger.warning("OPENAI_API_KEY not set - AI reasoning disabled")

        # Trading state
        self.daily_trades = 0
        self.last_trade_date = None
        self.open_positions = {}
        self.trade_history = []

    def evaluate_trade(
        self,
        pair: str,
        lstm_signal: str,
        lstm_confidence: float,
        current_price: float,
        atr: float
    ) -> Dict:
        """
        Valuta se eseguire un trade combinando tutti i fattori.

        Args:
            pair: Coppia forex
            lstm_signal: Segnale LSTM (BUY/SELL/HOLD)
            lstm_confidence: Confidenza LSTM (0-1)
            current_price: Prezzo corrente
            atr: ATR corrente

        Returns:
            Dizionario con decisione e reasoning
        """
        result = {
            'pair': pair,
            'timestamp': datetime.now().isoformat(),
            'lstm_signal': lstm_signal,
            'lstm_confidence': lstm_confidence,
            'should_trade': False,
            'final_signal': 'HOLD',
            'confidence': 0.0,
            'reasoning': '',
            'stop_loss': None,
            'take_profit': None,
            'risk_checks': {},
        }

        # Step 1: Basic LSTM filter
        if lstm_signal == 'HOLD':
            result['reasoning'] = "LSTM signal is HOLD - no action needed"
            return result

        if lstm_confidence < TRADING_CONFIG['min_lstm_confidence']:
            result['reasoning'] = f"LSTM confidence too low: {lstm_confidence:.1%} < {TRADING_CONFIG['min_lstm_confidence']:.1%}"
            return result

        # Step 2: Risk checks
        risk_checks = self._check_risk_rules(pair)
        result['risk_checks'] = risk_checks

        if not risk_checks['passed']:
            result['reasoning'] = f"Risk check failed: {risk_checks['reason']}"
            return result

        # Step 3: Macro analysis
        macro_context = self.macro_analyzer.get_macro_context(pair)

        if macro_context['avoid_trading']:
            result['reasoning'] = f"Macro restriction: {macro_context['avoid_reason']}"
            return result

        # Step 4: News sentiment
        news_analysis = self.news_analyzer.analyze_news_for_pair(pair)

        # Step 5: AI Reasoning (if available)
        if self.openai_client:
            ai_decision = self._get_ai_decision(
                pair=pair,
                lstm_signal=lstm_signal,
                lstm_confidence=lstm_confidence,
                macro_context=macro_context,
                news_analysis=news_analysis,
                current_price=current_price,
                atr=atr
            )
            result.update(ai_decision)
        else:
            # Fallback: Rule-based decision
            result.update(self._rule_based_decision(
                lstm_signal=lstm_signal,
                lstm_confidence=lstm_confidence,
                macro_bias=macro_context['macro_bias'],
                news_bias=news_analysis['overall_bias']
            ))

        # Calculate SL/TP if trade approved
        if result['should_trade']:
            if result['final_signal'] == 'BUY':
                result['stop_loss'] = current_price - atr * TRADING_CONFIG['stop_loss_atr']
                result['take_profit'] = current_price + atr * TRADING_CONFIG['take_profit_atr']
            elif result['final_signal'] == 'SELL':
                result['stop_loss'] = current_price + atr * TRADING_CONFIG['stop_loss_atr']
                result['take_profit'] = current_price - atr * TRADING_CONFIG['take_profit_atr']

        return result

    def _check_risk_rules(self, pair: str) -> Dict:
        """Verifica regole di risk management."""
        # Reset daily counter
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today

        checks = {
            'passed': True,
            'reason': '',
            'details': {}
        }

        # Max daily trades
        if self.daily_trades >= TRADING_CONFIG['max_trades_per_day']:
            checks['passed'] = False
            checks['reason'] = f"Max daily trades reached ({TRADING_CONFIG['max_trades_per_day']})"
            return checks

        # Max open positions
        if len(self.open_positions) >= TRADING_CONFIG['max_open_positions']:
            checks['passed'] = False
            checks['reason'] = f"Max open positions reached ({TRADING_CONFIG['max_open_positions']})"
            return checks

        # Already have position on this pair
        if pair in self.open_positions:
            checks['passed'] = False
            checks['reason'] = f"Already have position on {pair}"
            return checks

        # Session filter
        hour = datetime.now().hour
        if not (TRADING_CONFIG['active_hours'][0] <= hour <= TRADING_CONFIG['active_hours'][1]):
            checks['passed'] = False
            checks['reason'] = f"Outside active hours ({TRADING_CONFIG['active_hours']})"
            return checks

        return checks

    def _get_ai_decision(
        self,
        pair: str,
        lstm_signal: str,
        lstm_confidence: float,
        macro_context: Dict,
        news_analysis: Dict,
        current_price: float,
        atr: float
    ) -> Dict:
        """
        Usa GPT-5.1 Responses API per prendere una decisione di trading informata.
        """
        prompt = f"""You are an expert forex trading AI assistant. Analyze the following data and decide whether to execute a trade.

## TECHNICAL ANALYSIS (LSTM Model)
- Pair: {pair}
- Signal: {lstm_signal}
- Confidence: {lstm_confidence:.1%}
- Current Price: {current_price:.5f}
- ATR (14): {atr:.5f}

## MACRO CONTEXT
- Macro Bias: {macro_context['macro_bias']}
- Trading Restriction: {macro_context['avoid_trading']}
- Upcoming Events: {len(macro_context['upcoming_events'])} events

## NEWS SENTIMENT
- Overall Bias: {news_analysis['overall_bias']}
- Base Currency ({pair[:3]}) Sentiment: {news_analysis['base_sentiment']:+.2f}
- Quote Currency ({pair[3:6]}) Sentiment: {news_analysis['quote_sentiment']:+.2f}
- Confidence: {news_analysis['confidence']:.1%}

## TRADING RULES
- Only trade if LSTM confidence > 70%
- Prefer trades where technical, macro, and sentiment align
- Avoid trading before high-impact news events
- Risk: 1% per trade, R:R ratio 1:2

## YOUR TASK
Analyze all factors and provide your decision in the following JSON format:
{{
    "should_trade": true/false,
    "final_signal": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your decision"
}}

Respond ONLY with the JSON, no other text."""

        try:
            # GPT-5.1 Responses API
            response = self.openai_client.responses.create(
                model=AGENT_CONFIG['model'],
                reasoning={"effort": AGENT_CONFIG['reasoning_effort']},
                input=[
                    {"role": "system", "content": "You are an expert forex trading AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse response - usa output_text della Responses API
            response_text = response.output_text.strip()

            # Extract JSON from response
            if response_text.startswith('{'):
                decision = json.loads(response_text)
            else:
                # Try to find JSON in response
                import re
                json_match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

            logger.info(f"AI Decision for {pair}: {decision}")
            return decision

        except Exception as e:
            logger.error(f"AI decision error: {e}")
            # Fallback to rule-based
            return self._rule_based_decision(
                lstm_signal=lstm_signal,
                lstm_confidence=lstm_confidence,
                macro_bias=macro_context['macro_bias'],
                news_bias=news_analysis['overall_bias']
            )

    def _rule_based_decision(
        self,
        lstm_signal: str,
        lstm_confidence: float,
        macro_bias: str,
        news_bias: str
    ) -> Dict:
        """
        Decisione basata su regole quando AI non è disponibile.
        """
        result = {
            'should_trade': False,
            'final_signal': 'HOLD',
            'confidence': 0.0,
            'reasoning': ''
        }

        # Count aligned signals
        signals_aligned = 0
        total_signals = 3

        # LSTM signal
        if lstm_signal in ['BUY', 'SELL']:
            signals_aligned += 1

        # Macro alignment
        if (lstm_signal == 'BUY' and macro_bias == 'bullish') or \
           (lstm_signal == 'SELL' and macro_bias == 'bearish'):
            signals_aligned += 1
        elif macro_bias == 'neutral':
            signals_aligned += 0.5

        # News alignment
        if (lstm_signal == 'BUY' and news_bias == 'bullish') or \
           (lstm_signal == 'SELL' and news_bias == 'bearish'):
            signals_aligned += 1
        elif news_bias == 'neutral':
            signals_aligned += 0.5

        # Decision logic
        alignment_score = signals_aligned / total_signals

        if alignment_score >= 0.7:
            result['should_trade'] = True
            result['final_signal'] = lstm_signal
            result['confidence'] = min(lstm_confidence, alignment_score)
            result['reasoning'] = f"Strong alignment ({alignment_score:.0%}): LSTM={lstm_signal}, Macro={macro_bias}, News={news_bias}"
        elif alignment_score >= 0.5:
            # Partial alignment - trade with lower confidence
            if lstm_confidence > 0.8:  # Only if LSTM is very confident
                result['should_trade'] = True
                result['final_signal'] = lstm_signal
                result['confidence'] = lstm_confidence * 0.8
                result['reasoning'] = f"Partial alignment ({alignment_score:.0%}) but high LSTM confidence"
            else:
                result['reasoning'] = f"Insufficient alignment ({alignment_score:.0%}) and LSTM confidence"
        else:
            result['reasoning'] = f"Signals conflict - LSTM={lstm_signal}, Macro={macro_bias}, News={news_bias}"

        return result

    def record_trade(self, pair: str, signal: str, entry_price: float, sl: float, tp: float):
        """Registra un trade aperto."""
        self.open_positions[pair] = {
            'signal': signal,
            'entry_price': entry_price,
            'stop_loss': sl,
            'take_profit': tp,
            'entry_time': datetime.now(),
        }
        self.daily_trades += 1

        logger.info(f"Trade recorded: {signal} {pair} @ {entry_price:.5f}")

    def close_trade(self, pair: str, exit_price: float, reason: str):
        """Chiude un trade e registra il risultato."""
        if pair not in self.open_positions:
            return

        position = self.open_positions[pair]

        # Calculate PnL
        if position['signal'] == 'BUY':
            pnl_pips = (exit_price - position['entry_price']) * 10000
        else:
            pnl_pips = (position['entry_price'] - exit_price) * 10000

        trade_record = {
            **position,
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'exit_reason': reason,
            'pnl_pips': pnl_pips,
        }

        self.trade_history.append(trade_record)
        del self.open_positions[pair]

        logger.info(f"Trade closed: {pair} | PnL: {pnl_pips:+.1f} pips | Reason: {reason}")

    def get_summary(self) -> str:
        """Ottiene un riepilogo dello stato corrente."""
        lines = [
            "=" * 50,
            "TRADING AGENT STATUS",
            "=" * 50,
            f"Daily trades: {self.daily_trades}/{TRADING_CONFIG['max_trades_per_day']}",
            f"Open positions: {len(self.open_positions)}/{TRADING_CONFIG['max_open_positions']}",
            f"Total trades (history): {len(self.trade_history)}",
            "",
            "Open Positions:",
        ]

        if self.open_positions:
            for pair, pos in self.open_positions.items():
                lines.append(f"  {pair}: {pos['signal']} @ {pos['entry_price']:.5f}")
        else:
            lines.append("  None")

        return "\n".join(lines)


# Singleton instance
_decision_engine = None


def get_decision_engine() -> DecisionEngine:
    """Get singleton DecisionEngine instance."""
    global _decision_engine
    if _decision_engine is None:
        _decision_engine = DecisionEngine()
    return _decision_engine
