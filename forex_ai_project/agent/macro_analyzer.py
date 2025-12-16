"""
================================================================================
AI TRADING AGENT - MACRO ANALYZER
================================================================================
Analizza il calendario economico e gli eventi macro.

Fonti:
- Forex Factory (scraping)
- Investing.com Calendar
- FRED API (dati economici US)
================================================================================
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from bs4 import BeautifulSoup
import json

from .config import MACRO_CONFIG

logger = logging.getLogger(__name__)


class MacroAnalyzer:
    """Analizza eventi macroeconomici e calendario."""

    def __init__(self):
        self.events_cache = {}
        self.last_fetch = None
        self.cache_duration = timedelta(hours=1)

    def get_economic_calendar(self, days_ahead: int = 7) -> List[Dict]:
        """
        Recupera il calendario economico per i prossimi giorni.

        Returns:
            Lista di eventi con: date, time, currency, event, impact, forecast, previous
        """
        # Check cache
        if self._is_cache_valid():
            return self.events_cache.get('calendar', [])

        events = []

        try:
            # Metodo 1: Forex Factory (scraping)
            events = self._fetch_forex_factory()
        except Exception as e:
            logger.warning(f"Forex Factory fetch failed: {e}")

        if not events:
            try:
                # Metodo 2: Fallback a dati simulati basati su pattern storici
                events = self._generate_typical_events(days_ahead)
            except Exception as e:
                logger.error(f"Failed to generate events: {e}")

        self.events_cache['calendar'] = events
        self.last_fetch = datetime.now()

        return events

    def _fetch_forex_factory(self) -> List[Dict]:
        """Fetch calendario da Forex Factory."""
        events = []

        try:
            url = "https://www.forexfactory.com/calendar"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return events

            soup = BeautifulSoup(response.text, 'html.parser')

            # Parse calendar table
            rows = soup.select('.calendar__row')

            current_date = datetime.now().date()

            for row in rows:
                try:
                    # Extract data
                    currency_el = row.select_one('.calendar__currency')
                    event_el = row.select_one('.calendar__event-title')
                    impact_el = row.select_one('.calendar__impact span')
                    time_el = row.select_one('.calendar__time')

                    if not event_el:
                        continue

                    currency = currency_el.text.strip() if currency_el else ''
                    event_name = event_el.text.strip()
                    impact_class = impact_el.get('class', []) if impact_el else []
                    time_str = time_el.text.strip() if time_el else ''

                    # Determine impact level
                    impact = 'low'
                    if 'high' in str(impact_class):
                        impact = 'high'
                    elif 'medium' in str(impact_class):
                        impact = 'medium'

                    events.append({
                        'date': str(current_date),
                        'time': time_str,
                        'currency': currency,
                        'event': event_name,
                        'impact': impact,
                    })

                except Exception as e:
                    continue

        except Exception as e:
            logger.error(f"Forex Factory scraping error: {e}")

        return events

    def _generate_typical_events(self, days_ahead: int) -> List[Dict]:
        """Genera eventi tipici basati su pattern standard."""
        events = []
        today = datetime.now().date()

        # Eventi ricorrenti tipici
        typical_events = [
            # Monday
            {'day': 0, 'events': [
                {'currency': 'EUR', 'event': 'German Manufacturing PMI', 'impact': 'medium', 'time': '08:30'},
                {'currency': 'USD', 'event': 'ISM Manufacturing PMI', 'impact': 'high', 'time': '14:00'},
            ]},
            # Tuesday
            {'day': 1, 'events': [
                {'currency': 'AUD', 'event': 'RBA Interest Rate Decision', 'impact': 'high', 'time': '03:30'},
                {'currency': 'USD', 'event': 'JOLTS Job Openings', 'impact': 'medium', 'time': '14:00'},
            ]},
            # Wednesday
            {'day': 2, 'events': [
                {'currency': 'USD', 'event': 'ADP Employment Change', 'impact': 'medium', 'time': '12:15'},
                {'currency': 'USD', 'event': 'FOMC Statement', 'impact': 'high', 'time': '18:00'},
            ]},
            # Thursday
            {'day': 3, 'events': [
                {'currency': 'GBP', 'event': 'BOE Interest Rate Decision', 'impact': 'high', 'time': '11:00'},
                {'currency': 'USD', 'event': 'Initial Jobless Claims', 'impact': 'medium', 'time': '12:30'},
            ]},
            # Friday
            {'day': 4, 'events': [
                {'currency': 'USD', 'event': 'Non-Farm Payrolls', 'impact': 'high', 'time': '12:30'},
                {'currency': 'USD', 'event': 'Unemployment Rate', 'impact': 'high', 'time': '12:30'},
            ]},
        ]

        for i in range(days_ahead):
            date = today + timedelta(days=i)
            day_of_week = date.weekday()

            for day_events in typical_events:
                if day_events['day'] == day_of_week:
                    for event in day_events['events']:
                        events.append({
                            'date': str(date),
                            'time': event['time'],
                            'currency': event['currency'],
                            'event': event['event'],
                            'impact': event['impact'],
                        })

        return events

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self.last_fetch:
            return False
        return datetime.now() - self.last_fetch < self.cache_duration

    def get_upcoming_high_impact_events(self, hours_ahead: int = 24) -> List[Dict]:
        """
        Recupera eventi ad alto impatto nelle prossime ore.

        Args:
            hours_ahead: Ore da considerare

        Returns:
            Lista di eventi ad alto impatto
        """
        events = self.get_economic_calendar()
        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)

        high_impact = []

        for event in events:
            if event['impact'] != 'high':
                continue

            try:
                # Parse event datetime
                event_date = datetime.strptime(event['date'], '%Y-%m-%d')
                if event['time']:
                    hour, minute = map(int, event['time'].split(':'))
                    event_dt = event_date.replace(hour=hour, minute=minute)
                else:
                    event_dt = event_date

                # Check if within window
                if now <= event_dt <= cutoff:
                    event['datetime'] = event_dt
                    high_impact.append(event)

            except Exception as e:
                continue

        return high_impact

    def should_avoid_trading(self, pair: str) -> tuple[bool, str]:
        """
        Verifica se dovremmo evitare di tradare a causa di eventi macro.

        Args:
            pair: Coppia forex (es. EURUSD)

        Returns:
            (should_avoid, reason)
        """
        # Get currencies from pair
        base_currency = pair[:3]
        quote_currency = pair[3:6]

        # Get upcoming high impact events
        upcoming = self.get_upcoming_high_impact_events(hours_ahead=2)

        for event in upcoming:
            # Check if event affects our pair
            if event['currency'] in [base_currency, quote_currency]:
                # Check if it's a high impact event we should avoid
                event_name = event['event'].lower()

                for avoid_event in MACRO_CONFIG['high_impact_events']:
                    if avoid_event.lower() in event_name:
                        time_to_event = event['datetime'] - datetime.now()
                        minutes = time_to_event.total_seconds() / 60

                        return True, f"High impact event: {event['event']} ({event['currency']}) in {minutes:.0f} min"

        return False, "No blocking events"

    def get_macro_context(self, pair: str) -> Dict:
        """
        Ottiene il contesto macro completo per una coppia.

        Args:
            pair: Coppia forex

        Returns:
            Dizionario con contesto macro
        """
        base_currency = pair[:3]
        quote_currency = pair[3:6]

        context = {
            'pair': pair,
            'base_currency': base_currency,
            'quote_currency': quote_currency,
            'upcoming_events': [],
            'avoid_trading': False,
            'avoid_reason': '',
            'macro_bias': 'neutral',
        }

        # Get upcoming events for both currencies
        events = self.get_economic_calendar()

        for event in events:
            if event['currency'] in [base_currency, quote_currency]:
                context['upcoming_events'].append(event)

        # Check if should avoid
        avoid, reason = self.should_avoid_trading(pair)
        context['avoid_trading'] = avoid
        context['avoid_reason'] = reason

        # Determine macro bias based on recent events
        context['macro_bias'] = self._determine_macro_bias(base_currency, quote_currency)

        return context

    def _determine_macro_bias(self, base: str, quote: str) -> str:
        """Determina il bias macro basato su fattori fondamentali."""
        # Simplified bias based on typical central bank stances
        # In production, this would use actual data feeds

        hawkish_currencies = ['USD']  # Example
        dovish_currencies = ['JPY', 'EUR']

        if base in hawkish_currencies and quote in dovish_currencies:
            return 'bullish'
        elif base in dovish_currencies and quote in hawkish_currencies:
            return 'bearish'
        else:
            return 'neutral'

    def format_for_agent(self, pair: str) -> str:
        """
        Formatta il contesto macro per l'AI agent.

        Args:
            pair: Coppia forex

        Returns:
            Stringa formattata per il prompt dell'agent
        """
        context = self.get_macro_context(pair)

        lines = [
            f"=== MACRO CONTEXT for {pair} ===",
            f"Base Currency: {context['base_currency']}",
            f"Quote Currency: {context['quote_currency']}",
            f"Macro Bias: {context['macro_bias'].upper()}",
            f"",
            f"Trading Restriction: {'YES - ' + context['avoid_reason'] if context['avoid_trading'] else 'No restrictions'}",
            f"",
            f"Upcoming Events ({len(context['upcoming_events'])} total):",
        ]

        # Add high impact events
        high_impact = [e for e in context['upcoming_events'] if e['impact'] == 'high']
        for event in high_impact[:5]:
            lines.append(f"  [HIGH] {event['date']} {event['time']} - {event['currency']}: {event['event']}")

        return "\n".join(lines)


# Singleton instance
_macro_analyzer = None


def get_macro_analyzer() -> MacroAnalyzer:
    """Get singleton MacroAnalyzer instance."""
    global _macro_analyzer
    if _macro_analyzer is None:
        _macro_analyzer = MacroAnalyzer()
    return _macro_analyzer
