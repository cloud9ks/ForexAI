"""
================================================================================
AI TRADING AGENT - NEWS SENTIMENT ANALYZER
================================================================================
Analizza news e sentiment di mercato.

Fonti:
- NewsAPI.org
- RSS Feeds (Reuters, Bloomberg)
- Twitter/X sentiment (opzionale)
================================================================================
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import re
from collections import defaultdict

from .config import NEWS_CONFIG, NEWS_API_KEY

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """Analizza sentiment dalle news di mercato."""

    def __init__(self):
        self.news_cache = {}
        self.last_fetch = None
        self.cache_duration = timedelta(minutes=30)

        # Sentiment keywords
        self.bullish_keywords = [
            'surge', 'rally', 'soar', 'jump', 'gain', 'rise', 'bullish',
            'optimism', 'growth', 'strong', 'hawkish', 'tighten', 'hike',
            'beat expectations', 'better than expected', 'upbeat'
        ]

        self.bearish_keywords = [
            'plunge', 'crash', 'fall', 'drop', 'decline', 'bearish',
            'pessimism', 'weakness', 'weak', 'dovish', 'cut', 'ease',
            'miss expectations', 'worse than expected', 'downbeat'
        ]

        # Currency-specific keywords
        self.currency_keywords = {
            'USD': ['dollar', 'usd', 'fed', 'federal reserve', 'us economy', 'america'],
            'EUR': ['euro', 'eur', 'ecb', 'eurozone', 'european', 'germany'],
            'GBP': ['pound', 'gbp', 'sterling', 'boe', 'bank of england', 'uk', 'britain'],
            'JPY': ['yen', 'jpy', 'boj', 'bank of japan', 'japan'],
            'AUD': ['aussie', 'aud', 'rba', 'australia', 'australian'],
            'CHF': ['franc', 'chf', 'snb', 'swiss', 'switzerland'],
            'CAD': ['loonie', 'cad', 'boc', 'canada', 'canadian'],
            'NZD': ['kiwi', 'nzd', 'rbnz', 'new zealand'],
        }

    def fetch_news(self, query: str = "forex currency", max_results: int = 20) -> List[Dict]:
        """
        Fetch news da NewsAPI.

        Args:
            query: Query di ricerca
            max_results: Numero massimo di risultati

        Returns:
            Lista di articoli
        """
        if not NEWS_API_KEY:
            logger.warning("NEWS_API_KEY not set, using mock data")
            return self._get_mock_news()

        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': max_results,
                'apiKey': NEWS_API_KEY,
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data.get('status') == 'ok':
                return data.get('articles', [])
            else:
                logger.error(f"NewsAPI error: {data.get('message')}")
                return self._get_mock_news()

        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return self._get_mock_news()

    def _get_mock_news(self) -> List[Dict]:
        """Returns mock news for testing when API is not available."""
        return [
            {
                'title': 'Fed signals potential rate pause amid economic uncertainty',
                'description': 'Federal Reserve officials indicated they may hold rates steady.',
                'publishedAt': datetime.now().isoformat(),
                'source': {'name': 'Reuters'}
            },
            {
                'title': 'EUR/USD rises on ECB hawkish comments',
                'description': 'The euro gained against the dollar after ECB members suggested more tightening.',
                'publishedAt': datetime.now().isoformat(),
                'source': {'name': 'Bloomberg'}
            },
            {
                'title': 'UK inflation remains sticky, GBP strengthens',
                'description': 'Higher than expected inflation data pushed pound higher.',
                'publishedAt': datetime.now().isoformat(),
                'source': {'name': 'FT'}
            },
        ]

    def analyze_sentiment(self, text: str) -> float:
        """
        Analizza il sentiment di un testo.

        Args:
            text: Testo da analizzare

        Returns:
            Score da -1 (bearish) a +1 (bullish)
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        bullish_count = sum(1 for kw in self.bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in self.bearish_keywords if kw in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        # Score from -1 to +1
        score = (bullish_count - bearish_count) / total

        return score

    def get_currency_mentions(self, text: str) -> Dict[str, int]:
        """
        Conta le menzioni di ogni valuta nel testo.

        Args:
            text: Testo da analizzare

        Returns:
            Dict con conteggio per ogni valuta
        """
        text_lower = text.lower()
        mentions = defaultdict(int)

        for currency, keywords in self.currency_keywords.items():
            for keyword in keywords:
                mentions[currency] += len(re.findall(r'\b' + keyword + r'\b', text_lower))

        return dict(mentions)

    def analyze_news_for_pair(self, pair: str) -> Dict:
        """
        Analizza tutte le news rilevanti per una coppia.

        Args:
            pair: Coppia forex (es. EURUSD)

        Returns:
            Dizionario con analisi completa
        """
        base = pair[:3]
        quote = pair[3:6]

        # Fetch news
        news = self.fetch_news(query=f"{base} {quote} forex")

        analysis = {
            'pair': pair,
            'base_currency': base,
            'quote_currency': quote,
            'news_count': len(news),
            'base_sentiment': 0.0,
            'quote_sentiment': 0.0,
            'overall_bias': 'neutral',
            'confidence': 0.0,
            'key_headlines': [],
            'timestamp': datetime.now().isoformat(),
        }

        if not news:
            return analysis

        base_sentiments = []
        quote_sentiments = []

        for article in news:
            title = article.get('title', '')
            description = article.get('description', '')
            full_text = f"{title} {description}"

            # Get sentiment
            sentiment = self.analyze_sentiment(full_text)

            # Get currency mentions
            mentions = self.get_currency_mentions(full_text)

            # Assign sentiment to currencies based on mentions
            if mentions.get(base, 0) > 0:
                base_sentiments.append(sentiment)
            if mentions.get(quote, 0) > 0:
                quote_sentiments.append(sentiment)

            # Store key headlines
            if abs(sentiment) > 0.3:
                analysis['key_headlines'].append({
                    'title': title,
                    'sentiment': sentiment,
                    'source': article.get('source', {}).get('name', 'Unknown')
                })

        # Calculate average sentiments
        if base_sentiments:
            analysis['base_sentiment'] = sum(base_sentiments) / len(base_sentiments)
        if quote_sentiments:
            analysis['quote_sentiment'] = sum(quote_sentiments) / len(quote_sentiments)

        # Determine overall bias
        # If base is more bullish than quote, pair should go up
        sentiment_diff = analysis['base_sentiment'] - analysis['quote_sentiment']

        if sentiment_diff > NEWS_CONFIG['bullish_threshold']:
            analysis['overall_bias'] = 'bullish'
        elif sentiment_diff < NEWS_CONFIG['bearish_threshold']:
            analysis['overall_bias'] = 'bearish'
        else:
            analysis['overall_bias'] = 'neutral'

        # Confidence based on news count and sentiment strength
        analysis['confidence'] = min(1.0, (len(news) / 10) * abs(sentiment_diff))

        return analysis

    def get_market_sentiment(self) -> Dict:
        """
        Ottiene il sentiment generale del mercato.

        Returns:
            Dizionario con sentiment per ogni valuta
        """
        news = self.fetch_news(query="forex currency central bank")

        sentiments = {currency: [] for currency in self.currency_keywords.keys()}

        for article in news:
            title = article.get('title', '')
            description = article.get('description', '')
            full_text = f"{title} {description}"

            sentiment = self.analyze_sentiment(full_text)
            mentions = self.get_currency_mentions(full_text)

            for currency, count in mentions.items():
                if count > 0:
                    sentiments[currency].append(sentiment)

        # Calculate averages
        result = {}
        for currency, scores in sentiments.items():
            if scores:
                avg = sum(scores) / len(scores)
                result[currency] = {
                    'sentiment': avg,
                    'news_count': len(scores),
                    'bias': 'bullish' if avg > 0.1 else ('bearish' if avg < -0.1 else 'neutral')
                }
            else:
                result[currency] = {
                    'sentiment': 0.0,
                    'news_count': 0,
                    'bias': 'neutral'
                }

        return result

    def format_for_agent(self, pair: str) -> str:
        """
        Formatta l'analisi news per l'AI agent.

        Args:
            pair: Coppia forex

        Returns:
            Stringa formattata per il prompt
        """
        analysis = self.analyze_news_for_pair(pair)

        lines = [
            f"=== NEWS SENTIMENT for {pair} ===",
            f"News analyzed: {analysis['news_count']}",
            f"",
            f"Base ({analysis['base_currency']}) Sentiment: {analysis['base_sentiment']:+.2f}",
            f"Quote ({analysis['quote_currency']}) Sentiment: {analysis['quote_sentiment']:+.2f}",
            f"",
            f"Overall Bias: {analysis['overall_bias'].upper()}",
            f"Confidence: {analysis['confidence']:.1%}",
            f"",
            f"Key Headlines:",
        ]

        for headline in analysis['key_headlines'][:5]:
            sentiment_emoji = "📈" if headline['sentiment'] > 0 else "📉"
            lines.append(f"  {sentiment_emoji} [{headline['source']}] {headline['title'][:60]}...")

        if not analysis['key_headlines']:
            lines.append("  No significant headlines")

        return "\n".join(lines)


# Singleton instance
_news_analyzer = None


def get_news_analyzer() -> NewsSentimentAnalyzer:
    """Get singleton NewsSentimentAnalyzer instance."""
    global _news_analyzer
    if _news_analyzer is None:
        _news_analyzer = NewsSentimentAnalyzer()
    return _news_analyzer
