#!/usr/bin/env python3
"""
================================================================================
REAL NEWS ENABLED - Drop-in Replacement
================================================================================

This file replaces your scout tools with REAL news sources.

USAGE:
1. Save this as real_news_tools.py in your NewsScrape folder
2. In scout_council_swarm.py, REPLACE:

   from crewai.tools import BaseTool

   WITH:

   from real_news_tools import NewsSearchTool, SocialMediaMonitorTool, AcademicSearchTool

3. That's it! Run normally.

The tools will automatically:
- Use REAL news APIs if keys are configured
- Fall back to RSS if NewsAPI rate limited
- Fall back to mock only if all else fails
================================================================================
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict
from urllib.parse import quote

# Try importing optional dependencies
try:
    from newsapi import NewsApiClient
    HAS_NEWSAPI = True
except ImportError:
    HAS_NEWSAPI = False

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

try:
    import praw
    HAS_REDDIT = True
except ImportError:
    HAS_REDDIT = False

try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False

from crewai.tools import BaseTool

# ============================================================================
# REAL NEWS SEARCH
# ============================================================================

class NewsSearchTool(BaseTool):
    """Search real news from NewsAPI + RSS feeds."""
    name: str = "news_search"
    description: str = "Search for real news articles from NewsAPI and major RSS feeds"

    _instance = None
    _newsapi = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_api()
        return cls._instance

    def _init_api(self):
        """Initialize NewsAPI if key available."""
        api_key = os.getenv("NEWSAPI_KEY")
        if HAS_NEWSAPI and api_key:
            try:
                self._newsapi = NewsApiClient(api_key=api_key)
                print("    ✅ NewsAPI connected")
            except Exception as e:
                print(f"    ⚠️  NewsAPI init failed: {e}")
                self._newsapi = None
        else:
            if not api_key:
                print("    ⚠️  NEWSAPI_KEY not set in .env")
            if not HAS_NEWSAPI:
                print("    ⚠️  newsapi-python not installed")
            self._newsapi = None

    def _run(self, query: str) -> str:
        results = []

        # Try NewsAPI first
        if self._newsapi:
            try:
                newsapi_results = self._search_newsapi(query)
                results.extend(newsapi_results)
            except Exception as e:
                print(f"    ⚠️  NewsAPI search failed: {e}")

        # Try RSS feeds
        if HAS_FEEDPARSER:
            try:
                rss_results = self._search_rss(query)
                results.extend(rss_results)
            except Exception as e:
                print(f"    ⚠️  RSS search failed: {e}")

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(r)

        if unique:
            print(f"    ✅ Found {len(unique)} real news articles")
            return json.dumps(unique[:10])

        # Emergency fallback
        print(f"    ⚠️  No real news found, using Google News link")
        return json.dumps([{
            "title": f"Search Google News: {query}",
            "url": f"https://news.google.com/search?q={quote(query)}",
            "body": f"No API results. Search Google News directly for '{query}'",
            "source": "Google News",
            "reliability": "tier_2_reputable"
        }])

    def _search_newsapi(self, query: str) -> List[Dict]:
        """Search NewsAPI."""
        from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')

        response = self._newsapi.get_everything(
            q=query,
            from_param=from_date,
            language='en',
            sort_by='relevancy',
            page_size=10
        )

        articles = []
        for article in response.get('articles', []):
            articles.append({
                "title": article.get('title', ''),
                "url": article.get('url', ''),
                "body": article.get('description', '') or article.get('content', '')[:300],
                "source": article.get('source', {}).get('name', 'Unknown'),
                "published_at": article.get('publishedAt', ''),
                "reliability": "tier_1_verified" if article.get('source', {}).get('name') in ['Reuters', 'Associated Press', 'BBC News'] else "tier_2_reputable"
            })

        print(f"    📰 NewsAPI: {len(articles)} articles")
        return articles

    def _search_rss(self, query: str) -> List[Dict]:
        """Search major RSS feeds."""
        feeds = {
            "bbc": "http://feeds.bbci.co.uk/news/world/rss.xml",
            "reuters": "http://feeds.reuters.com/reuters/topNews",
            "guardian": "https://www.theguardian.com/world/rss",
        }

        results = []
        query_lower = query.lower()

        for name, url in feeds.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    if query_lower in title.lower() or any(word in title.lower() for word in query_lower.split()):
                        results.append({
                            "title": title,
                            "url": entry.get('link', ''),
                            "body": summary[:300],
                            "source": name.upper(),
                            "published_at": entry.get('published', ''),
                            "reliability": "tier_1_verified" if name == "reuters" else "tier_2_reputable"
                        })
            except:
                continue

        print(f"    📡 RSS: {len(results)} articles")
        return results

# ============================================================================
# REAL SOCIAL MEDIA
# ============================================================================

class SocialMediaMonitorTool(BaseTool):
    """Monitor real Reddit trends."""
    name: str = "social_media_monitor"
    description: str = "Monitor real social media trends from Reddit"

    _instance = None
    _reddit = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_reddit()
        return cls._instance

    def _init_reddit(self):
        """Initialize Reddit if credentials available."""
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")

        if HAS_REDDIT and client_id and client_secret:
            try:
                self._reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent="NewsReels/1.0"
                )
                print("    ✅ Reddit API connected")
            except Exception as e:
                print(f"    ⚠️  Reddit init failed: {e}")
                self._reddit = None
        else:
            self._reddit = None

    def _run(self, platform: str, topic: str) -> str:
        results = []

        # Search Reddit (real data)
        if self._reddit and platform.lower() in ['reddit', 'all']:
            try:
                reddit_results = self._search_reddit(topic)
                results.extend(reddit_results)
            except Exception as e:
                print(f"    ⚠️  Reddit search failed: {e}")

        if results:
            print(f"    ✅ Found {len(results)} real Reddit trends")
            return json.dumps(results)

        # Fallback
        print(f"    ⚠️  Using fallback trends")
        return json.dumps([{
            "headline": f"Trending: {topic.title()}",
            "platform": "Social Media",
            "engagement": "High engagement",
            "confidence": 0.6,
            "urgency": 0.5,
            "novelty": 0.7
        }])

    def _search_reddit(self, topic: str) -> List[Dict]:
        """Search Reddit hot posts."""
        results = []
        subreddits = ['news', 'worldnews', 'technology']

        for sub_name in subreddits:
            try:
                sub = self._reddit.subreddit(sub_name)
                for post in sub.search(topic, sort='hot', time_filter='day', limit=2):
                    if post.score > 100:
                        results.append({
                            "headline": post.title,
                            "platform": "Reddit",
                            "engagement": f"{post.score} upvotes, {post.num_comments} comments",
                            "subreddit": f"r/{sub_name}",
                            "confidence": min(post.upvote_ratio, 0.9),
                            "urgency": 0.7 if post.score > 1000 else 0.5,
                            "novelty": 0.8,
                            "url": f"https://reddit.com{post.permalink}"
                        })
            except:
                continue

        print(f"    📱 Reddit: {len(results)} posts")
        return results

# ============================================================================
# REAL ACADEMIC SEARCH
# ============================================================================

class AcademicSearchTool(BaseTool):
    """Search real arXiv papers."""
    name: str = "academic_search"
    description: str = "Search real academic papers from arXiv"

    def _run(self, query: str, source_type: str = "all") -> str:
        if not HAS_ARXIV:
            return json.dumps([])

        try:
            search = arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            results = []
            for paper in search.results():
                results.append({
                    "title": paper.title,
                    "url": paper.pdf_url,
                    "body": paper.summary[:300],
                    "source": "arXiv",
                    "published_at": paper.published.isoformat(),
                    "reliability": "tier_1_verified",
                    "authors": [str(a) for a in paper.authors[:3]]
                })

            print(f"    📚 arXiv: {len(results)} papers")
            return json.dumps(results)

        except Exception as e:
            print(f"    ⚠️  arXiv error: {e}")
            return json.dumps([])

# ============================================================================
# AUTO-DETECTION
# ============================================================================

def check_real_news_available():
    """Check what real news sources are available."""
    status = {
        "NewsAPI": HAS_NEWSAPI and bool(os.getenv("NEWSAPI_KEY")),
        "RSS Feeds": HAS_FEEDPARSER,
        "Reddit": HAS_REDDIT and bool(os.getenv("REDDIT_CLIENT_ID")),
        "arXiv": HAS_ARXIV
    }

    print("\n📊 Real News Sources Available:")
    for source, available in status.items():
        icon = "✅" if available else "❌"
        print(f"   {icon} {source}")

    return status

# Run check on import
if __name__ == "__main__":
    check_real_news_available()