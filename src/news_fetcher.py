from typing import List, Dict, Optional

from .rss_fetcher import fetch_latest_news
from .wikipedia_fetcher import (
    fetch_wikipedia_year,
    fetch_wikipedia_events,
    parse_wikipedia_fragment,
)

__all__ = [
    "fetch_latest_news",
    "fetch_wikipedia_year",
    "fetch_wikipedia_events",
    "parse_wikipedia_fragment",
]
