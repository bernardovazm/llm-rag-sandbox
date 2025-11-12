from typing import List, Dict, Optional
import requests
import datetime
import xml.etree.ElementTree as ET

DEFAULT_SOURCES = [
    "https://rss.cnn.com/rss/edition.rss",
    "http://feeds.reuters.com/reuters/topNews",
]


def _parse_rss(feed_xml: str, limit: int) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(feed_xml)
    except Exception:
        return items
    for item in root.findall('.//item')[:limit]:
        title = item.findtext('title') or ''
        desc = item.findtext('description') or item.findtext('summary') or ''
        link = item.findtext('link') or ''
        pub = item.findtext('pubDate') or ''
        items.append({'title': title.strip(), 'summary': desc.strip(), 'link': link.strip(), 'published': pub.strip()})
    return items


def fetch_latest_news(sources: Optional[List[str]] = None, limit_per_source: int = 5) -> List[Dict[str, str]]:
    srcs = sources or DEFAULT_SOURCES
    all_items: List[Dict[str, str]] = []
    now = datetime.datetime.utcnow().isoformat() + "Z"
    for url in srcs:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            items = _parse_rss(r.text, limit_per_source)
            for it in items:
                it['fetched_at'] = now
                text = it.get('title', '') + '\n\n' + (it.get('summary', '') or '') + '\n\n' + it.get('link', '')
                it['text'] = text
                all_items.append(it)
        except Exception:
            continue
    return all_items
