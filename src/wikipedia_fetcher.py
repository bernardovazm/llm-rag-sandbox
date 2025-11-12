from typing import List, Optional
import requests
import re
import html

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

USER_AGENT = "news-fetcher/1.0 (+https://example.org)"


def fetch_wikipedia_year(year: Optional[int] = None) -> List[dict]:
    try:
        y = year or __import__("datetime").datetime.utcnow().year
    except Exception:
        import datetime as _dt

        y = year or _dt.datetime.now().year

    urls = [f"https://pt.wikipedia.org/wiki/{y}", f"https://en.wikipedia.org/wiki/{y}"]

    for url in urls:
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
            if r.status_code != 200:
                continue
            html_text = r.text
            m = re.search(r'<meta\s+property="og:description"\s+content="([^"]+)"', html_text, flags=re.I)
            if m:
                summary = re.sub(r"\s+", " ", m.group(1)).strip()
            else:
                m2 = re.search(r'<meta\s+name="description"\s+content="([^"]+)"', html_text, flags=re.I)
                if m2:
                    summary = re.sub(r"\s+", " ", m2.group(1)).strip()
                else:
                    summary = None
                    for match in re.finditer(r"<p>(.*?)</p>", html_text, flags=re.S | re.I):
                        inner = re.sub(r"<.*?>", "", match.group(1)).strip()
                        if inner and len(inner) > 60 and not inner.lower().startswith("nota"):
                            summary = re.sub(r"\s+", " ", inner)
                            break

            if not summary:
                continue

            item = {
                "title": f"Wikipedia: {y} ({'pt' if 'pt.wikipedia' in url else 'en'})",
                "summary": summary,
                "link": url,
                "published": str(y),
                "fetched_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "text": f"Wikipedia {y}\n\n" + summary + "\n\n" + url,
            }
            return [item]
        except Exception:
            continue

    return []


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"(.+?[\.\?\!])(?:\s|$)", s)
    if m:
        return m.group(1).strip()
    return s[:200].rsplit(" ", 1)[0] + ("..." if len(s) > 200 else "")


def _clean_text_artifacts(s: str) -> str:
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"<[^>]*>", "", s)
    s = re.sub(r"/td>", "", s)
    s = re.sub(r"<tr[^>]*>", "", s)
    s = re.sub(r"\bpadding:[^\s>]+", "", s)
    s = re.sub(r"\badding:[^\s>]+", "", s)
    s = re.sub(r"\btd>\s*", "", s)
    s = re.sub(r"<tr[^>]*>", "", s)
    s = re.sub(r"[<>\"\x00-\x1f]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _events_from_dom(html_text: str, max_items: int) -> List[str]:
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    month_link_re = re.compile(r'_(?:[0-9]{1,2})$')
    date_text_re = re.compile(r'^(?:\d{1,2}\s+de\s+\w+|\w+\s+\d{1,2})$', flags=re.I)
    content = soup.find(id="mw-content-text") or soup
    res = []
    seen = set()
    for li in content.find_all('li'):
        first_a = None
        for child in li.children:
            if getattr(child, 'name', None) == 'a':
                first_a = child
                break
            if getattr(child, 'name', None) in ('span', 'small'):
                continue
            if isinstance(child, str) and date_text_re.match(child.strip()):
                first_a = None
                break

        if not first_a:
            continue
        href = first_a.get('href', '')
        atext = first_a.get_text(separator=' ').strip()
        if not (month_link_re.search(href) or date_text_re.match(atext)):
            continue
        date_text = atext
        nested = li.find('ul') or li.find('ol')
        items = []
        if nested:
            for inner in nested.find_all('li', recursive=False):
                try:
                    inner_clone = BeautifulSoup(str(inner), 'html.parser')
                    for bad in inner_clone.find_all(['table', 'tr', 'td', 'style', 'script', 'sup']):
                        bad.decompose()
                    inner_text = inner_clone.get_text(separator=' ').strip()
                except Exception:
                    inner_text = inner.get_text(separator=' ').strip()
                items.append(inner_text)
        else:
            try:
                li_clone = BeautifulSoup(str(li), 'html.parser')
                for bad in li_clone.find_all(['table', 'tr', 'td', 'style', 'script', 'sup']):
                    bad.decompose()
                full = li_clone.get_text(separator=' ').strip()
            except Exception:
                full = li.get_text(separator=' ').strip()
            if full.startswith(date_text):
                rest = full[len(date_text):].strip(' -–—:')
                if rest:
                    items.append(rest)

        for raw in items:
            txt = _clean_text_artifacts(raw)
            if not txt:
                continue
            sent = _first_sentence(txt)
            candidate = f"{date_text}: {sent}" if date_text else sent
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if candidate and candidate not in seen:
                res.append(candidate)
                seen.add(candidate)
                if len(res) >= max_items:
                    return res
    return res


def _events_from_section_html(html_text: str, L: str, y: int, max_items: int) -> List[str]:
    sec_re = r'(?:<span[^>]+class="mw-headline"[^>]*>\s*Eventos\s*</span>)'
    if L == "en":
        sec_re = r'(?:<span[^>]+class="mw-headline"[^>]*>\s*Events\s*</span>)'
    m = re.search(sec_re, html_text, flags=re.I)
    results = []
    seen = set()
    if not m:
        return results
    start = m.end()
    tail = html_text[start:start + 20000]
    end_m = re.search(r'<h2[^>]*>', tail, flags=re.I)
    section_html = tail if not end_m else tail[: end_m.start()]
    items = []
    for mm in re.finditer(r'<li>(.*?)</li>', section_html, flags=re.S | re.I):
        items.append(mm.group(1))
    if not items:
        for mm in re.finditer(r'<p>(.*?)</p>', section_html, flags=re.S | re.I):
            items.append(mm.group(1))

    for raw in items:
        txt = re.sub(r'<[^>]*>', '', raw)
        txt = re.sub(r'\[[^\]]*\]', '', txt)
        txt = html.unescape(txt).strip()
        date_match = re.match(r'^(\d{1,2}\s+de\s+\w+|\w+\s+\d{1,2})\s*[\-–—:]*\s*(.*)$', txt, flags=re.I)
        if date_match:
            date_part = date_match.group(1).strip()
            rest = date_match.group(2).strip()
            candidate = f"{date_part}: {rest}"
        else:
            candidate = _first_sentence(txt)
        candidate = re.sub(r'\s+', ' ', candidate).strip()
        if not candidate or len(candidate) < 10:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        results.append(candidate)
        if len(results) >= max_items:
            return results
    return results


def _events_by_page_scan(html_text: str, max_items: int) -> List[str]:
    results = []
    seen = set()
    page_items = []
    for mm in re.finditer(r'<li>([^<]{0,400}?)</li>', html_text, flags=re.S | re.I):
        page_items.append(mm.group(1))
    if not page_items:
        for mm in re.finditer(r'<p>([^<]{0,400}?)</p>', html_text, flags=re.S | re.I):
            page_items.append(mm.group(1))

    date_re = re.compile(r'^(\d{1,2}\s+de\s+\w+|\w+\s+\d{1,2})', flags=re.I)
    for raw in page_items:
        txt = re.sub(r'<[^>]*>', '', raw)
        txt = re.sub(r'\[[^\]]*\]', '', txt)
        txt = html.unescape(txt).strip()
        if not txt:
            continue
        mdate = date_re.match(txt)
        if not mdate:
            continue
        date_part = mdate.group(1).strip()
        rest = txt[mdate.end():].lstrip(' -–—:')
        if rest:
            candidate = f"{date_part}: {_first_sentence(rest)}"
        else:
            candidate = date_part
        candidate = re.sub(r'\s+', ' ', candidate).strip()
        if len(candidate) < 10:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        results.append(candidate)
        if len(results) >= max_items:
            return results

    months_pt = "janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
    months_en = "january|february|march|april|may|june|july|august|september|october|november|december"
    date_pattern_pt = re.compile(rf'\b(\d{{1,2}}\s+de\s+(?:{months_pt}))\b', flags=re.I)
    date_pattern_en = re.compile(rf'\b(\d{{1,2}}\s+(?:{months_en})|(?:{months_en})\s+\d{{1,2}})\b', flags=re.I)
    for dp in date_pattern_pt.finditer(html_text):
        start = max(0, dp.start() - 120)
        snippet = html_text[start: dp.end() + 300]
        txt = re.sub(r'<[^>]*>', '', snippet)
        txt = re.sub(r'\[[^\]]*\]', '', txt)
        txt = html.unescape(txt).strip()
        sent = _first_sentence(txt)
        if sent and sent not in seen:
            results.append(f"{dp.group(1)}: {sent}")
            seen.add(sent)
            if len(results) >= max_items:
                return results
    for dp in date_pattern_en.finditer(html_text):
        start = max(0, dp.start() - 120)
        snippet = html_text[start: dp.end() + 300]
        txt = re.sub(r'<[^>]*>', '', snippet)
        txt = re.sub(r'\[[^\]]*\]', '', txt)
        txt = html.unescape(txt).strip()
        sent = _first_sentence(txt)
        if sent and sent not in seen:
            results.append(f"{dp.group(1)}: {sent}")
            seen.add(sent)
            if len(results) >= max_items:
                return results
    return results


def fetch_wikipedia_events(year: Optional[int] = None, lang: str = "pt", max_items: int = 10) -> List[str]:
    try:
        y = year or __import__("datetime").datetime.utcnow().year
    except Exception:
        import datetime as _dt

        y = year or _dt.datetime.now().year

    langs = [lang]
    if lang != "en":
        langs.append("en")

    events: List[str] = []
    for L in langs:
        url = f"https://{L}.wikipedia.org/wiki/{y}"
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
            if r.status_code != 200:
                continue
            html_text = r.text
            dom_events = _events_from_dom(html_text, max_items)
            for e in dom_events:
                if e not in events:
                    events.append(e)
                if len(events) >= max_items:
                    return events
            sec_events = _events_from_section_html(html_text, L, y, max_items - len(events))
            for e in sec_events:
                if e not in events:
                    events.append(e)
                if len(events) >= max_items:
                    return events
            scan_events = _events_by_page_scan(html_text, max_items - len(events))
            for e in scan_events:
                if e not in events:
                    events.append(e)
                if len(events) >= max_items:
                    return events
        except Exception:
            continue
    return events


def parse_wikipedia_fragment(html_fragment: str, max_items: int = 10) -> List[str]:
    results: List[str] = []
    seen = set()
    try:
        soup = BeautifulSoup(html_fragment, "html.parser") if BeautifulSoup is not None else None
    except Exception:
        soup = None

    if soup is None:
        pattern = re.compile(r'<a\s+href="(/wiki/[^"\s]+_\d{1,2})"[^>]*>([^<]+)</a>\s*<ul>(.*?)</ul>', flags=re.S | re.I)
        for m in pattern.finditer(html_fragment):
            date_text = m.group(2).strip()
            inner_html = m.group(3)
            for li_m in re.finditer(r'<li>(.*?)</li>', inner_html, flags=re.S | re.I):
                raw = li_m.group(1)
                txt = re.sub(r'<[^>]*>', '', raw)
                txt = re.sub(r'\[[^\]]*\]', '', txt)
                txt = html.unescape(txt).strip()
                if not txt:
                    continue
                sent = _first_sentence(txt)
                candidate = f"{date_text}: {sent}"
                candidate = re.sub(r'\s+', ' ', candidate).strip()
                if candidate and candidate not in seen:
                    results.append(candidate)
                    seen.add(candidate)
                    if len(results) >= max_items:
                        return results
        return results

    date_link_re = re.compile(r"/wiki/[^_]+_\d{1,2}$")

    for a in soup.find_all('a', href=True):
        href = a['href']
        if not date_link_re.search(href):
            continue
        date_text = a.get_text(separator=' ').strip()
        li = a.find_parent('li')
        if li is None:
            continue

        nested = li.find('ul') or li.find('ol')
        items = []
        if nested:
            for inner in nested.find_all('li', recursive=False):
                try:
                    inner_clone = BeautifulSoup(str(inner), 'html.parser')
                    for bad in inner_clone.find_all(['sup', 'table']):
                        bad.decompose()
                    txt = inner_clone.get_text(separator=' ').strip()
                except Exception:
                    txt = inner.get_text(separator=' ').strip()
                items.append(txt)
        else:
            try:
                li_clone = BeautifulSoup(str(li), 'html.parser')
                for bad in li_clone.find_all(['sup', 'table']):
                    bad.decompose()
                full = li_clone.get_text(separator=' ').strip()
            except Exception:
                full = li.get_text(separator=' ').strip()
            if full.startswith(date_text):
                rest = full[len(date_text):].strip(' -–—:')
                if rest:
                    items.append(rest)

        for raw in items:
            txt = re.sub(r'\[[^\]]*\]', '', raw)
            txt = re.sub(r'\s+', ' ', txt).strip()
            txt = html.unescape(txt)
            if not txt:
                continue
            sent = _first_sentence(txt)
            candidate = f"{date_text}: {sent}" if date_text else sent
            candidate = re.sub(r'\s+', ' ', candidate).strip()
            if candidate and candidate not in seen:
                results.append(candidate)
                seen.add(candidate)
                if len(results) >= max_items:
                    return results

    return results
