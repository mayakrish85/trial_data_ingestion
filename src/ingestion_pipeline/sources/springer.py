# IGNORE FOR NOW -- THE PIPELINE DOES NOT SUPPORT SPRINGER FULLTEXT.
from __future__ import annotations

import os
import re
import html
import time
from typing import Tuple, Optional, Dict, Any
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup, element

from ingestion_pipeline.preprocessing.xml_cleaning import (
    section_to_nested_dict,
    collapse_body_to_section,
    extract_abstract_text,
)

# =============================================================================
# Rate limiting & robustness
# =============================================================================

class _RateLimiter:
    """Simple sliding-window limiter: at most `max_requests` per `window_secs`."""
    def __init__(self, max_requests: int, window_secs: int = 60):
        self.max = max(1, int(max_requests))
        self.win = max(1, int(window_secs))
        self._timestamps = deque()

    def acquire(self):
        now = time.time()
        # Drop timestamps older than window
        while self._timestamps and now - self._timestamps[0] > self.win:
            self._timestamps.popleft()
        # If at capacity, sleep until oldest timestamp exits window
        if len(self._timestamps) >= self.max:
            sleep_for = self.win - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.time())


def _parse_retry_after(header_val: Optional[str]) -> float:
    """Return seconds to wait from Retry-After header (seconds or HTTP-date)."""
    if not header_val:
        return 0.0
    # Seconds
    try:
        return float(header_val)
    except (TypeError, ValueError):
        pass
    # HTTP-date
    try:
        dt = parsedate_to_datetime(header_val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return 0.0


def _springer_session() -> requests.Session:
    """Session with connection pooling + retry that respects Retry-After."""
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "fulltext-pipeline/1.4"})
    return s


# Client-side RPM limiter (configurable via env)
_SPRINGER_RPM = int(os.getenv("SPRINGER_RPM", "90"))  # default ~1.5 rps
_SPRINGER_LIMITER = _RateLimiter(_SPRINGER_RPM, 60)

# Optional debug directory to dump raw responses (set SPRINGER_DEBUG_DIR)
_SPRINGER_DEBUG_DIR = os.getenv("SPRINGER_DEBUG_DIR") or ""


# =============================================================================
# XML helpers (namespace-agnostic + unescape)
# =============================================================================

def _find_ns(tag_or_soup, local: str):
    """Find first tag whose localname == local (ignores prefixes like jats:, ns2:)."""
    return tag_or_soup.find(lambda t: isinstance(t, element.Tag) and t.name.split(":")[-1] == local)

def _find_all_ns(tag_or_soup, local: str, recursive: bool = True):
    """Find all tags whose localname == local (ignores prefixes)."""
    return tag_or_soup.find_all(lambda t: isinstance(t, element.Tag) and t.name.split(":")[-1] == local,
                                recursive=recursive)

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _multi_unescape(s: str, max_rounds: int = 3) -> str:
    """Unescape HTML entities repeatedly up to max_rounds (handles double-escaped payloads)."""
    prev = s
    for _ in range(max_rounds):
        cur = html.unescape(prev)
        if cur == prev:
            break
        prev = cur
    return prev

def _soup_xml(payload: bytes | str) -> BeautifulSoup:
    return BeautifulSoup(payload, "lxml-xml")


def _extract_article_from_record(rec) -> Tuple[Optional[element.Tag], Optional[BeautifulSoup]]:
    """
    Given a <record>, return (article_tag, soup_for_that_article) handling:
      - direct <article> children
      - <xml> wrappers with escaped JATS
      - raw text fallback scanning for <article>...</article>
    """
    # A) direct JATS
    art = _find_ns(rec, "article")
    if art:
        return art, rec

    # B) <xml> wrapper with escaped content
    xml_tag = _find_ns(rec, "xml")
    if xml_tag:
        raw = xml_tag.get_text() or xml_tag.decode_contents(formatter="minimal")
        if raw:
            unescaped = _multi_unescape(raw)
            inner = _soup_xml(unescaped)
            art = _find_ns(inner, "article")
            if art:
                return art, inner

    # C) last-ditch: regex scan within record text
    txt = rec.get_text() or ""
    m = re.search(r"<article\b[\s\S]*?</article>", _multi_unescape(txt), flags=re.IGNORECASE)
    if m:
        inner = _soup_xml(m.group(0))
        art = _find_ns(inner, "article")
        if art:
            return art, inner

    return None, None


# =============================================================================
# Public API
# =============================================================================

def try_springer_jats(
    doi: str,
    timeout: int = 45,
    session: "requests.Session | None" = None,
) -> Tuple[Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]], Optional[str]]:
    """
    Fetch Springer OA JATS for DOI.

    Returns:
      (title, sections_dict, {"abstract": abstract_text}) on success
      (None, error_string) on failure

    Notes:
      - Namespace-agnostic tag matching (jats:article, ns2:body, etc.).
      - Handles escaped inner <xml> payloads (double-unescape if needed).
      - Chooses the record whose DOI matches the requested DOI when possible.
      - Enforces polite client-side rate limiting and honors Retry-After on 429.
    """
    api_key = os.getenv("SPRINGER_API_KEY", "")
    if not api_key:
        return None, "SPRINGER_API_KEY not set"

    base = "https://api.springernature.com/openaccess/jats"
    params = {"q": f"doi:{doi}", "api_key": api_key}
    sess = session or _springer_session()

    max_attempts = 6
    backoff = 0.75  # exponential fallback when no Retry-After is provided
    last_err = None
    resp = None

    for attempt in range(1, max_attempts + 1):
        _SPRINGER_LIMITER.acquire()
        try:
            resp = sess.get(base, params=params, timeout=timeout)
        except requests.RequestException as e:
            last_err = f"Springer request error: {e}"
            # Retry on network error
            if attempt < max_attempts:
                time.sleep(backoff ** attempt)
                continue
            return None, last_err

        # Success
        if resp.status_code == 200 and resp.content:
            break

        # 429 rate-limited: honor Retry-After
        if resp.status_code == 429:
            wait = _parse_retry_after(resp.headers.get("Retry-After")) or (backoff ** attempt)
            time.sleep(min(wait, 60.0))
            continue

        # Server errors: backoff
        if resp.status_code in (500, 502, 503, 504):
            if attempt < max_attempts:
                time.sleep(backoff ** attempt)
                continue
            return None, f"Springer HTTP {resp.status_code}"

        # Other codes: fail fast at last attempt
        last_err = f"Springer HTTP {resp.status_code}"
        if attempt < max_attempts:
            time.sleep(backoff ** attempt)
            continue
        return None, last_err

    if not resp or not (resp.content and resp.content.strip()):
        return None, last_err or "Springer: empty response"

    # Optional debug dump of the outer response
    if _SPRINGER_DEBUG_DIR:
        try:
            os.makedirs(_SPRINGER_DEBUG_DIR, exist_ok=True)
            with open(os.path.join(_SPRINGER_DEBUG_DIR, f"outer_{re.sub('[^a-zA-Z0-9]+','_', doi)}.xml"), "wb") as f:
                f.write(resp.content)
        except Exception:
            pass

    outer = _soup_xml(resp.content)
    records = _find_all_ns(outer, "record", recursive=True)
    if not records:
        return None, "Springer: no <record>"

    requested = _norm(doi)
    best: Optional[Tuple[element.Tag, BeautifulSoup]] = None

    for rec in records:
        art, soup = _extract_article_from_record(rec)
        if not art or not soup:
            continue

        # Prefer DOI-matching article
        art_doi_tag = soup.find(
            lambda t: isinstance(t, element.Tag)
            and t.name.split(":")[-1] == "article-id"
            and (t.get("pub-id-type") or "").lower() == "doi"
        )
        art_doi = _norm(art_doi_tag.get_text(strip=True) if art_doi_tag else None)
        if art_doi == requested:
            best = (art, soup)
            break
        if best is None:
            best = (art, soup)

    if not best:
        return None, "Springer: no JATS <article> found"

    article, soup = best

    # Optional debug dump of the selected article
    if _SPRINGER_DEBUG_DIR:
        try:
            path = os.path.join(_SPRINGER_DEBUG_DIR, f"article_{re.sub('[^a-zA-Z0-9]+','_', doi)}.xml")
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(article))
        except Exception:
            pass

    # ---- Parse title
    title_tag = _find_ns(article, "article-title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # ---- Abstract (front matter)
    front = _find_ns(article, "front") or article
    abstract_text = extract_abstract_text(front)

    # ---- Body defines "full text"
    body = _find_ns(article, "body")
    if not body:
        sub = _find_ns(article, "sub-article")
        if sub:
            body = _find_ns(sub, "body")

    if not body:
        # IMPORTANT: do not synthesize body from the entire article;
        # caller will treat this as abstract-only if require_fulltext=True
        return (title, {}, {"abstract": abstract_text}), None

    # Prefer top-level sections; otherwise collapse whole body into a single section
    secs = _find_all_ns(body, "sec", recursive=False)
    if secs:
        sections: Dict[str, Any] = {}
        for sec in secs:
            try:
                sections.update(section_to_nested_dict(sec))
            except Exception:
                # Be resilient to odd section trees; skip on parse error
                continue
    else:
        sections = collapse_body_to_section(body)

    if not sections and not abstract_text:
        return None, "Springer: no sections/text"

    return (title, sections, {"abstract": abstract_text}), None
