from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Iterable, List
import os, re, time, requests
from bs4 import BeautifulSoup

from ingestion_pipeline.preprocessing.xml_cleaning import (
    section_to_nested_dict,
    linearize_body_to_fulltext,
    extract_abstract_text,
)

# ---------------- DOI → PMCID (single) ----------------

def doi_to_pmcid(
    doi: str,
    timeout: int = 45,
    max_retries: int = 3,
    backoff: float = 1.5,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a DOI to PMCID via NCBI idconv; fallback to Europe PMC."""
    idconv_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(
                idconv_url,
                params={"ids": doi, "format": "json"},
                timeout=timeout,
                headers={"User-Agent": "fulltext-pipeline/1.1"},
            )
            if r.status_code == 200:
                data = r.json()
                for rec in data.get("records", []):
                    if rec.get("pmcid"):
                        return rec["pmcid"], None
                last_err = "idconv: no PMCID"
            else:
                last_err = f"idconv HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = f"idconv error: {e}"
        time.sleep(backoff ** (attempt + 1))
    # Europe PMC fallback
    try:
        epmc = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        r = requests.get(
            epmc,
            params={"query": f"doi:{doi}", "format": "json"},
            timeout=timeout,
            headers={"User-Agent": "fulltext-pipeline/1.1"},
        )
        if r.status_code == 200:
            for h in (r.json().get("resultList") or {}).get("result") or []:
                if h.get("pmcid"):
                    return h["pmcid"], None
            return None, f"EuropePMC: no PMCID | {last_err}"
        else:
            return None, f"EuropePMC HTTP {r.status_code} | {last_err}"
    except requests.RequestException as e:
        return None, f"EuropePMC error: {e} | {last_err}"

# ---------------- DOI → PMCID (batched) ----------------

def doi_to_pmcid_fetch_batch(
    dois: Iterable[str],
    timeout: int = 45,
    session: Optional[requests.Session] = None,
    backoff: float = 1.5,
    max_retries: int = 3,
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Resolve a *single* batch of DOIs to PMCIDs via idconv (one HTTP call).
    Returns (mapping_doi_norm->pmcid, failures[(doi_norm, reason)]).
    """
    sess = session or requests.Session()
    base = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    chunk = [(d or "").strip().lower() for d in dois if d]
    out: Dict[str, str] = {}
    fails: List[Tuple[str, str]] = []
    last_err = None
    for attempt in range(max_retries):
        try:
            r = sess.get(
                base,
                params={"ids": ",".join(chunk), "format": "json"},
                timeout=timeout,
                headers={"User-Agent": "fulltext-pipeline/1.2"},
            )
            if r.status_code == 200:
                recs = (r.json().get("records") or [])
                resolved = set()
                for rec in recs:
                    doi = (rec.get("doi") or "").strip().lower()
                    pmcid = rec.get("pmcid")
                    if doi and pmcid:
                        out[doi] = pmcid
                        resolved.add(doi)
                for d in chunk:
                    if d not in resolved:
                        fails.append((d, "idconv: no PMCID"))
                return out, fails
            last_err = f"idconv HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = f"idconv error: {e}"
        time.sleep(backoff ** (attempt + 1))
    for d in chunk:
        fails.append((d, last_err or "idconv: unknown error"))
    return out, fails

# ---------------- PMC JATS parse helpers ----------------

def _article_pmcid(article_tag) -> Optional[str]:
    for aid in article_tag.find_all("article-id"):
        if (aid.get("pub-id-type") or "").lower() == "pmcid":
            txt = aid.get_text(strip=True)
            if txt:
                return re.sub(r"^PMC", "PMC", txt, flags=re.I)
    return None

def _parse_article(article_tag):
    """Parse a single <article> tag to (title, sections, {'abstract': text})."""
    # title
    title = "Untitled"
    tg = article_tag.find("title-group")
    if tg and tg.find("article-title"):
        title = tg.find("article-title").get_text(strip=True)

    # abstract from front matter
    front = article_tag.find("front") or article_tag
    abstract_text = extract_abstract_text(front)

    # body defines full text; do NOT synthesize from whole article if missing
    body = article_tag.find("body")
    if not body:
        sub = article_tag.find("sub-article")
        if sub and sub.find("body"):
            body = sub.find("body")

    if not body:
        sections: Dict[str, Any] = {}
        return (title, sections, {"abstract": abstract_text}), None

    secs = body.find_all("sec", recursive=False)
    if secs:
        sections = {}
        for s in secs:
            sections.update(section_to_nested_dict(s))
    else:
        sections = linearize_body_to_fulltext(body)  # body only

    if not sections and not abstract_text:
        return None, "No sections/text"
    return (title, sections, {"abstract": abstract_text}), None

# ---------------- PMC JATS (single) ----------------

def try_pmc_jats(
    pmcid: str,
    timeout: int = 45,
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]], Optional[str]]:
    """Fetch one PMC article as JATS, returning (title, sections, {'abstract': ...})."""
    pmc_num = re.sub(r"\D", "", pmcid or "")
    if not pmc_num:
        return None, "Invalid PMCID"
    headers = {"User-Agent": "fulltext-pipeline/1.2"}
    sess = session or requests.Session()

    # EFetch
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pmc", "id": pmc_num, "rettype": "xml"}
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    try:
        r = sess.get(efetch_url, params=params, timeout=timeout, headers=headers)
        if r.status_code == 200 and r.content.strip():
            soup = BeautifulSoup(r.content, "lxml-xml")
            art = soup.find("article") or (soup.find("pmc-articleset") and soup.find("pmc-articleset").find("article"))
            if art:
                return _parse_article(art)
    except requests.RequestException:
        pass

    # OAI-PMH fallback
    try:
        oai = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
        r = sess.get(oai, params={"verb": "GetRecord", "identifier": f"oai:pubmedcentral.nih.gov:{pmc_num}", "metadataPrefix": "pmc"},
                     timeout=timeout, headers=headers)
        if r.status_code == 200 and r.content.strip():
            soup = BeautifulSoup(r.content, "lxml-xml")
            art = soup.find("article") or (soup.find("pmc-articleset") and soup.find("pmc-articleset").find("article"))
            if art:
                return _parse_article(art)
    except requests.RequestException:
        pass

    # Europe PMC fullTextXML fallback
    try:
        epmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMC{pmc_num}/fullTextXML"
        r = sess.get(epmc_url, timeout=timeout, headers=headers)
        if r.status_code == 200 and r.content.strip():
            soup = BeautifulSoup(r.content, "lxml-xml")
            art = soup.find("article") or (soup.find("pmc-articleset") and soup.find("pmc-articleset").find("article"))
            if art:
                return _parse_article(art)
    except requests.RequestException:
        pass

    return None, "PMC/EPMC: no JATS <article> found"

# ---------------- PMC JATS (batched) ----------------

def try_pmc_jats_fetch_batch(
    pmcids: Iterable[str],
    timeout: int = 45,
    session: Optional[requests.Session] = None,
) -> Tuple[Dict[str, Tuple[str, Dict[str, Any], Dict[str, Any]]], List[Tuple[str, str]]]:
    """
    Fetch a *single batch* of PMC articles via EFetch (one HTTP call).
    Returns (mapping 'PMCxxxx' -> (title, sections, {'abstract': ...}), failures[(pmcid, reason)]).
    """
    sess = session or requests.Session()
    ids = [re.sub(r"\D", "", p or "") for p in pmcids if p]
    out: Dict[str, Tuple[str, Dict[str, Any], Dict[str, Any]]] = {}
    fails: List[Tuple[str, str]] = []
    if not ids:
        return out, fails

    efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pmc", "id": ",".join(ids), "rettype": "xml"}
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key

    r = sess.get(efetch, params=params, timeout=timeout, headers={"User-Agent": "fulltext-pipeline/1.2"})
    if r.status_code != 200 or not r.content.strip():
        for pid in ids:
            fails.append((f"PMC{pid}", f"EFetch HTTP {r.status_code}"))
        return out, fails

    soup = BeautifulSoup(r.content, "lxml-xml")
    arts = soup.find_all("article")
    seen = set()
    for art in arts:
        pmcid_tag = _article_pmcid(art)  # e.g. "PMC1234567"
        if not pmcid_tag:
            continue
        parsed, perr = _parse_article(art)
        if parsed:
            out[pmcid_tag] = parsed
            seen.add(re.sub(r"\D", "", pmcid_tag))

    for pid in ids:
        if pid not in seen:
            fails.append((f"PMC{pid}", "EFetch: article not found in response"))
    return out, fails