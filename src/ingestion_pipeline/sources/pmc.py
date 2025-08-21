from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import re, requests
from bs4 import BeautifulSoup
from ingestion_pipeline.preprocessing.xml_cleaning import (
    section_to_nested_dict, collapse_body_to_section, linearize_body_to_fulltext
)

def doi_to_pmcid(doi: str, timeout: int = 45, max_retries: int = 3, backoff: float = 1.5) -> Tuple[Optional[str], Optional[str]]:
    idconv_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(idconv_url, params={"ids": doi, "format": "json"},
                             timeout=timeout, headers={"User-Agent": "fulltext-pipeline/1.1"})
            if r.status_code == 200:
                data = r.json()
                recs = data.get("records", [])
                for rec in recs:
                    pmcid = rec.get("pmcid")
                    if pmcid:
                        return pmcid, None
                last_err = "idconv: no PMCID"
            else:
                last_err = f"idconv HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = f"idconv error: {e}"
        import time
        time.sleep(backoff ** (attempt + 1))

    try:
        epmc = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        r = requests.get(epmc, params={"query": f"doi:{doi}", "format": "json"}, timeout=timeout,
                         headers={"User-Agent": "fulltext-pipeline/1.1"})
        if r.status_code == 200:
            data = r.json()
            hits = (data.get("resultList") or {}).get("result") or []
            for h in hits:
                pmcid = h.get("pmcid")
                if pmcid:
                    return pmcid, None
            return None, f"EuropePMC: no PMCID | {last_err}"
        else:
            return None, f"EuropePMC HTTP {r.status_code} | {last_err}"
    except requests.RequestException as e:
        return None, f"EuropePMC error: {e} | {last_err}"

def _find_article_node(xml_soup):
    art = xml_soup.find("article")
    if art: return art
    pas = xml_soup.find("pmc-articleset")
    return pas.find("article") if pas else None

def _parse_pmc_article_node(article_tag):
    title = "Untitled"
    tg = article_tag.find("title-group")
    if tg and tg.find("article-title"):
        title = tg.find("article-title").get_text(strip=True)
    else:
        at = article_tag.find("article-title")
        if at:
            title = at.get_text(strip=True)

    body = article_tag.find("body")
    if not body:
        sub = article_tag.find("sub-article")
        if sub:
            body = sub.find("body")

    if body:
        secs = body.find_all("sec", recursive=False)
        if secs:
            sections = {}
            for s in secs:
                sections.update(section_to_nested_dict(s))
        else:
            sections = linearize_body_to_fulltext(body)
    else:
        art_copy = BeautifulSoup(str(article_tag), "lxml-xml")
        for tagname in ["front", "back", "ref-list", "permissions", "copyright-statement"]:
            t = art_copy.find(tagname)
            if t: t.decompose()
        sections = linearize_body_to_fulltext(art_copy)

    if not sections:
        return None, "PMC XML has no sections or usable text"
    return (title, sections), None

def try_pmc_jats(pmcid: str, timeout: int = 45):
    pmc_num = re.sub(r"\D", "", pmcid or "")
    if not pmc_num:
        return None, "Invalid PMCID"

    headers = {"User-Agent": "fulltext-pipeline/1.2"}

    try:
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        r = requests.get(efetch_url, params={"db": "pmc", "id": pmc_num, "rettype": "xml"},
                         timeout=timeout, headers=headers)
        if r.status_code == 200 and r.content.strip():
            soup = BeautifulSoup(r.content, "lxml-xml")
            article = _find_article_node(soup)
            if article:
                return _parse_pmc_article_node(article)
    except requests.RequestException:
        pass

    try:
        oai = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
        r = requests.get(oai, params={"verb": "GetRecord", "identifier": f"oai:pubmedcentral.nih.gov:{pmc_num}", "metadataPrefix": "pmc"},
                         timeout=timeout, headers=headers)
        if r.status_code == 200 and r.content.strip():
            soup = BeautifulSoup(r.content, "lxml-xml")
            article = _find_article_node(soup)
            if article:
                return _parse_pmc_article_node(article)
    except requests.RequestException:
        pass

    try:
        epmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/PMC{pmc_num}/fullTextXML"
        r = requests.get(epmc_url, timeout=timeout, headers=headers)
        if r.status_code == 200 and r.content.strip():
            soup = BeautifulSoup(r.content, "lxml-xml")
            article = _find_article_node(soup)
            if article:
                return _parse_pmc_article_node(article)
    except requests.RequestException:
        pass

    return None, "PMC/EPMC: no JATS <article> found"
