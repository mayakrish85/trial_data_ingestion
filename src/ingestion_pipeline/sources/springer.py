from __future__ import annotations
import os, requests
from typing import Tuple, Optional, Dict, Any
from bs4 import BeautifulSoup
from ingestion_pipeline.preprocessing.xml_cleaning import section_to_nested_dict, collapse_body_to_section

def try_springer_jats(doi: str, timeout: int = 45) -> Tuple[Optional[Tuple[str, Dict[str, Any]]], Optional[str]]:
    api_key = os.getenv("SPRINGER_API_KEY", "ee9adb4e1ec15a5d05475f45fc50e768")
    if not api_key:
        return None, "SPRINGER_API_KEY not set"
    
    url = f"https://api.springernature.com/openaccess/jats?q=doi:{doi}&api_key={api_key}"

    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "fulltext-pipeline/1.0"})
    except requests.RequestException as e:
        return None, f"Springer request error: {e}"
    
    if r.status_code != 200:
        return None, f"Springer HTTP {r.status_code}"
    soup = BeautifulSoup(r.content, "lxml-xml")
    body = soup.find("body")
    title_tag = soup.find("article-title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"
    
    if body is None:
        return None, "Springer: no <body>"
    secs = body.find_all("sec", recursive=False)
    sections = {}
    if secs:
        for sec in secs:
            sections.update(section_to_nested_dict(sec))
    else:
        sections = collapse_body_to_section(body)
    if not sections:
        return None, "Springer: no sections/text"
    return (title, sections), None
