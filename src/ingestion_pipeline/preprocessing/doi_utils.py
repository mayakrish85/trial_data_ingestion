import os, re
from typing import Optional
try:
    import bibtexparser
except ImportError:
    bibtexparser = None
import pandas as pd

DOI_PAT = re.compile(r"10\.\d{4,9}/\S+", re.I)

def normalize_doi(x: Optional[str]) -> Optional[str]:
    if not isinstance(x, str):
        return None
    x = x.strip()
    x = re.sub(r"^https?://(dx\.)?doi\.org/", "", x, flags=re.I)
    x = x.replace("\u200b", "").strip().lower()
    return x or None

def _strip_braces(s: str | None) -> str | None:
    if not isinstance(s, str): return None
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

def _doi_from_any(s: str | None) -> Optional[str]:
    if not isinstance(s, str): return None
    m = DOI_PAT.search(s)
    return m.group(0) if m else None

def read_bib_to_df(path: str) -> pd.DataFrame:
    if not bibtexparser:
        raise RuntimeError("bibtexparser is required to read .bib files. `pip install bibtexparser`.")
    with open(path, "r", encoding="utf-8") as f:
        db = bibtexparser.load(f)
    rows = []
    for entry in db.entries:
        fields = {k.lower(): v for k, v in entry.items()}
        doi = fields.get("doi") or _doi_from_any(fields.get("url", "")) or _doi_from_any(fields.get("howpublished", ""))
        title = _strip_braces(fields.get("title"))
        journal = _strip_braces(fields.get("journal") or fields.get("journaltitle") or fields.get("booktitle"))
        if doi:
            rows.append({"doi": doi, "journal": journal, "title": title})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No DOIs found in the .bib file.")
    df["doi_norm"] = df["doi"].apply(normalize_doi)
    df = df.dropna(subset=["doi_norm"]).drop_duplicates(subset=["doi_norm"]).reset_index(drop=True)
    return df[["doi", "journal", "title", "doi_norm"]]

def load_input_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if "doi" not in df.columns:
            raise ValueError("CSV must contain a 'doi' column.")
        if "journal" not in df.columns:
            df["journal"] = None
        df["doi_norm"] = df["doi"].apply(normalize_doi)
        df = df.dropna(subset=["doi_norm"]).drop_duplicates(subset=["doi_norm"]).reset_index(drop=True)
        return df
    elif ext in (".bib", ".bibtex"):
        return read_bib_to_df(path)
    else:
        raise ValueError(f"Unsupported input type: {ext}. Use .csv or .bib")
