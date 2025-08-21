from __future__ import annotations
import os, json, time
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
from ingestion_pipeline.preprocessing.doi_utils import normalize_doi, load_input_df
from ingestion_pipeline.sources.springer import try_springer_jats
from ingestion_pipeline.sources.pmc import doi_to_pmcid, try_pmc_jats

def canonicalize_record(*, doi: str, title: str, sections: Dict[str, Any],
                        source: str, pmcid: Optional[str], journal: Optional[str]) -> Dict[str, Any]:
    return {
        "doi": doi,
        "title": title,
        "journal": (journal if (isinstance(journal, str) and journal.strip()) else None),
        "source": source,
        "pmcid": (pmcid if pmcid else None),
        "sections": sections or {}
    }

def load_existing(out_path: str):
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return [], set()
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    doi_seen = set()
    for rec in data:
        doi_norm = normalize_doi(rec.get("doi"))
        if doi_norm:
            doi_seen.add(doi_norm)
    return data, doi_seen

def save_json(records: List[Dict[str, Any]], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def get_fulltext_canonical(doi_raw: str, journal: Optional[str], timeout: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    doi_norm = normalize_doi(doi_raw)
    if not doi_norm:
        return None, "Invalid DOI"
    springer_res, s_err = try_springer_jats(doi_norm, timeout=timeout)
    if springer_res:
        title, sections = springer_res
        return canonicalize_record(doi=doi_raw, title=title, sections=sections, source="springer_oa", pmcid=None, journal=journal), None
    pmcid, conv_err = doi_to_pmcid(doi_norm, timeout=timeout)
    if not pmcid:
        return None, f"No PMC: {conv_err} | Springer err: {s_err}"
    pmc_res, p_err = try_pmc_jats(pmcid, timeout=timeout)
    if pmc_res:
        title, sections = pmc_res
        return canonicalize_record(doi=doi_raw, title=title, sections=sections, source="pmc", pmcid=pmcid, journal=journal), None
    return None, f"PMC parse failed: {p_err} | Springer err: {s_err}"

def run_fulltext(input_path: str, output_dir: str, throttle_sec: float = 1.0, request_timeout: int = 45) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "fulltext_articles.json")
    skipped_csv = os.path.join(output_dir, "fulltext_skipped.csv")
    summary_json = os.path.join(output_dir, "fulltext_summary.json")
    df = load_input_df(input_path)
    if "journal" not in df.columns:
        df["journal"] = None
    df["doi_norm"] = df["doi"].apply(normalize_doi)
    df = df.dropna(subset=["doi_norm"]).drop_duplicates(subset=["doi_norm"]).reset_index(drop=True)
    records, doi_seen = load_existing(out_json)
    appended = 0
    skipped = 0
    failures: List[dict] = []
    for _, row in df.iterrows():
        doi_raw = row["doi"]
        journal = row.get("journal")
        doi_norm = row["doi_norm"]
        if doi_norm in doi_seen:
            skipped += 1
            continue
        rec, err = get_fulltext_canonical(doi_raw, journal, timeout=request_timeout)
        if rec:
            records.append(rec)
            doi_seen.add(doi_norm)
            appended += 1
        else:
            failures.append({"doi": doi_raw, "journal": journal, "reason": err})
        time.sleep(throttle_sec)
    save_json(records, out_json)
    if failures:
        pd.DataFrame(failures).to_csv(skipped_csv, index=False)
    summary = {
        "input_unique_doi": int(len(df)),
        "appended": int(appended),
        "skipped_existing": int(skipped),
        "failures": int(len(failures)),
        "out_json": out_json,
        "skipped_csv": skipped_csv if failures else None
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
