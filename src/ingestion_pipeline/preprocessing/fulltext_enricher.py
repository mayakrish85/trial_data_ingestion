# src/ingestion_pipeline/preprocessing/fulltext_enricher.py
from __future__ import annotations

import os
import json
import time
from typing import Tuple, Optional, Dict, Any, List

import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ingestion_pipeline.preprocessing.doi_utils import normalize_doi, load_input_df
from ingestion_pipeline.preprocessing.xml_cleaning import sections_to_text

# --- Try batched PMC helpers; fall back to single-item versions if missing ---
HAVE_BATCH = True
try:
    from ingestion_pipeline.sources.pmc import (
        doi_to_pmcid_fetch_batch,    # (mapping, failures)
        try_pmc_jats_fetch_batch,    # (mapping, failures)
        try_pmc_jats,                # single fallback
        doi_to_pmcid,                # single fallback
    )
except Exception:
    HAVE_BATCH = False
    from ingestion_pipeline.sources.pmc import try_pmc_jats, doi_to_pmcid  # type: ignore

    def doi_to_pmcid_fetch_batch(dois, timeout=45, session=None, **_):
        out, fails = {}, []
        for d in dois:
            pmcid, err = doi_to_pmcid(d, timeout=timeout)  # type: ignore
            if pmcid:
                out[d] = pmcid
            else:
                fails.append((d, err or "idconv failed"))
        return out, fails

    def try_pmc_jats_fetch_batch(pmcids, timeout=45, session=None, **_):
        out, fails = {}, []
        for p in pmcids:
            res, err = try_pmc_jats(p, timeout=timeout)  # type: ignore
            if res:
                out[p] = res
            else:
                fails.append((p, err or "EFetch failed"))
        return out, fails


# ---------------- helpers ----------------
def _unpack_result(res) -> Tuple[Optional[str], dict, Optional[str]]:
    """
    Accept (title, sections) or (title, sections, {'abstract': ...}).
    Returns (title, sections, abstract_text).
    """
    abstract = None
    if isinstance(res, tuple) and len(res) == 3:
        title, sections, extras = res
        if isinstance(extras, dict):
            abstract = extras.get("abstract")
        return title, sections or {}, abstract
    if isinstance(res, tuple) and len(res) == 2:
        title, sections = res
        return title, sections or {}, None
    return None, {}, None

def _body_len(sections: dict) -> int:
    return len((sections_to_text(sections or {}) or "").strip())

def canonicalize_record(
    *, doi: str, title: str, sections: Dict[str, Any], source: str, pmcid: Optional[str], journal: Optional[str]
) -> Dict[str, Any]:
    return {
        "doi": doi,
        "title": title,
        "journal": (journal if (isinstance(journal, str) and journal.strip()) else None),
        "source": source,
        "pmcid": (pmcid if pmcid else None),
        "sections": sections or {},
    }

def load_existing(out_path: str):
    """Return (records_list, seen_doi_norm_set) from an existing JSON file, if present."""
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return [], set()
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    seen = set()
    for rec in data:
        dn = normalize_doi(rec.get("doi"))
        if dn:
            seen.add(dn)
    return data, seen

def save_json(obj: Any, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True
               ) if os.path.dirname(out_path) else None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------- main (PMC-first only; no Springer) ----------------
def run_fulltext(
    input_path: str,
    output_dir: str,
    throttle_sec: float = 0.0,          # per-ARTICLE delay (keep 0 for speed)
    request_timeout: int = 45,
    show_progress: bool = True,
    idconv_chunk: int = 150,            # tuned for speed
    efetch_chunk: int = 80,
    batch_workers: int = 4,             # parallel batch requests
    batch_throttle_sec: float = 0.10,   # polite delay after each batch
    require_fulltext: bool = True,      # keep only full-text records
    min_fulltext_chars: int = 200,      # body length threshold
    skip_pmc_single_fallback: bool = True,  # OFF single fallbacks for speed
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "fulltext_articles_AUG25.json")
    skipped_csv = os.path.join(output_dir, "fulltext_skipped.csv")
    summary_json = os.path.join(output_dir, "fulltext_summary.json")

    # Load & dedupe input
    df = load_input_df(input_path)
    if "journal" not in df.columns:
        df["journal"] = None
    df["doi_norm"] = df["doi"].apply(normalize_doi)
    df = df.dropna(subset=["doi_norm"]).drop_duplicates(subset=["doi_norm"]).reset_index(drop=True)

    # Resume support
    records, doi_seen = load_existing(out_json)
    failures: List[dict] = []
    appended = 0

    # Worklist: exclude already present
    todo_rows: List[pd.Series] = [row for _, row in df.iterrows() if row["doi_norm"] not in doi_seen]
    skipped_existing = len(df) - len(todo_rows)

    # ---------- Stage A: DOI -> PMCID (batched, parallel) ----------
    import requests
    session = requests.Session()  # reuse connections to NCBI
    dois_norm = [normalize_doi(r["doi"]) for r in todo_rows]
    doi_batches = [dois_norm[i:i + idconv_chunk] for i in range(0, len(dois_norm), idconv_chunk)]

    doi_to_pmc: Dict[str, str] = {}
    conv_fails_all: List[Tuple[str, str]] = []

    def idconv_worker(batch: List[str]):
        m, fails = doi_to_pmcid_fetch_batch(batch, timeout=request_timeout, session=session)
        if batch_throttle_sec:
            time.sleep(batch_throttle_sec)
        return m, fails

    if doi_batches:
        p = tqdm(total=len(doi_batches), disable=not show_progress, desc="IDConv (DOI→PMCID)", unit="batch")
        with ThreadPoolExecutor(max_workers=max(1, int(batch_workers))) as pool:
            futures = [pool.submit(idconv_worker, b) for b in doi_batches]
            for fut in as_completed(futures):
                m, fails = fut.result()
                doi_to_pmc.update(m)
                conv_fails_all.extend(fails)
                p.update(1)
        p.close()

    # ---------- Stage B: PMC EFetch (batched, parallel) ----------
    pmcids = [doi_to_pmc[d] for d in dois_norm if d in doi_to_pmc]
    pmc_batches = [pmcids[i:i + efetch_chunk] for i in range(0, len(pmcids), efetch_chunk)]
    pmc_map: Dict[str, Any] = {}
    pmc_fails_all: List[Tuple[str, str]] = []

    def efetch_worker(batch: List[str]):
        m, fails = try_pmc_jats_fetch_batch(batch, timeout=request_timeout, session=session)
        if batch_throttle_sec:
            time.sleep(batch_throttle_sec)
        return m, fails

    if pmc_batches:
        p = tqdm(total=len(pmc_batches), disable=not show_progress, desc="EFetch (PMC JATS)", unit="batch")
        with ThreadPoolExecutor(max_workers=max(1, int(batch_workers))) as pool:
            futures = [pool.submit(efetch_worker, b) for b in pmc_batches]
            for fut in as_completed(futures):
                m, fails = fut.result()
                pmc_map.update(m)
                pmc_fails_all.extend(fails)
                p.update(1)
        p.close()

    # ---------- Stage C: Assemble (simple & fast; no Springer) ----------
    p = tqdm(total=len(todo_rows), disable=not show_progress, desc="Assemble", unit="article")
    for row in todo_rows:
        doi_raw = row["doi"]
        journal = row.get("journal")
        dn = normalize_doi(doi_raw)

        pmcid = doi_to_pmc.get(dn)
        if not pmcid:
            reason = next((r for d, r in conv_fails_all if d == dn), None) or "No PMCID"
            failures.append({"doi": doi_raw, "journal": journal, "reason": reason})
            p.update(1)
            continue

        parsed = pmc_map.get(pmcid)
        if not parsed and not skip_pmc_single_fallback:
            # Try a single-item fallback if you really want to squeeze a few more
            single_res, _ = try_pmc_jats(pmcid, timeout=request_timeout)
            parsed = single_res or None

        if parsed:
            title, sections, _ = _unpack_result(parsed)
            if require_fulltext and _body_len(sections) < max(0, int(min_fulltext_chars)):
                failures.append({"doi": doi_raw, "journal": journal, "reason": "abstract_only"})
            else:
                rec = canonicalize_record(
                    doi=doi_raw, title=title, sections=sections, source="pmc", pmcid=pmcid, journal=journal
                )
                records.append(rec)
                doi_seen.add(dn)
                appended += 1
        else:
            reason = "PMC fetch failed (batched only)" if skip_pmc_single_fallback else "PMC fetch failed (batched + single)"
            failures.append({"doi": doi_raw, "journal": journal, "reason": reason})

        if throttle_sec:
            time.sleep(throttle_sec)
        p.update(1)
    p.close()

    # ---------- Persist ----------
    save_json(records, out_json)
    if failures:
        pd.DataFrame(failures).to_csv(skipped_csv, index=False)

    summary = {
        "input_unique_doi": int(len(df)),
        "appended": int(appended),
        "skipped_existing": int(skipped_existing),
        "failures": int(len(failures)),
        "out_json": out_json,
        "skipped_csv": skipped_csv if failures else None,
        "batched_helpers": bool(HAVE_BATCH),
        # no Springer fields in this reverted version
        "min_fulltext_chars": int(min_fulltext_chars),
        "skip_pmc_single_fallback": bool(skip_pmc_single_fallback),
        "idconv_chunk": int(idconv_chunk),
        "efetch_chunk": int(efetch_chunk),
        "batch_workers": int(batch_workers),
    }
    save_json(summary, summary_json)
    return summary
