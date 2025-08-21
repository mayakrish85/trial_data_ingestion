from typing import List, Dict, Any
import bibtexparser

def parse_bib_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        db = bibtexparser.load(f)
    records = []
    for entry in db.entries:
        rec = {
            "id": entry.get("ID") or entry.get("id"),
            "title": entry.get("title"),
            "doi": entry.get("doi"),
            "url": entry.get("url") or entry.get("link"),
            "source": "bib",
            "full_text": None,
            "meta": entry,
        }
        records.append(rec)
    return records
