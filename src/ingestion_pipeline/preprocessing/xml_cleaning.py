from __future__ import annotations
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup

# Tags we don't want in body text
DROP_TAGS = {
    "fig", "fig-group", "table", "table-wrap", "graphic", "media", "alternatives",
    "inline-formula", "disp-formula", "tex-math", "ref-list", "license", "permissions",
    "copyright-statement", "supplementary-material", "fn", "fn-group"
}

def section_to_nested_dict(sec_tag) -> Dict[str, Any]:
    """Convert a JATS <sec> subtree to a nested dict structure."""
    title_tag = sec_tag.find("title")
    section_title = title_tag.get_text(strip=True).title() if title_tag else "Untitled Section"
    # strip noisy tags inside this section
    for t in sec_tag.find_all(list(DROP_TAGS)):
        t.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in sec_tag.find_all("p", recursive=False)]
    text = " ".join(paragraphs).strip()
    section_dict: Dict[str, Any] = {}
    if text:
        section_dict["text"] = text
    for child in sec_tag.find_all("sec", recursive=False):
        child_dict = section_to_nested_dict(child)
        for k, v in child_dict.items():
            section_dict[k] = v
    return {section_title: section_dict}

def collapse_body_to_section(body) -> Dict[str, Any]:
    """If there are no explicit sections, collapse body to one full-text block."""
    full_text = body.get_text(" ", strip=True) if body else ""
    return {"Full Text": {"text": full_text}} if full_text else {}

def linearize_body_to_fulltext(body) -> Dict[str, Any]:
    """Linearize paragraphs, lists, and quotes inside <body> to a single Full Text block."""
    body_copy = BeautifulSoup(str(body), "lxml-xml")
    for t in body_copy.find_all(list(DROP_TAGS)):
        t.decompose()
    chunks: List[str] = []
    for p in body_copy.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if txt:
            chunks.append(txt)
    for lst in body_copy.find_all("list"):
        items = [li.get_text(" ", strip=True) for li in lst.find_all("list-item", recursive=False)]
        items = [it for it in items if it]
        if items:
            chunks.append("\n".join(f"• {it}" for it in items))
    for dq in body_copy.find_all(["disp-quote", "boxed-text"]):
        txt = dq.get_text(" ", strip=True)
        if txt:
            chunks.append(txt)
    full_text = "\n\n".join([c for c in chunks if c])
    return {"Full Text": {"text": full_text}} if full_text else {}

def sections_to_text(sections: Dict[str, Any]) -> str:
    """Flatten nested sections to a single string of body text."""
    out: List[str] = []
    def dfs(node: Dict[str, Any]):
        txt = node.get("text")
        if isinstance(txt, str) and txt.strip():
            out.append(txt.strip())
        for k, v in node.items():
            if isinstance(v, dict) and k != "text":
                dfs(v)
    for _, block in sections.items():
        if isinstance(block, dict):
            dfs(block)
    return "\n\n".join(out).strip()

def extract_abstract_text(root) -> Optional[str]:
    """
    Extract abstract text from a JATS document:
    supports <abstract>, <trans-abstract>, and structured abstracts with <sec>.
    """
    if root is None:
        return None
    soup = BeautifulSoup(str(root), "lxml-xml") if not isinstance(root, BeautifulSoup) else root

    abstracts = soup.find_all(["abstract", "trans-abstract"])
    parts: List[str] = []
    for ab in abstracts:
        secs = ab.find_all("sec", recursive=False)
        if secs:
            for sec in secs:
                title = sec.find("title")
                if title and title.get_text(strip=True):
                    parts.append(title.get_text(strip=True))
                for p in sec.find_all("p", recursive=False):
                    txt = p.get_text(" ", strip=True)
                    if txt:
                        parts.append(txt)
        else:
            ps = ab.find_all("p", recursive=False)
            if not ps:
                txt = ab.get_text(" ", strip=True)
                if txt:
                    parts.append(txt)
            else:
                for p in ps:
                    txt = p.get_text(" ", strip=True)
                    if txt:
                        parts.append(txt)
    text = "\n\n".join([p for p in parts if p]).strip()
    return text or None