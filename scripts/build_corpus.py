"""
scripts/build_corpus.py
-----------------------
Build the chest-imaging RAG corpus from two open sources:

  1. StatPearls (public domain, via NCBI E-utilities + Bookshelf)
  2. Wikipedia medical articles (CC-BY-SA, via MediaWiki API)

Output schema matches what build_rag_index.py expects:
  data/rag/raw/<slug>.json with {title, source_url, license, source,
                                 sections, finding_keys, slug}

No scraping. Both APIs are official, rate-limited politely, and return
clean structured content.

Usage:
    python -m scripts.build_corpus --output-dir data/rag/raw
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "RadAgent-Research/0.1 (academic research; "
    "rad-agent-research@example.org)"
)
NCBI_RATE_LIMIT_SEC = 0.4   # NCBI allows 3 req/s without API key
WIKI_RATE_LIMIT_SEC = 0.5

# StatPearls + Wikipedia query targets, keyed by NIH-14 class.
# Each entry: (slug, search_term, finding_keys, wiki_title)
# - search_term is what we send to NCBI to find the StatPearls article
# - wiki_title is the canonical Wikipedia article title
TARGETS: list[dict] = [
    {"slug": "cardiomegaly", "search": "Cardiomegaly",
     "wiki": "Cardiomegaly", "keys": ["Cardiomegaly"]},
    {"slug": "pleural-effusion", "search": "Pleural Effusion",
     "wiki": "Pleural effusion", "keys": ["Effusion"]},
    {"slug": "pneumothorax", "search": "Pneumothorax",
     "wiki": "Pneumothorax", "keys": ["Pneumothorax"]},
    {"slug": "pulmonary-edema", "search": "Pulmonary Edema",
     "wiki": "Pulmonary edema", "keys": ["Edema"]},
    {"slug": "emphysema", "search": "Emphysema",
     "wiki": "Emphysema", "keys": ["Emphysema"]},
    {"slug": "pulmonary-fibrosis", "search": "Pulmonary Fibrosis",
     "wiki": "Pulmonary fibrosis", "keys": ["Fibrosis"]},
    {"slug": "consolidation", "search": "Pulmonary Consolidation",
     "wiki": "Pulmonary consolidation", "keys": ["Consolidation"]},
    {"slug": "pneumonia", "search": "Pneumonia",
     "wiki": "Pneumonia", "keys": ["Pneumonia"]},
    {"slug": "atelectasis", "search": "Atelectasis",
     "wiki": "Atelectasis", "keys": ["Atelectasis"]},
    {"slug": "lung-nodule", "search": "Pulmonary Nodule",
     "wiki": "Pulmonary nodule", "keys": ["Nodule"]},
    {"slug": "lung-mass", "search": "Lung Cancer",
     "wiki": "Lung cancer", "keys": ["Mass", "Nodule"]},
    {"slug": "pleural-thickening", "search": "Pleural Plaque",
     "wiki": "Pleural disease", "keys": ["Pleural_Thickening"]},
    {"slug": "hiatal-hernia", "search": "Hiatal Hernia",
     "wiki": "Hiatal hernia", "keys": ["Hernia"]},
    {"slug": "infiltrate", "search": "Pulmonary Infiltrate",
     "wiki": "Infiltration (medical)", "keys": ["Infiltration"]},
    # Cross-cutting reference articles
    {"slug": "chest-radiograph", "search": "Chest Radiograph Interpretation",
     "wiki": "Chest radiograph", "keys": []},
    {"slug": "lobar-pneumonia", "search": "Lobar Pneumonia",
     "wiki": "Lobar pneumonia", "keys": ["Pneumonia", "Consolidation"]},
    {"slug": "aspiration-pneumonia", "search": "Aspiration Pneumonia",
     "wiki": "Aspiration pneumonia", "keys": ["Pneumonia", "Consolidation"]},
    {"slug": "viral-pneumonia", "search": "Viral Pneumonia",
     "wiki": "Viral pneumonia", "keys": ["Pneumonia"]},
    {"slug": "tension-pneumothorax", "search": "Tension Pneumothorax",
     "wiki": "Pneumothorax", "keys": ["Pneumothorax"]},
    {"slug": "empyema", "search": "Empyema",
     "wiki": "Empyema", "keys": ["Effusion"]},
    {"slug": "hemothorax", "search": "Hemothorax",
     "wiki": "Hemothorax", "keys": ["Effusion"]},
    {"slug": "copd", "search": "Chronic Obstructive Pulmonary Disease",
     "wiki": "Chronic obstructive pulmonary disease",
     "keys": ["Emphysema", "Infiltration"]},
    {"slug": "ipf", "search": "Idiopathic Pulmonary Fibrosis",
     "wiki": "Idiopathic pulmonary fibrosis", "keys": ["Fibrosis"]},
    {"slug": "ild", "search": "Interstitial Lung Disease",
     "wiki": "Interstitial lung disease",
     "keys": ["Fibrosis", "Infiltration"]},
    {"slug": "lung-cancer-staging", "search": "Lung Cancer Staging",
     "wiki": "Lung cancer staging", "keys": ["Mass", "Nodule"]},
    {"slug": "ards", "search": "Acute Respiratory Distress Syndrome",
     "wiki": "Acute respiratory distress syndrome",
     "keys": ["Edema", "Infiltration", "Consolidation"]},
    {"slug": "heart-failure", "search": "Congestive Heart Failure",
     "wiki": "Heart failure",
     "keys": ["Cardiomegaly", "Edema", "Effusion"]},
    {"slug": "asbestos-related-lung", "search": "Asbestosis",
     "wiki": "Asbestosis",
     "keys": ["Fibrosis", "Pleural_Thickening"]},
]


# --------------------------- HTTP helpers ---------------------------

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/html, */*",
    })
    return s


def _polite_get(session, url, rate_limit, params=None, max_retries=3):
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  [http] {url}: {e}", flush=True)
            time.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code == 200:
            time.sleep(rate_limit)
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = float(resp.headers.get("Retry-After", backoff))
            print(f"  [http] {resp.status_code}, backing off {wait:.1f}s",
                  flush=True)
            time.sleep(wait)
            backoff *= 2
            continue
        print(f"  [http] {resp.status_code} {url}", flush=True)
        return None
    return None


# --------------------------- StatPearls via NCBI ---------------------------

def fetch_statpearls(session, search_term: str) -> Optional[dict]:
    """Find a StatPearls article matching search_term and fetch sections.

    Strategy:
      1. esearch on the 'books' database with [book] StatPearls filter.
      2. Take the top hit's NBK ID.
      3. Fetch the rendered HTML from Bookshelf.
      4. Parse <h2>/<h3> sections like the Radiopaedia parser did.
    """
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "books",
        "term": f'"{search_term}"[Title] AND statpearls[Book]',
        "retmode": "json",
        "retmax": 5,
    }
    resp = _polite_get(session, esearch_url, NCBI_RATE_LIMIT_SEC, params=params)
    if resp is None:
        return None
    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"  [ncbi] non-JSON response for {search_term}", flush=True)
        return None
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        print(f"  [ncbi] no StatPearls hit for '{search_term}'", flush=True)
        return None
    nbk_id = ids[0]

    # Fetch the rendered HTML chapter.
    book_url = f"https://www.ncbi.nlm.nih.gov/books/NBK{nbk_id}/"
    resp = _polite_get(session, book_url, NCBI_RATE_LIMIT_SEC)
    if resp is None:
        # Try the un-prefixed id (some return the NBK already)
        if not nbk_id.startswith("NBK"):
            book_url = f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/"
            resp = _polite_get(session, book_url, NCBI_RATE_LIMIT_SEC)
        if resp is None:
            return None
        url_used = book_url
    else:
        url_used = book_url

    sections = _parse_bookshelf_html(resp.text)
    if not sections:
        return None

    # Pull title from the page if we can
    soup = BeautifulSoup(resp.text, "lxml")
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else search_term

    return {
        "title": title,
        "source_url": url_used,
        "license": "Public Domain (StatPearls / NCBI Bookshelf)",
        "source": "statpearls",
        "sections": sections,
    }


def _parse_bookshelf_html(html: str) -> list[dict]:
    """Bookshelf pages have <h2>/<h3> section headers and <p> body text.

    We strip references, figure captions, and citation widgets.
    """
    soup = BeautifulSoup(html, "lxml")
    body = (
        soup.select_one("div.book-content")
        or soup.select_one("div#maincontent")
        or soup.select_one("article")
        or soup.find("body")
    )
    if body is None:
        return []
    # Strip noise
    for sel in ["script", "style", "nav", "noscript",
                "div.fig", "div.figure", "figcaption",
                "div.bk_prnt", "ul.inline_list", "div.refdesc",
                "div.bk_align_grid", "section.ref-list", "div.ref-list",
                ".permissions", ".citation-form"]:
        for el in body.select(sel):
            el.decompose()

    sections: list[dict] = []
    current = "Overview"
    buf: list[str] = []
    for el in body.find_all(["h2", "h3", "p", "li"], recursive=True):
        if el.name in ("h2", "h3"):
            if buf:
                txt = "\n".join(buf).strip()
                if txt:
                    sections.append({"section": current, "text": txt})
                buf = []
            current = el.get_text(strip=True) or current
        else:
            t = el.get_text(" ", strip=True)
            # Skip obvious noise
            if not t or len(t) < 10:
                continue
            if t.lower().startswith(("figure ", "fig.", "table ", "see fig")):
                continue
            buf.append(t)
    if buf:
        txt = "\n".join(buf).strip()
        if txt:
            sections.append({"section": current, "text": txt})
    # Drop sections that are obviously references-only or boilerplate
    sections = [s for s in sections
                if not _is_ref_section(s["section"])
                and len(s["text"]) >= 100]
    return sections


_REF_SECTION_RE = re.compile(
    r"^(references?|further reading|bibliography|disclosure|"
    r"author information|review questions|continuing education)$",
    re.IGNORECASE,
)


def _is_ref_section(name: str) -> bool:
    return bool(_REF_SECTION_RE.match(name.strip()))


# --------------------------- Wikipedia ---------------------------

def fetch_wikipedia(session, page_title: str) -> Optional[dict]:
    """Fetch the plain-text content of a Wikipedia article via the API."""
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "inprop": "url",
        "redirects": 1,
        "titles": page_title,
    }
    resp = _polite_get(session, api, WIKI_RATE_LIMIT_SEC, params=params)
    if resp is None:
        return None
    try:
        data = resp.json()
    except json.JSONDecodeError:
        return None
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    if "missing" in page:
        print(f"  [wiki] missing: {page_title}", flush=True)
        return None
    extract = page.get("extract", "")
    if not extract or len(extract) < 500:
        return None
    title = page.get("title", page_title)
    url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}")

    sections = _parse_wiki_extract(extract)
    if not sections:
        return None

    return {
        "title": title,
        "source_url": url,
        "license": "CC-BY-SA-4.0",
        "source": "wikipedia",
        "sections": sections,
    }


_WIKI_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
_WIKI_SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading",
    "notes", "bibliography", "sources", "footnotes",
}


def _parse_wiki_extract(extract: str) -> list[dict]:
    """Wikipedia exsectionformat=wiki yields == Heading == markers."""
    parts = _WIKI_HEADING_RE.split(extract)
    # Split returns: [intro, eq1, head1, body1, eq2, head2, body2, ...]
    sections = []
    intro = parts[0].strip()
    if intro and len(intro) >= 100:
        sections.append({"section": "Overview", "text": intro})
    i = 1
    while i + 2 < len(parts):
        heading = parts[i + 1].strip()
        body = parts[i + 2].strip()
        i += 3
        if heading.lower() in _WIKI_SKIP_SECTIONS:
            continue
        if len(body) < 100:
            continue
        sections.append({"section": heading, "text": body})
    return sections


# --------------------------- Driver ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="data/rag/raw")
    p.add_argument("--skip-statpearls", action="store_true")
    p.add_argument("--skip-wikipedia", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _make_session()

    n_done = n_skip = n_fail = 0
    t0 = time.time()
    for i, t in enumerate(TARGETS):
        slug = t["slug"]
        # We write two files per target: <slug>__statpearls.json + <slug>__wiki.json
        sp_path = out_dir / f"{slug}__statpearls.json"
        wk_path = out_dir / f"{slug}__wiki.json"
        progress = f"[{i+1:02d}/{len(TARGETS)}] {slug}"

        # StatPearls
        if not args.skip_statpearls:
            if sp_path.exists():
                n_skip += 1
                print(f"{progress} statpearls: skip (exists)", flush=True)
            else:
                print(f"{progress} statpearls: '{t['search']}' ...", flush=True)
                art = fetch_statpearls(session, t["search"])
                if art is None:
                    n_fail += 1
                else:
                    art["finding_keys"] = t["keys"]
                    art["slug"] = f"{slug}__statpearls"
                    with open(sp_path, "w", encoding="utf-8") as f:
                        json.dump(art, f, indent=2, ensure_ascii=False)
                    chars = sum(len(s["text"]) for s in art["sections"])
                    print(f"  [ok] sp: {len(art['sections'])} sec, "
                          f"{chars:,} chars  -> {sp_path.name}", flush=True)
                    n_done += 1

        # Wikipedia
        if not args.skip_wikipedia:
            if wk_path.exists():
                n_skip += 1
                print(f"{progress} wiki: skip (exists)", flush=True)
            else:
                print(f"{progress} wiki: '{t['wiki']}' ...", flush=True)
                art = fetch_wikipedia(session, t["wiki"])
                if art is None:
                    n_fail += 1
                else:
                    art["finding_keys"] = t["keys"]
                    art["slug"] = f"{slug}__wiki"
                    with open(wk_path, "w", encoding="utf-8") as f:
                        json.dump(art, f, indent=2, ensure_ascii=False)
                    chars = sum(len(s["text"]) for s in art["sections"])
                    print(f"  [ok] wk: {len(art['sections'])} sec, "
                          f"{chars:,} chars  -> {wk_path.name}", flush=True)
                    n_done += 1

    elapsed = time.time() - t0
    print(f"\n[done] new={n_done} skipped={n_skip} failed={n_fail} "
          f"in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
