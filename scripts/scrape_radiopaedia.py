"""
scripts/scrape_radiopaedia.py
-----------------------------
Polite scraper for a curated list of Radiopaedia chest-imaging articles.
Hard rate limit: 3 sec. Resumable. Outputs data/rag/raw/<slug>.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://radiopaedia.org"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "RadAgent-Research/0.1 (academic; rad-agent-research@example.org)"
)
ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
ACCEPT_LANG = "en-US,en;q=0.9"
RATE_LIMIT_SEC = 3.0
MAX_RETRIES = 4

ARTICLES: list[tuple[str, list[str]]] = [
    ("cardiomegaly",                ["Cardiomegaly"]),
    ("pleural-effusion",            ["Effusion"]),
    ("pneumothorax",                ["Pneumothorax"]),
    ("pulmonary-oedema",            ["Edema"]),
    ("pulmonary-emphysema",         ["Emphysema"]),
    ("pulmonary-fibrosis",          ["Fibrosis"]),
    ("pulmonary-consolidation",     ["Consolidation"]),
    ("pneumonia",                   ["Pneumonia"]),
    ("atelectasis",                 ["Atelectasis"]),
    ("pulmonary-mass",              ["Mass"]),
    ("pulmonary-nodule",            ["Nodule"]),
    ("pleural-thickening",          ["Pleural_Thickening"]),
    ("hiatus-hernia",               ["Hernia"]),
    ("pulmonary-infiltrate",        ["Infiltration"]),
    ("chest-radiograph",            []),
    ("cardiothoracic-ratio",        ["Cardiomegaly"]),
    ("kerley-lines",                ["Edema"]),
    ("cardiogenic-pulmonary-oedema", ["Edema"]),
    ("non-cardiogenic-pulmonary-oedema", ["Edema"]),
    ("simple-pneumothorax",         ["Pneumothorax"]),
    ("tension-pneumothorax",        ["Pneumothorax"]),
    ("hydropneumothorax",           ["Pneumothorax", "Effusion"]),
    ("haemothorax",                 ["Effusion"]),
    ("empyema",                     ["Effusion"]),
    ("loculated-pleural-effusion",  ["Effusion"]),
    ("transudate-vs-exudate",       ["Effusion"]),
    ("lobar-pneumonia",             ["Pneumonia", "Consolidation"]),
    ("bronchopneumonia",            ["Pneumonia"]),
    ("aspiration-pneumonia",        ["Pneumonia", "Consolidation"]),
    ("viral-pneumonia",             ["Pneumonia"]),
    ("round-pneumonia",             ["Pneumonia", "Mass"]),
    ("lobar-collapse",              ["Atelectasis"]),
    ("plate-atelectasis",           ["Atelectasis"]),
    ("rounded-atelectasis",         ["Atelectasis", "Mass"]),
    ("solitary-pulmonary-nodule",   ["Nodule"]),
    ("multiple-pulmonary-nodules",  ["Nodule"]),
    ("cavitating-pulmonary-nodule", ["Nodule", "Mass"]),
    ("lung-cancer-staging",         ["Mass", "Nodule"]),
    ("primary-lung-cancer",         ["Mass", "Nodule"]),
    ("centrilobular-emphysema",     ["Emphysema"]),
    ("paraseptal-emphysema",        ["Emphysema"]),
    ("panlobular-emphysema",        ["Emphysema"]),
    ("idiopathic-pulmonary-fibrosis", ["Fibrosis"]),
    ("usual-interstitial-pneumonia", ["Fibrosis", "Infiltration"]),
    ("nonspecific-interstitial-pneumonia", ["Fibrosis", "Infiltration"]),
    ("ground-glass-opacification",  ["Infiltration"]),
    ("airspace-opacification",      ["Consolidation", "Infiltration"]),
    ("interstitial-lung-disease",   ["Fibrosis", "Infiltration"]),
    ("apical-pleural-cap",          ["Pleural_Thickening"]),
    ("calcified-pleural-plaques",   ["Pleural_Thickening"]),
    ("asbestos-related-pleural-disease", ["Pleural_Thickening"]),
    ("hiatal-hernia",               ["Hernia"]),
    ("morgagni-hernia",             ["Hernia"]),
    ("bochdalek-hernia",            ["Hernia"]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="data/rag/raw")
    p.add_argument("--rate-limit", type=float, default=RATE_LIMIT_SEC)
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    return p.parse_args()


def _polite_get(session, url, rate_limit, max_retries):
    backoff = 4.0
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=30, allow_redirects=True)
        except requests.RequestException as e:
            print(f"  [http] {url}: {e} (attempt {attempt+1})", flush=True)
            time.sleep(backoff + random.uniform(0, 1))
            backoff *= 2
            continue
        if resp.status_code == 200:
            time.sleep(rate_limit)
            return resp
        if resp.status_code == 404:
            print(f"  [http] 404 {url}", flush=True)
            time.sleep(rate_limit)
            return None
        if resp.status_code in (406, 429, 500, 502, 503, 504):
            wait = float(resp.headers.get("Retry-After", backoff))
            print(f"  [http] {resp.status_code} on {url}, backing off {wait:.1f}s",
                  flush=True)
            time.sleep(wait + random.uniform(0, 2))
            backoff *= 2
            continue
        print(f"  [http] {resp.status_code} {url}", flush=True)
        return None
    print(f"  [http] giving up on {url}", flush=True)
    return None


def _parse_article(html, source_url):
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else "(untitled)"
    body = (
        soup.select_one("div.body.user-generated-content")
        or soup.select_one("div.user-generated-content")
        or soup.select_one("article")
        or soup.find("body")
    )
    if body is None:
        return None
    for sel in ["script", "style", "nav", "aside", "noscript",
                ".edit-link", ".reference-list-toggle", ".js-toolbar"]:
        for el in body.select(sel):
            el.decompose()

    sections = []
    current_heading = "Overview"
    current_chunks: list[str] = []
    for el in body.find_all(["h2", "h3", "h4", "p", "li"], recursive=True):
        if el.name in ("h2", "h3"):
            if current_chunks:
                txt = "\n".join(current_chunks).strip()
                if txt:
                    sections.append({"section": current_heading, "text": txt})
                current_chunks = []
            current_heading = el.get_text(strip=True) or current_heading
        else:
            t = el.get_text(" ", strip=True)
            if t:
                current_chunks.append(t)
    if current_chunks:
        txt = "\n".join(current_chunks).strip()
        if txt:
            sections.append({"section": current_heading, "text": txt})
    if not sections:
        return None
    return {
        "title": title,
        "source_url": source_url,
        "license": "CC-BY-NC-SA",
        "source": "radiopaedia",
        "sections": sections,
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": ACCEPT,
        "Accept-Language": ACCEPT_LANG,
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    print(f"[scrape] {len(ARTICLES)} articles, {args.rate_limit}s rate limit, "
          f"output={out_dir}", flush=True)

    n_done = n_skip = n_fail = 0
    t0 = time.time()
    for i, (slug, finding_keys) in enumerate(ARTICLES):
        out_path = out_dir / f"{slug}.json"
        if out_path.exists():
            n_skip += 1
            continue
        url = urljoin(BASE_URL, f"/articles/{slug}")
        print(f"[{i+1:02d}/{len(ARTICLES)}] {slug} ...", flush=True)
        resp = _polite_get(session, url, args.rate_limit, args.max_retries)
        if resp is None:
            n_fail += 1
            continue
        parsed = _parse_article(resp.text, source_url=url)
        if parsed is None:
            print(f"  [parse] no sections extracted from {slug}", flush=True)
            n_fail += 1
            continue
        parsed["finding_keys"] = finding_keys
        parsed["slug"] = slug
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        n_chars = sum(len(s["text"]) for s in parsed["sections"])
        print(f"  [ok] {len(parsed['sections'])} sections, ~{n_chars:,} chars",
              flush=True)
        n_done += 1

    elapsed = time.time() - t0
    print(f"\n[done] new={n_done} skipped={n_skip} failed={n_fail} "
          f"in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
