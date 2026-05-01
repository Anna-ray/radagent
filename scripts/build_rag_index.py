"""
scripts/build_rag_index.py
--------------------------
Chunk scraped Radiopaedia articles, embed with BGE-M3, build FAISS index.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=str, default="data/rag/raw")
    p.add_argument("--output-dir", type=str, default="data/rag")
    p.add_argument("--max-tokens", type=int, default=480)
    p.add_argument("--overlap-tokens", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _split_by_tokens(text, enc, max_tokens, overlap):
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return [text]
    chunks = []
    step = max_tokens - overlap
    if step <= 0:
        step = max_tokens
    for start in range(0, len(ids), step):
        sub = ids[start:start + max_tokens]
        if not sub:
            break
        chunks.append(enc.decode(sub))
        if start + max_tokens >= len(ids):
            break
    return chunks


def _build_chunks_for_article(article, enc, max_tokens, overlap):
    out = []
    slug = article["slug"]
    title = article["title"]
    src = article["source_url"]
    fkeys = article.get("finding_keys", [])
    license_ = article.get("license", "CC-BY-NC-SA")
    source = article.get("source", "radiopaedia")
    for sec in article["sections"]:
        section_name = sec["section"]
        section_text = sec["text"]
        prefix = f"{title} - {section_name}: "
        sub_chunks = _split_by_tokens(section_text, enc, max_tokens, overlap)
        for k, txt in enumerate(sub_chunks):
            chunk_text = prefix + txt
            out.append({
                "chunk_id": f"{slug}::{section_name}::{k}",
                "text": chunk_text,
                "title": title,
                "section": section_name,
                "source_url": src,
                "finding_keys": fkeys,
                "license": license_,
                "source": source,
            })
    return out


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No raw articles found in {raw_dir}")
    print(f"[build] {len(files)} raw articles", flush=True)

    enc = tiktoken.get_encoding("cl100k_base")
    all_chunks = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            article = json.load(f)
        chunks = _build_chunks_for_article(article, enc, args.max_tokens, args.overlap_tokens)
        all_chunks.extend(chunks)
    print(f"[build] {len(all_chunks)} chunks", flush=True)
    if not all_chunks:
        raise RuntimeError("No chunks produced.")

    print(f"[embed] loading {EMBED_MODEL} on {args.device} ...", flush=True)
    model = SentenceTransformer(EMBED_MODEL, device=args.device)
    texts = [c["text"] for c in all_chunks]
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    print(f"[embed] {emb.shape} in {time.time()-t0:.1f}s", flush=True)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    index_path = out_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"[out] {index_path}  ntotal={index.ntotal}", flush=True)

    chunks_path = out_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"[out] {chunks_path}", flush=True)

    manifest = {
        "embed_model": EMBED_MODEL,
        "embed_dim": int(emb.shape[1]),
        "n_chunks": len(all_chunks),
        "n_articles": len(files),
        "max_tokens": args.max_tokens,
        "overlap_tokens": args.overlap_tokens,
        "license_summary": "Radiopaedia content under CC-BY-NC-SA. "
                           "Attribution required via source_url.",
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[out] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
