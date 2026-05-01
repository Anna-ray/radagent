"""
scripts/fix_statpearls_titles.py
--------------------------------
One-shot repair: rewrite StatPearls articles whose title is "Bookshelf"
with a sensible title derived from the slug. Wikipedia files untouched.

Then re-run scripts.build_rag_index to regenerate index + chunks.
"""
from __future__ import annotations

import json
from pathlib import Path

# Maps slug stem -> proper title
SLUG_TO_TITLE = {
    "cardiomegaly":            "Cardiomegaly",
    "pleural-effusion":        "Pleural Effusion",
    "pneumothorax":            "Pneumothorax",
    "pulmonary-edema":         "Pulmonary Edema",
    "emphysema":               "Emphysema",
    "pulmonary-fibrosis":      "Pulmonary Fibrosis",
    "consolidation":           "Pulmonary Consolidation",
    "pneumonia":               "Pneumonia",
    "atelectasis":             "Atelectasis",
    "lung-nodule":             "Pulmonary Nodule",
    "lung-mass":               "Lung Cancer",
    "pleural-thickening":      "Pleural Plaque",
    "hiatal-hernia":           "Hiatal Hernia",
    "infiltrate":              "Pulmonary Infiltrate",
    "chest-radiograph":        "Chest Radiograph Interpretation",
    "lobar-pneumonia":         "Lobar Pneumonia",
    "aspiration-pneumonia":    "Aspiration Pneumonia",
    "viral-pneumonia":         "Viral Pneumonia",
    "tension-pneumothorax":    "Tension Pneumothorax",
    "empyema":                 "Empyema",
    "hemothorax":              "Hemothorax",
    "copd":                    "Chronic Obstructive Pulmonary Disease",
    "ipf":                     "Idiopathic Pulmonary Fibrosis",
    "ild":                     "Interstitial Lung Disease",
    "lung-cancer-staging":     "Lung Cancer Staging",
    "ards":                    "Acute Respiratory Distress Syndrome",
    "heart-failure":           "Congestive Heart Failure",
    "asbestos-related-lung":   "Asbestosis",
}


def main():
    raw_dir = Path("data/rag/raw")
    n_fixed = 0
    n_skip = 0
    for fp in sorted(raw_dir.glob("*__statpearls.json")):
        with open(fp, encoding="utf-8") as f:
            art = json.load(f)
        if art.get("title") != "Bookshelf":
            n_skip += 1
            continue
        # slug field is e.g. "cardiomegaly__statpearls"; take left of "__"
        stem = art["slug"].split("__", 1)[0]
        new_title = SLUG_TO_TITLE.get(stem)
        if new_title is None:
            print(f"  [skip] no title mapping for {stem}", flush=True)
            continue
        art["title"] = new_title
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(art, f, indent=2, ensure_ascii=False)
        print(f"  [ok] {fp.name}: 'Bookshelf' -> '{new_title}'", flush=True)
        n_fixed += 1
    print(f"\n[done] fixed={n_fixed} skipped={n_skip}", flush=True)


if __name__ == "__main__":
    main()
