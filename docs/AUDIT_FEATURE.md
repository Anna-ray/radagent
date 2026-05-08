# RadAgent Self-Audit Layer

## Overview

The self-audit layer runs **after** the VLM generates its report and flags claims that disagree with the specialist, Grad-CAM, or retrieved passages. This provides a second layer of verification to catch potential hallucinations.

## Architecture

```
VLM Report → Audit Layer → Flagged Claims
                ↓
    ┌───────────┼───────────┐
    │           │           │
Numerical   Spatial    Citation
  Audit      Audit       Audit
    ↓           ↓           ↓
Specialist  Grad-CAM   Retrieved
  Probs     Heatmaps   Passages
```

## Three Audit Types

### 1. Numerical Audit
**Question**: Does the VLM claim match the specialist's probability?

- If VLM asserts a finding but `specialist_prob < threshold`, flag it
- If VLM omits a finding but `specialist_prob >= threshold`, flag it
- Default threshold: 0.5

**Example**:
```
Claim: "Pleural effusion is present in the right costophrenic angle"
Specialist: Effusion probability = 0.12
→ FLAGGED: VLM asserts Effusion but specialist p=0.12 < 0.5
```

### 2. Spatial Audit
**Question**: Does the VLM's anatomical localization match Grad-CAM attention?

- Parse anatomical region from claim ("left lower lobe", "right upper", etc.)
- Divide image into 3×3 grid (upper/mid/lower × left/center/right)
- Find grid cell containing Grad-CAM peak
- Flag if claim region ≠ peak region

**Example**:
```
Claim: "Cardiomegaly with left-sided prominence"
Grad-CAM peak: center-center (cardiac silhouette)
→ PASS: Spatial agreement

Claim: "Right upper lobe infiltrate"
Grad-CAM peak: left-lower
→ FLAGGED: Spatial mismatch
```

### 3. Citation Audit
**Question**: Does the cited passage actually support the claim?

- Embed claim and passage with BGE-M3 (same embedder as RAG)
- Compute cosine similarity
- Flag if similarity < threshold
- Default threshold: 0.4

**Example**:
```
Claim: "Cardiomegaly is defined by a cardiothoracic ratio >0.5 [1]"
Passage [1]: "Cardiomegaly on chest radiograph is typically diagnosed when..."
Similarity: 0.82
→ PASS: Strong citation

Claim: "Pneumothorax requires immediate chest tube placement [2]"
Passage [2]: "Pneumothorax is air in the pleural space..."
Similarity: 0.28
→ FLAGGED: Weak citation (passage doesn't discuss treatment)
```

## Usage

### From Command Line

```bash
python scripts/run_audit.py \
    --run-dir runs/nih14_convnextv2_base_384/predict_one_test \
    --output audit_results.json \
    --numerical-threshold 0.5 \
    --citation-threshold 0.4
```

**Input files** (expected in `--run-dir`):
- `structured_findings.json` — specialist probabilities
- `gradcam_*.png` — Grad-CAM heatmaps (one per finding)
- `retrieved_passages.json` — RAG results
- `vlm_report.txt` — VLM-generated report

**Output**: `audit_results.json`

### From Python

```python
from radagent.inference.audit import audit_report
from sentence_transformers import SentenceTransformer

# Load embedder
embedder = SentenceTransformer("BAAI/bge-m3")

# Run audit
results = audit_report(
    report_text=vlm_report,
    specialist_probs={"Cardiomegaly": 0.998, "Effusion": 0.12, ...},
    gradcam_dict={"Cardiomegaly": np.array(...), ...},
    retrieved_passages=[{"text": "...", ...}, ...],
    embedder=embedder,
    image_shape=(384, 384),
    numerical_threshold=0.5,
    citation_threshold=0.4,
)

# Check summary
print(f"Flagged {results['summary']['flagged_claims']} / {results['summary']['total_claims']} claims")

# Inspect flagged claims
for claim in results["claims"]:
    if claim["any_flagged"]:
        print(f"❌ {claim['claim']}")
        print(f"   Severity: {claim['max_severity']:.2f}")
```

## Output Format

```json
{
  "claims": [
    {
      "claim": "Cardiomegaly is present with cardiothoracic ratio >0.5 [1]",
      "citations": [1],
      "audits": {
        "numerical": [
          {
            "finding": "Cardiomegaly",
            "passed": true,
            "reason": "Agreement: specialist p=0.998, threshold=0.5",
            "severity": 0.0
          }
        ],
        "spatial": [
          {
            "finding": "Cardiomegaly",
            "passed": true,
            "reason": "Spatial agreement: claim=(None,center), peak=(mid,center)",
            "severity": 0.0
          }
        ],
        "citation": [
          {
            "citation_id": 1,
            "passed": true,
            "reason": "Strong citation: similarity=0.82",
            "severity": 0.0
          }
        ]
      },
      "any_flagged": false,
      "max_severity": 0.0
    }
  ],
  "summary": {
    "total_claims": 12,
    "flagged_claims": 2,
    "flag_rate": 0.167
  }
}
```

## Severity Levels

- **0.0**: No issue (audit passed)
- **0.1-0.3**: Minor disagreement (low severity)
- **0.4-0.6**: Moderate disagreement (medium severity)
- **0.7-1.0**: Major disagreement (high severity)

Severity is computed as:
- **Numerical**: `1.0 - specialist_prob` (when VLM asserts but specialist says no)
- **Spatial**: `0.5` (one dimension mismatch) or `1.0` (both dimensions mismatch)
- **Citation**: `1.0 - cosine_similarity`

## Integration with Pipeline

The audit layer is **post-hoc** and does not modify the existing pipeline. To integrate:

1. Run the normal pipeline (specialist → RAG → Grad-CAM → VLM)
2. Save intermediate outputs (findings, gradcams, passages, report)
3. Run `scripts/run_audit.py` on the output directory
4. Review flagged claims before presenting to user

**Future work**: Real-time audit in the WebSocket pipeline with UI indicators for flagged claims.

## Limitations

- **Claim parsing**: Uses simple sentence splitting and regex for citations. May miss complex sentence structures.
- **Anatomical parsing**: Keyword-based. May miss synonyms or complex anatomical descriptions.
- **Citation audit**: Semantic similarity doesn't guarantee logical entailment (passage may be topically related but not support the specific claim).
- **No negation handling**: "No cardiomegaly" and "Cardiomegaly" both trigger the same finding keyword match.

## Example Session

```bash
$ python scripts/run_audit.py --run-dir runs/predict_one_test

[audit] Loading files from runs/predict_one_test
[audit] Loaded 14 specialist probabilities
[audit] Loaded 3 Grad-CAM heatmaps
[audit] Loaded 9 retrieved passages
[audit] Loaded VLM report (1247 chars)
[audit] Loading embedder: BAAI/bge-m3
[audit] Embedder loaded on cuda
[audit] Running audit...
[audit] Results saved to runs/predict_one_test/audit_results.json

============================================================
AUDIT SUMMARY
============================================================
Total claims:   8
Flagged claims: 1
Flag rate:      12.5%
============================================================

FLAGGED CLAIMS:

[3] Mild pleural effusion is noted in the right costophrenic angle...
    Max severity: 0.88
    ❌ numerical: VLM asserts 'Effusion' but specialist probability is 0.12 (< 0.5)

[audit] Done.
```

## Configuration

All thresholds are tunable:

```python
# Conservative (fewer flags, higher precision)
audit_report(..., numerical_threshold=0.7, citation_threshold=0.5)

# Aggressive (more flags, higher recall)
audit_report(..., numerical_threshold=0.3, citation_threshold=0.3)
```

Recommended defaults (balanced):
- `numerical_threshold=0.5` (specialist probability)
- `citation_threshold=0.4` (cosine similarity)

---

**Status**: Implemented in `audit-feature` branch. Not yet integrated into main pipeline.