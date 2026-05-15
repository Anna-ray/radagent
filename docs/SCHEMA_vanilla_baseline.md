# Vanilla Baseline Output Schema

**Purpose:** Define the JSON schema for vanilla baseline output so Priority 2 dashboard can build against a stable contract.

**Author:** Rayane Aggoune  
**Date:** May 13, 2026

---

## Output File Structure

```
runs/vanilla_baseline/
└── <image_id>/
    ├── output.json          # Main output (schema below)
    └── metadata.json        # Request metadata
```

---

## Main Output Schema (`output.json`)

```json
{
  "request_id": "vanilla_baseline_abc123",
  "image_path": "path/to/image.jpg",
  "timestamp": "2026-05-13T12:00:00Z",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "api_provider": "Featherless",
  "grounding_status": "UNGROUNDED",
  
  "report": {
    "findings": "Bilateral infiltrates present. Possible pneumonia. Cardiomegaly noted. Pleural effusion on the right side.",
    "impression": "Acute cardiopulmonary process. Recommend clinical correlation and follow-up imaging.",
    "full_text": "FINDINGS:\nBilateral infiltrates present. Possible pneumonia. Cardiomegaly noted. Pleural effusion on the right side.\n\nIMPRESSION:\nAcute cardiopulmonary process. Recommend clinical correlation and follow-up imaging."
  },
  
  "fabricated_claims": [
    {
      "claim": "Bilateral infiltrates present",
      "confidence": "stated_as_fact",
      "evidence_provided": false,
      "hallucination_risk": "high",
      "reason": "Specialist confidence on Infiltration is 0.04, below 0.38 threshold — claim unsupported by image evidence"
    },
    {
      "claim": "Possible pneumonia",
      "confidence": "hedged",
      "evidence_provided": false,
      "hallucination_risk": "medium",
      "reason": "Specialist confidence on Pneumonia is 0.12, below 0.41 threshold — hedged language reduces risk but still unsupported"
    },
    {
      "claim": "Cardiomegaly noted",
      "confidence": "stated_as_fact",
      "evidence_provided": false,
      "hallucination_risk": "high",
      "reason": "Specialist confidence on Cardiomegaly is 0.23, below 0.42 threshold — claim stated as fact without evidence"
    },
    {
      "claim": "Pleural effusion on the right side",
      "confidence": "stated_as_fact",
      "evidence_provided": false,
      "hallucination_risk": "high",
      "reason": "Specialist confidence on Effusion is 0.08, below 0.45 threshold — laterality claim (right side) adds specificity without evidence"
    }
  ],
  
  "metadata": {
    "prompt": "You are a radiologist. Describe what you see in this chest X-ray. Provide findings and impression.",
    "temperature": 0.7,
    "max_tokens": 500,
    "processing_time_ms": 1234,
    "warning": "UNGROUNDED VANILLA — for comparison only. No specialist findings, no retrieved evidence, no citation requirement."
  }
}
```

---

## Metadata File Schema (`metadata.json`)

```json
{
  "request_id": "vanilla_baseline_abc123",
  "image_filename": "sample_001.jpg",
  "image_size_bytes": 123456,
  "timestamp_utc": "2026-05-13T12:00:00Z",
  "featherless_api_key_used": true,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct"
}
```

---

## Dashboard Integration Contract

### For Side-by-Side Comparison Panel (Priority 2)

The dashboard will consume:

1. **Vanilla Baseline** (left pane, red):
   - Source: `runs/vanilla_baseline/<image_id>/output.json`
   - Display: `report.full_text`
   - Highlight: `fabricated_claims[]` with red underline
   - Badge: "❌ UNGROUNDED (raw VLM)"

2. **RadAgent Grounded** (right pane, green):
   - Source: Existing v1 pipeline output
   - Display: VLM report with citations
   - Highlight: Citations as clickable links
   - Badge: "✅ RadAgent (grounded)"

### Comparison Logic

```javascript
// Dashboard will render:
{
  left: {
    title: "❌ Ungrounded (raw VLM)",
    subtitle: "Image-only prompt — no specialist, no RAG, no citations",
    content: vanilla_baseline.report.full_text,
    highlights: vanilla_baseline.fabricated_claims.map(c => ({
      text: c.claim,
      color: "red",
      tooltip: `Hallucination risk: ${c.hallucination_risk}`
    }))
  },
  right: {
    title: "✅ RadAgent (grounded)",
    subtitle: "Conditioned on specialist + RAG. Every claim cites evidence.",
    content: radagent_output.report,
    highlights: radagent_output.citations.map(c => ({
      text: c.claim,
      color: "green",
      tooltip: c.evidence_refs.join(", ")
    }))
  }
}
```

---

## Key Differences from RadAgent Output

| Field | Vanilla Baseline | RadAgent (v1) |
|-------|------------------|---------------|
| `grounding_status` | "UNGROUNDED" | "GROUNDED" |
| `specialist_findings` | ❌ None | ✅ 14-class calibrated |
| `retrieved_evidence` | ❌ None | ✅ RAG passages |
| `citations` | ❌ None | ✅ [n] references |
| `fabricated_claims` | ✅ Detected | ❌ N/A |
| `confidence_bands` | ❌ None | ✅ Per-class |
| `gradcam_heatmaps` | ❌ None | ✅ Per-finding |

---

## Validation Rules

1. **Required Fields:**
   - `request_id` (string, 16+ chars)
   - `grounding_status` (must be "UNGROUNDED")
   - `report.full_text` (string, non-empty)
   - `fabricated_claims` (array, may be empty)
   - `metadata.warning` (must contain "UNGROUNDED")

3. **Field Descriptions:**
   - `claim`: Exact text extracted from VLM report
   - `confidence`: How definitively the claim was stated
   - `evidence_provided`: Always `false` for vanilla baseline
   - `hallucination_risk`: Severity assessment based on specialist contradiction
   - `reason`: Technical explanation referencing specialist confidence vs threshold

2. **Fabricated Claims Detection:**
   - Parse report for definitive statements
   - Flag any claim without hedging ("possible", "may", "suggest")
   - Mark all as `evidence_provided: false`
   - Assign hallucination risk: high/medium/low

3. **Compatibility:**
   - Must be JSON-serializable
   - Must not exceed 1 MB file size
   - Must be UTF-8 encoded

---

## Example Usage

```bash
# Generate vanilla baseline
python scripts/run_vanilla_baseline.py \
  --image data/sample_001.jpg \
  --output runs/vanilla_baseline

# Output files:
# runs/vanilla_baseline/sample_001/output.json
# runs/vanilla_baseline/sample_001/metadata.json

# Dashboard reads both:
# - Vanilla: runs/vanilla_baseline/sample_001/output.json
# - RadAgent: runs/agentic_rag/sample_001/output.json
```

---

## Notes

- This schema is **stable** and will not change during Priority 2 implementation
- Dashboard can safely build against this contract
- Any schema changes require updating this document first
- The `fabricated_claims` array is auto-detected by simple heuristics (no LLM call needed)

---

**Schema Version:** 1.0  
**Last Updated:** May 13, 2026  
**Status:** ✅ APPROVED for Priority 2 implementation