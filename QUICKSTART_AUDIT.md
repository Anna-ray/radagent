# Quick Start: Running the Audit Layer

## Prerequisites

You need a completed RadAgent pipeline run with these output files:
- `structured_findings.json`
- `gradcam_*.png` files
- `retrieved_passages.json`
- `vlm_report.txt`

## Step 1: Prepare Test Data

First, run a single prediction to generate the required files:

```bash
# Make sure you're on the audit-feature branch
git checkout audit-feature

# Run a single prediction (this creates the output directory)
python scripts/predict_one.py \
    --config configs/nih14_convnextv2_base.yaml \
    --image path/to/your/chest_xray.png \
    --checkpoint runs/nih14_convnextv2_base_384/best.pt \
    --calibration runs/nih14_convnextv2_base_384/calibration.json \
    --bands runs/nih14_convnextv2_base_384/calibration_bands.json \
    --rag-index data/rag/index.faiss \
    --rag-chunks data/rag/chunks.jsonl \
    --rag-manifest data/rag/manifest.json \
    --output-dir runs/test_audit \
    --gradcam
```

This will create `runs/test_audit/` with all required files.

## Step 2: Run the Audit

```bash
python scripts/run_audit.py \
    --run-dir runs/test_audit \
    --output audit_results.json
```

**Expected output:**
```
[audit] Loading files from runs/test_audit
[audit] Loaded 14 specialist probabilities
[audit] Loaded 3 Grad-CAM heatmaps
[audit] Loaded 9 retrieved passages
[audit] Loaded VLM report (1247 chars)
[audit] Loading embedder: BAAI/bge-m3
[audit] Embedder loaded on cuda
[audit] Running audit...
[audit] Results saved to runs/test_audit/audit_results.json

============================================================
AUDIT SUMMARY
============================================================
Total claims:   8
Flagged claims: 1
Flag rate:      12.5%
============================================================

FLAGGED CLAIMS:

[3] Mild pleural effusion is noted...
    Max severity: 0.88
    ❌ numerical: VLM asserts 'Effusion' but specialist probability is 0.12 (< 0.5)

[audit] Done.
```

## Step 3: Review Results

Open the generated `audit_results.json`:

```bash
# Windows
notepad runs/test_audit/audit_results.json

# Linux/Mac
cat runs/test_audit/audit_results.json | jq .
```

## Quick Test Without Full Pipeline

If you don't have a full pipeline run yet, you can test with mock data:

```bash
# Create test directory
mkdir -p runs/mock_test

# Create minimal test files
cat > runs/mock_test/structured_findings.json << 'EOF'
{
  "findings": [
    {"name": "Cardiomegaly", "calibrated_probability": 0.998},
    {"name": "Effusion", "calibrated_probability": 0.12},
    {"name": "Pneumonia", "calibrated_probability": 0.05}
  ]
}
EOF

cat > runs/mock_test/retrieved_passages.json << 'EOF'
{
  "Cardiomegaly": [
    {
      "text": "Cardiomegaly on chest radiograph is diagnosed when the cardiothoracic ratio exceeds 0.5",
      "title": "Cardiomegaly",
      "section": "Imaging",
      "source_url": "https://example.com"
    }
  ]
}
EOF

cat > runs/mock_test/vlm_report.txt << 'EOF'
FINDINGS:
Cardiomegaly is present with cardiothoracic ratio >0.5 [1].
Mild pleural effusion is noted in the right costophrenic angle.

IMPRESSION:
Enlarged cardiac silhouette consistent with cardiomegaly.

RECOMMENDATIONS:
Clinical correlation recommended.
EOF

# Run audit (will skip Grad-CAM audit due to missing files)
python scripts/run_audit.py --run-dir runs/mock_test
```

## Troubleshooting

### Error: "Missing structured_findings.json"
**Solution**: Run `predict_one.py` first to generate the required files.

### Error: "Import sentence_transformers could not be resolved"
**Solution**: Install dependencies:
```bash
pip install sentence-transformers
```

### Error: "CUDA out of memory"
**Solution**: Use CPU for embedder:
```bash
# Edit scripts/run_audit.py line 148:
# Change: device = "cuda" if torch.cuda.is_available() else "cpu"
# To:     device = "cpu"
```

Or reduce batch size by processing claims one at a time.

### No Grad-CAM files found
**Warning only** - spatial audit will be skipped. To generate Grad-CAMs:
```bash
python scripts/predict_one.py ... --gradcam
```

## Advanced Usage

### Custom Thresholds

```bash
# More conservative (fewer flags)
python scripts/run_audit.py \
    --run-dir runs/test_audit \
    --numerical-threshold 0.7 \
    --citation-threshold 0.5

# More aggressive (more flags)
python scripts/run_audit.py \
    --run-dir runs/test_audit \
    --numerical-threshold 0.3 \
    --citation-threshold 0.3
```

### Different Image Size

```bash
python scripts/run_audit.py \
    --run-dir runs/test_audit \
    --image-size 512  # if you used 512x512 during inference
```

### Different Embedder

```bash
python scripts/run_audit.py \
    --run-dir runs/test_audit \
    --embedder-model "sentence-transformers/all-MiniLM-L6-v2"  # faster, smaller
```

## Integration with Existing Workflow

### Option 1: Post-Process Existing Runs

If you already have pipeline outputs:

```bash
# Find existing run directories
ls runs/nih14_convnextv2_base_384/

# Run audit on each
for dir in runs/nih14_convnextv2_base_384/predict_*; do
    python scripts/run_audit.py --run-dir "$dir"
done
```

### Option 2: Add to predict_one.py

Add at the end of `scripts/predict_one.py`:

```python
# After saving all outputs
if args.run_audit:
    from radagent.inference.audit import audit_report
    from sentence_transformers import SentenceTransformer
    
    embedder = SentenceTransformer("BAAI/bge-m3")
    audit_results = audit_report(...)
    
    with open(output_dir / "audit_results.json", "w") as f:
        json.dump(audit_results, f, indent=2)
```

## What to Look For

### High Severity Flags (>0.7)
- **Numerical**: VLM invented a finding the specialist strongly disagrees with
- **Spatial**: VLM localized to completely wrong region
- **Citation**: Citation is irrelevant to the claim

### Medium Severity Flags (0.4-0.7)
- **Numerical**: Borderline disagreement
- **Spatial**: Partial mismatch (one dimension correct)
- **Citation**: Weak topical relevance

### Low Severity Flags (<0.4)
- Minor disagreements, may be acceptable

## Next Steps

1. ✅ Run audit on your test cases
2. ✅ Review flagged claims manually
3. ✅ Tune thresholds based on your tolerance
4. ✅ Consider integrating into main pipeline after hackathon

---

**Questions?** See `docs/AUDIT_FEATURE.md` for detailed documentation.