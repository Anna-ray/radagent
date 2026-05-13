# Model Card — RadAgent Specialist (NIH-14 ConvNeXt-V2)

This model card follows the format of Mitchell et al. (2019, "Model Cards for
Model Reporting") adapted for medical imaging. It accompanies the trained
checkpoint at `runs/nih14_convnextv2_base_384/best.pt`.

## Model Details

- **Person developing model**: Rayane Aggoune (solo researcher, PhD candidate in AI)
- **Model date**: April 2026
- **Model version**: v1.0 (`nih14_convnextv2_base_384/best.pt`, 1.42 GB)
- **Model type**: 14-class multi-label image classifier
- **Architecture**: ConvNeXt-V2 Base, FCMAE-pretrained backbone, 2-layer linear head
- **Input resolution**: 384×384, 3-channel RGB (CLAHE-replicated grayscale)
- **Training procedure**:
  - Asymmetric Loss (γ⁻=4, γ⁺=1, clip=0.05)
  - EMA decay 0.9999
  - AdamW, discriminative LR (head 5e-4, backbone 5e-5), cosine warmup
  - Patient-disjoint train/val split, weighted sampler
  - bf16 AMP, gradient checkpointing
  - 12 epochs, batch size 16, single 16 GB GPU
- **Calibration**: Temperature scaling (T = 0.1979) + per-class F1-optimal thresholds + reliability-derived confidence bands

## Intended Use

### Primary intended uses
- Research: chest X-ray multi-label finding detection on NIH-14 and similar academic datasets
- Educational: teaching about calibration, attribution, and grounded multimodal AI
- A backbone for systems that combine specialist outputs with retrieval and language models (as RadAgent itself does)

### Primary intended users
- ML researchers working on medical imaging
- Educators and students in AI/medicine intersection
- Hackathon / competition participants

### Out-of-scope uses
- ❌ Clinical diagnosis or any patient-facing decision
- ❌ Triage workflows in real hospitals
- ❌ Liability-bearing decisions
- ❌ Settings outside the NIH-14 label distribution (e.g., pediatric CXR, lateral views, CT slices)

## Factors

### Relevant factors
- **Patient demographics**: NIH-14 is dominated by adult patients at the NIH Clinical Center (US, single-institution). Performance on other populations is not evaluated.
- **Imaging device**: NIH-14 frontal X-rays only. Performance on other modalities or non-frontal views is not evaluated.
- **Label noise**: NIH-14 labels are extracted from radiology reports via NLP (DNorm, MetaMap), with documented label noise especially for Pneumonia, Infiltration, and Pneumothorax.

### Evaluation factors
The reported metrics are stratified by class but not by demographic factors,
because NIH-14 does not distribute reliable per-patient demographic metadata.
This is a limitation.

## Metrics

### Primary metric
- **Macro AUC** on the official NIH-14 Wang et al. test split (N=25,596)

### Bootstrap protocol
- Image-level resampling, 1000 iterations
- 95% confidence intervals reported

## Evaluation Data

- **Dataset**: NIH ChestX-ray14 (Wang et al. 2017)
- **Split**: Official Wang et al. train/val/test (test = 25,596 images)
- **Preprocessing**: CLAHE (clip=2.5, fixed at eval), 3-channel replication, 384×384 resize, ImageNet normalization
- **Test-time augmentation**: Horizontal flip averaging (default) — disable with `--no-tta` for ablation

## Training Data

- **Dataset**: NIH ChestX-ray14 (Wang et al. 2017)
- **Train+val pool**: ~86,000 images after patient-disjoint val carve-out
- **Patient-disjoint split**: Critical — without it, val/train share patients and AUC inflates 3-5%
- **Class imbalance**: Severe — Hernia 0.2% positive, Infiltration 18% positive. Handled via weighted sampler + Asymmetric Loss

## Quantitative Analyses

### Per-class AUC on official test split (N=25,596)

| Class | AUC | n_pos (val) | Notes |
|---|---|---|---|
| Emphysema | 0.933 | 128 | High AUC, distinctive radiographic pattern |
| Hernia | 0.918 | 14 | High AUC despite tiny n_pos (rare but distinctive) |
| Cardiomegaly | 0.887 | 184 | Strong |
| Pneumothorax | 0.880 | 267 | Strong |
| Edema | 0.847 | 134 | Strong |
| Effusion | 0.835 | 979 | Solid |
| Mass | 0.832 | 339 | Solid |
| Fibrosis | 0.820 | 130 | Solid |
| Atelectasis | 0.776 | 934 | Common, label noise pulls AUC down |
| Pleural_Thickening | 0.778 | 235 | Subtle finding |
| Nodule | 0.781 | 509 | Small lesion difficulty |
| Consolidation | 0.748 | 309 | Boundary class with Pneumonia/Infiltration |
| Pneumonia | 0.726 | 106 | Heavy label noise (NIH NLP labeler) |
| Infiltration | 0.711 | 1476 | Heavy label noise |

### Aggregate metrics

| Metric | Value | 95% CI |
|---|---|---|
| Macro AUC | 0.8194 | [0.8151, 0.8232] |
| Micro AUC | 0.8634 | [0.8613, 0.8654] |
| Mean F1 | 0.356 | — |
| Mean AP | 0.301 | — |

## Ethical Considerations

- **Clinical safety**: Outputs MUST be paired with grounding (RAG + Grad-CAM) before being shown to anyone in a clinical role. The bare classifier should not be deployed alone.
- **Demographic bias**: NIH-14 is single-institution US adult population. Predictions on under-represented groups (pediatric, geriatric, non-US populations, under-represented ethnicities) are not validated.
- **Label noise**: Some "predictions" may simply be the model fitting noise patterns from the NLP-extracted labels. Especially Pneumonia and Infiltration.
- **Adversarial use**: A confidently-wrong AI radiology tool could harm patients. The grounding layers (RAG, Grad-CAM) and confidence bands exist specifically to surface model uncertainty rather than hide it.

## Caveats and Recommendations

- **Use the calibrated probabilities, not raw sigmoids.** Temperature 0.1979 sharpens the under-confident sigmoids that ASL+EMA produces. Without calibration, "high" confidence ranges look wrong.
- **Use the per-class confidence bands, not a single global threshold.** Rare classes (Hernia, Edema, Pneumonia) need different cutoffs than common ones.
- **Treat rare classes (Hernia, n_val_pos=14) with special skepticism.** Bands fall back to percentile method; CI is wide.
- **Re-eval on your population before any deployment.** This model has only been validated on NIH-14.

## Citation

```bibtex
@misc{aggoune2026radagent_specialist,
  author = {Aggoune, Rayane},
  title  = {RadAgent Specialist: ConvNeXt-V2 NIH-14 Multi-label Classifier},
  year   = {2026},
  note   = {Model card for the RadAgent project, AMD Developer Hackathon 2026},
  url    = {https://github.com/Anna-ray/radagent}
}
```

