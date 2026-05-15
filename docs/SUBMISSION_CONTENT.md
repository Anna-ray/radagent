# RadAgent v2 — Milan AI Week Submission Content
Author: Rayane Aggoune (Sétif, Algeria)

## Tagline
Every claim cites its evidence. Every action carries a receipt.
No patient data leaves the hospital.

## 200-word project summary
RadAgent v2 addresses a fundamental barrier in medical imaging AI: vision-language models hallucinate findings, which blocks adoption in clinical workflows. The project combines a four-pillar architecture to enable trustworthy radiology assistance. A grounded specialist pipeline uses calibrated ConvNeXt-V2 predictions and retrieval-augmented evidence rather than raw image-to-text reasoning. Federated learning trains across NIH ChestX-ray14 and CheXpert nodes with FedAvg so no patient images leave hospital boundaries. Autonomous workflow agents plan, execute, and replan when confidence thresholds are not met, while an auditable SHA-256 receipt chain records every action and decision. The novel CriticAgent actively challenges system outputs, visibly disagreeing with decisions when evidence is weak or uncertainty is high, which makes skepticism explicit and traceable. The stack uses Featherless for Qwen2.5-VL vision-language inference, Google Gemini 2.0 Flash for reasoning, a mock Speechmatics-compatible dictation architecture for voice audit demos, and Vultr for deployment infrastructure. Built solo in 8 days from Sétif, Algeria, RadAgent v2 demonstrates a production-oriented path toward safe imaging AI. Proof includes NIH-14 Macro AUC 0.819 for the specialist and federated training that preserves hospital-local data.

## Sponsor prize eligibility
Claimed: Main Prize, Vultr Sponsor Prize, Google Gemini track
Architecture supports (mocked for demo): Featherless, Speechmatics

## Tech stack
- Vision-language: Qwen2.5-VL via Featherless / vLLM-ROCm
- Reasoning: Google Gemini 2.0 Flash (Google AI Studio free tier)
- Specialist: ConvNeXt-V2 (NIH ChestX-ray14, Macro AUC 0.819)
- Retrieval: BGE-M3 + FAISS over 1,078 chunks (Wikipedia + StatPearls)
- Federated learning: FedAvg with SHA-256 audit chain (NIH-14 + CheXpert)
- Voice: Speechmatics-compatible architecture (mock used for demo)
- DICOM: pydicom-based router supporting 11 modalities
- Deployment: Vultr Cloud Compute (Frankfurt region)
- Audit: SHA-256 hash-linked receipts with CLI verifier

## Repo + Demo URLs
- Repo: https://github.com/Anna-ray/radagent
  - Branch: feature/v2-milan
- Live demo: <will be filled after Vultr deploy>
- Video: <will be filled after Canva production>

## Limitations (honest disclosure)
- Single real agent (CriticAgent); other components are deterministic modules orchestrated by a planner
- Federated learning is FedAvg only; secure aggregation and differential privacy planned for v2.1
- MURA bone-xray specialist scaffolded; weights training v2.1
- Vultr deployment serves pre-cached demo traces; live inference runs on the developer's GPU workstation
