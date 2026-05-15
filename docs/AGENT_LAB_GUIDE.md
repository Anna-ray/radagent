# RadAgent v2 - Agent Lab Implementation Guide

**Author:** Rayane Aggoune (Sétif, Algeria)  
**Project:** Milan AI Week 2026 - AI Agent Olympics  
**Repository:** https://github.com/Anna-ray/radagent  
**Branch:** `feature/v2-milan`

---

## 🎯 PROJECT OVERVIEW

RadAgent v2 is an **auditable, federated, autonomous radiology agent** demonstrating four core pillars:

1. **GROUNDED** - Every claim cites evidence from medical literature
2. **FEDERATED** - No patient data leaves the hospital (provably)
3. **AUTONOMOUS** - Agents replan on roadblocks without human intervention
4. **AUDITABLE** - Every action carries a SHA-256 hash-linked receipt

---

## 📦 WHAT'S BEEN BUILT (65% Complete)

### ✅ Track A: Federated Learning (100%)
**Location:** `radagent/federated/`

Real FedAvg implementation with:
- Hospital node client with local training
- FedAvg server with weighted aggregation
- SHA-256 audit chain linking rounds
- NIH ChestX-ray14 + CheXpert data loaders
- 14-class label harmonization
- Fallback to non-IID NIH-14 split if CheXpert unavailable

**Files:**
- `radagent/federated/server.py` (239 lines)
- `radagent/federated/client.py` (189 lines)
- `radagent/data/cxr_datasets.py` (339 lines)
- `scripts/run_federated_demo.py` (378 lines)
- `tests/test_federated.py` (239 lines)

**Run Demo:**
```bash
python scripts/run_federated_demo.py \
  --nih-root /path/to/nih14 \
  --chexpert-root /path/to/chexpert \
  --rounds 5 \
  --samples-per-node 5000 \
  --audit-dir runs/federated_demo
```

---

### ✅ Track B: Workflow Autonomy (100%)
**Location:** `radagent/autonomy/`

Autonomous workflow tools with:
- 4 RAG-grounded tools (triage, route, schedule, flag)
- Per-action confidence floors (0.65-0.85)
- Replan triggers for roadblocks
- Gemini Flash for fast routing decisions
- SHA-256 audit chain

**Files:**
- `radagent/autonomy/tools.py` (398 lines)
- `radagent/autonomy/halt.py` (64 lines)
- `radagent/autonomy/planner.py` (368 lines)
- `scripts/run_autonomy_demo.py` (243 lines)

**Run Demo:**
```bash
python scripts/run_autonomy_demo.py \
  --image path/to/xray.jpg \
  --findings path/to/findings.json \
  --inject-roadblock confidence \
  --audit-dir runs/autonomy_demo
```

---

### ✅ Track C: Universal Modality Router (100%)
**Location:** `radagent/modality/`

Universal DICOM routing with:
- 11 modalities registered (2 production, 9 registered, 1 fallback)
- DICOM I/O with pydicom
- Graceful fallback for unknown modalities
- Preprocessing registry (7 modality-specific functions)

**Registered Modalities:**
- **Production:** chest_xray, bone_xray
- **Registered:** chest_ct, mammography, mri_brain, mri_msk, ultrasound, pet_ct, nuclear_med, xray_angio
- **Fallback:** other (unknown)

**Files:**
- `radagent/modality/registry.yaml` (130 lines)
- `radagent/modality/router.py` (192 lines)
- `radagent/modality/dicom_io.py` (153 lines)
- `radagent/modality/preprocessing.py` (213 lines)
- `scripts/run_modality_demo.py` (207 lines)

**Run Demo:**
```bash
python scripts/run_modality_demo.py \
  --input path/to/study.dcm \
  --audit-dir runs/modality_demo
```

---

### ✅ Track F: Voice Dictation (100%)
**Location:** `radagent/voice/`

Voice-driven dictation auditing with:
- Speechmatics real-time STT integration
- Dictation auditor using Gemini Flash
- Discrepancy detection (RECONSIDER/CONFIRM/CONSISTENT)
- Mock implementations for testing without API keys

**Files:**
- `radagent/voice/transcriber.py` (237 lines)
- `radagent/voice/dictation_auditor.py` (390 lines)
- `scripts/run_dictation_demo.py` (219 lines)
- `tests/test_voice.py` (276 lines)

**Run Demo:**
```bash
python scripts/run_dictation_demo.py \
  --audio path/to/dictation.wav \
  --findings path/to/specialist_findings.json \
  --audit-dir runs/dictation_demo
```

---

### ✅ Cross-Cutting: Audit Chain Verifier
**Location:** `radagent/audit/`

SHA-256 audit chain verifier:
- CLI tool to verify hash chain integrity
- Validates all audit records in a run directory
- Detects broken chains or tampered records

**Files:**
- `radagent/audit/verify.py` (127 lines)

**Run Verifier:**
```bash
python -m radagent.audit.verify runs/federated_demo
```

---

## ⏸️ REMAINING WORK (35%)

### Track D: MURA Bone Specialist (Placeholder)
**Status:** Scaffolding needed  
**Priority:** Medium (can be marked as "registered" without training)

**Required Files:**
- `radagent/specialists/mura/dataset.py`
- `radagent/specialists/mura/train.py`
- `radagent/specialists/mura/calibrate.py`
- `radagent/specialists/mura/infer.py`
- `tests/test_mura_specialist.py`

**Action:** Create placeholder scaffolding with download instructions

---

### Track E: Dashboard Extensions
**Status:** Not started  
**Priority:** High (needed for video recording)

**Required Panels:**
1. Modality badge (top of every run)
2. Side-by-side comparison (vanilla vs RadAgent)
3. Dictation audit panel (transcript + discrepancies)
4. Autonomous queue panel (tool calls + confidence bands)
5. Federation network panel (hospital nodes + weight updates)

**Files to Modify:**
- `radagent/app/static/index.html`
- `radagent/app/server.py`

---

### Track E: Vultr Deployment
**Status:** Not started  
**Priority:** High (needed for public URL)

**Required Files:**
- `Dockerfile`
- `docker-compose.yml`
- `docs/VULTR_DEPLOYMENT.md`

**Action:** Containerize dashboard + autonomy backend

---

### Master Demo Script
**Status:** Not started  
**Priority:** High

**Required File:**
- `scripts/run_full_demo.py`

**Action:** Orchestrate all 6 demo scenes in sequence

---

### Documentation
**Status:** Partially complete  
**Priority:** High

**Required Files:**
- `docs/DEMO_SCRIPT.md` (scene-to-command mapping)
- `README.md` (v2 rewrite with quickstart)
- `docs/SUBMISSION_CONTENT.md` (form content)

---

## 🚀 QUICK START

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys (optional, mock implementations available)
export GEMINI_API_KEY="your-key"
export SPEECHMATICS_API_KEY="your-key"
```

### Run Individual Demos

**Federated Learning (Scene 5):**
```bash
python scripts/run_federated_demo.py \
  --nih-root data/nih14 \
  --rounds 5 \
  --audit-dir runs/fed_demo
```

**Autonomy (Scene 4):**
```bash
python scripts/run_autonomy_demo.py \
  --image data/sample.jpg \
  --inject-roadblock confidence \
  --audit-dir runs/auto_demo
```

**Modality Router (Scene 3):**
```bash
python scripts/run_modality_demo.py \
  --input data/sample.dcm \
  --audit-dir runs/mod_demo
```

**Voice Dictation (Scene 2.5):**
```bash
python scripts/run_dictation_demo.py \
  --audio data/dictation.wav \
  --mock \
  --audit-dir runs/voice_demo
```

### Verify Audit Chain
```bash
python -m radagent.audit.verify runs/fed_demo
```

---

## 📊 CODE METRICS

**Total Production Code:** 4,692 lines  
**Files Created:** 24 production files  
**Git Commits:** 5 clean commits  
**Test Coverage:** Comprehensive for Tracks A, F  
**Author:** Rayane Aggoune (100% of commits)

---

## 🎬 DEMO SCENE MAPPING

| Scene | Description | Status | Script |
|-------|-------------|--------|--------|
| 1 | Vanilla baseline (Featherless) | ⏸️ Pending | TBD |
| 2 | RadAgent CXR pipeline | ✅ Ready | v1 pipeline |
| 2.5 | Voice dictation auditing | ✅ Ready | `run_dictation_demo.py` |
| 3 | Universal modality router | ✅ Ready | `run_modality_demo.py` |
| 4 | Autonomy with replan | ✅ Ready | `run_autonomy_demo.py` |
| 5 | Federation reveal | ✅ Ready | `run_federated_demo.py` |
| 6 | Close (4 pillars) | ✅ Ready | Canva overlay |

---

## 🔗 REPOSITORY STRUCTURE

```
radagent/
├── federated/          # Track A: FedAvg + audit chain
├── autonomy/           # Track B: Workflow tools + planner
├── modality/           # Track C: Universal router
├── voice/              # Track F: STT + dictation auditor
├── audit/              # Audit chain verifier
├── specialists/        # Specialist models (v1 + MURA placeholder)
├── data/               # Data loaders
├── inference/          # v1 agentic-rag pipeline
├── rag/                # RAG retriever
├── app/                # Dashboard (needs extension)
└── utils/              # Utilities

scripts/
├── run_federated_demo.py    # Scene 5
├── run_autonomy_demo.py     # Scene 4
├── run_modality_demo.py     # Scene 3
├── run_dictation_demo.py    # Scene 2.5
└── run_full_demo.py         # TBD: Master script

tests/
├── test_federated.py        # Track A tests
└── test_voice.py            # Track F tests

docs/
├── AGENT_LAB_GUIDE.md       # This file
├── V2_IMPLEMENTATION_STATUS.md
├── RADAGENT_V2_DELIVERABLE.md
└── DEMO_SCRIPT.md           # TBD
```

---

## 🏆 COMPETITIVE ADVANTAGES

1. **Real Federated Learning** - Not simulated, uses real NIH-14 + CheXpert
2. **Provable Privacy** - Zero patient data in audit logs (test-verified)
3. **Autonomous Replanning** - Agents handle roadblocks with replan triggers
4. **Universal Routing** - 11 modalities with graceful fallback
5. **Voice Auditing** - Radiologist dictation vs specialist findings
6. **Complete Audit Trail** - SHA-256 hash chain with CLI verifier

---

## 📅 TIMELINE

**Current Status:** 65% complete  
**Remaining Work:** 8-10 hours  
**Deadline:** May 19, 2026, 13:00 CET  
**Status:** ✅ On track

---

## 🔧 TROUBLESHOOTING

### API Keys Not Set
All modules have mock implementations. Use `--mock` flag or omit API keys to use mocks.

### DICOM Files Not Loading
Ensure `pydicom` is installed: `pip install pydicom`

### Federated Demo Fails
If CheXpert unavailable, script automatically falls back to non-IID NIH-14 split.

### Audit Chain Broken
Run verifier: `python -m radagent.audit.verify <run_dir>`

---

## 📞 CONTACT

**Author:** Rayane Aggoune  
**Location:** Sétif, Algeria  
**Repository:** https://github.com/Anna-ray/radagent  
**Branch:** feature/v2-milan  
**License:** MIT

---

**Last Updated:** May 13, 2026  
**Version:** v2.0-alpha (65% complete)