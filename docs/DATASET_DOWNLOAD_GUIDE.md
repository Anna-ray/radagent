# RadAgent v2 - Dataset Download Guide

**Author:** Rayane Aggoune  
**Last Updated:** May 13, 2026

This guide provides instructions for downloading the datasets required for RadAgent v2 federated learning experiments.

---

## 📦 Required Datasets

### 1. NIH ChestX-ray14 (Hospital A Node)
**Size:** ~42 GB  
**Images:** 112,120 frontal-view X-rays  
**Classes:** 14 thoracic diseases  
**Status:** ✅ Already available in v1

**Download:**
```bash
# Official NIH download page
# https://nihcc.app.box.com/v/ChestXray-NIHCC

# Or use the NIH Chest X-ray dataset from Kaggle
# https://www.kaggle.com/datasets/nih-chest-xrays/data
```

**Expected Structure:**
```
data/nih14/
├── images/
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ...
├── Data_Entry_2017.csv
└── BBox_List_2017.csv
```

---

### 2. Stanford CheXpert (Hospital B Node)
**Size:** ~11 GB (frontal views only)  
**Images:** 224,316 chest radiographs  
**Classes:** 14 observations (harmonized to NIH-14 schema)  
**Status:** ⏸️ Needs download

#### Download Instructions

**Step 1: Request Access**
1. Visit: https://stanfordmlgroup.github.io/competitions/chexpert/
2. Click "Download Dataset"
3. Fill out the data use agreement form
4. You'll receive a download link via email (usually within 24 hours)

**Step 2: Download Dataset**
```bash
# After receiving the download link, use wget or curl
wget -O CheXpert-v1.0-small.zip "YOUR_DOWNLOAD_LINK_HERE"

# Or download the full dataset (recommended for production)
wget -O CheXpert-v1.0.zip "YOUR_DOWNLOAD_LINK_HERE"
```

**Step 3: Extract Dataset**
```bash
# Extract to data directory
unzip CheXpert-v1.0-small.zip -d data/

# Rename for consistency
mv data/CheXpert-v1.0-small data/chexpert
```

**Expected Structure:**
```
data/chexpert/
├── train/
│   ├── patient00001/
│   │   └── study1/
│   │       └── view1_frontal.jpg
│   └── ...
├── valid/
│   └── ...
├── train.csv
└── valid.csv
```

**Step 4: Filter Frontal Views Only**
```bash
# Run the preprocessing script (included in RadAgent)
python scripts/preprocess_chexpert.py \
  --input data/chexpert \
  --output data/chexpert_frontal \
  --frontal-only
```

---

### 3. Stanford MURA (Optional - Track D)
**Size:** ~6 GB  
**Images:** 40,561 musculoskeletal radiographs  
**Classes:** 7 body parts (binary normal/abnormal)  
**Status:** ⏸️ Optional for Track D

#### Download Instructions

**Step 1: Request Access**
1. Visit: https://stanfordmlgroup.github.io/competitions/mura/
2. Fill out the data use agreement
3. Receive download link via email

**Step 2: Download Dataset**
```bash
wget -O MURA-v1.1.zip "YOUR_DOWNLOAD_LINK_HERE"
unzip MURA-v1.1.zip -d data/
mv data/MURA-v1.1 data/mura
```

**Expected Structure:**
```
data/mura/
├── train/
│   ├── XR_ELBOW/
│   ├── XR_FINGER/
│   ├── XR_FOREARM/
│   ├── XR_HAND/
│   ├── XR_HUMERUS/
│   ├── XR_SHOULDER/
│   └── XR_WRIST/
├── valid/
│   └── ...
├── train_image_paths.csv
└── valid_image_paths.csv
```

---

## 🚀 Quick Start After Download

### Verify Dataset Structure
```bash
# Check NIH-14
python -c "from radagent.data.cxr_datasets import build_nih_loader; \
           loader, _ = build_nih_loader('data/nih14', 100); \
           print(f'NIH-14: {len(loader.dataset)} samples')"

# Check CheXpert
python -c "from radagent.data.cxr_datasets import build_chexpert_loader; \
           loader, _ = build_chexpert_loader('data/chexpert', 100); \
           print(f'CheXpert: {len(loader.dataset)} samples')"
```

### Run Federated Demo
```bash
python scripts/run_federated_demo.py \
  --nih-root data/nih14 \
  --chexpert-root data/chexpert \
  --test-root data/chexpert \
  --rounds 5 \
  --samples-per-node 5000 \
  --audit-dir runs/federated_demo
```

---

## 🔄 Fallback Strategy

If CheXpert is unavailable or download is delayed:

**Option 1: Use Non-IID NIH-14 Split (Automatic)**
```bash
# The script automatically falls back if CheXpert is missing
python scripts/run_federated_demo.py \
  --nih-root data/nih14 \
  --rounds 5 \
  --samples-per-node 5000 \
  --audit-dir runs/federated_demo
```

The script will:
1. Detect missing CheXpert
2. Split NIH-14 into two non-IID partitions
3. Simulate two hospitals with different disease distributions
4. Add a note to the audit log explaining the substitution

**Option 2: Use Public Subsets**
```bash
# Download CheXpert demo subset (smaller, no registration required)
# Note: This is NOT the official dataset, just for testing
wget https://example.com/chexpert_demo_1000.zip
```

---

## 📊 Dataset Statistics

### NIH ChestX-ray14
- **Total Images:** 112,120
- **Patients:** 30,805
- **Image Size:** 1024×1024 (original)
- **Format:** PNG
- **Labels:** Multi-label (0-14 diseases per image)
- **Split:** 70% train, 10% val, 20% test

### CheXpert
- **Total Images:** 224,316
- **Patients:** 65,240
- **Image Size:** Variable (resized to 320×320)
- **Format:** JPG
- **Labels:** Multi-label with uncertainty (-1, 0, 1)
- **Split:** Train/valid provided

### Label Harmonization (NIH-14 ↔ CheXpert)
```
NIH-14 Class          → CheXpert Mapping
─────────────────────────────────────────
Atelectasis           → Atelectasis (direct)
Cardiomegaly          → Cardiomegaly (direct)
Consolidation         → Consolidation (direct)
Edema                 → Edema (direct)
Effusion              → Pleural Effusion
Emphysema             → (not in CheXpert, mask=0)
Fibrosis              → (not in CheXpert, mask=0)
Hernia                → (not in CheXpert, mask=0)
Infiltration          → Lung Opacity
Mass                  → (not in CheXpert, mask=0)
Nodule                → (not in CheXpert, mask=0)
Pleural_Thickening    → (not in CheXpert, mask=0)
Pneumonia             → Pneumonia (direct)
Pneumothorax          → Pneumothorax (direct)
```

**Uncertainty Handling:**
- CheXpert `-1` (uncertain) → `label_mask = 0` (excluded from loss)
- CheXpert `0` (negative) → `label = 0`
- CheXpert `1` (positive) → `label = 1`

---

## ⚠️ Important Notes

### Data Use Agreements
- **NIH ChestX-ray14:** Public domain, no restrictions
- **CheXpert:** Requires signed data use agreement
- **MURA:** Requires signed data use agreement

### Privacy & Ethics
- All datasets are de-identified
- No patient data leaves the hospital in federated learning
- Audit logs contain ZERO raw patient data (test-verified)

### Storage Requirements
- **Minimum:** 50 GB (NIH-14 only with fallback)
- **Recommended:** 100 GB (NIH-14 + CheXpert)
- **Full:** 150 GB (NIH-14 + CheXpert + MURA)

### Processing Time
- **NIH-14 download:** ~2-4 hours (depending on connection)
- **CheXpert download:** ~1-2 hours
- **CheXpert preprocessing:** ~30 minutes
- **MURA download:** ~30 minutes

---

## 🛠️ Troubleshooting

### CheXpert Download Link Expired
Request a new link from Stanford. Links typically expire after 7 days.

### Disk Space Issues
Use the "small" version of CheXpert (11 GB vs 439 GB full version).

### Label Mismatch Errors
Ensure you're using the harmonized loader from `radagent.data.cxr_datasets`.

### Slow Download
Use a download manager like `aria2c` for resumable downloads:
```bash
aria2c -x 16 -s 16 "YOUR_DOWNLOAD_LINK"
```

---

## 📞 Support

**Dataset Issues:**
- NIH-14: https://nihcc.app.box.com/v/ChestXray-NIHCC
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
- MURA: https://stanfordmlgroup.github.io/competitions/mura/

**RadAgent Issues:**
- GitHub: https://github.com/Anna-ray/radagent/issues
- Author: Rayane Aggoune

---

**Last Updated:** May 13, 2026  
**RadAgent Version:** v2.0-alpha