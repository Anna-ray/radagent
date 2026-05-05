# RadAgent Specialist | NIH-14 Official Test Set

_Generated: 2026-04-30T08:13:19Z_

## Summary

- **N test cases:** 25,596
- **Macro AUC:** 0.819 [0.815, 0.823]
- **Micro AUC:** 0.863 [0.861, 0.865]
- **Mean F1 (per-class optimal threshold):** 0.356
- **Mean AP:** 0.301
- **Bootstrap iterations:** 1,000
- **CI level:** 95%
- **Backbone:** convnextv2_base.fcmae_ft_in22k_in1k_384
- **Image size:** 384
- **Checkpoint:** runs/nih14_convnextv2_base_384/best.pt
- **TTA:** horizontal flip
- **Temperature:** 0.1979
- **Inference time:** 285.2s (89.7 img/s)

## Per-class results

| Finding | AUC | AUC 95% CI | AP | F1 | Sens | Spec | Threshold |
|---|---|---|---|---|---|---|---|
| Atelectasis | 0.776 | [0.768, 0.784] | 0.347 | 0.384 | 0.683 | 0.725 | 0.426 |
| Cardiomegaly | 0.887 | [0.877, 0.896] | 0.334 | 0.398 | 0.489 | 0.958 | 0.497 |
| Effusion | 0.835 | [0.829, 0.841] | 0.529 | 0.524 | 0.787 | 0.730 | 0.492 |
| Infiltration | 0.711 | [0.703, 0.718] | 0.406 | 0.446 | 0.870 | 0.362 | 0.351 |
| Mass | 0.832 | [0.822, 0.842] | 0.339 | 0.405 | 0.501 | 0.929 | 0.496 |
| Nodule | 0.781 | [0.768, 0.794] | 0.269 | 0.325 | 0.368 | 0.939 | 0.473 |
| Pneumonia | 0.726 | [0.704, 0.746] | 0.057 | 0.096 | 0.142 | 0.960 | 0.426 |
| Pneumothorax | 0.880 | [0.873, 0.886] | 0.456 | 0.513 | 0.674 | 0.889 | 0.444 |
| Consolidation | 0.748 | [0.738, 0.758] | 0.157 | 0.252 | 0.565 | 0.777 | 0.446 |
| Edema | 0.847 | [0.836, 0.858] | 0.180 | 0.216 | 0.662 | 0.833 | 0.552 |
| Emphysema | 0.933 | [0.925, 0.940] | 0.464 | 0.531 | 0.676 | 0.961 | 0.499 |
| Fibrosis | 0.820 | [0.800, 0.840] | 0.107 | 0.178 | 0.186 | 0.984 | 0.495 |
| Pleural_Thickening | 0.778 | [0.765, 0.791] | 0.152 | 0.211 | 0.255 | 0.945 | 0.581 |
| Hernia | 0.918 | [0.884, 0.948] | 0.413 | 0.507 | 0.430 | 0.999 | 0.433 |
| **MEAN** | **0.819** | **[0.815, 0.823]** | **0.301** | **0.356** | -- | -- | -- |

_Per-class thresholds are F1-optimal on the validation set. Sensitivity and specificity are reported at those thresholds._
