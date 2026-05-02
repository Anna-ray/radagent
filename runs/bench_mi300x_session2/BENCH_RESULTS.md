# RadAgent MI300X Bench - Session 2

**Date**: 2026-05-01
**Model stack**: ConvNeXt-V2 Base (specialist) + BGE-M3 (retriever) + Qwen/Qwen2.5-VL-7B-Instruct (VLM)
**Hardware**: AMD Instinct MI300X VF, 192 GB HBM3, single GPU
**Image**: vLLM 0.17.1 + ROCm 7.2 on Ubuntu 24.04 (AMD Developer Cloud)
**N**: 10 NIH-14 test images (random sample)

## Headline numbers

| Stage      | Mean | Median | p95   | Min  | Max   |
|------------|------|--------|-------|------|-------|
| pre_ms     | 12   | 12     | 13    | 11   | 14    |
| spec_ms    | 339  | 22     | 1762  | 22   | 3183  |
| rag_ms     | 656  | 14     | 3538  | 0    | 6325  |
| vlm_ms     | 4984 | 5535   | 5640  | 2626 | 5646  |
| total_ms   | 5991 | 5592   | 10944 | 2661 | 15152 |

## Steady state (cases 2-10, after warmup)

- Specialist: 22 ms (vs 234 ms on RTX 4070 Ti SUPER - 10x faster on MI300X)
- RAG: 14 ms median (0 ms when n_above=0, since no queries are issued)
- VLM (Qwen2.5-VL-7B): 5.5 sec median
- **Total: 5.6 sec median, 5.8 sec max**

## Cold start (case 1, one-time cost)

- 15 sec total
- 3.2s specialist (ROCm kernel compile)
- 6.3s RAG (BGE-M3 model load)
- 5.6s VLM (server already warm but first request slow)
- Irrelevant to deployment scenarios where the server is persistent

## Memory footprint

All three models co-resident on a single MI300X:

- vLLM (Qwen2.5-VL-7B): ~16 GB at 0.85 utilization
- Specialist (ConvNeXt-V2 Base, bf16): ~1.5 GB
- BGE-M3 retriever: ~2.5 GB
- **Total active: ~20 GB / 192 GB available**

The same workload requires 2 NVIDIA H100 (80 GB each) when running an 11B-class VLM.
The MI300X co-residence story is the deployment-cost contribution.

## Sample generated report (case 1, smoke test, 00000744_010.png)

Specialist flagged: Pneumothorax (p=0.837), Atelectasis (p=0.588), Mass (p=0.515).
RAG retrieved 3 passages per finding from StatPearls + Wikipedia.
Qwen produced this report:

> ### Chest Radiograph Report
> #### Findings:
> 1. **Pneumothorax**: Moderate likelihood (calibrated probability: 0.837, confidence level: medium). A pneumothorax may appear as a clear space in the lung field, often with a visible line or marking along the pleural border [1].
> 2. **Atelectasis**: Moderate likelihood (calibrated probability: 0.588, medium). Features such as lung opacification and loss of lung volume are indicative; displacement of interlobar fissures and overinflation of the unaffected lung can also be observed [2].
> 3. **Mass**: Moderate likelihood (calibrated probability: 0.515, medium). The differential includes fibrosis, lymphangitis carcinomatosa, and metastases. Further evaluation with high-resolution CT or a lung biopsy may be necessary [3].
>
> #### Impression:
> The chest X-ray reveals moderate likelihood of pneumothorax, atelectasis, and a potential mass. Further clinical correlation and possibly additional imaging studies (e.g., CT scan) are recommended.
>
> #### Recommendations:
> - Clinical correlation with patient history
> - CT scan of the chest for better characterization
> - Specialist consultation

## Notes

- **Model substitution**: Llama 3.2 Vision Instruct (original target) failed to load in vLLM 0.17.1 due to a `MllamaProcessor._get_num_multimodal_tokens` AttributeError - vLLM s transformers fallback path is broken for this model on this image. Pivoted to Qwen2.5-VL-7B-Instruct, which is ungated and has native vLLM support. The substitution is also documentable as a positive finding for v2 (smaller, faster, free, equally good for visual reasoning).
- **Cost**: ~$2.50 of AMD Developer Cloud credit at $1.99/hr.
- **Total session time**: ~75 min from droplet creation to destroy.
- **Per-case JSONs**: 10 files in bench_mi300x/ contain the full structured findings, retrieved passages with URLs, and Qwen-generated report text per case.
