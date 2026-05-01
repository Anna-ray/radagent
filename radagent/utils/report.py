"""
radagent.utils.report
---------------------
Render evaluation results as a clean markdown report ready to drop
into the hackathon submission or paper.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence


def _fmt(v: float, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and (v != v)):  # NaN
        return "  --  "
    return f"{v:.{nd}f}"


def _fmt_ci(lo: float, hi: float, nd: int = 3) -> str:
    if (lo != lo) or (hi != hi):
        return "[--, --]"
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def render_markdown_report(
    metrics: dict,
    classes: Sequence[str],
    title: str = "RadAgent Specialist | NIH-14 Official Test Set",
    extra_meta: dict | None = None,
) -> str:
    pc = metrics["per_class"]
    lines: list[str] = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z_")
    lines.append("")

    # ----- summary block -----
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- **N test cases:** {metrics['n_samples']:,}"
    )
    lines.append(
        f"- **Macro AUC:** {_fmt(metrics['macro_auc'])} "
        f"{_fmt_ci(*metrics['macro_auc_ci'])}"
    )
    lines.append(
        f"- **Micro AUC:** {_fmt(metrics['micro_auc'])} "
        f"{_fmt_ci(*metrics['micro_auc_ci'])}"
    )
    lines.append(f"- **Mean F1 (per-class optimal threshold):** {_fmt(metrics['mean_f1'])}")
    lines.append(f"- **Mean AP:** {_fmt(metrics['mean_ap'])}")
    lines.append(f"- **Bootstrap iterations:** {metrics['n_bootstrap']:,}")
    lines.append(f"- **CI level:** {int(metrics['ci_level']*100)}%")
    if extra_meta:
        for k, v in extra_meta.items():
            lines.append(f"- **{k}:** {v}")
    lines.append("")

    # ----- per-class table -----
    lines.append("## Per-class results")
    lines.append("")
    lines.append("| Finding | AUC | AUC 95% CI | AP | F1 | Sens | Spec | Threshold |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c, name in enumerate(classes):
        auc = pc["auc"][c]
        auc_lo, auc_hi = pc["auc_ci"][c]
        ap = pc["ap"][c]
        f1 = pc["f1"][c]
        sn = pc["sens"][c]
        sp = pc["spec"][c]
        th = pc["thresholds"][c]
        lines.append(
            f"| {name} | {_fmt(auc)} | {_fmt_ci(auc_lo, auc_hi)} "
            f"| {_fmt(ap)} | {_fmt(f1)} | {_fmt(sn)} | {_fmt(sp)} | {_fmt(th)} |"
        )

    macro = metrics["macro_auc"]
    lines.append(
        f"| **MEAN** | **{_fmt(macro)}** | "
        f"**{_fmt_ci(*metrics['macro_auc_ci'])}** | "
        f"**{_fmt(metrics['mean_ap'])}** | "
        f"**{_fmt(metrics['mean_f1'])}** | -- | -- | -- |"
    )
    lines.append("")
    lines.append(
        "_Per-class thresholds are F1-optimal on the validation set. "
        "Sensitivity and specificity are reported at those thresholds._"
    )
    lines.append("")
    return "\n".join(lines)
