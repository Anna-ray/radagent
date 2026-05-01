"""
scripts/train.py
----------------
Usage:
    python -m scripts.train --config configs/nih14_convnextv2_base.yaml

Wires together: config → datasets → model → loss → optimizer → trainer.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from radagent.data.dataset import (
    NIHChestXray14,
    build_eval_transforms,
    build_train_transforms,
    load_nih14_dataframe,
    make_weighted_sampler,
    patient_disjoint_split,
)
from radagent.losses.asl import AsymmetricLoss
from radagent.models.specialist import SpecialistCXR
from radagent.training.trainer import Trainer
from radagent.utils.training_utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint (best.pt or last.pt) to resume from")
    return p.parse_args()
    return p.parse_args()


def build_loaders(cfg: dict):
    d = cfg["data"]
    train_val_df, _test_df = load_nih14_dataframe(
        labels_csv=d["labels_csv"],
        train_split_txt=d["train_split_txt"],
        test_split_txt=d["test_split_txt"],
    )
    train_df, val_df = patient_disjoint_split(
        train_val_df,
        val_fraction=d["val_fraction"],
        seed=cfg["experiment"]["seed"],
    )
    print(f"[data] train={len(train_df)}  val={len(val_df)}")

    train_tfms = build_train_transforms(
        image_size=d["image_size"],
        affine_deg=cfg["augment"]["random_affine_degrees"],
        affine_trans=cfg["augment"]["random_affine_translate"],
        elastic_alpha=cfg["augment"]["elastic_alpha"],
        elastic_sigma=cfg["augment"]["elastic_sigma"],
        rrc_scale=tuple(cfg["augment"]["random_resized_crop_scale"]),
        hflip_prob=cfg["augment"]["hflip_prob"],
    )
    eval_tfms = build_eval_transforms(image_size=d["image_size"])

    train_ds = NIHChestXray14(
        labels_df=train_df,
        images_dir=d["images_dir"],
        classes=d["classes"],
        image_size=d["image_size"],
        is_train=True,
        clahe_clip_jitter=tuple(cfg["augment"]["clahe_clip_limit_jitter"]),
        train_transforms=train_tfms,
        eval_transforms=eval_tfms,
    )
    val_ds = NIHChestXray14(
        labels_df=val_df,
        images_dir=d["images_dir"],
        classes=d["classes"],
        image_size=d["image_size"],
        is_train=False,
        train_transforms=train_tfms,
        eval_transforms=eval_tfms,
    )

    if cfg["train"]["use_weighted_sampler"]:
        sampler = make_weighted_sampler(train_ds.label_matrix)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        shuffle=shuffle,
        num_workers=d["num_workers"],
        pin_memory=d["pin_memory"],
        persistent_workers=d["persistent_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=d["num_workers"],
        pin_memory=d["pin_memory"],
        persistent_workers=d["persistent_workers"],
    )
    return train_loader, val_loader


def build_model_and_loss(cfg: dict):
    classes = cfg["data"]["classes"]
    model = SpecialistCXR(
        timm_name=cfg["model"]["name"],
        num_classes=len(classes),
        pretrained=cfg["model"]["pretrained"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
        grad_checkpointing=cfg["model"]["grad_checkpointing"],
    )
    loss_type = cfg["loss"]["type"]
    if loss_type == "asl":
        loss_fn = AsymmetricLoss(
            gamma_neg=cfg["loss"]["asl_gamma_neg"],
            gamma_pos=cfg["loss"]["asl_gamma_pos"],
            clip=cfg["loss"]["asl_clip"],
        )
    elif loss_type == "bce_pos_weight":
        # Fallback only — kept for ablations
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(loss_type)
    return model, loss_fn


def build_optimizer(cfg: dict, model: SpecialistCXR):
    groups = model.parameter_groups(
        lr_backbone=cfg["optim"]["lr_backbone"],
        lr_head=cfg["optim"]["lr_head"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    return torch.optim.AdamW(groups, betas=tuple(cfg["optim"]["betas"]))


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — RadAgent requires a GPU.")
    device = torch.device("cuda")
    print(f"[device] {torch.cuda.get_device_name(0)}")
    print(f"[vram]   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    out_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(cfg)
    model, loss_fn = build_model_and_loss(cfg)
    optimizer = build_optimizer(cfg, model)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        classes=cfg["data"]["classes"],
        cfg=cfg,
        device=device,
        output_dir=str(out_dir),
    )

    start_epoch = 1
    if args.resume:
        start_epoch = trainer.resume_from(args.resume)

    trainer.fit(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
