#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Disease-specific survival prognosis from multi-view Cine CMR.

MAIN-CODE-UPDATED PROGNOSIS VERSION:
    - image input organization and encoder definitions are aligned with the latest diagnosis main code
    - diagnosis encoder checkpoint is loaded from the main full-view hier_mean experiment
    - survival stage keeps disease-specific hier_attn / hybrid-crossattention aggregation

Independent HCM/DCM hybrid cross-attention survival variant:
    - shared slice/view attention pooling
    - disease-specific cross-attention residuals
    - configurable HCM/DCM loss weights
    - configurable HCM/DCM residual gates
    - configurable HCM/DCM Cox head capacity/dropout
    - DCM-weighted early stopping


Prognosis pipeline:
    1) Load a trained multi-view Video Swin classifier / encoder.
    2) Either extract frozen patient-level features, or extract clip-level features and fine-tune
       only the slice/view aggregation layers (agg_only).
    3) Train disease-specific Cox survival heads:
           HCM patients -> HCM Cox head
           DCM patients -> DCM Cox head
           NC patients  -> not used in prognosis loss
    4) Evaluate C-index, Kaplan-Meier risk stratification and log-rank p value.

Input image data structure:
    root/
      NC/patient_x/Cine4CH-15_1_xxx.nii ...
      HCM/patient_y/CineSAX-6_25_xxx.nii ...
      DCM/patient_z/...

NIfTI filename format:
    Cine2CH-6_1_cine_tf2d14_retro_iPAT.nii
    view      = Cine2CH
    slice_idx = 6
    frame_idx = 1

Survival CSV required columns by default:
    patient_id, disease_type, class_label, event, time_to_event, censored, event_type, prog_available

Notes:
    - No manual ROI delineation is required. A frozen automated ventricular
      segmentation model derives explicit SAX cine functional descriptors for
      the final DCM model.
    - event_type is not used in the first Cox loss, but is saved/statistically summarized.
    - time_to_event is assumed to be in DAYS in this script.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# =============================================================================
# 运行配置区
# =============================================================================
# CONFIGURATION RULE: RUN_CONFIG below is the only editable configuration source.
# No later FINAL_* block may override checkpoint, split, GPU, data, or model parameters.

RUN_CONFIG = {
    # ---------------- GPU ----------------
    "gpu_mode": "single",                    # "single" or "ddp"; DDP only accelerates feature extraction
    "visible_gpus": "0,1,2,3",      # DCM-only run uses all four physical GPUs
    "auto_launch_ddp": False,
    "ddp_master_port": 0,                # 0 = automatically choose a free port; avoids EADDRINUSE 29500

    # ---------------- Data ----------------
    "data_path": "/data/datasets/CMR/Project-FM",
    "selected_views": "Cine2CH,Cine3CH,Cine4CH,CineSAX",
    "cohort_required_views": "Cine2CH,Cine3CH,Cine4CH,CineSAX",  # align with main diagnosis complete-case cohort
    "allow_missing_selected_views": False,
    "slice_plan": "Cine2CH:3,Cine3CH:3,Cine4CH:3,CineSAX:8",
    "num_frames": 13,
    "image_size": 224,
    "min_frames_per_slice": 1,
    "use_cache": True,
    "cache_dir": "",

    # Survival table. Put your processed HCM+DCM CSV path here.
    "prognosis_csv": "/data/projects/MRI/datasets/survivaldata/cmr_survival_final.csv",
    "prognosis_id_col": "patient_id",
    "prognosis_disease_col": "disease_type",
    "prognosis_class_col": "class_label",
    "prognosis_event_col": "event",
    "prognosis_time_col": "time_to_event",
    "prognosis_censor_col": "censored",
    "prognosis_event_type_col": "event_type",
    "prognosis_available_col": "prog_available",
    "patient_id_zfill": 0,                 # normally 0; set e.g. 10 only if your image folder IDs are zero-padded to 10 digits

    # Use previous classification split if you want exactly the same patient split.
    # It should contain split_train.csv, split_val.csv, split_test.csv or train_split.csv etc.
    "split_source_dir": "/data/projects/MRI_New/20260725_backbone_baseline_with_r3d18_results_20260725_233138/exp_videoswin_kinetics_hiermean_full",
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "seed": 42,

    # ---------------- Encoder / checkpoint ----------------
    # Recommended: point this to your exp6 four-view classification best_model.pt
    "classification_ckpt": "/data/projects/MRI_New/20260725_backbone_baseline_with_r3d18_results_20260725_233138/exp_videoswin_kinetics_hiermean_full/best_model.pt",
    # "classification_ckpt":"",
    # "classification_ckpt": '/data/projects/MRI/20260705_04_main_diagnosis_full_hiermean_results_main_full_hiermean_seed42/main_diagnosis_full_hiermean/best_model.pt',

    # If classification_ckpt is empty, the backbone can still load raw Video Swin weights below.
    "weights_path": "/data/projects/MRI/checkpoints/swin3d_t-7615ae03.pth",
    "pretrained": True,
    "diagnosis_encoder_agg": "hier_mean",  # main diagnosis encoder aggregation used for clean checkpoint loading
    "survival_agg": "hier_attn",       # prognosis-specific aggregation; keep attention for survival risk modeling
    "agg": "hier_attn",                # backward-compatible alias for survival_agg
    "dropout": 0.20,
    "backbone_chunk_size": 0,
    "amp": True,

    # ---------------- Feature extraction ----------------
    "batch_size": 1,                        # per-GPU if DDP
    "workers": 2,
    "force_reextract_features": False,     # reuse frozen multilevel features after the first extraction

    # ---------------- Survival fine-tuning mode ----------------
    # frozen_feature: use fixed patient-level features, only train HCM/DCM Cox heads.
    # agg_only: cache clip-level Video Swin features, then fine-tune slice/view attention pooling + Cox heads.
    # cross_attention: cache clip-level features, then train disease-specific HCM/DCM query cross-attention + Cox heads.
    # hybrid_cross_attention: agg-only shared patient feature + disease-specific cross-attention residual + Cox heads.
    #                  These modes do NOT update Video Swin backbone.
    "fine_tune_mode": "hybrid_cross_attention",    # "frozen_feature" / "agg_only" / "cross_attention" / "hybrid_cross_attention"
    "feature_lr": 3e-5,                      # learning rate for aggregation / cross-attention layers
    "agg_finetune_dropout": 0.05,            # dropout inside slice/view attention pooling during agg_only training
    "cross_attn_heads": 4,                   # disease-specific cross-attention heads
    "cross_attn_layers": 1,                  # keep 1 first; more layers may overfit survival data
    "cross_attn_dropout": 0.10,              # dropout inside cross-attention block
    # Hybrid residual gates: fused = shared_agg_feat + gate * disease_cross_feat
    # DCM can be initialized with a smaller gate to preserve the more stable shared agg-only feature.
    "hybrid_cross_gate_init": 0.50,          # fallback if disease-specific gate init is not provided
    "hcm_cross_gate_init": 0.20,
    "dcm_cross_gate_init": 0.10,

    # ---------------- Survival head training ----------------
    "survival_epochs": 100,
    "survival_lr": 1e-4,
    "survival_weight_decay": 1e-3,
    "survival_hidden_dim": 128,
    "survival_dropout": 0.30,
    # Optional disease-specific head capacity; 0 means use survival_hidden_dim / survival_dropout.
    "hcm_hidden_dim": 128,
    "dcm_hidden_dim": 128,
    "hcm_dropout": 0.20,
    "dcm_dropout": 0.3,
    "survival_head_type": "mlp",           # "mlp" or "linear"
    # DCM-priority Cox training and model selection.
    "disease_mode": "DCM_only",     # "both_independent" / "joint" / "DCM_only" / "HCM_only"
    # The following values are overridden automatically for HCM_only and DCM_only child runs.
    "hcm_loss_weight": 0.0,
    "dcm_loss_weight": 1.0,
    "early_stop_metric": "dcm",
    "hcm_monitor_weight": 0.0,
    "dcm_monitor_weight": 1.0,
    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,
    "risk_tie_eps": 1e-8,
    # Optional L2 penalty on risk magnitude; usually keep 0.
    "risk_l2": 0.0,

    # ---------------- Evaluation / plots ----------------
    "risk_group_source": "train_median",   # train_median / split_median
    "eval_horizons_days": "365,730,1095,1460,1825", # fixed horizons in days: 1y/2y/3y/4y/5y by default
    # 默认主方案：Cox-only 训练，然后直接用 Cox risk score 评估 1/2/3/4/5 年事件预测 AUC。
    # 如果要额外训练固定时间窗 BCE head，再把 train_horizon_heads=True 且 horizon_loss_weight>0。
    "horizon_loss_weight": 0.0,            # 0 = 不训练 horizon BCE head，只用 Cox risk score 评估时间窗事件
    "train_horizon_heads": False,          # True = 训练 HCM/DCM 固定时间窗事件 head
    "use_horizon_pos_weight": True,        # 训练 BCE head 时，对正样本加权以处理事件不平衡
    "horizon_pos_weight_max": 20.0,        # 防止极端正样本权重过大
    "horizon_threshold_mode": "youden",   # 用验证集选测试阈值：youden / f1 / median
    "horizon_threshold": 0.5,              # 仅当没有可用验证阈值时作为兜底
    "plot_km": True,
    "plot_history": True,

    # ---------------- Multi-seed standalone mode ----------------
    # True: this single file will run multiple seeds sequentially and summarize results.
    # Child runs are launched from this same file, so no external runner/old script is required.
    "multiseed_mode": False,
    "multiseed_seeds": "42,123,2024,3407,777",
    "multiseed_output_root": "auto",
    "multiseed_continue_on_error": False,

    # ---------------- Independent disease runner ----------------
    # If disease_mode="both_independent", this single file launches itself twice:
    #   1) HCM_only hybrid model
    #   2) DCM_only hybrid model
    # No external old scripts are required. Each child run still uses DDP for feature extraction if gpu_mode="ddp".
    "both_independent_modes": "DCM_only",
    "both_output_root": "auto",
    "both_continue_on_error": False,

    # ---------------- IO ----------------
    "output_dir": "auto",
    "experiment_name": "dcm_nested_oof_fixed_k_seed42",
    "run_id": "",
    "resume_head": "",
    "eval_only": False,
    "print_freq": 20,

    # ---------------- Active diagnosis multilevel survival experiment ----------------
    "multilevel_variants": "diagnosis_patient_only,diagnosis_multilevel_residual",
    "multilevel_seeds": "42,2024,3407",
    "multilevel_epochs": 500,
    "multilevel_patience": 80,
    "multilevel_eval_every": 5,
    "multilevel_lr": 3e-4,
    "multilevel_weight_decay": 1e-3,
    "multilevel_dropout": 0.20,
    "multilevel_head_hidden": 64,
    "multilevel_adapter_dim": 64,
    "multilevel_attention_hidden": 64,
    "multilevel_attention_heads": 4,
    "multilevel_residual_gate_init": 0.10,
    "multilevel_residual_gate_max": 0.50,
    "multilevel_clip_gate_init": 0.10,
    "multilevel_clip_gate_max": 0.50,
    "multilevel_view_dropout": 0.10,
    "multilevel_rank_weight": 0.05,
    "multilevel_rank_tau": 0.10,
    "multilevel_rank_max_pairs": 30000,
    "multilevel_residual_l2": 1e-4,
    "multilevel_grad_clip": 5.0,

    # ---------------- Diagnosis-to-prognosis fine-tuning ----------------
    # The diagnosis network/checkpoint stays unchanged. A private encoder copy
    # is initialized from it and adapted with DCM survival supervision only.
    # Screening default: run one candidate with one seed only. The completed
    # direct/frozen baselines are reused from the previous locked run.
    "prognosis_variants": "diagnosis_finetune",
    "prognosis_seeds": "42",
    "prognosis_epochs": 60,
    "prognosis_patience": 12,
    "prognosis_eval_every": 2,
    "prognosis_head_warmup_epochs": 4,
    "prognosis_pooling_warmup_epochs": 4,
    "prognosis_unfreeze_last_stages": 2,
    "prognosis_backbone_lr": 1e-6,
    "prognosis_aggregation_lr": 1e-5,
    "prognosis_head_lr": 1e-4,
    "prognosis_weight_decay": 1e-3,
    "prognosis_head_hidden": 128,
    "prognosis_dropout": 0.30,
    "prognosis_pairs_per_epoch": 256,
    "prognosis_pair_batch_size": 2,
    "prognosis_eval_batch_size": 2,
    "prognosis_gradient_accumulation": 4,
    "prognosis_grad_clip": 5.0,

    # ---------------- Full-risk-set single-seed screening ----------------
    "fullrisk_variants": "K_oof_late_fusion",
    "fusion_bottleneck_dim": 64,
    "fusion_reuse_feature_cache": True,
    "multilevel_token_dim": 768,
    "evaluate_test_after_selection": True,  # final test: evaluate validation-selected model only
    "residual_fusion_dropout": 0.50,
    "residual_fusion_max_gate": 0.50,
    "residual_fusion_weight_decay": 0.01,
    "risk_fusion_dropout": 0.10,
    "risk_fusion_max_gate": 0.50,
    "risk_fusion_weight_decay": 0.05,
    "risk_fusion_lr": 0.01,
    "motion_flow_phases": 16,
    "motion_flow_size": 48,
    "motion_fusion_max_gate": 0.30,
    "cardiac_segmentation_model": "/data/projects/MRI_New/pretrained_models/ventricular_short_axis_3label/models/model.pt",
    "segmentation_inference_batch_size": 128,  # 32 SAX frames per 4090
    "segmentation_cox_dropout": 0.10,
    "segmentation_cox_lr": 0.01,
    "segmentation_cox_weight_decay": 0.01,
    "late_fusion_folds": 5,
    "late_fusion_min_phenotype_weight": 0.20,
    "late_fusion_max_phenotype_weight": 0.80,
    "late_fusion_weight_steps": 25,
    "fullrisk_extract_batch_size": 2,
    "fullrisk_head_hidden": 64,
    "fullrisk_head_dropout": 0.20,
    "fullrisk_head_lr": 3e-4,
    "fullrisk_weight_decay": 1e-3,
    "fullrisk_head_epochs": 300,
    "fullrisk_patience": 40,
    "fullrisk_eval_every": 5,
    "fullrisk_adapt_epochs": 3,
    "fullrisk_adapt_batch_size": 2,
    "fullrisk_unfreeze_last_stages": 1,
    "fullrisk_encoder_lr": 1e-6,
    "fullrisk_grad_clip": 5.0,

    # ---------------- Legacy general head settings (inactive in final DCM K run) ----------------
    "final_feature_mode": 'topdev3_proto6',
    "final_experts": '2ch,3ch,4ch,sax,lax,full',
    "final_pca_dim": 64,
    "final_bottleneck_dim": 16,
    "final_topdev_m": 3,
    "final_proto_k": 6,
    "final_lambda": 0.45,
    "final_seed_aggregate": 'median',
    "final_seeds": '42,2024,3407,777,2026,123,456,789,1024,2048',
    "final_head_device": 'auto',
    "final_head_epochs": 500,
    "final_head_patience": 80,
    "final_head_eval_every": 5,
    "final_head_lr": 0.0003,
    "final_head_weight_decay": 0.001,
    "final_head_dropout": 0.1,
    "final_grad_clip": 5.0,
    "horizons_days": '365,730,1095,1460,1825',
    "expected_train_n": 783,
    "expected_train_events": 181,
    "expected_val_n": 168,
    "expected_val_events": 39,
    "expected_test_n": 168,
    "expected_test_events": 42,
    # Fixed expected counts are historical audit references only. Raw/diagnosis
    # patient-level alignment remains strict regardless of this switch.
    "strict_expected_split_counts": False,
}


def _apply_env_run_config_overrides():
    raw = os.environ.get("CMR_RUN_CONFIG_OVERRIDES", "").strip()
    if not raw:
        return
    try:
        overrides = json.loads(raw)
        if isinstance(overrides, dict):
            for protected_key in ["split_source_dir", "classification_ckpt"]:
                if protected_key in overrides:
                    print(f"[WARN] Ignoring environment override for protected key: {protected_key}")
                    overrides.pop(protected_key, None)
            RUN_CONFIG.update(overrides)
    except Exception as e:
        print(f"[WARN] CMR_RUN_CONFIG_OVERRIDES parse failed: {e}")

_apply_env_run_config_overrides()

# Must be set before importing torch.
if RUN_CONFIG.get("visible_gpus", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(RUN_CONFIG["visible_gpus"])

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.models.video as video_models

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# =============================================================================
# Basic utils
# =============================================================================
CLASS_TO_IDX = {"NC": 0, "HCM": 1, "DCM": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
ALL_VIEWS = ["Cine2CH", "Cine3CH", "Cine4CH", "CineSAX"]
DEFAULT_SLICE_PLAN = {"Cine2CH": 3, "Cine3CH": 3, "Cine4CH": 3, "CineSAX": 8}


FUSION_AGG_CHOICES = [
    "hier_attn",
    "hier_mean",
    "flat_mil",
    "flat_mean",
    "concat_mlp",
    "late_fusion",
    "view_transformer",
]


def canonicalize_agg(agg: str) -> str:
    """Use the same aggregation naming convention as the latest diagnosis main code.

    For prognosis, only hier_attn and hier_mean are instantiated by the survival
    models. The other names are accepted here so configuration/reporting stays
    consistent with the diagnosis code, but survival aggregation will reject
    unsupported modes explicitly.
    """
    a = str(agg or "hier_attn").lower().strip()
    aliases = {
        "attention": "hier_attn",
        "attn": "hier_attn",
        "hier_attention": "hier_attn",
        "hierarchical_attention": "hier_attn",
        "mean": "hier_mean",
        "hierarchical_mean": "hier_mean",
        "mil": "flat_mil",
        "flat_attention": "flat_mil",
        "flat_attn": "flat_mil",
        "late": "late_fusion",
        "transformer": "view_transformer",
        "view_token_transformer": "view_transformer",
    }
    a = aliases.get(a, a)
    if a not in FUSION_AGG_CHOICES:
        raise ValueError(f"Unknown agg={agg!r}. Supported: {FUSION_AGG_CHOICES}; aliases attention/mean are also accepted.")
    return a


def make_hierarchical_poolers(dim: int, agg: str, dropout: float = 0.05):
    agg = canonicalize_agg(agg)
    if agg == "hier_attn":
        return MaskedAttentionPool(dim, dropout=dropout), MaskedAttentionPool(dim, dropout=dropout)
    if agg == "hier_mean":
        return MaskedMeanPool(), MaskedMeanPool()
    raise ValueError(f"Survival feature aggregation supports only hier_attn/hier_mean, got {agg!r}")


def safe_torch_load(path, map_location="cpu", weights_only=True):
    """Avoid torch.load FutureWarning when possible, while staying compatible with old PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        if weights_only:
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)
        raise


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed() -> Tuple[torch.device, int, int, bool]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))
        device = torch.device(f"cuda:{local_rank}")
        return device, dist.get_rank(), dist.get_world_size(), True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, 0, 1, False


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_csv_list(s: str) -> List[str]:
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def validate_views(views: Sequence[str]):
    for v in views:
        if v not in ALL_VIEWS:
            raise ValueError(f"Unknown view: {v}. Supported: {ALL_VIEWS}")


def parse_slice_plan(s: str) -> Dict[str, int]:
    if s is None or str(s).strip() == "":
        return dict(DEFAULT_SLICE_PLAN)
    plan = dict(DEFAULT_SLICE_PLAN)
    for item in str(s).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"slice_plan item must be View:K, got {item}")
        view, val = item.split(":", 1)
        view = view.strip()
        if view not in ALL_VIEWS:
            raise ValueError(f"Unknown view in slice_plan: {view}")
        plan[view] = int(val)
    return plan


def ensure_dir(p: Path):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_shared_run_id(user_run_id: str = "") -> str:
    user_run_id = str(user_run_id or "").strip()
    if user_run_id:
        return user_run_id
    env_run_id = os.environ.get("CMR_RUN_ID", "").strip()
    if env_run_id:
        return env_run_id
    if is_dist_avail_and_initialized():
        obj = [now_string() if get_rank() == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]
    return now_string()


def build_output_dir(args, selected_views: List[str]) -> Path:
    run_id = get_shared_run_id(getattr(args, "run_id", ""))
    script_stem = Path(sys.argv[0]).stem
    output_dir = str(getattr(args, "output_dir", "auto") or "auto").strip()
    if output_dir.lower() in {"", "auto", "none"}:
        root = Path(f"./{script_stem}_results_{run_id}")
    else:
        root = Path(output_dir)
    exp_name = str(getattr(args, "experiment_name", "") or "").strip()
    if exp_name.lower() in {"", "auto", "none"}:
        view_tag = "_".join([v.replace("Cine", "") for v in selected_views])
        exp_name = f"survival_{view_tag}"
    return root / exp_name


def make_id_candidates(pid, zfill: int = 0) -> List[str]:
    """
    Generate robust patient-id candidates for matching imaging folders to prognosis CSV.

    Handles cases such as:
        imaging folder: 0020271138_20181016
        prognosis id:   0020271138

    The matching strategy deliberately keeps both zero-padded and unpadded numeric IDs,
    because many hospital IDs contain leading zeros.
    """
    raw = str(pid).strip()
    cands = [raw]

    # Common separators in patient folder names: ID_date, ID-date, DCM_ID_date, etc.
    for tok in re.split(r"[_\-\s/\\]+", raw):
        tok = str(tok).strip()
        if tok:
            cands.append(tok)

    # Extract digit runs. This catches strings like DCM0020271138_20181016.
    # Keep long runs first because patient IDs are usually longer than dates.
    digit_runs = re.findall(r"\d+", raw)
    digit_runs = sorted(digit_runs, key=len, reverse=True)
    cands.extend(digit_runs)

    # Numeric variants with/without leading zeros.
    expanded = []
    for c in cands:
        c = str(c).strip()
        if not c:
            continue
        expanded.append(c)
        if zfill and c.isdigit():
            expanded.append(c.zfill(int(zfill)))
        if c.isdigit():
            try:
                expanded.append(str(int(c)))  # remove leading zeros
            except Exception:
                pass

    # preserve order unique
    out = []
    for c in expanded:
        if c not in out:
            out.append(c)
    return out


def resolve_prognosis_id(image_patient_id: str, prognosis_table: Dict[str, object], prognosis_table_unpad: Dict[str, object], zfill: int = 0) -> Tuple[Optional[str], Optional[object]]:
    """Return (matched_id, row) for one image patient_id, or (None, None)."""
    raw = str(image_patient_id).strip()

    # 1) Candidate matching: exact / split token / digit run / zfill / unpad
    for cand in make_id_candidates(raw, zfill):
        if cand in prognosis_table:
            return cand, prognosis_table[cand]
        if cand in prognosis_table_unpad:
            row = prognosis_table_unpad[cand]
            return str(row.get("patient_id", cand)), row

    # 2) Prefix / substring fallback. Sort IDs by length to avoid short accidental matches.
    for qid, row in sorted(prognosis_table.items(), key=lambda kv: len(str(kv[0])), reverse=True):
        qid = str(qid).strip()
        if not qid:
            continue
        if raw.startswith(qid) or qid in raw:
            return qid, row

    return None, None


def config_hash_for_features(args, selected_views: List[str], slice_plan: Dict[str, int]) -> str:
    obj = {
        "data_path": args.data_path,
        "selected_views": selected_views,
        "cohort_required_views": getattr(args, "cohort_required_views", ""),
        "allow_missing_selected_views": getattr(args, "allow_missing_selected_views", False),
        "slice_plan": {k: slice_plan.get(k, None) for k in ALL_VIEWS},
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "classification_ckpt": getattr(args, "classification_ckpt", ""),
        "weights_path": getattr(args, "weights_path", ""),
        "diagnosis_encoder_agg": getattr(args, "diagnosis_encoder_agg", getattr(args, "agg", "hier_attn")),
        "survival_agg": getattr(args, "survival_agg", getattr(args, "agg", "hier_attn")),
        "agg": getattr(args, "agg", "hier_attn"),
        "fine_tune_mode": getattr(args, "fine_tune_mode", "frozen_feature"),
        "prognosis_csv": getattr(args, "prognosis_csv", ""),
    }
    return hashlib.md5(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# =============================================================================
# Dataset
# =============================================================================
class MultiViewCardiacNiftiDataset(Dataset):
    filename_pattern = re.compile(r"^(Cine(?:2CH|3CH|4CH|SAX))-(\d+)_(\d+)_.*\.nii(?:\.gz)?$")

    def __init__(
        self,
        root_dir: str,
        selected_views: Sequence[str],
        slice_plan: Dict[str, int],
        num_frames: int = 13,
        image_size: int = 224,
        samples: Optional[List[Dict]] = None,
        mode: str = "eval",
        required_views: Optional[Sequence[str]] = None,
        allow_missing_selected_views: bool = False,
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
        min_frames_per_slice: int = 1,
        verbose: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.selected_views = list(selected_views)
        validate_views(self.selected_views)
        self.slice_plan = dict(slice_plan)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.mode = mode
        self.allow_missing_selected_views = bool(allow_missing_selected_views)
        self.required_views = list(required_views) if required_views else []
        validate_views(self.required_views)
        self.use_cache = bool(use_cache)
        self.cache_dir = Path(cache_dir) if cache_dir else self.root_dir / ".cmr_clip_cache"
        self.min_frames_per_slice = int(min_frames_per_slice)
        self.verbose = verbose

        self.nclips = sum(int(self.slice_plan.get(v, 0)) for v in self.selected_views)
        self.clip_view_ids_template = self._build_clip_view_ids_template()
        self.view_clip_ranges = self._build_view_clip_ranges()
        if self.use_cache:
            ensure_dir(self.cache_dir)

        self.samples = samples if samples is not None else self._collect_samples()

    def _build_clip_view_ids_template(self) -> torch.Tensor:
        ids = []
        for local_vid, v in enumerate(self.selected_views):
            ids.extend([local_vid] * int(self.slice_plan.get(v, 0)))
        return torch.tensor(ids, dtype=torch.long)

    def _build_view_clip_ranges(self) -> Dict[str, Tuple[int, int]]:
        ranges = {}
        start = 0
        for v in self.selected_views:
            k = int(self.slice_plan.get(v, 0))
            ranges[v] = (start, start + k)
            start += k
        return ranges

    def _parse_filename(self, path: Path):
        m = self.filename_pattern.match(path.name)
        if not m:
            return None, None, None
        return m.group(1), int(m.group(2)), int(m.group(3))

    def _organize_patient_files(self, patient_dir: Path) -> Dict[str, Dict[int, List[Dict]]]:
        organized: Dict[str, Dict[int, List[Dict]]] = {}
        files = list(patient_dir.glob("*.nii")) + list(patient_dir.glob("*.nii.gz"))
        for fp in files:
            view, sid, fid = self._parse_filename(fp)
            if view is None:
                continue
            organized.setdefault(view, {}).setdefault(sid, []).append({"frame_idx": fid, "file_path": str(fp)})
        for view in organized:
            for sid in organized[view]:
                organized[view][sid].sort(key=lambda x: x["frame_idx"])
        return organized

    def _patient_has_view(self, pdata: Dict, view: str) -> bool:
        if view not in pdata or not pdata[view]:
            return False
        return any(len(frames) >= self.min_frames_per_slice for frames in pdata[view].values())

    def _is_valid_patient(self, pdata: Dict) -> bool:
        if self.required_views:
            required = self.required_views
        elif self.allow_missing_selected_views:
            required = []
        else:
            required = self.selected_views
        for v in required:
            if not self._patient_has_view(pdata, v):
                return False
        if self.allow_missing_selected_views and not required:
            return any(self._patient_has_view(pdata, v) for v in self.selected_views)
        return True

    def _collect_samples(self) -> List[Dict]:
        samples = []
        if self.verbose:
            print0("开始扫描 NIfTI 数据...")
            print0(f"  selected_views = {self.selected_views}")
            if self.required_views:
                print0(f"  cohort_required_views = {self.required_views}")
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            class_dir = self.root_dir / cls_name
            if not class_dir.exists():
                print0(f"警告: 类别目录不存在: {class_dir}")
                continue
            patient_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
            valid_count = 0
            for pdir in patient_dirs:
                pdata = self._organize_patient_files(pdir)
                if self._is_valid_patient(pdata):
                    available = [v for v in ALL_VIEWS if self._patient_has_view(pdata, v)]
                    samples.append({
                        "patient_id": pdir.name,
                        "patient_dir": str(pdir),
                        "label": cls_idx,
                        "class_name": cls_name,
                        "organized_data": pdata,
                        "available_views": available,
                    })
                    valid_count += 1
            print0(f"{cls_name}: {valid_count}/{len(patient_dirs)} 有效")
        print0(f"数据扫描完成: 总有效病人 {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def normalize_image(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        p1, p99 = np.percentile(img, (1, 99))
        if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
            mn, mx = float(np.min(img)), float(np.max(img))
            if mx - mn < 1e-8:
                return np.zeros_like(img, dtype=np.float32)
            return ((img - mn) / (mx - mn + 1e-8)).astype(np.float32)
        img = np.clip(img, p1, p99)
        return ((img - p1) / (p99 - p1 + 1e-8)).astype(np.float32)

    def resize_with_pad(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        if h == self.image_size and w == self.image_size:
            return x
        scale = min(self.image_size / max(h, 1), self.image_size / max(w, 1))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        x = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        pad_h, pad_w = self.image_size - new_h, self.image_size - new_w
        pad_top, pad_left = pad_h // 2, pad_w // 2
        return F.pad(x, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0.0)

    def _load_frame(self, file_path: str) -> torch.Tensor:
        try:
            nimg = nib.load(file_path)
            arr = nimg.get_fdata(dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            elif arr.ndim > 3:
                arr = np.squeeze(arr)
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
            if arr.ndim != 2:
                raise ValueError(f"unexpected ndim={arr.ndim}")
            h, w = arr.shape
            if w > h:
                arr = arr.T
            arr = self.normalize_image(arr)
            x = torch.from_numpy(arr).unsqueeze(0)
            x = self.resize_with_pad(x)
            x = (x - 0.485) / 0.229
            return x.float()
        except Exception:
            return torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)

    def _select_frame_infos(self, frames_info: Sequence[Dict]) -> List[Dict]:
        frames_info = sorted(list(frames_info), key=lambda x: x["frame_idx"])
        n = len(frames_info)
        if n <= 0:
            return []
        idx = np.linspace(0, n - 1, self.num_frames, dtype=int).tolist()
        return [frames_info[i] for i in idx]

    def _cache_key_for_clip(self, frames_info: Sequence[Dict]) -> str:
        raw = "|".join([f'{x["frame_idx"]}:{x["file_path"]}' for x in frames_info])
        raw += f"|T={self.num_frames}|S={self.image_size}|norm=v2"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _load_one_slice_clip(self, frames_info: Sequence[Dict]) -> torch.Tensor:
        selected = self._select_frame_infos(frames_info)
        if not selected:
            return torch.zeros(1, self.num_frames, self.image_size, self.image_size, dtype=torch.float32)
        if self.use_cache:
            key = self._cache_key_for_clip(selected)
            cache_path = self.cache_dir / f"{key}.pt"
            if cache_path.exists():
                try:
                    return safe_torch_load(cache_path, map_location="cpu", weights_only=True)
                except Exception:
                    pass
        frames = [self._load_frame(x["file_path"]) for x in selected]
        clip = torch.stack(frames, dim=1).contiguous()
        if self.use_cache:
            key = self._cache_key_for_clip(selected)
            cache_path = self.cache_dir / f"{key}.pt"
            tmp_path = self.cache_dir / f"{key}.{os.getpid()}.{random.randint(0, 10**9)}.tmp"
            try:
                torch.save(clip, tmp_path)
                os.replace(tmp_path, cache_path)
            except Exception:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
        return clip

    def _select_slice_ids(self, view_slices: Dict[int, List[Dict]], k: int) -> List[int]:
        valid_slice_ids = [sid for sid in sorted(view_slices.keys()) if len(view_slices[sid]) >= self.min_frames_per_slice]
        if len(valid_slice_ids) == 0 or k <= 0:
            return []
        if len(valid_slice_ids) >= k:
            pos = np.linspace(0, len(valid_slice_ids) - 1, k, dtype=int).tolist()
            return [valid_slice_ids[i] for i in pos]
        return valid_slice_ids

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        pdata = sample["organized_data"]
        videos, clip_mask, view_mask = [], [], []
        for v in self.selected_views:
            k = int(self.slice_plan.get(v, 0))
            view_slices = pdata.get(v, {})
            chosen_slice_ids = self._select_slice_ids(view_slices, k)
            real_count = 0
            for sid in chosen_slice_ids:
                videos.append(self._load_one_slice_clip(view_slices[sid]))
                clip_mask.append(1)
                real_count += 1
            for _ in range(k - len(chosen_slice_ids)):
                videos.append(torch.zeros(1, self.num_frames, self.image_size, self.image_size, dtype=torch.float32))
                clip_mask.append(0)
            view_mask.append(1 if real_count > 0 else 0)
        if len(videos) != self.nclips:
            raise RuntimeError(f"Internal error: got {len(videos)} clips, expected {self.nclips}")
        return {
            "videos": torch.stack(videos, dim=0).float(),
            "clip_mask": torch.tensor(clip_mask, dtype=torch.bool),
            "clip_view_ids": self.clip_view_ids_template.clone(),
            "view_mask": torch.tensor(view_mask, dtype=torch.bool),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "patient_id": sample["patient_id"],
            "class_name": sample["class_name"],
            "available_views": sample.get("available_views", []),
            "disease_type": sample.get("disease_type", sample["class_name"]),
            "event": torch.tensor(float(sample.get("event", -1)), dtype=torch.float32),
            "time_to_event": torch.tensor(float(sample.get("time_to_event", -1)), dtype=torch.float32),
            "censored": torch.tensor(float(sample.get("censored", -1)), dtype=torch.float32),
            "prog_available": torch.tensor(float(sample.get("prog_available", 0)), dtype=torch.float32),
            "event_type": sample.get("event_type", ""),
        }


# =============================================================================
# Model: feature extractor and survival heads
# =============================================================================
class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 2, 128)
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        has_any = mask.any(dim=1, keepdim=True).float()
        return pooled * has_any, weights * has_any


class MaskedMeanPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        w = mask.float()
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (x * w.unsqueeze(-1)).sum(dim=1) / denom
        weights = w / denom
        return pooled * mask.any(dim=1, keepdim=True).float(), weights


class MultiViewVideoSwinFeatureExtractor(nn.Module):
    def __init__(self, selected_views, slice_plan, weights_path=None, pretrained=False, agg="attention", dropout=0.35, backbone_chunk_size=0):
        super().__init__()
        self.selected_views = list(selected_views)
        self.slice_plan = dict(slice_plan)
        self.num_views = len(self.selected_views)
        self.nclips = sum(int(self.slice_plan.get(v, 0)) for v in self.selected_views)
        self.backbone_chunk_size = int(backbone_chunk_size)

        self.backbone = video_models.swin3d_t(weights=None)
        self.feature_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        if pretrained:
            self.load_raw_swin_weights(weights_path)

        self.view_embeddings = nn.Parameter(torch.zeros(self.num_views, self.feature_dim))
        nn.init.trunc_normal_(self.view_embeddings, std=0.02)
        self.agg = canonicalize_agg(agg)
        self.slice_pool, self.view_pool = make_hierarchical_poolers(self.feature_dim, self.agg, dropout=0.05)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )
        self.view_clip_ranges = self._build_view_clip_ranges()
        self.last_attention = None
        print0(f"✅ Feature extractor initialized: views={self.selected_views}, nclips={self.nclips}, dim={self.feature_dim}")

    def _build_view_clip_ranges(self) -> Dict[int, Tuple[int, int]]:
        ranges = {}
        start = 0
        for local_vid, v in enumerate(self.selected_views):
            k = int(self.slice_plan.get(v, 0))
            ranges[local_vid] = (start, start + k)
            start += k
        return ranges

    def load_raw_swin_weights(self, weights_path: Optional[str]):
        if not weights_path or not Path(weights_path).exists():
            print0(f"⚠️ raw Video Swin weights not found: {weights_path}")
            return
        print0(f"🚀 Loading raw Video Swin weights: {weights_path}")
        ckpt = safe_torch_load(str(weights_path), map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        else:
            state = ckpt
        clean = {}
        for k, v in state.items():
            kk = k
            if kk.startswith("module."):
                kk = kk[len("module."):]
            if kk.startswith("backbone."):
                kk = kk[len("backbone."):]
            if kk.startswith("head.") or ".head." in kk:
                continue
            clean[kk] = v
        missing, unexpected = self.backbone.load_state_dict(clean, strict=False)
        print0(f"🎉 raw weights loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    def load_classification_checkpoint(self, ckpt_path: str):
        if not ckpt_path:
            return
        p = Path(ckpt_path)
        if not p.exists():
            print0(f"⚠️ classification_ckpt not found: {p}")
            return
        print0(f"🚀 Loading classification checkpoint: {p}")
        ckpt = safe_torch_load(str(p), map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        else:
            state = ckpt
        clean = {}
        for k, v in state.items():
            kk = k
            if kk.startswith("module."):
                kk = kk[len("module."):]
            clean[kk] = v
        msg = self.load_state_dict(clean, strict=False)
        print0(f"🎉 classification checkpoint loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        if msg.missing_keys:
            print0(f"  missing examples: {msg.missing_keys[:8]}")
        if msg.unexpected_keys:
            print0(f"  unexpected examples: {msg.unexpected_keys[:8]}")

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_chunk_size and self.backbone_chunk_size > 0 and x.shape[0] > self.backbone_chunk_size:
            outs = []
            for s in range(0, x.shape[0], self.backbone_chunk_size):
                outs.append(self.backbone(x[s:s + self.backbone_chunk_size]))
            return torch.cat(outs, dim=0)
        return self.backbone(x)

    def extract_clip_features(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract frozen Video Swin clip-level features before slice/view aggregation.

        Used by fine_tune_mode=agg_only: the expensive Video Swin backbone is cached,
        then only slice/view aggregation layers and disease-specific Cox heads are trained.
        """
        videos = batch["videos"]
        clip_mask = batch["clip_mask"]
        clip_view_ids = batch["clip_view_ids"]
        B, N, C, T, H, W = videos.shape
        flat_mask = clip_mask.reshape(-1).bool()
        flat_videos = videos.reshape(B * N, C, T, H, W)
        feats_flat = torch.zeros(B * N, self.feature_dim, device=videos.device, dtype=torch.float32)
        if flat_mask.any():
            x_valid = flat_videos[flat_mask].expand(-1, 3, -1, -1, -1).contiguous()
            feat_valid = self._run_backbone(x_valid).float()
            feats_flat[flat_mask] = feat_valid
        return {
            "clip_features": feats_flat.reshape(B, N, self.feature_dim),
            "clip_mask": clip_mask.bool(),
            "clip_view_ids": clip_view_ids.long(),
        }

    def forward_features(self, batch: Dict) -> torch.Tensor:
        videos = batch["videos"]
        clip_mask = batch["clip_mask"]
        clip_view_ids = batch["clip_view_ids"]
        B, N, C, T, H, W = videos.shape
        flat_mask = clip_mask.reshape(-1).bool()
        flat_view_ids = clip_view_ids.reshape(-1).long()
        flat_videos = videos.reshape(B * N, C, T, H, W)
        feats_flat = torch.zeros(B * N, self.feature_dim, device=videos.device, dtype=torch.float32)
        if flat_mask.any():
            x_valid = flat_videos[flat_mask].expand(-1, 3, -1, -1, -1).contiguous()
            feat_valid = self._run_backbone(x_valid).float()
            feat_valid = feat_valid + self.view_embeddings[flat_view_ids[flat_mask]]
            feats_flat[flat_mask] = feat_valid
        feats = feats_flat.reshape(B, N, self.feature_dim)
        view_feats, view_valids, slice_weights = [], [], []
        for local_vid in range(self.num_views):
            start, end = self.view_clip_ranges[local_vid]
            vf, sw = self.slice_pool(feats[:, start:end, :], clip_mask[:, start:end])
            view_feats.append(vf)
            view_valids.append(clip_mask[:, start:end].any(dim=1))
            slice_weights.append(sw.detach().cpu())
        view_feats = torch.stack(view_feats, dim=1)
        view_mask = torch.stack(view_valids, dim=1)
        patient_feat, view_weights = self.view_pool(view_feats, view_mask)
        self.last_attention = {
            "view_weights": view_weights.detach().cpu(),
            "slice_weights": slice_weights,
            "selected_views": list(self.selected_views),
        }
        return patient_feat

    def forward(self, batch: Dict):
        return self.forward_features(batch)


class DiseaseSpecificCoxHeads(nn.Module):
    """Disease-specific Cox heads plus optional fixed-horizon event heads.

    Supports separate HCM/DCM hidden dimensions and dropout so DCM can be made
    slightly stronger without changing the HCM branch.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.3,
                 head_type: str = "mlp", num_horizons: int = 0,
                 hcm_hidden_dim: Optional[int] = None, dcm_hidden_dim: Optional[int] = None,
                 hcm_dropout: Optional[float] = None, dcm_dropout: Optional[float] = None):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_horizons = int(num_horizons)
        hcm_hidden_dim = int(hcm_hidden_dim or hidden_dim)
        dcm_hidden_dim = int(dcm_hidden_dim or hidden_dim)
        hcm_dropout = float(dropout if hcm_dropout is None else hcm_dropout)
        dcm_dropout = float(dropout if dcm_dropout is None else dcm_dropout)

        def make_head(out_dim: int, hd: int, do: float):
            if head_type == "linear":
                return nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, out_dim))
            return nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hd),
                nn.GELU(),
                nn.Dropout(do),
                nn.Linear(hd, out_dim),
            )

        self.hcm_head = make_head(1, hcm_hidden_dim, hcm_dropout)
        self.dcm_head = make_head(1, dcm_hidden_dim, dcm_dropout)
        if self.num_horizons > 0:
            self.hcm_event_head = make_head(self.num_horizons, hcm_hidden_dim, hcm_dropout)
            self.dcm_event_head = make_head(self.num_horizons, dcm_hidden_dim, dcm_dropout)
        else:
            self.hcm_event_head = None
            self.dcm_event_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {
            "hcm_risk": self.hcm_head(x).squeeze(-1),
            "dcm_risk": self.dcm_head(x).squeeze(-1),
        }
        if self.num_horizons > 0:
            out["hcm_event_logits"] = self.hcm_event_head(x)
            out["dcm_event_logits"] = self.dcm_event_head(x)
        return out

class ClipAggregationSurvivalModel(nn.Module):
    """Trainable slice/view aggregation on cached Video Swin clip features."""
    def __init__(self, selected_views, slice_plan, feature_dim: int, agg: str = "hier_attn",
                 agg_dropout: float = 0.05, survival_hidden_dim: int = 128,
                 survival_dropout: float = 0.30, survival_head_type: str = "mlp", num_horizons: int = 0,
                 hcm_hidden_dim: Optional[int] = None, dcm_hidden_dim: Optional[int] = None,
                 hcm_dropout: Optional[float] = None, dcm_dropout: Optional[float] = None):
        super().__init__()
        self.selected_views = list(selected_views)
        self.slice_plan = dict(slice_plan)
        self.num_views = len(self.selected_views)
        self.nclips = sum(int(self.slice_plan.get(v, 0)) for v in self.selected_views)
        self.feature_dim = int(feature_dim)
        self.view_embeddings = nn.Parameter(torch.zeros(self.num_views, self.feature_dim))
        nn.init.trunc_normal_(self.view_embeddings, std=0.02)
        self.agg = canonicalize_agg(agg)
        self.slice_pool, self.view_pool = make_hierarchical_poolers(self.feature_dim, self.agg, dropout=agg_dropout)
        self.survival_heads = DiseaseSpecificCoxHeads(
            feature_dim=self.feature_dim,
            hidden_dim=survival_hidden_dim,
            dropout=survival_dropout,
            head_type=survival_head_type,
            num_horizons=num_horizons,
            hcm_hidden_dim=hcm_hidden_dim,
            dcm_hidden_dim=dcm_hidden_dim,
            hcm_dropout=hcm_dropout,
            dcm_dropout=dcm_dropout,
        )
        self.view_clip_ranges = self._build_view_clip_ranges()

    def _build_view_clip_ranges(self) -> Dict[int, Tuple[int, int]]:
        ranges = {}
        start = 0
        for local_vid, v in enumerate(self.selected_views):
            k = int(self.slice_plan.get(v, 0))
            ranges[local_vid] = (start, start + k)
            start += k
        return ranges

    def load_aggregation_from_classification_checkpoint(self, ckpt_path: str):
        if not ckpt_path or not Path(ckpt_path).exists():
            print0(f"agg_only: classification_ckpt not found, aggregation starts from random init: {ckpt_path}")
            return
        ckpt = safe_torch_load(str(ckpt_path), map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        clean = {}
        for k, v in state.items():
            kk = k[len("module."):] if k.startswith("module.") else k
            if kk.startswith("view_embeddings") or kk.startswith("slice_pool.") or kk.startswith("view_pool."):
                clean[kk] = v
        # Shape-safe loading: only load keys that exist with exactly matching shapes.
        own = self.state_dict()
        clean = {k: v for k, v in clean.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
        msg = self.load_state_dict(clean, strict=False)
        print0(f"agg_only aggregation init loaded: keys={len(clean)}, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}, survival_agg={getattr(self, 'agg', 'cross_attn')}")

    def aggregate_features(self, clip_features: torch.Tensor, clip_mask: torch.Tensor, clip_view_ids: torch.Tensor) -> torch.Tensor:
        clip_mask = clip_mask.bool()
        clip_view_ids = clip_view_ids.long()
        feats = clip_features.float() + self.view_embeddings[clip_view_ids]
        view_feats, view_valids = [], []
        for local_vid in range(self.num_views):
            start, end = self.view_clip_ranges[local_vid]
            vf, _ = self.slice_pool(feats[:, start:end, :], clip_mask[:, start:end])
            view_feats.append(vf)
            view_valids.append(clip_mask[:, start:end].any(dim=1))
        view_feats = torch.stack(view_feats, dim=1)
        view_mask = torch.stack(view_valids, dim=1)
        patient_feat, _ = self.view_pool(view_feats, view_mask)
        return patient_feat

    def forward(self, clip_features: torch.Tensor, clip_mask: torch.Tensor, clip_view_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        patient_feat = self.aggregate_features(clip_features, clip_mask, clip_view_ids)
        return self.survival_heads(patient_feat)

    def aggregation_parameters(self):
        params = [self.view_embeddings]
        params += list(self.slice_pool.parameters())
        params += list(self.view_pool.parameters())
        return params


class CrossAttentionBlock(nn.Module):
    """One lightweight query-to-clip cross-attention block.

    Disease-specific query tokens read cached Video Swin clip features.
    This keeps Video Swin frozen and only learns how HCM/DCM should fuse multi-view cine tokens.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.10, ffn_ratio: float = 2.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = ~mask.bool()
        q = self.q_norm(queries)
        kv = self.kv_norm(tokens)
        attn_out, attn_weights = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        queries = queries + self.drop(attn_out)
        queries = queries + self.ffn(queries)
        return queries, attn_weights


class DiseaseSpecificCrossAttentionSurvivalModel(nn.Module):
    """HCM/DCM disease-specific cross-attention on cached clip features.

    The model outputs two different patient representations:
    HCM query -> HCM-specific feature -> HCM Cox head
    DCM query -> DCM-specific feature -> DCM Cox head
    """
    def __init__(self, selected_views, slice_plan, feature_dim: int,
                 num_heads: int = 4, num_layers: int = 1, cross_dropout: float = 0.10,
                 survival_hidden_dim: int = 128, survival_dropout: float = 0.20,
                 survival_head_type: str = "mlp", num_horizons: int = 0,
                 hcm_hidden_dim: Optional[int] = None, dcm_hidden_dim: Optional[int] = None,
                 hcm_dropout: Optional[float] = None, dcm_dropout: Optional[float] = None):
        super().__init__()
        self.selected_views = list(selected_views)
        self.slice_plan = dict(slice_plan)
        self.num_views = len(self.selected_views)
        self.nclips = sum(int(self.slice_plan.get(v, 0)) for v in self.selected_views)
        self.feature_dim = int(feature_dim)
        self.view_embeddings = nn.Parameter(torch.zeros(self.num_views, self.feature_dim))
        self.clip_pos_embeddings = nn.Parameter(torch.zeros(self.nclips, self.feature_dim))
        self.disease_queries = nn.Parameter(torch.zeros(2, self.feature_dim))
        nn.init.trunc_normal_(self.view_embeddings, std=0.02)
        nn.init.trunc_normal_(self.clip_pos_embeddings, std=0.02)
        nn.init.trunc_normal_(self.disease_queries, std=0.02)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(self.feature_dim, num_heads=num_heads, dropout=cross_dropout)
            for _ in range(int(num_layers))
        ])
        self.out_norm = nn.LayerNorm(self.feature_dim)
        self.survival_heads = DiseaseSpecificCoxHeads(
            feature_dim=self.feature_dim,
            hidden_dim=survival_hidden_dim,
            dropout=survival_dropout,
            head_type=survival_head_type,
            num_horizons=num_horizons,
            hcm_hidden_dim=hcm_hidden_dim,
            dcm_hidden_dim=dcm_hidden_dim,
            hcm_dropout=hcm_dropout,
            dcm_dropout=dcm_dropout,
        )
        self.last_attention = None

    def load_aggregation_from_classification_checkpoint(self, ckpt_path: str):
        if not ckpt_path or not Path(ckpt_path).exists():
            print0(f"cross_attention: classification_ckpt not found, starts from random init: {ckpt_path}")
            return
        ckpt = safe_torch_load(str(ckpt_path), map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        clean = {}
        for k, v in state.items():
            kk = k[len("module."):] if k.startswith("module.") else k
            if kk == "view_embeddings" and tuple(v.shape) == tuple(self.view_embeddings.shape):
                clean[kk] = v
        own = self.state_dict()
        clean = {k: v for k, v in clean.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
        msg = self.load_state_dict(clean, strict=False)
        print0(f"cross_attention init loaded: keys={len(clean)}, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    def encode_disease_features(self, clip_features: torch.Tensor, clip_mask: torch.Tensor, clip_view_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = clip_features.shape
        clip_mask = clip_mask.bool()
        clip_view_ids = clip_view_ids.long()
        pos = self.clip_pos_embeddings[:N].unsqueeze(0)
        tokens = clip_features.float() + self.view_embeddings[clip_view_ids] + pos
        tokens = self.input_norm(tokens)
        queries = self.disease_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        last_attn = None
        for blk in self.blocks:
            queries, last_attn = blk(queries, tokens, clip_mask)
        queries = self.out_norm(queries)
        self.last_attention = {
            "attn_weights": last_attn.detach().cpu() if last_attn is not None else None,
            "query_names": ["HCM", "DCM"],
            "selected_views": list(self.selected_views),
        }
        return queries[:, 0, :], queries[:, 1, :]

    def forward(self, clip_features: torch.Tensor, clip_mask: torch.Tensor, clip_view_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        hcm_feat, dcm_feat = self.encode_disease_features(clip_features, clip_mask, clip_view_ids)
        out = {
            "hcm_risk": self.survival_heads.hcm_head(hcm_feat).squeeze(-1),
            "dcm_risk": self.survival_heads.dcm_head(dcm_feat).squeeze(-1),
        }
        if self.survival_heads.num_horizons > 0:
            out["hcm_event_logits"] = self.survival_heads.hcm_event_head(hcm_feat)
            out["dcm_event_logits"] = self.survival_heads.dcm_event_head(dcm_feat)
        return out

    def aggregation_parameters(self):
        params = [self.view_embeddings, self.clip_pos_embeddings, self.disease_queries]
        params += list(self.input_norm.parameters())
        params += list(self.blocks.parameters())
        params += list(self.out_norm.parameters())
        return params



class HybridCrossAttentionSurvivalModel(nn.Module):
    """Hybrid disease-specific survival model on cached clip features.

    Shared agg-only attention pooling gives a stable patient feature, while
    HCM/DCM learnable queries add disease-specific cross-attention residuals.
    """
    def __init__(self, selected_views, slice_plan, feature_dim: int,
                 agg: str = "attention", agg_dropout: float = 0.05,
                 num_heads: int = 4, num_layers: int = 1, cross_dropout: float = 0.10,
                 hybrid_gate_init: float = 0.50,
                 hcm_gate_init: Optional[float] = None, dcm_gate_init: Optional[float] = None,
                 survival_hidden_dim: int = 128, survival_dropout: float = 0.20,
                 survival_head_type: str = "mlp", num_horizons: int = 0,
                 hcm_hidden_dim: Optional[int] = None, dcm_hidden_dim: Optional[int] = None,
                 hcm_dropout: Optional[float] = None, dcm_dropout: Optional[float] = None):
        super().__init__()
        self.selected_views = list(selected_views)
        self.slice_plan = dict(slice_plan)
        self.num_views = len(self.selected_views)
        self.nclips = sum(int(self.slice_plan.get(v, 0)) for v in self.selected_views)
        self.feature_dim = int(feature_dim)

        self.view_embeddings = nn.Parameter(torch.zeros(self.num_views, self.feature_dim))
        nn.init.trunc_normal_(self.view_embeddings, std=0.02)
        self.agg = canonicalize_agg(agg)
        self.slice_pool, self.view_pool = make_hierarchical_poolers(self.feature_dim, self.agg, dropout=agg_dropout)

        self.clip_pos_embeddings = nn.Parameter(torch.zeros(self.nclips, self.feature_dim))
        self.disease_queries = nn.Parameter(torch.zeros(2, self.feature_dim))
        nn.init.trunc_normal_(self.clip_pos_embeddings, std=0.02)
        nn.init.trunc_normal_(self.disease_queries, std=0.02)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(self.feature_dim, num_heads=num_heads, dropout=cross_dropout)
            for _ in range(int(num_layers))
        ])
        self.cross_out_norm = nn.LayerNorm(self.feature_dim)

        def _gate_logit(x):
            x = float(x)
            x = min(max(x, 1e-4), 1.0 - 1e-4)
            return math.log(x / (1.0 - x))

        hcm_gate_init = hybrid_gate_init if hcm_gate_init is None else hcm_gate_init
        dcm_gate_init = hybrid_gate_init if dcm_gate_init is None else dcm_gate_init
        self.gate_logits = nn.Parameter(torch.tensor(
            [[float(_gate_logit(hcm_gate_init))], [float(_gate_logit(dcm_gate_init))]],
            dtype=torch.float32,
        ))
        self.fuse_norm_hcm = nn.LayerNorm(self.feature_dim)
        self.fuse_norm_dcm = nn.LayerNorm(self.feature_dim)

        self.survival_heads = DiseaseSpecificCoxHeads(
            feature_dim=self.feature_dim,
            hidden_dim=survival_hidden_dim,
            dropout=survival_dropout,
            head_type=survival_head_type,
            num_horizons=num_horizons,
            hcm_hidden_dim=hcm_hidden_dim,
            dcm_hidden_dim=dcm_hidden_dim,
            hcm_dropout=hcm_dropout,
            dcm_dropout=dcm_dropout,
        )
        self.view_clip_ranges = self._build_view_clip_ranges()
        self.last_attention = None

    def _build_view_clip_ranges(self) -> Dict[int, Tuple[int, int]]:
        ranges = {}
        start = 0
        for local_vid, v in enumerate(self.selected_views):
            k = int(self.slice_plan.get(v, 0))
            ranges[local_vid] = (start, start + k)
            start += k
        return ranges

    def load_aggregation_from_classification_checkpoint(self, ckpt_path: str):
        if not ckpt_path or not Path(ckpt_path).exists():
            print0(f"hybrid_cross_attention: classification_ckpt not found, starts from random init: {ckpt_path}")
            return
        ckpt = safe_torch_load(str(ckpt_path), map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        clean = {}
        for k, v in state.items():
            kk = k[len("module."):] if k.startswith("module.") else k
            if kk.startswith("view_embeddings") or kk.startswith("slice_pool.") or kk.startswith("view_pool."):
                clean[kk] = v
        # Survival-specific attention can be intentionally different from the diagnosis head.
        # Load only compatible diagnosis aggregation parameters, usually view_embeddings.
        own = self.state_dict()
        clean = {k: v for k, v in clean.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
        msg = self.load_state_dict(clean, strict=False)
        print0(f"hybrid_cross_attention init loaded: keys={len(clean)}, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}, survival_agg={getattr(self, 'agg', 'unknown')}")

    def aggregate_shared_features(self, clip_features: torch.Tensor, clip_mask: torch.Tensor, clip_view_ids: torch.Tensor):
        clip_mask = clip_mask.bool()
        clip_view_ids = clip_view_ids.long()
        feats = clip_features.float() + self.view_embeddings[clip_view_ids]
        view_feats, view_valids, slice_weights = [], [], []
        for local_vid in range(self.num_views):
            start, end = self.view_clip_ranges[local_vid]
            vf, sw = self.slice_pool(feats[:, start:end, :], clip_mask[:, start:end])
            view_feats.append(vf)
            view_valids.append(clip_mask[:, start:end].any(dim=1))
            slice_weights.append(sw.detach().cpu())
        view_feats = torch.stack(view_feats, dim=1)
        view_mask = torch.stack(view_valids, dim=1)
        patient_feat, view_weights = self.view_pool(view_feats, view_mask)
        return patient_feat, view_weights, slice_weights

    def encode_cross_features(self, clip_features: torch.Tensor, clip_mask: torch.Tensor, clip_view_ids: torch.Tensor):
        B, N, D = clip_features.shape
        clip_mask = clip_mask.bool()
        clip_view_ids = clip_view_ids.long()
        pos = self.clip_pos_embeddings[:N].unsqueeze(0)
        tokens = clip_features.float() + self.view_embeddings[clip_view_ids] + pos
        tokens = self.input_norm(tokens)
        queries = self.disease_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        last_attn = None
        for blk in self.blocks:
            queries, last_attn = blk(queries, tokens, clip_mask)
        queries = self.cross_out_norm(queries)
        return queries[:, 0, :], queries[:, 1, :], last_attn

    def forward(self, clip_features: torch.Tensor, clip_mask: torch.Tensor, clip_view_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_feat, view_weights, slice_weights = self.aggregate_shared_features(clip_features, clip_mask, clip_view_ids)
        hcm_cross, dcm_cross, cross_attn = self.encode_cross_features(clip_features, clip_mask, clip_view_ids)
        gates = torch.sigmoid(self.gate_logits)
        hcm_feat = self.fuse_norm_hcm(shared_feat + gates[0].view(1, 1) * hcm_cross)
        dcm_feat = self.fuse_norm_dcm(shared_feat + gates[1].view(1, 1) * dcm_cross)
        self.last_attention = {
            "view_weights": view_weights.detach().cpu(),
            "slice_weights": slice_weights,
            "cross_attn_weights": cross_attn.detach().cpu() if cross_attn is not None else None,
            "query_names": ["HCM", "DCM"],
            "selected_views": list(self.selected_views),
            "hybrid_gates": gates.detach().cpu().view(-1).tolist(),
        }
        out = {
            "hcm_risk": self.survival_heads.hcm_head(hcm_feat).squeeze(-1),
            "dcm_risk": self.survival_heads.dcm_head(dcm_feat).squeeze(-1),
        }
        if self.survival_heads.num_horizons > 0:
            out["hcm_event_logits"] = self.survival_heads.hcm_event_head(hcm_feat)
            out["dcm_event_logits"] = self.survival_heads.dcm_event_head(dcm_feat)
        return out

    def aggregation_parameters(self):
        params = [self.view_embeddings, self.clip_pos_embeddings, self.disease_queries, self.gate_logits]
        params += list(self.slice_pool.parameters())
        params += list(self.view_pool.parameters())
        params += list(self.input_norm.parameters())
        params += list(self.blocks.parameters())
        params += list(self.cross_out_norm.parameters())
        params += list(self.fuse_norm_hcm.parameters())
        params += list(self.fuse_norm_dcm.parameters())
        return params


# =============================================================================
# Prognosis table / split
# =============================================================================
def read_prognosis_table(args) -> pd.DataFrame:
    p = Path(str(args.prognosis_csv))
    if not p.exists():
        raise FileNotFoundError(f"prognosis_csv not found: {p}")
    df = pd.read_csv(p, dtype={args.prognosis_id_col: str})
    df[args.prognosis_id_col] = df[args.prognosis_id_col].astype(str).str.strip()
    # Normalize key columns into standard names.
    rename = {
        args.prognosis_id_col: "patient_id",
        args.prognosis_disease_col: "disease_type",
        args.prognosis_class_col: "class_label",
        args.prognosis_event_col: "event",
        args.prognosis_time_col: "time_to_event",
        args.prognosis_censor_col: "censored",
        args.prognosis_event_type_col: "event_type",
        args.prognosis_available_col: "prog_available",
    }
    for old, new in list(rename.items()):
        if old not in df.columns:
            if new in ["event_type"]:
                df[new] = ""
            elif new in ["prog_available"]:
                df[new] = 1
            elif new in ["censored"]:
                df[new] = 1 - pd.to_numeric(df.get("event", 0), errors="coerce").fillna(0)
            else:
                raise ValueError(f"Missing required prognosis column: {old}")
        elif old != new:
            df = df.rename(columns={old: new})

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    if int(getattr(args, "patient_id_zfill", 0) or 0) > 0:
        z = int(args.patient_id_zfill)
        df["patient_id"] = df["patient_id"].apply(lambda x: x.zfill(z) if str(x).isdigit() else str(x))
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df["time_to_event"] = pd.to_numeric(df["time_to_event"], errors="coerce")
    df["censored"] = pd.to_numeric(df["censored"], errors="coerce").fillna(1 - df["event"].fillna(0))
    df["prog_available"] = pd.to_numeric(df["prog_available"], errors="coerce").fillna(1)
    df["disease_type"] = df["disease_type"].astype(str).str.upper().str.strip()
    df["event_type"] = df["event_type"].fillna("").astype(str)
    return df


def attach_prognosis_to_samples(samples: List[Dict], prog_df: pd.DataFrame, args) -> List[Dict]:
    # Build lookup tables. patient_id must remain string because many IDs have leading zeros.
    prog_df = prog_df.copy()
    prog_df["patient_id"] = prog_df["patient_id"].astype(str).str.strip()
    table = {str(r["patient_id"]).strip(): r for _, r in prog_df.iterrows()}

    # Also allow unpadded numeric lookup, e.g. 0020271138 <-> 20271138.
    table_unpad = {}
    for k, r in table.items():
        if str(k).isdigit():
            try:
                table_unpad[str(int(k))] = r
            except Exception:
                pass

    matched, usable = 0, 0
    unmatched_examples = []
    matched_examples = []
    out = []
    zfill = int(getattr(args, "patient_id_zfill", 0) or 0)

    for s in samples:
        image_pid = str(s["patient_id"]).strip()
        matched_pid, row = resolve_prognosis_id(image_pid, table, table_unpad, zfill=zfill)

        if row is None:
            if len(unmatched_examples) < 20:
                unmatched_examples.append(image_pid)
            continue

        ns = dict(s)
        ns["matched_prognosis_id"] = str(matched_pid)
        ns["disease_type"] = str(row.get("disease_type", s["class_name"])).upper().strip()
        ns["event"] = float(row.get("event", np.nan)) if pd.notna(row.get("event", np.nan)) else np.nan
        ns["time_to_event"] = float(row.get("time_to_event", np.nan)) if pd.notna(row.get("time_to_event", np.nan)) else np.nan
        ns["censored"] = float(row.get("censored", np.nan)) if pd.notna(row.get("censored", np.nan)) else np.nan
        ns["prog_available"] = float(row.get("prog_available", 1)) if pd.notna(row.get("prog_available", 1)) else 0
        ns["event_type"] = str(row.get("event_type", ""))

        matched += 1
        if len(matched_examples) < 10:
            matched_examples.append(f"{image_pid} -> {matched_pid}")

        # Only HCM/DCM with usable survival labels enter prognosis training.
        if ns["class_name"] in {"HCM", "DCM"} and ns["disease_type"] in {"HCM", "DCM"} and ns["prog_available"] == 1:
            if np.isfinite(ns["event"]) and np.isfinite(ns["time_to_event"]) and ns["time_to_event"] > 0:
                if int(ns["event"]) in {0, 1}:
                    out.append(ns)
                    usable += 1

    print0(f"✅ prognosis matched={matched}/{len(samples)}, usable HCM/DCM={usable}")
    if matched_examples:
        print0("匹配示例:")
        for x in matched_examples:
            print0(f"  {x}")
    if unmatched_examples:
        print0("未匹配影像 patient_id 示例（前20个）:")
        for x in unmatched_examples:
            print0(f"  {x}")
    return out

def stratified_split_survival_samples(samples: List[Dict], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_key = defaultdict(list)
    for s in samples:
        key = (s["class_name"], int(s["event"]))
        by_key[key].append(s)
    train, val, test = [], [], []
    for key, items in sorted(by_key.items(), key=lambda x: str(x[0])):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def _find_split_file(split_dir: Path, split: str) -> Optional[Path]:
    candidates = [
        split_dir / f"split_{split}.csv",
        split_dir / f"{split}_split.csv",
        split_dir / f"{split}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def split_from_existing_csv(samples: List[Dict], split_source_dir: str):
    split_dir = Path(split_source_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"split_source_dir does not exist: {split_dir}")
    sample_map = {}
    for s in samples:
        sample_map[(s["class_name"], str(s["patient_id"]))] = s
        sample_map[(s["class_name"], str(s["patient_id"]).lstrip("0"))] = s
    outs = []
    for split in ["train", "val", "test"]:
        fp = _find_split_file(split_dir, split)
        if fp is None:
            raise FileNotFoundError(f"cannot find {split} split csv under {split_dir}")
        df = pd.read_csv(fp, dtype={"patient_id": str})
        rows = []
        for _, r in df.iterrows():
            pid = str(r["patient_id"]).strip()
            cls = str(r.get("class_name", "")).strip()
            s = sample_map.get((cls, pid)) or sample_map.get((cls, pid.lstrip("0")))
            if s is not None:
                rows.append(s)
        outs.append(rows)
        print0(f"Loaded {split} from {fp.name}: {len(rows)} prognosis samples")
    return tuple(outs)


def save_survival_split_csv(samples: List[Dict], path: Path):
    rows = []
    for s in samples:
        rows.append({
            "patient_id": s["patient_id"],
            "matched_prognosis_id": s.get("matched_prognosis_id", ""),
            "class_name": s["class_name"],
            "disease_type": s.get("disease_type", s["class_name"]),
            "event": int(s.get("event", -1)),
            "time_to_event": float(s.get("time_to_event", -1)),
            "censored": float(s.get("censored", -1)),
            "event_type": s.get("event_type", ""),
            "available_views": ";".join(s.get("available_views", [])),
            "patient_dir": s.get("patient_dir", ""),
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def summarize_survival_samples(samples: List[Dict]) -> Dict:
    d = {}
    for disease in ["HCM", "DCM"]:
        ss = [s for s in samples if s["class_name"] == disease]
        if not ss:
            d[disease] = {"n": 0, "events": 0, "censored": 0}
        else:
            ev = sum(int(s["event"]) == 1 for s in ss)
            d[disease] = {"n": len(ss), "events": int(ev), "censored": int(len(ss) - ev)}
    return d


# =============================================================================
# Feature extraction
# =============================================================================
def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device, args) -> List[Dict]:
    model.eval()
    rows = []
    start = time.time()
    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        fine_mode = str(getattr(args, "fine_tune_mode", "frozen_feature")).lower()
        with torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
            if fine_mode in {"agg_only", "cross_attention", "cross_attn", "hybrid_cross_attention", "hybrid_cross_attn", "hybrid"}:
                clip_out = model.extract_clip_features(batch)
                clip_feats_np = clip_out["clip_features"].detach().cpu().numpy()
                clip_mask_np = clip_out["clip_mask"].detach().cpu().numpy()
                clip_view_ids_np = clip_out["clip_view_ids"].detach().cpu().numpy()
                feats_np = None
            else:
                feats = model(batch).float()
                feats_np = feats.detach().cpu().numpy()
                clip_feats_np = clip_mask_np = clip_view_ids_np = None
        labels = batch["label"].detach().cpu().numpy().tolist()
        events = batch["event"].detach().cpu().numpy().tolist()
        times = batch["time_to_event"].detach().cpu().numpy().tolist()
        censored = batch["censored"].detach().cpu().numpy().tolist()
        pids = batch["patient_id"]
        class_names = batch["class_name"]
        event_types = batch.get("event_type", [""] * len(pids))
        for i in range(len(pids)):
            row = {
                "patient_id": str(pids[i]),
                "class_name": str(class_names[i]),
                "label": int(labels[i]),
                "event": int(events[i]),
                "time_to_event": float(times[i]),
                "censored": float(censored[i]),
                "event_type": str(event_types[i]) if isinstance(event_types, (list, tuple)) else "",
            }
            if fine_mode in {"agg_only", "cross_attention", "cross_attn", "hybrid_cross_attention", "hybrid_cross_attn", "hybrid"}:
                row["clip_features"] = clip_feats_np[i].astype(np.float32)
                row["clip_mask"] = clip_mask_np[i].astype(bool)
                row["clip_view_ids"] = clip_view_ids_np[i].astype(np.int64)
            else:
                row["feature"] = feats_np[i].astype(np.float32)
            rows.append(row)
        if is_main_process() and args.print_freq > 0 and (step + 1) % args.print_freq == 0:
            print(f"  Feature extraction step {step+1:04d}/{len(loader):04d} | elapsed={time.time()-start:.1f}s")

    if is_dist_avail_and_initialized():
        gathered = [None for _ in range(get_world_size())]
        dist.all_gather_object(gathered, rows)
        merged = []
        for g in gathered:
            merged.extend(g)
        rows = merged
    # Deduplicate padded DDP samples.
    dedup = {}
    for r in rows:
        dedup[r["patient_id"]] = r
    return list(dedup.values())


def save_feature_rows(rows: List[Dict], out_npz: Path, out_csv: Path):
    ensure_dir(out_npz.parent)
    is_clip = bool(rows) and ("clip_features" in rows[0])
    if is_clip:
        clip_features = np.stack([r["clip_features"] for r in rows], axis=0).astype(np.float32) if rows else np.empty((0, 0, 0), dtype=np.float32)
        clip_mask = np.stack([r["clip_mask"] for r in rows], axis=0).astype(bool) if rows else np.empty((0, 0), dtype=bool)
        clip_view_ids = np.stack([r["clip_view_ids"] for r in rows], axis=0).astype(np.int64) if rows else np.empty((0, 0), dtype=np.int64)
        np.savez_compressed(out_npz, clip_features=clip_features, clip_mask=clip_mask, clip_view_ids=clip_view_ids)
    else:
        feat = np.stack([r["feature"] for r in rows], axis=0).astype(np.float32) if rows else np.empty((0, 0), dtype=np.float32)
        np.savez_compressed(out_npz, features=feat)
    meta = []
    for i, r in enumerate(rows):
        meta.append({k: r[k] for k in ["patient_id", "class_name", "label", "event", "time_to_event", "censored", "event_type"]})
        meta[-1]["feature_index"] = i
        meta[-1]["feature_mode"] = "clip" if is_clip else "patient"
    pd.DataFrame(meta).to_csv(out_csv, index=False, encoding="utf-8-sig")


def load_feature_rows(npz_path: Path, csv_path: Path) -> List[Dict]:
    data = np.load(npz_path)
    meta = pd.read_csv(csv_path, dtype={"patient_id": str})
    rows = []
    is_clip = "clip_features" in data.files
    for _, r in meta.iterrows():
        idx = int(r["feature_index"])
        row = {
            "patient_id": str(r["patient_id"]),
            "class_name": str(r["class_name"]),
            "label": int(r["label"]),
            "event": int(r["event"]),
            "time_to_event": float(r["time_to_event"]),
            "censored": float(r["censored"]),
            "event_type": str(r.get("event_type", "")),
        }
        if is_clip:
            row["clip_features"] = data["clip_features"][idx].astype(np.float32)
            row["clip_mask"] = data["clip_mask"][idx].astype(bool)
            row["clip_view_ids"] = data["clip_view_ids"][idx].astype(np.int64)
        else:
            row["feature"] = data["features"][idx].astype(np.float32)
        rows.append(row)
    return rows


def rows_to_tensors(rows: List[Dict], device: torch.device):
    X = torch.tensor(np.stack([r["feature"] for r in rows], axis=0), dtype=torch.float32, device=device)
    y = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long, device=device)
    event = torch.tensor([int(r["event"]) for r in rows], dtype=torch.float32, device=device)
    timev = torch.tensor([float(r["time_to_event"]) for r in rows], dtype=torch.float32, device=device)
    return X, y, event, timev


def rows_to_clip_tensors(rows: List[Dict], device: torch.device):
    X = torch.tensor(np.stack([r["clip_features"] for r in rows], axis=0), dtype=torch.float32, device=device)
    clip_mask = torch.tensor(np.stack([r["clip_mask"] for r in rows], axis=0), dtype=torch.bool, device=device)
    clip_view_ids = torch.tensor(np.stack([r["clip_view_ids"] for r in rows], axis=0), dtype=torch.long, device=device)
    y = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long, device=device)
    event = torch.tensor([int(r["event"]) for r in rows], dtype=torch.float32, device=device)
    timev = torch.tensor([float(r["time_to_event"]) for r in rows], dtype=torch.float32, device=device)
    return X, clip_mask, clip_view_ids, y, event, timev


def parse_horizons_days(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    out = []
    for item in str(s).split(","):
        item = item.strip()
        if item:
            out.append(int(round(float(item))))
    return out


def build_horizon_targets_np(rows: List[Dict], horizons_days: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Censoring-aware fixed-window labels. Uncertain censored-before-horizon samples are masked out."""
    labels = np.zeros((len(rows), len(horizons_days)), dtype=np.float32)
    masks = np.zeros((len(rows), len(horizons_days)), dtype=bool)
    for i, r in enumerate(rows):
        e = int(r["event"])
        t = float(r["time_to_event"])
        for j, h in enumerate(horizons_days):
            if e == 1:
                labels[i, j] = 1.0 if t <= h else 0.0
                masks[i, j] = True
            else:
                if t >= h:
                    labels[i, j] = 0.0
                    masks[i, j] = True
                else:
                    labels[i, j] = 0.0
                    masks[i, j] = False
    return labels, masks


def build_horizon_targets_tensors(rows: List[Dict], horizons_days: Sequence[int], device: torch.device):
    labels, masks = build_horizon_targets_np(rows, horizons_days)
    return torch.tensor(labels, dtype=torch.float32, device=device), torch.tensor(masks, dtype=torch.bool, device=device)


def selected_event_logits(outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Optional[torch.Tensor]:
    if "hcm_event_logits" not in outputs or "dcm_event_logits" not in outputs:
        return None
    return torch.where((labels == CLASS_TO_IDX["HCM"]).unsqueeze(1), outputs["hcm_event_logits"], outputs["dcm_event_logits"])


# =============================================================================
# Survival losses and metrics
# =============================================================================
class CoxPHLoss(nn.Module):
    def forward(self, risk: torch.Tensor, timev: torch.Tensor, event: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        risk = risk.reshape(-1)[mask]
        timev = timev.reshape(-1)[mask]
        event = event.reshape(-1)[mask]
        if risk.numel() < 2 or event.sum() <= 0:
            return risk.sum() * 0.0
        order = torch.argsort(timev, descending=True)
        risk = risk[order]
        event = event[order]
        log_cumsum = torch.logcumsumexp(risk, dim=0)
        return -((risk - log_cumsum) * event).sum() / event.sum().clamp(min=1.0)


def get_selected_risk(outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    return torch.where(labels == CLASS_TO_IDX["HCM"], outputs["hcm_risk"], outputs["dcm_risk"])


def concordance_index(timev, event, risk) -> float:
    timev = np.asarray(timev, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    n = len(timev)
    comparable = 0.0
    concordant = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if timev[i] == timev[j]:
                continue
            # smaller time with event is the case expected to have higher risk
            if timev[i] < timev[j] and event[i] == 1:
                comparable += 1
                if risk[i] > risk[j] + 1e-12:
                    concordant += 1
                elif abs(risk[i] - risk[j]) <= 1e-12:
                    concordant += 0.5
            elif timev[j] < timev[i] and event[j] == 1:
                comparable += 1
                if risk[j] > risk[i] + 1e-12:
                    concordant += 1
                elif abs(risk[i] - risk[j]) <= 1e-12:
                    concordant += 0.5
    return float(concordant / comparable) if comparable > 0 else float("nan")


def km_curve(timev, event):
    timev = np.asarray(timev, dtype=float)
    event = np.asarray(event, dtype=int)
    unique_times = np.sort(np.unique(timev[event == 1]))
    xs, ys = [0.0], [1.0]
    surv = 1.0
    for t in unique_times:
        at_risk = np.sum(timev >= t)
        d = np.sum((timev == t) & (event == 1))
        if at_risk > 0:
            xs.extend([t, t])
            ys.extend([surv, surv * (1.0 - d / at_risk)])
            surv = ys[-1]
    if len(timev) > 0:
        xs.append(float(np.max(timev)))
        ys.append(surv)
    return np.asarray(xs), np.asarray(ys)


def logrank_p_value(timev, event, group):
    timev = np.asarray(timev, dtype=float)
    event = np.asarray(event, dtype=int)
    group = np.asarray(group, dtype=int)
    event_times = np.sort(np.unique(timev[event == 1]))
    O1 = E1 = V1 = 0.0
    for t in event_times:
        at = timev >= t
        n1 = np.sum(at & (group == 1))
        n0 = np.sum(at & (group == 0))
        n = n1 + n0
        d1 = np.sum((timev == t) & (event == 1) & (group == 1))
        d0 = np.sum((timev == t) & (event == 1) & (group == 0))
        d = d1 + d0
        if n <= 1 or d == 0:
            continue
        e1 = d * n1 / n
        v1 = (n1 * n0 * d * (n - d)) / (n * n * max(n - 1, 1))
        O1 += d1
        E1 += e1
        V1 += v1
    if V1 <= 0:
        return float("nan")
    chi2 = (O1 - E1) ** 2 / V1
    # Survival of chi-square df=1: erfc(sqrt(x/2))
    return float(math.erfc(math.sqrt(chi2 / 2.0)))


def evaluate_risks(rows: List[Dict], risks: np.ndarray, thresholds: Dict[str, float]) -> Dict:
    df = pd.DataFrame([{k: r[k] for k in ["patient_id", "class_name", "label", "event", "time_to_event", "censored", "event_type"]} for r in rows])
    df["risk"] = risks.astype(float)
    metrics = {}
    for disease in ["HCM", "DCM"]:
        sub = df[df["class_name"] == disease].copy()
        if len(sub) == 0:
            continue
        metrics[f"{disease}_n"] = int(len(sub))
        metrics[f"{disease}_events"] = int(sub["event"].sum())
        metrics[f"{disease}_cindex"] = concordance_index(sub["time_to_event"].values, sub["event"].values, sub["risk"].values)
        thr = thresholds.get(disease, float(sub["risk"].median()))
        group = (sub["risk"].values >= thr).astype(int)
        metrics[f"{disease}_risk_threshold"] = float(thr)
        metrics[f"{disease}_logrank_p"] = logrank_p_value(sub["time_to_event"].values, sub["event"].values, group)
    metrics["overall_n"] = int(len(df))
    metrics["overall_events"] = int(df["event"].sum())
    metrics["overall_cindex"] = concordance_index(df["time_to_event"].values, df["event"].values, df["risk"].values)
    return metrics


def predict_risks(heads: nn.Module, rows: List[Dict], device: torch.device) -> np.ndarray:
    heads.eval()
    with torch.no_grad():
        X, y, event, timev = rows_to_tensors(rows, device)
        out = heads(X)
        risk = get_selected_risk(out, y)
    return risk.detach().cpu().numpy().astype(np.float32)


def get_train_thresholds(train_rows: List[Dict], train_risks: np.ndarray) -> Dict[str, float]:
    df = pd.DataFrame([{"class_name": r["class_name"], "risk": float(rv)} for r, rv in zip(train_rows, train_risks)])
    thresholds = {}
    for disease in ["HCM", "DCM"]:
        sub = df[df["class_name"] == disease]
        if len(sub) > 0:
            thresholds[disease] = float(sub["risk"].median())
    return thresholds


def save_risk_predictions(rows: List[Dict], risks: np.ndarray, thresholds: Dict[str, float], path: Path):
    records = []
    for r, risk in zip(rows, risks):
        disease = r["class_name"]
        thr = thresholds.get(disease, float(np.median(risks)))
        records.append({
            "patient_id": r["patient_id"],
            "class_name": disease,
            "event": int(r["event"]),
            "time_to_event": float(r["time_to_event"]),
            "censored": float(r["censored"]),
            "event_type": r.get("event_type", ""),
            "risk": float(risk),
            "risk_threshold": float(thr),
            "risk_group": "high" if float(risk) >= thr else "low",
        })
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")


def save_event_type_summary(all_rows_by_split: Dict[str, List[Dict]], out_dir: Path):
    records = []
    for split, rows in all_rows_by_split.items():
        for r in rows:
            records.append({"split": split, "class_name": r["class_name"], "event": int(r["event"]), "event_type": r.get("event_type", "")})
    if not records:
        return
    df = pd.DataFrame(records)
    summary = df[df["event"] == 1].groupby(["split", "class_name", "event_type"]).size().reset_index(name="count")
    summary.to_csv(out_dir / "event_type_summary.csv", index=False, encoding="utf-8-sig")


def plot_km_for_split(rows: List[Dict], risks: np.ndarray, thresholds: Dict[str, float], out_dir: Path, split: str):
    if not HAS_MPL:
        return
    df = pd.DataFrame([{k: r[k] for k in ["patient_id", "class_name", "event", "time_to_event"]} for r in rows])
    df["risk"] = risks
    for disease in ["HCM", "DCM"]:
        sub = df[df["class_name"] == disease].copy()
        if len(sub) < 4:
            continue
        thr = thresholds.get(disease, float(sub["risk"].median()))
        sub["group"] = (sub["risk"] >= thr).astype(int)
        pval = logrank_p_value(sub["time_to_event"].values, sub["event"].values, sub["group"].values)
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        for g, name in [(0, "Low risk"), (1, "High risk")]:
            ss = sub[sub["group"] == g]
            if len(ss) == 0:
                continue
            x, y = km_curve(ss["time_to_event"].values, ss["event"].values)
            ax.step(x / 365.0, y, where="post", label=f"{name} (n={len(ss)}, events={int(ss['event'].sum())})")
        ax.set_xlabel("Time after CMR (years)")
        ax.set_ylabel("Event-free survival")
        ax.set_title(f"{disease} Kaplan-Meier | {split} | log-rank p={pval:.3g}")
        ax.set_ylim(0, 1.02)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"km_{split}_{disease}.png", dpi=300)
        plt.close(fig)


def plot_survival_history(history: List[Dict], out_dir: Path):
    if not HAS_MPL or not history:
        return
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["epoch"], df["loss"], label="train Cox loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_overall_cindex"], label="val overall C-index", linestyle="--")
    ax2.set_ylabel("C-index")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "survival_training_history.png", dpi=300)
    plt.close(fig)


def compute_horizon_bce_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor, horizon_labels: torch.Tensor,
                             horizon_masks: torch.Tensor, args) -> torch.Tensor:
    """Disease-specific fixed-horizon BCE loss with optional positive-class weighting.

    This is only used when train_horizon_heads=True and horizon_loss_weight>0.
    For the main Cox-only setting, this function returns 0 and fixed-horizon metrics
    are computed from the Cox risk score instead.
    """
    if horizon_labels is None or horizon_labels.numel() == 0 or "hcm_event_logits" not in outputs:
        return outputs["hcm_risk"].sum() * 0.0

    use_pos_weight = bool(getattr(args, "use_horizon_pos_weight", True))
    pos_weight_max = float(getattr(args, "horizon_pos_weight_max", 50.0))

    total_loss = outputs["hcm_risk"].sum() * 0.0
    total_weight = 0.0

    for disease, key in [("HCM", "hcm_event_logits"), ("DCM", "dcm_event_logits")]:
        disease_mask = (labels == CLASS_TO_IDX[disease]).unsqueeze(1) & horizon_masks
        logits = outputs[key]

        for j in range(horizon_labels.shape[1]):
            m = disease_mask[:, j]
            if not m.any():
                continue
            y = horizon_labels[m, j]
            z = logits[m, j]

            loss_vec = F.binary_cross_entropy_with_logits(z, y, reduction="none")
            if use_pos_weight:
                pos = float(y.sum().item())
                neg = float(y.numel() - pos)
                if pos > 0:
                    pw = min(neg / max(pos, 1.0), pos_weight_max)
                else:
                    pw = 1.0
                weight = torch.where(y > 0.5, torch.full_like(y, pw), torch.ones_like(y))
            else:
                weight = torch.ones_like(y)

            total_loss = total_loss + (loss_vec * weight).sum()
            total_weight += float(weight.sum().item())

    return total_loss / max(total_weight, 1.0)

def predict_outputs(heads: nn.Module, rows: List[Dict], device: torch.device, horizons_days: Sequence[int]):
    heads.eval()
    with torch.no_grad():
        if rows and "clip_features" in rows[0]:
            Xc, clip_mask, clip_view_ids, y, event, timev = rows_to_clip_tensors(rows, device)
            out = heads(Xc, clip_mask, clip_view_ids)
        else:
            X, y, event, timev = rows_to_tensors(rows, device)
            out = heads(X)
        risk = get_selected_risk(out, y)
        logits = selected_event_logits(out, y)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32) if logits is not None else None
    return risk.detach().cpu().numpy().astype(np.float32), probs


def _best_binary_threshold(y_true: np.ndarray, score: np.ndarray, mode: str = "youden") -> float:
    """Select a binary threshold on validation data."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    if len(score) == 0:
        return 0.5
    if mode == "median" or len(np.unique(y_true)) < 2:
        return float(np.median(score))

    candidates = np.unique(score)
    if len(candidates) > 512:
        candidates = np.quantile(score, np.linspace(0.01, 0.99, 512))
    best_thr = float(np.median(score))
    best_value = -1e18

    for thr in candidates:
        pred = (score >= thr).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        tn = int(((y_true == 0) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-12)
        value = f1 if mode == "f1" else (sens + spec - 1.0)
        if value > best_value:
            best_value = value
            best_thr = float(thr)
    return best_thr


def select_horizon_thresholds_from_val(rows: List[Dict], scores: np.ndarray, horizons_days: Sequence[int],
                                       mode: str = "youden") -> Dict[str, float]:
    """Select group- and horizon-specific thresholds using validation data only.

    Scores can be Cox risk scores or event-head probabilities. Higher score = higher risk.
    Keys are like HCM_365d, DCM_1095d, overall_1825d.
    """
    thresholds: Dict[str, float] = {}
    if len(rows) == 0 or len(horizons_days) == 0:
        return thresholds
    labels_np, masks_np = build_horizon_targets_np(rows, horizons_days)
    class_names = np.array([r["class_name"] for r in rows])
    score_arr = np.asarray(scores)
    if score_arr.ndim == 1:
        score_arr = np.repeat(score_arr[:, None], len(horizons_days), axis=1)

    for group in ["HCM", "DCM", "overall"]:
        gmask = np.ones(len(rows), dtype=bool) if group == "overall" else (class_names == group)
        for j, h in enumerate(horizons_days):
            m = gmask & masks_np[:, j]
            key = f"{group}_{int(h)}d"
            if m.sum() == 0:
                continue
            thresholds[key] = _best_binary_threshold(labels_np[m, j], score_arr[m, j], mode=mode)
    return thresholds


def evaluate_horizon_scores(rows: List[Dict], scores: Optional[np.ndarray], horizons_days: Sequence[int],
                            thresholds: Optional[Dict[str, float]] = None,
                            default_threshold: float = 0.5,
                            score_name: str = "cox_risk") -> Dict:
    """Evaluate fixed-window event prediction from scores.

    Main recommended setting: scores = Cox risk scores.
    Optional setting: scores = event-head sigmoid probabilities.
    Thresholds are selected on validation data and reused for test.
    """
    metrics = {}
    if scores is None or len(horizons_days) == 0 or len(rows) == 0:
        return metrics
    labels_np, masks_np = build_horizon_targets_np(rows, horizons_days)
    class_names = np.array([r["class_name"] for r in rows])
    score_arr = np.asarray(scores).astype(float)
    if score_arr.ndim == 1:
        score_arr = np.repeat(score_arr[:, None], len(horizons_days), axis=1)
    thresholds = thresholds or {}

    for group in ["HCM", "DCM", "overall"]:
        gmask = np.ones(len(rows), dtype=bool) if group == "overall" else (class_names == group)
        for j, h in enumerate(horizons_days):
            m = gmask & masks_np[:, j]
            prefix = f"{group}_{int(h)}d"
            metrics[f"{prefix}_score_source"] = score_name
            metrics[f"{prefix}_n"] = int(m.sum())
            if m.sum() == 0:
                continue
            y_true = labels_np[m, j].astype(int)
            y_score = score_arr[m, j].astype(float)
            thr = thresholds.get(prefix, default_threshold)
            y_pred = (y_score >= thr).astype(int)
            metrics[f"{prefix}_threshold"] = float(thr)
            metrics[f"{prefix}_events"] = int(y_true.sum())
            metrics[f"{prefix}_non_events"] = int(len(y_true) - y_true.sum())
            if len(np.unique(y_true)) >= 2:
                metrics[f"{prefix}_auc"] = float(roc_auc_score(y_true, y_score))
                metrics[f"{prefix}_ap"] = float(average_precision_score(y_true, y_score))
            else:
                metrics[f"{prefix}_auc"] = None
                metrics[f"{prefix}_ap"] = None
            metrics[f"{prefix}_acc"] = float(accuracy_score(y_true, y_pred))
            metrics[f"{prefix}_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics[f"{prefix}_recall_sensitivity"] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics[f"{prefix}_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            metrics[f"{prefix}_specificity"] = float(tn / max(tn + fp, 1))
    return metrics


def evaluate_horizon_predictions(rows: List[Dict], probs: Optional[np.ndarray], horizons_days: Sequence[int], threshold: float = 0.5) -> Dict:
    return evaluate_horizon_scores(rows, probs, horizons_days, thresholds=None, default_threshold=threshold, score_name="event_head")

def save_risk_and_horizon_predictions(rows: List[Dict], risks: np.ndarray, probs: Optional[np.ndarray],
                                      thresholds: Dict[str, float], horizons_days: Sequence[int], path: Path,
                                      horizon_thresholds: Optional[Dict[str, float]] = None,
                                      head_horizon_thresholds: Optional[Dict[str, float]] = None):
    labels_np, masks_np = build_horizon_targets_np(rows, horizons_days) if horizons_days else (None, None)
    horizon_thresholds = horizon_thresholds or {}
    head_horizon_thresholds = head_horizon_thresholds or {}
    records = []
    for i, (r, risk) in enumerate(zip(rows, risks)):
        disease = r["class_name"]
        thr = thresholds.get(disease, float(np.median(risks)))
        rec = {
            "patient_id": r["patient_id"],
            "class_name": disease,
            "event": int(r["event"]),
            "time_to_event": float(r["time_to_event"]),
            "censored": float(r["censored"]),
            "event_type": r.get("event_type", ""),
            "risk": float(risk),
            "risk_threshold": float(thr),
            "risk_group": "high" if float(risk) >= thr else "low",
        }
        if horizons_days:
            for j, h in enumerate(horizons_days):
                key = f"{disease}_{int(h)}d"
                key_overall = f"overall_{int(h)}d"
                hthr = horizon_thresholds.get(key, horizon_thresholds.get(key_overall, float(np.median(risks))))
                available = bool(masks_np[i, j])
                rec[f"event_within_{int(h)}d_label"] = int(labels_np[i, j]) if available else -1
                rec[f"event_within_{int(h)}d_available"] = int(available)
                rec[f"event_within_{int(h)}d_cox_risk_score"] = float(risk)
                rec[f"event_within_{int(h)}d_cox_risk_threshold"] = float(hthr)
                rec[f"event_within_{int(h)}d_cox_risk_pred"] = int(float(risk) >= float(hthr))
                if probs is not None:
                    hpthr = head_horizon_thresholds.get(key, head_horizon_thresholds.get(key_overall, 0.5))
                    rec[f"event_within_{int(h)}d_head_prob"] = float(probs[i, j])
                    rec[f"event_within_{int(h)}d_head_threshold"] = float(hpthr)
                    rec[f"event_within_{int(h)}d_head_pred"] = int(float(probs[i, j]) >= float(hpthr))
        records.append(rec)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")

def save_horizon_summary_tables(all_metrics: Dict, out_dir: Path, horizons_days: Sequence[int]):
    records = []
    for split in ["train", "val", "test"]:
        hm = all_metrics.get(split, {}).get("horizon_metrics", {})
        for group in ["HCM", "DCM", "overall"]:
            for h in horizons_days:
                prefix = f"{group}_{int(h)}d"
                if f"{prefix}_n" not in hm:
                    continue
                records.append({
                    "split": split,
                    "group": group,
                    "horizon_days": int(h),
                    "horizon_years": round(float(h) / 365.0, 3),
                    "n": hm.get(f"{prefix}_n"),
                    "events": hm.get(f"{prefix}_events"),
                    "threshold": hm.get(f"{prefix}_threshold"),
                    "score_source": hm.get(f"{prefix}_score_source"),
                    "auc": hm.get(f"{prefix}_auc"),
                    "average_precision": hm.get(f"{prefix}_ap"),
                    "accuracy": hm.get(f"{prefix}_acc"),
                    "precision": hm.get(f"{prefix}_precision"),
                    "sensitivity": hm.get(f"{prefix}_recall_sensitivity"),
                    "specificity": hm.get(f"{prefix}_specificity"),
                    "f1": hm.get(f"{prefix}_f1"),
                })
    if records:
        pd.DataFrame(records).to_csv(out_dir / "horizon_event_metrics_summary.csv", index=False, encoding="utf-8-sig")




def save_horizon_label_statistics(rows_by_split: Dict[str, List[Dict]], out_dir: Path, horizons_days: Sequence[int]):
    """Save censoring-aware event-count statistics for each fixed horizon.

    Example output columns include event_ratio_available, so you can directly read:
        HCM 1-year: 8/149
        DCM 1-year: 17/160
        overall 1-year: 25/309
    """
    if not horizons_days:
        return

    rows_by_split = dict(rows_by_split)
    all_rows = []
    for name in ["train", "val", "test"]:
        all_rows.extend(rows_by_split.get(name, []))
    rows_by_split["all"] = all_rows

    records = []
    for split, rows in rows_by_split.items():
        if len(rows) == 0:
            continue
        class_names = np.array([r.get("class_name", "") for r in rows])
        events_all = np.array([int(r.get("event", 0)) for r in rows], dtype=int)
        times_all = np.array([float(r.get("time_to_event", 0)) for r in rows], dtype=float)
        labels_np, masks_np = build_horizon_targets_np(rows, horizons_days)

        for group in ["HCM", "DCM", "overall"]:
            gmask = np.ones(len(rows), dtype=bool) if group == "overall" else (class_names == group)
            total_n = int(gmask.sum())
            if total_n == 0:
                continue
            total_observed_events = int(events_all[gmask].sum())
            total_censored = int(total_n - total_observed_events)
            median_followup_days = float(np.median(times_all[gmask]))
            mean_followup_days = float(np.mean(times_all[gmask]))

            for j, h in enumerate(horizons_days):
                available_mask = gmask & masks_np[:, j]
                unavailable_mask = gmask & (~masks_np[:, j])
                available_n = int(available_mask.sum())
                uncertain_n = int(unavailable_mask.sum())
                if available_n > 0:
                    event_n = int(labels_np[available_mask, j].sum())
                    non_event_n = int(available_n - event_n)
                    event_rate_available = float(event_n / available_n)
                else:
                    event_n = 0
                    non_event_n = 0
                    event_rate_available = None

                records.append({
                    "split": split,
                    "group": group,
                    "horizon_days": int(h),
                    "horizon_years": round(float(h) / 365.0, 3),
                    "total_n": total_n,
                    "total_observed_events_any_time": total_observed_events,
                    "total_censored_any_time": total_censored,
                    "median_followup_days": median_followup_days,
                    "mean_followup_days": mean_followup_days,
                    "available_n": available_n,
                    "events_within_horizon": event_n,
                    "non_events_within_horizon": non_event_n,
                    "uncertain_censored_before_horizon": uncertain_n,
                    "event_rate_available": event_rate_available,
                    "event_rate_total": float(event_n / total_n) if total_n > 0 else None,
                    "event_ratio_available": f"{event_n}/{available_n}" if available_n > 0 else "0/0",
                })

    if records:
        df = pd.DataFrame(records)
        df.to_csv(out_dir / "horizon_event_label_statistics.csv", index=False, encoding="utf-8-sig")
        compact = df[df["split"].isin(["test", "all"])][[
            "split", "group", "horizon_years", "event_ratio_available",
            "events_within_horizon", "available_n", "uncertain_censored_before_horizon",
            "event_rate_available",
        ]]
        compact.to_csv(out_dir / "horizon_event_label_statistics_compact.csv", index=False, encoding="utf-8-sig")

        test_df = df[df["split"] == "test"]
        if len(test_df) > 0:
            print0("\n========== Fixed-horizon label statistics on TEST ==========")
            for h in horizons_days:
                hdf = test_df[test_df["horizon_days"] == int(h)]
                pieces = []
                for group in ["HCM", "DCM", "overall"]:
                    r = hdf[hdf["group"] == group]
                    if len(r) > 0:
                        rr = r.iloc[0]
                        pieces.append(f"{group} {round(h/365.0, 1)}y: {rr['event_ratio_available']}")
                if pieces:
                    print0("  " + " | ".join(pieces))


def train_survival_heads(train_rows, val_rows, test_rows, feature_dim: int, args, out_dir: Path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    horizons_days = parse_horizons_days(getattr(args, "eval_horizons_days", ""))
    train_horizon_heads = bool(getattr(args, "train_horizon_heads", False)) and float(getattr(args, "horizon_loss_weight", 0.0)) > 0 and len(horizons_days) > 0
    use_clip_agg = bool(train_rows) and ("clip_features" in train_rows[0])
    if use_clip_agg:
        selected_views = parse_csv_list(args.selected_views)
        slice_plan = parse_slice_plan(args.slice_plan)
        survival_agg = canonicalize_agg(getattr(args, "survival_agg", getattr(args, "agg", "hier_attn")))
        fine_mode = str(getattr(args, "fine_tune_mode", "agg_only")).lower()
        if fine_mode in {"hybrid_cross_attention", "hybrid_cross_attn", "hybrid"}:
            heads = HybridCrossAttentionSurvivalModel(
                selected_views=selected_views,
                slice_plan=slice_plan,
                feature_dim=feature_dim,
                agg=survival_agg,
                agg_dropout=float(getattr(args, "agg_finetune_dropout", 0.05)),
                num_heads=int(getattr(args, "cross_attn_heads", 4)),
                num_layers=int(getattr(args, "cross_attn_layers", 1)),
                cross_dropout=float(getattr(args, "cross_attn_dropout", 0.10)),
                hybrid_gate_init=float(getattr(args, "hybrid_cross_gate_init", 0.50)),
                hcm_gate_init=float(getattr(args, "hcm_cross_gate_init", getattr(args, "hybrid_cross_gate_init", 0.50))),
                dcm_gate_init=float(getattr(args, "dcm_cross_gate_init", getattr(args, "hybrid_cross_gate_init", 0.50))),
                survival_hidden_dim=args.survival_hidden_dim,
                survival_dropout=args.survival_dropout,
                survival_head_type=args.survival_head_type,
                num_horizons=len(horizons_days) if train_horizon_heads else 0,
                hcm_hidden_dim=int(getattr(args, "hcm_hidden_dim", 0)) or None,
                dcm_hidden_dim=int(getattr(args, "dcm_hidden_dim", 0)) or None,
                hcm_dropout=float(getattr(args, "hcm_dropout", args.survival_dropout)),
                dcm_dropout=float(getattr(args, "dcm_dropout", args.survival_dropout)),
            ).to(device)
            heads.load_aggregation_from_classification_checkpoint(getattr(args, "classification_ckpt", ""))
            print0(f"fine_tune_mode=hybrid_cross_attention: training survival_agg={survival_agg} pooling + disease-specific cross-attention residual + survival heads on cached clip features.")
        elif fine_mode in {"cross_attention", "cross_attn"}:
            heads = DiseaseSpecificCrossAttentionSurvivalModel(
                selected_views=selected_views,
                slice_plan=slice_plan,
                feature_dim=feature_dim,
                num_heads=int(getattr(args, "cross_attn_heads", 4)),
                num_layers=int(getattr(args, "cross_attn_layers", 1)),
                cross_dropout=float(getattr(args, "cross_attn_dropout", 0.10)),
                survival_hidden_dim=args.survival_hidden_dim,
                survival_dropout=args.survival_dropout,
                survival_head_type=args.survival_head_type,
                num_horizons=len(horizons_days) if train_horizon_heads else 0,
                hcm_hidden_dim=int(getattr(args, "hcm_hidden_dim", 0)) or None,
                dcm_hidden_dim=int(getattr(args, "dcm_hidden_dim", 0)) or None,
                hcm_dropout=float(getattr(args, "hcm_dropout", args.survival_dropout)),
                dcm_dropout=float(getattr(args, "dcm_dropout", args.survival_dropout)),
            ).to(device)
            heads.load_aggregation_from_classification_checkpoint(getattr(args, "classification_ckpt", ""))
            print0("fine_tune_mode=cross_attention: training disease-specific HCM/DCM query cross-attention + survival heads on cached clip features.")
        else:
            heads = ClipAggregationSurvivalModel(
                selected_views=selected_views,
                slice_plan=slice_plan,
                feature_dim=feature_dim,
                agg=survival_agg,
                agg_dropout=float(getattr(args, "agg_finetune_dropout", 0.05)),
                survival_hidden_dim=args.survival_hidden_dim,
                survival_dropout=args.survival_dropout,
                survival_head_type=args.survival_head_type,
                num_horizons=len(horizons_days) if train_horizon_heads else 0,
                hcm_hidden_dim=int(getattr(args, "hcm_hidden_dim", 0)) or None,
                dcm_hidden_dim=int(getattr(args, "dcm_hidden_dim", 0)) or None,
                hcm_dropout=float(getattr(args, "hcm_dropout", args.survival_dropout)),
                dcm_dropout=float(getattr(args, "dcm_dropout", args.survival_dropout)),
            ).to(device)
            heads.load_aggregation_from_classification_checkpoint(getattr(args, "classification_ckpt", ""))
            print0(f"fine_tune_mode=agg_only: training survival_agg={survival_agg} slice/view aggregation + survival heads on cached clip features.")
    else:
        heads = DiseaseSpecificCoxHeads(
            feature_dim=feature_dim,
            hidden_dim=args.survival_hidden_dim,
            dropout=args.survival_dropout,
            head_type=args.survival_head_type,
            num_horizons=len(horizons_days) if train_horizon_heads else 0,
        ).to(device)
        print0("fine_tune_mode=frozen_feature: training survival heads on fixed patient features.")
    if args.resume_head and Path(args.resume_head).exists():
        ckpt = safe_torch_load(args.resume_head, map_location="cpu", weights_only=True)
        heads.load_state_dict(ckpt.get("heads", ckpt), strict=False)
        print0(f"Loaded survival head checkpoint: {args.resume_head}")

    if use_clip_agg:
        opt = torch.optim.AdamW([
            {"params": heads.aggregation_parameters(), "lr": float(getattr(args, "feature_lr", 1e-5)), "weight_decay": args.survival_weight_decay},
            {"params": heads.survival_heads.parameters(), "lr": args.survival_lr, "weight_decay": args.survival_weight_decay},
        ])
        Xc_train, clip_mask_train, clip_view_ids_train, y_train, e_train, t_train = rows_to_clip_tensors(train_rows, device)
        X_train = None
    else:
        opt = torch.optim.AdamW(heads.parameters(), lr=args.survival_lr, weight_decay=args.survival_weight_decay)
        X_train, y_train, e_train, t_train = rows_to_tensors(train_rows, device)
    cox = CoxPHLoss()
    horizon_y_train, horizon_m_train = build_horizon_targets_tensors(train_rows, horizons_days, device) if horizons_days else (None, None)

    history = []
    best_val = -1e9
    best_epoch = 0
    no_improve = 0
    best_path = out_dir / "best_survival_heads.pt"
    print0(f"Fixed-horizon event prediction: horizons_days={horizons_days}")
    print0(f"  Main mode: Cox-only risk-score horizon evaluation")
    print0(f"  train_horizon_heads={train_horizon_heads}, horizon_loss_weight={getattr(args, 'horizon_loss_weight', 0.0)}, threshold_mode={getattr(args, 'horizon_threshold_mode', 'youden')}")
    print0(f"  Disease mode={getattr(args, 'disease_mode', 'joint')} | hcm_loss_weight={getattr(args, 'hcm_loss_weight', 1.0)}, dcm_loss_weight={getattr(args, 'dcm_loss_weight', 1.0)}, early_stop_metric={getattr(args, 'early_stop_metric', 'overall')}")
    print0(f"  gates: hcm_cross_gate_init={getattr(args, 'hcm_cross_gate_init', getattr(args, 'hybrid_cross_gate_init', 0.5))}, dcm_cross_gate_init={getattr(args, 'dcm_cross_gate_init', getattr(args, 'hybrid_cross_gate_init', 0.5))}")
    print0(f"  heads: hcm_hidden_dim={getattr(args, 'hcm_hidden_dim', getattr(args, 'survival_hidden_dim', 128))}, dcm_hidden_dim={getattr(args, 'dcm_hidden_dim', getattr(args, 'survival_hidden_dim', 128))}, hcm_dropout={getattr(args, 'hcm_dropout', getattr(args, 'survival_dropout', 0.2))}, dcm_dropout={getattr(args, 'dcm_dropout', getattr(args, 'survival_dropout', 0.2))}")

    if not args.eval_only:
        for epoch in range(1, args.survival_epochs + 1):
            heads.train()
            opt.zero_grad(set_to_none=True)
            if use_clip_agg:
                out = heads(Xc_train, clip_mask_train, clip_view_ids_train)
            else:
                out = heads(X_train)
            hcm_mask = (y_train == CLASS_TO_IDX["HCM"])
            dcm_mask = (y_train == CLASS_TO_IDX["DCM"])
            loss_hcm = cox(out["hcm_risk"], t_train, e_train, hcm_mask)
            loss_dcm = cox(out["dcm_risk"], t_train, e_train, dcm_mask)
            hcm_w = float(getattr(args, "hcm_loss_weight", 1.0))
            dcm_w = float(getattr(args, "dcm_loss_weight", 1.0))
            loss_cox = hcm_w * loss_hcm + dcm_w * loss_dcm
            loss_event = compute_horizon_bce_loss(out, y_train, horizon_y_train, horizon_m_train, args) if train_horizon_heads else loss_cox * 0.0
            loss = loss_cox + float(getattr(args, "horizon_loss_weight", 0.0)) * loss_event
            if args.risk_l2 and args.risk_l2 > 0:
                selected_risk = get_selected_risk(out, y_train)
                loss = loss + float(args.risk_l2) * selected_risk.pow(2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(heads.parameters(), 5.0)
            opt.step()

            train_risk, train_probs = predict_outputs(heads, train_rows, device, horizons_days)
            thresholds = get_train_thresholds(train_rows, train_risk)
            val_risk, val_probs = predict_outputs(heads, val_rows, device, horizons_days)
            val_metrics = evaluate_risks(val_rows, val_risk, thresholds)
            val_horizon_thresholds = select_horizon_thresholds_from_val(
                val_rows, val_risk, horizons_days, mode=str(getattr(args, "horizon_threshold_mode", "youden"))
            )
            val_horizon_metrics = evaluate_horizon_scores(
                val_rows, val_risk, horizons_days, thresholds=val_horizon_thresholds,
                default_threshold=float(getattr(args, "horizon_threshold", 0.5)), score_name="cox_risk"
            )
            val_overall = float(val_metrics.get("overall_cindex", np.nan))
            val_hcm = float(val_metrics.get("HCM_cindex", np.nan))
            val_dcm = float(val_metrics.get("DCM_cindex", np.nan))

            vals_for_bal = [v for v in [val_hcm, val_dcm] if not math.isnan(v)]
            val_balanced = float(np.mean(vals_for_bal)) if vals_for_bal else val_overall
            val_min = float(np.min(vals_for_bal)) if vals_for_bal else val_overall
            hcm_mw = float(getattr(args, "hcm_monitor_weight", 0.35))
            dcm_mw = float(getattr(args, "dcm_monitor_weight", 0.65))
            if not math.isnan(val_hcm) and not math.isnan(val_dcm):
                denom = max(hcm_mw + dcm_mw, 1e-8)
                val_dcm_weighted = float((hcm_mw * val_hcm + dcm_mw * val_dcm) / denom)
            else:
                val_dcm_weighted = val_balanced

            metric_name = str(getattr(args, "early_stop_metric", "overall")).lower()
            if metric_name in {"balanced", "mean"}:
                val_score = val_balanced
            elif metric_name in {"dcm_weighted", "weighted_dcm"}:
                val_score = val_dcm_weighted
            elif metric_name == "dcm":
                val_score = val_dcm if not math.isnan(val_dcm) else val_balanced
            elif metric_name == "hcm":
                val_score = val_hcm if not math.isnan(val_hcm) else val_balanced
            elif metric_name in {"min", "worst"}:
                val_score = val_min
            else:
                val_score = val_overall
            if math.isnan(val_score):
                vals = [v for k, v in val_metrics.items() if k.endswith("_cindex") and not math.isnan(v)]
                val_score = float(np.mean(vals)) if vals else -1e9

            rec = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_cox": float(loss_cox.item()),
                "loss_event": float(loss_event.item()) if horizons_days else 0.0,
                "loss_hcm": float(loss_hcm.item()),
                "loss_dcm": float(loss_dcm.item()),
                "weighted_loss_hcm": float(hcm_w * loss_hcm.item()),
                "weighted_loss_dcm": float(dcm_w * loss_dcm.item()),
                "val_overall_cindex": val_overall,
                "val_HCM_cindex": val_hcm,
                "val_DCM_cindex": val_dcm,
                "val_balanced_cindex": val_balanced,
                "val_dcm_weighted_cindex": val_dcm_weighted,
                "monitor_metric": metric_name,
                "monitor_score": float(val_score),
            }
            for k, v in val_horizon_metrics.items():
                if k.endswith("_auc") and v is not None:
                    rec[f"val_{k}"] = float(v)
            history.append(rec)
            if epoch % max(1, int(args.print_freq)) == 0 or epoch == 1:
                msg = (
                    f"Epoch {epoch:03d}/{args.survival_epochs} | "
                    f"loss={rec['loss']:.4f} | cox={rec['loss_cox']:.4f} | event={rec['loss_event']:.4f} | "
                    f"overall={rec['val_overall_cindex']:.4f} | HCM={rec['val_HCM_cindex']:.4f} | "
                    f"DCM={rec['val_DCM_cindex']:.4f} | bal={rec['val_balanced_cindex']:.4f} | "
                    f"monitor={rec['monitor_metric']}:{rec['monitor_score']:.4f}"
                )
                auc_items = [v for k, v in val_horizon_metrics.items() if k.startswith("overall_") and k.endswith("_auc") and v is not None]
                if auc_items:
                    msg += f" | val_horizon_auc_mean={float(np.mean(auc_items)):.4f}"
                print0(msg)

            improved = val_score > best_val + args.early_stop_min_delta
            if improved:
                best_val = val_score
                best_epoch = epoch
                no_improve = 0
                torch.save({"model": heads.state_dict(), "heads": heads.state_dict(), "feature_dim": feature_dim, "epoch": epoch, "best_val_score": best_val, "horizons_days": horizons_days, "use_clip_agg": use_clip_agg, "config": vars(args)}, best_path)
            else:
                no_improve += 1
                if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                    print0(f"🛑 Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_{str(getattr(args, 'early_stop_metric', 'overall'))}={best_val:.4f}")
                    break
        pd.DataFrame(history).to_csv(out_dir / "survival_history.csv", index=False, encoding="utf-8-sig")
        plot_survival_history(history, out_dir)

    if best_path.exists():
        ckpt = safe_torch_load(best_path, map_location="cpu", weights_only=True)
        state_to_load = ckpt.get("model", ckpt.get("heads", ckpt))
        heads.load_state_dict(state_to_load, strict=False)
        best_epoch = int(ckpt.get("epoch", best_epoch))
        best_val = float(ckpt.get("best_val_score", best_val))
        horizons_days = ckpt.get("horizons_days", horizons_days)

    train_risk, train_probs = predict_outputs(heads, train_rows, device, horizons_days)
    thresholds = get_train_thresholds(train_rows, train_risk)

    # Fixed-horizon thresholds must be chosen on validation data, not on test.
    val_risk_for_thr, val_probs_for_thr = predict_outputs(heads, val_rows, device, horizons_days)
    horizon_thresholds = select_horizon_thresholds_from_val(
        val_rows, val_risk_for_thr, horizons_days, mode=str(getattr(args, "horizon_threshold_mode", "youden"))
    )
    head_horizon_thresholds = {}
    if train_horizon_heads and val_probs_for_thr is not None:
        head_horizon_thresholds = select_horizon_thresholds_from_val(
            val_rows, val_probs_for_thr, horizons_days, mode=str(getattr(args, "horizon_threshold_mode", "youden"))
        )

    all_metrics = {
        "best_epoch": best_epoch,
        "best_val_score": best_val,
        "risk_thresholds": thresholds,
        "horizon_thresholds_from_val": horizon_thresholds,
        "head_horizon_thresholds_from_val": head_horizon_thresholds,
        "horizons_days": horizons_days,
        "horizon_score_source": "cox_risk",
        "train_horizon_heads": bool(train_horizon_heads),
        "fine_tune_mode": str(getattr(args, "fine_tune_mode", "agg_only")) if use_clip_agg else "frozen_feature",
        "diagnosis_encoder_agg": str(getattr(args, "diagnosis_encoder_agg", getattr(args, "agg", ""))),
        "survival_agg": str(getattr(args, "survival_agg", getattr(args, "agg", ""))),
    }
    for split, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        risk, probs = predict_outputs(heads, rows, device, horizons_days)
        save_risk_and_horizon_predictions(
            rows, risk, probs, thresholds, horizons_days, out_dir / f"risk_predictions_{split}.csv",
            horizon_thresholds=horizon_thresholds, head_horizon_thresholds=head_horizon_thresholds
        )
        metrics = evaluate_risks(rows, risk, thresholds)
        # Recommended main fixed-window metrics: use Cox risk score directly.
        metrics["horizon_metrics"] = evaluate_horizon_scores(
            rows, risk, horizons_days, thresholds=horizon_thresholds,
            default_threshold=float(getattr(args, "horizon_threshold", 0.5)), score_name="cox_risk"
        )
        # Optional: if horizon heads are trained, also save their metrics separately.
        if train_horizon_heads and probs is not None:
            metrics["horizon_head_metrics"] = evaluate_horizon_scores(
                rows, probs, horizons_days, thresholds=head_horizon_thresholds,
                default_threshold=float(getattr(args, "horizon_threshold", 0.5)), score_name="event_head"
            )
        all_metrics[split] = metrics
        plot_km_for_split(rows, risk, thresholds, out_dir, split)

    save_horizon_summary_tables(all_metrics, out_dir, horizons_days)
    save_horizon_label_statistics({"train": train_rows, "val": val_rows, "test": test_rows}, out_dir, horizons_days)
    with open(out_dir / "survival_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    return all_metrics




def filter_samples_by_disease_mode(rows: List[Dict], disease_mode: str) -> List[Dict]:
    """Filter rows/samples for standalone disease-specific prognosis training.

    DCM_only means only DCM patients are used for feature extraction, training,
    validation, testing, threshold selection, horizon metrics, and KM plots.
    This run is intended to tune an independent DCM prognosis model.
    """
    mode = str(disease_mode or "joint").strip().lower()
    if mode in {"", "joint", "both", "all"}:
        return list(rows)
    if mode in {"dcm", "dcm_only", "dcmonly", "dcm-only"}:
        return [r for r in rows if str(r.get("class_name", "")).upper() == "DCM"]
    if mode in {"hcm", "hcm_only", "hcmonly", "hcm-only"}:
        return [r for r in rows if str(r.get("class_name", "")).upper() == "HCM"]
    raise ValueError(f"Unknown disease_mode={disease_mode!r}. Use joint / DCM_only / HCM_only.")

# =============================================================================
# Main workflow
# =============================================================================
def add_bool_arg(parser, name: str, default: bool, help_text: str = ""):
    parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default, help=help_text)


def parse_args():
    cfg = RUN_CONFIG
    p = argparse.ArgumentParser("Disease-specific survival prognosis from multi-view Cine CMR")
    for k, v in cfg.items():
        if isinstance(v, bool):
            add_bool_arg(p, k, v)
        elif isinstance(v, int):
            p.add_argument(f"--{k}", type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(f"--{k}", type=float, default=v)
        else:
            p.add_argument(f"--{k}", type=str, default=v)
    return p.parse_args()


def get_free_tcp_port() -> int:
    """Ask OS for an available local TCP port, then close it. Used for torchrun --master_port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def resolve_master_port(preferred=0) -> int:
    """Return preferred port if >0, otherwise a free port. Avoids torchrun default 29500 conflicts."""
    try:
        preferred = int(preferred)
    except Exception:
        preferred = 0
    if preferred > 0:
        return preferred
    env_port = os.environ.get("MASTER_PORT", "")
    try:
        env_port_i = int(env_port) if env_port else 0
    except Exception:
        env_port_i = 0
    if env_port_i > 0:
        return env_port_i
    return get_free_tcp_port()


def maybe_auto_launch_ddp():
    cfg = RUN_CONFIG
    if str(cfg.get("gpu_mode", "single")).lower() != "ddp":
        return False
    if not cfg.get("auto_launch_ddp", True):
        return False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return False
    visible = str(cfg.get("visible_gpus", "")).strip()
    if not visible:
        return False
    nproc = len([x for x in visible.split(",") if x.strip()])
    if nproc <= 1:
        return False
    run_id = str(cfg.get("run_id", "") or "").strip() or now_string()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["CMR_RUN_ID"] = run_id

    # IMPORTANT: preserve parent-provided overrides.
    # In independent HCM/DCM mode, the parent process passes disease_mode=HCM_only/DCM_only
    # through CMR_RUN_CONFIG_OVERRIDES. The previous version overwrote that environment
    # variable with only {"run_id": ...}, so DDP workers fell back to the default
    # disease_mode="both_independent" and crashed inside torchrun.
    try:
        existing_overrides = json.loads(env.get("CMR_RUN_CONFIG_OVERRIDES", "") or "{}")
        if not isinstance(existing_overrides, dict):
            existing_overrides = {}
    except Exception:
        existing_overrides = {}
    existing_overrides["run_id"] = run_id
    env["CMR_RUN_CONFIG_OVERRIDES"] = json.dumps(existing_overrides, ensure_ascii=False)

    master_port = resolve_master_port(cfg.get("ddp_master_port", 0))
    env["MASTER_PORT"] = str(master_port)
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", f"--master_port={master_port}", sys.argv[0]]
    print(f"🚀 自动启动 DDP 特征提取: {nproc} GPUs | master_port={master_port} | {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    return True


def build_dataloader(dataset: Dataset, args, distributed: bool, shuffle: bool = False):
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=False,
    )
    return loader



# =============================================================================
# Standalone multi-seed runner
# =============================================================================
def parse_seed_list(seed_text) -> List[int]:
    if isinstance(seed_text, (list, tuple)):
        return [int(x) for x in seed_text]
    out = []
    for x in str(seed_text or "").replace(";", ",").split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def should_run_multiseed_parent() -> bool:
    if os.environ.get("CMR_MULTI_SEED_CHILD", "0") == "1":
        return False
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        return False
    return bool(RUN_CONFIG.get("multiseed_mode", False))


def make_multiseed_output_root() -> Path:
    root = str(RUN_CONFIG.get("multiseed_output_root", "auto") or "auto").strip()
    if root.lower() in {"", "auto", "none"}:
        script_stem = Path(sys.argv[0]).stem
        return Path.cwd() / f"{script_stem}_results_{now_string()}"
    return Path(root)


def flatten_metric_dict(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        kk = f"{prefix}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_metric_dict(v, prefix=f"{kk}."))
        else:
            out[kk] = v
    return out


def read_seed_metrics(seed_dir: Path) -> dict:
    metrics_path = seed_dir / "survival_metrics.json"
    if not metrics_path.exists():
        hits = list(seed_dir.rglob("survival_metrics.json"))
        if hits:
            metrics_path = hits[0]
    if not metrics_path.exists():
        return {"metrics_missing": 1}
    with open(metrics_path, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)
    test_metrics = all_metrics.get("test", all_metrics)
    return flatten_metric_dict(test_metrics)


def summarize_multiseed_results(output_root: Path, seeds: List[int]):
    rows = []
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        row = {"seed": int(seed), "seed_dir": str(seed_dir)}
        row.update(read_seed_metrics(seed_dir))
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = output_root / "survival_seed_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    numeric_cols = []
    for c in df.columns:
        if c in {"seed", "seed_dir"}:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            numeric_cols.append(c)
            df[c] = vals
    stat_rows = []
    for c in numeric_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        stat_rows.append({
            "metric": c,
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        })
    stat_df = pd.DataFrame(stat_rows)
    stat_csv = output_root / "survival_seed_summary_mean_std.csv"
    stat_df.to_csv(stat_csv, index=False, encoding="utf-8-sig")

    key_metrics = [
        "HCM_cindex", "DCM_cindex", "overall_cindex",
        "HCM_logrank_p", "DCM_logrank_p",
        "horizon_metrics.overall_365d_auc", "horizon_metrics.overall_730d_auc",
        "horizon_metrics.overall_1095d_auc", "horizon_metrics.overall_1460d_auc",
        "horizon_metrics.overall_1825d_auc",
        "horizon_metrics.HCM_365d_auc", "horizon_metrics.DCM_365d_auc",
    ]
    md_lines = ["# Multi-seed survival summary", "", f"Seeds: {seeds}", "", "| Metric | mean ± std | min | max |", "|---|---:|---:|---:|"]
    for m in key_metrics:
        r = stat_df[stat_df["metric"] == m]
        if len(r) == 0:
            continue
        rr = r.iloc[0]
        md_lines.append(f"| {m} | {rr['mean']:.4f} ± {rr['std']:.4f} | {rr['min']:.4f} | {rr['max']:.4f} |")
    (output_root / "survival_seed_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print0("\n========== Multi-seed summary saved ==========")
    print0(f"{out_csv}")
    print0(f"{stat_csv}")
    print0(str(output_root / "survival_seed_summary.md"))
    if len(stat_df):
        print0("\nKey metrics:")
        for m in key_metrics:
            r = stat_df[stat_df["metric"] == m]
            if len(r):
                rr = r.iloc[0]
                print0(f"  {m}: {rr['mean']:.4f} ± {rr['std']:.4f}")


def tail_text(path: Path, n: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def run_multiseed_parent():
    seeds = parse_seed_list(RUN_CONFIG.get("multiseed_seeds", "42,123,2024,3407,777"))
    output_root = make_multiseed_output_root()
    ensure_dir(output_root)
    with open(output_root / "multiseed_parent_config.json", "w", encoding="utf-8") as f:
        json.dump(RUN_CONFIG, f, ensure_ascii=False, indent=2)

    visible = str(RUN_CONFIG.get("visible_gpus", "") or "").strip()
    nproc = len([x for x in visible.split(",") if x.strip()]) if visible else 1
    gpu_mode = str(RUN_CONFIG.get("gpu_mode", "single")).lower()
    script_path = str(Path(__file__).resolve())

    print0("========== Standalone multi-seed survival runner ==========")
    print0(f"Script: {script_path}")
    print0(f"Output root: {output_root}")
    print0(f"Seeds: {seeds}")
    print0(f"GPU mode: {gpu_mode} | visible_gpus={visible}")
    print0("This is a single-file runner: it launches this same script for each seed, not an external old script.")

    failed = []
    for seed in seeds:
        seed = int(seed)
        seed_dir = output_root / f"seed_{seed}"
        ensure_dir(seed_dir)
        log_path = seed_dir / "runner_stdout_stderr.log"
        overrides = {
            "multiseed_mode": False,
            "seed": seed,
            "output_dir": str(output_root),
            "experiment_name": f"seed_{seed}",
            "run_id": f"seed_{seed}",
            "auto_launch_ddp": False,
            "fine_tune_mode": str(RUN_CONFIG.get("fine_tune_mode", "hybrid_cross_attention")),
            "feature_lr": float(RUN_CONFIG.get("feature_lr", 3e-5)),
            "survival_dropout": float(RUN_CONFIG.get("survival_dropout", 0.20)),
            "hcm_loss_weight": float(RUN_CONFIG.get("hcm_loss_weight", 1.0)),
            "dcm_loss_weight": float(RUN_CONFIG.get("dcm_loss_weight", 1.5)),
            "early_stop_metric": str(RUN_CONFIG.get("early_stop_metric", "dcm_weighted")),
            "hcm_monitor_weight": float(RUN_CONFIG.get("hcm_monitor_weight", 0.35)),
            "dcm_monitor_weight": float(RUN_CONFIG.get("dcm_monitor_weight", 0.65)),
            "hcm_cross_gate_init": float(RUN_CONFIG.get("hcm_cross_gate_init", 0.50)),
            "dcm_cross_gate_init": float(RUN_CONFIG.get("dcm_cross_gate_init", 0.25)),
            "hcm_hidden_dim": int(RUN_CONFIG.get("hcm_hidden_dim", 128)),
            "dcm_hidden_dim": int(RUN_CONFIG.get("dcm_hidden_dim", 256)),
            "hcm_dropout": float(RUN_CONFIG.get("hcm_dropout", 0.20)),
            "dcm_dropout": float(RUN_CONFIG.get("dcm_dropout", 0.15)),
            "train_horizon_heads": False,
            "horizon_loss_weight": 0.0,
        }
        env = os.environ.copy()
        env["CMR_MULTI_SEED_CHILD"] = "1"
        env["CMR_RUN_ID"] = f"seed_{seed}"
        env["CMR_RUN_CONFIG_OVERRIDES"] = json.dumps(overrides, ensure_ascii=False)
        if visible:
            env["CUDA_VISIBLE_DEVICES"] = visible
        if gpu_mode == "ddp" and nproc > 1:
            # Use a unique free port for every seed. This avoids: EADDRINUSE, address already in use, port 29500.
            master_port = resolve_master_port(0)
            env["MASTER_PORT"] = str(master_port)
            cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", f"--master_port={master_port}", script_path]
        else:
            master_port = 0
            cmd = [sys.executable, script_path]

        print0(f"\n========== Running seed={seed} ==========")
        print0(f"Output: {seed_dir}")
        print0((f"DDP master_port: {master_port}" if master_port else "DDP master_port: N/A"))
        print0("Command: " + " ".join(cmd))
        start = time.time()
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_f.write(line)
                log_f.flush()
            code = proc.wait()
        elapsed = time.time() - start
        print0(f"seed={seed} finished with code={code}, elapsed={elapsed:.1f}s")
        if code != 0:
            failed.append(seed)
            print0(f"[ERROR] seed={seed} failed. Last log lines:\n{tail_text(log_path, 120)}")
            if not bool(RUN_CONFIG.get("multiseed_continue_on_error", False)):
                raise subprocess.CalledProcessError(code, cmd)

    summarize_multiseed_results(output_root, seeds)
    if failed:
        print0(f"[WARN] Failed seeds: {failed}")


# =============================================================================
# Independent HCM/DCM runner: run two disease-specific models from this same file
# =============================================================================
def should_run_both_independent_parent() -> bool:
    if os.environ.get("CMR_BOTH_CHILD", "0") == "1":
        return False
    if os.environ.get("CMR_MULTI_SEED_CHILD", "0") == "1":
        return False
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        return False
    mode = str(RUN_CONFIG.get("disease_mode", "") or "").strip().lower()
    return mode in {"both", "both_independent", "independent", "hcm_dcm"}


def make_both_output_root() -> Path:
    root = str(RUN_CONFIG.get("both_output_root", "auto") or "auto").strip()
    if root.lower() in {"", "auto", "none"}:
        script_stem = Path(sys.argv[0]).stem
        return Path.cwd() / f"{script_stem}_results_{now_string()}"
    return Path(root)



def child_overrides_for_disease(mode: str, output_root: Path) -> dict:
    mode = str(mode).strip()
    base = {
        "disease_mode": mode,
        "output_dir": str(output_root),
        "multiseed_mode": False,
        "both_output_root": str(output_root),
        "force_reextract_features": bool(RUN_CONFIG.get("force_reextract_features", False)),
        "diagnosis_encoder_agg": str(RUN_CONFIG.get("diagnosis_encoder_agg", "hier_mean")),
        "survival_agg": str(RUN_CONFIG.get("survival_agg", RUN_CONFIG.get("agg", "hier_attn"))),
        "agg": str(RUN_CONFIG.get("survival_agg", RUN_CONFIG.get("agg", "hier_attn"))),
    }
    if mode == "HCM_only":
        base.update({
            "experiment_name": "hcm_only_hybrid_crossattn_fourviews",
            "hcm_loss_weight": 1.0,
            "dcm_loss_weight": 0.0,
            "early_stop_metric": "hcm",
            "hcm_monitor_weight": 1.0,
            "dcm_monitor_weight": 0.0,
            "hcm_hidden_dim": int(RUN_CONFIG.get("hcm_hidden_dim", 128)),
            "hcm_dropout": float(RUN_CONFIG.get("hcm_dropout", 0.20)),
        })
    elif mode == "DCM_only":
        base.update({
            "experiment_name": "dcm_only_hybrid_crossattn_fourviews",
            "hcm_loss_weight": 0.0,
            "dcm_loss_weight": 1.0,
            "early_stop_metric": "dcm",
            "hcm_monitor_weight": 0.0,
            "dcm_monitor_weight": 1.0,
            "dcm_hidden_dim": int(RUN_CONFIG.get("dcm_hidden_dim", 256)),
            "dcm_dropout": float(RUN_CONFIG.get("dcm_dropout", 0.15)),
        })
    else:
        raise ValueError(f"Unsupported independent disease mode: {mode}")
    return base


def find_survival_metrics(exp_dir: Path) -> Optional[Path]:
    direct = exp_dir / "survival_metrics.json"
    if direct.exists():
        return direct
    hits = list(exp_dir.rglob("survival_metrics.json"))
    if hits:
        return hits[0]
    return None


def flatten_dict(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        kk = f"{prefix}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=f"{kk}."))
        else:
            out[kk] = v
    return out


def summarize_independent_hcm_dcm(output_root: Path, modes: List[str]):
    rows = []
    for mode in modes:
        exp_name = "hcm_only_hybrid_crossattn_fourviews" if mode == "HCM_only" else "dcm_only_hybrid_crossattn_fourviews"
        exp_dir = output_root / exp_name
        mpath = find_survival_metrics(exp_dir)
        row = {"mode": mode, "experiment_dir": str(exp_dir), "metrics_path": str(mpath) if mpath else ""}
        if mpath and mpath.exists():
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    metrics_all = json.load(f)
                test_metrics = metrics_all.get("test", metrics_all)
                row.update(flatten_dict(test_metrics))
            except Exception as e:
                row["read_error"] = str(e)
        else:
            row["metrics_missing"] = 1
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = output_root / "independent_hcm_dcm_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    compact_rows = []
    for _, r in df.iterrows():
        mode = r.get("mode", "")
        disease = "HCM" if mode == "HCM_only" else "DCM"
        compact_rows.append({
            "mode": mode,
            "cindex": r.get(f"{disease}_cindex", r.get("overall_cindex", np.nan)),
            "logrank_p": r.get(f"{disease}_logrank_p", np.nan),
            "auc_1y": r.get(f"horizon_metrics.{disease}_365d_auc", r.get("horizon_metrics.overall_365d_auc", np.nan)),
            "auc_2y": r.get(f"horizon_metrics.{disease}_730d_auc", r.get("horizon_metrics.overall_730d_auc", np.nan)),
            "auc_3y": r.get(f"horizon_metrics.{disease}_1095d_auc", r.get("horizon_metrics.overall_1095d_auc", np.nan)),
            "auc_4y": r.get(f"horizon_metrics.{disease}_1460d_auc", r.get("horizon_metrics.overall_1460d_auc", np.nan)),
            "auc_5y": r.get(f"horizon_metrics.{disease}_1825d_auc", r.get("horizon_metrics.overall_1825d_auc", np.nan)),
            "experiment_dir": r.get("experiment_dir", ""),
        })
    compact = pd.DataFrame(compact_rows)
    compact_path = output_root / "independent_hcm_dcm_summary_compact.csv"
    compact.to_csv(compact_path, index=False, encoding="utf-8-sig")

    md_path = output_root / "independent_hcm_dcm_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Independent HCM/DCM prognosis summary\n\n")
        try:
            f.write(compact.to_markdown(index=False))
        except Exception:
            f.write(compact.to_csv(index=False))
        f.write("\n")
    print0("\n========== Independent HCM/DCM summary saved ==========")
    print0(str(csv_path))
    print0(str(compact_path))
    print0(str(md_path))
    print0("\nCompact summary:")
    print0(compact.to_string(index=False))


def run_both_independent_parent():
    output_root = make_both_output_root()
    ensure_dir(output_root)
    modes = parse_csv_list(RUN_CONFIG.get("both_independent_modes", "HCM_only,DCM_only"))
    modes = [m for m in modes if m in {"HCM_only", "DCM_only"}]
    if not modes:
        modes = ["HCM_only", "DCM_only"]
    print0("========== Independent HCM/DCM survival runner ==========")
    print0(f"Script: {Path(sys.argv[0]).resolve()}")
    print0(f"Output root: {output_root}")
    print0(f"Modes: {modes}")
    print0("This is a single-file runner: it launches this same script for HCM_only and DCM_only, not external old scripts.")
    with open(output_root / "parent_run_config.json", "w", encoding="utf-8") as f:
        json.dump(RUN_CONFIG, f, ensure_ascii=False, indent=2)

    failed = []
    for mode in modes:
        print0(f"\n========== Running {mode} ==========")
        overrides = child_overrides_for_disease(mode, output_root)
        env = os.environ.copy()
        env["CMR_BOTH_CHILD"] = "1"
        env["CMR_RUN_CONFIG_OVERRIDES"] = json.dumps(overrides, ensure_ascii=False)
        cmd = [sys.executable, sys.argv[0]]
        log_dir = output_root / ("hcm_only_hybrid_crossattn_fourviews" if mode == "HCM_only" else "dcm_only_hybrid_crossattn_fourviews")
        ensure_dir(log_dir)
        log_path = log_dir / "parent_stdout_stderr.log"
        print0("Command:", " ".join(cmd))
        print0("Overrides:", json.dumps(overrides, ensure_ascii=False))
        start = time.time()
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_f.write(line)
                log_f.flush()
            code = proc.wait()
        elapsed = time.time() - start
        print0(f"{mode} finished with code={code}, elapsed={elapsed:.1f}s")
        if code != 0:
            failed.append(mode)
            print0(f"[ERROR] {mode} failed. See log: {log_path}")
            if not bool(RUN_CONFIG.get("both_continue_on_error", False)):
                raise subprocess.CalledProcessError(code, cmd)
    summarize_independent_hcm_dcm(output_root, modes)
    if failed:
        print0(f"[WARN] Failed modes: {failed}")



# =============================================================================
# Final all-in-one HCM/DCM retrain workflow
# =============================================================================
# This section intentionally does NOT call older scripts with subprocess.
# It reuses the classes/functions defined above in this same file and runs:
#   1) HCM four-view hybrid_cross_attention Cox retraining
#   2) DCM four-view hybrid_cross_attention Cox retraining
#   3) DCM LAX-only (2CH+3CH+4CH) fixed-feature PCA128 Ridge-Cox retraining

import copy

FINAL_FIXED_SPLIT_DIR = None  # deprecated: use args.split_source_dir from top RUN_CONFIG
FINAL_OUT_STEM = "20260721_final_hcm_dcm_allinone_retrain_v3_lax_final"


def _query_free_gpu_with_nvidia_smi() -> str:
    """Return physical GPU id with largest free memory. Empty string if unavailable."""
    try:
        import subprocess as _sp
        out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=_sp.DEVNULL,
        )
        best_idx, best_mem = "", -1
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            idx, mem = parts[0], int(float(parts[1]))
            if mem > best_mem:
                best_idx, best_mem = idx, mem
        if best_idx:
            print(f"[GPU auto] selected physical GPU {best_idx} with free memory {best_mem} MiB")
        return best_idx
    except Exception as e:
        print(f"[WARN] GPU auto selection failed: {e}. Falling back to current CUDA visibility.")
        return ""


def configure_runtime_gpu(visible_gpus: str, device_arg: str = "auto") -> torch.device:
    """Configure a single visible GPU before CUDA context is created."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    visible = str(visible_gpus or "").strip()
    if str(device_arg).lower() == "cpu" or not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return torch.device("cpu")
    if visible.lower() in {"", "auto", "none"}:
        picked = _query_free_gpu_with_nvidia_smi()
        if picked:
            os.environ["CUDA_VISIBLE_DEVICES"] = picked
            visible = picked
    else:
        # Use only the first GPU for this all-in-one final retrain.
        first = [x.strip() for x in visible.split(",") if x.strip()][0]
        os.environ["CUDA_VISIBLE_DEVICES"] = first
        visible = first
    # Important: after CUDA_VISIBLE_DEVICES is set, logical cuda:0 is the selected physical GPU.
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def clone_args(args):
    return argparse.Namespace(**vars(args))


def configure_disease_args(base_args, disease_mode: str, out_root: Path, experiment_name: str):
    args = clone_args(base_args)
    args.gpu_mode = "single"
    args.auto_launch_ddp = False
    args.visible_gpus = str(getattr(base_args, "visible_gpus", "") or "")
    args.output_dir = str(out_root)
    args.experiment_name = experiment_name
    args.split_source_dir = str(getattr(base_args, "split_source_dir", RUN_CONFIG.get("split_source_dir", "")))
    args.disease_mode = disease_mode
    args.both_independent_modes = disease_mode
    args.multiseed_mode = False
    args.force_reextract_features = bool(getattr(base_args, "force_reextract_features", True))
    args.batch_size = int(getattr(base_args, "batch_size", 1))
    args.workers = int(getattr(base_args, "workers", 2))
    args.diagnosis_encoder_agg = canonicalize_agg(getattr(args, "diagnosis_encoder_agg", "hier_mean"))
    args.survival_agg = canonicalize_agg(getattr(args, "survival_agg", getattr(args, "agg", "hier_attn")))
    args.agg = args.survival_agg
    args.fine_tune_mode = "hybrid_cross_attention"
    args.train_horizon_heads = False
    args.horizon_loss_weight = 0.0
    args.eval_only = False
    if disease_mode == "HCM_only":
        args.hcm_loss_weight = 1.0
        args.dcm_loss_weight = 0.0
        args.early_stop_metric = "hcm"
        args.hcm_monitor_weight = 1.0
        args.dcm_monitor_weight = 0.0
        args.hcm_dropout = float(getattr(args, "hcm_dropout", 0.20))
        args.hcm_hidden_dim = int(getattr(args, "hcm_hidden_dim", 128))
    elif disease_mode == "DCM_only":
        args.hcm_loss_weight = 0.0
        args.dcm_loss_weight = 1.0
        args.early_stop_metric = "dcm"
        args.hcm_monitor_weight = 0.0
        args.dcm_monitor_weight = 1.0
        args.dcm_dropout = float(getattr(args, "dcm_dropout", 0.30))
        args.dcm_hidden_dim = int(getattr(args, "dcm_hidden_dim", 128))
    else:
        raise ValueError(f"Unsupported disease_mode for final retrain: {disease_mode}")
    return args


def collect_split_samples_for_args(args):
    selected_views = parse_csv_list(args.selected_views)
    validate_views(selected_views)
    required_views = parse_csv_list(args.cohort_required_views)
    validate_views(required_views)
    slice_plan = parse_slice_plan(args.slice_plan)

    base_dataset = MultiViewCardiacNiftiDataset(
        root_dir=args.data_path,
        selected_views=selected_views,
        slice_plan=slice_plan,
        num_frames=args.num_frames,
        image_size=args.image_size,
        samples=None,
        mode="eval",
        required_views=required_views,
        allow_missing_selected_views=args.allow_missing_selected_views,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir if args.cache_dir else None,
        min_frames_per_slice=args.min_frames_per_slice,
        verbose=True,
    )
    prog_df = read_prognosis_table(args)
    prog_samples = attach_prognosis_to_samples(base_dataset.samples, prog_df, args)
    if len(prog_samples) == 0:
        raise RuntimeError("No usable HCM/DCM prognosis samples after matching. Check patient_id and prognosis_csv.")
    if args.split_source_dir:
        train_samples, val_samples, test_samples = split_from_existing_csv(prog_samples, args.split_source_dir)
    else:
        train_samples, val_samples, test_samples = stratified_split_survival_samples(prog_samples, args.train_ratio, args.val_ratio, args.seed)
    train_samples = filter_samples_by_disease_mode(train_samples, args.disease_mode)
    val_samples = filter_samples_by_disease_mode(val_samples, args.disease_mode)
    test_samples = filter_samples_by_disease_mode(test_samples, args.disease_mode)
    if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError(f"No samples remain after disease_mode={args.disease_mode}.")
    return selected_views, required_views, slice_plan, {"train": train_samples, "val": val_samples, "test": test_samples}


def extract_or_load_features_for_disease(args, selected_views, required_views, slice_plan, split_to_samples, out_dir: Path, device: torch.device):
    cfg_hash = config_hash_for_features(args, selected_views, slice_plan)
    feature_dir = out_dir / f"features_{cfg_hash}"
    ensure_dir(feature_dir)
    need_extract = bool(getattr(args, "force_reextract_features", True))
    for split in ["train", "val", "test"]:
        if not (feature_dir / f"{split}_features.npz").exists() or not (feature_dir / f"{split}_meta.csv").exists():
            need_extract = True
    if need_extract:
        print0(f"Extracting frozen Video Swin clip features into: {feature_dir}")
        model = MultiViewVideoSwinFeatureExtractor(
            selected_views=selected_views,
            slice_plan=slice_plan,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            agg=getattr(args, "diagnosis_encoder_agg", getattr(args, "agg", "hier_mean")),
            dropout=args.dropout,
            backbone_chunk_size=args.backbone_chunk_size,
        ).to(device)
        if args.classification_ckpt:
            model.load_classification_checkpoint(args.classification_ckpt)
        model.eval()
        feature_rows_by_split = {}
        for split in ["train", "val", "test"]:
            ds = MultiViewCardiacNiftiDataset(
                root_dir=args.data_path,
                selected_views=selected_views,
                slice_plan=slice_plan,
                num_frames=args.num_frames,
                image_size=args.image_size,
                samples=split_to_samples[split],
                mode="eval",
                required_views=required_views,
                allow_missing_selected_views=args.allow_missing_selected_views,
                use_cache=args.use_cache,
                cache_dir=args.cache_dir if args.cache_dir else None,
                min_frames_per_slice=args.min_frames_per_slice,
                verbose=False,
            )
            loader = build_dataloader(ds, args, distributed=False, shuffle=False)
            rows = extract_features(model, loader, device, args)
            save_feature_rows(rows, feature_dir / f"{split}_features.npz", feature_dir / f"{split}_meta.csv")
            feature_rows_by_split[split] = rows
            print0(f"Saved {split} features: {len(rows)}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print0(f"Using cached features: {feature_dir}")
        feature_rows_by_split = {}
        for split in ["train", "val", "test"]:
            feature_rows_by_split[split] = load_feature_rows(feature_dir / f"{split}_features.npz", feature_dir / f"{split}_meta.csv")
    return feature_rows_by_split, feature_dir


def run_neural_disease_allinone(base_args, disease_mode: str, root: Path, experiment_name: str, device: torch.device):
    args = configure_disease_args(base_args, disease_mode, root, experiment_name)
    selected_views, required_views, slice_plan, split_to_samples = collect_split_samples_for_args(args)
    out_dir = root / experiment_name
    ensure_dir(out_dir)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print0("\n" + "=" * 80)
    print0(f"Running all-in-one neural retrain: {disease_mode} -> {out_dir}")
    print0("Survival split summary:")
    print0("  Train:", summarize_survival_samples(split_to_samples["train"]))
    print0("  Val:  ", summarize_survival_samples(split_to_samples["val"]))
    print0("  Test: ", summarize_survival_samples(split_to_samples["test"]))
    save_survival_split_csv(split_to_samples["train"], out_dir / "split_train_survival.csv")
    save_survival_split_csv(split_to_samples["val"], out_dir / "split_val_survival.csv")
    save_survival_split_csv(split_to_samples["test"], out_dir / "split_test_survival.csv")

    feature_rows_by_split, feature_dir = extract_or_load_features_for_disease(
        args, selected_views, required_views, slice_plan, split_to_samples, out_dir, device
    )
    save_event_type_summary(feature_rows_by_split, out_dir)
    if "clip_features" in feature_rows_by_split["train"][0]:
        feature_dim = int(feature_rows_by_split["train"][0]["clip_features"].shape[-1])
    else:
        feature_dim = int(feature_rows_by_split["train"][0]["feature"].shape[0])
    metrics = train_survival_heads(
        feature_rows_by_split["train"],
        feature_rows_by_split["val"],
        feature_rows_by_split["test"],
        feature_dim=feature_dim,
        args=args,
        out_dir=out_dir,
    )
    print0(f"\n{disease_mode} test metrics:")
    print0(json.dumps(metrics.get("test", {}), ensure_ascii=False, indent=2))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "args": args,
        "out_dir": out_dir,
        "feature_dir": feature_dir,
        "rows": feature_rows_by_split,
        "metrics": metrics,
        "selected_views": selected_views,
    }


class FinalLinearCox(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).view(-1)


def row_array(rows: List[Dict], key: str, dtype=float):
    return np.asarray([r[key] for r in rows], dtype=dtype)


def dcm_lax_feature_matrix(rows: List[Dict], selected_views: List[str]) -> np.ndarray:
    # LAX-only = long-axis Cine views: 2CH + 3CH + 4CH.
    # Each view is first averaged across its valid clips, then the three view-level vectors are concatenated.
    lax_view_names = ["Cine2CH", "Cine3CH", "Cine4CH"]
    lax_view_ids = [selected_views.index(v) for v in lax_view_names]
    feats = []
    for r in rows:
        clip = np.asarray(r["clip_features"], dtype=np.float32)
        mask = np.asarray(r["clip_mask"], dtype=bool)
        vids = np.asarray(r["clip_view_ids"], dtype=np.int64)
        view_feats = []
        for vid in lax_view_ids:
            m = mask & (vids == vid)
            if m.sum() <= 0:
                view_feats.append(np.zeros((clip.shape[-1],), dtype=np.float32))
            else:
                view_feats.append(clip[m].mean(axis=0).astype(np.float32))
        feats.append(np.concatenate(view_feats, axis=0))
    return np.stack(feats, axis=0).astype(np.float32)


def standardize_and_pca(Xtr, Xva, Xte, pca_dim: int, seed: int):
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    Xtr_s = (Xtr - mean) / std
    Xva_s = (Xva - mean) / std
    Xte_s = (Xte - mean) / std
    if pca_dim and pca_dim < Xtr_s.shape[1]:
        from sklearn.decomposition import PCA
        svd_solver = "randomized" if pca_dim < min(Xtr_s.shape) else "auto"
        pca = PCA(n_components=pca_dim, random_state=seed, svd_solver=svd_solver)
        Xtr_p = pca.fit_transform(Xtr_s).astype(np.float32)
        Xva_p = pca.transform(Xva_s).astype(np.float32)
        Xte_p = pca.transform(Xte_s).astype(np.float32)
        state = {"mean": mean, "std": std, "pca_components": pca.components_, "pca_mean": pca.mean_, "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_))}
        return Xtr_p, Xva_p, Xte_p, state
    return Xtr_s.astype(np.float32), Xva_s.astype(np.float32), Xte_s.astype(np.float32), {"mean": mean, "std": std}


def fit_final_ridge_cox(Xtr, ytr_t, ytr_e, Xva, yva_t, yva_e, device: torch.device, l2=1e-4, lr=1e-2, max_epochs=1200, patience=120):
    model = FinalLinearCox(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = CoxPHLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ttr_t = torch.tensor(ytr_t, dtype=torch.float32, device=device)
    etr_t = torch.tensor(ytr_e, dtype=torch.float32, device=device)
    mtr_t = torch.ones_like(etr_t, dtype=torch.bool)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        risk = model(Xtr_t)
        loss = loss_fn(risk, ttr_t, etr_t, mtr_t)
        if l2 > 0:
            w = model.linear.weight.view(-1)
            loss = loss + float(l2) * torch.sum(w * w)
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_risk = model(Xva_t).detach().cpu().numpy().astype(float)
            val_c = concordance_index(yva_t, yva_e, val_risk)
            if val_c > best_val + 1e-6:
                best_val = float(val_c)
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 10
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_score": best_val, "best_epoch": best_epoch}


def predict_final_linear(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(float)


def eval_array_metrics(timev, event, risk, threshold):
    out = {
        "cindex": concordance_index(timev, event, risk),
        "logrank_p": logrank_p_value(timev, event, (risk >= threshold).astype(int)),
    }
    # Use existing horizon evaluator by constructing lightweight rows.
    rows = []
    for i, (t, e) in enumerate(zip(timev, event)):
        rows.append({"patient_id": str(i), "class_name": "DCM", "label": CLASS_TO_IDX["DCM"], "event": int(e), "time_to_event": float(t), "censored": int(1 - int(e)), "event_type": ""})
    horizons_days = parse_horizons_days(str(RUN_CONFIG.get("eval_horizons_days", "365,730,1095,1460,1825")))
    hmet = evaluate_horizon_scores(rows, np.asarray(risk, dtype=float), horizons_days, thresholds={}, score_name="cox_risk")
    for k, v in hmet.items():
        if k.startswith("overall_") and k.endswith("_auc"):
            # overall_365d_auc -> auc_1y
            days = int(k.split("_")[1].replace("d", ""))
            year = int(round(days / 365))
            out[f"auc_{year}y"] = float(v) if v is not None else float("nan")
    return out


def save_final_risk_predictions(rows: List[Dict], risk: np.ndarray, threshold: float, path: Path):
    recs = []
    for r, rr in zip(rows, risk):
        recs.append({
            "patient_id": r.get("patient_id", ""),
            "class_name": r.get("class_name", ""),
            "label": r.get("label", ""),
            "event": int(r.get("event", 0)),
            "time_to_event": float(r.get("time_to_event", 0)),
            "censored": int(r.get("censored", 0)),
            "event_type": r.get("event_type", ""),
            "risk": float(rr),
            "risk_threshold": float(threshold),
            "risk_group": "high" if float(rr) >= threshold else "low",
        })
    pd.DataFrame(recs).to_csv(path, index=False, encoding="utf-8-sig")


def train_dcm_lax_pca128_ridge(dcm_result: Dict, root: Path, base_args, device: torch.device):
    out_dir = root / "dcm_lax_pca128_ridgecox_retrain"
    ensure_dir(out_dir)
    rows = dcm_result["rows"]
    selected_views = dcm_result["selected_views"]
    Xtr0 = dcm_lax_feature_matrix(rows["train"], selected_views)
    Xva0 = dcm_lax_feature_matrix(rows["val"], selected_views)
    Xte0 = dcm_lax_feature_matrix(rows["test"], selected_views)
    Xtr, Xva, Xte, state = standardize_and_pca(Xtr0, Xva0, Xte0, pca_dim=128, seed=int(getattr(base_args, "seed", 42)))
    ytr_t = row_array(rows["train"], "time_to_event", float)
    ytr_e = row_array(rows["train"], "event", int)
    yva_t = row_array(rows["val"], "time_to_event", float)
    yva_e = row_array(rows["val"], "event", int)
    yte_t = row_array(rows["test"], "time_to_event", float)
    yte_e = row_array(rows["test"], "event", int)
    # CPU is usually sufficient and avoids GPU contention.
    fit_device = torch.device("cpu") if str(getattr(base_args, "traditional_device", "cpu")).lower() == "cpu" else device
    model, fit_info = fit_final_ridge_cox(Xtr, ytr_t, ytr_e, Xva, yva_t, yva_e, fit_device, l2=1e-4)
    risk_tr = predict_final_linear(model, Xtr, fit_device)
    risk_va = predict_final_linear(model, Xva, fit_device)
    risk_te = predict_final_linear(model, Xte, fit_device)
    threshold = float(np.median(risk_tr))
    metrics = {
        "best_epoch": fit_info["best_epoch"],
        "best_val_score": fit_info["best_val_score"],
        "feature_family": "LAX-only concat view mean (2CH+3CH+4CH)",
        "pca_dim": 128,
        "l2": 1e-4,
        "pca_explained_variance_ratio_sum": state.get("explained_variance_ratio_sum", None),
        "train": eval_array_metrics(ytr_t, ytr_e, risk_tr, threshold),
        "val": eval_array_metrics(yva_t, yva_e, risk_va, threshold),
        "test": eval_array_metrics(yte_t, yte_e, risk_te, threshold),
    }
    save_final_risk_predictions(rows["train"], risk_tr, threshold, out_dir / "risk_predictions_train.csv")
    save_final_risk_predictions(rows["val"], risk_va, threshold, out_dir / "risk_predictions_val.csv")
    save_final_risk_predictions(rows["test"], risk_te, threshold, out_dir / "risk_predictions_test.csv")
    with open(out_dir / "survival_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({"feature_family": "LAX-only (Cine2CH+Cine3CH+Cine4CH)", "pca_dim": 128, "l2": 1e-4, "note": "validation-selected DCM-specific traditional Cox candidate retrained from current DCM features"}, f, ensure_ascii=False, indent=2)
    print0("\nDCM LAX-only PCA128 Ridge-Cox test metrics:")
    print0(json.dumps(metrics["test"], ensure_ascii=False, indent=2))
    return {"out_dir": out_dir, "metrics": metrics}


def extract_compact_record(name: str, disease: str, result: Dict):
    metrics = result["metrics"]
    test = metrics.get("test", {})
    if disease == "HCM":
        cindex = test.get("HCM_cindex", test.get("overall_cindex", float("nan")))
        p = test.get("HCM_logrank_p", test.get("overall_logrank_p", float("nan")))
        n = test.get("HCM_n", test.get("overall_n", float("nan")))
        events = test.get("HCM_events", test.get("overall_events", float("nan")))
    elif disease == "DCM" and "DCM_cindex" in test:
        cindex = test.get("DCM_cindex", test.get("overall_cindex", float("nan")))
        p = test.get("DCM_logrank_p", test.get("overall_logrank_p", float("nan")))
        n = test.get("DCM_n", test.get("overall_n", float("nan")))
        events = test.get("DCM_events", test.get("overall_events", float("nan")))
    else:
        cindex = test.get("cindex", float("nan"))
        p = test.get("logrank_p", float("nan"))
        n = 168
        events = 40
    rec = {"model": name, "disease": disease, "cindex": cindex, "logrank_p": p, "n": n, "events": events, "experiment_dir": str(result["out_dir"])}
    # Horizon AUCs.
    hdict = None
    if "horizon_metrics" in test:
        hdict = test["horizon_metrics"]
    for y, days in [(1, 365), (2, 730), (3, 1095), (4, 1460), (5, 1825)]:
        val = float("nan")
        if hdict:
            keys = [f"{disease}_{days}d_auc", f"overall_{days}d_auc"]
            for k in keys:
                if k in hdict and hdict[k] is not None:
                    val = hdict[k]
                    break
        elif f"auc_{y}y" in test:
            val = test[f"auc_{y}y"]
        rec[f"auc_{y}y"] = val
    rec["best_epoch"] = metrics.get("best_epoch", float("nan"))
    rec["best_val_score"] = metrics.get("best_val_score", float("nan"))
    return rec



# =============================================================================
# Clean final HCM prognosis pipeline: feature re-extraction + two low-capacity Cox
# heads + fixed diagnosis-anchored risk-level correction.
#
# This section is intentionally written as normal Python functions. It does not
# embed older scripts as source strings and does not import previous v18/v20/v21
# result code. The feature extraction utilities above are defined directly in
# this same file.
# =============================================================================

# FINAL_HCM_CONFIG was removed intentionally.
# All shared and final HCM settings now live in RUN_CONFIG at the top of this file.


def hcm_parse_strs(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").split(",") if x.strip()]


def hcm_parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s or "").split(",") if x.strip()]


def hcm_parse_horizons(s: str) -> List[int]:
    return hcm_parse_ints(s)


def hcm_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def hcm_final_root(args) -> Path:
    out = str(getattr(args, "output_dir", "auto") or "auto").strip()
    exp = str(getattr(args, "experiment_name", "final_hcm_diagnosis_anchored_lambda045_clean_reextract"))
    if out.lower() in {"", "auto", "none"}:
        root = Path.cwd() / f"20260724_final_hcm_lambda045_clean_results_{hcm_now()}" / exp
    else:
        root = Path(out) / exp
    return ensure_dir(root)


def hcm_seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def hcm_choose_head_device(name: str) -> torch.device:
    name = str(name or "auto").lower()
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


HCM_VIEW_MAP = {
    0: "2ch", 1: "3ch", 2: "4ch", 3: "sax",
    "0": "2ch", "1": "3ch", "2": "4ch", "3": "sax",
    "Cine2CH": "2ch", "Cine3CH": "3ch", "Cine4CH": "4ch", "CineSAX": "sax",
    "2ch": "2ch", "3ch": "3ch", "4ch": "4ch", "sax": "sax",
}


def hcm_view_name(x) -> str:
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if isinstance(x, np.generic):
        x = x.item()
    return HCM_VIEW_MAP.get(x, HCM_VIEW_MAP.get(str(x), str(x).lower()))


def hcm_row_pid(row: Dict) -> str:
    for key in ["patient_id", "pid", "id", "patient"]:
        if key in row:
            return str(row[key])
    return ""


def hcm_row_time(row: Dict) -> float:
    for key in ["time_to_event", "time", "duration", "survival_time"]:
        if key in row:
            return float(row[key])
    raise KeyError(f"Cannot find survival time. Available row keys={list(row.keys())}")


def hcm_row_event(row: Dict) -> int:
    for key in ["event", "events", "event_observed", "status"]:
        if key in row:
            return int(float(row[key]))
    raise KeyError(f"Cannot find event indicator. Available row keys={list(row.keys())}")


def hcm_clip_features(row: Dict) -> np.ndarray:
    arr = np.asarray(row["clip_features"], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"clip_features must be [n_clips, dim], got {arr.shape}, patient={hcm_row_pid(row)}")
    return arr


def hcm_clip_mask(row: Dict) -> np.ndarray:
    feats = hcm_clip_features(row)
    mask = row.get("clip_mask", None)
    if mask is None:
        return np.ones((feats.shape[0],), dtype=bool)
    mask = np.asarray(mask).astype(bool).reshape(-1)
    if len(mask) != feats.shape[0]:
        raise ValueError(f"clip_mask length mismatch: {len(mask)} vs {feats.shape[0]}, patient={hcm_row_pid(row)}")
    return mask


def hcm_clip_view_ids(row: Dict) -> np.ndarray:
    feats = hcm_clip_features(row)
    vids = row.get("clip_view_ids", None)
    if vids is None:
        if feats.shape[0] == 17:
            return np.asarray([0] * 3 + [1] * 3 + [2] * 3 + [3] * 8)
        return np.zeros((feats.shape[0],), dtype=int)
    vids = np.asarray(vids).reshape(-1)
    if len(vids) != feats.shape[0]:
        raise ValueError(f"clip_view_ids length mismatch: {len(vids)} vs {feats.shape[0]}, patient={hcm_row_pid(row)}")
    return vids


def hcm_indices_for_expert(row: Dict, expert: str) -> np.ndarray:
    expert = str(expert).lower()
    mask = hcm_clip_mask(row)
    names = np.asarray([hcm_view_name(v) for v in hcm_clip_view_ids(row)])
    if expert == "full":
        return np.where(mask)[0]
    if expert == "lax":
        return np.where(mask & np.isin(names, ["2ch", "3ch", "4ch"]))[0]
    return np.where(mask & (names == expert))[0]


def hcm_assert_splits_aligned(raw: Dict[str, List[Dict]], diag: Dict[str, List[Dict]], cfg: dict) -> None:
    print("\n========== HCM raw/diagnosis feature alignment audit ==========")
    for split in ["train", "val", "test"]:
        rrows, drows = raw[split], diag[split]
        if len(rrows) != len(drows):
            raise RuntimeError(f"{split}: row count mismatch raw={len(rrows)} diagnosis={len(drows)}")
        n = len(rrows)
        events = int(sum(hcm_row_event(r) for r in rrows))
        exp_n = int(cfg.get(f"expected_{split}_n", 0) or 0)
        exp_e = int(cfg.get(f"expected_{split}_events", 0) or 0)
        strict_expected = bool(cfg.get("strict_expected_split_counts", False))

        # The raw/diagnosis row-by-row alignment checks below are always strict.
        # The expected_* values are only historical audit references and can become
        # stale whenever split_source_dir changes. Do not confuse a stale expected
        # count with a feature-alignment or checkpoint-loading failure.
        count_mismatches = []
        if exp_n and n != exp_n:
            count_mismatches.append(f"expected n={exp_n}, got n={n}")
        if exp_e and events != exp_e:
            count_mismatches.append(f"expected events={exp_e}, got events={events}")
        if count_mismatches:
            msg = f"{split}: " + "; ".join(count_mismatches)
            if strict_expected:
                raise RuntimeError(msg)
            print(f"[WARN] {msg}. Continuing because strict_expected_split_counts=False.")
        maes = []
        for i, (rr, dd) in enumerate(zip(rrows, drows)):
            if hcm_row_pid(rr) != hcm_row_pid(dd):
                raise RuntimeError(f"{split}[{i}]: patient_id mismatch raw={hcm_row_pid(rr)} diagnosis={hcm_row_pid(dd)}")
            if abs(hcm_row_time(rr) - hcm_row_time(dd)) > 1e-6:
                raise RuntimeError(f"{split}[{i}]: time mismatch")
            if hcm_row_event(rr) != hcm_row_event(dd):
                raise RuntimeError(f"{split}[{i}]: event mismatch")
            if not np.array_equal(hcm_clip_mask(rr), hcm_clip_mask(dd)):
                raise RuntimeError(f"{split}[{i}]: clip_mask mismatch")
            if not np.array_equal(hcm_clip_view_ids(rr), hcm_clip_view_ids(dd)):
                raise RuntimeError(f"{split}[{i}]: clip_view_ids mismatch")
            rf, df = hcm_clip_features(rr), hcm_clip_features(dd)
            if rf.shape != df.shape:
                raise RuntimeError(f"{split}[{i}]: feature shape mismatch raw={rf.shape} diagnosis={df.shape}")
            maes.append(float(np.abs(rf - df).mean()))
        print(json.dumps({
            "split": split,
            "n": n,
            "events": events,
            "feature_mae_mean": float(np.mean(maes)),
            "feature_mae_min": float(np.min(maes)),
            "feature_mae_max": float(np.max(maes)),
        }, ensure_ascii=False, indent=2))
        if float(np.mean(maes)) < 1e-8:
            raise RuntimeError(f"{split}: raw and diagnosis features are effectively identical.")
    print("PASS: raw and diagnosis feature rows are aligned and numerically different.\n")


def hcm_kmeans(x: np.ndarray, k: int, seed: int = 42, n_iter: int = 25, max_points: int = 20000) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        raise ValueError("empty input for kmeans")
    x_fit = x[rng.choice(len(x), min(len(x), max_points), replace=False)] if len(x) > max_points else x
    if len(x_fit) < k:
        out = np.zeros((k, x.shape[1]), dtype=np.float32)
        out[:len(x_fit)] = x_fit
        for i in range(len(x_fit), k):
            out[i] = x_fit[i % len(x_fit)]
        return out
    centers = x_fit[rng.choice(len(x_fit), k, replace=False)].copy()
    for _ in range(n_iter):
        dist = ((x_fit[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        new = centers.copy()
        for j in range(k):
            m = labels == j
            if m.any():
                new[j] = x_fit[m].mean(axis=0)
        if np.allclose(new, centers):
            break
        centers = new
    return centers.astype(np.float32)


def hcm_fit_prototypes(raw_train: List[Dict], diag_train: List[Dict], experts: List[str], k: int, seed: int) -> Dict[str, np.ndarray]:
    out = {}
    for expert in experts:
        clips = []
        for rows in [raw_train, diag_train]:
            for row in rows:
                idx = hcm_indices_for_expert(row, expert)
                if len(idx):
                    clips.append(hcm_clip_features(row)[idx])
        if clips:
            x = np.concatenate(clips, axis=0)
        else:
            x = np.zeros((k, hcm_clip_features(raw_train[0]).shape[-1]), dtype=np.float32)
        out[expert] = hcm_kmeans(x, k=k, seed=seed)
    return out


def hcm_vectorize_clips(clips: np.ndarray, centers: Optional[np.ndarray], topdev_m: int, proto_k: int) -> np.ndarray:
    # final_feature_mode is fixed to topdev3_proto6.
    if clips.size == 0:
        d = int(centers.shape[1]) if centers is not None else 768
        return np.zeros((d * 3 + int(proto_k),), dtype=np.float32)
    clips = np.asarray(clips, dtype=np.float32)
    mu = clips.mean(axis=0)
    sd = clips.std(axis=0)
    dist = np.linalg.norm(clips - mu[None, :], axis=1)
    order = np.argsort(-dist)
    top = clips[order[:min(int(topdev_m), len(order))]]
    if len(top) < int(topdev_m):
        top = np.concatenate([top, np.repeat(mu[None, :], int(topdev_m) - len(top), axis=0)], axis=0)
    top_mean = top.mean(axis=0)
    if centers is None:
        pdist = np.zeros((int(proto_k),), dtype=np.float32)
    else:
        pdist = np.linalg.norm(centers - mu[None, :], axis=1).astype(np.float32)
    return np.concatenate([mu, sd, top_mean, pdist], axis=0).astype(np.float32)


def hcm_build_expert_vectors(rows: List[Dict], experts: List[str], prototypes: Dict[str, np.ndarray], cfg: dict) -> Dict[str, np.ndarray]:
    out = {}
    topdev_m = int(cfg.get("final_topdev_m", 3))
    proto_k = int(cfg.get("final_proto_k", 6))
    for expert in experts:
        centers = prototypes.get(expert)
        vals = []
        for row in rows:
            idx = hcm_indices_for_expert(row, expert)
            clips = hcm_clip_features(row)[idx] if len(idx) else np.zeros((0, hcm_clip_features(row).shape[-1]), dtype=np.float32)
            vals.append(hcm_vectorize_clips(clips, centers, topdev_m, proto_k))
        out[expert] = np.stack(vals, axis=0).astype(np.float32)
    return out


class HcmPCA:
    def __init__(self, mean: np.ndarray, std: np.ndarray, components: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.components = components.astype(np.float32)


def hcm_fit_pca(x: np.ndarray, dim: int) -> HcmPCA:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0)
    xc = x - mean[None, :]
    std = xc.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = xc / std[None, :]
    try:
        _, _, vt = np.linalg.svd(xs, full_matrices=False)
        comp = vt[:int(dim)]
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(123)
        comp = rng.normal(size=(int(dim), x.shape[1])).astype(np.float32)
        comp /= (np.linalg.norm(comp, axis=1, keepdims=True) + 1e-8)
    if comp.shape[0] < int(dim):
        comp = np.concatenate([comp, np.zeros((int(dim) - comp.shape[0], x.shape[1]), dtype=np.float32)], axis=0)
    return HcmPCA(mean, std, comp)


def hcm_apply_pca(x: np.ndarray, state: HcmPCA) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float32) - state.mean[None, :]) / state.std[None, :]
    return (z @ state.components.T).astype(np.float32)


class HcmPrepared:
    def __init__(self, raw, diag, pids, times, events):
        self.raw = raw
        self.diag = diag
        self.pids = pids
        self.times = times
        self.events = events


def hcm_prepare_features(raw_splits: Dict[str, List[Dict]], diag_splits: Dict[str, List[Dict]], cfg: dict) -> HcmPrepared:
    experts = hcm_parse_strs(cfg.get("final_experts", "2ch,3ch,4ch,sax,lax,full"))
    pca_dim = int(cfg.get("final_pca_dim", 64))
    proto_k = int(cfg.get("final_proto_k", 6))
    prototypes = hcm_fit_prototypes(raw_splits["train"], diag_splits["train"], experts, proto_k, seed=42)
    raw_vec = {sp: hcm_build_expert_vectors(raw_splits[sp], experts, prototypes, cfg) for sp in ["train", "val", "test"]}
    diag_vec = {sp: hcm_build_expert_vectors(diag_splits[sp], experts, prototypes, cfg) for sp in ["train", "val", "test"]}
    pcas: Dict[str, HcmPCA] = {}
    for expert in experts:
        fit_x = np.concatenate([raw_vec["train"][expert], diag_vec["train"][expert]], axis=0)
        pcas[expert] = hcm_fit_pca(fit_x, pca_dim)
    raw_arr, diag_arr = {}, {}
    for sp in ["train", "val", "test"]:
        raw_parts, diag_parts = [], []
        for expert in experts:
            raw_parts.append(hcm_apply_pca(raw_vec[sp][expert], pcas[expert]))
            diag_parts.append(hcm_apply_pca(diag_vec[sp][expert], pcas[expert]))
        raw_arr[sp] = np.stack(raw_parts, axis=1).astype(np.float32)   # [N, E, pca_dim]
        diag_arr[sp] = np.stack(diag_parts, axis=1).astype(np.float32)
    return HcmPrepared(
        raw=raw_arr,
        diag=diag_arr,
        pids={sp: np.asarray([hcm_row_pid(r) for r in raw_splits[sp]]).astype(str) for sp in ["train", "val", "test"]},
        times={sp: np.asarray([hcm_row_time(r) for r in raw_splits[sp]], dtype=np.float32) for sp in ["train", "val", "test"]},
        events={sp: np.asarray([hcm_row_event(r) for r in raw_splits[sp]], dtype=np.int64) for sp in ["train", "val", "test"]},
    )


class HcmExpertMeanCox(nn.Module):
    def __init__(self, in_dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(int(in_dim), int(bottleneck_dim)),
            nn.LayerNorm(int(bottleneck_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.risk = nn.Linear(int(bottleneck_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)       # [N, E, B]
        h = h.mean(dim=1)      # [N, B]
        return self.risk(h).squeeze(-1)


def hcm_cox_ph_loss(risk: torch.Tensor, time_to_event: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time_to_event, descending=True)
    r = risk[order]
    e = event[order].float()
    log_cum = torch.logcumsumexp(r, dim=0)
    return -((r - log_cum) * e).sum() / torch.clamp(e.sum(), min=1.0)


def hcm_cindex(time_to_event: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    time_to_event = np.asarray(time_to_event, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    comparable = 0.0
    concordant = 0.0
    for i in range(len(time_to_event)):
        if event[i] != 1:
            continue
        js = np.where(time_to_event > time_to_event[i])[0]
        if len(js) == 0:
            continue
        comparable += float(len(js))
        concordant += float(np.sum(risk[i] > risk[js]))
        concordant += 0.5 * float(np.sum(risk[i] == risk[js]))
    return float(concordant / comparable) if comparable > 0 else float("nan")


def hcm_binary_auc(y: np.ndarray, risk: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    risk = np.asarray(risk, dtype=float)
    pos = risk[y == 1]
    neg = risk[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = pd.Series(all_scores).rank(method="average").values
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def hcm_horizon_auc(time_to_event: np.ndarray, event: np.ndarray, risk: np.ndarray, horizon: int) -> Tuple[float, int, int]:
    time_to_event = np.asarray(time_to_event, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    usable = ((event == 1) & (time_to_event <= float(horizon))) | (time_to_event > float(horizon))
    if usable.sum() < 3:
        return float("nan"), int(usable.sum()), 0
    y = ((event == 1) & (time_to_event <= float(horizon)))[usable].astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), int(usable.sum()), int(y.sum())
    return hcm_binary_auc(y, risk[usable]), int(usable.sum()), int(y.sum())


def hcm_logrank_p(time_to_event: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Two-group log-rank p-value with a no-lifelines fallback.

    The previous clean scripts returned NaN whenever lifelines was not installed,
    although a pure NumPy logrank_p_value implementation is already available
    in this file. For final prognosis reporting and plotting, p-values must be
    computed deterministically instead of silently becoming NaN.
    """
    time_to_event = np.asarray(time_to_event, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if len(risk) == 0 or not np.all(np.isfinite(risk)):
        return float("nan")
    high = risk >= np.median(risk)
    low = ~high
    if high.sum() < 2 or low.sum() < 2:
        return float("nan")
    group = high.astype(int)

    # Prefer lifelines if present, but do not require it.
    try:
        from lifelines.statistics import logrank_test
        res = logrank_test(
            time_to_event[high],
            time_to_event[low],
            event_observed_A=event[high],
            event_observed_B=event[low],
        )
        p = float(res.p_value)
        if np.isfinite(p):
            return p
    except Exception:
        pass

    # Fallback: in-file implementation using the chi-square(1) survival function.
    try:
        p = float(logrank_p_value(time_to_event, event, group))
        return p if np.isfinite(p) else float("nan")
    except Exception:
        return float("nan")


def hcm_eval(time_to_event: np.ndarray, event: np.ndarray, risk: np.ndarray, cfg: dict, prefix: str = "") -> Dict[str, float]:
    out = {
        f"{prefix}n": int(len(time_to_event)),
        f"{prefix}events": int(np.asarray(event).sum()),
        f"{prefix}cindex": hcm_cindex(time_to_event, event, risk),
        f"{prefix}logrank_p": hcm_logrank_p(time_to_event, event, risk),
    }
    for h in hcm_parse_horizons(cfg.get("horizons_days", "365,730,1095,1460,1825")):
        auc, n_h, e_h = hcm_horizon_auc(time_to_event, event, risk, h)
        label = {365: "1y", 730: "2y", 1095: "3y", 1460: "4y", 1825: "5y"}.get(int(h), f"{h}d")
        out[f"{prefix}auc_{label}_auc"] = auc
        out[f"{prefix}auc_{label}_n"] = n_h
        out[f"{prefix}auc_{label}_events"] = e_h
    return out


def hcm_train_one_seed(x_by_split: Dict[str, np.ndarray], prepared: HcmPrepared, variant: str, seed: int, cfg: dict, out_dir: Path) -> Tuple[Dict, Dict[str, np.ndarray]]:
    hcm_seed_everything(int(seed))
    device = hcm_choose_head_device(str(cfg.get("final_head_device", "auto")))
    model = HcmExpertMeanCox(
        in_dim=int(x_by_split["train"].shape[-1]),
        bottleneck_dim=int(cfg.get("final_bottleneck_dim", 16)),
        dropout=float(cfg.get("final_head_dropout", 0.10)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("final_head_lr", 3e-4)), weight_decay=float(cfg.get("final_head_weight_decay", 1e-3)))
    train_x = torch.as_tensor(x_by_split["train"], dtype=torch.float32, device=device)
    train_t = torch.as_tensor(prepared.times["train"], dtype=torch.float32, device=device)
    train_e = torch.as_tensor(prepared.events["train"], dtype=torch.float32, device=device)
    val_x = torch.as_tensor(x_by_split["val"], dtype=torch.float32, device=device)
    best_val = -1.0
    best_epoch = -1
    best_state = None
    bad = 0
    max_epochs = int(cfg.get("final_head_epochs", 500))
    patience = int(cfg.get("final_head_patience", 80))
    eval_every = int(cfg.get("final_head_eval_every", 5))
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        risk = model(train_x)
        loss = hcm_cox_ph_loss(risk, train_t, train_e)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("final_grad_clip", 5.0)))
        opt.step()
        if epoch == 1 or epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_risk = model(val_x).detach().cpu().numpy().astype(float)
            val_c = hcm_cindex(prepared.times["val"], prepared.events["val"], val_risk)
            if val_c > best_val:
                best_val = float(val_c)
                best_epoch = int(epoch)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += eval_every
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    risks = {}
    with torch.no_grad():
        for sp in ["train", "val", "test"]:
            x = torch.as_tensor(x_by_split[sp], dtype=torch.float32, device=device)
            risks[sp] = model(x).detach().cpu().numpy().astype(np.float32)
    row = {
        "variant": variant,
        "seed": int(seed),
        "feature_mode": str(cfg.get("final_feature_mode", "topdev3_proto6")),
        "pca_dim": int(cfg.get("final_pca_dim", 64)),
        "bottleneck_dim": int(cfg.get("final_bottleneck_dim", 16)),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val),
    }
    for sp in ["train", "val", "test"]:
        row.update(hcm_eval(prepared.times[sp], prepared.events[sp], risks[sp], cfg, prefix=f"{sp}_"))
    pred_dir = ensure_dir(out_dir / "seed_predictions" / f"{variant}_seed{seed}")
    for sp in ["train", "val", "test"]:
        pd.DataFrame({
            "patient_id": prepared.pids[sp],
            "time_to_event": prepared.times[sp],
            "event": prepared.events[sp],
            "risk": risks[sp],
            "class_name": "HCM",
        }).to_csv(pred_dir / f"risk_predictions_{sp}.csv", index=False, encoding="utf-8-sig")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, risks


def hcm_zscore_from_train(train_risk: np.ndarray, risk: np.ndarray) -> np.ndarray:
    train_risk = np.asarray(train_risk, dtype=float)
    mu = float(train_risk.mean())
    sd = float(train_risk.std())
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return ((np.asarray(risk, dtype=float) - mu) / sd).astype(np.float32)


def hcm_aggregate_seed_risks(seed_risks: List[np.ndarray], method: str) -> np.ndarray:
    stack = np.stack(seed_risks, axis=0).astype(np.float32)
    if str(method).lower() == "mean":
        return stack.mean(axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def hcm_save_prediction_table(prepared: HcmPrepared, risks_by_split: Dict[str, np.ndarray], out_dir: Path, prefix: str) -> None:
    pred_dir = ensure_dir(out_dir / "risk_predictions" / prefix)
    train_threshold = float(np.median(risks_by_split["train"]))
    for sp in ["train", "val", "test"]:
        risk = risks_by_split[sp]
        pd.DataFrame({
            "patient_id": prepared.pids[sp],
            "time_to_event": prepared.times[sp],
            "event": prepared.events[sp],
            "risk": risk,
            "risk_group": np.where(risk >= train_threshold, "high", "low"),
            "risk_threshold_train_median": train_threshold,
            "class_name": "HCM",
        }).to_csv(pred_dir / f"risk_predictions_{sp}.csv", index=False, encoding="utf-8-sig")


def hcm_build_final_risklevel(prepared: HcmPrepared, raw_seed_risks: Dict[int, Dict[str, np.ndarray]], diag_seed_risks: Dict[int, Dict[str, np.ndarray]], cfg: dict, out_dir: Path) -> pd.DataFrame:
    lam = float(cfg.get("final_lambda", 0.45))
    seeds = hcm_parse_ints(cfg.get("final_seeds"))
    agg = str(cfg.get("final_seed_aggregate", "median"))
    model_defs = {
        "diagnosis_only_lambda0": 0.0,
        "raw_only_lambda1": 1.0,
        f"hcm_diaganchored_lambda{str(lam).replace('.', 'p')}": lam,
    }
    rows = []
    for model_name, cur_lam in model_defs.items():
        split_seed_risks: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
        for seed in seeds:
            if seed not in raw_seed_risks or seed not in diag_seed_risks:
                raise RuntimeError(f"Missing seed risks for seed={seed}")
            for sp in ["train", "val", "test"]:
                z_diag = hcm_zscore_from_train(diag_seed_risks[seed]["train"], diag_seed_risks[seed][sp])
                z_raw = hcm_zscore_from_train(raw_seed_risks[seed]["train"], raw_seed_risks[seed][sp])
                split_seed_risks[sp].append(((1.0 - cur_lam) * z_diag + cur_lam * z_raw).astype(np.float32))
        final_risks = {sp: hcm_aggregate_seed_risks(split_seed_risks[sp], agg) for sp in ["train", "val", "test"]}
        hcm_save_prediction_table(prepared, final_risks, out_dir, model_name)
        row = {
            "model": model_name,
            "lambda": float(cur_lam),
            "is_final_candidate": bool(abs(cur_lam - lam) < 1e-12),
            "feature_mode": str(cfg.get("final_feature_mode", "topdev3_proto6")),
            "pca_dim": int(cfg.get("final_pca_dim", 64)),
            "bottleneck_dim": int(cfg.get("final_bottleneck_dim", 16)),
            "n_seeds": int(len(seeds)),
            "seed_aggregate": agg,
        }
        for sp in ["train", "val", "test"]:
            row.update(hcm_eval(prepared.times[sp], prepared.events[sp], final_risks[sp], cfg, prefix=f"{sp}_"))
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values(["is_final_candidate", "val_cindex"], ascending=[False, False])
    metrics.to_csv(out_dir / "hcm_final_lambda045_metrics.csv", index=False, encoding="utf-8-sig")
    # One-row summary with deltas against the two fixed controls.
    final_row = metrics[metrics["is_final_candidate"]].iloc[0].to_dict()
    raw_row = metrics[metrics["model"] == "raw_only_lambda1"].iloc[0].to_dict()
    diag_row = metrics[metrics["model"] == "diagnosis_only_lambda0"].iloc[0].to_dict()
    summary = dict(final_row)
    for metric in ["val_cindex", "test_cindex", "test_auc_1y_auc", "test_auc_3y_auc", "test_auc_5y_auc"]:
        summary[f"raw_ref_{metric}"] = raw_row.get(metric, np.nan)
        summary[f"delta_vs_raw_{metric}"] = final_row.get(metric, np.nan) - raw_row.get(metric, np.nan)
        summary[f"diagnosis_ref_{metric}"] = diag_row.get(metric, np.nan)
        summary[f"delta_vs_diagnosis_{metric}"] = final_row.get(metric, np.nan) - diag_row.get(metric, np.nan)
    pd.DataFrame([summary]).to_csv(out_dir / "hcm_final_lambda045_summary.csv", index=False, encoding="utf-8-sig")
    return metrics


def hcm_clean_final_main() -> None:
    # Install final defaults before parse_args() builds the CLI.
    # RUN_CONFIG is already complete; no late configuration override.
    args = parse_args()
    args.gpu_mode = "single"
    args.auto_launch_ddp = False
    args.disease_mode = "HCM_only"
    args.both_independent_modes = "HCM_only"
    args.fine_tune_mode = "hybrid_cross_attention"  # clip-level feature extraction mode
    args.train_horizon_heads = False
    args.horizon_loss_weight = 0.0
    args.force_reextract_features = bool(getattr(args, "force_reextract_features", True))
    args.diagnosis_encoder_agg = canonicalize_agg(getattr(args, "diagnosis_encoder_agg", "hier_mean"))
    args.survival_agg = canonicalize_agg(getattr(args, "survival_agg", getattr(args, "agg", "hier_attn")))
    args.agg = args.survival_agg

    # Final resolved configuration audit: these values come from the top RUN_CONFIG.
    print(f"Resolved split_source_dir  : {args.split_source_dir}")
    print(f"Resolved classification_ckpt: {args.classification_ckpt}")
    if not Path(str(args.split_source_dir)).is_dir():
        raise FileNotFoundError(f"split_source_dir does not exist: {args.split_source_dir}")
    if not Path(str(args.classification_ckpt)).is_file():
        raise FileNotFoundError(f"classification_ckpt does not exist: {args.classification_ckpt}")

    # Use one selected GPU for re-extraction; the head training also uses this CUDA device by default.
    device = configure_runtime_gpu(str(getattr(args, "visible_gpus", "") or ""), device_arg="auto")
    hcm_seed_everything(int(getattr(args, "seed", 42)))
    run_dir = hcm_final_root(args)
    print("=" * 80)
    print("Clean final HCM prognosis: diagnosis-anchored risk-level correction")
    print(f"Run dir: {run_dir}")
    print(f"Feature extraction device: {device}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print("Clean standalone implementation: no embedded old-script source strings and no dependency on prior result directories.")
    print("=" * 80)
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # Build fixed HCM split once.
    selected_views, required_views, slice_plan, split_to_samples = collect_split_samples_for_args(args)
    print("\nHCM survival split summary:")
    print("  Train:", summarize_survival_samples(split_to_samples["train"]))
    print("  Val:  ", summarize_survival_samples(split_to_samples["val"]))
    print("  Test: ", summarize_survival_samples(split_to_samples["test"]))
    save_survival_split_csv(split_to_samples["train"], run_dir / "split_train_survival.csv")
    save_survival_split_csv(split_to_samples["val"], run_dir / "split_val_survival.csv")
    save_survival_split_csv(split_to_samples["test"], run_dir / "split_test_survival.csv")

    # 1) diagnosis_ckpt features.
    diag_args = clone_args(args)
    diag_args.classification_ckpt = str(getattr(args, "classification_ckpt", ""))
    diag_args.force_reextract_features = bool(getattr(args, "force_reextract_features", True))
    diag_stage = ensure_dir(run_dir / "diagnosis_ckpt_feature_extraction")
    print("\n" + "=" * 80)
    print("Extracting HCM diagnosis_ckpt features")
    print(f"classification_ckpt: {diag_args.classification_ckpt}")
    diag_rows, diag_feature_dir = extract_or_load_features_for_disease(
        diag_args, selected_views, required_views, slice_plan, split_to_samples, diag_stage, device
    )

    # 2) raw_swin features.
    raw_args = clone_args(args)
    raw_args.classification_ckpt = ""
    raw_args.force_reextract_features = bool(getattr(args, "force_reextract_features", True))
    raw_stage = ensure_dir(run_dir / "raw_swin_feature_extraction")
    print("\n" + "=" * 80)
    print("Extracting HCM raw_swin features")
    print("classification_ckpt: [EMPTY: raw Video Swin only]")
    raw_rows, raw_feature_dir = extract_or_load_features_for_disease(
        raw_args, selected_views, required_views, slice_plan, split_to_samples, raw_stage, device
    )

    with open(run_dir / "resolved_feature_dirs.json", "w", encoding="utf-8") as f:
        json.dump({"diagnosis_ckpt": str(diag_feature_dir), "raw_swin": str(raw_feature_dir)}, f, ensure_ascii=False, indent=2)

    cfg = vars(args).copy()
    cfg.update({
        "final_feature_mode": getattr(args, "final_feature_mode"),
        "final_experts": getattr(args, "final_experts"),
        "final_pca_dim": int(getattr(args, "final_pca_dim")),
        "final_bottleneck_dim": int(getattr(args, "final_bottleneck_dim")),
        "final_topdev_m": int(getattr(args, "final_topdev_m")),
        "final_proto_k": int(getattr(args, "final_proto_k")),
        "final_lambda": float(getattr(args, "final_lambda")),
        "final_seed_aggregate": getattr(args, "final_seed_aggregate"),
        "final_seeds": getattr(args, "final_seeds"),
        "final_head_device": getattr(args, "final_head_device"),
        "final_head_epochs": int(getattr(args, "final_head_epochs")),
        "final_head_patience": int(getattr(args, "final_head_patience")),
        "final_head_eval_every": int(getattr(args, "final_head_eval_every")),
        "final_head_lr": float(getattr(args, "final_head_lr")),
        "final_head_weight_decay": float(getattr(args, "final_head_weight_decay")),
        "final_head_dropout": float(getattr(args, "final_head_dropout")),
        "final_grad_clip": float(getattr(args, "final_grad_clip")),
        "horizons_days": str(getattr(args, "horizons_days")),
        "expected_train_n": int(getattr(args, "expected_train_n")),
        "expected_train_events": int(getattr(args, "expected_train_events")),
        "expected_val_n": int(getattr(args, "expected_val_n")),
        "expected_val_events": int(getattr(args, "expected_val_events")),
        "expected_test_n": int(getattr(args, "expected_test_n")),
        "expected_test_events": int(getattr(args, "expected_test_events")),
    })

    hcm_assert_splits_aligned(raw_rows, diag_rows, cfg)
    print("\n" + "=" * 80)
    print("Preparing low-capacity HCM View-Cox representation")
    print(f"feature_mode={cfg['final_feature_mode']} | PCA={cfg['final_pca_dim']} | bottleneck={cfg['final_bottleneck_dim']}")
    prepared = hcm_prepare_features(raw_rows, diag_rows, cfg)

    seeds = hcm_parse_ints(cfg.get("final_seeds"))
    head_out = ensure_dir(run_dir / "lowcapacity_cox_heads")
    raw_seed_risks: Dict[int, Dict[str, np.ndarray]] = {}
    diag_seed_risks: Dict[int, Dict[str, np.ndarray]] = {}
    single_rows = []
    for seed in seeds:
        print("\n" + "-" * 80)
        print(f"Training raw_only Cox head | seed={seed}")
        row, risks = hcm_train_one_seed(prepared.raw, prepared, "raw_only", int(seed), cfg, head_out)
        single_rows.append(row)
        raw_seed_risks[int(seed)] = risks
        print(json.dumps({"variant": "raw_only", "seed": int(seed), "val_cindex": row.get("val_cindex"), "test_cindex": row.get("test_cindex"), "test_auc_3y": row.get("test_auc_3y_auc")}, ensure_ascii=False, indent=2))

        print("\n" + "-" * 80)
        print(f"Training diagnosis_only Cox head | seed={seed}")
        row, risks = hcm_train_one_seed(prepared.diag, prepared, "diagnosis_only", int(seed), cfg, head_out)
        single_rows.append(row)
        diag_seed_risks[int(seed)] = risks
        print(json.dumps({"variant": "diagnosis_only", "seed": int(seed), "val_cindex": row.get("val_cindex"), "test_cindex": row.get("test_cindex"), "test_auc_3y": row.get("test_auc_3y_auc")}, ensure_ascii=False, indent=2))

    single_df = pd.DataFrame(single_rows)
    single_df.to_csv(run_dir / "hcm_single_seed_raw_diag_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Building final HCM risk-level correction")
    print(f"risk_final = {(1.0 - float(cfg['final_lambda'])):.2f} * z(diagnosis risk) + {float(cfg['final_lambda']):.2f} * z(raw risk)")
    final_dir = ensure_dir(run_dir / "final_lambda045_risklevel")
    final_metrics = hcm_build_final_risklevel(prepared, raw_seed_risks, diag_seed_risks, cfg, final_dir)

    compact_cols = [
        "model", "lambda", "is_final_candidate", "val_cindex", "test_cindex", "test_logrank_p",
        "test_auc_1y_auc", "test_auc_2y_auc", "test_auc_3y_auc", "test_auc_4y_auc", "test_auc_5y_auc",
        "n_seeds", "seed_aggregate", "feature_mode", "pca_dim", "bottleneck_dim",
    ]
    compact_cols = [c for c in compact_cols if c in final_metrics.columns]
    compact = final_metrics[compact_cols].copy()
    compact.to_csv(run_dir / "hcm_final_lambda045_compact.csv", index=False, encoding="utf-8-sig")
    with open(run_dir / "README_final_hcm_clean.md", "w", encoding="utf-8") as f:
        f.write("# Clean final HCM prognosis script\n\n")
        f.write("Pipeline: re-extract diagnosis_ckpt and raw_swin features, train raw_only and diagnosis_only low-capacity View-Cox heads with 10 seeds, then apply fixed λ=0.45 diagnosis-anchored risk-level correction.\n\n")
        f.write("Clean standalone implementation: no embedded old-script source strings and no dependency on previous result directories.\n\n")
        f.write("## Compact final metrics\n\n")
        try:
            f.write(compact.to_markdown(index=False))
        except Exception:
            f.write(compact.to_csv(index=False))
        f.write("\n")

    print("\n========== HCM final compact metrics ==========")
    print(compact.to_string(index=False))
    print(f"\nSaved all outputs to: {run_dir}")




# =============================================================================
# Diagnosis multilevel representation -> disease-specific survival residual
# =============================================================================
# This pipeline is the active entry point of this file. It preserves the exact
# diagnosis forward path (clip + view embedding + diagnosis slice pooling +
# diagnosis view pooling) and trains only a low-capacity prognosis residual and
# Cox head on frozen diagnosis representations.

MULTILEVEL_FEATURE_VERSION = "diagnosis_multilevel_v1"


def multilevel_feature_hash(args, disease: str) -> str:
    obj = {
        "version": MULTILEVEL_FEATURE_VERSION,
        "disease": disease,
        "data_path": args.data_path,
        "split_source_dir": args.split_source_dir,
        "classification_ckpt": args.classification_ckpt,
        "selected_views": args.selected_views,
        "slice_plan": args.slice_plan,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "diagnosis_encoder_agg": args.diagnosis_encoder_agg,
    }
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _multilevel_extract_method(self, batch: Dict) -> Dict[str, torch.Tensor]:
    """Run the exact diagnosis hierarchy and expose all frozen representations.

    Returns:
        backbone_clip_features: [B, 17, 768], before view embeddings
        diagnosis_clip_features: [B, 17, 768], after view embeddings
        diagnosis_view_features: [B, 4, 768], after diagnosis slice pooling
        diagnosis_patient_feature: [B, 768], after diagnosis view pooling
        diagnosis_logits: [B, 3]
    """
    videos = batch["videos"]
    clip_mask = batch["clip_mask"].bool()
    clip_view_ids = batch["clip_view_ids"].long()
    B, N, C, T, H, W = videos.shape
    flat_mask = clip_mask.reshape(-1)
    flat_view_ids = clip_view_ids.reshape(-1)
    flat_videos = videos.reshape(B * N, C, T, H, W)

    backbone_flat = torch.zeros(B * N, self.feature_dim, device=videos.device, dtype=torch.float32)
    diagnosis_flat = torch.zeros_like(backbone_flat)
    if flat_mask.any():
        x_valid = flat_videos[flat_mask].expand(-1, 3, -1, -1, -1).contiguous()
        backbone_valid = self._run_backbone(x_valid).float()
        diagnosis_valid = backbone_valid + self.view_embeddings[flat_view_ids[flat_mask]]
        backbone_flat[flat_mask] = backbone_valid
        diagnosis_flat[flat_mask] = diagnosis_valid

    backbone_clip = backbone_flat.reshape(B, N, self.feature_dim)
    diagnosis_clip = diagnosis_flat.reshape(B, N, self.feature_dim)

    view_features, view_valids, slice_weights = [], [], []
    for local_vid in range(self.num_views):
        start, end = self.view_clip_ranges[local_vid]
        vf, sw = self.slice_pool(diagnosis_clip[:, start:end, :], clip_mask[:, start:end])
        view_features.append(vf)
        view_valids.append(clip_mask[:, start:end].any(dim=1))
        slice_weights.append(sw)
    diagnosis_view = torch.stack(view_features, dim=1)
    view_mask = torch.stack(view_valids, dim=1)
    diagnosis_patient, view_weights = self.view_pool(diagnosis_view, view_mask)
    diagnosis_logits = self.classifier(diagnosis_patient)

    return {
        "backbone_clip_features": backbone_clip,
        "diagnosis_clip_features": diagnosis_clip,
        "diagnosis_view_features": diagnosis_view,
        "diagnosis_patient_feature": diagnosis_patient,
        "diagnosis_logits": diagnosis_logits,
        "clip_mask": clip_mask,
        "view_mask": view_mask,
        "clip_view_ids": clip_view_ids,
        "view_weights": view_weights,
    }


# Attach the method without changing the original diagnosis class definition.
MultiViewVideoSwinFeatureExtractor.extract_multilevel_features = _multilevel_extract_method


@torch.no_grad()
def extract_multilevel_split(model, loader, device, args) -> Dict[str, object]:
    model.eval()
    tensor_lists = defaultdict(list)
    records = []
    start_time = time.time()
    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type="cuda", enabled=(bool(args.amp) and device.type == "cuda")):
            out = model.extract_multilevel_features(batch)
        for key in [
            "backbone_clip_features", "diagnosis_clip_features", "diagnosis_view_features",
            "diagnosis_patient_feature", "diagnosis_logits", "clip_mask", "view_mask",
            "clip_view_ids", "view_weights",
        ]:
            tensor_lists[key].append(out[key].detach().cpu())

        pids = batch["patient_id"]
        classes = batch["class_name"]
        events = batch["event"].detach().cpu().numpy().tolist()
        times = batch["time_to_event"].detach().cpu().numpy().tolist()
        censored = batch["censored"].detach().cpu().numpy().tolist()
        event_types = batch.get("event_type", [""] * len(pids))
        for i in range(len(pids)):
            records.append({
                "patient_id": str(pids[i]),
                "class_name": str(classes[i]),
                "event": int(events[i]),
                "time_to_event": float(times[i]),
                "censored": float(censored[i]),
                "event_type": str(event_types[i]) if isinstance(event_types, (list, tuple)) else "",
            })
        if args.print_freq > 0 and ((step + 1) % int(args.print_freq) == 0):
            print0(f"  multilevel extraction {step+1:04d}/{len(loader):04d} | elapsed={time.time()-start_time:.1f}s")

    result = {k: torch.cat(v, dim=0) for k, v in tensor_lists.items()}
    result["meta"] = pd.DataFrame(records)
    return result


def save_multilevel_split(data: Dict[str, object], tensor_path: Path, meta_path: Path) -> None:
    payload = {k: v for k, v in data.items() if torch.is_tensor(v)}
    torch.save(payload, tensor_path)
    data["meta"].to_csv(meta_path, index=False, encoding="utf-8-sig")


def load_multilevel_split(tensor_path: Path, meta_path: Path) -> Dict[str, object]:
    payload = safe_torch_load(str(tensor_path), map_location="cpu", weights_only=True)
    payload["meta"] = pd.read_csv(meta_path, dtype={"patient_id": str})
    return payload


def extract_or_load_multilevel(args, split_to_samples, selected_views, required_views, slice_plan, run_dir, device):
    disease = "HCM" if str(args.disease_mode).lower().startswith("hcm") else "DCM"
    cache_dir = ensure_dir(run_dir / f"multilevel_features_{multilevel_feature_hash(args, disease)}")
    complete = all(
        (cache_dir / f"{sp}_multilevel.pt").exists() and (cache_dir / f"{sp}_meta.csv").exists()
        for sp in ["train", "val", "test"]
    )
    if complete and not bool(args.force_reextract_features):
        print0(f"Using cached diagnosis multilevel features: {cache_dir}")
        return {
            sp: load_multilevel_split(cache_dir / f"{sp}_multilevel.pt", cache_dir / f"{sp}_meta.csv")
            for sp in ["train", "val", "test"]
        }, cache_dir

    print0(f"Extracting exact diagnosis multilevel representations to: {cache_dir}")
    model = MultiViewVideoSwinFeatureExtractor(
        selected_views=selected_views,
        slice_plan=slice_plan,
        weights_path=args.weights_path,
        pretrained=bool(args.pretrained),
        agg=args.diagnosis_encoder_agg,
        dropout=float(args.dropout),
        backbone_chunk_size=int(args.backbone_chunk_size),
    ).to(device)
    model.load_classification_checkpoint(args.classification_ckpt)
    model.eval()

    outputs = {}
    for sp in ["train", "val", "test"]:
        ds = MultiViewCardiacNiftiDataset(
            root_dir=args.data_path,
            selected_views=selected_views,
            slice_plan=slice_plan,
            num_frames=int(args.num_frames),
            image_size=int(args.image_size),
            samples=split_to_samples[sp],
            mode="eval",
            required_views=required_views,
            allow_missing_selected_views=bool(args.allow_missing_selected_views),
            use_cache=bool(args.use_cache),
            cache_dir=args.cache_dir if args.cache_dir else None,
            min_frames_per_slice=int(args.min_frames_per_slice),
            verbose=False,
        )
        loader = build_dataloader(ds, args, distributed=False, shuffle=False)
        outputs[sp] = extract_multilevel_split(model, loader, device, args)
        save_multilevel_split(
            outputs[sp], cache_dir / f"{sp}_multilevel.pt", cache_dir / f"{sp}_meta.csv"
        )
        print0(
            f"Saved {sp}: n={len(outputs[sp]['meta'])}, "
            f"patient={tuple(outputs[sp]['diagnosis_patient_feature'].shape)}, "
            f"view={tuple(outputs[sp]['diagnosis_view_features'].shape)}, "
            f"clip={tuple(outputs[sp]['diagnosis_clip_features'].shape)}"
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs, cache_dir


def audit_multilevel_data(data_by_split, args):
    disease = "HCM" if str(args.disease_mode).lower().startswith("hcm") else "DCM"
    print0("\n========== Diagnosis multilevel representation audit ==========")
    expected = {
        "train": (int(args.expected_train_n), int(args.expected_train_events)),
        "val": (int(args.expected_val_n), int(args.expected_val_events)),
        "test": (int(args.expected_test_n), int(args.expected_test_events)),
    }
    for sp in ["train", "val", "test"]:
        d = data_by_split[sp]
        meta = d["meta"]
        n = len(meta)
        events = int(meta["event"].sum())
        exp_n, exp_e = expected[sp]
        if n != exp_n or events != exp_e:
            raise RuntimeError(f"{disease} {sp}: expected n/events={exp_n}/{exp_e}, got {n}/{events}")
        if not (meta["class_name"].astype(str).str.upper() == disease).all():
            raise RuntimeError(f"{sp}: contains non-{disease} patients")
        pf = d["diagnosis_patient_feature"]
        vf = d["diagnosis_view_features"]
        cf = d["diagnosis_clip_features"]
        if pf.ndim != 2 or vf.ndim != 3 or cf.ndim != 3:
            raise RuntimeError(f"{sp}: invalid multilevel feature shapes")
        if len(pf) != n or len(vf) != n or len(cf) != n:
            raise RuntimeError(f"{sp}: metadata/feature row mismatch")
        finite = bool(torch.isfinite(pf).all() and torch.isfinite(vf).all() and torch.isfinite(cf).all())
        print0(json.dumps({
            "split": sp, "disease": disease, "n": n, "events": events,
            "patient_shape": list(pf.shape), "view_shape": list(vf.shape), "clip_shape": list(cf.shape),
            "patient_std_mean": float(pf.float().std(dim=0).mean()),
            "finite": finite,
        }, ensure_ascii=False, indent=2))
        if not finite:
            raise RuntimeError(f"{sp}: non-finite diagnosis features")
    print0("PASS: exact diagnosis clip/view/patient representations are aligned.\n")


def _bounded_logit(init_value: float, max_value: float) -> float:
    ratio = min(max(float(init_value) / max(float(max_value), 1e-8), 1e-4), 1.0 - 1e-4)
    return math.log(ratio / (1.0 - ratio))


class DiagnosisPatientCoxBaseline(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patient_feature, **kwargs):
        return {
            "risk": self.head(patient_feature).squeeze(-1),
            "residual_gate": patient_feature.new_tensor(0.0),
            "clip_gate": patient_feature.new_tensor(0.0),
        }


class DiagnosisMultilevelSurvivalResidual(nn.Module):
    """Frozen diagnosis patient phenotype + low-capacity survival residual.

    The base path is exactly the diagnosis patient feature. The residual path
    re-reads diagnosis clip/view representations with survival supervision.
    Both gates are bounded and initialized small, so the model starts close to
    diagnosis-patient-feature Cox rather than replacing the diagnosis phenotype.
    """
    def __init__(self, dim: int, num_views: int, view_ranges, adapter_dim: int,
                 attention_hidden: int, num_heads: int, head_hidden: int,
                 dropout: float, gate_init: float, gate_max: float,
                 clip_gate_init: float, clip_gate_max: float):
        super().__init__()
        self.dim = int(dim)
        self.num_views = int(num_views)
        self.view_ranges = list(view_ranges)
        self.clip_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, attention_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden, 1),
        )
        self.clip_gate_max = float(clip_gate_max)
        self.clip_gate_logit = nn.Parameter(torch.tensor(
            _bounded_logit(clip_gate_init, clip_gate_max), dtype=torch.float32
        ))
        self.patient_query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, adapter_dim))
        self.view_key = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, adapter_dim))
        self.view_value = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, adapter_dim))
        self.view_pos = nn.Parameter(torch.zeros(num_views, adapter_dim))
        nn.init.trunc_normal_(self.view_pos, std=0.02)
        self.view_attention = nn.MultiheadAttention(
            embed_dim=adapter_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.residual_up = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, dim),
        )
        # Near-zero initialization protects the frozen diagnosis phenotype.
        nn.init.normal_(self.residual_up[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_up[-1].bias)
        self.gate_max = float(gate_max)
        self.residual_gate_logit = nn.Parameter(torch.tensor(
            _bounded_logit(gate_init, gate_max), dtype=torch.float32
        ))
        self.fuse_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def _survival_view_features(self, clip_features, diagnosis_view_features, clip_mask, view_dropout):
        B = clip_features.shape[0]
        working_mask = clip_mask.bool().clone()
        if self.training and view_dropout > 0:
            drop_flag = torch.rand(B, device=clip_features.device) < float(view_dropout)
            drop_view = torch.randint(0, self.num_views, (B,), device=clip_features.device)
            for b in torch.where(drop_flag)[0].tolist():
                s, e = self.view_ranges[int(drop_view[b])]
                working_mask[b, s:e] = False

        pooled = []
        for vid, (s, e) in enumerate(self.view_ranges):
            x = clip_features[:, s:e, :]
            m = working_mask[:, s:e]
            score = self.clip_score(x).squeeze(-1).masked_fill(~m, -1e4)
            w = torch.softmax(score, dim=1) * m.float()
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            p = (x * w.unsqueeze(-1)).sum(dim=1)
            has = m.any(dim=1, keepdim=True)
            p = torch.where(has, p, torch.zeros_like(p))
            pooled.append(p)
        survival_view = torch.stack(pooled, dim=1)
        clip_gate = self.clip_gate_max * torch.sigmoid(self.clip_gate_logit)
        blended_view = diagnosis_view_features + clip_gate * (survival_view - diagnosis_view_features)
        view_mask = torch.stack([
            working_mask[:, s:e].any(dim=1) for s, e in self.view_ranges
        ], dim=1)
        return blended_view, view_mask, clip_gate

    def forward(self, patient_feature, diagnosis_view_features, diagnosis_clip_features,
                clip_mask, view_dropout=0.0, **kwargs):
        view_features, view_mask, clip_gate = self._survival_view_features(
            diagnosis_clip_features, diagnosis_view_features, clip_mask, view_dropout
        )
        query = self.patient_query(patient_feature).unsqueeze(1)
        key = self.view_key(view_features) + self.view_pos.unsqueeze(0)
        value = self.view_value(view_features) + self.view_pos.unsqueeze(0)
        attended, attention = self.view_attention(
            query=query, key=key, value=value,
            key_padding_mask=~view_mask.bool(),
            need_weights=True, average_attn_weights=False,
        )
        residual = self.residual_up(attended.squeeze(1))
        gate = self.gate_max * torch.sigmoid(self.residual_gate_logit)
        fused = self.fuse_norm(patient_feature + gate * residual)
        return {
            "risk": self.head(fused).squeeze(-1),
            "residual_gate": gate,
            "clip_gate": clip_gate,
            "residual": residual,
            "view_attention": attention,
        }


def build_comparable_pairs(times: torch.Tensor, events: torch.Tensor):
    t = times.detach().cpu().numpy().astype(float)
    e = events.detach().cpu().numpy().astype(int)
    ii, jj = np.where((e[:, None] == 1) & (t[:, None] < t[None, :]))
    return torch.tensor(ii, dtype=torch.long), torch.tensor(jj, dtype=torch.long)


def pairwise_rank_loss(risk, pair_i, pair_j, tau: float, max_pairs: int):
    if pair_i.numel() == 0:
        return risk.sum() * 0.0
    if max_pairs > 0 and pair_i.numel() > max_pairs:
        idx = torch.randperm(pair_i.numel(), device=pair_i.device)[:max_pairs]
        pair_i = pair_i[idx]
        pair_j = pair_j[idx]
    margin = (risk[pair_i] - risk[pair_j]) / max(float(tau), 1e-6)
    return F.softplus(-margin).mean()


def split_tensors(data: Dict[str, object], device: torch.device):
    meta = data["meta"]
    return {
        "patient_feature": data["diagnosis_patient_feature"].float().to(device),
        "diagnosis_view_features": data["diagnosis_view_features"].float().to(device),
        "diagnosis_clip_features": data["diagnosis_clip_features"].float().to(device),
        "clip_mask": data["clip_mask"].bool().to(device),
        "diagnosis_logits": data["diagnosis_logits"].float().to(device),
        "time": torch.tensor(meta["time_to_event"].values, dtype=torch.float32, device=device),
        "event": torch.tensor(meta["event"].values, dtype=torch.float32, device=device),
    }


@torch.no_grad()
def predict_multilevel_model(model, tensors, view_dropout=0.0):
    model.eval()
    out = model(
        patient_feature=tensors["patient_feature"],
        diagnosis_view_features=tensors["diagnosis_view_features"],
        diagnosis_clip_features=tensors["diagnosis_clip_features"],
        clip_mask=tensors["clip_mask"],
        diagnosis_logits=tensors["diagnosis_logits"],
        view_dropout=view_dropout,
    )
    return out["risk"].detach().cpu().numpy().astype(np.float32), out


def evaluate_survival_array(meta: pd.DataFrame, risk: np.ndarray, train_threshold: float, horizons):
    time_np = meta["time_to_event"].values.astype(float)
    event_np = meta["event"].values.astype(int)
    group = (np.asarray(risk) >= float(train_threshold)).astype(int)
    result = {
        "n": int(len(meta)),
        "events": int(event_np.sum()),
        "cindex": concordance_index(time_np, event_np, risk),
        "logrank_p": logrank_p_value(time_np, event_np, group),
    }
    for h in horizons:
        auc, n_h, e_h = hcm_horizon_auc(time_np, event_np, risk, int(h))
        y = int(round(int(h) / 365.0))
        result[f"auc_{y}y"] = auc
        result[f"auc_{y}y_n"] = n_h
        result[f"auc_{y}y_events"] = e_h
    return result


def train_one_multilevel_seed(data_by_split, args, variant: str, seed: int, out_dir: Path, device):
    seed_everything(int(seed))
    tr = split_tensors(data_by_split["train"], device)
    va = split_tensors(data_by_split["val"], device)
    te = split_tensors(data_by_split["test"], device)
    dim = int(tr["patient_feature"].shape[-1])
    selected_views = parse_csv_list(args.selected_views)
    slice_plan = parse_slice_plan(args.slice_plan)
    ranges, start = [], 0
    for view in selected_views:
        end = start + int(slice_plan[view])
        ranges.append((start, end))
        start = end

    if variant == "diagnosis_patient_only":
        model = DiagnosisPatientCoxBaseline(
            dim=dim, hidden_dim=int(args.multilevel_head_hidden), dropout=float(args.multilevel_dropout)
        ).to(device)
    elif variant == "diagnosis_multilevel_residual":
        model = DiagnosisMultilevelSurvivalResidual(
            dim=dim,
            num_views=len(selected_views),
            view_ranges=ranges,
            adapter_dim=int(args.multilevel_adapter_dim),
            attention_hidden=int(args.multilevel_attention_hidden),
            num_heads=int(args.multilevel_attention_heads),
            head_hidden=int(args.multilevel_head_hidden),
            dropout=float(args.multilevel_dropout),
            gate_init=float(args.multilevel_residual_gate_init),
            gate_max=float(args.multilevel_residual_gate_max),
            clip_gate_init=float(args.multilevel_clip_gate_init),
            clip_gate_max=float(args.multilevel_clip_gate_max),
        ).to(device)
    else:
        raise ValueError(f"Unknown variant={variant}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.multilevel_lr),
        weight_decay=float(args.multilevel_weight_decay),
    )
    cox = CoxPHLoss()
    all_mask = torch.ones_like(tr["event"], dtype=torch.bool)
    pair_i, pair_j = build_comparable_pairs(tr["time"], tr["event"])
    pair_i, pair_j = pair_i.to(device), pair_j.to(device)

    best_val, best_epoch, stale = -1e9, 0, 0
    best_state = None
    history = []
    for epoch in range(1, int(args.multilevel_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(
            patient_feature=tr["patient_feature"],
            diagnosis_view_features=tr["diagnosis_view_features"],
            diagnosis_clip_features=tr["diagnosis_clip_features"],
            clip_mask=tr["clip_mask"],
            diagnosis_logits=tr["diagnosis_logits"],
            view_dropout=float(args.multilevel_view_dropout),
        )
        risk = out["risk"]
        loss_cox = cox(risk, tr["time"], tr["event"], all_mask)
        loss_rank = pairwise_rank_loss(
            risk, pair_i, pair_j,
            tau=float(args.multilevel_rank_tau),
            max_pairs=int(args.multilevel_rank_max_pairs),
        )
        loss = loss_cox + float(args.multilevel_rank_weight) * loss_rank
        if "residual" in out and float(args.multilevel_residual_l2) > 0:
            loss = loss + float(args.multilevel_residual_l2) * out["residual"].pow(2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(args.multilevel_grad_clip))
        optimizer.step()

        if epoch == 1 or epoch % int(args.multilevel_eval_every) == 0:
            val_risk, val_out = predict_multilevel_model(model, va)
            val_c = concordance_index(
                data_by_split["val"]["meta"]["time_to_event"].values,
                data_by_split["val"]["meta"]["event"].values,
                val_risk,
            )
            rec = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_cox": float(loss_cox.item()),
                "loss_rank": float(loss_rank.item()),
                "val_cindex": float(val_c),
                "residual_gate": float(val_out["residual_gate"].detach().cpu()),
                "clip_gate": float(val_out["clip_gate"].detach().cpu()),
            }
            history.append(rec)
            if epoch == 1 or epoch % max(int(args.print_freq), int(args.multilevel_eval_every)) == 0:
                print0(
                    f"{variant} seed={seed} epoch={epoch:03d} | loss={rec['loss']:.4f} "
                    f"cox={rec['loss_cox']:.4f} rank={rec['loss_rank']:.4f} "
                    f"val_c={rec['val_cindex']:.4f} gates={rec['residual_gate']:.3f}/{rec['clip_gate']:.3f}"
                )
            if val_c > best_val + float(args.early_stop_min_delta):
                best_val = float(val_c)
                best_epoch = int(epoch)
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += int(args.multilevel_eval_every)
                if stale >= int(args.multilevel_patience):
                    break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {variant}, seed={seed}")
    model.load_state_dict(best_state)
    ensure_dir(out_dir)
    torch.save({
        "model": best_state, "variant": variant, "seed": int(seed),
        "best_epoch": best_epoch, "best_val_cindex": best_val,
        "config": vars(args),
    }, out_dir / f"{variant}_seed{seed}_best.pt")
    pd.DataFrame(history).to_csv(out_dir / f"{variant}_seed{seed}_history.csv", index=False, encoding="utf-8-sig")

    risks, last_out = {}, None
    for sp, tensors in [("train", tr), ("val", va), ("test", te)]:
        risks[sp], last_out = predict_multilevel_model(model, tensors)
    threshold = float(np.median(risks["train"]))
    horizons = parse_horizons_days(args.eval_horizons_days)
    metrics = {
        "variant": variant,
        "seed": int(seed),
        "best_epoch": best_epoch,
        "best_val_cindex": best_val,
        "residual_gate": float(last_out["residual_gate"].detach().cpu()),
        "clip_gate": float(last_out["clip_gate"].detach().cpu()),
    }
    for sp in ["train", "val", "test"]:
        ev = evaluate_survival_array(data_by_split[sp]["meta"], risks[sp], threshold, horizons)
        for k, v in ev.items():
            metrics[f"{sp}_{k}"] = v
    return metrics, risks


def zscore_risk_from_train(train_risk, risk):
    mu = float(np.mean(train_risk))
    sd = float(np.std(train_risk))
    if sd < 1e-8:
        sd = 1.0
    return ((np.asarray(risk) - mu) / sd).astype(np.float32)


def save_multilevel_predictions(meta, risk, threshold, path, variant):
    df = meta.copy()
    df["risk"] = np.asarray(risk, dtype=float)
    df["risk_threshold"] = float(threshold)
    df["risk_group"] = np.where(df["risk"] >= float(threshold), "high", "low")
    df["variant"] = variant
    df.to_csv(path, index=False, encoding="utf-8-sig")


def aggregate_multilevel_seed_risks(data_by_split, risk_store, args, run_dir):
    rows = []
    horizons = parse_horizons_days(args.eval_horizons_days)
    variants = sorted({k[0] for k in risk_store.keys()})
    for variant in variants:
        seeds = sorted(k[1] for k in risk_store.keys() if k[0] == variant)
        ensemble = {}
        for sp in ["train", "val", "test"]:
            standardized = [
                zscore_risk_from_train(risk_store[(variant, seed)]["train"], risk_store[(variant, seed)][sp])
                for seed in seeds
            ]
            ensemble[sp] = np.mean(np.stack(standardized, axis=0), axis=0).astype(np.float32)
        threshold = float(np.median(ensemble["train"]))
        row = {"variant": variant, "aggregation": "train_zscore_mean", "n_seeds": len(seeds)}
        for sp in ["train", "val", "test"]:
            ev = evaluate_survival_array(data_by_split[sp]["meta"], ensemble[sp], threshold, horizons)
            for k, v in ev.items():
                row[f"{sp}_{k}"] = v
            save_multilevel_predictions(
                data_by_split[sp]["meta"], ensemble[sp], threshold,
                run_dir / f"risk_predictions_{sp}_{variant}_ensemble.csv", variant,
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    if {"diagnosis_patient_only", "diagnosis_multilevel_residual"}.issubset(set(summary["variant"])):
        base = summary[summary["variant"] == "diagnosis_patient_only"].iloc[0]
        idx = summary["variant"] == "diagnosis_multilevel_residual"
        summary.loc[idx, "delta_val_cindex_vs_patient_only"] = summary.loc[idx, "val_cindex"] - float(base["val_cindex"])
        summary.loc[idx, "delta_test_cindex_vs_patient_only"] = summary.loc[idx, "test_cindex"] - float(base["test_cindex"])
    summary.to_csv(run_dir / "diagnosis_multilevel_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def diagnosis_multilevel_survival_main():
    args = parse_args()
    disease = "HCM" if str(args.disease_mode).lower().startswith("hcm") else "DCM"
    expected_ckpt = str(RUN_CONFIG["classification_ckpt"])
    expected_split = str(RUN_CONFIG["split_source_dir"])
    if str(args.classification_ckpt) != expected_ckpt or str(args.split_source_dir) != expected_split:
        raise RuntimeError(
            "This experiment locks the top RUN_CONFIG checkpoint and split.\n"
            f"resolved ckpt={args.classification_ckpt}\nresolved split={args.split_source_dir}"
        )
    if not Path(args.classification_ckpt).is_file():
        raise FileNotFoundError(f"classification_ckpt not found: {args.classification_ckpt}")
    if not Path(args.split_source_dir).is_dir():
        raise FileNotFoundError(f"split_source_dir not found: {args.split_source_dir}")

    device = configure_runtime_gpu(str(args.visible_gpus or ""), device_arg="auto")
    seed_everything(int(args.seed))
    stamp = now_string()
    out_base = str(args.output_dir or "auto")
    if out_base.lower() in {"", "auto", "none"}:
        run_dir = ensure_dir(Path.cwd() / f"20260726_{disease.lower()}_diagnosis_multilevel_survival_results_{stamp}" / str(args.experiment_name))
    else:
        run_dir = ensure_dir(Path(out_base) / str(args.experiment_name))

    print0("=" * 88)
    print0(f"{disease} diagnosis multilevel survival residual experiment")
    print0(f"Run dir: {run_dir}")
    print0(f"split_source_dir   : {args.split_source_dir}")
    print0(f"classification_ckpt: {args.classification_ckpt}")
    print0(f"device={device}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print0("Exact diagnosis path: clip backbone -> view embedding -> diagnosis slice pool -> diagnosis view pool.")
    print0("Trainable prognosis path: diagnosis patient feature + bounded multilevel survival residual + Cox/ranking.")
    print0("=" * 88)

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    selected_views, required_views, slice_plan, split_to_samples = collect_split_samples_for_args(args)
    print0("Survival split summary:")
    for sp in ["train", "val", "test"]:
        print0(f"  {sp}: {summarize_survival_samples(split_to_samples[sp])}")
        save_survival_split_csv(split_to_samples[sp], run_dir / f"split_{sp}_survival.csv")

    data_by_split, feature_dir = extract_or_load_multilevel(
        args, split_to_samples, selected_views, required_views, slice_plan, run_dir, device
    )
    audit_multilevel_data(data_by_split, args)
    with open(run_dir / "resolved_feature_dir.json", "w", encoding="utf-8") as f:
        json.dump({"feature_dir": str(feature_dir)}, f, ensure_ascii=False, indent=2)

    variants = [x.strip() for x in str(args.multilevel_variants).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.multilevel_seeds).split(",") if x.strip()]
    head_dir = ensure_dir(run_dir / "survival_heads")
    all_metrics, risk_store = [], {}
    for variant in variants:
        for seed in seeds:
            print0("\n" + "-" * 88)
            print0(f"Training {disease} | variant={variant} | seed={seed}")
            metrics, risks = train_one_multilevel_seed(
                data_by_split, args, variant, seed, head_dir, device
            )
            all_metrics.append(metrics)
            risk_store[(variant, seed)] = risks
            print0(json.dumps({
                "variant": variant,
                "seed": seed,
                "best_epoch": metrics["best_epoch"],
                "val_cindex": metrics["val_cindex"],
                "test_cindex": metrics["test_cindex"],
                "test_logrank_p": metrics["test_logrank_p"],
                "test_auc_1y": metrics.get("test_auc_1y"),
                "test_auc_3y": metrics.get("test_auc_3y"),
                "test_auc_5y": metrics.get("test_auc_5y"),
                "residual_gate": metrics["residual_gate"],
                "clip_gate": metrics["clip_gate"],
            }, ensure_ascii=False, indent=2))

    pd.DataFrame(all_metrics).to_csv(
        run_dir / "diagnosis_multilevel_single_seed_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary = aggregate_multilevel_seed_risks(data_by_split, risk_store, args, run_dir)
    print0("\n========== Diagnosis multilevel ensemble summary ==========")
    show = [c for c in [
        "variant", "aggregation", "n_seeds", "val_cindex", "test_cindex", "test_logrank_p",
        "test_auc_1y", "test_auc_2y", "test_auc_3y", "test_auc_4y", "test_auc_5y",
        "delta_val_cindex_vs_patient_only", "delta_test_cindex_vs_patient_only",
    ] if c in summary.columns]
    print0(summary[show].to_string(index=False))
    print0(f"\nSaved all outputs to: {run_dir}")


if __name__ == "__main__":
    from prognosis_fullrisk_teacher_adapt_oof_fixed import run as prognosis_stage_run
    prognosis_stage_run(globals())
