#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-view Cine CMR three-class classification with Video Swin.

FUSION-BASELINE VERSION:
    Based on the original user code, with additional aggregation/fusion baselines
    for full-view comparison: flat_mean, flat_mil, hier_mean, hier_attn,
    concat_mlp, late_fusion, view_transformer.

Task:
    Patient-level NC / HCM / DCM classification from NIfTI cine MRI.

Data structure:
    root/
      NC/patient_x/Cine4CH-15_1_xxx.nii ...
      HCM/patient_y/CineSAX-6_25_xxx.nii ...
      DCM/patient_z/...

Filename format:
    Cine2CH-6_1_cine_tf2d14_retro_iPAT.nii
    view      = Cine2CH
    slice_idx = 6
    frame_idx = 1

Main design:
    patient -> views -> slices -> frames
    each slice cine clip -> shared torchvision swin3d_t -> slice feature
    slice aggregation -> view feature
    view aggregation  -> patient feature
    classifier        -> NC/HCM/DCM

Supports:
    - single GPU: python multiview_cardiac_swin_train_full.py ...
    - DDP multi-GPU: torchrun --nproc_per_node=8 multiview_cardiac_swin_train_full.py ...
    - local/offline Video Swin weight loading, no internet download
    - practical available-view experiments and complete-case strict cohort experiments
    - zero padding + masks for insufficient slices
    - NIfTI preprocessed clip cache to reduce CPU/IO bottleneck
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

RUN_CONFIG = {
    # ---------------- GPU ----------------
    # "gpu_mode": "single",                 # "single" 或 "ddp"
    # "visible_gpus": "0",                  # 单卡 "0"；8卡 "0,1,2,3,4,5,6,7"
    "gpu_mode": "ddp",                 # "single" 或 "ddp"
    "visible_gpus": "0,1,2,3",                  # 单卡 "0"；8卡 "0,1,2,3,4,5,6,7"
    "auto_launch_ddp": True,               # gpu_mode="ddp" 且用 python 启动时，自动拉起 torchrun

    # ---------------- Experiment mode ----------------
    # single：只跑下面 model_type 指定的一个模型；compare：按 compare_experiments 顺序依次跑多个模型。
    # compare 模式会复用同一套 data_path / selected_views / slice_plan / seed，保证模型对比的数据一致。
    "experiment_mode": "compare",           # "single" 或 "compare"
    # "compare_experiments": [
    #     {"experiment_name": "exp_videoswin_4ch_sax", "model_type": "videoswin"},
    #     {"experiment_name": "exp_resnet18_4ch_sax", "model_type": "resnet18", "pretrained": False},
    #     {"experiment_name": "exp_resnet50_4ch_sax", "model_type": "resnet50", "pretrained": False},
    #     {"experiment_name": "exp_densenet121_4ch_sax", "model_type": "densenet121", "pretrained": False},
    #     {"experiment_name": "exp_efficientnet_b0_4ch_sax", "model_type": "efficientnet_b0", "pretrained": False},
    #     {"experiment_name": "exp_convnext_tiny_4ch_sax", "model_type": "convnext_tiny", "pretrained": False},
    # ],

    "compare_experiments": [
        # ------------------------------------------------------------------
        # Multi-view fusion baselines: same cohort/split/backbone, only agg differs.
        # 这一组最关键：直接回答“你的 two-stage view-aware pooling 是否优于 flat MIL / simple fusion”。
        # ------------------------------------------------------------------
        {
            "experiment_name": "auto",
            "model_type": "videoswin",
            "agg": "flat_mean",
            "pretrained": True,
            "weights_path": "/data/projects/MRI/checkpoints/swin3d_t-7615ae03.pth",
        },
        {
            "experiment_name": "auto",
            "model_type": "videoswin",
            "agg": "flat_mil",
            "pretrained": True,
            "weights_path": "/data/projects/MRI/checkpoints/swin3d_t-7615ae03.pth",
        },
        {
            "experiment_name": "auto",
            "model_type": "videoswin",
            "agg": "hier_mean",
            "pretrained": True,
            "weights_path": "/data/projects/MRI/checkpoints/swin3d_t-7615ae03.pth",
        },
        {
            "experiment_name": "auto",
            "model_type": "videoswin",
            "agg": "hier_attn",
            "pretrained": True,
            "weights_path": "/data/projects/MRI/checkpoints/swin3d_t-7615ae03.pth",
        },
        {
            "experiment_name": "auto",
            "model_type": "videoswin",
            "agg": "concat_mlp",
            "pretrained": True,
            "weights_path": "/data/projects/MRI/checkpoints/swin3d_t-7615ae03.pth",
        },
        {
            "experiment_name": "auto",
            "model_type": "videoswin",
            "agg": "late_fusion",
            "pretrained": True,
            "weights_path": "/data/projects/MRI/checkpoints/swin3d_t-7615ae03.pth",
        },
        {
            "experiment_name": "auto",
            "model_type": "videoswin",
            "agg": "view_transformer",
            "pretrained": True,
            "weights_path": "/data/projects/MRI/checkpoints/swin3d_t-7615ae03.pth",
        },
    ],

    # ---------------- Data ----------------
    "data_path": "/data/datasets/CMR/Project-FM",
    "selected_views": "Cine2CH,Cine3CH,Cine4CH,CineSAX",#,Cine3CH,Cine4CH,CineSAX
    # 真实应用版：留空即可；严格公平版：例如 "Cine2CH,Cine3CH,Cine4CH,CineSAX"
    "cohort_required_views": "Cine2CH,Cine3CH,Cine4CH,CineSAX",
    # False = 必须拥有 selected_views 里的全部视图才入组；True = 有至少一个 selected view 就入组，缺失视图用 mask 忽略
    "allow_missing_selected_views": False,
    "slice_plan": "Cine2CH:3,Cine3CH:3,Cine4CH:3,CineSAX:8",
    "num_frames": 13,
    "image_size": 224,
    "min_frames_per_slice": 1,
    "use_cache": True,
    "cache_dir": "",                       # 留空则默认 data_path/.cmr_clip_cache
    "augment": True,

    # ---------------- Split ----------------
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "seed": 42,

    # ---------------- Model ----------------
    # model_type 可选：videoswin / resnet18 / resnet50 / densenet121 / efficientnet_b0 / convnext_tiny
    # videoswin 是本文主模型；其他是 2D CNN 对比模型，仍使用同一套 view-slice-frame 聚合框架。
    "model_type": "videoswin",
    "model_name": "",
    "model_weights_path": "",              # 2D CNN 对比模型的本地权重；留空则随机初始化/不加载
    "frame_samples": 0,                     # 2D CNN 每个 slice clip 取几帧；0 表示使用 num_frames 的全部帧，例如 num_frames=8 就用8帧

    # ---------------- Prognosis interface, default off ----------------
    # 先做三分类时留空；后面做预后/生存分析时填入总 CSV，即可把 event/time/censor 信息接入 Dataset。
    "prognosis_csv": "",
    "prognosis_id_col": "受试者ID",
    "prognosis_event_col": "是否发生结局事件",
    "prognosis_time_col": "发生时间",
    "prognosis_censor_col": "是否为删失样本",

    "pretrained": True,
    "weights_path": "/data/projects/MRI/checkpoints/swin3d_t-7615ae03.pth",
    "agg": "hier_attn",                    # fusion: hier_attn/hier_mean/flat_mil/flat_mean/concat_mlp/late_fusion/view_transformer; 兼容旧值 attention/mean
    "dropout": 0.35,
    "freeze_backbone": False,
    "backbone_chunk_size": 0,               # 显存不够时可设 8/16/32；0 表示不切块
    "compile": False,

    # ---------------- Optimization ----------------
    "epochs": 50,
    "batch_size": 1,                        # DDP 时是每张卡 batch size
    "workers": 4,                           # DDP 时是每个进程/每张卡的 workers
    "lr": 1e-4,
    "backbone_lr_mult": 0.1,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "label_smoothing": 0.0,
    "grad_accum_steps": 1,
    "grad_clip": 1.0,
    "amp": True,
    "scheduler": "cosine",                 # "cosine" / "step" / "none"
    "early_stop_patience": 8,              # 早停耐心轮数；<=0 表示关闭早停
    "early_stop_min_delta": 1e-4,           # val_macro_f1 至少提升这么多才算 improved

    # ---------------- Visualization ----------------
    "plot_roc": True,
    "plot_history": True,
    "save_attention_viz": False,
    "vis_num_samples": 0,

    # ---------------- IO ----------------
    # output_dir="auto" 时，会自动生成：./当前脚本文件名_results_YYYYMMDD_HHMMSS/experiment_name
    # 例如：./multiview_cardiac_swin_train_configurable_results_20260424_153012/exp_4ch_sax
    "output_dir": "./final_diagnosis_results/fusion_strategy_runs",
    "experiment_name": "fusion_strategy_full_view",  # compare 子实验会覆盖该名称
    "run_id": "fusion_full4view_seed42",    # 可手动修改，留空则自动用当前时间
    "resume": "",
    "eval_only": False,
    "print_freq": 20,
}

# 支持对比实验子进程用环境变量覆盖 RUN_CONFIG。必须发生在 import torch 之前。
def _apply_env_run_config_overrides():
    raw = os.environ.get("CMR_RUN_CONFIG_OVERRIDES", "").strip()
    if not raw:
        return
    try:
        overrides = json.loads(raw)
        if isinstance(overrides, dict):
            RUN_CONFIG.update(overrides)
    except Exception as e:
        print(f"[WARN] CMR_RUN_CONFIG_OVERRIDES 解析失败: {e}")

_apply_env_run_config_overrides()

# 必须在 import torch 之前设置可见 GPU
if RUN_CONFIG.get("visible_gpus", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(RUN_CONFIG["visible_gpus"])

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.models.video as video_models
import torchvision.models as tv_models
try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# -----------------------------
# Basic utils
# -----------------------------

CLASS_TO_IDX = {"NC": 0, "HCM": 1, "DCM": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
ALL_VIEWS = ["Cine2CH", "Cine3CH", "Cine4CH", "CineSAX"]
DEFAULT_SLICE_PLAN = {"Cine2CH": 3, "Cine3CH": 3, "Cine4CH": 3, "CineSAX": 8}


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
    """Initialize DDP if launched with torchrun."""
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size, distributed


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
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


def parse_slice_plan(s: str) -> Dict[str, int]:
    if s is None or str(s).strip() == "":
        return dict(DEFAULT_SLICE_PLAN)
    plan = dict(DEFAULT_SLICE_PLAN)
    for item in s.split(","):
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


def validate_views(views: Sequence[str]):
    for v in views:
        if v not in ALL_VIEWS:
            raise ValueError(f"Unknown view: {v}. Supported: {ALL_VIEWS}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def now_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_shared_run_id(user_run_id: str = "") -> str:
    """
    生成所有进程一致的实验时间戳。
    - 单卡：直接用当前时间；
    - python 自动拉起 DDP：父进程会通过 CMR_RUN_ID 传给所有子进程；
    - 手动 torchrun：rank0 生成后 broadcast 给其他 rank，避免不同进程目录不一致。
    """
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
    """
    默认按“当前代码文件名 + 时间”创建结果目录，方便同时跑很多版。
    output_dir="auto" 或留空：
        ./当前脚本文件名_results_YYYYMMDD_HHMMSS/experiment_name
    output_dir="./xxx"：
        ./xxx/experiment_name
    """
    run_id = get_shared_run_id(getattr(args, "run_id", ""))
    script_stem = Path(sys.argv[0]).stem

    output_dir = str(getattr(args, "output_dir", "auto") or "auto").strip()
    if output_dir.lower() in {"", "auto", "none"}:
        root = Path(f"./{script_stem}_results_{run_id}")
    else:
        root = Path(output_dir)

    exp_name = str(getattr(args, "experiment_name", "") or "").strip()
    if exp_name.lower() in {"", "auto", "none"}:
        # view_tag = "_".join([v.replace("Cine", "") for v in selected_views])
        # exp_name = f"exp_{view_tag}"
        # 自动生成名称：exp_模型名_视图名
        view_tag = "_".join([v.replace("Cine", "") for v in selected_views])
        model_tag = str(getattr(args, "model_type", "model")).lower()
        exp_name = f"exp_{model_tag}_{view_tag}"
        
        # # ====== 核心修改：把真实的拼装名字覆盖回 args 里！ ======
    args.experiment_name = exp_name

    return root / exp_name

def safe_torch_load(path, map_location="cpu", weights_only=True):
    """
    兼容不同 PyTorch 版本的 torch.load。
    新版本使用 weights_only=True，避免 FutureWarning；
    老版本如果不支持该参数，则自动回退。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        # 某些旧 checkpoint 如果不是纯 tensor/state_dict，weights_only=True 可能失败
        # 这种情况下再回退到普通 torch.load
        if weights_only:
            return torch.load(path, map_location=map_location)
        raise

# -----------------------------
# NIfTI dataset
# -----------------------------

class MultiViewCardiacNiftiDataset(Dataset):
    """
    Returns fixed patient-level tensors:
        videos:        [Nclips, 1, T, H, W]
        clip_mask:     [Nclips], 1 for real slice, 0 for padding/missing slice
        clip_view_ids: [Nclips], local selected view index, fixed by selected_views
        view_mask:     [V], 1 if view has at least one real slice, else 0
        label:         scalar long
    """

    filename_pattern = re.compile(r"^(Cine(?:2CH|3CH|4CH|SAX))-(\d+)_(\d+)_.*\.nii(?:\.gz)?$")

    def __init__(
        self,
        root_dir: str,
        selected_views: Sequence[str],
        slice_plan: Dict[str, int],
        num_frames: int = 13,
        image_size: int = 224,
        samples: Optional[List[Dict]] = None,
        mode: str = "train",
        required_views: Optional[Sequence[str]] = None,
        allow_missing_selected_views: bool = False,
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
        augment: bool = False,
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
        self.allow_missing_selected_views = allow_missing_selected_views
        self.required_views = list(required_views) if required_views else []
        validate_views(self.required_views)
        self.use_cache = bool(use_cache)
        self.cache_dir = Path(cache_dir) if cache_dir else self.root_dir / ".cmr_clip_cache"
        self.augment = bool(augment and mode == "train")
        self.min_frames_per_slice = int(min_frames_per_slice)
        self.verbose = verbose

        self.nclips = sum(int(self.slice_plan.get(v, 0)) for v in self.selected_views)
        self.clip_view_ids_template = self._build_clip_view_ids_template()
        self.view_clip_ranges = self._build_view_clip_ranges()

        if self.use_cache:
            ensure_dir(self.cache_dir)

        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._collect_samples()

    def _build_clip_view_ids_template(self) -> torch.Tensor:
        ids = []
        for local_vid, v in enumerate(self.selected_views):
            k = int(self.slice_plan.get(v, 0))
            ids.extend([local_vid] * k)
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

    def _patient_has_view(self, patient_data: Dict, view: str) -> bool:
        if view not in patient_data or not patient_data[view]:
            return False
        for _, frames in patient_data[view].items():
            if len(frames) >= self.min_frames_per_slice:
                return True
        return False

    def _is_valid_patient(self, patient_data: Dict) -> bool:
        if self.required_views:
            required = self.required_views
        elif self.allow_missing_selected_views:
            required = []
        else:
            required = self.selected_views

        for v in required:
            if not self._patient_has_view(patient_data, v):
                return False

        if self.allow_missing_selected_views and not required:
            return any(self._patient_has_view(patient_data, v) for v in self.selected_views)
        return True

    def _collect_samples(self) -> List[Dict]:
        samples = []
        if self.verbose:
            print0("开始扫描 NIfTI 数据...")
            print0(f"  selected_views = {self.selected_views}")
            if self.required_views:
                print0(f"  cohort_required_views = {self.required_views}")
            elif self.allow_missing_selected_views:
                print0("  allow_missing_selected_views = True: 只要至少存在一个 selected view 即可")
            else:
                print0("  require selected_views: 每个样本必须具备当前实验所选视图")

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
        # robust percentile normalization per frame
        p1, p99 = np.percentile(img, (1, 99))
        if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
            mn, mx = float(np.min(img)), float(np.max(img))
            denom = mx - mn
            if denom < 1e-8:
                return np.zeros_like(img, dtype=np.float32)
            img = (img - mn) / (denom + 1e-8)
            return img.astype(np.float32)
        img = np.clip(img, p1, p99)
        img = (img - p1) / (p99 - p1 + 1e-8)
        return img.astype(np.float32)

    def resize_with_pad(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, H, W]
        _, h, w = x.shape
        if h == self.image_size and w == self.image_size:
            return x
        scale = min(self.image_size / max(h, 1), self.image_size / max(w, 1))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        x = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

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
            x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
            x = self.resize_with_pad(x)
            # ImageNet-style normalization after [0,1]
            x = (x - 0.485) / 0.229
            return x.float()
        except Exception as e:
            # Avoid printing too much in multi-worker training.
            return torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)

    def _cache_key_for_clip(self, frames_info: Sequence[Dict]) -> str:
        # Include selected frame file paths, image size and T in the key.
        raw = "|".join([f'{x["frame_idx"]}:{x["file_path"]}' for x in frames_info])
        raw += f"|T={self.num_frames}|S={self.image_size}|norm=v2"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _select_frame_infos(self, frames_info: Sequence[Dict]) -> List[Dict]:
        frames_info = sorted(list(frames_info), key=lambda x: x["frame_idx"])
        n = len(frames_info)
        if n <= 0:
            return []
        if n >= self.num_frames:
            idx = np.linspace(0, n - 1, self.num_frames, dtype=int).tolist()
            return [frames_info[i] for i in idx]
        # Not enough frames: repeat last by linspace over available frames.
        idx = np.linspace(0, n - 1, self.num_frames, dtype=int).tolist()
        return [frames_info[i] for i in idx]

    def _load_one_slice_clip(self, frames_info: Sequence[Dict]) -> torch.Tensor:
        """Return [1, T, H, W]."""
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

        frames = [self._load_frame(x["file_path"]) for x in selected]  # each [1,H,W]
        clip = torch.stack(frames, dim=1).contiguous()  # [1,T,H,W]

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
        # Not enough slices: use real slices only. The rest will be zero-padded with mask=0.
        return valid_slice_ids

    def _augment_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """Apply same random affine/brightness to all frames of one slice clip. Input [1,T,H,W]."""
        if not self.augment:
            return clip
        angle = random.uniform(-5.0, 5.0)
        trans_x = random.randint(-5, 5)
        trans_y = random.randint(-5, 5)
        scale = random.uniform(0.98, 1.02)
        brightness = random.uniform(0.85, 1.15)

        c, t, h, w = clip.shape
        frames = clip.permute(1, 0, 2, 3).contiguous()  # [T,1,H,W]
        out = []
        for i in range(t):
            x = frames[i]
            x = TF.pad(x, padding=16, fill=0.0)
            x = TF.affine(
                x,
                angle=angle,
                translate=[trans_x, trans_y],
                scale=scale,
                shear=0.0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            x = TF.center_crop(x, [self.image_size, self.image_size])
            x = torch.clamp(x * brightness, -3.0, 3.0)
            out.append(x)
        return torch.stack(out, dim=1).contiguous()  # [1,T,H,W]

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        pdata = sample["organized_data"]

        videos: List[torch.Tensor] = []
        clip_mask: List[int] = []
        view_mask: List[int] = []

        for v in self.selected_views:
            k = int(self.slice_plan.get(v, 0))
            if k <= 0:
                view_mask.append(0)
                continue

            view_slices = pdata.get(v, {})
            chosen_slice_ids = self._select_slice_ids(view_slices, k)
            real_count = 0

            for sid in chosen_slice_ids:
                clip = self._load_one_slice_clip(view_slices[sid])
                clip = self._augment_clip(clip)
                videos.append(clip)
                clip_mask.append(1)
                real_count += 1

            # Zero padding for insufficient slices.
            num_pad = k - len(chosen_slice_ids)
            for _ in range(num_pad):
                videos.append(torch.zeros(1, self.num_frames, self.image_size, self.image_size, dtype=torch.float32))
                clip_mask.append(0)

            view_mask.append(1 if real_count > 0 else 0)

        if len(videos) != self.nclips:
            raise RuntimeError(f"Internal error: got {len(videos)} clips, expected {self.nclips}")

        return {
            "videos": torch.stack(videos, dim=0).float(),  # [N,1,T,H,W]
            "clip_mask": torch.tensor(clip_mask, dtype=torch.bool),
            "clip_view_ids": self.clip_view_ids_template.clone(),
            "view_mask": torch.tensor(view_mask, dtype=torch.bool),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "patient_id": sample["patient_id"],
            "class_name": sample["class_name"],
            "available_views": sample["available_views"],
            "prog_available": torch.tensor(sample.get("prog_available", 0), dtype=torch.float32),
            "event": torch.tensor(sample.get("event", -1), dtype=torch.float32),
            "time_to_event": torch.tensor(sample.get("time_to_event", -1), dtype=torch.float32),
            "censored": torch.tensor(sample.get("censored", -1), dtype=torch.float32),
        }


# -----------------------------
# Splitting
# -----------------------------

def stratified_split_samples(
    samples: List[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for s in samples:
        by_label[int(s["label"])].append(s)

    train, val, test = [], [], []
    for label, items in sorted(by_label.items()):
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

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def save_split_csv(samples: List[Dict], path: Path):
    rows = []
    for s in samples:
        rows.append({
            "patient_id": s["patient_id"],
            "class_name": s["class_name"],
            "label": s["label"],
            "available_views": ";".join(s.get("available_views", [])),
            "patient_dir": s.get("patient_dir", ""),
            "prog_available": s.get("prog_available", 0),
            "event": s.get("event", -1),
            "time_to_event": s.get("time_to_event", -1),
            "censored": s.get("censored", -1),
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def class_count_string(samples: List[Dict]) -> str:
    counts = defaultdict(int)
    for s in samples:
        counts[s["class_name"]] += 1
    return ", ".join([f"{k}:{counts.get(k, 0)}" for k in ["NC", "HCM", "DCM"]])


# -----------------------------
# Model
# -----------------------------

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
        """
        x:    [B, N, D]
        mask: [B, N] bool, True=valid
        returns pooled [B,D], weights [B,N]
        """
        mask = mask.bool()
        scores = self.score(x).squeeze(-1)  # [B,N]
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weights = weights / denom
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        # If a row is all invalid, return zeros.
        has_any = mask.any(dim=1, keepdim=True).float()
        pooled = pooled * has_any
        weights = weights * has_any
        return pooled, weights


class MaskedMeanPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        w = mask.float()
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (x * w.unsqueeze(-1)).sum(dim=1) / denom
        weights = w / denom
        pooled = pooled * mask.any(dim=1, keepdim=True).float()
        return pooled, weights


FUSION_AGG_CHOICES = [
    "hier_attn",        # your current method: per-view slice attention -> view attention
    "hier_mean",        # per-view slice mean -> view mean
    "flat_mil",         # all clips/slices/views as one MIL bag with attention
    "flat_mean",        # all clips/slices/views as one bag with mean pooling
    "concat_mlp",       # per-view features concatenated then classified by MLP
    "late_fusion",      # per-view logits averaged by valid-view mask
    "view_transformer", # per-view tokens go through a small Transformer encoder
]


def canonicalize_agg(agg: str) -> str:
    """Keep old configs runnable while making the fusion strategy explicit."""
    a = str(agg or "hier_attn").lower().strip()
    aliases = {
        "attention": "hier_attn",
        "attn": "hier_attn",
        "mean": "hier_mean",
        "mil": "flat_mil",
        "flat_attention": "flat_mil",
        "flat_attn": "flat_mil",
        "hier_attention": "hier_attn",
        "hierarchical_attention": "hier_attn",
        "hierarchical_mean": "hier_mean",
        "transformer": "view_transformer",
        "view_token_transformer": "view_transformer",
        "late": "late_fusion",
    }
    a = aliases.get(a, a)
    if a not in FUSION_AGG_CHOICES:
        raise ValueError(f"Unknown agg='{agg}'. Supported: {FUSION_AGG_CHOICES}; old aliases attention/mean are also accepted.")
    return a


class ViewTokenTransformerPool(nn.Module):
    """Small masked Transformer over view-level tokens.

    This is a reasonable multi-view fusion baseline, not the proposed default.
    It keeps the backbone and per-view slice aggregation identical, then replaces
    view attention pooling with a shallow Transformer over view tokens.
    """
    def __init__(self, dim: int, num_views: int, num_heads: int = 4, depth: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_views = int(num_views)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_views + 1, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=max(1, min(num_heads, dim // 64 if dim >= 64 else 1)),
            dim_feedforward=max(dim * 2, 256),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, view_feats: torch.Tensor, view_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        view_feats: [B,V,D]
        view_mask:  [B,V] bool, True=valid
        returns patient_feat [B,D], diagnostic view_weights [B,V]
        """
        B, V, D = view_feats.shape
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, view_feats], dim=1) + self.pos_embed[:, :V + 1, :]
        # src_key_padding_mask: True means masked/ignored. CLS is always valid.
        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=view_feats.device),
            ~view_mask.bool(),
        ], dim=1)
        out = self.encoder(tokens, src_key_padding_mask=pad_mask)
        patient_feat = self.norm(out[:, 0, :])
        # PyTorch TransformerEncoder does not expose attention weights; use valid-view prior for reporting.
        w = view_mask.float()
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1.0)
        return patient_feat, w


class MultiViewAggregationMixin:
    """Shared multi-view aggregation code for VideoSwin and 2D CNN backbones."""

    def _make_classifier_head(self, input_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _init_multiview_aggregation(self, num_classes: int, agg: str, dropout: float):
        self.agg = canonicalize_agg(agg)

        # Only instantiate modules that are actually used by the chosen fusion strategy.
        # This avoids DDP "unused parameter" errors when find_unused_parameters=False.
        if self.agg == "hier_attn":
            self.slice_pool = MaskedAttentionPool(self.feature_dim, dropout=0.05)
            self.view_pool = MaskedAttentionPool(self.feature_dim, dropout=0.05)
            self.classifier = self._make_classifier_head(self.feature_dim, num_classes, dropout)
        elif self.agg == "hier_mean":
            self.slice_pool = MaskedMeanPool()
            self.view_pool = MaskedMeanPool()
            self.classifier = self._make_classifier_head(self.feature_dim, num_classes, dropout)
        elif self.agg == "flat_mil":
            self.flat_pool = MaskedAttentionPool(self.feature_dim, dropout=0.05)
            self.classifier = self._make_classifier_head(self.feature_dim, num_classes, dropout)
        elif self.agg == "flat_mean":
            self.flat_pool = MaskedMeanPool()
            self.classifier = self._make_classifier_head(self.feature_dim, num_classes, dropout)
        elif self.agg == "concat_mlp":
            self.slice_pool = MaskedAttentionPool(self.feature_dim, dropout=0.05)
            self.classifier = self._make_classifier_head(self.feature_dim * self.num_views, num_classes, dropout)
        elif self.agg == "late_fusion":
            self.slice_pool = MaskedAttentionPool(self.feature_dim, dropout=0.05)
            self.view_classifier = self._make_classifier_head(self.feature_dim, num_classes, dropout)
        elif self.agg == "view_transformer":
            self.slice_pool = MaskedAttentionPool(self.feature_dim, dropout=0.05)
            self.view_transformer = ViewTokenTransformerPool(self.feature_dim, self.num_views, dropout=dropout)
            self.classifier = self._make_classifier_head(self.feature_dim, num_classes, dropout)
        else:
            raise ValueError(f"Unsupported agg: {self.agg}")

    def _compute_view_features(self, feats: torch.Tensor, clip_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        view_feats, view_valids, slice_weights = [], [], []
        for local_vid in range(self.num_views):
            start, end = self.view_clip_ranges[local_vid]
            sf = feats[:, start:end, :]
            sm = clip_mask[:, start:end]
            vf, sw = self.slice_pool(sf, sm)
            view_feats.append(vf)
            view_valids.append(sm.any(dim=1))
            slice_weights.append(sw)
        return torch.stack(view_feats, dim=1), torch.stack(view_valids, dim=1), slice_weights

    def _flat_view_weights(self, flat_weights: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        view_weights, slice_weights = [], []
        for local_vid in range(self.num_views):
            start, end = self.view_clip_ranges[local_vid]
            local_w = flat_weights[:, start:end]
            slice_weights.append(local_w)
            view_weights.append(local_w.sum(dim=1))
        view_weights = torch.stack(view_weights, dim=1)
        denom = view_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        view_weights = view_weights / denom
        return view_weights, slice_weights

    def _masked_view_mean(self, view_feats: torch.Tensor, view_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = view_mask.float()
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1.0)
        patient_feat = torch.sum(view_feats * w.unsqueeze(-1), dim=1)
        return patient_feat, w

    def _aggregate_features(self, feats: torch.Tensor, clip_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Aggregate clip features into patient feature according to self.agg."""
        if self.agg in {"flat_mil", "flat_mean"}:
            patient_feat, flat_weights = self.flat_pool(feats, clip_mask)
            view_weights, slice_weights = self._flat_view_weights(flat_weights)
            aux = {"view_weights": view_weights, "slice_weights": slice_weights}
        else:
            view_feats, view_mask, slice_weights = self._compute_view_features(feats, clip_mask)
            if self.agg in {"hier_attn", "hier_mean"}:
                patient_feat, view_weights = self.view_pool(view_feats, view_mask)
            elif self.agg == "concat_mlp":
                # Keep a fixed view order; missing views are zeroed by mask before concatenation.
                view_feats_masked = view_feats * view_mask.float().unsqueeze(-1)
                patient_feat = view_feats_masked.reshape(view_feats.shape[0], self.num_views * self.feature_dim)
                view_weights = view_mask.float()
                view_weights = view_weights / view_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            elif self.agg == "late_fusion":
                # Patient feature is only returned for feature extraction/prognosis hooks; logits are computed per view.
                patient_feat, view_weights = self._masked_view_mean(view_feats, view_mask)
            elif self.agg == "view_transformer":
                patient_feat, view_weights = self.view_transformer(view_feats, view_mask)
            else:
                raise ValueError(f"Unsupported agg: {self.agg}")
            aux = {"view_feats": view_feats, "view_mask": view_mask, "view_weights": view_weights, "slice_weights": slice_weights}

        self.last_attention = {
            "view_weights": aux["view_weights"].detach().cpu(),
            "slice_weights": [w.detach().cpu() for w in aux["slice_weights"]],
            "selected_views": list(self.selected_views),
            "agg": self.agg,
        }
        return patient_feat, aux

    def _classify_from_aggregation(self, patient_feat: torch.Tensor, aux: Dict) -> torch.Tensor:
        if self.agg == "late_fusion":
            view_feats = aux["view_feats"]
            view_mask = aux["view_mask"]
            B, V, D = view_feats.shape
            logits_per_view = self.view_classifier(view_feats.reshape(B * V, D)).reshape(B, V, -1)
            w = view_mask.float()
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1.0)
            self.last_attention["view_weights"] = w.detach().cpu()
            return torch.sum(logits_per_view * w.unsqueeze(-1), dim=1)
        return self.classifier(patient_feat)


class MultiViewVideoSwinClassifier(MultiViewAggregationMixin, nn.Module):
    def __init__(
        self,
        selected_views: Sequence[str],
        slice_plan: Dict[str, int],
        num_classes: int = 3,
        weights_path: Optional[str] = None,
        pretrained: bool = False,
        agg: str = "hier_attn",
        dropout: float = 0.35,
        freeze_backbone: bool = False,
        backbone_chunk_size: int = 0,
    ):
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
            self.load_backbone_weights(weights_path)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.view_embeddings = nn.Parameter(torch.zeros(self.num_views, self.feature_dim))
        nn.init.trunc_normal_(self.view_embeddings, std=0.02)

        self.view_clip_ranges = self._build_view_clip_ranges()
        self._init_multiview_aggregation(num_classes=num_classes, agg=agg, dropout=dropout)
        print0(
            f"✅ MultiView Video Swin 初始化: views={self.selected_views}, nclips={self.nclips}, "
            f"dim={self.feature_dim}, agg={self.agg}"
        )

    def _build_view_clip_ranges(self) -> Dict[int, Tuple[int, int]]:
        ranges = {}
        start = 0
        for local_vid, v in enumerate(self.selected_views):
            k = int(self.slice_plan.get(v, 0))
            ranges[local_vid] = (start, start + k)
            start += k
        return ranges

    def load_backbone_weights(self, weights_path: Optional[str]):
        if not weights_path:
            print0("⚠️ pretrained=True 但未提供 weights_path，将从随机初始化训练")
            return
        p = Path(weights_path)
        if not p.exists():
            print0(f"⚠️ 找不到权重文件: {p}，将从随机初始化训练")
            return
        print0(f"🚀 加载本地 Video Swin 权重: {p}")
        ckpt = torch.load(str(p), map_location="cpu")
        if isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            else:
                state = ckpt
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
        print0(f"🎉 权重加载完成: missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print0(f"  missing examples: {missing[:5]}")
        if len(unexpected) > 0:
            print0(f"  unexpected examples: {unexpected[:5]}")

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M,3,T,H,W]
        if self.backbone_chunk_size and self.backbone_chunk_size > 0 and x.shape[0] > self.backbone_chunk_size:
            outs = []
            for s in range(0, x.shape[0], self.backbone_chunk_size):
                outs.append(self.backbone(x[s:s + self.backbone_chunk_size]))
            return torch.cat(outs, dim=0)
        return self.backbone(x)

    def _extract_clip_features(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        videos = batch["videos"]          # [B,N,1,T,H,W]
        clip_mask = batch["clip_mask"]    # [B,N]
        clip_view_ids = batch["clip_view_ids"]  # [B,N]

        B, N, C, T, H, W = videos.shape
        assert N == self.nclips, f"input Nclips={N}, model nclips={self.nclips}"

        flat_mask = clip_mask.reshape(-1).bool()
        flat_view_ids = clip_view_ids.reshape(-1).long()
        flat_videos = videos.reshape(B * N, C, T, H, W)

        feats_flat = torch.zeros(B * N, self.feature_dim, device=videos.device, dtype=torch.float32)
        if flat_mask.any():
            x_valid = flat_videos[flat_mask]
            x_valid = x_valid.expand(-1, 3, -1, -1, -1).contiguous()
            feat_valid = self._run_backbone(x_valid).float()
            valid_view_ids = flat_view_ids[flat_mask]
            feat_valid = feat_valid + self.view_embeddings[valid_view_ids]
            feats_flat[flat_mask] = feat_valid
        return feats_flat.reshape(B, N, self.feature_dim), clip_mask

    def forward_features(self, batch: Dict) -> torch.Tensor:
        feats, clip_mask = self._extract_clip_features(batch)
        patient_feat, _aux = self._aggregate_features(feats, clip_mask)
        return patient_feat

    def forward(self, batch: Dict, return_features: bool = False):
        feats, clip_mask = self._extract_clip_features(batch)
        patient_feat, aux = self._aggregate_features(feats, clip_mask)
        logits = self._classify_from_aggregation(patient_feat, aux)
        if return_features:
            return logits, patient_feat
        return logits



class MultiView2DCNNClassifier(MultiViewAggregationMixin, nn.Module):
    """2D CNN 对比模型：使用同一套 view-slice-frame 输入与聚合逻辑。"""
    def __init__(self, selected_views: Sequence[str], slice_plan: Dict[str, int], num_classes: int = 3,
                 backbone_name: str = "resnet50", weights_path: Optional[str] = None,
                 pretrained: bool = False, agg: str = "hier_attn", dropout: float = 0.35,
                 freeze_backbone: bool = False, backbone_chunk_size: int = 0, frame_samples: int = 0):
        super().__init__()
        self.selected_views = list(selected_views)
        self.slice_plan = dict(slice_plan)
        self.num_views = len(self.selected_views)
        self.nclips = sum(int(self.slice_plan.get(v, 0)) for v in self.selected_views)
        self.backbone_name = backbone_name.lower()
        self.backbone_chunk_size = int(backbone_chunk_size)
        # frame_samples <= 0 表示使用输入 clip 的全部 T 帧；这样 2D 对比模型也能和主模型一样看 num_frames 帧。
        self.frame_samples = int(frame_samples)
        self.backbone, self.feature_dim = self._create_2d_backbone(self.backbone_name)
        if pretrained and weights_path:
            self.load_backbone_weights(weights_path)
        elif pretrained and not weights_path:
            print0(f"⚠️ {self.backbone_name} pretrained=True 但 model_weights_path 为空；内网环境不联网下载，当前为随机初始化。")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.view_embeddings = nn.Parameter(torch.zeros(self.num_views, self.feature_dim))
        nn.init.trunc_normal_(self.view_embeddings, std=0.02)
        self.view_clip_ranges = self._build_view_clip_ranges()
        self._init_multiview_aggregation(num_classes=num_classes, agg=agg, dropout=dropout)
        print0(
            f"✅ MultiView 2D CNN 初始化: backbone={self.backbone_name}, views={self.selected_views}, "
            f"nclips={self.nclips}, dim={self.feature_dim}, frame_samples={self.frame_samples}, agg={self.agg}"
        )

    def _build_view_clip_ranges(self) -> Dict[int, Tuple[int, int]]:
        ranges, start = {}, 0
        for local_vid, v in enumerate(self.selected_views):
            k = int(self.slice_plan.get(v, 0)); ranges[local_vid] = (start, start + k); start += k
        return ranges

    def _create_2d_backbone(self, name: str):
        if name in {"resnet18", "resnet50"}:
            model = getattr(tv_models, name)(weights=None)
            feature_dim = model.fc.in_features
            model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
            model.fc = nn.Identity()
            return model, feature_dim
        if name == "densenet121":
            model = tv_models.densenet121(weights=None)
            old = model.features.conv0
            model.features.conv0 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
            model.classifier = nn.Identity()
            return model, 1024
        if name in {"efficientnet_b0", "convnext_tiny"}:
            if not HAS_TIMM:
                raise RuntimeError(f"{name} 需要 timm，但当前环境未安装 timm。")
            model = timm.create_model(name, pretrained=False, num_classes=0, in_chans=1)
            return model, model.num_features
        raise ValueError(f"Unsupported 2D backbone: {name}")

    def load_backbone_weights(self, weights_path: str):
        p = Path(weights_path)
        if not p.exists():
            print0(f"⚠️ 找不到 2D backbone 权重文件: {p}，将从随机初始化训练"); return
        print0(f"🚀 加载本地 2D backbone 权重: {p}")
        ckpt = torch.load(str(p), map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        clean = {}
        for k, v in state.items():
            kk = k
            for prefix in ["module.", "backbone."]:
                if kk.startswith(prefix): kk = kk[len(prefix):]
            if kk.startswith(("fc.", "classifier.", "head.")) or ".head." in kk: continue
            if hasattr(v, "ndim") and v.ndim == 4 and v.shape[1] == 3 and ("conv1.weight" in kk or "conv0.weight" in kk or "patch_embed.proj.weight" in kk):
                v = v.mean(dim=1, keepdim=True)
            clean[kk] = v
        msg = self.backbone.load_state_dict(clean, strict=False)
        print0(f"🎉 2D 权重加载完成: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_chunk_size and self.backbone_chunk_size > 0 and x.shape[0] > self.backbone_chunk_size:
            outs = []
            for s in range(0, x.shape[0], self.backbone_chunk_size): outs.append(self.backbone(x[s:s + self.backbone_chunk_size]))
            return torch.cat(outs, dim=0)
        return self.backbone(x)

    def _extract_clip_features(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        videos, clip_mask, clip_view_ids = batch["videos"], batch["clip_mask"], batch["clip_view_ids"]
        B, N, C, T, H, W = videos.shape
        flat_mask = clip_mask.reshape(-1).bool(); flat_view_ids = clip_view_ids.reshape(-1).long(); flat_videos = videos.reshape(B * N, C, T, H, W)
        feats_flat = torch.zeros(B * N, self.feature_dim, device=videos.device, dtype=torch.float32)
        if flat_mask.any():
            x_valid = flat_videos[flat_mask]
            use_frames = T if self.frame_samples <= 0 else min(int(self.frame_samples), T)
            f_idx = torch.linspace(0, T - 1, use_frames, device=videos.device).long()
            x_frames = x_valid.index_select(dim=2, index=f_idx)
            M = x_frames.shape[0]
            x_frames = x_frames.permute(0, 2, 1, 3, 4).reshape(M * use_frames, 1, H, W).contiguous()
            frame_feats = self._run_backbone(x_frames).float().reshape(M, use_frames, self.feature_dim)
            feat_valid = frame_feats.mean(dim=1)
            feat_valid = feat_valid + self.view_embeddings[flat_view_ids[flat_mask]]
            feats_flat[flat_mask] = feat_valid
        return feats_flat.reshape(B, N, self.feature_dim), clip_mask

    def forward_features(self, batch: Dict) -> torch.Tensor:
        feats, clip_mask = self._extract_clip_features(batch)
        patient_feat, _aux = self._aggregate_features(feats, clip_mask)
        return patient_feat

    def forward(self, batch: Dict, return_features: bool = False):
        feats, clip_mask = self._extract_clip_features(batch)
        patient_feat, aux = self._aggregate_features(feats, clip_mask)
        logits = self._classify_from_aggregation(patient_feat, aux)
        return (logits, patient_feat) if return_features else logits


def build_model(args, selected_views: Sequence[str], slice_plan: Dict[str, int]) -> nn.Module:
    model_type = str(args.model_type).lower().strip(); model_name = str(getattr(args, "model_name", "") or "").lower().strip()
    if model_type in {"videoswin", "swin3d", "video_swin"}:
        return MultiViewVideoSwinClassifier(selected_views=selected_views, slice_plan=slice_plan, num_classes=3,
            weights_path=args.weights_path if args.weights_path else None, pretrained=args.pretrained, agg=args.agg,
            dropout=args.dropout, freeze_backbone=args.freeze_backbone, backbone_chunk_size=args.backbone_chunk_size)
    backbone_name = model_name or model_type
    if model_type in {"cnn2d", "resnet", "efficientnet", "densenet", "convnext"}:
        defaults = {"resnet": "resnet50", "efficientnet": "efficientnet_b0", "densenet": "densenet121", "convnext": "convnext_tiny", "cnn2d": "resnet50"}
        backbone_name = model_name or defaults[model_type]
    if backbone_name in {"resnet18", "resnet50", "densenet121", "efficientnet_b0", "convnext_tiny"}:
        return MultiView2DCNNClassifier(selected_views=selected_views, slice_plan=slice_plan, num_classes=3,
            backbone_name=backbone_name, weights_path=getattr(args, "model_weights_path", "") or None, pretrained=args.pretrained,
            agg=args.agg, dropout=args.dropout, freeze_backbone=args.freeze_backbone, backbone_chunk_size=args.backbone_chunk_size,
            frame_samples=args.frame_samples)
    raise ValueError(f"Unknown model_type/model_name: {model_type}/{model_name}")


def attach_prognosis_info(samples: List[Dict], args) -> None:
    for s in samples:
        s.setdefault("prog_available", 0); s.setdefault("event", -1); s.setdefault("time_to_event", -1); s.setdefault("censored", -1)
    csv_path = str(getattr(args, "prognosis_csv", "") or "").strip()
    if not csv_path: return
    p = Path(csv_path)
    if not p.exists(): print0(f"⚠️ prognosis_csv 不存在: {p}，将只做三分类。"); return
    df = pd.read_csv(p)
    id_col, event_col, time_col, censor_col = args.prognosis_id_col, args.prognosis_event_col, args.prognosis_time_col, args.prognosis_censor_col
    if id_col not in df.columns: print0(f"⚠️ prognosis_csv 中找不到 ID 列 {id_col}，将只做三分类。"); return
    table = {str(r[id_col]): r for _, r in df.iterrows()}; matched = 0
    for s in samples:
        pid = str(s["patient_id"])
        if pid not in table: continue
        r = table[pid]
        try:
            s["prog_available"] = 1
            s["event"] = float(r[event_col]) if event_col in df.columns and pd.notna(r[event_col]) else -1
            s["time_to_event"] = float(r[time_col]) if time_col in df.columns and pd.notna(r[time_col]) else -1
            s["censored"] = float(r[censor_col]) if censor_col in df.columns and pd.notna(r[censor_col]) else -1
            matched += 1
        except Exception: pass
    print0(f"✅ 已接入预后/生存信息: matched={matched}/{len(samples)}；当前训练仍默认只计算三分类 loss。")


class CoxPHLoss(nn.Module):
    """预后阶段备用 Cox partial likelihood loss。当前脚本默认不启用。"""
    def forward(self, risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        risk = risk.reshape(-1)[mask.bool()]; time = time.reshape(-1)[mask.bool()]; event = event.reshape(-1)[mask.bool()]
        if risk.numel() == 0 or event.sum() <= 0: return risk.sum() * 0.0
        order = torch.argsort(time, descending=True); risk = risk[order]; event = event[order]
        log_cumsum = torch.logcumsumexp(risk, dim=0)
        return -((risk - log_cumsum) * event).sum() / event.sum().clamp(min=1.0)

# -----------------------------
# Train / Eval
# -----------------------------

def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def get_lr(optimizer):
    for g in optimizer.param_groups:
        return g["lr"]
    return 0.0


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Return parameter counts for reporting model complexity."""
    target = model.module if isinstance(model, DDP) else model
    total = sum(p.numel() for p in target.parameters())
    trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "params_total_m": float(total / 1e6),
        "params_trainable_m": float(trainable / 1e6),
    }


def get_peak_cuda_memory_mb(device: torch.device) -> float:
    """Peak allocated CUDA memory in MB, reduced by max across ranks."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    value = torch.tensor([torch.cuda.max_memory_allocated(device) / (1024 ** 2)], dtype=torch.float64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def reset_peak_cuda_memory(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    args,
) -> Dict:
    model.train()
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]

        with torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
            logits = model(batch)
            loss = criterion(logits, labels)
            loss = loss / args.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(loader):
            if args.grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            loss_value = loss.item() * args.grad_accum_steps
            pred = torch.argmax(logits, dim=1)
            total_correct += (pred == labels).sum().item()
            total_count += labels.numel()
            total_loss += loss_value * labels.numel()

        if is_main_process() and args.print_freq > 0 and (step + 1) % args.print_freq == 0:
            elapsed = time.time() - start_time
            ips = total_count / max(elapsed, 1e-6)
            # print(f"  Epoch {epoch:03d} Step {step+1:04d}/{len(loader):04d} | "
            #       f"loss {total_loss/max(total_count,1):.4f} | "
            #       f"acc {100*total_correct/max(total_count,1):.2f}% | "
            #       f"lr {get_lr(optimizer):.2e} | {ips:.2f} samples/s")

    # Reduce across ranks.
    stats = torch.tensor([total_loss, total_correct, total_count], dtype=torch.float64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_avg = (stats[0] / stats[2].clamp(min=1)).item()
    acc = (stats[1] / stats[2].clamp(min=1)).item()
    return {"loss": loss_avg, "accuracy": acc}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args, phase: str = "val") -> Dict:
    """Evaluate and record both end-to-end time and model-forward time.

    eval_time_sec includes DataLoader + CPU/GPU transfer + forward.
    forward_time_sec is synchronized model forward time only.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    local_rows = []
    local_forward_time = 0.0
    local_forward_count = 0
    eval_start = time.perf_counter()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fwd_start = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
            logits = model(batch)
            loss = criterion(logits, labels)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        local_forward_time += time.perf_counter() - fwd_start
        local_forward_count += int(labels.numel())

        probs = torch.softmax(logits.float(), dim=1)
        preds = torch.argmax(probs, dim=1)
        total_loss += loss.item() * labels.numel()

        labels_np = labels.detach().cpu().numpy().tolist()
        preds_np = preds.detach().cpu().numpy().tolist()
        probs_np = probs.detach().cpu().numpy().tolist()
        pids = batch["patient_id"]
        for pid, y, p, prob in zip(pids, labels_np, preds_np, probs_np):
            local_rows.append({
                "patient_id": pid,
                "label": int(y),
                "prediction": int(p),
                "prob_NC": float(prob[0]),
                "prob_HCM": float(prob[1]),
                "prob_DCM": float(prob[2]),
                "loss_weight": 1.0,
            })

    eval_elapsed_local = time.perf_counter() - eval_start

    if is_dist_avail_and_initialized():
        gathered = [None for _ in range(get_world_size())]
        dist.all_gather_object(gathered, local_rows)
        rows = []
        for g in gathered:
            rows.extend(g)

        timing_tensor = torch.tensor(
            [eval_elapsed_local, local_forward_time, float(local_forward_count)],
            dtype=torch.float64,
            device=device,
        )
        elapsed_tensor = timing_tensor[0:1].clone()
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        forward_count_tensor = timing_tensor[1:3].clone()
        dist.all_reduce(forward_count_tensor, op=dist.ReduceOp.SUM)
        eval_time_sec = float(elapsed_tensor.item())
        forward_time_sec = float(forward_count_tensor[0].item())
        forward_count = int(forward_count_tensor[1].item())
    else:
        rows = local_rows
        eval_time_sec = float(eval_elapsed_local)
        forward_time_sec = float(local_forward_time)
        forward_count = int(local_forward_count)

    # DDP DistributedSampler may pad duplicate samples. Deduplicate by patient_id.
    dedup = {}
    for r in rows:
        dedup[r["patient_id"]] = r
    rows = list(dedup.values())

    if len(rows) == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "rows": [],
            "labels": np.array([]),
            "preds": np.array([]),
            "probs": np.array([]),
            "eval_time_sec": eval_time_sec,
            "forward_time_sec": forward_time_sec,
            "num_samples": 0,
            "throughput_samples_per_sec": 0.0,
            "forward_latency_ms_per_patient": 0.0,
            "end_to_end_latency_ms_per_patient": 0.0,
        }

    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    preds = np.array([r["prediction"] for r in rows], dtype=np.int64)
    probs = np.array([[r["prob_NC"], r["prob_HCM"], r["prob_DCM"]] for r in rows], dtype=np.float32)
    acc = accuracy_score(labels, preds)
    avg_loss = total_loss / max(len(local_rows), 1)
    num_unique = len(rows)
    throughput = num_unique / max(eval_time_sec, 1e-9)
    forward_latency = 1000.0 * forward_time_sec / max(forward_count, 1)
    end_to_end_latency = 1000.0 * eval_time_sec / max(num_unique, 1)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "rows": rows,
        "labels": labels,
        "preds": preds,
        "probs": probs,
        "eval_time_sec": float(eval_time_sec),
        "forward_time_sec": float(forward_time_sec),
        "num_samples": int(num_unique),
        "throughput_samples_per_sec": float(throughput),
        "forward_latency_ms_per_patient": float(forward_latency),
        "end_to_end_latency_ms_per_patient": float(end_to_end_latency),
    }

def summarize_metrics(results: Dict) -> Dict:
    y = results["labels"]
    p = results["preds"]
    prob = results["probs"]
    metrics = {}
    if len(y) == 0:
        return metrics
    metrics["accuracy"] = float(accuracy_score(y, p))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y, p))
    metrics["macro_precision"] = float(precision_score(y, p, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(y, p, average="macro", zero_division=0))
    metrics["macro_f1"] = float(f1_score(y, p, average="macro", zero_division=0))
    try:
        # Only valid if every class appears in y.
        metrics["ovr_auc_macro"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except Exception:
        metrics["ovr_auc_macro"] = None
    return metrics


def save_eval_outputs(results: Dict, out_prefix: Path):
    ensure_dir(out_prefix.parent)
    rows = results["rows"]
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["true_class"] = df["label"].map(IDX_TO_CLASS)
        df["pred_class"] = df["prediction"].map(IDX_TO_CLASS)
        df["correct"] = df["label"] == df["prediction"]
    df.to_csv(str(out_prefix) + "_predictions.csv", index=False, encoding="utf-8-sig")

    y, p = results["labels"], results["preds"]
    report = classification_report(y, p, labels=[0, 1, 2], target_names=["NC", "HCM", "DCM"], digits=4, zero_division=0)
    with open(str(out_prefix) + "_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    pd.DataFrame(cm, index=["true_NC", "true_HCM", "true_DCM"], columns=["pred_NC", "pred_HCM", "pred_DCM"]).to_csv(
        str(out_prefix) + "_confusion_matrix.csv", encoding="utf-8-sig"
    )

    metrics = summarize_metrics(results)
    for k in [
        "eval_time_sec",
        "forward_time_sec",
        "num_samples",
        "throughput_samples_per_sec",
        "forward_latency_ms_per_patient",
        "end_to_end_latency_ms_per_patient",
        "params_total",
        "params_trainable",
        "params_total_m",
        "params_trainable_m",
        "peak_cuda_memory_mb",
        "total_train_time_sec",
        "avg_epoch_time_sec",
        "best_epoch",
        "best_val_macro_f1",
    ]:
        if k in results:
            metrics[k] = results[k]
    with open(str(out_prefix) + "_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if HAS_MPL:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["NC", "HCM", "DCM"])
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["NC", "HCM", "DCM"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(str(out_prefix) + "_confusion_matrix.png", dpi=300)
        plt.close(fig)


    if HAS_MPL:
        try:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y, classes=[0, 1, 2])
            fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111)
            for i, name in enumerate(["NC", "HCM", "DCM"]):
                fpr, tpr, _ = roc_curve(y_bin[:, i], results["probs"][:, i]); roc_i = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{name} AUC={roc_i:.3f}")
            fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), results["probs"].ravel())
            ax.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro AUC={auc(fpr_micro, tpr_micro):.3f}")
            ax.plot([0, 1], [0, 1], linestyle=":"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title("One-vs-rest ROC"); ax.legend(loc="lower right"); fig.tight_layout()
            fig.savefig(str(out_prefix) + "_roc_curves.png", dpi=300); plt.close(fig)
        except Exception as e:
            print0(f"ROC 绘制失败: {e}")

    return metrics, report



def save_history_plot(history: List[Dict], out_dir: Path):
    if not HAS_MPL or len(history) == 0: return
    df = pd.DataFrame(history); fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121); ax1.plot(df["epoch"], df["train_loss"], label="train_loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
    ax2 = fig.add_subplot(122); ax2.plot(df["epoch"], df["train_acc"], label="train_acc"); ax2.plot(df["epoch"], df["val_acc"], label="val_acc"); ax2.plot(df["epoch"], df["val_macro_f1"], label="val_macro_f1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score"); ax2.legend(); fig.tight_layout(); fig.savefig(out_dir / "history_plot.png", dpi=300); plt.close(fig)



# =============================
# Explanation-panel visualization
# =============================

def _safe_denormalize_frame(x: torch.Tensor) -> np.ndarray:
    """Convert one normalized frame tensor [H,W] or [1,H,W] to [0,1] numpy."""
    if x.dim() == 3:
        x = x.squeeze(0)
    x = x.detach().float().cpu()
    x = x * 0.229 + 0.485
    x = torch.clamp(x, 0.0, 1.0)
    return x.numpy()


def _compute_motion_curve(clip: torch.Tensor) -> np.ndarray:
    """clip: [1,T,H,W], returns length T-1 frame-to-frame pixel-difference curve."""
    c = clip.detach().float().cpu()
    if c.dim() == 4:
        c = c.squeeze(0)
    c = torch.clamp(c * 0.229 + 0.485, 0.0, 1.0)
    if c.shape[0] < 2:
        return np.array([], dtype=np.float32)
    diff = torch.mean(torch.abs(c[1:] - c[:-1]), dim=(1, 2))
    return diff.numpy()


def _choose_representative_clip(batch: Dict, target_model: nn.Module, selected_views: Sequence[str]) -> Dict:
    """Choose best view and best slice/clip according to attention, restricted to selected_views only."""
    att = getattr(target_model, "last_attention", None) or {}
    view_mask = batch["view_mask"][0].detach().bool().cpu().numpy().astype(bool)
    clip_mask = batch["clip_mask"][0].detach().bool().cpu().numpy().astype(bool)

    if "view_weights" in att:
        vw = att["view_weights"][0]
        vw = vw.detach().cpu().numpy() if torch.is_tensor(vw) else np.asarray(vw)
        vw = vw[:len(selected_views)].astype(np.float32)
        vw = np.where(view_mask[:len(selected_views)], vw, 0.0)
        if vw.sum() > 0:
            vw = vw / (vw.sum() + 1e-8)
        view_weights = vw
    else:
        vw = view_mask[:len(selected_views)].astype(np.float32)
        view_weights = vw / (vw.sum() + 1e-8) if vw.sum() > 0 else vw

    valid_view_indices = np.where(view_mask[:len(selected_views)])[0]
    best_view_idx = 0 if len(valid_view_indices) == 0 else int(valid_view_indices[np.argmax(view_weights[valid_view_indices])])

    start, end = target_model.view_clip_ranges.get(best_view_idx, (0, 0))
    local_clip_mask = clip_mask[start:end]
    local_slice_weights = None
    if "slice_weights" in att and best_view_idx < len(att["slice_weights"]):
        sw = att["slice_weights"][best_view_idx][0]
        sw = sw.detach().cpu().numpy() if torch.is_tensor(sw) else np.asarray(sw)
        sw = sw[:max(0, end - start)].astype(np.float32)
        sw = np.where(local_clip_mask, sw, 0.0)
        if sw.sum() > 0:
            sw = sw / (sw.sum() + 1e-8)
        local_slice_weights = sw

    valid_local = np.where(local_clip_mask)[0]
    if len(valid_local) == 0:
        valid_global = np.where(clip_mask)[0]
        chosen_clip_idx = int(valid_global[0]) if len(valid_global) > 0 else 0
        best_slice_local = 0
    else:
        if local_slice_weights is not None and local_slice_weights.sum() > 0:
            best_slice_local = int(valid_local[np.argmax(local_slice_weights[valid_local])])
        else:
            best_slice_local = int(valid_local[len(valid_local) // 2])
        chosen_clip_idx = int(start + best_slice_local)

    return {
        "view_weights": view_weights,
        "best_view_idx": best_view_idx,
        "best_view_name": selected_views[best_view_idx] if best_view_idx < len(selected_views) else "Unknown",
        "best_slice_local": int(best_slice_local),
        "clip_idx": int(chosen_clip_idx),
        "slice_weights": local_slice_weights,
    }


def _get_gradcam_target_layer(target_model: nn.Module):
    if isinstance(target_model, MultiViewVideoSwinClassifier):
        return target_model.backbone.features[-1], "videoswin"
    if isinstance(target_model, MultiView2DCNNClassifier):
        name = target_model.backbone_name
        b = target_model.backbone
        if name in {"resnet18", "resnet50"} and hasattr(b, "layer4"):
            return b.layer4[-1], "cnn2d"
        if name == "densenet121" and hasattr(b, "features"):
            return b.features, "cnn2d"
        if name == "efficientnet_b0" and hasattr(b, "blocks"):
            return b.blocks[-1], "cnn2d"
        if name == "convnext_tiny":
            if hasattr(b, "stages"):
                return b.stages[-1], "cnn2d"
            if hasattr(b, "blocks"):
                return b.blocks[-1], "cnn2d"
    return None, "unknown"


def _normalize_cam(cam: torch.Tensor, image_size: int) -> np.ndarray:
    cam = cam.detach().float().cpu()
    if cam.dim() == 2:
        cam = cam.unsqueeze(0).unsqueeze(0)
    elif cam.dim() == 3:
        cam = cam.unsqueeze(0)
    cam = F.interpolate(cam, size=(image_size, image_size), mode="bilinear", align_corners=False)
    cam = cam.squeeze()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.numpy()


def _gradcam_for_selected_clip(target_model: nn.Module, batch: Dict, clip_idx: int, target_class: Optional[int] = None) -> Optional[np.ndarray]:
    layer, family = _get_gradcam_target_layer(target_model)
    if layer is None:
        return None

    activations, gradients = [], []

    def fwd_hook(_module, _inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        activations.append(out)

    def bwd_hook(_module, _grad_in, grad_out):
        gout = grad_out[0]
        if isinstance(gout, (tuple, list)):
            gout = gout[0]
        gradients.append(gout)

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    was_training = target_model.training
    target_model.eval()
    try:
        target_model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = target_model(batch)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1)[0].item())
            score = logits[0, int(target_class)]
            score.backward(retain_graph=False)

        if not activations or not gradients:
            return None
        act, grad = activations[-1], gradients[-1]
        flat_mask = batch["clip_mask"].reshape(-1).detach().bool().cpu()
        valid_positions = torch.where(flat_mask)[0].tolist()
        if not valid_positions:
            return None
        valid_order = valid_positions.index(int(clip_idx)) if int(clip_idx) in valid_positions else 0
        image_size = int(batch["videos"].shape[-1])

        if family == "videoswin":
            if act.dim() != 5 or grad.dim() != 5:
                return None
            if act.shape[-1] == target_model.feature_dim:
                act_cf = act.permute(0, 4, 1, 2, 3)
                grad_cf = grad.permute(0, 4, 1, 2, 3)
            elif act.shape[1] == target_model.feature_dim:
                act_cf, grad_cf = act, grad
            else:
                act_cf = act.permute(0, 4, 1, 2, 3)
                grad_cf = grad.permute(0, 4, 1, 2, 3)
            idx = min(valid_order, act_cf.shape[0] - 1)
            a, g = act_cf[idx:idx + 1], grad_cf[idx:idx + 1]
            weights = g.mean(dim=(2, 3, 4), keepdim=True)
            cam3d = torch.relu((weights * a).sum(dim=1))
            cam2d = cam3d.mean(dim=1)
            return _normalize_cam(cam2d, image_size)

        if family == "cnn2d":
            if act.dim() != 4 or grad.dim() != 4:
                return None
            T = int(batch["videos"].shape[3])
            use_frames = T if getattr(target_model, "frame_samples", 0) <= 0 else min(int(target_model.frame_samples), T)
            idx = min(valid_order * use_frames + use_frames // 2, act.shape[0] - 1)
            a, g = act[idx:idx + 1], grad[idx:idx + 1]
            weights = g.mean(dim=(2, 3), keepdim=True)
            cam2d = torch.relu((weights * a).sum(dim=1))
            return _normalize_cam(cam2d, image_size)
        return None
    except Exception:
        return None
    finally:
        h1.remove(); h2.remove(); target_model.zero_grad(set_to_none=True)
        if was_training:
            target_model.train()


def _make_explanation_panel(target_model: nn.Module, batch: Dict, selected_views: Sequence[str], out_path: Path, row: Dict) -> Dict:
    if not HAS_MPL:
        return row
    with torch.no_grad():
        logits = target_model(batch)
        prob = torch.softmax(logits.float(), dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(prob))
    true_label = int(batch["label"][0].detach().cpu().item())
    pid = str(batch["patient_id"][0])

    choice = _choose_representative_clip(batch, target_model, selected_views)
    view_weights = choice["view_weights"]
    best_view_name = choice["best_view_name"]
    best_view_idx = choice["best_view_idx"]
    clip_idx = choice["clip_idx"]
    best_slice_local = choice["best_slice_local"]

    clip = batch["videos"][0, clip_idx].detach().cpu()
    T = clip.shape[1]
    mid_frame = T // 2
    original_img = _safe_denormalize_frame(clip[:, mid_frame])
    motion = _compute_motion_curve(clip)
    cam = _gradcam_for_selected_clip(target_model, batch, clip_idx=clip_idx, target_class=pred)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Patient Explanation Panel: {pid}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    xlabels = list(selected_views)
    if view_weights is None or len(view_weights) == 0:
        ax.text(0.5, 0.5, "No View Attention", ha="center", va="center")
        view_weights = np.zeros(len(xlabels), dtype=np.float32)
    ax.bar(xlabels, view_weights[:len(xlabels)])
    ax.set_ylim(0, 1.0); ax.set_ylabel("Attention weight"); ax.set_title("View Importance")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    if motion.size > 0:
        x = np.arange(1, len(motion) + 1)
        ax.plot(x, motion, marker="o"); ax.fill_between(x, motion, alpha=0.2)
        ax.set_xlabel("Frame transition"); ax.set_ylabel("Mean |I(t)-I(t-1)|")
    else:
        ax.text(0.5, 0.5, "Not enough frames", ha="center", va="center")
    ax.set_title(f"Cardiac Motion Curve ({best_view_name})"); ax.grid(alpha=0.25)

    ax = axes[0, 2]
    ax.imshow(original_img, cmap="gray")
    ax.set_title(f"Original Image\n{best_view_name} | slice slot {best_slice_local} | frame {mid_frame}/{T - 1}")
    ax.axis("off")

    ax = axes[1, 0]
    if cam is not None:
        ax.imshow(original_img, cmap="gray"); ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    else:
        ax.imshow(original_img, cmap="gray")
        ax.text(0.5, 0.5, "Grad-CAM Failed", ha="center", va="center", transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("Grad-CAM Heatmap"); ax.axis("off")

    ax = axes[1, 1]
    if cam is not None:
        im = ax.imshow(cam, cmap="jet", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No CAM mask", ha="center", va="center")
    ax.set_title("Attention Mask"); ax.axis("off")

    ax = axes[1, 2]; ax.axis("off")
    conf = float(prob[pred])
    available_views = batch.get("available_views", [[""]])[0]
    av_text = ", ".join(map(str, available_views)) if isinstance(available_views, (list, tuple)) else str(available_views)
    status = "CORRECT" if pred == true_label else "WRONG"
    summary = (
        f"Patient ID: {pid}\n"
        f"True: {IDX_TO_CLASS.get(true_label, true_label)}\n"
        f"Pred: {IDX_TO_CLASS.get(pred, pred)} ({conf:.2%})\n"
        f"Status: {status}\n\n"
        f"Selected Views: {', '.join(xlabels)}\n"
        f"Available Views: {av_text}\n"
        f"Best View: {best_view_name}\n"
        f"Best Slice Slot: {best_slice_local}\n"
        f"Selected Frame: {mid_frame}/{T - 1}\n\n"
        f"Prob NC:  {prob[0]:.4f}\n"
        f"Prob HCM: {prob[1]:.4f}\n"
        f"Prob DCM: {prob[2]:.4f}"
    )
    ax.text(0.02, 0.5, summary, fontsize=11, va="center", family="monospace", bbox=dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor="gray", alpha=0.95))
    ax.set_title("Prediction Summary")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    row.update({
        "true": IDX_TO_CLASS.get(true_label, str(true_label)),
        "pred": IDX_TO_CLASS.get(pred, str(pred)),
        "correct": bool(pred == true_label),
        "confidence": conf,
        "best_view": best_view_name,
        "best_view_index_in_selected_views": int(best_view_idx),
        "best_slice_slot": int(best_slice_local),
        "selected_clip_index": int(clip_idx),
        "selected_frame": int(mid_frame),
        "cam_available": cam is not None,
    })
    return row


def export_attention_visualizations(model: nn.Module, dataset: Dataset, device: torch.device, out_dir: Path, args, selected_views: Sequence[str]):
    """
    Export 2x3 explanation panels after testing.
    View-attention axes are restricted to selected_views only. For example, if the experiment uses
    [Cine4CH, CineSAX], Cine2CH and Cine3CH are not computed or plotted.
    """
    if not getattr(args, "save_attention_viz", False):
        return
    if not HAS_MPL:
        print0("matplotlib 不可用，跳过可视化。")
        return
    target_model = model.module if isinstance(model, DDP) else model
    target_model.eval()
    vis_dir = out_dir / "explanation_panels"
    ensure_dir(vis_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    rows = []
    for i, batch in enumerate(loader):
        if i >= int(getattr(args, "vis_num_samples", 12)):
            break
        batch = move_batch_to_device(batch, device)
        pid = str(batch["patient_id"][0])
        with torch.no_grad():
            logits = target_model(batch)
            prob = torch.softmax(logits.float(), dim=1)[0].detach().cpu().numpy()
            pred = int(np.argmax(prob))
        att = getattr(target_model, "last_attention", None) or {}
        if "view_weights" in att:
            view_w = att["view_weights"][0]
            view_w = view_w.detach().cpu().numpy() if torch.is_tensor(view_w) else np.asarray(view_w)
            view_w = view_w[:len(selected_views)]
        else:
            view_mask = batch["view_mask"][0].detach().cpu().numpy().astype(np.float32)[:len(selected_views)]
            view_w = view_mask / (view_mask.sum() + 1e-8)

        row = {
            "patient_id": pid,
            "pred": IDX_TO_CLASS[pred],
            "prob_NC": float(prob[0]),
            "prob_HCM": float(prob[1]),
            "prob_DCM": float(prob[2]),
            "selected_views": ";".join(selected_views),
        }
        # Only selected views are written to CSV.
        for v, w in zip(selected_views, view_w):
            row[f"view_weight_{v}"] = float(w)
        safe_pid = re.sub(r"[^A-Za-z0-9_.-]+", "_", pid)
        panel_path = vis_dir / f"{i:03d}_{safe_pid}_explanation_panel.png"
        try:
            row = _make_explanation_panel(target_model, batch, selected_views, panel_path, row)
        except Exception as e:
            row["panel_error"] = str(e)
            print0(f"可视化失败: patient={pid}, error={e}")
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(vis_dir / "explanation_summary.csv", index=False, encoding="utf-8-sig")
        df.to_csv(vis_dir / "attention_summary.csv", index=False, encoding="utf-8-sig")
        print0(f"✅ 可解释性面板已保存: {vis_dir}")


def build_optimizer(model: nn.Module, args):
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)
    param_groups = [
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ]
    return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args, steps_per_epoch: int):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.2)
    return None


# -----------------------------
# Main
# -----------------------------

def add_bool_arg(parser, name: str, default: bool, help_text: str = ""):
    """布尔参数默认由 RUN_CONFIG 决定；命令行也可用 --amp / --no-amp 临时覆盖。"""
    parser.add_argument(
        f"--{name}",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _comparison_root_dir(run_id: str) -> Path:
    output_dir = str(RUN_CONFIG.get("output_dir", "auto") or "auto").strip()
    if output_dir.lower() in {"", "auto", "none"}:
        return Path(f"./{Path(sys.argv[0]).stem}_results_{run_id}")
    return Path(output_dir) / f"fusion_strategy_{run_id}"


def _collect_comparison_summary(root: Path, experiments: List[Dict]):
    rows = []
    for exp in experiments:
        exp_name = str(exp.get("experiment_name", "") or "").strip()
        if not exp_name:
            continue
        metrics_path = root / exp_name / "test_metrics.json"
        hist_path = root / exp_name / "history.csv"
        profile_path = root / exp_name / "model_profile.json"
        row = {"experiment_name": exp_name, "model_type": exp.get("model_type", RUN_CONFIG.get("model_type", "")), "agg": exp.get("agg", RUN_CONFIG.get("agg", "")), "selected_views": exp.get("selected_views", RUN_CONFIG.get("selected_views", ""))}
        if metrics_path.exists():
            try:
                row.update(json.loads(metrics_path.read_text(encoding="utf-8")))
            except Exception as e:
                row["metrics_error"] = str(e)
        else:
            row["metrics_missing"] = str(metrics_path)
        if hist_path.exists():
            try:
                h = pd.read_csv(hist_path)
                if len(h) > 0:
                    if "val_macro_f1" in h.columns:
                        row["best_val_macro_f1"] = float(h["val_macro_f1"].max())
                        row["best_epoch"] = int(h.loc[h["val_macro_f1"].idxmax(), "epoch"])
                    if "time_sec" in h.columns:
                        row["total_epoch_time_sec"] = float(h["time_sec"].sum())
                        row["avg_epoch_time_sec"] = float(h["time_sec"].mean())
                        row["min_epoch_time_sec"] = float(h["time_sec"].min())
            except Exception:
                pass
        if profile_path.exists():
            try:
                prof = json.loads(profile_path.read_text(encoding="utf-8"))
                for k in ["params_total_m", "params_trainable_m", "nclips", "num_frames", "image_size"]:
                    if k in prof:
                        row[k] = prof[k]
            except Exception:
                pass
        rows.append(row)
    if rows:
        ensure_dir(root)
        df = pd.DataFrame(rows)
        df.to_csv(root / "comparison_summary.csv", index=False, encoding="utf-8-sig")
        if HAS_MPL:
            try:
                plot_cols = [c for c in ["accuracy", "macro_f1", "throughput_samples_per_sec", "forward_latency_ms_per_patient", "params_total_m", "peak_cuda_memory_mb"] if c in df.columns]
                if plot_cols:
                    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(10, 3.0 * len(plot_cols)))
                    if len(plot_cols) == 1:
                        axes = [axes]
                    labels = df["experiment_name"].astype(str).tolist()
                    for ax, col in zip(axes, plot_cols):
                        ax.bar(labels, df[col].astype(float))
                        ax.set_title(col)
                        ax.tick_params(axis="x", rotation=30)
                    fig.tight_layout()
                    fig.savefig(root / "comparison_summary_bars.png", dpi=300)
                    plt.close(fig)
            except Exception as e:
                print(f"对比实验汇总图绘制失败: {e}")
        print(f"\n✅ 对比实验汇总已保存: {root / 'comparison_summary.csv'}")


def maybe_run_compare_parent():
    """compare 模式：母舰调度逻辑，确保命名、数据、seed 绝对对齐。"""
    mode = str(RUN_CONFIG.get("experiment_mode", "single")).lower().strip()
    already_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    is_child = os.environ.get("CMR_COMPARE_CHILD", "0") == "1"
    if mode != "compare" or already_distributed or is_child:
        return

    experiments = RUN_CONFIG.get("compare_experiments", [])
    if not isinstance(experiments, list) or len(experiments) == 0:
        print("[WARN] experiment_mode='compare' 但 compare_experiments 为空，将按 single 运行。")
        return

    run_id = os.environ.get("CMR_RUN_ID", "").strip() or str(RUN_CONFIG.get("run_id", "")).strip() or now_string()
    root = _comparison_root_dir(run_id)
    ensure_dir(root)

    print("=" * 80)
    print(f"🚀 进入对比实验模式，共 {len(experiments)} 个实验")
    print(f"共同 run_id: {run_id}")
    print(f"共同输出根目录: {root}")
    print("=" * 80)

    # 记录真实的实验名字列表，用于最后汇总
    actual_exp_names = []

    for idx, exp in enumerate(experiments, 1):
        overrides = dict(exp)
        overrides["experiment_mode"] = "single"
        overrides["auto_launch_ddp"] = RUN_CONFIG.get("auto_launch_ddp", True)
        overrides.setdefault("selected_views", RUN_CONFIG.get("selected_views", ""))
        overrides.setdefault("run_id", run_id)
        overrides.setdefault("output_dir", str(root)) # 强制指向母舰创建的根目录

        # ====== 核心：在这里计算出真实名字，并强制覆盖给子进程 ======
        exp_name_raw = str(overrides.get("experiment_name", "auto")).strip()
        if not exp_name_raw or exp_name_raw.lower() == "auto":
            view_tag = "_".join([v.replace("Cine", "") for v in parse_csv_list(overrides.get("selected_views", ""))])
            model_tag = str(overrides.get("model_type", "model")).lower()
            agg_tag = str(overrides.get("agg", RUN_CONFIG.get("agg", ""))).lower().strip()
            agg_part = f"_{agg_tag}" if agg_tag else ""
            exp_name = f"exp_{idx:02d}_{model_tag}{agg_part}_{view_tag}"
        else:
            exp_name = exp_name_raw
        
        overrides["experiment_name"] = exp_name
        actual_exp_names.append(overrides) # 保存带有真实名字的配置

        env = os.environ.copy()
        env["CMR_COMPARE_CHILD"] = "1"
        env["CMR_RUN_ID"] = run_id
        env["CMR_RUN_CONFIG_OVERRIDES"] = json.dumps(overrides, ensure_ascii=False)
        
        cmd = [sys.executable, sys.argv[0]] + sys.argv[1:]
        print(f"\n[{idx}/{len(experiments)}] 正在运行: {exp_name}")
        subprocess.run(cmd, env=env, check=True)

    # 全部跑完后，使用带有真实名字的列表进行汇总
    _collect_comparison_summary(root, actual_exp_names)
    sys.exit(0)


def maybe_auto_launch_ddp():
    """
    让多卡也能用 `python xxx.py` 启动。
    当 RUN_CONFIG['gpu_mode']='ddp' 且当前不是 torchrun 子进程时，自动重新用 torchrun 拉起。
    """
    gpu_mode = str(RUN_CONFIG.get("gpu_mode", "single")).lower().strip()
    auto_launch = bool(RUN_CONFIG.get("auto_launch_ddp", True))
    already_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if gpu_mode != "ddp" or already_distributed or not auto_launch:
        return

    visible = str(RUN_CONFIG.get("visible_gpus", "0")).strip()
    nproc = len([x for x in visible.split(",") if x.strip()])
    if nproc <= 1:
        print("[WARN] gpu_mode='ddp' 但 visible_gpus 只有 1 张卡，将按单卡运行。")
        return

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--standalone",
        sys.argv[0],
    ] + sys.argv[1:]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["CMR_RUN_ID"] = env.get("CMR_RUN_ID", str(RUN_CONFIG.get("run_id", "")).strip() or now_string())
    print("=" * 80)
    print(f"检测到 RUN_CONFIG['gpu_mode']='ddp'，自动启动 {nproc} 卡 DDP：")
    print(" ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, env=env, check=True)
    sys.exit(0)


def parse_args():
    cfg = RUN_CONFIG
    parser = argparse.ArgumentParser(description="Multi-view Video Swin for cardiac cine NIfTI classification")

    # Experiment mode
    parser.add_argument("--experiment_mode", type=str, default=cfg.get("experiment_mode", "single"), choices=["single", "compare"],
                        help="single 跑一个模型；compare 按 RUN_CONFIG['compare_experiments'] 顺序依次跑多个实验")

    # Data
    parser.add_argument("--data_path", type=str, default=cfg["data_path"], help="Root path containing NC/HCM/DCM folders")
    parser.add_argument("--selected_views", type=str, default=cfg["selected_views"],
                        help="Views used as model input, e.g. Cine4CH,CineSAX or Cine2CH,Cine3CH,Cine4CH,CineSAX")
    parser.add_argument("--cohort_required_views", type=str, default=cfg["cohort_required_views"],
                        help="Strict complete-case cohort filter. If set, all experiments use patients with these views, while selected_views controls model input.")
    add_bool_arg(parser, "allow_missing_selected_views", cfg["allow_missing_selected_views"],
                 "If true, include patients with at least one selected view; missing views are masked. Default requires all selected views.")
    parser.add_argument("--slice_plan", type=str, default=cfg["slice_plan"],
                        help="Max slice slots per view. Insufficient slices are zero-padded and masked.")
    parser.add_argument("--num_frames", type=int, default=cfg["num_frames"])
    parser.add_argument("--image_size", type=int, default=cfg["image_size"])
    parser.add_argument("--min_frames_per_slice", type=int, default=cfg["min_frames_per_slice"])
    add_bool_arg(parser, "use_cache", cfg["use_cache"], "Cache preprocessed slice clips to reduce NIfTI IO")
    parser.add_argument("--cache_dir", type=str, default=cfg["cache_dir"], help="Default: data_path/.cmr_clip_cache")
    add_bool_arg(parser, "augment", cfg["augment"], "Use simple spatial/intensity augmentation for training")

    # Prognosis/survival interface; empty prognosis_csv means classification-only.
    parser.add_argument("--prognosis_csv", type=str, default=cfg.get("prognosis_csv", ""))
    parser.add_argument("--prognosis_id_col", type=str, default=cfg.get("prognosis_id_col", "受试者ID"))
    parser.add_argument("--prognosis_event_col", type=str, default=cfg.get("prognosis_event_col", "是否发生结局事件"))
    parser.add_argument("--prognosis_time_col", type=str, default=cfg.get("prognosis_time_col", "发生时间"))
    parser.add_argument("--prognosis_censor_col", type=str, default=cfg.get("prognosis_censor_col", "是否为删失样本"))

    # Train split
    parser.add_argument("--train_ratio", type=float, default=cfg["train_ratio"])
    parser.add_argument("--val_ratio", type=float, default=cfg["val_ratio"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])

    # Model
    parser.add_argument("--model_type", type=str, default=cfg.get("model_type", "videoswin"), help="videoswin / resnet18 / resnet50 / densenet121 / efficientnet_b0 / convnext_tiny")
    parser.add_argument("--model_name", type=str, default=cfg.get("model_name", ""), help="Optional backbone name for cnn2d/resnet/efficientnet/densenet/convnext")
    parser.add_argument("--model_weights_path", type=str, default=cfg.get("model_weights_path", ""), help="Local 2D backbone weight path for comparison models")
    parser.add_argument("--frame_samples", type=int, default=cfg.get("frame_samples", 0), help="For 2D CNN baselines: number of frames sampled from each slice clip. 0 means use all num_frames frames.")
    add_bool_arg(parser, "pretrained", cfg["pretrained"], "Load local weights; no internet download is used")
    parser.add_argument("--weights_path", type=str, default=cfg["weights_path"], help="Local .pth path. No internet download is used.")
    parser.add_argument("--agg", type=str, default=cfg["agg"], choices=FUSION_AGG_CHOICES + ["attention", "mean"],
                        help="Multi-view fusion strategy. attention/mean are legacy aliases for hier_attn/hier_mean.")
    parser.add_argument("--dropout", type=float, default=cfg["dropout"])
    add_bool_arg(parser, "freeze_backbone", cfg["freeze_backbone"], "Freeze Video Swin backbone")
    parser.add_argument("--backbone_chunk_size", type=int, default=cfg["backbone_chunk_size"],
                        help="If >0, process valid clips through backbone in chunks to reduce memory")
    add_bool_arg(parser, "compile", cfg["compile"], "Use torch.compile; first epoch may be slower")

    # Optimization
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch_size", type=int, default=cfg["batch_size"], help="Per-GPU batch size under DDP")
    parser.add_argument("--workers", type=int, default=cfg["workers"], help="DataLoader workers per process/GPU")
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--backbone_lr_mult", type=float, default=cfg["backbone_lr_mult"])
    parser.add_argument("--min_lr", type=float, default=cfg["min_lr"])
    parser.add_argument("--weight_decay", type=float, default=cfg["weight_decay"])
    parser.add_argument("--label_smoothing", type=float, default=cfg["label_smoothing"])
    parser.add_argument("--grad_accum_steps", type=int, default=cfg["grad_accum_steps"])
    parser.add_argument("--grad_clip", type=float, default=cfg["grad_clip"])
    add_bool_arg(parser, "amp", cfg["amp"], "Use mixed precision")
    parser.add_argument("--scheduler", type=str, default=cfg["scheduler"], choices=["cosine", "step", "none"])
    parser.add_argument("--early_stop_patience", type=int, default=cfg.get("early_stop_patience", 0),
                        help="Early stopping patience by validation macro-F1. <=0 disables early stopping.")
    parser.add_argument("--early_stop_min_delta", type=float, default=cfg.get("early_stop_min_delta", 0.0),
                        help="Minimum validation macro-F1 improvement required to reset early stopping counter.")

    # Visualization
    add_bool_arg(parser, "plot_roc", cfg.get("plot_roc", True), "Save multiclass ROC curves")
    add_bool_arg(parser, "plot_history", cfg.get("plot_history", True), "Save training history plot")
    add_bool_arg(parser, "save_attention_viz", cfg.get("save_attention_viz", True), "Save view/slice attention visualizations for a few test samples")
    parser.add_argument("--vis_num_samples", type=int, default=cfg.get("vis_num_samples", 12))

    # IO
    parser.add_argument("--output_dir", type=str, default=cfg["output_dir"],
                        help="Use 'auto' to create ./script_name_results_YYYYMMDD_HHMMSS/experiment_name")
    parser.add_argument("--experiment_name", type=str, default=cfg["experiment_name"])
    parser.add_argument("--run_id", type=str, default=cfg.get("run_id", ""),
                        help="Optional fixed run id. Empty means current timestamp.")
    parser.add_argument("--resume", type=str, default=cfg["resume"], help="Resume checkpoint path")
    add_bool_arg(parser, "eval_only", cfg["eval_only"], "Only evaluate checkpoint")
    parser.add_argument("--print_freq", type=int, default=cfg["print_freq"])

    return parser.parse_args()

def main():
    maybe_run_compare_parent()
    maybe_auto_launch_ddp()
    args = parse_args()
    device, rank, world_size, distributed = setup_distributed()
    seed_everything(args.seed + rank)

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    selected_views = parse_csv_list(args.selected_views)
    validate_views(selected_views)
    cohort_required_views = parse_csv_list(args.cohort_required_views)
    validate_views(cohort_required_views)
    slice_plan = parse_slice_plan(args.slice_plan)

    out_dir = build_output_dir(args, selected_views)
    if is_main_process():
        ensure_dir(out_dir)
        resolved_config = vars(args).copy()
        resolved_config["resolved_output_dir"] = str(out_dir)
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(resolved_config, f, ensure_ascii=False, indent=2)

    print0("=" * 80)
    print0("Multi-view CMR 三分类 / 对比实验框架")
    print0(f"model_type={args.model_type}, model_name={getattr(args, 'model_name', '')}")
    print0(f"device={device}, distributed={distributed}, world_size={world_size}")
    print0(f"selected_views={selected_views}")
    print0(f"slice_plan={slice_plan}")
    print0(f"num_frames={args.num_frames}, image_size={args.image_size}")
    print0(f"output_dir={out_dir}")
    print0("=" * 80)

    # Scan once to get samples.
    base_dataset = MultiViewCardiacNiftiDataset(
        root_dir=args.data_path,
        selected_views=selected_views,
        slice_plan=slice_plan,
        num_frames=args.num_frames,
        image_size=args.image_size,
        mode="val",
        required_views=cohort_required_views,
        allow_missing_selected_views=args.allow_missing_selected_views,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir if args.cache_dir else None,
        augment=False,
        min_frames_per_slice=args.min_frames_per_slice,
        verbose=True,
    )
    if len(base_dataset.samples) < 3:
        raise RuntimeError("有效样本太少，请检查 data_path、selected_views、cohort_required_views 或文件命名格式。")

    # 预后/生存接口：默认 prognosis_csv 为空，不影响三分类；填写后会随 split CSV 一起保存。
    attach_prognosis_info(base_dataset.samples, args)

    train_samples, val_samples, test_samples = stratified_split_samples(
        base_dataset.samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print0(f"Train: {len(train_samples)} | {class_count_string(train_samples)}")
    print0(f"Val:   {len(val_samples)} | {class_count_string(val_samples)}")
    print0(f"Test:  {len(test_samples)} | {class_count_string(test_samples)}")

    if is_main_process():
        save_split_csv(train_samples, out_dir / "split_train.csv")
        save_split_csv(val_samples, out_dir / "split_val.csv")
        save_split_csv(test_samples, out_dir / "split_test.csv")

    # Recreate datasets with split samples and train/val modes.
    common_ds_kwargs = dict(
        root_dir=args.data_path,
        selected_views=selected_views,
        slice_plan=slice_plan,
        num_frames=args.num_frames,
        image_size=args.image_size,
        required_views=cohort_required_views,
        allow_missing_selected_views=args.allow_missing_selected_views,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir if args.cache_dir else None,
        min_frames_per_slice=args.min_frames_per_slice,
        verbose=False,
    )
    train_ds = MultiViewCardiacNiftiDataset(samples=train_samples, mode="train", augment=args.augment, **common_ds_kwargs)
    val_ds = MultiViewCardiacNiftiDataset(samples=val_samples, mode="val", augment=False, **common_ds_kwargs)
    test_ds = MultiViewCardiacNiftiDataset(samples=test_samples, mode="val", augment=False, **common_ds_kwargs)

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if distributed else None
    test_sampler = DistributedSampler(test_ds, shuffle=False, drop_last=False) if distributed else None

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=(train_sampler is None), drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, sampler=test_sampler, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_model(args, selected_views, slice_plan).to(device)
    model_profile = count_parameters(model)
    model_profile.update({
        "model_type": str(args.model_type),
        "selected_views": ",".join(selected_views),
        "nclips": int(sum(int(slice_plan.get(v, 0)) for v in selected_views)),
        "num_frames": int(args.num_frames),
        "image_size": int(args.image_size),
        "slice_plan": dict(slice_plan),
        # === 新增：将完整的网络结构字符串放入 JSON 中 ===
        "network_structure": repr(model)  
    })
    if is_main_process():
        with open(out_dir / "model_profile.json", "w", encoding="utf-8") as f:
            json.dump(model_profile, f, ensure_ascii=False, indent=2)
        print0(
            f"模型复杂度: total={model_profile['params_total_m']:.2f}M, "
            f"trainable={model_profile['params_trainable_m']:.2f}M, "
            f"nclips={model_profile['nclips']}, T={model_profile['num_frames']}\n"
            f"✅ 网络完整结构已合并保存至: {out_dir / 'model_profile.json'}"
        )

        # ================= 新增功能：保存网络结构 =================
        model_repr = repr(model)  # 获取 PyTorch 模型的完整结构字符串
        with open(out_dir / "network_structure.txt", "w", encoding="utf-8") as f:
            f.write(model_repr)
        print0(f"✅ 已将网络详细结构保存至: {out_dir / 'network_structure.txt'}")
        # ==========================================================

    if args.compile:
        print0("使用 torch.compile 编译模型；首次运行可能较慢。")
        model = torch.compile(model)

    if distributed:
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))], output_device=int(os.environ.get("LOCAL_RANK", 0)), find_unused_parameters=False,broadcast_buffers=False)

    optimizer = build_optimizer(model.module if isinstance(model, DDP) else model, args)
    scheduler = build_scheduler(optimizer, args, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    reset_peak_cuda_memory(device)
    experiment_wall_start = time.perf_counter()

    start_epoch = 1
    best_val_f1 = -1.0
    no_improve_epochs = 0
    best_path = out_dir / "best_model.pt"

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        target_model = model.module if isinstance(model, DDP) else model
        target_model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        no_improve_epochs = int(ckpt.get("no_improve_epochs", 0))
        print0(f"已恢复训练: {args.resume}, start_epoch={start_epoch}, best_val_f1={best_val_f1:.4f}, no_improve_epochs={no_improve_epochs}")

    history = []

    if args.eval_only:
        if not args.resume:
            print0("eval_only 模式建议通过 --resume 指定模型权重。现在将评估当前初始化模型。")
        test_results = evaluate(model, test_loader, device, args, phase="test")
        test_results.update(model_profile)
        test_results["peak_cuda_memory_mb"] = get_peak_cuda_memory_mb(device)
        if is_main_process():
            metrics, report = save_eval_outputs(test_results, out_dir / "test")
            print(report)
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        cleanup_distributed()
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_stats = run_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        val_results = evaluate(model, val_loader, device, args, phase="val")
        val_metrics = summarize_metrics(val_results)

        if scheduler is not None:
            scheduler.step()

        val_f1 = val_metrics.get("macro_f1", 0.0) or 0.0
        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "val_acc": val_metrics.get("accuracy", 0.0),
            "val_bal_acc": val_metrics.get("balanced_accuracy", 0.0),
            "val_macro_f1": val_metrics.get("macro_f1", 0.0),
            "val_auc_macro": val_metrics.get("ovr_auc_macro", None),
            "lr": get_lr(optimizer),
            "time_sec": epoch_time,
            "best_val_macro_f1": best_val_f1,
            "no_improve_epochs": no_improve_epochs,
        }
        history.append(row)

        if is_main_process():
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={row['train_loss']:.4f} train_acc={100*row['train_acc']:.2f}% | "
                f"val_acc={100*row['val_acc']:.2f}% val_macro_f1={row['val_macro_f1']:.4f} | "
                f"time={epoch_time:.1f}s"
            )
            pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
            if args.plot_history:
                save_history_plot(history, out_dir)

        improved = val_f1 > (best_val_f1 + args.early_stop_min_delta)
        if improved:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            if is_main_process():
                target_model = model.module if isinstance(model, DDP) else model
                ckpt = {
                    "epoch": epoch,
                    "model": target_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_val_f1": best_val_f1,
                    "no_improve_epochs": no_improve_epochs,
                    "args": vars(args),
                    "selected_views": selected_views,
                    "slice_plan": slice_plan,
                }
                torch.save(ckpt, best_path)
                print(f"  ✅ 保存新的最佳模型: {best_path} | best_val_macro_f1={best_val_f1:.4f}")
        else:
            no_improve_epochs += 1
            if is_main_process():
                if args.early_stop_patience > 0:
                    print(f"  ↳ val_macro_f1 未提升：{no_improve_epochs}/{args.early_stop_patience} | best={best_val_f1:.4f}")
                else:
                    print(f"  ↳ val_macro_f1 未提升：{no_improve_epochs} | best={best_val_f1:.4f} | 早停关闭")

        should_stop = args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience
        if distributed:
            dist.barrier()
        if should_stop:
            print0(f"🛑 触发早停：连续 {no_improve_epochs} 轮 val_macro_f1 未提升，最佳 val_macro_f1={best_val_f1:.4f}")
            break

    total_train_time_sec = time.perf_counter() - experiment_wall_start
    avg_epoch_time_sec = float(np.mean([h.get("time_sec", 0.0) for h in history])) if len(history) > 0 else 0.0
    best_epoch = int(max(history, key=lambda x: x.get("val_macro_f1", -1)).get("epoch", 0)) if len(history) > 0 else 0

    # Load best and test.
    if best_path.exists():
        if distributed:
            dist.barrier()
        ckpt = torch.load(best_path, map_location="cpu")
        target_model = model.module if isinstance(model, DDP) else model
        target_model.load_state_dict(ckpt["model"], strict=True)
        print0(f"加载最佳模型进行测试: {best_path}")

    test_results = evaluate(model, test_loader, device, args, phase="test")
    test_results.update(model_profile)
    test_results["peak_cuda_memory_mb"] = get_peak_cuda_memory_mb(device)
    test_results["total_train_time_sec"] = float(total_train_time_sec)
    test_results["avg_epoch_time_sec"] = float(avg_epoch_time_sec)
    test_results["best_epoch"] = int(best_epoch)
    test_results["best_val_macro_f1"] = float(best_val_f1)
    if is_main_process():
        metrics, report = save_eval_outputs(test_results, out_dir / "test")
        print("\n" + "=" * 80)
        print("Test Classification Report")
        print("=" * 80)
        print(report)
        print("Test Metrics:")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        export_attention_visualizations(model, test_ds, device, out_dir, args, selected_views)
        print(f"结果已保存到: {out_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
