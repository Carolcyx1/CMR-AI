"""Fair direct-vs-diagnosis prognosis screening with exact full-risk-set Cox.

The image encoder is first evaluated as an initialization.  A disease-specific
full-risk-set Cox model becomes the prognosis teacher.  A private encoder copy
is then adapted to that teacher, after which an exact Cox model is refitted.
No diagnosis loss, logits, or diagnosis teacher is used in prognosis training.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


VERSION = "diagnosis_to_prognosis_deterministic_raw_encoder_v6"
RAW_ENCODER_FEATURE_POLICY = "raw_view_embeddings_deterministic_zero_v2"


def _prognosis_feature_dir(category):
    """Return the centralized persistent feature directory."""
    configured = os.environ.get("FINAL_PROGNOSIS_FEATURE_ROOT", "").strip()
    root = (
        Path(configured)
        if configured
        else Path(__file__).resolve().parent / "features"
    )
    directory = root / category
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class FullRiskCoxHead(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class EncoderRiskStudent(nn.Module):
    def __init__(self, encoder, teacher_head):
        super().__init__()
        self.encoder = encoder
        self.risk_head = teacher_head

    def forward(self, batch):
        return self.risk_head(self.encoder.forward_features(batch))


class AdaptiveFusionCoxHead(nn.Module):
    """Information-bottleneck branches with patient-wise confidence fusion."""

    def __init__(self, input_dims, bottleneck_dim, hidden, dropout):
        super().__init__()
        self.input_dims = [int(x) for x in input_dims]
        bottleneck_dim = int(bottleneck_dim)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            for dim in self.input_dims
        ])
        self.confidence = nn.ModuleList([
            nn.Linear(bottleneck_dim, 1) for _ in self.input_dims
        ])
        self.risk = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, features):
        if not isinstance(features, (tuple, list)):
            features = (features,)
        encoded = [layer(x) for layer, x in zip(self.branches, features)]
        logits = torch.cat(
            [gate(x) for gate, x in zip(self.confidence, encoded)], dim=1
        )
        weights = torch.softmax(logits, dim=1)
        fused = sum(
            x * weights[:, index:index + 1]
            for index, x in enumerate(encoded)
        )
        return self.risk(fused).squeeze(-1)


class ResidualDiagnosisCoxHead(nn.Module):
    """Raw prognosis risk plus a deliberately small diagnosis-risk correction."""

    def __init__(self, raw_dim, diagnosis_dim, bottleneck_dim, dropout, max_gate):
        super().__init__()
        self.raw_risk = nn.Sequential(
            nn.LayerNorm(int(raw_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(raw_dim), 1),
        )
        self.diagnosis_residual = nn.Sequential(
            nn.LayerNorm(int(diagnosis_dim)),
            nn.Linear(int(diagnosis_dim), int(bottleneck_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(bottleneck_dim), 1),
        )
        # Starts near zero, so diagnosis information must earn its contribution.
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.max_gate = float(max_gate)

    def forward(self, features):
        raw, diagnosis = features
        gate = torch.sigmoid(self.gate_logit) * self.max_gate
        return (
            self.raw_risk(raw).squeeze(-1)
            + gate * self.diagnosis_residual(diagnosis).squeeze(-1)
        )


class SharedTokenResidualCoxHead(nn.Module):
    """Parameter-shared multilevel encoder to prevent flattened-feature memorization."""

    def __init__(self, raw_dim, token_dim, projection_dim, dropout, max_gate):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_tokens = 21  # patient + 4 views + 8 raw stats + 8 diagnosis stats
        projection_dim = int(projection_dim)
        self.raw_risk = nn.Sequential(
            nn.LayerNorm(int(raw_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(raw_dim), 1),
        )
        # Exactly the same projection is used for every level and every view.
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, projection_dim),
            nn.GELU(),
        )
        # patient plus mean/std/max summaries for three token groups.
        summary_dim = projection_dim * 10
        self.residual_risk = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(summary_dim, 1),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.max_gate = float(max_gate)

    @staticmethod
    def _summarize(tokens):
        return torch.cat(
            [
                tokens.mean(dim=1),
                tokens.std(dim=1, unbiased=False),
                tokens.amax(dim=1),
            ],
            dim=1,
        )

    def forward(self, features):
        raw, rich = features
        if rich.shape[1] != self.num_tokens * self.token_dim:
            raise ValueError(
                f"Expected {self.num_tokens * self.token_dim} multilevel values, "
                f"got {rich.shape[1]}"
            )
        tokens = self.shared_projection(
            rich.reshape(rich.shape[0], self.num_tokens, self.token_dim)
        )
        summary = torch.cat(
            [
                tokens[:, 0],
                self._summarize(tokens[:, 1:5]),
                self._summarize(tokens[:, 5:13]),
                self._summarize(tokens[:, 13:21]),
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.gate_logit) * self.max_gate
        return (
            self.raw_risk(raw).squeeze(-1)
            + gate * self.residual_risk(summary).squeeze(-1)
        )


class RiskLevelFusionCoxHead(nn.Module):
    """Two independent low-dimensional risks fused only at score level."""

    def __init__(self, raw_dim, diagnosis_dim, dropout, max_gate):
        super().__init__()
        self.raw_risk = nn.Sequential(
            nn.Dropout(float(dropout)), nn.Linear(int(raw_dim), 1, bias=False)
        )
        self.diagnosis_risk = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(diagnosis_dim), 1, bias=False),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.max_gate = float(max_gate)

    def forward(self, features):
        raw, diagnosis = features
        gate = torch.sigmoid(self.gate_logit) * self.max_gate
        return (
            self.raw_risk(raw).squeeze(-1)
            + gate * self.diagnosis_risk(diagnosis).squeeze(-1)
        )


class MotionAugmentedResidualCoxHead(nn.Module):
    """F-style frozen phenotype residual plus explicit cine-motion risk."""

    def __init__(
        self, raw_dim, diagnosis_dim, motion_dim, bottleneck_dim,
        dropout, diagnosis_max_gate, motion_max_gate,
    ):
        super().__init__()
        self.raw_risk = nn.Sequential(
            nn.LayerNorm(int(raw_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(raw_dim), 1),
        )
        self.diagnosis_risk = nn.Sequential(
            nn.LayerNorm(int(diagnosis_dim)),
            nn.Linear(int(diagnosis_dim), int(bottleneck_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(bottleneck_dim), 1),
        )
        self.motion_risk = nn.Sequential(
            nn.LayerNorm(int(motion_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(motion_dim), 1),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.motion_gate_logit = nn.Parameter(torch.tensor(-2.5))
        self.diagnosis_max_gate = float(diagnosis_max_gate)
        self.motion_max_gate = float(motion_max_gate)

    def forward(self, features):
        raw, diagnosis, motion = features
        diagnosis_gate = (
            torch.sigmoid(self.gate_logit) * self.diagnosis_max_gate
        )
        motion_gate = (
            torch.sigmoid(self.motion_gate_logit) * self.motion_max_gate
        )
        return (
            self.raw_risk(raw).squeeze(-1)
            + diagnosis_gate * self.diagnosis_risk(diagnosis).squeeze(-1)
            + motion_gate * self.motion_risk(motion).squeeze(-1)
        )


class SegmentationLinearCoxHead(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(float(dropout)), nn.Linear(int(dim), 1, bias=False)
        )

    def forward(self, features):
        if isinstance(features, (tuple, list)):
            features = features[0]
        return self.net(features).squeeze(-1)


class LateRiskFusionModel(nn.Module):
    def __init__(
        self, phenotype_head, function_head, alpha,
        phenotype_mean, phenotype_std, function_mean, function_std,
    ):
        super().__init__()
        self.phenotype_head = phenotype_head
        self.function_head = function_head
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("phenotype_mean", torch.tensor(float(phenotype_mean)))
        self.register_buffer("phenotype_std", torch.tensor(float(phenotype_std)))
        self.register_buffer("function_mean", torch.tensor(float(function_mean)))
        self.register_buffer("function_std", torch.tensor(float(function_std)))

    def forward(self, features):
        raw, multilevel, functional = features
        phenotype = self.phenotype_head((raw, multilevel))
        function = self.function_head((functional,))
        phenotype = (phenotype - self.phenotype_mean) / self.phenotype_std
        function = (function - self.function_mean) / self.function_std
        return self.alpha * phenotype + (1.0 - self.alpha) * function


class CrossFoldLateFusionModel(nn.Module):
    """Disease-specific prognosis Teacher ensemble built from OOF folds."""

    def __init__(
        self, phenotype_heads, function_heads, alpha,
        phenotype_means, phenotype_stds, function_means, function_stds,
    ):
        super().__init__()
        self.phenotype_heads = nn.ModuleList(phenotype_heads)
        self.function_heads = nn.ModuleList(function_heads)
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        for name, values in (
            ("phenotype_means", phenotype_means),
            ("phenotype_stds", phenotype_stds),
            ("function_means", function_means),
            ("function_stds", function_stds),
        ):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))

    def forward(self, features):
        raw, multilevel, functional = features
        phenotype_predictions, function_predictions = [], []
        for index, (phenotype_head, function_head) in enumerate(
            zip(self.phenotype_heads, self.function_heads)
        ):
            phenotype = phenotype_head((raw, multilevel))
            function = function_head((functional,))
            phenotype_predictions.append(
                (phenotype - self.phenotype_means[index])
                / self.phenotype_stds[index]
            )
            function_predictions.append(
                (function - self.function_means[index])
                / self.function_stds[index]
            )
        phenotype = torch.stack(phenotype_predictions).mean(dim=0)
        function = torch.stack(function_predictions).mean(dim=0)
        return self.alpha * phenotype + (1.0 - self.alpha) * function


def _view_mean_std(clip_features, clip_mask, clip_view_ids, num_views):
    outputs = []
    for view_id in range(int(num_views)):
        valid = clip_mask.bool() & (clip_view_ids.long() == view_id)
        weight = valid.float().unsqueeze(2)
        denom = weight.sum(dim=1).clamp_min(1.0)
        mean = (clip_features.float() * weight).sum(dim=1) / denom
        variance = (
            (clip_features.float() - mean.unsqueeze(1)).square() * weight
        ).sum(dim=1) / denom
        outputs.extend([mean, variance.clamp_min(0).sqrt()])
    return torch.cat(outputs, dim=1)


def _build_frozen_multilevel_vector(split_data, num_views):
    """View/slice distribution features; the diagnosis encoder stays frozen."""
    raw_stats = _view_mean_std(
        split_data["backbone_clip_features"],
        split_data["clip_mask"],
        split_data["clip_view_ids"],
        num_views,
    )
    diagnosis_stats = _view_mean_std(
        split_data["diagnosis_clip_features"],
        split_data["clip_mask"],
        split_data["clip_view_ids"],
        num_views,
    )
    diagnosis_views = split_data["diagnosis_view_features"].float().flatten(1)
    diagnosis_patient = split_data["diagnosis_patient_feature"].float()
    return torch.cat(
        [diagnosis_patient, diagnosis_views, raw_stats, diagnosis_stats], dim=1
    )


def _device_batch(items, device):
    batch = default_collate(items)
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _dataset(ns, samples, views, required, plan, args, mode):
    return ns["MultiViewCardiacNiftiDataset"](
        root_dir=args.data_path,
        selected_views=views,
        slice_plan=plan,
        num_frames=int(args.num_frames),
        image_size=int(args.image_size),
        samples=samples,
        mode=mode,
        required_views=required,
        allow_missing_selected_views=bool(args.allow_missing_selected_views),
        use_cache=bool(args.use_cache),
        cache_dir=args.cache_dir if args.cache_dir else None,
        min_frames_per_slice=int(args.min_frames_per_slice),
        verbose=False,
    )


def _new_encoder(ns, views, plan, args, diagnosis_init, device):
    encoder = ns["MultiViewVideoSwinFeatureExtractor"](
        selected_views=views,
        slice_plan=plan,
        weights_path=args.weights_path,
        pretrained=bool(args.pretrained),
        agg=args.diagnosis_encoder_agg,
        dropout=float(args.dropout),
        backbone_chunk_size=int(args.backbone_chunk_size),
    )
    if diagnosis_init:
        encoder.load_classification_checkpoint(args.classification_ckpt)
    else:
        # The raw/Kinetics control has never learned anatomical view identity.
        # A frozen random view embedding would create arbitrary view-specific
        # offsets shared by A_direct, A_matched, F, and K raw branches.
        with torch.no_grad():
            encoder.view_embeddings.zero_()
        ns["print0"](
            "Raw patient encoder: view_embeddings set deterministically to zero "
            f"(policy={RAW_ENCODER_FEATURE_POLICY})"
        )
    return encoder.to(device)


@torch.no_grad()
def extract_features(encoder, dataset, device, args):
    encoder.eval()
    batch_size = max(1, int(args.fullrisk_extract_batch_size))
    forward = encoder
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        forward = nn.DataParallel(encoder)
    outputs = []
    for start in range(0, len(dataset), batch_size):
        items = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        batch = _device_batch(items, device)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=(bool(args.amp) and device.type == "cuda"),
        ):
            feature = forward(batch)
        outputs.append(feature.detach().float().cpu())
    features = torch.cat(outputs, dim=0)
    if not torch.isfinite(features).all():
        raise FloatingPointError("Encoder produced non-finite prognosis features.")
    return features


def _motion_descriptors(batch, num_views):
    """Low-cost cine motion/phase descriptors, aggregated separately per view."""
    videos = batch["videos"].float()
    mask = batch["clip_mask"].bool()
    view_ids = batch["clip_view_ids"].long()
    # A robust intensity-time curve for every cine slice.
    curve = videos.mean(dim=(2, 4, 5))
    delta = curve[:, :, 1:] - curve[:, :, :-1]
    scale = curve.std(dim=2).clamp_min(1e-4)
    desc = torch.stack(
        [
            curve.mean(dim=2),
            curve.std(dim=2),
            curve.amax(dim=2) - curve.amin(dim=2),
            delta.abs().mean(dim=2),
            delta.abs().amax(dim=2),
            curve.argmax(dim=2).float() / max(curve.shape[2] - 1, 1),
            curve.argmin(dim=2).float() / max(curve.shape[2] - 1, 1),
        ],
        dim=2,
    )
    desc[:, :, :5] = desc[:, :, :5] / scale.unsqueeze(2)
    outputs = []
    for view_id in range(int(num_views)):
        valid = mask & (view_ids == view_id)
        weight = valid.float().unsqueeze(2)
        denom = weight.sum(dim=1).clamp_min(1.0)
        mean = (desc * weight).sum(dim=1) / denom
        variance = ((desc - mean.unsqueeze(1)).square() * weight).sum(dim=1) / denom
        outputs.extend([mean, variance.clamp_min(0).sqrt()])
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def extract_features_and_motion(encoder, dataset, device, args, num_views):
    encoder.eval()
    batch_size = max(1, int(args.fullrisk_extract_batch_size))
    forward = encoder
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        forward = nn.DataParallel(encoder)
    features, motion = [], []
    for start in range(0, len(dataset), batch_size):
        items = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        batch = _device_batch(items, device)
        motion.append(_motion_descriptors(batch, num_views).cpu())
        with torch.amp.autocast(
            device_type="cuda",
            enabled=(bool(args.amp) and device.type == "cuda"),
        ):
            feature = forward(batch)
        features.append(feature.detach().float().cpu())
    result = torch.cat(features), torch.cat(motion)
    if not all(torch.isfinite(x).all() for x in result):
        raise FloatingPointError("Encoder or motion branch produced non-finite features.")
    return result


def _one_clip_optical_flow_descriptor(clip, phases=16, size=48):
    import cv2

    frames = clip.squeeze(0).float().cpu().numpy()
    indices = np.linspace(0, len(frames) - 1, int(phases), dtype=int)
    frames = frames[indices]
    resized = np.stack([
        cv2.resize(frame, (int(size), int(size)), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]).astype(np.float32)
    # Frame-wise robust scaling removes scanner intensity scale while retaining motion.
    lo = np.percentile(resized, 2, axis=(1, 2), keepdims=True)
    hi = np.percentile(resized, 98, axis=(1, 2), keepdims=True)
    resized = np.clip((resized - lo) / np.maximum(hi - lo, 1e-5), 0, 1)
    resized = np.ascontiguousarray((resized * 255.0).astype(np.uint8))
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    rx, ry = xx - (size - 1) / 2, yy - (size - 1) / 2
    radius = np.sqrt(rx * rx + ry * ry).clip(1.0)
    magnitude_curve, p90_curve, divergence_curve = [], [], []
    coherence_curve, radial_curve = [], []
    for index in range(len(resized) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            resized[index], resized[index + 1], None,
            0.5, 2, 9, 2, 5, 1.1, 0,
        )
        fx, fy = flow[..., 0], flow[..., 1]
        magnitude = np.sqrt(fx * fx + fy * fy)
        mean_magnitude = float(magnitude.mean())
        divergence = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
        radial = (fx * rx + fy * ry) / radius
        magnitude_curve.append(mean_magnitude)
        p90_curve.append(float(np.percentile(magnitude, 90)))
        divergence_curve.append(float(np.mean(np.abs(divergence))))
        coherence_curve.append(
            float(np.sqrt(fx.mean() ** 2 + fy.mean() ** 2) / (mean_magnitude + 1e-6))
        )
        radial_curve.append(float(radial.mean()))
    mag = np.asarray(magnitude_curve, dtype=np.float32)
    p90 = np.asarray(p90_curve, dtype=np.float32)
    div = np.asarray(divergence_curve, dtype=np.float32)
    coh = np.asarray(coherence_curve, dtype=np.float32)
    radial = np.asarray(radial_curve, dtype=np.float32)
    return np.asarray([
        mag.mean(), mag.std(), mag.max(),
        p90.mean(), p90.max(),
        div.mean(), div.max(),
        coh.mean(), coh.std(),
        radial.min(), radial.max(), np.mean(np.abs(radial)),
        float(mag.argmax()) / max(len(mag) - 1, 1),
        float(radial.argmin()) / max(len(radial) - 1, 1),
    ], dtype=np.float32)


def _patient_optical_flow_descriptor(item, num_views, phases, size):
    videos = item["videos"]
    mask = item["clip_mask"].bool()
    view_ids = item["clip_view_ids"].long()
    descriptor_dim = 14
    outputs = []
    for view_id in range(int(num_views)):
        descriptors = [
            _one_clip_optical_flow_descriptor(
                videos[index], phases=phases, size=size
            )
            for index in range(len(videos))
            if bool(mask[index]) and int(view_ids[index]) == view_id
        ]
        if descriptors:
            values = np.stack(descriptors)
            outputs.extend([values.mean(axis=0), values.std(axis=0)])
        else:
            outputs.extend([
                np.zeros(descriptor_dim, np.float32),
                np.zeros(descriptor_dim, np.float32),
            ])
    return np.concatenate(outputs).astype(np.float32)


def _attach_optical_flow_features(
    ns, args, views, required, plan, samples, disease, features,
):
    signature = {
        "version": "farneback_functional_v1",
        "disease": disease,
        "views": list(views),
        "plan": dict(plan),
        "frames": int(args.num_frames),
        "size": int(args.image_size),
        "phases": int(args.motion_flow_phases),
        "flow_size": int(args.motion_flow_size),
        "split_source": str(args.split_source_dir),
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = _prognosis_feature_dir("motion")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{disease.lower()}_{digest}.pt"
    patient_ids = {
        split: [str(x["patient_id"]) for x in samples[split]]
        for split in ("train", "val", "test")
    }
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("patient_ids") == patient_ids:
            features["optical_motion"] = cached["features"]
            ns["print0"](f"Reusing optical-flow cache: {cache_path}")
            return features, cache_path
    motion = {}
    for split in ("train", "val", "test"):
        dataset = _dataset(
            ns, samples[split], views, required, plan, args, "eval"
        )
        rows = []
        for index in range(len(dataset)):
            rows.append(_patient_optical_flow_descriptor(
                dataset[index], len(views),
                int(args.motion_flow_phases), int(args.motion_flow_size),
            ))
            if (index + 1) % 100 == 0:
                ns["print0"](
                    f"Optical flow {split}: {index + 1}/{len(dataset)}"
                )
        motion[split] = torch.from_numpy(np.stack(rows)).float()
    torch.save(
        {"patient_ids": patient_ids, "features": motion}, cache_path
    )
    features["optical_motion"] = motion
    ns["print0"](
        f"Saved optical-flow features dim={motion['train'].shape[1]}: {cache_path}"
    )
    return features, cache_path


def _curve_statistics(curves):
    """curves: [K, T, C], returns per-slice [K, C*8]."""
    phases = max(curves.shape[1] - 1, 1)
    minimum = curves.amin(dim=1)
    maximum = curves.amax(dim=1)
    amplitude = maximum - minimum
    return torch.cat(
        [
            curves.mean(dim=1),
            curves.std(dim=1, unbiased=False),
            minimum,
            maximum,
            amplitude,
            amplitude / maximum.clamp_min(1e-6),
            curves.argmin(dim=1).float() / phases,
            curves.argmax(dim=1).float() / phases,
        ],
        dim=1,
    )


@torch.no_grad()
def _segmentation_functional_descriptor(
    model, item, sax_view_id, device, inference_batch_size,
):
    valid = item["clip_mask"].bool() & (
        item["clip_view_ids"].long() == int(sax_view_id)
    )
    clips = item["videos"][valid]
    if len(clips) == 0:
        return torch.zeros(145, dtype=torch.float32)
    # Dataset tensors are ImageNet-standardized; restore their [0, 1] intensity.
    clips = (clips.float() * 0.229 + 0.485).clamp(0, 1)
    k, _, t, h, w = clips.shape
    frames = clips.permute(0, 2, 1, 3, 4).reshape(k * t, 1, h, w)
    predictions = []
    for start in range(0, len(frames), int(inference_batch_size)):
        x = frames[start:start + int(inference_batch_size)].to(device)
        x = F.interpolate(
            x, size=(256, 256), mode="bilinear", align_corners=False
        )
        predictions.append(model(x).argmax(dim=1).cpu())
    mask = torch.cat(predictions).reshape(k, t, 256, 256)
    areas = torch.stack(
        [(mask == label).sum(dim=(2, 3)) for label in (1, 2, 3)],
        dim=2,
    ).float() / float(256 * 256)
    per_slice = _curve_statistics(areas)
    lv = areas[:, :, 0].clamp_min(1e-6)
    ratios = torch.stack(
        [
            (areas[:, :, 2] / lv).mean(dim=1),
            (areas[:, :, 2] / lv).std(dim=1, unbiased=False),
            (areas[:, :, 1] / lv).mean(dim=1),
            (areas[:, :, 1] / lv).std(dim=1, unbiased=False),
        ],
        dim=1,
    )
    per_slice = torch.cat([per_slice, ratios], dim=1)  # K x 28
    across_slices = torch.cat(
        [
            per_slice.mean(dim=0),
            per_slice.std(dim=0, unbiased=False),
            per_slice.amin(dim=0),
            per_slice.amax(dim=0),
        ]
    )  # 112
    global_curves = areas.sum(dim=0, keepdim=True)
    global_stats = _curve_statistics(global_curves).squeeze(0)  # 24
    phase_dispersion = torch.cat(
        [
            areas.argmin(dim=1).float().std(dim=0, unbiased=False) / max(t - 1, 1),
            areas.argmax(dim=1).float().std(dim=0, unbiased=False) / max(t - 1, 1),
        ]
    )  # 6
    foreground_reliability = torch.stack(
        [(areas[:, :, index] > 20 / (256 * 256)).float().mean() for index in range(3)]
    )
    descriptor = torch.cat(
        [across_slices, global_stats, phase_dispersion, foreground_reliability]
    ).float()
    if descriptor.numel() != 145 or not torch.isfinite(descriptor).all():
        raise RuntimeError("Invalid segmentation-derived functional descriptor")
    return descriptor


def _attach_segmentation_functional_features(
    ns, args, views, required, plan, samples, device, disease, features,
):
    from monai.networks.nets import UNet

    model_path = Path(args.cardiac_segmentation_model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Cardiac segmentation model missing: {model_path}")
    signature = {
        "version": "monai_sax_3label_functional_v1",
        "model": str(model_path),
        "model_size": model_path.stat().st_size,
        "views": list(views),
        "plan": dict(plan),
        "frames": int(args.num_frames),
        "split_source": str(args.split_source_dir),
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = _prognosis_feature_dir("segmentation")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{disease.lower()}_{digest}.pt"
    patient_ids = {
        split: [str(x["patient_id"]) for x in samples[split]]
        for split in ("train", "val", "test")
    }
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("patient_ids") == patient_ids:
            features["segmentation_functional"] = cached["features"]
            ns["print0"](f"Reusing segmentation-functional cache: {cache_path}")
            return features, cache_path
    model = UNet(
        spatial_dims=2, in_channels=1, out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2), num_res_units=2,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    sax_view_id = list(views).index("CineSAX")
    output = {}
    for split in ("train", "val", "test"):
        dataset = _dataset(
            ns, samples[split], views, required, plan, args, "eval"
        )
        rows = []
        for index in range(len(dataset)):
            rows.append(_segmentation_functional_descriptor(
                model, dataset[index], sax_view_id, device,
                int(args.segmentation_inference_batch_size),
            ))
            if (index + 1) % 100 == 0:
                ns["print0"](
                    f"Cardiac segmentation {split}: {index + 1}/{len(dataset)}"
                )
        output[split] = torch.stack(rows)
    torch.save({"patient_ids": patient_ids, "features": output}, cache_path)
    features["segmentation_functional"] = output
    ns["print0"](f"Saved segmentation-functional features: {cache_path}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, cache_path


def _meta_tensors(samples, device):
    time = torch.tensor(
        [float(item["time_to_event"]) for item in samples],
        dtype=torch.float32,
        device=device,
    )
    event = torch.tensor(
        [float(item["event"]) for item in samples],
        dtype=torch.float32,
        device=device,
    )
    return time, event


@torch.no_grad()
def _head_risk(head, features, device):
    head.eval()
    return (
        head(features.to(device))
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def fit_exact_cox(ns, features, samples, val_features, val_samples, args, device, tag):
    """Fit one head with exact Cox loss over every training patient."""
    torch.manual_seed(int(args.seed))
    head = FullRiskCoxHead(
        features.shape[1],
        args.fullrisk_head_hidden,
        args.fullrisk_head_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(args.fullrisk_head_lr),
        weight_decay=float(args.fullrisk_weight_decay),
    )
    x_train = features.to(device)
    x_val = val_features.to(device)
    time_train, event_train = _meta_tensors(samples, device)
    time_val = np.asarray([float(x["time_to_event"]) for x in val_samples])
    event_val = np.asarray([int(x["event"]) for x in val_samples])
    mask = torch.ones_like(event_train, dtype=torch.bool)
    cox = ns["CoxPHLoss"]()
    best_score, best_epoch, best_state, stale = -math.inf, 0, None, 0
    history = []
    for epoch in range(1, int(args.fullrisk_head_epochs) + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        risk = head(x_train)
        loss = cox(risk, time_train, event_train, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"{tag}: non-finite exact Cox loss at epoch={epoch}"
            )
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), float(args.fullrisk_grad_clip))
        optimizer.step()
        if epoch == 1 or epoch % int(args.fullrisk_eval_every) == 0:
            head.eval()
            with torch.no_grad():
                val_risk = head(x_val).detach().cpu().numpy()
            val_c = ns["concordance_index"](time_val, event_val, val_risk)
            history.append(
                {"epoch": epoch, "loss": float(loss.item()), "val_cindex": val_c}
            )
            if np.isfinite(val_c) and val_c > best_score + float(args.early_stop_min_delta):
                best_score, best_epoch, stale = float(val_c), epoch, 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                }
            else:
                stale += int(args.fullrisk_eval_every)
                if stale >= int(args.fullrisk_patience):
                    break
    if best_state is None:
        raise RuntimeError(f"{tag}: exact Cox head produced no valid checkpoint")
    head.load_state_dict(best_state)
    return head, history, best_epoch, best_score


def _to_device_features(features, device):
    if isinstance(features, (tuple, list)):
        return tuple(x.to(device) for x in features)
    return features.to(device)


def fit_fusion_cox(
    ns, train_features, train_samples, val_features, val_samples,
    args, device, tag, fixed_epochs=None,
):
    """Fit a cached multi-branch bottleneck/fusion model with exact Cox loss."""
    torch.manual_seed(int(args.seed))
    input_dims = [int(x.shape[1]) for x in train_features]
    is_flat_residual = (
        tag.startswith("E_residual_b")
        or tag.startswith("F_multilevel_residual_b")
        or tag.startswith("A_raw_multilevel_residual_b")
    )
    is_shared_residual = tag.startswith("G_shared_multilevel_k")
    is_risk_fusion = tag.startswith("H_risk_fusion_p")
    is_motion_residual = tag.startswith((
        "I_optical_motion_b", "J_segmentation_functional_b"
    ))
    is_segmentation_linear = tag.startswith("S_segmentation_linear")
    is_residual = (
        is_flat_residual or is_shared_residual
        or is_risk_fusion or is_motion_residual
    )
    if is_segmentation_linear:
        head = SegmentationLinearCoxHead(
            input_dims[0], float(args.segmentation_cox_dropout)
        ).to(device)
    elif is_motion_residual:
        bottleneck = int(tag.rsplit("b", 1)[1])
        head = MotionAugmentedResidualCoxHead(
            raw_dim=input_dims[0],
            diagnosis_dim=input_dims[1],
            motion_dim=input_dims[2],
            bottleneck_dim=bottleneck,
            dropout=float(args.residual_fusion_dropout),
            diagnosis_max_gate=float(args.residual_fusion_max_gate),
            motion_max_gate=float(args.motion_fusion_max_gate),
        ).to(device)
    elif is_risk_fusion:
        head = RiskLevelFusionCoxHead(
            raw_dim=input_dims[0],
            diagnosis_dim=input_dims[1],
            dropout=float(args.risk_fusion_dropout),
            max_gate=float(args.risk_fusion_max_gate),
        ).to(device)
    elif is_shared_residual:
        projection_dim = int(tag.rsplit("k", 1)[1])
        head = SharedTokenResidualCoxHead(
            raw_dim=input_dims[0],
            token_dim=int(args.multilevel_token_dim),
            projection_dim=projection_dim,
            dropout=float(args.residual_fusion_dropout),
            max_gate=float(args.residual_fusion_max_gate),
        ).to(device)
    elif is_flat_residual:
        bottleneck = int(tag.rsplit("b", 1)[1])
        head = ResidualDiagnosisCoxHead(
            raw_dim=input_dims[0],
            diagnosis_dim=input_dims[1],
            bottleneck_dim=bottleneck,
            dropout=float(args.residual_fusion_dropout),
            max_gate=float(args.residual_fusion_max_gate),
        ).to(device)
    else:
        head = AdaptiveFusionCoxHead(
            input_dims=input_dims,
            bottleneck_dim=int(args.fusion_bottleneck_dim),
            hidden=int(args.fullrisk_head_hidden),
            dropout=float(args.fullrisk_head_dropout),
        ).to(device)
    if is_segmentation_linear:
        weight_decay = float(args.segmentation_cox_weight_decay)
    elif is_risk_fusion:
        weight_decay = float(args.risk_fusion_weight_decay)
    elif is_residual:
        weight_decay = float(args.residual_fusion_weight_decay)
    else:
        weight_decay = float(args.fullrisk_weight_decay)
    learning_rate = (
        float(args.segmentation_cox_lr) if is_segmentation_linear
        else (
            float(args.risk_fusion_lr)
            if is_risk_fusion else float(args.fullrisk_head_lr)
        )
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    x_train = _to_device_features(train_features, device)
    x_val = _to_device_features(val_features, device)
    time_train, event_train = _meta_tensors(train_samples, device)
    time_val = np.asarray([float(x["time_to_event"]) for x in val_samples])
    event_val = np.asarray([int(x["event"]) for x in val_samples])
    cox, mask = ns["CoxPHLoss"](), torch.ones_like(event_train, dtype=torch.bool)
    best_score, best_epoch, best_state, stale = -math.inf, 0, None, 0
    max_epochs = (
        int(fixed_epochs)
        if fixed_epochs is not None
        else int(args.fullrisk_head_epochs)
    )
    history = []
    for epoch in range(1, max_epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        risk = head(x_train)
        loss = cox(risk, time_train, event_train, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{tag}: non-finite Cox loss at epoch={epoch}")
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), float(args.fullrisk_grad_clip))
        optimizer.step()
        if fixed_epochs is not None:
            if epoch == max_epochs:
                history.append({
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "val_cindex": np.nan,
                    "training_mode": "fixed_epoch_outer_refit",
                })
        elif epoch == 1 or epoch % int(args.fullrisk_eval_every) == 0:
            head.eval()
            with torch.no_grad():
                val_risk = head(x_val).detach().cpu().numpy()
            val_c = ns["concordance_index"](time_val, event_val, val_risk)
            history.append(
                {"epoch": epoch, "loss": float(loss.item()), "val_cindex": val_c}
            )
            if np.isfinite(val_c) and val_c > best_score + float(args.early_stop_min_delta):
                best_score, best_epoch, stale = float(val_c), epoch, 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                }
            else:
                stale += int(args.fullrisk_eval_every)
                if stale >= int(args.fullrisk_patience):
                    break
    if fixed_epochs is not None:
        return head, history, max_epochs, float("nan")
    if best_state is None:
        raise RuntimeError(f"{tag}: fusion Cox head produced no valid checkpoint")
    head.load_state_dict(best_state)
    return head, history, best_epoch, best_score


@torch.no_grad()
def _model_risk(head, features, device):
    head.eval()
    return (
        head(_to_device_features(features, device))
        .detach().float().cpu().numpy().astype(np.float32)
    )


def _standardize_feature_sets(feature_sets):
    """Train-only normalization; returns normalized train/val/test tensors."""
    output = {}
    for name, by_split in feature_sets.items():
        train = by_split["train"].float()
        mean = train.mean(dim=0, keepdim=True)
        std = train.std(dim=0, keepdim=True).clamp_min(1e-5)
        output[name] = {
            split: (tensor.float() - mean) / std
            for split, tensor in by_split.items()
        }
    return output


def _add_train_fitted_pca(feature_sets, dimensions, seed):
    """Unsupervised train-only PCA; validation/test never influence components."""
    for dim in dimensions:
        dim = int(dim)
        for source in ("raw", "multilevel"):
            train = feature_sets[source]["train"].cpu().numpy()
            pca = PCA(
                n_components=dim,
                svd_solver="randomized",
                random_state=int(seed),
            )
            transformed = {
                split: torch.from_numpy(
                    pca.fit_transform(tensor.cpu().numpy())
                    if split == "train"
                    else pca.transform(tensor.cpu().numpy())
                ).float()
                for split, tensor in feature_sets[source].items()
            }
            # Normalize PCA scores using training statistics only.
            mean = transformed["train"].mean(dim=0, keepdim=True)
            std = transformed["train"].std(dim=0, keepdim=True).clamp_min(1e-5)
            feature_sets[f"{source}_pca{dim}"] = {
                split: (tensor - mean) / std
                for split, tensor in transformed.items()
            }
    return feature_sets


def _freeze_for_adaptation(student, args):
    for parameter in student.parameters():
        parameter.requires_grad = False
    # The teacher head remains fixed: only the prognosis encoder copy adapts.
    student.encoder.view_embeddings.requires_grad = True
    for module in (student.encoder.slice_pool, student.encoder.view_pool):
        for parameter in module.parameters():
            parameter.requires_grad = True
    stages = list(student.encoder.backbone.features.children())
    n_last = max(0, int(args.fullrisk_unfreeze_last_stages))
    for stage in stages[-n_last:] if n_last else []:
        for parameter in stage.parameters():
            parameter.requires_grad = True


def adapt_encoder(ns, encoder, teacher_head, dataset, teacher_risk, args, device, tag):
    """Adapt only the prognosis encoder to disease-specific teacher ordering."""
    frozen_head = FullRiskCoxHead(
        encoder.feature_dim,
        args.fullrisk_head_hidden,
        args.fullrisk_head_dropout,
    ).to(device)
    frozen_head.load_state_dict(teacher_head.state_dict())
    student = EncoderRiskStudent(encoder, frozen_head).to(device)
    _freeze_for_adaptation(student, args)
    forward = student
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        forward = nn.DataParallel(student)
    parameters = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.fullrisk_encoder_lr),
        weight_decay=float(args.fullrisk_weight_decay),
    )
    rng = random.Random(int(args.seed))
    teacher_risk = np.asarray(teacher_risk, dtype=np.float32)
    teacher_mean = float(np.mean(teacher_risk))
    teacher_std = max(float(np.std(teacher_risk)), 1e-4)
    indices = list(range(len(dataset)))
    batch_size = max(1, int(args.fullrisk_adapt_batch_size))
    history = []
    for epoch in range(1, int(args.fullrisk_adapt_epochs) + 1):
        rng.shuffle(indices)
        forward.train()
        # Keep the fixed risk head deterministic.
        student.risk_head.eval()
        running, seen = 0.0, 0
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = _device_batch([dataset[index] for index in batch_indices], device)
            target = torch.tensor(
                (teacher_risk[batch_indices] - teacher_mean) / teacher_std,
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            # Keep adaptation in FP32. Per-batch standardization made the
            # derivative explode when two predictions were nearly tied; it
            # was the source of NaNs in the first full-risk screening run.
            with torch.amp.autocast(device_type="cuda", enabled=False):
                prediction = forward(batch).float()
                prediction = (prediction - teacher_mean) / teacher_std
                loss = F.smooth_l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{tag}: non-finite prognosis distillation loss "
                    f"at epoch={epoch}, batch_start={start}"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, float(args.fullrisk_grad_clip))
            optimizer.step()
            if not all(
                torch.isfinite(parameter).all() for parameter in parameters
            ):
                raise FloatingPointError(
                    f"{tag}: encoder parameters became non-finite "
                    f"at epoch={epoch}, batch_start={start}"
                )
            running += float(loss.detach().cpu()) * len(batch_indices)
            seen += len(batch_indices)
        history.append({"epoch": epoch, "distill_loss": running / max(seen, 1)})
        ns["print0"](
            f"{tag}: prognosis-teacher adaptation epoch={epoch:02d} "
            f"loss={history[-1]['distill_loss']:.4f}"
        )
    return encoder, history


def _evaluate(ns, samples, risk, threshold, horizons):
    meta = pd.DataFrame(
        {
            "patient_id": [str(x["patient_id"]) for x in samples],
            "time_to_event": [float(x["time_to_event"]) for x in samples],
            "event": [int(x["event"]) for x in samples],
        }
    )
    return ns["evaluate_survival_array"](meta, risk, threshold, horizons), meta


def run_variant(ns, variant, args, views, required, plan, samples, device, out_dir):
    diagnosis_init = variant == "diagnosis_to_prognosis"
    encoder = _new_encoder(ns, views, plan, args, diagnosis_init, device)
    datasets_eval = {
        split: _dataset(ns, samples[split], views, required, plan, args, "eval")
        for split in ("train", "val", "test")
    }
    initial = {
        split: extract_features(encoder, datasets_eval[split], device, args)
        for split in ("train", "val", "test")
    }
    teacher, teacher_history, teacher_epoch, teacher_val = fit_exact_cox(
        ns, initial["train"], samples["train"],
        initial["val"], samples["val"], args, device, f"{variant}/teacher",
    )
    teacher_risk = _head_risk(teacher, initial["train"], device)
    train_augmented = _dataset(
        ns, samples["train"], views, required, plan, args, "train"
    )
    encoder, adapt_history = adapt_encoder(
        ns, encoder, teacher, train_augmented, teacher_risk,
        args, device, f"{variant}/encoder",
    )
    adapted = {
        split: extract_features(encoder, datasets_eval[split], device, args)
        for split in ("train", "val", "test")
    }
    final_head, final_history, final_epoch, final_val = fit_exact_cox(
        ns, adapted["train"], samples["train"],
        adapted["val"], samples["val"], args, device, f"{variant}/final",
    )
    risks = {
        split: _head_risk(final_head, adapted[split], device)
        for split in ("train", "val", "test")
    }
    threshold = float(np.median(risks["train"]))
    horizons = ns["parse_horizons_days"](args.eval_horizons_days)
    row = {
        "variant": variant,
        "seed": int(args.seed),
        "teacher_best_epoch": teacher_epoch,
        "teacher_val_cindex": teacher_val,
        "final_best_epoch": final_epoch,
        "final_val_cindex": final_val,
    }
    for split in ("train", "val", "test"):
        metrics, meta = _evaluate(
            ns, samples[split], risks[split], threshold, horizons
        )
        row.update({f"{split}_{key}": value for key, value in metrics.items()})
        meta["risk"] = risks[split]
        meta.to_csv(
            out_dir / f"{variant}_{split}_predictions.csv",
            index=False, encoding="utf-8-sig",
        )
    torch.save(
        {
            "version": VERSION,
            "variant": variant,
            "encoder": {
                key: value.detach().cpu()
                for key, value in encoder.state_dict().items()
            },
            "cox_head": {
                key: value.detach().cpu()
                for key, value in final_head.state_dict().items()
            },
            "metrics": row,
        },
        out_dir / f"{variant}_best.pt",
    )
    pd.DataFrame(teacher_history).to_csv(
        out_dir / f"{variant}_teacher_cox_history.csv", index=False
    )
    pd.DataFrame(adapt_history).to_csv(
        out_dir / f"{variant}_encoder_adapt_history.csv", index=False
    )
    pd.DataFrame(final_history).to_csv(
        out_dir / f"{variant}_final_cox_history.csv", index=False
    )
    del encoder, teacher, final_head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _feature_cache_path(args, disease, views, plan):
    signature = {
        "version": VERSION,
        "raw_encoder_feature_policy": RAW_ENCODER_FEATURE_POLICY,
        "disease": disease,
        "views": list(views),
        "plan": dict(plan),
        "frames": int(args.num_frames),
        "size": int(args.image_size),
        "weights": str(args.weights_path),
        "diagnosis_checkpoint": str(args.classification_ckpt),
        "aggregation": str(args.diagnosis_encoder_agg),
        "split_source": str(args.split_source_dir),
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    directory = _prognosis_feature_dir("patient_level")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{disease.lower()}_{digest}.pt"


def _extract_abcd_features(
    ns, args, views, required, plan, samples, device, disease,
):
    cache_path = _feature_cache_path(args, disease, views, plan)
    patient_ids = {
        split: [str(x["patient_id"]) for x in samples[split]]
        for split in ("train", "val", "test")
    }
    if bool(args.fusion_reuse_feature_cache) and cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("patient_ids") == patient_ids:
            ns["print0"](f"Reusing prognosis feature cache: {cache_path}")
            return cached["features"], cache_path
        ns["print0"]("Feature cache patient order changed; extracting again.")

    datasets = {
        split: _dataset(ns, samples[split], views, required, plan, args, "eval")
        for split in ("train", "val", "test")
    }
    raw, motion = {}, {}
    raw_encoder = _new_encoder(ns, views, plan, args, False, device)
    for split in ("train", "val", "test"):
        raw[split], motion[split] = extract_features_and_motion(
            raw_encoder, datasets[split], device, args, len(views)
        )
    del raw_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    diagnosis = {}
    diagnosis_encoder = _new_encoder(ns, views, plan, args, True, device)
    for split in ("train", "val", "test"):
        diagnosis[split] = extract_features(
            diagnosis_encoder, datasets[split], device, args
        )
    del diagnosis_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    features = {"raw": raw, "diagnosis": diagnosis, "motion": motion}
    torch.save(
        {
            "version": VERSION,
            "patient_ids": patient_ids,
            "features": features,
        },
        cache_path,
    )
    ns["print0"](f"Saved reusable prognosis feature cache: {cache_path}")
    return features, cache_path


def _attach_frozen_multilevel_features(
    ns, args, views, required, plan, samples, device, disease, features,
):
    persistent_root = (
        _prognosis_feature_dir("multilevel") / disease.lower()
    )
    persistent_root.mkdir(parents=True, exist_ok=True)
    multi, cache_dir = ns["extract_or_load_multilevel"](
        args, samples, views, required, plan, persistent_root, device
    )
    rich = {}
    for split in ("train", "val", "test"):
        expected = [str(x["patient_id"]) for x in samples[split]]
        observed = multi[split]["meta"]["patient_id"].astype(str).tolist()
        if observed != expected:
            raise RuntimeError(
                f"{split}: frozen multilevel feature order does not match split CSV"
            )
        rich[split] = _build_frozen_multilevel_vector(
            multi[split], len(views)
        )
    features["multilevel"] = rich
    ns["print0"](
        f"Frozen multilevel prognosis vector dim={rich['train'].shape[1]} | "
        f"cache={cache_dir}"
    )
    return features, cache_dir


def _attach_raw_multilevel_features(
    ns, args, views, required, plan, samples, device, disease, features,
):
    """Extract the structure-matched multilevel vector without diagnosis supervision."""
    persistent_root = (
        _prognosis_feature_dir("raw_multilevel") / disease.lower()
    )
    persistent_root.mkdir(parents=True, exist_ok=True)
    raw_args = copy.copy(args)
    raw_args.classification_ckpt = ""
    multi, cache_dir = ns["extract_or_load_multilevel"](
        raw_args, samples, views, required, plan, persistent_root, device
    )
    rich = {}
    for split in ("train", "val", "test"):
        expected = [str(x["patient_id"]) for x in samples[split]]
        observed = multi[split]["meta"]["patient_id"].astype(str).tolist()
        if observed != expected:
            raise RuntimeError(
                f"{split}: raw multilevel feature order does not match split CSV"
            )
        rich[split] = _build_frozen_multilevel_vector(
            multi[split], len(views)
        )
    features["raw_multilevel"] = rich
    ns["print0"](
        f"Raw multilevel matched-control vector dim={rich['train'].shape[1]} | "
        f"cache={cache_dir}"
    )
    return features, cache_dir


def _variant_features(features, variant, split):
    if variant == "A_direct":
        return (features["raw"][split],)
    if variant == "B_diagnosis":
        return (features["diagnosis"][split],)
    if variant == "C_raw_diagnosis_fusion":
        return (features["raw"][split], features["diagnosis"][split])
    if variant == "D_raw_diagnosis_motion_fusion":
        return (
            features["raw"][split],
            features["diagnosis"][split],
            features["motion"][split],
        )
    if variant.startswith("E_residual_b"):
        return (features["raw"][split], features["diagnosis"][split])
    if variant.startswith("F_multilevel_residual_b"):
        return (features["raw"][split], features["multilevel"][split])
    if variant.startswith("A_raw_multilevel_residual_b"):
        return (
            features["raw"][split], features["raw_multilevel"][split]
        )
    if variant.startswith("G_shared_multilevel_k"):
        return (features["raw"][split], features["multilevel"][split])
    if variant.startswith("H_risk_fusion_p"):
        dim = int(variant.rsplit("p", 1)[1])
        return (
            features[f"raw_pca{dim}"][split],
            features[f"multilevel_pca{dim}"][split],
        )
    if variant.startswith("I_optical_motion_b"):
        return (
            features["raw"][split],
            features["multilevel"][split],
            features["optical_motion"][split],
        )
    if variant.startswith("J_segmentation_functional_b"):
        return (
            features["raw"][split],
            features["multilevel"][split],
            features["segmentation_functional"][split],
        )
    if variant == "S_segmentation_linear":
        return (features["segmentation_functional"][split],)
    if variant in {"K_oof_late_fusion", "L_fold_teacher_ensemble"}:
        return (
            features["raw"][split],
            features["multilevel"][split],
            features["segmentation_functional"][split],
        )
    if variant == "K_raw_matched_oof_late_fusion":
        return (
            features["raw"][split],
            features["raw_multilevel"][split],
            features["segmentation_functional"][split],
        )
    raise ValueError(f"Unknown A/B/C/D variant: {variant}")


def _subset_features(features, indices):
    return tuple(x[indices] for x in features)


def _subset_samples(samples, indices):
    return [samples[int(index)] for index in indices]


def _normalize_features_from_indices(features, fit_indices):
    """Apply preprocessing fitted only on the specified training indices."""
    normalized = []
    for tensor in features:
        fit = tensor[fit_indices].float()
        mean = fit.mean(dim=0, keepdim=True)
        std = fit.std(dim=0, keepdim=True).clamp_min(1e-5)
        normalized.append((tensor.float() - mean) / std)
    return tuple(normalized)


def _cindex_from_samples(ns, samples, risk):
    return ns["concordance_index"](
        np.asarray([float(x["time_to_event"]) for x in samples]),
        np.asarray([int(x["event"]) for x in samples]),
        np.asarray(risk),
    )


def fit_nested_single_expert_oof(
    ns, features, train_samples, args, device, variant, root,
):
    """Fully nested OOF risk for one fixed prognosis expert."""
    events = np.asarray([int(x["event"]) for x in train_samples])
    splitter = StratifiedKFold(
        n_splits=int(args.late_fusion_folds),
        shuffle=True,
        random_state=int(args.seed),
    )
    oof_risk = np.full(len(events), np.nan, dtype=np.float32)
    outer_fold_assignment = np.full(len(events), -1, dtype=np.int16)
    fold_rows = []
    for fold, (outer_fit_idx, outer_hold_idx) in enumerate(
        splitter.split(np.zeros(len(events)), events), start=1
    ):
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=int(args.seed) + 2000 + fold,
        )
        inner_fit_rel, inner_val_rel = next(
            inner_splitter.split(
                np.zeros(len(outer_fit_idx)), events[outer_fit_idx]
            )
        )
        inner_fit_idx = outer_fit_idx[inner_fit_rel]
        inner_val_idx = outer_fit_idx[inner_val_rel]
        inner_features = _normalize_features_from_indices(
            features, inner_fit_idx
        )
        _, _, selected_epoch, inner_val_cindex = fit_fusion_cox(
            ns,
            _subset_features(inner_features, inner_fit_idx),
            _subset_samples(train_samples, inner_fit_idx),
            _subset_features(inner_features, inner_val_idx),
            _subset_samples(train_samples, inner_val_idx),
            args, device, variant,
        )
        outer_features = _normalize_features_from_indices(
            features, outer_fit_idx
        )
        outer_head, _, _, _ = fit_fusion_cox(
            ns,
            _subset_features(outer_features, outer_fit_idx),
            _subset_samples(train_samples, outer_fit_idx),
            _subset_features(outer_features, outer_fit_idx),
            _subset_samples(train_samples, outer_fit_idx),
            args, device, variant,
            fixed_epochs=selected_epoch,
        )
        fit_risk = _model_risk(
            outer_head,
            _subset_features(outer_features, outer_fit_idx), device,
        )
        hold_risk = _model_risk(
            outer_head,
            _subset_features(outer_features, outer_hold_idx), device,
        )
        risk_mean = float(fit_risk.mean())
        risk_std = max(float(fit_risk.std()), 1e-5)
        oof_risk[outer_hold_idx] = (hold_risk - risk_mean) / risk_std
        outer_fold_assignment[outer_hold_idx] = fold
        fold_rows.append({
            "fold": fold,
            "outer_train_n": len(outer_fit_idx),
            "inner_fit_n": len(inner_fit_idx),
            "inner_val_n": len(inner_val_idx),
            "outer_holdout_n": len(outer_hold_idx),
            "inner_selected_epoch": selected_epoch,
            "outer_refit_epochs": selected_epoch,
            "inner_val_cindex": inner_val_cindex,
            "outer_oof_cindex": _cindex_from_samples(
                ns, _subset_samples(train_samples, outer_hold_idx),
                hold_risk,
            ),
        })
    if not np.isfinite(oof_risk).all() or np.any(
        outer_fold_assignment < 1
    ):
        raise RuntimeError(f"{variant}: incomplete fully nested OOF risk")
    pd.DataFrame(fold_rows).to_csv(
        root / f"{variant}_nested_oof_fold_metrics.csv", index=False
    )
    pd.DataFrame({
        "patient_id": [x["patient_id"] for x in train_samples],
        "time_to_event": [x["time_to_event"] for x in train_samples],
        "event": [x["event"] for x in train_samples],
        "outer_fold": outer_fold_assignment,
        "nested_oof_risk": oof_risk,
    }).to_csv(
        root / f"{variant}_nested_oof_predictions.csv", index=False
    )
    return oof_risk, _cindex_from_samples(ns, train_samples, oof_risk)


def fit_oof_late_fusion(
    ns, features, samples, args, device, root,
    phenotype_feature_key="multilevel", output_prefix="K_oof",
    model_keys=("K_oof_late_fusion", "L_fold_teacher_ensemble"),
):
    """OOF-select the risk weight, then fit both experts on the full train set."""
    phenotype = (
        features["raw"]["train"],
        features[phenotype_feature_key]["train"],
    )
    functional = (features["segmentation_functional"]["train"],)
    events = np.asarray([int(x["event"]) for x in samples["train"]])
    splitter = StratifiedKFold(
        n_splits=int(args.late_fusion_folds),
        shuffle=True,
        random_state=int(args.seed),
    )
    oof_phenotype = np.full(len(events), np.nan, dtype=np.float32)
    oof_function = np.full(len(events), np.nan, dtype=np.float32)
    nested_oof_fused = np.full(len(events), np.nan, dtype=np.float32)
    nested_oof_alpha = np.full(len(events), np.nan, dtype=np.float32)
    outer_fold_assignment = np.full(len(events), -1, dtype=np.int16)
    fold_rows = []
    phenotype_heads, function_heads = [], []
    phenotype_means, phenotype_stds = [], []
    function_means, function_stds = [], []
    for fold, (fit_idx, hold_idx) in enumerate(
        splitter.split(np.zeros(len(events)), events), start=1
    ):
        # The outer holdout is reserved exclusively for OOF prediction.
        # Early stopping is performed on a validation subset drawn only from
        # the outer training fold, preventing optimistic OOF estimates.
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=int(args.seed) + 1000 + fold,
        )
        inner_fit_rel, inner_val_rel = next(
            inner_splitter.split(
                np.zeros(len(fit_idx)), events[fit_idx]
            )
        )
        inner_fit_idx = fit_idx[inner_fit_rel]
        inner_val_idx = fit_idx[inner_val_rel]
        fold_phenotype = _normalize_features_from_indices(
            phenotype, inner_fit_idx
        )
        fold_functional = _normalize_features_from_indices(
            functional, inner_fit_idx
        )
        phenotype_head, _, phenotype_epoch, phenotype_inner_c = fit_fusion_cox(
            ns,
            _subset_features(fold_phenotype, inner_fit_idx),
            _subset_samples(samples["train"], inner_fit_idx),
            _subset_features(fold_phenotype, inner_val_idx),
            _subset_samples(samples["train"], inner_val_idx),
            args, device, "F_multilevel_residual_b8",
        )
        function_head, _, function_epoch, function_inner_c = fit_fusion_cox(
            ns,
            _subset_features(fold_functional, inner_fit_idx),
            _subset_samples(samples["train"], inner_fit_idx),
            _subset_features(fold_functional, inner_val_idx),
            _subset_samples(samples["train"], inner_val_idx),
            args, device, "S_segmentation_linear",
        )
        fit_phenotype_risk = _model_risk(
            phenotype_head,
            _subset_features(fold_phenotype, inner_fit_idx), device
        )
        fit_function_risk = _model_risk(
            function_head,
            _subset_features(fold_functional, inner_fit_idx), device
        )
        phenotype_mean = float(fit_phenotype_risk.mean())
        phenotype_std = max(float(fit_phenotype_risk.std()), 1e-5)
        function_mean = float(fit_function_risk.mean())
        function_std = max(float(fit_function_risk.std()), 1e-5)
        inner_val_phenotype_risk = _model_risk(
            phenotype_head,
            _subset_features(fold_phenotype, inner_val_idx), device,
        )
        inner_val_function_risk = _model_risk(
            function_head,
            _subset_features(fold_functional, inner_val_idx), device,
        )
        inner_val_zp = (
            inner_val_phenotype_risk - phenotype_mean
        ) / phenotype_std
        inner_val_zf = (
            inner_val_function_risk - function_mean
        ) / function_std
        inner_alpha_candidates = []
        for candidate_alpha in np.linspace(
            float(args.late_fusion_min_phenotype_weight),
            float(args.late_fusion_max_phenotype_weight),
            int(args.late_fusion_weight_steps),
        ):
            candidate_score = _cindex_from_samples(
                ns,
                _subset_samples(samples["train"], inner_val_idx),
                float(candidate_alpha) * inner_val_zp
                + (1.0 - float(candidate_alpha)) * inner_val_zf,
            )
            inner_alpha_candidates.append(
                (float(candidate_score), float(candidate_alpha))
            )
        inner_alpha_score, fold_alpha = max(inner_alpha_candidates)

        # Reinitialize and refit on the complete outer-training fold using the
        # epochs selected solely inside that fold. The outer holdout remains
        # untouched until this refit is complete.
        outer_phenotype = _normalize_features_from_indices(
            phenotype, fit_idx
        )
        outer_functional = _normalize_features_from_indices(
            functional, fit_idx
        )
        phenotype_head, _, _, _ = fit_fusion_cox(
            ns,
            _subset_features(outer_phenotype, fit_idx),
            _subset_samples(samples["train"], fit_idx),
            _subset_features(outer_phenotype, fit_idx),
            _subset_samples(samples["train"], fit_idx),
            args, device, "F_multilevel_residual_b8",
            fixed_epochs=phenotype_epoch,
        )
        function_head, _, _, _ = fit_fusion_cox(
            ns,
            _subset_features(outer_functional, fit_idx),
            _subset_samples(samples["train"], fit_idx),
            _subset_features(outer_functional, fit_idx),
            _subset_samples(samples["train"], fit_idx),
            args, device, "S_segmentation_linear",
            fixed_epochs=function_epoch,
        )
        fit_phenotype_risk = _model_risk(
            phenotype_head,
            _subset_features(outer_phenotype, fit_idx), device,
        )
        fit_function_risk = _model_risk(
            function_head,
            _subset_features(outer_functional, fit_idx), device,
        )
        phenotype_mean = float(fit_phenotype_risk.mean())
        phenotype_std = max(float(fit_phenotype_risk.std()), 1e-5)
        function_mean = float(fit_function_risk.mean())
        function_std = max(float(fit_function_risk.std()), 1e-5)
        hold_phenotype_risk = _model_risk(
            phenotype_head,
            _subset_features(outer_phenotype, hold_idx), device,
        )
        hold_function_risk = _model_risk(
            function_head,
            _subset_features(outer_functional, hold_idx), device,
        )
        oof_phenotype[hold_idx] = (
            hold_phenotype_risk - phenotype_mean
        ) / phenotype_std
        oof_function[hold_idx] = (
            hold_function_risk - function_mean
        ) / function_std
        nested_oof_fused[hold_idx] = (
            fold_alpha * oof_phenotype[hold_idx]
            + (1.0 - fold_alpha) * oof_function[hold_idx]
        )
        nested_oof_alpha[hold_idx] = fold_alpha
        outer_fold_assignment[hold_idx] = fold
        phenotype_outer_c = _cindex_from_samples(
            ns, _subset_samples(samples["train"], hold_idx),
            hold_phenotype_risk,
        )
        function_outer_c = _cindex_from_samples(
            ns, _subset_samples(samples["train"], hold_idx),
            hold_function_risk,
        )
        fold_rows.append({
            "fold": fold,
            "outer_train_n": len(fit_idx),
            "inner_fit_n": len(inner_fit_idx),
            "inner_val_n": len(inner_val_idx),
            "outer_holdout_n": len(hold_idx),
            "phenotype_best_epoch": phenotype_epoch,
            "phenotype_outer_refit_epochs": phenotype_epoch,
            "phenotype_inner_val_cindex": phenotype_inner_c,
            "phenotype_outer_oof_cindex": phenotype_outer_c,
            "function_best_epoch": function_epoch,
            "function_outer_refit_epochs": function_epoch,
            "function_inner_val_cindex": function_inner_c,
            "function_outer_oof_cindex": function_outer_c,
            "inner_selected_alpha": fold_alpha,
            "inner_alpha_selection_cindex": inner_alpha_score,
        })
        phenotype_heads.append(phenotype_head)
        function_heads.append(function_head)
        phenotype_means.append(phenotype_mean)
        phenotype_stds.append(phenotype_std)
        function_means.append(function_mean)
        function_stds.append(function_std)
    if (
        not np.isfinite(oof_phenotype).all()
        or not np.isfinite(oof_function).all()
        or not np.isfinite(nested_oof_fused).all()
        or not np.isfinite(nested_oof_alpha).all()
        or np.any(outer_fold_assignment < 1)
    ):
        raise RuntimeError("Incomplete OOF risk predictions")
    zp = oof_phenotype
    zf = oof_function
    candidates = []
    for alpha in np.linspace(
        float(args.late_fusion_min_phenotype_weight),
        float(args.late_fusion_max_phenotype_weight),
        int(args.late_fusion_weight_steps),
    ):
        score = _cindex_from_samples(
            ns, samples["train"], float(alpha) * zp + (1.0 - float(alpha)) * zf
        )
        candidates.append((float(score), float(alpha)))
    apparent_global_alpha_oof_cindex, alpha = max(candidates)
    nested_oof_cindex = _cindex_from_samples(
        ns, samples["train"], nested_oof_fused
    )
    pd.DataFrame(fold_rows).to_csv(
        root / f"{output_prefix}_fold_metrics.csv", index=False
    )
    pd.DataFrame(
        {"patient_id": [x["patient_id"] for x in samples["train"]],
         "time_to_event": [x["time_to_event"] for x in samples["train"]],
         "event": [x["event"] for x in samples["train"]],
         "outer_fold": outer_fold_assignment,
         "phenotype_oof_risk": oof_phenotype,
         "function_oof_risk": oof_function,
         "nested_fused_oof_risk": nested_oof_fused,
         "nested_fold_alpha": nested_oof_alpha}
    ).to_csv(root / f"{output_prefix}_predictions.csv", index=False)

    phenotype_head, phenotype_history, phenotype_epoch, _ = fit_fusion_cox(
        ns, phenotype, samples["train"],
        (features["raw"]["val"], features[phenotype_feature_key]["val"]), samples["val"],
        args, device, "F_multilevel_residual_b8",
    )
    function_head, function_history, function_epoch, _ = fit_fusion_cox(
        ns, functional, samples["train"],
        (features["segmentation_functional"]["val"],), samples["val"],
        args, device, "S_segmentation_linear",
    )
    # Full-training risk moments map both experts to the OOF fusion scale.
    train_p = _model_risk(phenotype_head, phenotype, device)
    train_f = _model_risk(function_head, functional, device)
    model = LateRiskFusionModel(
        phenotype_head, function_head, alpha,
        float(train_p.mean()), max(float(train_p.std()), 1e-5),
        float(train_f.mean()), max(float(train_f.std()), 1e-5),
    ).to(device)
    pd.DataFrame(phenotype_history).to_csv(
        root / "K_full_phenotype_history.csv", index=False
    )
    pd.DataFrame(function_history).to_csv(
        root / "K_full_function_history.csv", index=False
    )
    ensemble = CrossFoldLateFusionModel(
        phenotype_heads, function_heads, alpha,
        phenotype_means, phenotype_stds, function_means, function_stds,
    ).to(device)
    models = {}
    if "K_oof_late_fusion" in model_keys:
        models["K_oof_late_fusion"] = model
    if "L_fold_teacher_ensemble" in model_keys:
        models["L_fold_teacher_ensemble"] = ensemble
    if "K_raw_matched_oof_late_fusion" in model_keys:
        models["K_raw_matched_oof_late_fusion"] = model
    return models, {
        "best_epoch": f"F{phenotype_epoch}/S{function_epoch}",
        "oof_cindex": nested_oof_cindex,
        "nested_oof_cindex": nested_oof_cindex,
        "global_alpha_apparent_oof_cindex": apparent_global_alpha_oof_cindex,
        "oof_phenotype_weight": alpha,
    }


def run(ns):
    args = ns["parse_args"]()
    visible = str(args.visible_gpus or "").strip()
    if visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    disease = "HCM" if str(args.disease_mode).lower().startswith("hcm") else "DCM"
    views, required, plan, split_result = ns["collect_split_samples_for_args"](args)
    samples = {key: list(split_result[key]) for key in ("train", "val", "test")}
    stamp = ns["now_string"]()
    root = ns["ensure_dir"](
        Path.cwd()
        / f"20260729_{disease.lower()}_diagnosis_to_prognosis_abcd_{stamp}"
        / str(args.experiment_name)
    )
    with open(root / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    variants = [x.strip() for x in str(args.fullrisk_variants).split(",") if x.strip()]
    allowed = {
        "A_direct",
        "B_diagnosis",
        "C_raw_diagnosis_fusion",
        "D_raw_diagnosis_motion_fusion",
        "E_residual_b8",
        "E_residual_b16",
        "F_multilevel_residual_b4",
        "F_multilevel_residual_b8",
        "A_raw_multilevel_residual_b8",
        "G_shared_multilevel_k4",
        "G_shared_multilevel_k8",
        "H_risk_fusion_p16",
        "H_risk_fusion_p32",
        "I_optical_motion_b8",
        "J_segmentation_functional_b8",
        "S_segmentation_linear",
        "K_oof_late_fusion",
        "L_fold_teacher_ensemble",
        "K_raw_matched_oof_late_fusion",
    }
    if set(variants) - allowed:
        raise ValueError(f"Unsupported fullrisk variants: {variants}")
    ns["print0"](
        f"{disease} cached A/B/C/D screening | variants={variants} | "
        f"seed={args.seed} | GPUs={visible}"
    )
    features, cache_path = _extract_abcd_features(
        ns, args, views, required, plan, samples, device, disease
    )
    multilevel_cache = None
    if any(
        x.startswith((
            "F_multilevel_residual_b",
            "G_shared_multilevel_k",
            "H_risk_fusion_p",
            "I_optical_motion_b",
            "J_segmentation_functional_b",
            "K_oof_late_fusion",
            "L_fold_teacher_ensemble",
        ))
        for x in variants
    ):
        features, multilevel_cache = _attach_frozen_multilevel_features(
            ns, args, views, required, plan, samples, device, disease, features
        )
    raw_multilevel_cache = None
    if any(
        x.startswith("A_raw_multilevel_residual_b")
        or x == "K_raw_matched_oof_late_fusion"
        for x in variants
    ):
        features, raw_multilevel_cache = _attach_raw_multilevel_features(
            ns, args, views, required, plan, samples, device, disease, features
        )
    motion_cache = None
    if any(x.startswith("I_optical_motion_b") for x in variants):
        features, motion_cache = _attach_optical_flow_features(
            ns, args, views, required, plan, samples, disease, features
        )
    segmentation_cache = None
    if any(
        x.startswith("J_segmentation_functional_b")
        or x in {
            "S_segmentation_linear",
            "K_oof_late_fusion",
            "L_fold_teacher_ensemble",
            "K_raw_matched_oof_late_fusion",
        }
        for x in variants
    ):
        features, segmentation_cache = _attach_segmentation_functional_features(
            ns, args, views, required, plan, samples, device, disease, features
        )
    features = _standardize_feature_sets(features)
    pca_dimensions = sorted({
        int(x.rsplit("p", 1)[1])
        for x in variants if x.startswith("H_risk_fusion_p")
    })
    if pca_dimensions:
        features = _add_train_fitted_pca(
            features, pca_dimensions, int(args.seed)
        )
        ns["print0"](
            f"Train-only PCA risk features prepared: {pca_dimensions}"
        )
    rows, fitted = [], {}
    horizons = ns["parse_horizons_days"](args.eval_horizons_days)
    fusion_variants = {
        "K_oof_late_fusion", "L_fold_teacher_ensemble",
        "K_raw_matched_oof_late_fusion",
    }
    for variant in [x for x in variants if x not in fusion_variants]:
        train_x = _variant_features(features, variant, "train")
        val_x = _variant_features(features, variant, "val")
        _, nested_oof_cindex = fit_nested_single_expert_oof(
            ns, train_x, samples["train"], args, device, variant, root
        )
        head, history, best_epoch, best_val = fit_fusion_cox(
            ns, train_x, samples["train"], val_x, samples["val"],
            args, device, variant,
        )
        row = {
            "variant": variant,
            "seed": int(args.seed),
            "best_epoch": best_epoch,
            "val_cindex_selection": best_val,
            "nested_oof_cindex": nested_oof_cindex,
            "oof_cindex": nested_oof_cindex,
            "test_cindex": np.nan,
        }
        train_risk = _model_risk(head, train_x, device)
        val_risk = _model_risk(head, val_x, device)
        threshold = float(np.median(train_risk))
        for split, risk in (("train", train_risk), ("val", val_risk)):
            metrics, meta = _evaluate(
                ns, samples[split], risk, threshold, horizons
            )
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
            meta["risk"] = risk
            meta.to_csv(root / f"{variant}_{split}_predictions.csv", index=False)
        fitted[variant] = (head, threshold)
        rows.append(row)
        pd.DataFrame(history).to_csv(
            root / f"{variant}_cox_history.csv", index=False
        )
    requested_fusions = [x for x in variants if x in fusion_variants]
    if requested_fusions:
        fusion_models, oof_info_by_variant = {}, {}
        standard_fusions = [
            x for x in requested_fusions
            if x in {"K_oof_late_fusion", "L_fold_teacher_ensemble"}
        ]
        if standard_fusions:
            models, info = fit_oof_late_fusion(
                ns, features, samples, args, device, root,
                model_keys=tuple(standard_fusions),
            )
            fusion_models.update(models)
            oof_info_by_variant.update({x: info for x in standard_fusions})
        if "K_raw_matched_oof_late_fusion" in requested_fusions:
            models, info = fit_oof_late_fusion(
                ns, features, samples, args, device, root,
                phenotype_feature_key="raw_multilevel",
                output_prefix="K_raw_matched_oof",
                model_keys=("K_raw_matched_oof_late_fusion",),
            )
            fusion_models.update(models)
            oof_info_by_variant["K_raw_matched_oof_late_fusion"] = info
        for variant in requested_fusions:
            model = fusion_models[variant]
            oof_info = oof_info_by_variant[variant]
            train_x = _variant_features(features, variant, "train")
            val_x = _variant_features(features, variant, "val")
            train_risk = _model_risk(model, train_x, device)
            val_risk = _model_risk(model, val_x, device)
            threshold = float(np.median(train_risk))
            row = {
                "variant": variant,
                "seed": int(args.seed),
                "best_epoch": (
                    oof_info["best_epoch"]
                    if variant != "L_fold_teacher_ensemble"
                    else "5-fold teachers"
                ),
                "val_cindex_selection": _cindex_from_samples(
                    ns, samples["val"], val_risk
                ),
                "test_cindex": np.nan,
                "oof_cindex": oof_info["oof_cindex"],
                "nested_oof_cindex": oof_info["nested_oof_cindex"],
                "global_alpha_apparent_oof_cindex": (
                    oof_info["global_alpha_apparent_oof_cindex"]
                ),
                "oof_phenotype_weight": oof_info["oof_phenotype_weight"],
            }
            for split, risk in (("train", train_risk), ("val", val_risk)):
                metrics, meta = _evaluate(
                    ns, samples[split], risk, threshold, horizons
                )
                row.update({
                    f"{split}_{key}": value for key, value in metrics.items()
                })
                meta["risk"] = risk
                meta.to_csv(
                    root / f"{variant}_{split}_predictions.csv", index=False
                )
            fitted[variant] = (model, threshold)
            rows.append(row)

    summary = pd.DataFrame(rows)
    best_index = summary["val_cindex_selection"].astype(float).idxmax()
    selected = str(summary.loc[best_index, "variant"])
    summary["selected_on_validation"] = summary["variant"].eq(selected)
    selected_head, threshold = fitted[selected]
    evaluate_test = bool(getattr(args, "evaluate_test_after_selection", True))
    if evaluate_test:
        test_x = _variant_features(features, selected, "test")
        test_risk = _model_risk(selected_head, test_x, device)
        test_metrics, test_meta = _evaluate(
            ns, samples["test"], test_risk, threshold, horizons
        )
        for key, value in test_metrics.items():
            summary.loc[best_index, f"test_{key}"] = value
        test_meta["risk"] = test_risk
        test_meta.to_csv(root / f"{selected}_test_predictions.csv", index=False)
        if isinstance(selected_head, LateRiskFusionModel):
            raw, multilevel, functional = (
                tensor.to(device) for tensor in test_x
            )
            selected_head.eval()
            with torch.no_grad():
                phenotype_risk = (
                    selected_head.phenotype_head((raw, multilevel))
                    - selected_head.phenotype_mean
                ) / selected_head.phenotype_std
                function_risk = (
                    selected_head.function_head((functional,))
                    - selected_head.function_mean
                ) / selected_head.function_std
            component_meta = test_meta[
                ["patient_id", "time_to_event", "event"]
            ].copy()
            component_meta["phenotype_risk"] = (
                phenotype_risk.detach().float().cpu().numpy()
            )
            component_meta["function_risk"] = (
                function_risk.detach().float().cpu().numpy()
            )
            component_meta.to_csv(
                root / "K_test_component_predictions.csv", index=False
            )
    direct_val = summary.loc[
        summary["variant"] == "A_direct", "val_cindex_selection"
    ]
    if len(direct_val):
        summary["delta_val_cindex_vs_A"] = (
            summary["val_cindex_selection"] - float(direct_val.iloc[0])
        )
    torch.save(
        {
            "version": VERSION,
            "selected_by": "validation_cindex_only",
            "selected_variant": selected,
            "feature_cache": str(cache_path),
            "multilevel_feature_cache": (
                str(multilevel_cache) if multilevel_cache is not None else ""
            ),
            "raw_multilevel_feature_cache": (
                str(raw_multilevel_cache)
                if raw_multilevel_cache is not None else ""
            ),
            "motion_feature_cache": (
                str(motion_cache) if motion_cache is not None else ""
            ),
            "segmentation_feature_cache": (
                str(segmentation_cache) if segmentation_cache is not None else ""
            ),
            "cox_head": {
                key: value.detach().cpu()
                for key, value in selected_head.state_dict().items()
            },
            "metrics": summary.loc[best_index].to_dict(),
        },
        root / "selected_prognosis_model.pt",
    )
    summary.to_csv(root / "fullrisk_screen_summary.csv", index=False)
    compact_columns = [
        "variant", "selected_on_validation", "best_epoch", "train_cindex",
        "val_cindex_selection", "delta_val_cindex_vs_A", "test_cindex",
    ]
    compact = summary[
        [column for column in compact_columns if column in summary.columns]
    ].copy()
    compact.to_csv(root / "key_metrics.csv", index=False)
    ns["print0"]("\n========== Key prognosis metrics ==========")
    ns["print0"](compact.to_string(index=False))
    selected_row = summary.loc[best_index]
    selected_message = (
        f"SELECTED={selected} | "
        f"VAL_CINDEX={float(selected_row['val_cindex_selection']):.4f}"
    )
    if evaluate_test:
        selected_message += (
            f" | TEST_CINDEX={float(selected_row['test_cindex']):.4f}"
        )
    else:
        selected_message += " | TEST=LOCKED"
    ns["print0"](selected_message)
    ns["print0"](f"Saved to: {root}")
