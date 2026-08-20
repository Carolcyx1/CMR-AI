"""Final DCM prognosis: fixed K versus matched and simple direct controls.

Set ``FORCE_RETRAIN`` below to True to retrain K, A_matched, and A_direct
for all three fixed seeds. Otherwise matching completed runs are reused.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent if CODE_ROOT.name == "final_prognosis_code" else CODE_ROOT

# False: reuse complete matching runs and only refresh summaries/bootstrap.
# True: retrain all 3 models × all 3 seeds before summaries/bootstrap.
FORCE_RETRAIN = False

# GPU IDs visible to every DCM training run.
# Examples: "0" = single GPU; "0,1" = two GPUs; "0,1,2,3" = four GPUs.
# To run DCM and HCM concurrently on four GPUs, use "0,1" here and "2,3"
# in the HCM final script.
VISIBLE_GPUS = "0,1,2,3"
ACTIVE_VISIBLE_GPUS = os.environ.get(
    "FINAL_PROGNOSIS_VISIBLE_GPUS", VISIBLE_GPUS
)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = Path(os.environ.get(
    "FINAL_PROGNOSIS_DISEASE_ROOT",
    ROOT / f"DCM_final_prognosis_{RUN_TIMESTAMP}",
))
TRAINING_RUNS = RUN_ROOT / "training_runs"

BASE_SCRIPT = CODE_ROOT / "dcm_nested_oof_fixed_k.py"
CONTROL_ENTRY = CODE_ROOT / "dcm_matched_direct_entry.py"
NESTED_MODULE = CODE_ROOT / "prognosis_fullrisk_teacher_adapt_oof_fixed.py"
CONTROL_RUN_HASH = hashlib.sha256(
    BASE_SCRIPT.read_bytes()
    + CONTROL_ENTRY.read_bytes()
    + NESTED_MODULE.read_bytes()
).hexdigest()[:10]
K_RUN_HASH = hashlib.sha256(
    BASE_SCRIPT.read_bytes() + NESTED_MODULE.read_bytes()
).hexdigest()[:10]
OUTPUT = RUN_ROOT / "three_seed_summary"
BOOTSTRAP_OUTPUT = RUN_ROOT / "paired_bootstrap"
SEEDS = (42, 2024, 3407)
VARIANTS = {
    "K": "K_oof_late_fusion",
    "A_matched": "A_raw_multilevel_residual_b8",
    "A_direct": "A_direct",
}
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260730


def cindex(frame_or_time, risk, event=None):
    if event is None:
        frame = frame_or_time
        time = frame["time_to_event"].to_numpy(dtype=float)
        event = frame["event"].to_numpy(dtype=int)
    else:
        time = np.asarray(frame_or_time, dtype=float)
        event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    comparable = (
        (time[:, None] < time[None, :])
        & (event[:, None] == 1)
    )
    denominator = int(comparable.sum())
    if denominator == 0:
        return np.nan
    concordant = risk[:, None] > risk[None, :] + 1e-12
    tied = np.abs(risk[:, None] - risk[None, :]) <= 1e-12
    return float(
        (
            concordant[comparable].sum()
            + 0.5 * tied[comparable].sum()
        )
        / denominator
    )


def experiment_name(label, seed):
    if label == "K":
        return f"dcm_nested_oof_fixed_{K_RUN_HASH}_seed{seed}"
    return (
        f"dcm_nested_oof_{label.lower()}_{CONTROL_RUN_HASH}_seed{seed}"
    )


def nested_path(label, directory):
    if label == "K":
        return directory / "K_oof_predictions.csv"
    return directory / f"{VARIANTS[label]}_nested_oof_predictions.csv"


def completed(label, directory):
    variant = VARIANTS[label]
    return (
        (directory / f"{variant}_test_predictions.csv").is_file()
        and nested_path(label, directory).is_file()
    )


def find_completed(label, seed):
    name = experiment_name(label, seed)
    candidates = list(ROOT.glob(
        f"20260729_dcm_diagnosis_to_prognosis_abcd_*/{name}"
    ))
    candidates += list(ROOT.glob(
        f"DCM_final_prognosis_*/training_runs/{label}_seed{seed}"
    ))
    candidates += list(ROOT.glob(
        f"final_prognosis_*/DCM/training_runs/{label}_seed{seed}"
    ))
    canonical = TRAINING_RUNS / f"{label}_seed{seed}"
    if canonical.exists():
        candidates.append(canonical)
    valid = [path for path in candidates if completed(label, path)]
    return max(valid, key=lambda x: x.stat().st_mtime) if valid else None


def register_training_run(label, seed, source, move=False):
    TRAINING_RUNS.mkdir(parents=True, exist_ok=True)
    target = TRAINING_RUNS / f"{label}_seed{seed}"
    if target.exists() or target.is_symlink():
        return target
    if move:
        source_parent = source.parent
        shutil.move(str(source), str(target))
        try:
            source_parent.rmdir()
        except OSError:
            pass
    else:
        shutil.copytree(source.resolve(), target)
    return target


def run_variant(label, seed, force_retrain=False):
    if not force_retrain:
        existing = find_completed(label, seed)
        if existing is not None:
            return register_training_run(label, seed, existing)
    entry = BASE_SCRIPT if label == "K" else CONTROL_ENTRY
    command = [
        sys.executable,
        str(entry),
        "--seed", str(seed),
        "--visible_gpus", ACTIVE_VISIBLE_GPUS,
        "--fullrisk_variants", VARIANTS[label],
        "--experiment_name", experiment_name(label, seed),
        "--evaluate_test_after_selection",
    ]
    log_dir = TRAINING_RUNS / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}_seed{seed}.log"
    print(
        f"[DCM] Training {label}, seed {seed} "
        f"(details: {log_path})",
        flush=True,
    )
    with open(log_path, "w", encoding="utf-8") as log_handle:
        try:
            subprocess.run(
                command, cwd=ROOT, check=True,
                stdout=log_handle, stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            print(f"[DCM] Training failed; inspect {log_path}")
            raise
    result = find_completed(label, seed)
    if result is None:
        raise RuntimeError(
            f"{label} seed {seed} finished without expected predictions"
        )
    return register_training_run(label, seed, result, move=True)


def load_predictions(label, directory):
    variant = VARIANTS[label]
    result = {
        split: pd.read_csv(
            directory / f"{variant}_{split}_predictions.csv"
        )
        for split in ("train", "val", "test")
    }
    nested = pd.read_csv(nested_path(label, directory))
    if label == "K":
        required = {
            "patient_id", "time_to_event", "event", "outer_fold",
            "nested_fused_oof_risk",
        }
        missing = required - set(nested.columns)
        if missing:
            raise RuntimeError(
                f"K nested OOF audit columns missing: {sorted(missing)}"
            )
        nested = nested.rename(
            columns={"nested_fused_oof_risk": "nested_oof_risk"}
        )
    result["nested_oof"] = nested
    return result


def assert_aligned(reference, candidate, tag):
    if (
        reference["patient_id"].astype(str).tolist()
        != candidate["patient_id"].astype(str).tolist()
    ):
        raise RuntimeError(f"Cohort/order mismatch: {tag}/patient_id")
    if not np.allclose(
        reference["time_to_event"].to_numpy(dtype=float),
        candidate["time_to_event"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(f"Cohort mismatch: {tag}/time_to_event")
    if not np.array_equal(
        reference["event"].to_numpy(dtype=int),
        candidate["event"].to_numpy(dtype=int),
    ):
        raise RuntimeError(f"Cohort mismatch: {tag}/event")


def standardize_full_training(predictions):
    train = predictions["train"]["risk"].to_numpy(dtype=float)
    mean = float(train.mean())
    std = max(float(train.std(ddof=0)), 1e-5)
    return {
        split: (
            predictions[split]["risk"].to_numpy(dtype=float) - mean
        ) / std
        for split in ("train", "val", "test")
    }


def standardize_nested_oof(predictions):
    risk = predictions["nested_oof"][
        "nested_oof_risk"
    ].to_numpy(dtype=float)
    return (risk - float(risk.mean())) / max(
        float(risk.std(ddof=0)), 1e-5
    )


def paired_bootstrap(test_frame):
    time = test_frame["time_to_event"].to_numpy(dtype=float)
    event = test_frame["event"].to_numpy(dtype=int)
    k_risk = test_frame[
        "K_risk_three_seed_mean"
    ].to_numpy(dtype=float)
    comparators = {
        label: test_frame[
            f"{label}_risk_three_seed_mean"
        ].to_numpy(dtype=float)
        for label in ("A_matched", "A_direct")
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    k_boot = np.empty(N_BOOTSTRAP, dtype=float)
    control_boot = {
        label: np.empty(N_BOOTSTRAP, dtype=float)
        for label in comparators
    }
    for iteration in range(N_BOOTSTRAP):
        index = rng.integers(0, len(time), size=len(time))
        k_boot[iteration] = cindex(
            time[index], k_risk[index], event[index]
        )
        for label, risk in comparators.items():
            control_boot[label][iteration] = cindex(
                time[index], risk[index], event[index]
            )

    observed_k = cindex(time, k_risk, event)
    rows = []
    distribution_frames = []
    for label, risk in comparators.items():
        valid = np.isfinite(k_boot) & np.isfinite(control_boot[label])
        if int(valid.sum()) < 0.95 * N_BOOTSTRAP:
            raise RuntimeError(
                f"Too few valid bootstrap samples for {label}: "
                f"{int(valid.sum())}"
            )
        valid_k = k_boot[valid]
        valid_control = control_boot[label][valid]
        delta = valid_k - valid_control
        observed_control = cindex(time, risk, event)
        delta_ci = np.quantile(delta, [0.025, 0.975])
        control_ci = np.quantile(valid_control, [0.025, 0.975])
        p_value = min(
            1.0,
            2.0 * min(
                (np.count_nonzero(delta <= 0) + 1)
                / (len(delta) + 1),
                (np.count_nonzero(delta >= 0) + 1)
                / (len(delta) + 1),
            ),
        )
        rows.append({
            "comparison": f"K_vs_{label}",
            "k_cindex": observed_k,
            "k_ci95_low": np.quantile(valid_k, 0.025),
            "k_ci95_high": np.quantile(valid_k, 0.975),
            "control_cindex": observed_control,
            "control_ci95_low": control_ci[0],
            "control_ci95_high": control_ci[1],
            "delta_cindex": observed_k - observed_control,
            "delta_ci95_low": delta_ci[0],
            "delta_ci95_high": delta_ci[1],
            "paired_bootstrap_p_two_sided": p_value,
            "n_bootstrap_valid": int(valid.sum()),
        })
        distribution_frames.append(pd.DataFrame({
            "comparison": label,
            "iteration": np.flatnonzero(valid),
            "K_cindex": valid_k,
            "control_cindex": valid_control,
            "delta_cindex": delta,
        }))
    BOOTSTRAP_OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        BOOTSTRAP_OUTPUT / "paired_bootstrap_summary.csv", index=False
    )
    pd.concat(distribution_frames, ignore_index=True).to_csv(
        BOOTSTRAP_OUTPUT / "paired_bootstrap_distributions.csv",
        index=False,
    )
    return summary


def main(force_retrain=False):
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    mode = "FORCE RETRAIN" if force_retrain else "REUSE IF COMPLETE"
    print(f"[DCM] Mode: {mode}")
    print(f"[DCM] GPUs: {ACTIVE_VISIBLE_GPUS}")
    directories = {
        (label, seed): run_variant(label, seed, force_retrain)
        for label in VARIANTS
        for seed in SEEDS
    }
    print("[DCM] All 3 models × 3 seeds are ready.")
    predictions = {
        key: load_predictions(key[0], directory)
        for key, directory in directories.items()
    }
    reference = predictions[("K", 42)]
    for (label, seed), by_split in predictions.items():
        for split in ("train", "val", "test"):
            assert_aligned(
                reference[split], by_split[split],
                f"{label}/seed{seed}/{split}",
            )
        assert_aligned(
            reference["train"], by_split["nested_oof"],
            f"{label}/seed{seed}/train-vs-nested_oof",
        )
        assert_aligned(
            reference["nested_oof"], by_split["nested_oof"],
            f"{label}/seed{seed}/nested_oof",
        )
    for seed in SEEDS:
        reference_fold = predictions[
            ("K", seed)
        ]["nested_oof"]["outer_fold"].to_numpy(dtype=int)
        for label in VARIANTS:
            candidate_fold = predictions[
                (label, seed)
            ]["nested_oof"]["outer_fold"].to_numpy(dtype=int)
            if not np.array_equal(reference_fold, candidate_fold):
                raise RuntimeError(
                    f"Outer-fold mismatch: {label}/seed{seed}"
                )

    full_risks = {
        key: standardize_full_training(value)
        for key, value in predictions.items()
    }
    oof_risks = {
        key: standardize_nested_oof(value)
        for key, value in predictions.items()
    }
    ensemble = {
        label: {
            split: np.mean(
                np.stack([
                    full_risks[(label, seed)][split]
                    for seed in SEEDS
                ]),
                axis=0,
            )
            for split in ("train", "val", "test")
        }
        for label in VARIANTS
    }
    ensemble_oof = {
        label: np.mean(
            np.stack([oof_risks[(label, seed)] for seed in SEEDS]),
            axis=0,
        )
        for label in VARIANTS
    }

    single_rows = []
    for label in VARIANTS:
        for seed in SEEDS:
            by_split = predictions[(label, seed)]
            single_rows.append({
                "model": label,
                "seed": seed,
                "train_oof_cindex": cindex(
                    by_split["nested_oof"],
                    by_split["nested_oof"]["nested_oof_risk"],
                ),
                "val_cindex": cindex(
                    by_split["val"], by_split["val"]["risk"]
                ),
                "test_cindex": cindex(
                    by_split["test"], by_split["test"]["risk"]
                ),
                "result_directory": str(directories[(label, seed)]),
            })
    single = pd.DataFrame(single_rows)
    aggregate_rows = []
    for label in VARIANTS:
        subset = single[single["model"] == label]
        aggregate_rows.append({
            "model": label,
            "n_seeds": len(SEEDS),
            "train_oof_cindex_mean": subset["train_oof_cindex"].mean(),
            "train_oof_cindex_sd": subset["train_oof_cindex"].std(ddof=1),
            "val_cindex_mean": subset["val_cindex"].mean(),
            "val_cindex_sd": subset["val_cindex"].std(ddof=1),
            "test_cindex_mean": subset["test_cindex"].mean(),
            "test_cindex_sd": subset["test_cindex"].std(ddof=1),
            "three_seed_train_oof_cindex": cindex(
                reference["nested_oof"], ensemble_oof[label]
            ),
            "three_seed_val_cindex": cindex(
                reference["val"], ensemble[label]["val"]
            ),
            "three_seed_test_cindex": cindex(
                reference["test"], ensemble[label]["test"]
            ),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    for control in ("A_matched", "A_direct"):
        for split in ("val", "test"):
            control_value = float(
                aggregate.loc[
                    aggregate["model"] == control,
                    f"three_seed_{split}_cindex",
                ].iloc[0]
            )
            aggregate[f"delta_{split}_vs_{control}"] = (
                aggregate[f"three_seed_{split}_cindex"] - control_value
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    single.to_csv(OUTPUT / "single_seed_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT / "three_seed_summary.csv", index=False)
    with open(OUTPUT / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump({
            "control_run_hash": CONTROL_RUN_HASH,
            "locked_k_run_hash": K_RUN_HASH,
            "base_script": str(BASE_SCRIPT),
            "control_entry": str(CONTROL_ENTRY),
            "nested_module": str(NESTED_MODULE),
            "nested_module_version": (
                "diagnosis_to_prognosis_deterministic_raw_encoder_v6"
            ),
            "raw_encoder_feature_policy": (
                "raw_view_embeddings_deterministic_zero_v2"
            ),
            "seeds": list(SEEDS),
            "variants": VARIANTS,
            "K_policy": (
                "retrain all when FORCE_RETRAIN=True; otherwise reuse locked v6"
            ),
            "control_selection": (
                "each control trained and selected independently on validation"
            ),
        }, handle, indent=2)

    oof_frame = reference["nested_oof"][
        ["patient_id", "time_to_event", "event", "outer_fold"]
    ].copy()
    for label in VARIANTS:
        for seed in SEEDS:
            oof_frame[f"{label}_nested_oof_seed{seed}"] = oof_risks[
                (label, seed)
            ]
        oof_frame[f"{label}_nested_oof_three_seed_mean"] = (
            ensemble_oof[label]
        )
    oof_frame.to_csv(
        OUTPUT / "three_seed_train_nested_oof_predictions.csv",
        index=False,
    )
    output_frames = {}
    for split in ("train", "val", "test"):
        frame = reference[split][
            ["patient_id", "time_to_event", "event"]
        ].copy()
        for label in VARIANTS:
            for seed in SEEDS:
                frame[f"{label}_risk_seed{seed}"] = full_risks[
                    (label, seed)
                ][split]
            frame[f"{label}_risk_three_seed_mean"] = ensemble[label][
                split
            ]
        frame.to_csv(
            OUTPUT / f"three_seed_{split}_predictions.csv", index=False
        )
        output_frames[split] = frame
    pd.DataFrame([
        {
            "model": label,
            "train_median_threshold": float(
                np.median(ensemble[label]["train"])
            ),
        }
        for label in VARIANTS
    ]).to_csv(OUTPUT / "locked_train_risk_thresholds.csv", index=False)

    bootstrap = paired_bootstrap(output_frames["test"])
    display = aggregate[[
        "model",
        "three_seed_train_oof_cindex",
        "three_seed_val_cindex",
        "three_seed_test_cindex",
    ]].rename(columns={
        "three_seed_train_oof_cindex": "OOF C-index",
        "three_seed_val_cindex": "Val C-index",
        "three_seed_test_cindex": "Test C-index",
    })
    comparisons = bootstrap[[
        "comparison", "delta_cindex", "delta_ci95_low",
        "delta_ci95_high", "paired_bootstrap_p_two_sided",
        "n_bootstrap_valid",
    ]].copy()
    comparisons["Delta C-index (95% CI)"] = comparisons.apply(
        lambda row: (
            f"{row['delta_cindex']:.3f} "
            f"({row['delta_ci95_low']:.3f}, "
            f"{row['delta_ci95_high']:.3f})"
        ),
        axis=1,
    )
    comparisons = comparisons[[
        "comparison", "Delta C-index (95% CI)",
        "paired_bootstrap_p_two_sided", "n_bootstrap_valid",
    ]].rename(columns={
        "comparison": "Comparison",
        "paired_bootstrap_p_two_sided": "P value",
        "n_bootstrap_valid": "Valid bootstrap",
    })
    display.to_csv(OUTPUT / "final_metrics_concise.csv", index=False)
    comparisons.to_csv(
        BOOTSTRAP_OUTPUT / "final_comparisons_concise.csv",
        index=False,
    )
    print("\n========== DCM FINAL RESULTS ==========")
    print(display.to_string(
        index=False,
        formatters={
            column: "{:.3f}".format
            for column in ("OOF C-index", "Val C-index", "Test C-index")
        },
    ))
    print("\nPaired bootstrap (10,000 repeats)")
    print(comparisons.to_string(
        index=False,
        formatters={"P value": "{:.4f}".format},
    ))
    print(f"\nResults: {OUTPUT}")
    print(f"Bootstrap: {BOOTSTRAP_OUTPUT}")
    print(f"Complete DCM run: {RUN_ROOT}")


if __name__ == "__main__":
    env_force = os.environ.get("FINAL_PROGNOSIS_FORCE_RETRAIN")
    selected_force = (
        FORCE_RETRAIN if env_force is None
        else env_force.strip().lower() in {"1", "true", "yes", "y"}
    )
    main(force_retrain=selected_force)
