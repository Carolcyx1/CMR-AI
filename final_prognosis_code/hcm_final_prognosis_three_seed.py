"""Final HCM prognosis: F versus matched/basic direct controls.

Runs three fixed seeds for all three models, builds patient-aligned ensembles,
and performs the final 10,000-repeat paired bootstrap comparisons. Completed
matching runs are reused by default. Set ``FORCE_RETRAIN`` below to True to
retrain all three models for all three fixed seeds.
"""

from __future__ import annotations

import subprocess
import sys
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent if CODE_ROOT.name == "final_prognosis_code" else CODE_ROOT

# False: reuse complete matching runs and only refresh summaries/bootstrap.
# True: retrain all 3 models × all 3 seeds before summaries/bootstrap.
FORCE_RETRAIN = False

# GPU IDs visible to every HCM training run.
# Examples: "0" = single GPU; "0,1" = two GPUs; "0,1,2,3" = four GPUs.
# To run DCM and HCM concurrently on four GPUs, use "2,3" here and "0,1"
# in the DCM final script.
VISIBLE_GPUS = "0,1,2,3"
ACTIVE_VISIBLE_GPUS = os.environ.get(
    "FINAL_PROGNOSIS_VISIBLE_GPUS", VISIBLE_GPUS
)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = Path(os.environ.get(
    "FINAL_PROGNOSIS_DISEASE_ROOT",
    ROOT / f"HCM_final_prognosis_{RUN_TIMESTAMP}",
))
TRAINING_RUNS = RUN_ROOT / "training_runs"

BASE_SCRIPT = CODE_ROOT / "hcm_nested_oof_fixed.py"
OUTPUT = RUN_ROOT / "three_seed_summary"
BOOTSTRAP_OUTPUT = RUN_ROOT / "paired_bootstrap"
NESTED_MODULE = CODE_ROOT / "prognosis_fullrisk_teacher_adapt_oof_fixed.py"
RUN_HASH = hashlib.sha256(
    BASE_SCRIPT.read_bytes() + NESTED_MODULE.read_bytes()
).hexdigest()[:10]
SEEDS = (42, 2024, 3407)
VARIANTS = {
    "F_two_stage": "F_multilevel_residual_b8",
    "A_matched": "A_raw_multilevel_residual_b8",
    "A_direct": "A_direct",
}
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260730


def cindex(frame, risk):
    time = frame["time_to_event"].to_numpy(dtype=float)
    event = frame["event"].to_numpy(dtype=int)
    risk = np.asarray(risk, dtype=float)
    comparable = (
        (time[:, None] < time[None, :])
        & (event[:, None] == 1)
    )
    concordant = risk[:, None] > risk[None, :] + 1e-12
    tied = np.abs(risk[:, None] - risk[None, :]) <= 1e-12
    denominator = int(comparable.sum())
    return float(
        (
            concordant[comparable].sum()
            + 0.5 * tied[comparable].sum()
        )
        / denominator
    )


def cindex_arrays(time, event, risk):
    """Harrell's C-index with protection for bootstrap samples without pairs."""
    time = np.asarray(time, dtype=float)
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


def paired_bootstrap(test_frame):
    """Compare the final HCM ensemble with both direct-prognosis controls."""
    time = test_frame["time_to_event"].to_numpy(dtype=float)
    event = test_frame["event"].to_numpy(dtype=int)
    f_risk = test_frame[
        "F_two_stage_risk_three_seed_mean"
    ].to_numpy(dtype=float)
    comparators = {
        "A_matched": test_frame[
            "A_matched_risk_three_seed_mean"
        ].to_numpy(dtype=float),
        "A_direct": test_frame[
            "A_direct_risk_three_seed_mean"
        ].to_numpy(dtype=float),
    }
    observed_f = cindex_arrays(time, event, f_risk)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    f_boot = np.empty(N_BOOTSTRAP, dtype=float)
    comparator_boot = {
        name: np.empty(N_BOOTSTRAP, dtype=float)
        for name in comparators
    }
    for iteration in range(N_BOOTSTRAP):
        index = rng.integers(
            0, len(test_frame), size=len(test_frame)
        )
        f_boot[iteration] = cindex_arrays(
            time[index], event[index], f_risk[index]
        )
        for name, risk in comparators.items():
            comparator_boot[name][iteration] = cindex_arrays(
                time[index], event[index], risk[index]
            )

    rows = []
    distributions = []
    for name, risk in comparators.items():
        observed_control = cindex_arrays(time, event, risk)
        valid = (
            np.isfinite(f_boot)
            & np.isfinite(comparator_boot[name])
        )
        if int(valid.sum()) < 0.95 * N_BOOTSTRAP:
            raise RuntimeError(
                f"Too few valid bootstrap samples for {name}: "
                f"{valid.sum()}"
            )
        valid_f = f_boot[valid]
        valid_control = comparator_boot[name][valid]
        delta = valid_f - valid_control
        delta_ci = np.quantile(delta, [0.025, 0.975])
        control_ci = np.quantile(
            valid_control, [0.025, 0.975]
        )
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
            "comparison": f"F_two_stage_vs_{name}",
            "f_cindex": observed_f,
            "f_ci95_low": np.quantile(valid_f, 0.025),
            "f_ci95_high": np.quantile(valid_f, 0.975),
            "control_cindex": observed_control,
            "control_ci95_low": control_ci[0],
            "control_ci95_high": control_ci[1],
            "delta_cindex": observed_f - observed_control,
            "delta_ci95_low": delta_ci[0],
            "delta_ci95_high": delta_ci[1],
            "paired_bootstrap_p_two_sided": p_value,
            "n_bootstrap_valid": int(valid.sum()),
        })
        distributions.append(pd.DataFrame({
            "comparison": name,
            "iteration": np.flatnonzero(valid),
            "F_two_stage_cindex": valid_f,
            "control_cindex": valid_control,
            "delta_cindex": delta,
        }))

    BOOTSTRAP_OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        BOOTSTRAP_OUTPUT / "paired_bootstrap_summary.csv",
        index=False,
    )
    pd.concat(distributions, ignore_index=True).to_csv(
        BOOTSTRAP_OUTPUT / "paired_bootstrap_distributions.csv",
        index=False,
    )
    return summary


def experiment_name(label, seed):
    return f"hcm_nested_oof_{label.lower()}_{RUN_HASH}_seed{seed}"


def find_completed(label, seed):
    variant = VARIANTS[label]
    name = experiment_name(label, seed)
    candidates = list(
        ROOT.glob(
            f"20260729_hcm_diagnosis_to_prognosis_abcd_*/{name}"
        )
    )
    candidates += list(ROOT.glob(
        f"HCM_final_prognosis_*/training_runs/{label}_seed{seed}"
    ))
    candidates += list(ROOT.glob(
        f"final_prognosis_*/HCM/training_runs/{label}_seed{seed}"
    ))
    canonical = TRAINING_RUNS / f"{label}_seed{seed}"
    if canonical.exists():
        candidates.append(canonical)
    valid = [
        path for path in candidates
        if (path / f"{variant}_test_predictions.csv").is_file()
        and (path / f"{variant}_nested_oof_predictions.csv").is_file()
    ]
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


def run_fixed(label, seed, force_retrain=False):
    if not force_retrain:
        existing = find_completed(label, seed)
        if existing is not None:
            return register_training_run(label, seed, existing)
    variant = VARIANTS[label]
    command = [
        sys.executable,
        str(BASE_SCRIPT),
        "--seed", str(seed),
        "--visible_gpus", ACTIVE_VISIBLE_GPUS,
        "--fullrisk_variants", variant,
        "--experiment_name", experiment_name(label, seed),
        "--evaluate_test_after_selection",
    ]
    log_dir = TRAINING_RUNS / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}_seed{seed}.log"
    print(
        f"[HCM] Training {label}, seed {seed} "
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
            print(f"[HCM] Training failed; inspect {log_path}")
            raise
    result = find_completed(label, seed)
    if result is None:
        raise RuntimeError(
            f"{label} seed {seed} finished without test predictions"
        )
    return register_training_run(label, seed, result, move=True)


def load_predictions(label, seed, directory):
    variant = VARIANTS[label]
    predictions = {
        split: pd.read_csv(
            directory / f"{variant}_{split}_predictions.csv"
        )
        for split in ("train", "val", "test")
    }
    predictions["nested_oof"] = pd.read_csv(
        directory / f"{variant}_nested_oof_predictions.csv"
    )
    return predictions


def assert_aligned(reference, candidate, tag):
    if (
        reference["patient_id"].astype(str).tolist()
        != candidate["patient_id"].astype(str).tolist()
    ):
        raise RuntimeError(f"Cohort mismatch: {tag}/patient_id")
    if not np.allclose(
        reference["time_to_event"].to_numpy(dtype=float),
        candidate["time_to_event"].to_numpy(dtype=float),
        rtol=0.0, atol=1e-6,
    ):
        raise RuntimeError(f"Cohort mismatch: {tag}/time_to_event")
    if not np.array_equal(
        reference["event"].to_numpy(dtype=int),
        candidate["event"].to_numpy(dtype=int),
    ):
        raise RuntimeError(f"Cohort mismatch: {tag}/event")


def standardize(predictions):
    train = predictions["train"]["risk"].to_numpy(dtype=float)
    mean = float(train.mean())
    std = max(float(train.std()), 1e-5)
    return {
        split: (
            predictions[split]["risk"].to_numpy(dtype=float) - mean
        ) / std
        for split in ("train", "val", "test")
    }


def main(force_retrain=False):
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    mode = "FORCE RETRAIN" if force_retrain else "REUSE IF COMPLETE"
    print(f"[HCM] Mode: {mode}")
    print(f"[HCM] GPUs: {ACTIVE_VISIBLE_GPUS}")
    directories = {
        (label, seed): run_fixed(label, seed, force_retrain)
        for label in VARIANTS
        for seed in SEEDS
    }
    print("[HCM] All 3 models × 3 seeds are ready.")
    predictions = {
        key: load_predictions(*key, directory)
        for key, directory in directories.items()
    }
    reference = predictions[("F_two_stage", 42)]
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
            predictions[("F_two_stage", 42)]["nested_oof"],
            by_split["nested_oof"],
            f"{label}/seed{seed}/nested_oof",
        )
    for seed in SEEDS:
        reference_fold = predictions[
            ("F_two_stage", seed)
        ]["nested_oof"]["outer_fold"].to_numpy(dtype=int)
        for label in VARIANTS:
            candidate_fold = predictions[
                (label, seed)
            ]["nested_oof"]["outer_fold"].to_numpy(dtype=int)
            if not np.array_equal(reference_fold, candidate_fold):
                raise RuntimeError(
                    f"Outer-fold assignment mismatch for {label}/seed{seed}"
                )

    risks = {
        key: standardize(value)
        for key, value in predictions.items()
    }
    nested_oof_risks = {
        key: (
            value["nested_oof"]["nested_oof_risk"].to_numpy(dtype=float)
            - float(
                value["nested_oof"]["nested_oof_risk"].mean()
            )
        )
        / max(
            float(value["nested_oof"]["nested_oof_risk"].std(ddof=0)),
            1e-5,
        )
        for key, value in predictions.items()
    }
    rows = []
    for label in VARIANTS:
        for seed in SEEDS:
            by_split = predictions[(label, seed)]
            rows.append({
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
    single = pd.DataFrame(rows)

    ensemble_risks = {}
    aggregate_rows = []
    for label in VARIANTS:
        ensemble_risks[label] = {
            split: np.mean(
                np.stack([
                    risks[(label, seed)][split] for seed in SEEDS
                ]),
                axis=0,
            )
            for split in ("train", "val", "test")
        }
        subset = single[single["model"] == label]
        aggregate_rows.append({
            "model": label,
            "n_seeds": len(SEEDS),
            "train_oof_cindex_mean": subset[
                "train_oof_cindex"
            ].mean(),
            "train_oof_cindex_sd": subset[
                "train_oof_cindex"
            ].std(ddof=1),
            "val_cindex_mean": subset["val_cindex"].mean(),
            "val_cindex_sd": subset["val_cindex"].std(ddof=1),
            "test_cindex_mean": subset["test_cindex"].mean(),
            "test_cindex_sd": subset["test_cindex"].std(ddof=1),
            "three_seed_train_oof_cindex": cindex(
                predictions[(label, 42)]["nested_oof"],
                np.mean(np.stack([
                    nested_oof_risks[(label, seed)]
                    for seed in SEEDS
                ]), axis=0),
            ),
            "three_seed_val_cindex": cindex(
                reference["val"], ensemble_risks[label]["val"]
            ),
            "three_seed_test_cindex": cindex(
                reference["test"], ensemble_risks[label]["test"]
            ),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["delta_three_seed_val_vs_direct"] = (
        aggregate["three_seed_val_cindex"]
        - float(
            aggregate.loc[
                aggregate["model"] == "A_direct",
                "three_seed_val_cindex",
            ].iloc[0]
        )
    )
    aggregate["delta_three_seed_test_vs_direct"] = (
        aggregate["three_seed_test_cindex"]
        - float(
            aggregate.loc[
                aggregate["model"] == "A_direct",
                "three_seed_test_cindex",
            ].iloc[0]
        )
    )
    for split in ("val", "test"):
        matched_value = float(
            aggregate.loc[
                aggregate["model"] == "A_matched",
                f"three_seed_{split}_cindex",
            ].iloc[0]
        )
        aggregate[f"delta_three_seed_{split}_vs_matched"] = (
            aggregate[f"three_seed_{split}_cindex"] - matched_value
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    single.to_csv(OUTPUT / "single_seed_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT / "three_seed_summary.csv", index=False)
    with open(OUTPUT / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_hash": RUN_HASH,
                "base_script": str(BASE_SCRIPT),
                "nested_module": str(NESTED_MODULE),
                "nested_module_version": (
                    "diagnosis_to_prognosis_deterministic_raw_encoder_v6"
                ),
                "raw_encoder_feature_policy": (
                    "raw_view_embeddings_deterministic_zero_v2"
                ),
                "raw_multilevel_cache_version": (
                    "raw_multilevel_zero_view_embedding_v2"
                ),
                "raw_view_embedding_policy": "deterministic_zero",
                "seeds": list(SEEDS),
                "variants": VARIANTS,
                "nested_oof_standardization": (
                    "within-seed cross-fitted risk distribution"
                ),
                "validation_test_standardization": (
                    "corresponding full-training model risk distribution"
                ),
            },
            handle, indent=2,
        )
    oof_frame = predictions[("F_two_stage", 42)]["nested_oof"][
        ["patient_id", "time_to_event", "event"]
    ].copy()
    for label in VARIANTS:
        for seed in SEEDS:
            oof_frame[f"{label}_nested_oof_seed{seed}"] = (
                nested_oof_risks[(label, seed)]
            )
        oof_frame[f"{label}_nested_oof_three_seed_mean"] = np.mean(
            np.stack([
                nested_oof_risks[(label, seed)] for seed in SEEDS
            ]),
            axis=0,
        )
    oof_frame.to_csv(
        OUTPUT / "three_seed_train_nested_oof_predictions.csv",
        index=False,
    )
    saved_split_frames = {}
    for split in ("train", "val", "test"):
        frame = reference[split][
            ["patient_id", "time_to_event", "event"]
        ].copy()
        for label in VARIANTS:
            for seed in SEEDS:
                frame[f"{label}_risk_seed{seed}"] = risks[
                    (label, seed)
                ][split]
            frame[f"{label}_risk_three_seed_mean"] = ensemble_risks[
                label
            ][split]
        frame.to_csv(
            OUTPUT / f"three_seed_{split}_predictions.csv", index=False
        )
        saved_split_frames[split] = frame
    thresholds = {
        label: float(np.median(ensemble_risks[label]["train"]))
        for label in VARIANTS
    }
    pd.DataFrame([
        {"model": label, "train_median_threshold": threshold}
        for label, threshold in thresholds.items()
    ]).to_csv(OUTPUT / "locked_train_risk_thresholds.csv", index=False)
    bootstrap = paired_bootstrap(saved_split_frames["test"])
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
    print("\n========== HCM FINAL RESULTS ==========")
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
    print(f"Complete HCM run: {RUN_ROOT}")


if __name__ == "__main__":
    env_force = os.environ.get("FINAL_PROGNOSIS_FORCE_RETRAIN")
    selected_force = (
        FORCE_RETRAIN if env_force is None
        else env_force.strip().lower() in {"1", "true", "yes", "y"}
    )
    main(force_retrain=selected_force)
