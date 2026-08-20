"""Unified DCM/HCM prognosis training, summary, bootstrap, and figure pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# --------------------------- USER SWITCHES ---------------------------
# Select either or both disease-specific pipelines.
RUN_DCM = True
RUN_HCM = True

# False: reuse complete matching runs and regenerate final result summaries.
# True: retrain all selected models × all 3 seeds before summarization.
FORCE_RETRAIN = False

# False: stop after disease result summaries and paired bootstrap.
# True: additionally generate the selected disease figures.
RUN_FIGURES = True

# GPU assignment. For concurrent disease scripts use separate GPU sets.
# This unified pipeline runs DCM then HCM, so sharing all four is safe.
DCM_VISIBLE_GPUS = "0,1,2,3"
HCM_VISIBLE_GPUS = "0,1,2,3"
# -------------------------------------------------------------------


CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent if CODE_ROOT.name == "final_prognosis_code" else CODE_ROOT
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE_ROOT = ROOT / f"final_prognosis_{STAMP}"

DISEASE_SCRIPTS = {
    "DCM": CODE_ROOT / "20260730_dcm_final_prognosis_three_seed.py",
    "HCM": CODE_ROOT / "20260730_hcm_final_prognosis_three_seed.py",
}


def run_disease(disease, visible_gpus):
    disease_root = ARCHIVE_ROOT / disease
    environment = os.environ.copy()
    environment["FINAL_PROGNOSIS_DISEASE_ROOT"] = str(disease_root)
    environment["FINAL_PROGNOSIS_FORCE_RETRAIN"] = (
        "1" if FORCE_RETRAIN else "0"
    )
    environment["FINAL_PROGNOSIS_VISIBLE_GPUS"] = visible_gpus
    print(
        f"\n[{disease}] Starting | retrain={FORCE_RETRAIN} "
        f"| GPUs={visible_gpus}"
    )
    subprocess.run(
        [sys.executable, str(DISEASE_SCRIPTS[disease])],
        cwd=ROOT, env=environment, check=True,
    )


def run_figures(active_diseases):
    environment = os.environ.copy()
    environment["FINAL_PROGNOSIS_ARCHIVE_ROOT"] = str(ARCHIVE_ROOT)
    environment["FINAL_PROGNOSIS_ACTIVE_DISEASES"] = ",".join(
        active_diseases
    )
    print(f"\n[FIGURES] Generating: {', '.join(active_diseases)}")
    subprocess.run(
        [sys.executable, str(CODE_ROOT / "20260730_final_prognosis_figures.py")],
        cwd=ROOT, env=environment, check=True,
    )


def main():
    selected = []
    if RUN_DCM:
        selected.append(("DCM", DCM_VISIBLE_GPUS))
    if RUN_HCM:
        selected.append(("HCM", HCM_VISIBLE_GPUS))
    if not selected:
        raise ValueError("At least one of RUN_DCM or RUN_HCM must be True")

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=False)
    print(f"Unified prognosis archive: {ARCHIVE_ROOT}")
    for disease, visible_gpus in selected:
        run_disease(disease, visible_gpus)
    if RUN_FIGURES:
        run_figures([disease for disease, _ in selected])
    print(f"\nCOMPLETE: {ARCHIVE_ROOT}")


if __name__ == "__main__":
    main()
