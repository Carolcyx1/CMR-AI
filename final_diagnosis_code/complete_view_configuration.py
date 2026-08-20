"""Train all 15 view subsets with the final diagnosis implementation.

This is the formal training-based view-configuration ablation.  It delegates
to the locked final diagnosis training code rather than copying an older
network implementation, so every subset differs only in ``selected_views``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "final_diagnosis_code" / "20260725_backbone_baseline_with_r3d18.py"
RUN_ID = f"view15_hiermean_seed42_{datetime.now():%Y%m%d_%H%M%S}"
OUTPUT = ROOT / "final_diagnosis_results" / "view_configuration_runs" / f"view_configuration_{datetime.now():%Y%m%d_%H%M%S}"
WEIGHTS = "/data/projects/MRI/checkpoints/swin3d_t-7615ae03.pth"
VIEWS = ("Cine2CH", "Cine3CH", "Cine4CH", "CineSAX")


def item(name, views):
    return {
        "experiment_name": name,
        "model_type": "videoswin",
        "selected_views": ",".join(views),
        "agg": "hier_mean",
        "pretrained": True,
        "weights_path": WEIGHTS,
    }


EXPERIMENTS = [
    item("view01_2CH_only", (VIEWS[0],)),
    item("view02_3CH_only", (VIEWS[1],)),
    item("view03_4CH_only", (VIEWS[2],)),
    item("view04_SAX_only", (VIEWS[3],)),
    item("view05_2CH_3CH", (VIEWS[0], VIEWS[1])),
    item("view06_2CH_4CH", (VIEWS[0], VIEWS[2])),
    item("view07_2CH_SAX", (VIEWS[0], VIEWS[3])),
    item("view08_3CH_4CH", (VIEWS[1], VIEWS[2])),
    item("view09_3CH_SAX", (VIEWS[1], VIEWS[3])),
    item("view10_4CH_SAX", (VIEWS[2], VIEWS[3])),
    item("view11_2CH_3CH_4CH", (VIEWS[0], VIEWS[1], VIEWS[2])),
    item("view12_2CH_3CH_SAX", (VIEWS[0], VIEWS[1], VIEWS[3])),
    item("view13_2CH_4CH_SAX", (VIEWS[0], VIEWS[2], VIEWS[3])),
    item("view14_3CH_4CH_SAX", (VIEWS[1], VIEWS[2], VIEWS[3])),
    item("view15_full_2CH_3CH_4CH_SAX", VIEWS),
]


def main():
    overrides = {
        "gpu_mode": "ddp", "visible_gpus": "0,1,2,3", "auto_launch_ddp": True,
        "experiment_mode": "compare", "compare_experiments": EXPERIMENTS,
        "cohort_required_views": ",".join(VIEWS), "allow_missing_selected_views": False,
        "seed": 42, "output_dir": str(OUTPUT), "run_id": RUN_ID,
    }
    env = os.environ.copy()
    env["CMR_RUN_CONFIG_OVERRIDES"] = json.dumps(overrides)
    print(f"Final diagnosis implementation: {BASE}")
    print(f"15-combination output root: {OUTPUT}")
    subprocess.run([sys.executable, str(BASE)], check=True, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    main()
