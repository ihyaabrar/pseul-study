"""Download the two CDC datasets into data/ and check for the two manual ones.

NHANES 2021-2023 and BRFSS 2024 are fetched from CDC with the same functions the
study used, so the files land where the loaders in
experiments/pseul_scopus_study/revision_v2/study_registry.py look for them.
The medication-adherence and Diabetes 130 files need a manual download; this
script only reports whether they are in place.

    python scripts/fetch_public_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "pseul_scopus_study"))

from run_pseul_brfss_diabetes_study import download_brfss  # noqa: E402
from run_pseul_nhanes_diabetes_study import read_xpt  # noqa: E402

NHANES_FILES = ["DEMO_L", "BMX_L", "BPXO_L", "DIQ_L", "GHB_L", "GLU_L",
                "TCHOL_L", "HDL_L", "PAQ_L", "SMQ_L"]

MANUAL = {
    "Medication adherence (Mendeley Data, doi:10.17632/zkp7sbbx64.2)":
        ROOT / "data" / "Final Prepared Dataset - Diabetes and Hypertension Data.xlsx",
    "Diabetes 130-US Hospitals (UCI, doi:10.24432/C5230J)":
        ROOT / "data" / "diabetes130" / "diabetic_data.csv",
}


def main() -> int:
    for name in NHANES_FILES:
        rows = len(read_xpt(name))
        print(f"NHANES {name}: {rows} rows")
    print(f"BRFSS 2024: {download_brfss()}")

    missing = 0
    for label, path in MANUAL.items():
        if path.exists():
            print(f"found    {label}")
        else:
            missing += 1
            print(f"MISSING  {label}\n         put it at {path.relative_to(ROOT)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
