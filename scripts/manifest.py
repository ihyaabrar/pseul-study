"""Write or check MANIFEST_SHA256.csv, the hash of every tracked file.

    python scripts/manifest.py          # check the files against the manifest
    python scripts/manifest.py --write  # rebuild it after an intentional change

.gitattributes turns off line-ending conversion, so a clone on any platform
has the same bytes as the files that were hashed.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.csv"


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.splitlines() if p and p != MANIFEST.name)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if "--write" in sys.argv:
        files = tracked_files()
        with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["path", "sha256", "bytes"])
            for rel in files:
                p = ROOT / rel
                writer.writerow([rel, digest(p), p.stat().st_size])
        print(f"wrote {len(files)} entries to {MANIFEST.name}")
        return 0

    with MANIFEST.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r["path"] for r in rows
           if not (ROOT / r["path"]).exists() or digest(ROOT / r["path"]) != r["sha256"]]
    for rel in bad:
        print(f"MISMATCH {rel}")
    print(f"{len(rows) - len(bad)}/{len(rows)} files match")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
