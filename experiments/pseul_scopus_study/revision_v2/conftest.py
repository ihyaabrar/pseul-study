"""Put the repository root on sys.path so the tests import the frozen method.

test_revision_v2.py imports it as `scripts.pseul`, which resolves only when the
repository root is importable. Sitting beside the tests, this conftest is picked
up whatever directory pytest is invoked from; a root-level one is skipped when
pytest infers a rootdir further down the tree.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
