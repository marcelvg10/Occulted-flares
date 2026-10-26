"""Run offline analytical/storage tests from any working directory."""
from pathlib import Path
import os
import sys
import unittest

WORK = Path(__file__).resolve().parents[1]
SOURCE = str(WORK / "src")
sys.path.insert(0, SOURCE)
# Child CLI checks use the same source checkout without an editable install.
os.environ["PYTHONPATH"] = SOURCE + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(WORK / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
