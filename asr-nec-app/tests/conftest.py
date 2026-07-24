from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.mkdtemp(prefix="asr-nec-app-tests-"))
atexit.register(shutil.rmtree, TEST_ROOT, True)

os.environ["DATABASE_URL"] = f"sqlite:///{(TEST_ROOT / 'test.db').as_posix()}"
os.environ["STORAGE_DIR"] = str(TEST_ROOT / "storage")
os.environ["UPLOAD_DIR"] = str(TEST_ROOT / "storage" / "uploads")
os.environ["FRONTEND_DIST_DIR"] = str(TEST_ROOT / "frontend-not-built")
os.environ["CORS_ORIGINS"] = "http://localhost:5009"
os.environ["NEC_SKIP_MODEL_LOAD"] = "true"
os.environ["NEC_RUNS_DIR"] = str(TEST_ROOT / "runs")
os.environ["EXAMPLES_FILE"] = str(TEST_ROOT / "examples.json")
os.environ["EXAMPLES_AUDIO_DIR"] = str(TEST_ROOT / "examples-audio")
