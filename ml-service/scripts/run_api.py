from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=False,
    )
