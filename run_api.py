"""Root launcher for DeepTrace AI FastAPI server."""

from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if (root / "deeptrace").exists():
    sys.path.insert(0, str(root))
elif (root / "DeepTrace-AI" / "deeptrace").exists():
    sys.path.insert(0, str(root / "DeepTrace-AI"))

import socket
import uvicorn
from deeptrace.api.app import app

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

if __name__ == "__main__":
    target_port = 8000
    if is_port_in_use(target_port):
        print(f"⚠️ Port {target_port} is already in use by another process.")
        # Find next free port
        for p in range(8001, 8020):
            if not is_port_in_use(p):
                target_port = p
                print(f"✅ Switching to available port: {target_port}")
                break

    print(f"🚀 Launching DeepTrace API at http://127.0.0.1:{target_port}")
    print(f"📖 Swagger Docs available at http://127.0.0.1:{target_port}/docs")
    uvicorn.run(app, host="127.0.0.1", port=target_port, log_level="info")
