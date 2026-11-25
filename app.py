"""DeepTrace AI - Unified Host-Ready Application Launcher.

Runs the complete DeepTrace AI system:
- Streamlit interactive forensics dashboard
- Auto-starts FastAPI backend on a free port or runs unified in-process
- No socket collisions, host-ready for local or cloud deployment.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if (ROOT / "DeepTrace-AI").exists() and str(ROOT / "DeepTrace-AI") not in sys.path:
    sys.path.insert(0, str(ROOT / "DeepTrace-AI"))


def find_free_port(preferred: int = 8000, max_tries: int = 20) -> int:
    """Find a free TCP port starting with preferred."""
    for p in range(preferred, preferred + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    # Fallback: let OS assign any available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_api_background(port: int) -> threading.Thread:
    """Start FastAPI server in a background daemon thread."""
    import uvicorn
    from deeptrace.api.app import app

    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return t


def main():
    print("=" * 60)
    print("🛡️  DeepTrace AI: Multimodal Forensic Investigation System")
    print("=" * 60)

    # 1. Start FastAPI API on a free port
    api_port = find_free_port(8000)
    try:
        start_api_background(api_port)
        time.sleep(0.5)
        print(f"✅ REST API running at:     http://localhost:{api_port}")
        print(f"📖 Swagger Docs at:        http://localhost:{api_port}/docs")
    except Exception as e:
        print(f"⚠️  Could not start background API: {e}")

    # 2. Launch Streamlit Dashboard
    dashboard_file = ROOT / "app_streamlit.py"
    if not dashboard_file.exists():
        dashboard_file = ROOT / "DeepTrace-AI" / "app_streamlit.py"

    dashboard_port = os.environ.get("PORT", "8501")
    print(f"🚀 Forensic Dashboard at:  http://localhost:{dashboard_port}")
    print("=" * 60)
    print("Press Ctrl+C to stop.\n")

    # Run Streamlit directly using sys.executable
    import subprocess
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_file),
        f"--server.port={dashboard_port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 DeepTrace AI shut down.")


if __name__ == "__main__":
    main()
