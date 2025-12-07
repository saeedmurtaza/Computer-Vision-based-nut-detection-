# TODO: implement
# launcher/launcher.py — O-HIVE Clean Project Launcher
import os
import sys
import socket
import webbrowser
import threading
import time
import urllib.request
import traceback

import logging
from pathlib import Path

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
LOG_DIR = Path(os.getenv("LOCALAPPDATA", Path.home())) / "O-HIVE" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "launcher.log"

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


# ---------------------------------------------------------
# Basler Pylon Runtime Discovery (PyInstaller compatible)
# ---------------------------------------------------------
def _add_pylon_runtime_dirs():
    """
    Make Pylon DLLs discoverable.
    Works when packaged as .exe or windowed Python app.
    """
    candidates = [
        r"C:\Program Files\Basler\pylon 8\Runtime\x64",
        r"C:\Program Files\Basler\pylon 7\Runtime\x64",
        r"C:\Program Files\Basler\pylon 6\Runtime\x64",
    ]
    for d in candidates:
        if os.path.isdir(d):
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                log(f"Added Pylon directory: {d}")
            except Exception as e:
                log(f"Failed to add Pylon directory {d}: {e}")

_add_pylon_runtime_dirs()


# ---------------------------------------------------------
# Uvicorn Logging (avoid STDOUT issues in packaged apps)
# ---------------------------------------------------------
def uvicorn_log_config():
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": str(LOG_PATH),
                "mode": "a",
                "encoding": "utf-8",
                "formatter": "standard",
                "level": "INFO",
            }
        },
        "loggers": {
            "uvicorn":        {"handlers": ["file"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"handlers": ["file"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["file"], "level": "INFO", "propagate": False},
        },
    }


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def port_free(port: int) -> bool:
    """Returns True if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) != 0


def healthcheck(port: int) -> bool:
    """Checks /api/status endpoint availability."""
    url = f"http://127.0.0.1:{port}/api/status"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------
# Launch FastAPI Server
# ---------------------------------------------------------
def run_server(port: int):
    try:
        import uvicorn
        from app.main import app

        log(f"Starting uvicorn at http://127.0.0.1:{port}")
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            reload=False,
            log_level="info",
            access_log=True,
            log_config=uvicorn_log_config(),
        )
    except Exception:
        log("Uvicorn crashed:\n" + traceback.format_exc())


def _message_box(title: str, text: str):
    """Windows popup for errors (used only on failure)."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)
    except Exception:
        pass


# ---------------------------------------------------------
# Launcher main
# ---------------------------------------------------------
def main():
    # Choose available port
    port = next((p for p in range(8000, 8011) if port_free(p)), 8000)
    url = f"http://127.0.0.1:{port}/"

    log("=" * 60)
    log(f"Launcher starting — port candidate: {port}")

    # Start FastAPI in background
    t = threading.Thread(target=run_server, args=(port,), daemon=True)
    t.start()

    # Wait for TCP server
    deadline = time.time() + 30
    while time.time() < deadline:
        if not port_free(port):
            log("TCP listener ready.")
            break
        time.sleep(0.2)
    else:
        log("Server did not open TCP port. Aborting.")
        _message_box("O-HIVE Launcher", f"Server failed to start.\nLog: {LOG_PATH}")
        return

    # Wait for /api/status
    deadline = time.time() + 30
    opened = False
    while time.time() < deadline:
        if healthcheck(port):
            log("Healthcheck OK. Opening browser.")
            try:
                webbrowser.open(url)
            except Exception:
                pass
            opened = True
            break
        time.sleep(0.3)

    if not opened:
        log("Healthcheck failed; opening main page anyway.")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    # Keep alive while server thread runs
    try:
        while t.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("Launcher crashed:\n" + traceback.format_exc())
        _message_box("O-HIVE Launcher", f"Launcher crashed.\nLog: {LOG_PATH}")
