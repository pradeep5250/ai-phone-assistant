"""
realtime_ui.py — Standalone 1-Click Launcher for OpenAI Realtime Voice Studio UI

Usage:
    python realtime_ui.py
    or
    python realtime_ui.py --port 8000
"""

import os
import sys
import time
import webbrowser
import threading
import argparse
import uvicorn
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))


def open_browser_delayed(url: str, delay_sec: float = 1.2):
    """Open default web browser after server starts."""
    time.sleep(delay_sec)
    print(f"🚀 Opening browser at: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[!] Could not automatically open browser: {e}")


def print_banner(host: str, port: int):
    voice = os.getenv("OPENAI_VOICE", "marin")
    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
    url = f"http://{host}:{port}/realtime"

    print("\n" + "=" * 68)
    print("  🎙️  MINTX SOLUTIONS AI — REALTIME VOICE STUDIO  🎙️")
    print("=" * 68)
    print(f"  🏢 Company:       Mintx Solutions")
    print(f"  ⚡ Model:         {model}")
    print(f"  🔊 Default Voice: {voice}")
    print(f"  🌐 Studio UI:     {url}")
    print("=" * 68)
    if not OPENAI_API_KEY:
        print("\n  ⚠️  WARNING: OPENAI_API_KEY is not set in .env file!")
    else:
        print("  ✅ OpenAI API Key loaded successfully.")
    print("  👉 Press Ctrl+C in this terminal to stop the server.\n" + "=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Mintx Solutions AI Realtime Voice Studio UI")
    parser.add_argument("--host", default=SERVER_HOST, help=f"Host address (default: {SERVER_HOST})")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help=f"Port number (default: {SERVER_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open the browser")
    args = parser.parse_args()

    host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{host}:{args.port}/realtime"

    print_banner(host, args.port)

    if not args.no_browser:
        threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()

    from realtime_server import app
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
