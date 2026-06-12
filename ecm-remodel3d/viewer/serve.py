#!/usr/bin/env python3
"""One command to view the 3D scene.

    python serve.py            # serves this folder and opens the browser

Why this is needed: index.html uses ES modules and fetches scene.json. Browsers
block both when a page is opened by double-click (file://), so the page would be
blank. Serving over http (what this does) fixes it.
"""

from __future__ import annotations

import http.server
import os
import socketserver
import threading
import webbrowser

PORT = int(os.environ.get("PORT", "8000"))


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map.setdefault(".json", "application/json")
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print(f"serving the 3D viewer at  {url}")
        print("press Ctrl+C to stop")
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
