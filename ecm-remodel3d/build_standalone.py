#!/usr/bin/env python3
"""Bundle viewer/index.html + viewer/scene.json into one double-clickable file.

Produces viewer/standalone.html with the scene embedded (no fetch), so it opens
directly from the file system — no local server needed. three.js still loads
from its CDN, so an internet connection is required the first time.

    python build_standalone.py
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "viewer")


def main():
    html = open(os.path.join(V, "index.html")).read()
    scene = open(os.path.join(V, "scene.json")).read()

    # 1) embed the scene as a global, just before the module script
    inject = ('<script>window.__SCENE__ = ' + scene + ';</script>\n'
              '<script type="module">')
    html = html.replace('<script type="module">', inject, 1)

    # 2) replace the network fetch with the embedded data
    fetch_block = """  let data;
  try {
    const res = await fetch('scene.json', {cache:'no-store'});
    if (!res.ok) throw new Error('scene.json '+res.status);
    data = await res.json();
  } catch(e){ showErr('scene.json 로드 실패: '+e.message+'\\n(로컬이면 http.server 로 서빙)'); return; }"""
    assert fetch_block in html, "fetch block not found — viewer/index.html changed?"
    html = html.replace(fetch_block, "  let data = window.__SCENE__;")

    # 3) drop the file:// warning (standalone works from file://)
    file_warn = """  if (location.protocol === 'file:') {
    document.getElementById('err').innerHTML =
      '⚠ 파일을 더블클릭(file://)으로 열면 브라우저 보안 때문에 3D가 로드되지 않습니다.<br>' +
      '터미널에서 &nbsp;<b>cd viewer &amp;&amp; python serve.py</b>&nbsp; 실행 후 ' +
      '열리는 &nbsp;<b>http://localhost:8000</b>&nbsp; 으로 보세요.';
    document.querySelector('.hint').style.display = 'none';
  }
"""
    html = html.replace(file_warn, "")

    # 4) run the CDN watchdog regardless of protocol
    html = html.replace("if (location.protocol !== 'file:') {\n    window.__bootTimer",
                        "if (true) {\n    window.__bootTimer")

    # 5) update the hint line
    html = html.replace(
        'scene.json 을 불러옵니다 · 로컬에서 열면 <code>python -m http.server</code> 로 서빙하세요',
        '단일 파일 뷰어 · three.js는 인터넷에서 로드됩니다')

    out = os.path.join(V, "standalone.html")
    open(out, "w").write(html)
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({kb:.0f} KB) — double-click to open")


if __name__ == "__main__":
    main()
