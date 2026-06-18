"""Render .mmd files to PNG via mermaid.ink."""
import base64
import os
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).parent

for mmd in sorted(HERE.glob("*.mmd")):
    with open(mmd, encoding="utf-8") as f:
        code = f.read()
    b64 = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{b64}?type=png&bgColor=FFFFFF"
    out = mmd.with_suffix(".png")
    print(f"Rendering {mmd.name} -> {out.name} ({len(code)} chars)")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        out.write_bytes(data)
        print(f"  OK {len(data)} bytes")
    except Exception as e:
        print(f"  FAILED: {e}")
