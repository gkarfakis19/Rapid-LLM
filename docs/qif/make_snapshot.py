#!/usr/bin/env python3
"""Regenerate board_snapshot.html: the board with status JSONs inlined (for artifact publishing)."""
import json, pathlib, datetime
root = pathlib.Path(__file__).parent
data = {}
for p in ["P1","P2","P3","P4","P5","P6"]:
    f = root / "status" / (p + ".json")
    data[p] = json.loads(f.read_text()) if f.exists() else None
h = (root / "index.html").read_text()
stamp = datetime.date.today().isoformat()
banner = ('<p style="background:#f6ead2;border:1px solid #e2d4b0;border-radius:6px;'
          'padding:.5rem .8rem;font:13px/1.5 \'Segoe UI\',system-ui,sans-serif">Snapshot of '
          + stamp + '. Plan-page links resolve on the localhost board, not here.</p>')
h = h.replace("<body><main>", "<body><main>" + banner, 1)
h = h.replace("<script>", "<script>window.STATUS_DATA = " + json.dumps(data) + ";</script>\n<script>", 1)
(root / "board_snapshot.html").write_text(h)
print("wrote", root / "board_snapshot.html")
