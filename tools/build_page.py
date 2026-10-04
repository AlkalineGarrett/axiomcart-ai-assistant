"""
Rebuild graph-walkthrough.html from the current sources.

    python tools/build_page.py                 # embed src/ + data.py
    python tools/build_page.py --capture artifacts/trace.json   # also bake in a real run

The page is self-contained: every source file, the demo data, and
optionally a captured trace are inlined between marker comments, so it
works offline from file://. Re-run this after editing anything in src/.

Note: the Graph Execution tab pins line numbers, so if you move code
around inside src/, check the highlighted ranges in STEPS still line up.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "graph-walkthrough.html"
NAMES = ["graph.py", "nodes.py", "state.py", "tools.py", "data.py", "main.py", "rag.py", "config.py", "voice.py"]
WANTED = ["PRODUCT_CATALOG", "ORDER_DATABASE", "SUPPORT_POLICIES"]


def read_sources() -> dict[str, str]:
    return {f"src/{n}": (ROOT / "src" / n).read_text(encoding="utf-8") for n in NAMES}


def parse_data(source: str) -> dict:
    """Lift the constants out of data.py without importing it (no deps needed)."""
    data = {}
    for node in ast.parse(source).body:
        targets = getattr(node, "targets", None) or (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for t in targets:
            if not isinstance(t, ast.Name) or t.id not in WANTED:
                continue
            v = node.value
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "strip":
                data[t.id] = ast.literal_eval(v.func.value).strip()   # SUPPORT_POLICIES
            else:
                data[t.id] = ast.literal_eval(v)
    missing = [w for w in WANTED if w not in data]
    if missing:
        sys.exit(f"could not parse from data.py: {missing}")
    return data


def inject(html: str, marker: str, payload: str) -> str:
    # A "</script>" anywhere in the payload would close the tag early. Inside a
    # JSON literal every "/" lives in a string, so escaping it is always safe.
    payload = payload.replace("</", "<\\/")
    html, count = re.subn(
        rf"/\*__{marker}__\*/.*?/\*__END_{marker}__\*/",
        lambda m: f"/*__{marker}__*/{payload}/*__END_{marker}__*/",   # function repl: used literally
        html,
        flags=re.S,
    )
    if count != 1:
        sys.exit(f"__{marker}__ marker found {count} times in {PAGE.name}, expected exactly 1")
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild the self-contained walkthrough page")
    ap.add_argument("--capture", metavar="TRACE.JSON",
                    help="bake a captured run into the Prompt Trace tab (default: artifacts/trace.json if present)")
    ap.add_argument("--no-capture", action="store_true", help="strip any baked-in capture")
    args = ap.parse_args()

    files = read_sources()
    data = parse_data(files["src/data.py"])
    html = PAGE.read_text(encoding="utf-8")

    html = inject(html, "FILES", json.dumps(files, ensure_ascii=False, indent=0))
    html = inject(html, "DATA", json.dumps(data, ensure_ascii=False, indent=0))
    print(f"embedded {len(files)} sources, {len(data['PRODUCT_CATALOG'])} products, "
          f"{len(data['ORDER_DATABASE'])} orders")

    cap_path = None
    if not args.no_capture:
        cap_path = pathlib.Path(args.capture) if args.capture else ROOT / "artifacts" / "trace.json"
        if not cap_path.exists():
            if args.capture:
                sys.exit(f"no such capture: {cap_path}")
            cap_path = None

    if cap_path:
        cap = json.loads(cap_path.read_text(encoding="utf-8"))
        if not isinstance(cap.get("events"), list):
            sys.exit(f"{cap_path} has no 'events' array — is it from src.trace_capture?")
        html = inject(html, "CAPTURE", json.dumps(cap, ensure_ascii=False))
        m = cap.get("meta", {})
        print(f"baked in {cap_path.name}: {len(cap['events'])} events, "
              f"{m.get('llm_calls', '?')} LLM calls, query {m.get('query', '?')!r}")
    else:
        html = inject(html, "CAPTURE", "null")
        print("no capture baked in (run src.trace_capture to make one)")

    PAGE.write_text(html, encoding="utf-8")
    print(f"-> {PAGE}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
