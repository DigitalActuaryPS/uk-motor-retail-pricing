"""Reflection checks. Writes data/processed/verify_report.json and exits non-zero on failure."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retailpricing.core import paths  # noqa: E402
from retailpricing.core.config import load_config  # noqa: E402
from retailpricing.reporting import build  # noqa: E402

results: dict[str, dict] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + (": " + detail if detail else ""))


# 1. Every README and executive summary number matches executed output
reg = build.Registry()
v = build.collect(reg)
for fname, render in (("docs/executive_summary.md", build.render_exec), ("README.md", build.render_readme)):
    on_disk = (ROOT / fname).read_text(encoding="utf-8")
    check(f"{fname} equals re-render from current outputs", on_disk == render(v))
saved = json.loads((ROOT / "docs" / "number_registry.json").read_text())
check("number registry matches recomputed values", {k: x["text"] for k, x in saved.items()} == {k: x["text"] for k, x in reg.items.items()})

ALLOWED = {"95", "6", "0", "1", "2", "3", "4", "5", "8", "9", "10", "15", "30", "13", "14", "R01", "6B", "2A", "4", "12", "22"}
for fname, render in (("docs/executive_summary.md", build.render_exec), ("README.md", build.render_readme)):
    text = render(v)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = "\n".join(l for l in text.splitlines() if not l.startswith("|"))
    for x in sorted(reg.items.values(), key=lambda i: -len(i["text"])):
        text = text.replace(x["text"], "")
    text = re.sub(r"`[^`]*`", "", text)
    left = [t for t in re.findall(r"[-+]?\d[\d,.]*%?", text) if t.strip("+-,.%") not in ALLOWED and t.strip(",.") not in ALLOWED]
    check(f"{fname} has no unregistered numbers", not left, ", ".join(left[:15]))

# 2. Claims-based targets absent from model fitting code
bad = re.compile(r"claim|severity|frequency|loss[_ ]?cost|pure[_ ]?premium", re.I)
hits = []
for sub in ("conversion", "retention", "elasticity", "optimisation"):
    for f in (ROOT / "src" / "retailpricing" / sub).glob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if bad.search(line) and not line.strip().startswith(("#", '"""', "'''")):
                hits.append(f"{f.relative_to(ROOT)}:{i}:{line.strip()[:90]}")
check("no claims, severity, frequency or loss terms in model-fitting code (non-comment lines)", not hits, "; ".join(hits[:6]))

# 3. Config statuses
cfg = load_config()
rows = list(cfg.iter_params())
missing = [p for p, n in rows if not (isinstance(n, dict) and n.get("status") and "source_note" in n and "value" in n)]
check("every config parameter has value, status and source_note", bool(rows) and not missing, f"{len(rows)} parameters" + (f"; missing {missing[:5]}" if missing else ""))
problems = cfg.validate()
check("config validate() reports no problems", not problems, "; ".join(problems[:5]))

# 4. gitignore
gi = (ROOT / ".gitignore").read_text()
check(".gitignore excludes manual quotes file", "data/manual/manual_quotes.csv" in gi)
tpl = (ROOT / "data" / "manual" / "manual_quotes_template.csv").read_text().strip().splitlines()
check("manual template is header only", len(tpl) == 1)
check("no manual quotes file shipped", not (ROOT / "data" / "manual" / "manual_quotes.csv").exists())

# 5. No em or en dashes in prose and code
dash = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")
offenders = []
for pat in ("*.md", "*.yaml", "*.py", "*.sql", "*.ipynb", "*.toml", "*.txt"):
    for f in ROOT.rglob(pat):
        if any(p in f.parts for p in (".git", "__pycache__", ".pytest_cache", "venv", ".venv", "node_modules")):
            continue
        if dash.search(f.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(f.relative_to(ROOT)))
check("no em or en dashes", not offenders, ", ".join(offenders[:8]))

# 6. TODO and placeholder scan
ph = re.compile(r"\bTODO\b|\bFIXME\b|NotImplementedError|lorem ipsum|placeholder", re.I)
ph_hits = []
for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "app").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
    if f.name == "verify.py":
        continue
    for i, line in enumerate(f.read_text().splitlines(), 1):
        if ph.search(line):
            ph_hits.append(f"{f.relative_to(ROOT)}:{i}")
check("no TODO, FIXME or stub markers", not ph_hits, ", ".join(ph_hits[:6]))

# 7. Determinism on the full-size dataset: two fresh generations with the same seed
code = (
    "import sys;sys.path.insert(0,'src');from retailpricing.core.config import load_config;"
    "from retailpricing.data_gen.generator import generate, frame_hash;"
    "c=load_config();d,_=generate(c);print(frame_hash(d))"
)
hs = [subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()[-1:] for _ in range(2)]
stored = json.loads((paths.data_processed() / "m1_summary.json").read_text())["sha256"]
check("same-seed full-size generator runs give identical hashes", hs[0] == hs[1] and hs[0] == [stored], f"{hs[0]} {hs[1]} stored={stored[:16]}")

# 8. Em-dash-free, status counts and statement of exit
out = ROOT / "data" / "processed" / "verify_report.json"
out.write_text(json.dumps(results, indent=1))
failed = [k for k, r in results.items() if not r["pass"]]
print(f"\n{len(results) - len(failed)} of {len(results)} checks passed")
sys.exit(1 if failed else 0)
