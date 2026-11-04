"""T-2344 dry run: world_plan zones for the given countries on a scratch copy of the plan with kyi, kha, ode
unheld (as the live step will see them). Usage: ua_dry.py <out.json> <ISO3,ISO3>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import world_plan as wp  # noqa: E402

out = Path(sys.argv[1])
isos = sys.argv[2].split(",")
doc = json.loads(wp.OUT.read_text(encoding="utf-8"))
for r in doc["metros"]:
    if r.get("id") in ("kyi", "kha", "ode") and r.get("status") == "held":
        r["status"] = "planned"
        r.pop("hold_reason", None)
copy = out.with_name(out.stem + ".base.json")
copy.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
wp.OUT = copy
wp.plan_zones(isos, str(out))
