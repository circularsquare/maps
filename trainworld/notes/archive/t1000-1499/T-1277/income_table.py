"""T-1277: write app/src/game/countryIncome.json, GDP per head at PPP by country (ISO 3166 alpha-2),
from the world field's country table (World Bank NY.GDP.PCAP.PP.KD). Rerun after
world_field_build.py.

    python trainworld/notes/T-1277/income_table.py   (from the maps root)"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = json.load(open(ROOT / "data/work/world_field.countries.json", encoding="utf-8"))
gdp = {c["iso2"]: round(c["gdp"]) for c in src["countries"] if c.get("iso2") and c.get("gdp")}
srcs = sorted({c.get("gdp_src") for c in src["countries"] if c.get("gdp")} - {None})
out = {"note": "GDP per head at PPP, constant 2021 international $ (T-1277)", "sources": srcs, "gdp": dict(sorted(gdp.items()))}
p = ROOT / "app/src/game/countryIncome.json"
p.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
print(f"{len(gdp)} countries -> {p}")
