"""Search aliases for every registered zone (T-825): the biggest cities and towns in its place list
(`data/packs/<id>.places.json`, OSM via places.py) with their other names, so "The Hague" or "Den
Haag" finds the zone holding it. Writes app/src/game/zone-aliases.json ({zone id: [names]}),
bundled with the app; the hand-edited display names and extra aliases are zone-names.json beside
it, which this never touches. Rerun after a join adds zones (seconds):

    ..\\venv\\Scripts\\python.exe pipeline\\zone_aliases.py
"""
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "app" / "src" / "game" / "registry.json"
PACKS = ROOT / "data" / "packs"
OUT = ROOT / "app" / "src" / "game" / "zone-aliases.json"

KINDS = {"city", "town"}
MIN_POP = 20_000  # a town people would name a start after
MAX_PLACES = 6  # per zone, biggest first; keeps the bundled file about 60 bytes a zone
# OSM alt names that are jokes or vandalism rather than names anyone searches by
NOT_NAMES = {"Swiftkirchen"}


def fold(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).casefold().strip()


def aliases(zone_id, own_name):
    path = PACKS / f"{zone_id}.places.json"
    if not path.exists():
        return []
    f = json.loads(path.read_text(encoding="utf-8"))
    kinds = f["kinds"]
    rows = [p for p in f["places"] if kinds[p[1]] in KINDS and p[4] >= MIN_POP]
    rows.sort(key=lambda p: -p[4])
    seen, out = {fold(own_name)}, []
    for p in rows[:MAX_PLACES]:
        for name in [p[0], *(p[5] if len(p) > 5 and p[5] else [])]:
            k = fold(name)
            if name in NOT_NAMES or not k or k in seen:
                continue
            seen.add(k)
            out.append(name)
    return out


def main():
    cities = json.loads(REGISTRY.read_text(encoding="utf-8"))["cities"]
    table = {c["id"]: a for c in cities if (a := aliases(c["id"], c["name"]))}
    text = "{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in table.items()) + "\n}\n"
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(OUT)
    print(f"{len(table)} of {len(cities)} zones with aliases, {sum(map(len, table.values()))} names, {len(text.encode()):,} bytes -> {OUT}")


if __name__ == "__main__":
    main()
