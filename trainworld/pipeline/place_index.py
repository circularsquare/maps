"""T-4004: the world name index the search box finds far places with, made from every registered zone's
place list (`data/packs/<id>.places.json`, places.py). No new data: the same rows, regrouped by name.

    ..\\venv\\Scripts\\python.exe -I pipeline\\place_index.py [--force]

Writes `data/packs/place-index/` whole (built as `place-index.part/`, then swapped in), so a reader never
sees half of it. By hand it refuses while a world run is live (a join compares data/packs), unless
--force; the runner runs it after each join (world_run.py), when the zones' place lists may have changed.

A search fetches one shard, not the world. Each word of a place's names (normalised exactly as the
app's `normalise`, game/nameMatch.ts) is keyed by its first KEY_LEN letters (2 for CJK, kana and
Hangul, where two characters are about a word); a word shorter than that is keyed whole, but only for
a name with no longer word; a name of several words is keyed joined too ("santafe"). A key with more
than CAP rows is "deep": its own shard keeps the rows named just that (up to CAP) and the TOP others
by weight (kind, then population), and its rows go one letter further down (a short word's key, never
split, is cut the same way: Burmese, which the normalising splits at every vowel sign). Keys are hashed (FNV-1a
over code points) into BUCKETS files per tier:

    place-index/index.json      {"format": 1, "buckets": N, "kinds": [...], "tiers": {"a": [kinds], "b": [kinds]},
                                 "keyLen": 3, "joined": true, "deep": {"a": [keys], "b": [keys]}, ...}
    place-index/a/<bucket>.json {"rows": [[name, kind index, lng, lat, population or 0, zone id, [other names]?], ...]}

Tier a is cities down to villages, fetched first; tier b neighbourhoods and hamlets. Data (c)
OpenStreetMap contributors, ODbL.
"""
import datetime
import json
import os
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "app" / "src" / "game" / "registry.json"
PACKS = ROOT / "data" / "packs"
OUT = PACKS / "place-index"

KINDS = ["city", "town", "borough", "suburb", "quarter", "village", "neighbourhood", "hamlet"]
TIERS = {"a": ["city", "town", "borough", "suburb", "village"], "b": ["quarter", "neighbourhood", "hamlet"]}
# search.ts PLACE_WEIGHT: equal matches, bigger kinds first, then by population
WEIGHT = {"city": 1e8, "town": 1e7, "borough": 1e7, "suburb": 5e6, "quarter": 4e6, "village": 3e6, "neighbourhood": 2e6, "hamlet": 0}
BUCKETS = 2048
KEY_LEN, WIDE_KEY_LEN = 3, 2
CAP, TOP = 2000, 400
PLAIN = {"ł": "l", "ø": "o", "đ": "d", "ħ": "h", "ı": "i", "ß": "ss", "æ": "ae", "œ": "oe", "þ": "th"}


def normalise(s):
    """The app's normalise (game/nameMatch.ts), step for step: NFD, U+0300-U+036F dropped, lower case,
    the letters NFD does not split, & as "and", every run of non-letters and non-digits one space. A
    character this Python's Unicode does not know (Cn) is kept as a letter: they are newer CJK ideographs,
    letters to the browser."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not 0x300 <= ord(c) <= 0x36F).lower()
    s = "".join(PLAIN.get(c, c) for c in s).replace("&", " and ")
    return " ".join("".join(c if unicodedata.category(c)[0] in "LN" or unicodedata.category(c) == "Cn" else " " for c in s).split())


def key_len(word):
    return WIDE_KEY_LEN if ord(word[0]) >= 0x2E80 else KEY_LEN


def fnv(key):
    h = 2166136261
    for c in key:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    return h


def words(names):
    """The words a place is keyed by: every word of any of its names long enough for a key, or, for a
    name with none, its short words whole; and a name of several words also joined ("santafe"), for a query
    whose own words are all too common or short. With them, its whole names, normalised and joined."""
    out, whole = set(), set()
    for n in names:
        key = normalise(n)
        ws = [w for w in key.split(" ") if w]
        long = [w for w in ws if len(w) >= key_len(w)]
        out.update(long or ws)
        if len(ws) > 1:
            out.add("".join(ws))
        whole.update((key, "".join(ws)))
    return out, whole


def best(ids, k, rows, cap):
    """Of rows sharing key k: those whose whole name is k first (up to CAP), then the best `cap` others."""
    named = sorted((r for r in ids if k in rows[r][3]), key=lambda r: -rows[r][1])[:CAP]
    return named + sorted((r for r in ids if k not in rows[r][3]), key=lambda r: -rows[r][1])[:cap]


def place(postings, plen, rows, deep, buckets):
    """Postings (row, word) sharing their first plen - 1 letters, grouped by the first plen."""
    groups = defaultdict(list)
    for r, w in postings:
        groups[w[:plen]].append((r, w))
    for k, g in groups.items():
        ids = {r for r, _ in g}
        if len(ids) <= CAP:
            buckets[fnv(k) % BUCKETS].update(ids)
            continue
        deep.add(k)
        exact = {r for r, w in g if len(w) <= plen}
        top = sorted(ids, key=lambda r: -rows[r][1])[:TOP]
        buckets[fnv(k) % BUCKETS].update(best(exact, k, rows, TOP), top)
        place([(r, w) for r, w in g if len(w) > plen], plen + 1, rows, deep, buckets)


def live_runs():
    sys.path.insert(0, str(Path(__file__).parent))
    import freeze
    return freeze.live_runs(ROOT)


def build():
    zones = [c["id"] for c in json.loads(REGISTRY.read_text(encoding="utf-8"))["cities"]]
    tier_of = {k: t for t, ks in TIERS.items() for k in ks}
    rows = {t: [] for t in TIERS}  # per tier: (row as written, weight, words)
    missing = 0
    for z in zones:
        path = PACKS / f"{z}.places.json"
        if not path.exists():
            missing += 1
            continue
        f = json.loads(path.read_text(encoding="utf-8"))
        for p in f["places"]:
            kind = f["kinds"][p[1]]
            others = p[5] if len(p) > 5 and p[5] else []
            row = [p[0], KINDS.index(kind), round(p[2], 4), round(p[3], 4), p[4], z] + ([others] if others else [])
            rows[tier_of[kind]].append((row, WEIGHT[kind] + p[4], *words([p[0], *others])))
    tmp = PACKS / "place-index.part"
    if tmp.exists():
        shutil.rmtree(tmp)
    man = {"format": 1, "buckets": BUCKETS, "kinds": KINDS, "tiers": TIERS, "keyLen": KEY_LEN, "wideKeyLen": WIDE_KEY_LEN, "joined": True,
           "built": datetime.date.today().isoformat(), "zones": len(zones) - missing,
           "source": "OpenStreetMap contributors, ODbL (the zones' place lists)", "deep": {}, "places": {}}
    sizes = {}
    for t, rs in rows.items():
        deep, buckets = set(), defaultdict(set)
        short, longer = [], []
        for i, (_, _, ws, _) in enumerate(rs):
            for w in ws:
                (longer if len(w) >= key_len(w) else short).append((i, w))
        by_word = defaultdict(set)
        for i, w in short:
            by_word[w].add(i)
        for w, ids in by_word.items():  # a short word is its own key, never split: past CAP, as a deep key's exact rows
            buckets[fnv(w) % BUCKETS].update(ids if len(ids) <= CAP else best(ids, w, rs, TOP))
        wide = [(i, w) for i, w in longer if key_len(w) == WIDE_KEY_LEN]
        place([(i, w) for i, w in longer if key_len(w) == KEY_LEN], KEY_LEN, rs, deep, buckets)
        place(wide, WIDE_KEY_LEN, rs, deep, buckets)
        d = tmp / t
        d.mkdir(parents=True)
        sz = []
        for b in range(BUCKETS):
            ids = sorted(buckets.get(b, ()), key=lambda i: -rs[i][1])
            text = json.dumps({"rows": [rs[i][0] for i in ids]}, ensure_ascii=False, separators=(",", ":"))
            (d / f"{b}.json").write_text(text, encoding="utf-8")
            sz.append((len(text.encode()), len(ids)))
        man["deep"][t] = sorted(deep)
        man["places"][t] = len(rs)
        sizes[t] = sz
    (tmp / "index.json").write_text(json.dumps(man, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    old = PACKS / "place-index.old"
    if old.exists():
        shutil.rmtree(old)
    if OUT.exists():
        os.replace(OUT, old)
    os.replace(tmp, OUT)
    if old.exists():
        shutil.rmtree(old)
    for t, sz in sizes.items():
        b = sorted(s for s, _ in sz)
        print(f"tier {t}: {man['places'][t]:,} places, {len(man['deep'][t])} deep keys, {BUCKETS} shards, {sum(b) / 1e6:.0f} MB; "
              f"shard median {b[len(b) // 2] / 1e3:.0f} KB, 99th {b[int(len(b) * .99)] / 1e3:.0f} KB, largest {b[-1] / 1e3:.0f} KB "
              f"({max(n for _, n in sz):,} rows)", flush=True)
    print(f"{man['zones']} zones ({missing} with no place list) -> {OUT}", flush=True)


def main():
    if "--force" not in sys.argv and not os.environ.get("TW_STEP_LOG"):
        runs = live_runs()
        if runs:
            sys.exit(f"not built: {', '.join(runs)} live (a join compares data/packs); --force to build anyway")
    build()


if __name__ == "__main__":
    main()
