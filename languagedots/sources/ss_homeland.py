"""South Sudan, Jonglei, Unity and Upper Nile: each county drawn on the language of the people whose
homeland it is, on the county's 2025 estimate. Ask 019's ruling (published estimates placed by
homeland, rows `modelled`); fix-ss, 2026-10-07. The record is sources/ss.md section 7.

    python sources/ss_homeland.py     -> data/normalized/ss_homeland.csv (county x node, counts)

No survey with a language or group question has sampled these three states openly (ss.md
section 1). What exists, and how it is used:

  * HOMELAND. The Conflict Sensitivity Resource Facility's county profiles (csrf-southsudan.org,
    read 2026-10-07; data/raw/ss/csrf_county_profiles_ethnic.json) name each county's ethnic
    groups, the main one first. Each county goes to its first-named group's language. That
    group's line is asserted against the saved profile below, so a change upstream shows.
  * MALAKAL, the one mixed county with a measured split: JICA's household survey of Malakal Town
    (884 households in 24 bomas; final report July 2014, appendix I, attachment II, "Tribe and
    Ethnic Group"; data/raw/ss/JICA_Malakal_Town_FR_AppendixI_12228961_05.pdf): Shilluk 445,
    Nuer 201, Dinka 134, others 104. Applied to the whole county; "others" on `africa_other`.
  * SMALL GROUPS the profiles name second, where a published figure exists and Joshua Project's
    point for the group lies in that county (data/raw/pg/joshuaproject_pgic.csv, ROG3 = OD):
    Kacipo-Balesi (JP "Suri", koe, 4,700) in Pibor; Opo (JP "Opo" and "Buldit", lgn, 7,400
    each) in Maiwut, the profile's "Koma". JP figures are scaled by the ten states' 2025
    estimate over JP's South Sudan total, as sources/sd_estimate.py does. A small group may take
    at most a fifth of its county (asserted).
  * NOT SPLIT (named second by the profiles, no figure for them in that county): Koma in
    Longochuk (JP's Komo, 26,000 with its point there, would be 35% of the county: JP's figure is
    national and the profile names the Nuer first), Anyuak in
    Akobo, Gawaar and Lou Nuer and Shilluk in Canal/Pigi, Jie in Pibor (a Toposa dialect,
    Glottolog jiye1239), Nuer in Maban, Padang Dinka in Panyikang, Shilluk in Renk. They are drawn
    as the county's first group.
  * Refugees from Sudan (Maban's camps: Uduk, Jumjum, Ingessana; Pariang's: Nuba) are outside
    the county estimates and are not drawn, as in the other seven states.
"""
import csv
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
LOOKUP = RD / "data" / "geo" / "ss" / "ss_lookup.csv"
COUNTIES = RD / "data" / "geo" / "ss" / "ss_counties.csv"
CSRF = HERE / "data" / "raw" / "ss" / "csrf_county_profiles_ethnic.json"
JP = HERE / "data" / "raw" / "pg" / "joshuaproject_pgic.csv"
OUT = HERE / "data" / "normalized" / "ss_homeland.csv"
STATES = ("SS03", "SS06", "SS07")
SOURCE_ID = "ss_homeland_csrf_2025"

NI = "nilosaharan.nilotic"
SU = "nilosaharan.surmic"
KO = "nilosaharan.koman"
DINKA, NUER, SHILLUK = f"{NI}.dinka", f"{NI}.nuer", f"{NI}.shilluk"

# county -> (first-named group as the CSRF profile prints it, node)
HOME = {
    "SS0301": ("Lou Nuer", NUER),                       # Akobo; Anyuak not split
    "SS0302": ("Gawaar Nuer", NUER),                    # Ayod
    "SS0303": ("Southeastern Dinka", DINKA),            # Bor South
    "SS0304": ("Padang Dinka", DINKA),                  # Canal/Pigi; Nuer, Shilluk not split
    "SS0305": ("Southeastern Dinka", DINKA),            # Duk
    "SS0306": ("Nuer", NUER),                           # Fangak
    "SS0307": ("Lou Nuer", NUER),                       # Nyirol
    "SS0308": ("Murle", f"{SU}.murle"),                 # Pibor; Kacipo below, Jie not split
    "SS0309": ("Anyuak", f"{NI}.anuak"),                # Pochalla
    "SS0310": ("Southeastern Dinka", DINKA),            # Twic East
    "SS0311": ("Lou Nuer", NUER),                       # Uror
    "SS0601": ("Padang Dinka", DINKA),                  # Abiemnhom (Alor, Ruweng)
    "SS0602": ("Western Jikany Nuer", NUER),            # Guit
    "SS0603": ("Jagey", NUER),                          # Koch
    "SS0604": ("Dok Nuer", NUER),                       # Leer
    "SS0605": ("Haak Nuer", NUER),                      # Mayendit
    "SS0606": ("Bul Nuer", NUER),                       # Mayom
    "SS0607": ("Nyuong Nuer", NUER),                    # Panyijiar
    "SS0608": ("Padang Dinka", DINKA),                  # Pariang (Ruweng)
    "SS0609": ("Leek Nuer", NUER),                      # Rubkona
    "SS0701": ("Padang Dinka", DINKA),                  # Baliet
    "SS0702": ("Shilluk", SHILLUK),                     # Fashoda
    "SS0703": ("Eastern Jikany Nuer", NUER),            # Longochuk; Komo below
    "SS0704": ("Eastern Jikany Nuer", NUER),            # Luakpiny/Nasir
    "SS0705": ("Mabanese", f"{NI}.mabaan"),             # Maban; Nuer not split
    "SS0706": ("Eastern Jikany Nuer", NUER),            # Maiwut; Opo below
    "SS0707": ("Shilluk", None),                        # Malakal: JICA split below
    "SS0708": ("Shilluk", SHILLUK),                     # Manyo
    "SS0709": ("Ageer", DINKA),                         # Melut (Ageer and Nyiel Dinka)
    "SS0710": ("Shilluk", SHILLUK),                     # Panyikang; Padang Dinka not split
    "SS0711": ("Abialang", DINKA),                      # Renk; Shilluk not split
    "SS0712": ("Eastern Jikany Nuer", NUER),            # Ulang
}
# JICA Malakal Town household survey, households by tribe (appendix I, A1-270, total row)
MALAKAL = {SHILLUK: 445, NUER: 201, DINKA: 134, "africa_other": 104}
# small groups: county, JP people-group names (ROL3 in brackets), node
SMALL = [
    ("SS0308", {"Suri"}, "koe", f"{SU}.kacipo"),
    # Komo (JP 26,000, point in Longochuk) left out: it would make Longochuk 35% Komo, more than
    # the profile's "Eastern Jikany Nuer ... and Koma" bears, and JP's one figure is national.
    ("SS0706", {"Opo", "Buldit"}, "lgn", f"{KO}.opo"),
]


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def largest_remainder(vals, total):
    f = np.asarray(vals, dtype=float)
    base = np.floor(f)
    k = int(round(total - base.sum()))
    base[np.argsort(-(f - base))[:k]] += 1
    return base.astype(int)


def main():
    lut = pd.read_csv(LOOKUP, dtype={"unit": str})
    cn = pd.read_csv(COUNTIES, dtype={"county": str, "unit": str})
    cn["pop_2025"] = cn["pop_2025"].astype(int)
    pop10 = int(lut["pop"].sum())
    mine = cn[cn["unit"].isin(STATES)].set_index("county")
    say(set(mine.index) == set(HOME), f"{len(mine)} counties in Jonglei, Unity and Upper Nile, "
        "each with a homeland line")
    for st in STATES:
        a = int(mine.loc[mine["unit"] == st, "pop_2025"].sum())
        b = int(lut.loc[lut["unit"] == st, "pop"].iloc[0])
        say(abs(a - b) <= 2, f"{st}: counties sum to the state's 2025 estimate ({a:,} against "
            f"{b:,}; the lookup's rounding)")

    prof = json.loads(CSRF.read_text(encoding="utf-8"))
    for c, (grp, _) in HOME.items():
        e = prof[c]
        text = e.get("ethnic") or e["desc"].split("Ethnic groups and languages:")[-1].strip()
        text = text.replace("￼", "")
        say(grp.lower() in text.lower()[:len(grp) + 40],
            f"{mine.loc[c, 'name']}: profile names {grp!r} first ({text[:60]!r})")

    txt = JP.read_text(encoding="utf-8-sig")
    jp = [r for r in csv.DictReader(io.StringIO(txt[txt.index("ROG3,"):])) if r["ROG3"] == "OD"]
    jp_total = sum(int(r["Population"]) for r in jp)
    scale = pop10 / jp_total
    print(f"  Joshua Project: {len(jp)} groups in South Sudan, {jp_total:,}; scale to the 2025 "
          f"estimate {pop10:,}: {scale:.4f}")

    rows = []
    for c, (grp, node) in HOME.items():
        p = int(mine.loc[c, "pop_2025"])
        name = mine.loc[c, "name"]
        parts = {}
        labels = {}
        if c == "SS0707":
            tot = sum(MALAKAL.values())
            for n, h in MALAKAL.items():
                parts[n] = p * h / tot
                labels[n] = f"JICA Malakal Town household survey: {h} of {tot} households"
        else:
            for cc, names, iso, n in SMALL:
                if cc != c:
                    continue
                hit = [r for r in jp if r["PeopNameInCountry"] in names]
                say(len(hit) == len(names) and all(r["ROL3"] == iso for r in hit),
                    f"JP {sorted(names)} found, language {iso}")
                v = sum(int(r["Population"]) for r in hit) * scale
                parts[n] = v
                labels[n] = (f"Joshua Project {' + '.join(sorted(names))} "
                             f"{sum(int(r['Population']) for r in hit):,} x {scale:.4f}")
            rest = p - sum(parts.values())
            say(rest > 0.8 * p, f"{name}: the first group keeps {rest / p:.0%}")
            parts[node] = parts.get(node, 0.0) + rest
            labels[node] = f"CSRF county profile, first-named group: {grp}; the rest of the county"
        ks = list(parts)
        cnt = largest_remainder([parts[k] for k in ks], p)
        for k, v in zip(ks, cnt):
            rows.append(dict(geo_id=c, geo_level="county", geo_name=name, source_category=k,
                             source_label=labels[k], count=int(v), tier="modelled",
                             source_id=SOURCE_ID, year=2025))
    out = pd.DataFrame(rows)
    out = out[out["count"] > 0]
    want = int(mine["pop_2025"].sum())
    say(int(out["count"].sum()) == want, f"drawn total {int(out['count'].sum()):,} = the three "
        "states' 2025 estimate")
    bad = [c for c, d in out.groupby("geo_id") if int(d["count"].sum()) != int(mine.loc[c, "pop_2025"])]
    say(not bad, f"every county sums to its 2025 estimate {bad}")

    nat = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("\n  drawn in the three states, against Joshua Project's South Sudan figure (scaled):")
    jp_by = {}
    for r in jp:
        jp_by[r["ROL3"]] = jp_by.get(r["ROL3"], 0) + int(r["Population"])
    iso_of = {NUER: ["nus"], DINKA: ["dip", "dks", "dik", "dib", "diw"], SHILLUK: ["shk"],
              f"{SU}.murle": ["mur"], f"{NI}.anuak": ["anu"], f"{NI}.mabaan": ["mfz"],
              f"{SU}.kacipo": ["koe"], f"{KO}.komo": ["xom"], f"{KO}.opo": ["lgn"]}
    for n, v in nat.items():
        j = sum(jp_by.get(i, 0) for i in iso_of.get(n, [])) * scale
        print(f"    {n:34s} {v:>10,}   JP (whole country) {j:>10,.0f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(HERE)}: {len(out)} rows, {out['source_category'].nunique()} nodes")


if __name__ == "__main__":
    main()
