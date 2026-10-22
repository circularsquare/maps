"""Nigeria: the languages of each state's homelands, for naming the part of MICS's "other
language" that the Afrobarometer cannot split, and for placing minority languages inside states.

    python sources/ng_homeland.py           -> data/normalized/ng_homeland.csv (one row per point)
    python sources/ng_mics.py               -> data/normalized/ng.csv (reads the table above)
    python sources/ng_homeland.py --blocks  -> also the generated blocks of taxonomy/tree.d/ng.txt
                                               and taxonomy/ng2022.py (between their >>> / <<<
                                               markers): the new answers ng.csv draws

Added 2026-10-19 (session eca0185b-ng; sources/ng.md §8). Anita's standing word: bias toward
splitting, and better a named modelled language than an unnamed wedge (AGENT_BRIEF §2, ask 019:
published speaker estimates placed by homeland, rows `modelled`).

SOURCES.
  * Joshua Project, people groups in countries (data/raw/pg/joshuaproject_pgic.csv, already in
    the repo for Papua New Guinea and South Sudan): Nigeria's 531 people groups, each with a
    population estimate, its primary language (ISO 639-3) and one point. The populations are
    ethnic, not speaker counts, and add up to 242 million against COD-PS's 217 million; they are
    used only as RELATIVE weights between the small languages of one state.
  * Glottolog 5 (data/raw/glottolog; CC BY): each ISO code's language (a dialect code is read as
    its language: Ezaa, Izii, Ikwo and Mgbo are Izi-Ezaa-Ikwo-Mgbo), its classification (which
    tree group a new node goes under, GROUP below), and its point, a second homeland point for
    placement.
  * COD-AB (religiondots' nga_admin2.shp): which state and LGA each point lies in.

WHICH LANGUAGE IS ALREADY AN ANSWER. EXISTING gives the ISO codes of every answer ng.csv already
draws, read one by one against Glottolog's names (`Bassa` is Bassa-Kontagora bsr, not Liberia's
bsq; `Ogoni` is Khana, Gokana and Tee; `Kambari` the four Kambari languages). A Joshua Project
group whose language is one of these counts towards that answer; every other language becomes a
new answer named by Glottolog, under the tree group its Glottolog classification gives.

NOT USED. Sign languages, English, Arabic, Pidgins and Glottolog's bookkeeping or unattested
entries; groups whose language is Hausa, Fulfulde or another MICS-nine language (14 Hausa-speaking
groups such as Butawa and Gamo-Ningi stay on Hausa).
"""
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
GLOTTO = HERE / "data" / "raw" / "glottolog"
JP = HERE / "data" / "raw" / "pg" / "joshuaproject_pgic.csv"
ADM2 = HERE.parent / "religiondots" / "data" / "raw" / "ng" / "shp" / "nga_admin2.shp"
OUT = HERE / "data" / "normalized" / "ng_homeland.csv"
TREE = HERE / "taxonomy" / "tree.d" / "ng.txt"
MAPPING = HERE / "taxonomy" / "ng2022.py"
N_JP = 531

# every answer ng.csv already draws -> its ISO 639-3 codes (checked against Glottolog's names)
EXISTING = {
    "Hausa": "hau", "Yoruba": "yor", "Igbo": "ibo", "Fula": "fuv fub ful", "Kanuri": "knc kby",
    "English": "eng", "Nigerian Pidgin": "pcm", "Tiv": "tiv", "Ekajuk": "eka", "Zarma": "dje",
    "Shuwa Arabic": "shu", "Busa": "bqp bus", "Bariba": "bba", "Nupe": "nup", "Igala": "igl",
    "Idoma": "idu", "Agatu": "agc", "Alago": "ala", "Yala": "yba", "Etulo": "utr", "Igede": "ige",
    "Yace": "ekr", "Ebira": "igb", "Gbagyi": "gbr gby", "Gade": "ged", "Ika": "ikk",
    "Ukwuani": "ukw", "Ikwere": "ikw", "Ekpeye": "ekp", "Ogba": "ogc", "Edo": "bin",
    "Urhobo": "urh", "Isoko": "iso", "Esan": "ish", "Yekhee": "ets", "Okpela": "atg",
    "Okpe": "oke", "Epie": "epi", "Degema": "deg", "Emai-Iuleha-Ora (Owan)": "ema",
    "Ijaw": "ijc", "Kalabari": "ijn", "Okrika": "okr", "Nembe": "ijs", "Ibani": "iby",
    "Ibibio": "ibb", "Efik": "efi", "Anaang": "anw", "Oron": "orx", "Obolo": "ann",
    "Ogoni": "gkn tkq", "Khana": "ogo", "Eleme": "elm", "Abua": "abn", "Ogbia": "ogb",
    "Yakurr": "yaz", "Mbembe": "mfn", "Tarok": "yer", "Berom": "bom", "Tyap": "kcg",
    "Jju": "kaj", "Gyong": "kdm", "Aten": "gne", "Hyam (Jaba)": "jab", "Adara": "kad",
    "Eggon": "ego", "Mada": "mda", "Ninzo": "nin", "Numana": "nbr", "Irigwe": "iri",
    "Izere": "izr", "Migili": "mgi", "Koro": "vkz vkn", "Kambari": "bqx kdl tsw asg",
    "Bassa": "bzw bsr", "Kurama": "krh", "Laru": "lan", "Lela (Dakarkari)": "dri",
    "Jukun": "jbu juk juw", "Kuteb": "kub", "Jibu": "jib", "Tigon": "nza",
    "Etkywan (Ichen)": "ich", "Ejagham": "etu", "Jarawa": "jrr", "Mambila": "mzk",
    "Ndoola": "ndr", "Bekwarra": "bkv", "Bette-Bendi": "btt", "Mumuye": "mzm",
    "Chamba": "ccg ndi", "Dza (Jenjo)": "jen", "Yungur": "yun", "Waja": "wja", "Tula": "tul",
    "Ngas": "anc", "Tangale": "tan", "Bura-Pabir": "bwr", "Marghi": "mrt", "Karekare": "kai",
    "Bade": "bde", "Kamwe": "hig", "Mwaghavul": "sur", "Gwandara": "gwn", "Tera": "ttr",
    "Bole": "bol", "Bachama": "bcy", "Zaar": "say", "Goemai": "ank", "Kibaku (Chibok)": "ckl",
    "Huba (Kilba)": "hbb", "Glavda": "glw", "Piapung": "pcw",
    # answers added 2026-10-19 with Afrobarometer R3 and R10
    "Izi-Ezaa-Ikwo-Mgbo": "izi eza izz iqw gmz", "Ukpet-Ehom": "akd", "Bokyi": "bky",
    "Itsekiri": "its", "Kupa": "kug", "Nyankpa (Yeskwa)": "yes", "Gun": "guw",
}
# the ten MICS answers and the lingua francas: never a homeland candidate
NOT_CANDIDATE = {"Hausa", "Yoruba", "Igbo", "Fula", "Kanuri", "Ijaw", "Tiv", "Ibibio", "Edo",
                 "English", "Nigerian Pidgin"}

# Glottolog classification (ancestor names) -> tree group, first match wins
GROUP = [
    ("Sign Language", None), ("Bookkeeping", None), ("Unattested", None), ("Pidgin", None),
    ("Indo-European", None), ("Semitic", None), ("Fula", None),
    ("Ijoid", "nigercongo.ijoid"),
    ("Delta Cross", "nigercongo.crossriver"),
    ("Benue-Congo Plateau", "nigercongo.plateau"),
    ("Kainji", "nigercongo.kainji"),
    ("Jukunoid", "nigercongo.jukunoid"),
    ("Wide Grassfields", "nigercongo.bantoid.grassfields"),
    ("Bantoid", "nigercongo.bantoid"),
    ("Edoid", "nigercongo.voltaniger.edoid"),
    ("Akpes-Edoid", "nigercongo.voltaniger"),
    ("Ebira-Nupoid", "nigercongo.voltaniger"), ("Idomoid", "nigercongo.voltaniger"),
    ("Igboid", "nigercongo.voltaniger"), ("Defoid", "nigercongo.voltaniger"),
    ("Oko-Eni-Osayen", "nigercongo.voltaniger"), ("Ukaan", "nigercongo.voltaniger"),
    ("Tula-Longuda", "nigercongo.adamawa"),     # as Waja and Tula already (sources/ng.md §5)
    ("Gur", "nigercongo.gur"),
    ("Gbe", "nigercongo.kwa"),
    ("Cameroun-Ubangian", "nigercongo.adamawa"), ("Gbaya-Manza-Ngbaka", "nigercongo.adamawa"),
    ("North Volta-Congo", "nigercongo.adamawa"),
    ("Benue-Congo", "nigercongo"),
    ("Mande", "nigercongo.mande"),
    ("Chadic", "afroasiatic.chadic"),
    ("Berber", "afroasiatic.berber"),
    ("Songhay", "nilosaharan.songhay"),
    ("Saharan", "nilosaharan"),
    ("Central Sudanic", "nilosaharan.centralsudanic"),
]


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def glottolog():
    L = pd.read_csv(GLOTTO / "languages.csv", keep_default_na=False)
    V = pd.read_csv(GLOTTO / "values.csv", keep_default_na=False,
                    usecols=["Language_ID", "Parameter_ID", "Value"])
    cls = V[V.Parameter_ID == "classification"].set_index("Language_ID")["Value"].to_dict()
    L = L.set_index("ID")
    return L, cls


def language_of(iso, L, by_iso):
    """ISO code -> Glottolog language row (a dialect's code read as its language)."""
    gid = by_iso.get(iso)
    if gid is None:
        return None
    r = L.loc[gid]
    if r["Level"] == "dialect" and r["Language_ID"]:
        return L.loc[r["Language_ID"]]
    return r


def group_of(gid, L, cls):
    names = [L.loc[x, "Name"] for x in cls.get(gid, "").split("/") if x in L.index]
    for key, grp in GROUP:
        if any(key == n or (key in ("Bookkeeping", "Unattested", "Sign Language", "Pidgin")
                            and n.startswith(key)) for n in names):
            return grp, names
    return None, names


def tree_index(with_ng=False):
    """Every node id and label written in tree.txt and the other fragments (and ng.txt's own
    hand-written part, with_ng)."""
    ids = {}
    files = [HERE / "taxonomy" / "tree.txt"] + sorted((HERE / "taxonomy" / "tree.d").glob("*.txt"))
    for f in files:
        if f.name == "ng.txt":
            if not with_ng:
                continue
            t = f.read_text(encoding="utf-8").split("# >>> generated")[0]
            for line in t.splitlines():
                if line.strip() and not line.lstrip().startswith("#") and "|" in line:
                    ids.setdefault(line.split("|")[0].strip(), line.split("|")[1].strip())
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if parts[0] and " " not in parts[0]:
                ids.setdefault(parts[0], parts[1])
    return ids


def build():
    import geopandas as gpd
    L, cls = glottolog()
    by_iso = {r: i for i, r in zip(L.index, L["ISO639P3code"]) if r}
    jp = pd.read_csv(JP, skiprows=1, encoding="utf-8-sig", low_memory=False)
    jp = jp[jp["ROG3"] == "NI"].copy()
    say(len(jp) == N_JP, f"Joshua Project: {len(jp)} people groups in Nigeria")
    iso2ans = {}
    for a, cs in EXISTING.items():
        for c in cs.split():
            say(c in by_iso, f"EXISTING {a}: {c} is a Glottolog ISO code")
            iso2ans[c] = a
    tree = tree_index()
    all_ids = tree_index(with_ng=True)
    import ng_afro as A                                      # noqa: E402
    sys.path.insert(0, str(HERE / "taxonomy"))
    adm = gpd.read_file(ADM2, engine="pyogrio")[["adm1_name", "adm2_pcode", "adm1_pcode",
                                                 "geometry"]].to_crs(4326)

    rows, new, skipped = [], {}, []
    for r in jp.itertuples():
        iso = r.ROL3
        lang = language_of(iso, L, by_iso)
        if iso in iso2ans:
            ans = iso2ans[iso]
        elif lang is None:
            skipped.append((r.PeopNameInCountry, iso, "no Glottolog language"))
            continue
        else:
            # another ISO of an existing answer's language (a dialect code)?
            same = [a for c, a in iso2ans.items()
                    if (language_of(c, L, by_iso) is not None
                        and language_of(c, L, by_iso).name == lang.name)]
            if same:
                ans = same[0]
            else:
                grp, names = group_of(lang.name, L, cls)
                if grp is None:
                    skipped.append((r.PeopNameInCountry, iso, " / ".join(names[:2]) or "-"))
                    continue
                ans = lang["Name"]
                nid = f"{grp}.{slug(ans)}"
                # a node another country already wrote for the same language: reuse its id
                hit = [k for k, v in tree.items()
                       if k.startswith(grp + ".") and k.rsplit(".", 1)[1] == slug(ans)]
                if hit:
                    nid = hit[0]
                if grp not in all_ids:
                    raise SystemExit(f"{ans}: its group {grp} is not a node")
                new.setdefault(ans, dict(node=nid, glotto=lang.name, iso=set(),
                                         label=tree[nid] if hit else ans, reused=bool(hit)))
                new[ans]["iso"].add(iso)
        # a homeland only if Glottolog lists Nigeria among the language's countries: Joshua
        # Project also pins migrant groups (Ewe in Oyo, Gourmanchema and Jenaama Bozo in Kebbi)
        home = lang is not None and "NG" in str(lang["Countries"]).split(";")
        rows.append(dict(answer=ans, iso=iso, source="joshuaproject", people=r.PeopNameInCountry,
                         pop=float(r.Population), lat=float(r.Latitude),
                         lon=float(r.Longitude), ng_home=home))
    # Glottolog's own points for every answer's language (placement only, pop 0)
    ans_isos = {}
    for a, cs in EXISTING.items():
        ans_isos.setdefault(a, set()).update(cs.split())
    for a, d in new.items():
        ans_isos.setdefault(a, set()).update(d["iso"])
    seen = set()
    for a, isos in ans_isos.items():
        for c in isos:
            lang = language_of(c, L, by_iso)
            if lang is None or (a, lang.name) in seen or not lang["Latitude"]:
                continue
            seen.add((a, lang.name))
            rows.append(dict(answer=a, iso=c, source="glottolog", people=lang["Name"], pop=0.0,
                             lat=float(lang["Latitude"]), lon=float(lang["Longitude"]),
                             ng_home="NG" in str(lang["Countries"]).split(";")))
    df = pd.DataFrame(rows)
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    j = gpd.sjoin(pts, adm, how="left", predicate="within").drop(columns="index_right")
    out = j["adm1_name"].isna()
    # a point just outside Nigeria (Glottolog puts cross-border languages' points in Cameroon,
    # Benin, Niger, Chad): its nearest LGA within 50 km, else not a Nigerian homeland point
    if out.any():
        a3 = adm.to_crs(3857)
        p3 = pts[out].to_crs(3857)
        nn = gpd.sjoin_nearest(p3, a3, how="left", max_distance=50_000, distance_col="d")
        nn = nn[~nn.index.duplicated()]
        for c in ("adm1_name", "adm2_pcode", "adm1_pcode"):
            j.loc[out, c] = nn[c].reindex(j.index[out]).to_numpy()
    lost = j[j["adm1_name"].isna()]
    j = j[j["adm1_name"].notna()].drop(columns="geometry")
    j["state"] = j["adm1_name"].map(lambda s: A.NORM.get(A.ckey(s), s))
    say(j["state"].nunique() <= A.N_STATES, f"points in {j['state'].nunique()} states")
    jpl = j[j["source"] == "joshuaproject"]
    print(f"  Joshua Project groups kept: {len(jpl)} of {len(jp)} ({len(skipped)} skipped: "
          + "; ".join(f"{p} [{i}: {why}]" for p, i, why in skipped) + ")")
    print(f"  {len(lost)} points more than 50 km outside Nigeria dropped "
          f"({', '.join(sorted(set(lost['answer'])))[:300]})")
    print(f"  {len(new)} new answers ({sum(d['reused'] for d in new.values())} on a node another "
          f"country already wrote)")
    j.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(HERE)} ({len(j)} points)")
    return j, new


def write_blocks(new, used):
    """The generated parts of tree.d/ng.txt and ng2022.py: only answers ng.csv draws (`used`)."""
    keep = {a: d for a, d in new.items() if a in used}
    # parents the hand-written part of ng.txt and tree.txt lack, repeated bare (AGENT_BRIEF §3)
    base = set()
    for line in (HERE / "taxonomy" / "tree.txt").read_text(encoding="utf-8").splitlines():
        if "|" in line and not line.lstrip().startswith("#"):
            base.add(line.split("|")[0].strip())
    hand = tree_index(with_ng=True)
    others = tree_index()
    have = base | set(hand)
    parents = set()
    for d in keep.values():
        p = d["node"]
        while "." in p:
            p = p.rsplit(".", 1)[0]
            if p not in have:
                parents.add(p)
    missing = sorted(p for p in parents if p not in others)
    if missing:
        raise SystemExit(f"generated nodes' parents defined nowhere: {missing}")
    tree_lines = [f"{p} | {others[p]}" for p in sorted(parents)]
    tree_lines += [f"{d['node']} | {d['label']}" for a, d in sorted(keep.items(),
                                                                    key=lambda kv: kv[1]["node"])]
    _replace_block(TREE, "#", tree_lines)
    map_lines = [f"    {a!r}: {d['node']!r},   # {d['glotto']}"
                 for a, d in sorted(keep.items())]
    _replace_block(MAPPING, "#", ["HOMELAND = {"] + map_lines + ["}"])
    print(f"  wrote {len(keep)} generated nodes into {TREE.name} and {MAPPING.name}")


def _replace_block(path, c, lines):
    start = f"{c} >>> generated by sources/ng_homeland.py, do not edit by hand"
    end = f"{c} <<< end of generated block"
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise SystemExit(f"{path}: the generated block's markers are missing")
    a, rest = text.split(start, 1)
    _, b = rest.split(end, 1)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(a + start + "\n" + "\n".join(lines) + "\n" + end + b, encoding="utf-8")
    os.replace(tmp, path)


def load():
    return pd.read_csv(OUT, dtype={"adm2_pcode": str, "adm1_pcode": str})


if __name__ == "__main__":
    sys.path.insert(0, str(HERE / "sources"))
    _, new_answers = build()
    if "--blocks" in sys.argv:
        # after sources/ng_mics.py: the generated nodes are the new answers ng.csv draws
        used = set(pd.read_csv(HERE / "data" / "normalized" / "ng.csv")["source_category"])
        write_blocks(new_answers, used)
