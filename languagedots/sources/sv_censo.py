"""El Salvador, VII Censo de Poblacion y VI de Vivienda 2024 (ONEC, Banco Central de Reserva):
"Habla otro idioma aparte del espanol? Cuales?", per distrito.

    python sources/sv_censo.py [--fetch]

Writes data/normalized/sv.csv: one row per (distrito, source_category), `count` as published.
The categories are the census's own labels:

    Inglés, Francés, Italiano, Náhuat, Pisbi, Potón, LESSA, Otro, Español
                     MENTIONS: people of 3 and over who named that language, several allowed
    Sí               PERSONS of 3 and over who said they speak a language besides Spanish
    Población        everyone counted in the distrito, all ages (the census's district table)

THE QUESTION presupposes Spanish: everyone is taken to speak it, and the census asks whether
they speak anything else, and which. The BCR's dashboard says the universe ("Total de personas
de 3 anos o mas que habla otro idioma") and that the percentages can sum to more than 100%
"debido que una persona puede hablar uno o mas idiomas". `Español` as an answer (3,170 people)
is someone who named Spanish among their languages anyway; countries/sv.py does not count it
twice.

THE SOURCE is the BCR's census GeoPortal (poblacion.bcr.gob.sv, an ArcGIS Hub; org
xSn1Pefr3M9zkw7Z), whose dashboards read open, keyless feature services. Three of them carry the
language figures and one the population; the 262 distritos are the pre-2024 municipalities,
which the 2024 reform kept as districts of 44 new municipalities. The distrito polygons come from
the same service as the language figures (written to data/raw/sv/distritos.geojson for
sources/sv_geo.py).

CHECKS (the script stops unless all hold):
  1. 262 distritos in every table, the same ids, unique.
  2. Two services agree on every distrito and language: `tabla_idioma` (the dashboard's table)
     and `Limites_idiomas` (its map layer); and the map layer's `Todos` equals `Sí` in a third
     service (`Si_habla_otro_idioma`).
  3. The distritos sum to the table's own aggregate rows: every municipio, every department and
     the national row, on every language.
  4. National totals equal the figures the BCR released (2026-03 press coverage, pinned below).
  5. Per distrito, every language's mentions are at most `Sí`, and `Sí` is at most the sum of
     mentions (each yes names at least one language).
  6. Population: 262 distritos sum to the department layer's 14 totals and to 5,922,921, and
     district names agree between the population and language tables once folded.
  7. Printed, not asserted: the earlier department-level release (30 Sept, before revision) gives
     Sí + No, the age-3+ universe; it is compared with population minus three fifths of the
     0-4 band (ages 0, 1 and 2 of 0-4) per department.
"""
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "sv"
NORM = HERE / "data" / "normalized"
BASE = "https://services8.arcgis.com/xSn1Pefr3M9zkw7Z/arcgis/rest/services/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# (service, layer, file, order field)
TABLES = [
    ("tabla_idioma", 0, "tabla_idioma.json", "FID"),
    ("Si_habla_otro_idioma___Copy", 0, "si_habla.json", "FID"),
    ("Limites_idiomas", 2, "limites_idiomas_distritos.json", "OBJECTID"),
    ("capas_población_1", 6, "poblacion_distritos.json", "OBJECTID_1"),
    ("capas_población_1", 3, "poblacion_departamentos.json", "OBJECTID_1"),
    ("Otro_30_09", 0, "idiomas_departamentos_30sep.json", "OBJECTID"),
]
N = 262
LANGS = ["Inglés", "Francés", "Italiano", "Náhuat", "Pisbi", "Potón", "LESSA", "Otro", "Español"]
# BCR, as reported 2026-03-07 (Diario El Mundo, Infobae: "427,368 ... 97.07% ingles")
PUBLISHED = {"Sí": 427368, "Inglés": 414887, "Francés": 16741, "Italiano": 6911, "Náhuat": 1135,
             "Pisbi": 24, "Potón": 32, "LESSA": 2025, "Otro": 13326}
POPULATION = 5922921


def query(svc, layer, order, geometry=False, where="1=1"):
    rows, off = [], 0
    while True:
        params = {"where": where, "outFields": "*", "f": "geojson" if geometry else "json",
                  "returnGeometry": "true" if geometry else "false", "outSR": 4326,
                  "resultOffset": off, "resultRecordCount": 1000, "orderByFields": order}
        url = (BASE + urllib.parse.quote(svc) + f"/FeatureServer/{layer}/query?"
               + urllib.parse.urlencode(params))
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180))
        if "error" in d:
            raise SystemExit(f"{svc}/{layer}: {d['error']}")
        feats = d["features"]
        rows += feats if geometry else [f["attributes"] for f in feats]
        more = d.get("exceededTransferLimit") or d.get("properties", {}).get("exceededTransferLimit")
        if not more and len(feats) < 1000:
            return rows
        off += len(feats)
        time.sleep(0.5)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for svc, layer, name, order in TABLES:
        rows = query(svc, layer, order)
        (RAW / name).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"  {svc}/{layer}: {len(rows)} rows -> {name}")
    feats = query("Limites_idiomas", 2, "OBJECTID", geometry=True, where="idiomas='Todos'")
    gj = {"type": "FeatureCollection", "features": feats}
    (RAW / "distritos.geojson").write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    print(f"  Limites_idiomas/2 polygons (idiomas = Todos): {len(feats)} -> distritos.geojson")


def load(name):
    return pd.DataFrame(json.loads((RAW / name).read_text(encoding="utf-8")))


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    t = load("tabla_idioma.json")
    t["id_distrito"] = t["id_distrito"].astype(str)
    real = t[t["id_distrito"] != "000000"].copy()
    agg = t[t["id_distrito"] == "000000"].copy()
    check(set(real["idiomas"]) == set(LANGS), f"languages {sorted(set(real['idiomas']))}")
    g = real.groupby(["id_distrito", "idiomas"])["tot"]
    check((g.nunique() == 1).all(), "tabla_idioma's repeated rows agree with each other")
    tab = g.first().unstack()[LANGS]
    names = real.drop_duplicates("id_distrito").set_index("id_distrito")
    check(len(tab) == N and tab.index.str.fullmatch(r"\d{6}").all(),
          f"{len(tab)} distritos in tabla_idioma, six-digit ids")

    lim = load("limites_idiomas_distritos.json")
    lim["id_distrito"] = lim["id_distrito"].astype(str)
    lt = lim.pivot_table(index="id_distrito", columns="idiomas", values="tot", aggfunc="first")
    check(set(lt.index) == set(tab.index), "Limites_idiomas has the same 262 distritos")
    diff = (tab - lt[LANGS].reindex(tab.index)).abs().to_numpy().sum()
    check(diff == 0, f"two services agree on every distrito and language (total |diff| {diff})")

    s = load("si_habla.json")
    s["id_distrito"] = s["id_distrito"].astype(str)
    s = s[s["id_distrito"] != "000000"]
    check(s["id_distrito"].nunique() == N and (s.groupby("id_distrito")["si"].nunique() == 1).all(),
          f"Si_habla: {s['id_distrito'].nunique()} distritos, one value each")
    si = s.drop_duplicates("id_distrito").set_index("id_distrito")["si"].reindex(tab.index)
    check((si == lt["Todos"].reindex(tab.index)).all(), "Sí equals the map layer's `Todos` everywhere")

    # 3. the table's own aggregate rows
    a = agg.drop_duplicates(["id_depto", "id_mun", "idiomas"])
    nat = a[a["departamento"] == "Nacional"].set_index("idiomas")["tot"]
    check(all(nat[l] == tab[l].sum() for l in LANGS), "distritos sum to the national row")
    dep_rows = a[(a["mun"] == "Seleccionar municipio")]
    dep_sum = real.drop_duplicates(["id_distrito", "idiomas"]).groupby(["departamento", "idiomas"])["tot"].sum()
    bad = [(r.departamento, r.idiomas) for r in dep_rows.itertuples()
           if dep_sum.get((r.departamento, r.idiomas)) != r.tot]
    check(len(dep_rows) == 14 * len(LANGS) and not bad,
          f"distritos sum to all {len(dep_rows) // len(LANGS)} department rows ({bad[:3]})")
    mun_rows = a[(a["mun"] != "Seleccionar municipio") & (a["departamento"] != "Nacional")]
    mun_sum = real.drop_duplicates(["id_distrito", "idiomas"]).groupby(["mun", "idiomas"])["tot"].sum()
    bad = [(r.mun, r.idiomas) for r in mun_rows.itertuples() if mun_sum.get((r.mun, r.idiomas)) != r.tot]
    check(len(mun_rows) == 44 * len(LANGS) and not bad,
          f"distritos sum to all {len(mun_rows) // len(LANGS)} municipio rows ({bad[:3]})")

    # 4. the released figures
    got = {**{l: int(tab[l].sum()) for l in LANGS if l in PUBLISHED}, "Sí": int(si.sum())}
    check(got == PUBLISHED, f"national totals equal the BCR's released figures {got}")

    # 5. per distrito
    m = tab.drop(columns="Español")
    check((m.le(si, axis=0)).all().all(), "no language has more mentions than Sí in any distrito")
    # Español counts here: a yes who named only Spanish (a foreign-born resident answering
    # about their own other language, presumably) has no other mention
    check((si <= tab.sum(axis=1)).all(), "Sí is at most the sum of mentions (Español included) "
                                         "in every distrito")
    short = int((si - m.sum(axis=1)).clip(lower=0).sum())
    print(f"     mentions per yes, nationally {m.to_numpy().sum() / si.sum():.4f} (other than "
          f"Spanish); {short:,} yeses across the distritos are not covered by a non-Spanish mention")

    # 6. population
    p = load("poblacion_distritos.json")
    p["id"] = p["Dis_id"].astype(int).map(lambda x: f"{x:06d}")
    check(set(p["id"]) == set(tab.index) and p["id"].is_unique, "population table: same 262 distritos")
    p = p.set_index("id").reindex(tab.index)
    pd_ = load("poblacion_departamentos.json")
    check(int(p["Total_Personas"].sum()) == POPULATION == int(pd_["Total"].sum()),
          f"population {int(p['Total_Personas'].sum()):,} = department layer = {POPULATION:,}")
    agebins = [c for c in p.columns if re.fullmatch(r"[HM]_\d+_(\d+|o_mas)", c)]
    check(len(agebins) == 40, f"{len(agebins)} age-sex bins (20 five-year bands, 95+ last)")
    check((p[agebins].sum(axis=1) == p["Total_Personas"]).all(), "age bins sum to each distrito's total")
    dep = p.assign(d=p.index.str[:2]).groupby("d")["Total_Personas"].sum()
    pdd = pd_.assign(d=pd_["dep_id"].astype(int).map(lambda x: f"{x:02d}")).set_index("d")["Total"]
    check((dep == pdd.reindex(dep.index)).all(), "distritos sum to the 14 department totals")
    nm = [(i, names.at[i, "distrito"], p.at[i, "Dis_min"]) for i in tab.index
          if fold(names.at[i, "distrito"]) != fold(p.at[i, "Dis_min"])]
    for x in nm:
        print(f"     name differs on {x[0]}: language table {x[1]!r}, population table {x[2]!r}")
    check(len(nm) <= 5, f"distrito names agree between the two tables ({len(nm)} differ)")
    check((si <= p["Total_Personas"]).all(), "Sí below the population in every distrito")

    # 7. the universe, printed
    w = load("idiomas_departamentos_30sep.json")
    sicol = next(c for c in w.columns if fold(c) == "si")
    w["d"] = w["id_depto"].astype(str).str.zfill(2)
    u3 = (p["Total_Personas"] - 0.6 * (p["H_0_4"] + p["M_0_4"])).groupby(p.index.str[:2]).sum()
    r = (w.set_index("d")[sicol] + w.set_index("d")["No_"]) / u3
    print(f"     age-3+ universe (30 Sept release, Sí + No) over population less 3/5 of 0-4, per "
          f"department: {r.min():.3f} to {r.max():.3f}; national {w[sicol].sum() + w['No_'].sum():,} "
          f"against {u3.sum():,.0f}")
    print(f"     30 Sept release Sí {w[sicol].sum():,} against the revised {int(si.sum()):,}")

    if not ok:
        raise SystemExit("sv_censo: checks FAILED, nothing written")

    rows = []
    for i in tab.index:
        base = dict(geo_id=i, geo_name=names.at[i, "distrito"], municipio=names.at[i, "mun"],
                    departamento=names.at[i, "departamento"], geo_level="distrito")
        for l in LANGS:
            rows.append({**base, "source_category": l, "count": int(tab.at[i, l])})
        rows.append({**base, "source_category": "Sí", "count": int(si[i])})
        rows.append({**base, "source_category": "Población", "count": int(p.at[i, "Total_Personas"])})
    out = pd.DataFrame(rows)
    NORM.mkdir(parents=True, exist_ok=True)
    tmp = NORM / "sv.csv.part"
    out.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(NORM / "sv.csv")
    print(f"  wrote {NORM / 'sv.csv'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
