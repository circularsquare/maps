"""Brazil: the settled immigrant-descended language communities the 2022 census cannot see
(it asked language only of indigenous people) -> data/normalized/br_settlers.csv
(unit = município code, node, count). Read by countries/br.py, which takes them out of the
Portuguese remainder. Rows are `modelled`: published speaker estimates placed by homeland
(ask 019's route; sources/eg.md is the model). Record: sources/br.md, "Settled communities".

    python sources/br_settlers.py

THE ESTIMATES (each a published figure for a named region, used as printed):
- Hunsrik (Riograndenser Hunsrik). Altenhofen, Morello et al., "Hunsrückisch: inventário de
  uma língua do Brasil" (IPOL/Garapuvu, 2018; data/raw/br/hunsrik_inventario_2018.pdf, p.120):
  80% of Rio Grande do Sul's ~980,000 German speakers (BIRS survey of 18-year-old conscripts,
  1988-90, x the 1991 population) = 784,000; Santa Catarina 294,000; southwestern Paraná
  98,000; a further 49,000 "in the other regions (Amazon, Espírito Santo, Argentina and
  Paraguay)" is not drawn, since it mixes countries and names no place.
- Talian. The same inventory's Table 6 (p.117): Italian speakers in Rio Grande do Sul, BIRS
  1990, 6.43% = 587,411. Italian spoken at home in Rio Grande do Sul is Talian (Brazilian
  Venetian koine); IPHAN's INDL page says no national estimate exists, so Santa Catarina,
  Paraná and the rest are not drawn.
- East Pomeranian (Pomerano). Espírito Santo, "estimated at 120,000" (Portuguese Wikipedia,
  Pomeranos, citing IPOL 2014, "Espírito Santo investe na preservação da língua pomerana":
  "cerca de 120 mil dos estimados 300 mil descendentes de pomeranos"). Descendants, which the
  sources use interchangeably with speakers; Santa Maria de Jetibá's linguistic census found
  most Pomeranians there fluent. Other states' figures are descendants of unknown source, not
  drawn.

PLACEMENT INSIDE EACH REGION (the region's total never moves):
- German varieties (Hunsrik, Pomerano): by Lutherans per município (Censo 2010, SIDRA 2094,
  "Igreja Evangélica Luterana", read from religiondots' normalized br.csv, read-only) as a share
  of the 2010 population, times the 2022 population. German Brazilians are the core of Brazil's
  Lutheran churches. Catholic German towns (Hunsrik is spoken by Catholics as much as
  Lutherans) are caught only where a municipal law makes Hunsrik co-official: those take their
  whole 2022 population as the weight. Southwestern Paraná's Hunsrik speakers came from Rio
  Grande do Sul's Catholic colonies, so there the weight is plain population over the
  mesorregião (IBGE 4107, Sudoeste Paranaense).
- Talian: plain 2022 population over the IBGE microrregiões that hold a município with Talian
  co-official (the Serra Gaúcha and the Quarta Colônia).
- No município is drawn above 70% of its people on one of these languages: IPOL gives 70% of
  Santa Maria de Jetibá's 34,000 people as of Pomeranian origin, the highest share any source
  gives. What a cap takes off is spread over the region's other municípios by the same weight.
  All the settler languages together are held to 90% of a município.

CO-OFFICIAL LAWS: Portuguese Wikipedia's Hunsriqueano rio-grandense and Língua pomerana pages
and the 2024 AFTM/Anafisco list "Municípios brasileiros que possuem línguas co-oficiais",
read 2026-10-06.
"""
import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "br"
NORM = ROOT / "data" / "normalized"
RD = ROOT.parent / "religiondots"
SETORES = RD / "data" / "geo" / "br" / "br_setores_2022.gpkg"

HUNSRIK = "indoeuropean.germanic.continental.hunsrik"
POMERANO = "indoeuropean.germanic.continental.lowgerman.pomeranian"
TALIAN = "indoeuropean.romance.talian"
CAP = 0.70
BOOST_MAX = 30_000   # co-official towns above this keep their Lutheran weight (Ijuí, 83,000)
JOINT_CAP = 0.90

COOFFICIAL = {
    HUNSRIK: [("Santa Maria do Herval", "RS"), ("Barão", "RS"), ("Harmonia", "RS"),
              ("Horizontina", "RS"), ("Ijuí", "RS"), ("Antônio Carlos", "SC"),
              ("São João do Oeste", "SC"), ("Ipumirim", "SC"), ("Itá", "SC")],
    POMERANO: [("Afonso Cláudio", "ES"), ("Domingos Martins", "ES"), ("Itarana", "ES"),
               ("Laranja da Terra", "ES"), ("Pancas", "ES"), ("Santa Maria de Jetibá", "ES"),
               ("Vila Pavão", "ES")],
    TALIAN: [("Serafina Corrêa", "RS"), ("Flores da Cunha", "RS"), ("Nova Roma do Sul", "RS"),
             ("Paraí", "RS"), ("Bento Gonçalves", "RS"), ("Fagundes Varela", "RS"),
             ("Antônio Prado", "RS"), ("Guabiju", "RS"), ("Camargo", "RS"),
             ("Caxias do Sul", "RS"), ("Ivorá", "RS"), ("Pinto Bandeira", "RS"),
             ("Nova Pádua", "RS"), ("Barão", "RS"), ("Casca", "RS")],
}

# (node, region kind, region key, speakers, weight rule)
ESTIMATES = [
    (HUNSRIK, "uf", "RS", 784_000, "lutheran"),
    (HUNSRIK, "uf", "SC", 294_000, "lutheran"),
    (HUNSRIK, "meso", 4107, 98_000, "pop"),
    (TALIAN, "talian", None, 587_411, "pop"),
    (POMERANO, "uf", "ES", 120_000, "lutheran_only"),
]


def municipios():
    ibge = json.loads((RAW / "ibge_municipios.json").read_text(encoding="utf-8"))
    sig = {str(m["regiao-imediata"]["regiao-intermediaria"]["UF"]["id"]):
           m["regiao-imediata"]["regiao-intermediaria"]["UF"]["sigla"] for m in ibge}
    rows = []
    for m in ibge:
        code = str(m["id"])
        mi = m["microrregiao"]
        rows.append((code, m["nome"], sig[code[:2]], mi["id"] if mi else None,
                     mi["mesorregiao"]["id"] if mi else None))
    return pd.DataFrame(rows, columns=["unit", "name", "uf", "micro", "meso"]).set_index("unit")


def lutheran_share():
    r = pd.read_csv(RD / "data" / "normalized" / "br.csv", dtype={"geo_id": str},
                    usecols=["geo_id", "source_category", "count", "year"])
    r = r[r["year"] == 2010]
    tot = r[r["source_category"] == "Total"].groupby("geo_id")["count"].sum()
    lut = r[r["source_category"].str.contains("Igreja Evangélica Luterana", regex=False)]
    lut = lut.groupby("geo_id")["count"].sum()
    return (lut / tot).fillna(0), int(lut.sum())


def population():
    import pyogrio
    p = pyogrio.read_dataframe(SETORES, read_geometry=False, columns=["unit", "pop"])
    p["unit"] = p["unit"].astype(str)
    return p.groupby("unit")["pop"].sum().astype(float)


def place(total, weight, pop, cap=CAP):
    """Spread `total` over municípios by `weight`, no município above cap x pop; what a cap
    takes off goes to the uncapped ones by weight. -> Series (may fall short if all cap)."""
    out = pd.Series(0.0, index=weight.index)
    free = weight[weight > 0].index
    left = float(total)
    for _ in range(50):
        w = weight[free]
        if left < 0.5 or w.sum() <= 0:
            break
        add = left * w / w.sum()
        room = cap * pop[free] - out[free]
        take = add.clip(upper=room)
        out[free] += take
        left -= float(take.sum())
        free = free[(room - take) > 0.5]
    return out


def main():
    ok = True

    def check(c, m):
        nonlocal ok
        print(("ok    " if c else "FAIL  ") + m)
        ok &= bool(c)

    mun = municipios()
    pop = population()
    # IBGE's list also has Boa Esperança do Norte (5101837), created after the census
    check(len(pop) == 5570 and set(pop.index) <= set(mun.index),
          f"{len(pop):,} municípios with 2022 population, all in IBGE's list")
    lshare, lsum = lutheran_share()
    print(f"  Lutherans 2010: {lsum:,} in {int((lshare > 0).sum()):,} municípios")
    key = {(n, u): c for c, n, u in zip(mun.index, mun["name"], mun["uf"])}
    co = {}
    for node, towns in COOFFICIAL.items():
        miss = [t for t in towns if t not in key]
        check(not miss, f"{node.split('.')[-1]}: {len(towns)} co-official towns found ({miss})")
        co[node] = {key[t] for t in towns if t in key}
    talian_micro = set(mun.loc[list(co[TALIAN]), "micro"]) if co[TALIAN] else set()
    talian_rs = mun[mun["micro"].isin(talian_micro) & (mun["uf"] == "RS")].index
    pomerano_rs = mun.index[mun["micro"] == mun.loc[key[("Canguçu", "RS")], "micro"]]
    print(f"  Pelotas microrregião, kept off Hunsrik: "
          + ", ".join(mun.loc[pomerano_rs, "name"]))

    parts = []
    for node, kind, k, n, rule in ESTIMATES:
        if kind == "uf":
            region = mun.index[mun["uf"] == k]
        elif kind == "meso":
            region = mun.index[mun["meso"] == k]
        else:
            region = talian_rs
        p = pop.reindex(region).fillna(0)
        if rule.startswith("lutheran"):
            w = lshare.reindex(region).fillna(0) * p
            if rule == "lutheran":
                boost = [u for u in co[node] if u in region and p[u] < BOOST_MAX]
                w[boost] = p[boost]
            if node == HUNSRIK:
                # the Pelotas microrregião's Lutherans are Pomeranians (Canguçu, São Lourenço
                # do Sul): part of the inventory's other 20%, not Hunsrik; left on Portuguese
                w[w.index.isin(pomerano_rs)] = 0
        else:
            w = p.copy()
        got = place(n, w, p)
        short = n - got.sum()
        top = got.sort_values(ascending=False).head(4)
        print(f"  {node.split('.')[-1]:9} {kind} {k}: {n:,} over {int((got > 0).sum()):,} "
              f"municípios (population {p.sum():,.0f}); capped {int((got >= CAP * p - 0.5).sum())}"
              f"; short {short:,.0f}; top " + ", ".join(
                  f"{mun.loc[u, 'name']} {v:,.0f} ({v / p[u]:.0%})" for u, v in top.items()))
        check(short < 1, f"{node.split('.')[-1]} {k}: the whole estimate placed")
        parts.append(pd.DataFrame({"unit": got.index, "node": node, "count": got.values}))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["count"] > 0].groupby(["unit", "node"], as_index=False)["count"].sum()
    s = df.groupby("unit")["count"].sum()
    over = s[s > JOINT_CAP * pop.reindex(s.index)]
    if len(over):
        f = (JOINT_CAP * pop.reindex(over.index) / over)
        df["count"] *= df["unit"].map(f).fillna(1.0)
        print(f"  held to {JOINT_CAP:.0%} jointly in {len(over)} municípios: "
              + ", ".join(mun.loc[over.index[:6], "name"]))
    check((df.groupby("unit")["count"].sum() <= pop.reindex(df["unit"].unique()) + 0.5).all(),
          "no município over its population")
    if not ok:
        raise SystemExit("checks failed")
    df.to_csv(NORM / "br_settlers.csv", index=False)
    print("wrote br_settlers.csv: " + ", ".join(
        f"{k.split('.')[-1]} {v:,.0f}" for k, v in df.groupby("node")["count"].sum().items()))


if __name__ == "__main__":
    main()
