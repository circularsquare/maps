"""Brazil, Censo Demográfico 2022: indigenous languages spoken at home, per município.

    python sources/br_censo.py [--fetch]

Writes
  data/normalized/br.csv               one row per (município, language label) as published:
                                       Tabela complementar 26, people counted once per language
                                       they named (up to three each)
  data/normalized/br_status.csv        per município, SIDRA 10392: indigenous people aged 2+ by
                                       how many indigenous languages they named
  data/processed/br_setor_indigenous.csv   per census setor, indigenous people (V01690), the
                                       placement weight for indigenous-language dots

THE QUESTION. Asked only of indigenous people aged 2 or over: "fala ou utiliza língua indígena
no domicílio?", then which, up to three, with no order of importance. Non-indigenous people are
not asked anything about language, so this table says nothing about German, Pomeranian, Talian
or Japanese.

THE TABLES.
  Tabela complementar 26 (xlsx, "Etnias e Línguas Indígenas", Resultados do Universo): every
    município x language label, 8,061 rows, 310 labels. A person who named two languages is
    in two rows, so a município's rows sum to MENTIONS, not people.
  SIDRA 10392 at N6: indigenous people aged 2+ per município by "status de declaração de língua"
    (one, two or three languages; não-determinada; mal definida; não sabe; does not speak one).
    It gives the people behind the mentions.
  Tabela complementar 21: the national total per language, for the per-label check.
  Agregados por setores, pessoas indígenas: V01690 per setor.

CHECKS (all must hold, or the script stops):
  1. per município, T26's rows sum to one + 2*two + 3*three + não-determinada + mal definida +
     não sabe from SIDRA 10392, exactly; and T26's three non-language rows equal SIDRA's three
     status counts, exactly.
  2. per label, T26 summed over municípios equals Tabela complementar 21's national figure.
  3. nationally: 474,856 speakers aged 2+, the figure IBGE publishes.
  4. reported, not enforced: setor V01690 summed per município against SIDRA's indigenous people
     aged 2+. IBGE suppresses small setor counts ("X"), so the setores fall short.
"""
import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "br"
NORM = HERE / "data" / "normalized"
PROC = HERE / "data" / "processed"

FTP = "http://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/"
ETN = FTP + ("Etnias_e_Linguas_Indigenas_principais_caracteristicas_sociodemograficas_"
             "Resultados_do_universo/Tabelas_complementares/xlsx/")
SETOR_ZIP = ("Agregados_por_Setores_Censitarios/Agregados_por_Setor_csv/"
             "Agregados_por_setores_pessoas_indigenas_BR.zip")
SIDRA = ("https://servicodados.ibge.gov.br/api/v3/agregados/10392/periodos/2022/variaveis/13245"
         "?localidades=N6[all]&classificacao=1168[all]|15[201]|1336[57961]|1440[58106]")
FILES = {
    "Tabela_complementar_26.xlsx": ETN + "Tabela_complementar_26.xlsx",
    "Tabela_complementar_21.xlsx": ETN + "Tabela_complementar_21.xlsx",
    "Agregados_por_setores_pessoas_indigenas_BR.zip": FTP + SETOR_ZIP,
    "sidra_10392_n6.json": SIDRA,
}
NATIONAL_SPEAKERS = 474_856
STATUS = {"Total": "total", "Declarou uma língua indígena": "one",
          "Declarou duas línguas indígenas": "two", "Declarou três línguas indígenas": "three",
          "Declaração não-determinada": "nd", "Declaração mal definida": "md",
          "Não sabe": "ns", "Não fala língua indígena no domicílio": "none"}
# T26's rows that are a status, not a language
NOT_LANG = {"Não determinada": "nd", "Mal definida": "md", "Não sabe": "ns"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        print("fetching", name)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            dest.write_bytes(r.read())


def read_t26():
    t = pd.read_excel(RAW / "Tabela_complementar_26.xlsx", header=None, skiprows=8, dtype=str)
    t.columns = ["uf", "uf_name", "geo_id", "geo_name", "code", "source_category", "count"]
    t = t[t["geo_id"].str.fullmatch(r"\d{7}", na=False)].copy()
    t["count"] = pd.to_numeric(t["count"].replace("-", "0")).astype(int)
    t["source_category"] = t["source_category"].str.strip()
    assert t.groupby("code")["source_category"].nunique().max() == 1
    return t


def read_status():
    js = json.loads((RAW / "sidra_10392_n6.json").read_text(encoding="utf-8"))
    rows = []
    for res in js[0]["resultados"]:
        cat = list(res["classificacoes"][0]["categoria"].values())[0]
        for s in res["series"]:
            v = s["serie"]["2022"]
            rows.append((s["localidade"]["id"], STATUS[cat], 0 if v in ("-", "...", "X", "..") else int(v)))
    s = pd.DataFrame(rows, columns=["geo_id", "status", "n"])
    s = s.pivot(index="geo_id", columns="status", values="n").fillna(0).astype(int)
    return s[list(STATUS.values())].reset_index()


def read_t21():
    """Pairs of (label, people) laid out in six column pairs; family rows are included and
    simply never match a language label."""
    t = pd.read_excel(RAW / "Tabela_complementar_21.xlsx", header=None, skiprows=5)
    out = {}
    for c in range(0, t.shape[1] - 1, 2):
        for name, v in zip(t[c], t[c + 1]):
            if isinstance(name, str) and pd.notna(v) and str(v).strip() not in ("", "-"):
                try:
                    out.setdefault(name.strip(), []).append(int(v))
                except ValueError:
                    pass
    return out


def read_setores():
    z = zipfile.ZipFile(RAW / "Agregados_por_setores_pessoas_indigenas_BR.zip")
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), sep=";",
                         usecols=["CD_SETOR", "V01690"], dtype=str)
    df["suppressed"] = df["V01690"].eq("X")
    df["indig"] = pd.to_numeric(df["V01690"], errors="coerce").fillna(0).astype(int)
    return df.rename(columns={"CD_SETOR": "setor"})[["setor", "indig", "suppressed"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not all((RAW / n).exists() for n in FILES):
        fetch()

    t = read_t26()
    st = read_status()
    print(f"T26: {len(t):,} rows, {t['geo_id'].nunique():,} municípios, "
          f"{t['source_category'].nunique()} labels, {t['count'].sum():,} mentions")

    # 1. per município, mentions = people x languages named
    m = t.groupby("geo_id")["count"].sum()
    flags = t[t["source_category"].isin(NOT_LANG)].pivot_table(
        index="geo_id", columns="source_category", values="count", aggfunc="sum").fillna(0)
    s = st.set_index("geo_id")
    want = s["one"] + 2 * s["two"] + 3 * s["three"] + s["nd"] + s["md"] + s["ns"]
    both = pd.concat([m.rename("t26"), want.rename("sidra")], axis=1).fillna(0)
    bad = both[both["t26"] != both["sidra"]]
    if len(bad):
        raise SystemExit(f"CHECK 1 failed in {len(bad)} municípios:\n{bad.head()}")
    for lab, col in NOT_LANG.items():
        got = flags.get(lab, pd.Series(dtype=float)).reindex(s.index).fillna(0)
        if (got != s[col]).any():
            raise SystemExit(f"CHECK 1 failed: T26 '{lab}' differs from SIDRA {col}")
    print(f"check 1 ok: {len(both):,} municípios, T26 mentions = SIDRA people x languages named, exactly")

    # 2. per label against the national table
    t21 = read_t21()
    nat = t.groupby("source_category")["count"].sum()
    miss, diff = [], []
    for lab, v in nat.items():
        if lab in NOT_LANG or lab == "Outra língua das Américas":
            continue
        cands = t21.get(lab) or t21.get(lab.replace(" (*)", "")) or t21.get(lab.replace(" (**)", ""))
        if not cands:
            miss.append(lab)
        elif v not in cands:
            diff.append((lab, v, cands))
    print(f"check 2: {len(nat) - len(miss) - len(diff) - 4} labels equal Tabela 21's national figure; "
          f"{len(miss)} not found there by name {miss[:8]}; {len(diff)} differ {diff[:8]}")
    # Five labels (Ka'apor, Tiriyó, Wapixana, Xavante, Yanomami) are 1-3 people short of
    # Tabela 21 in T26, which agrees with SIDRA per município exactly (check 1). Tolerated.
    if miss or any(min(abs(v - c) for c in cs) > 5 for _, v, cs in diff):
        raise SystemExit("CHECK 2 failed")

    # 3. national
    speakers = int((s["one"] + s["two"] + s["three"] + s["nd"] + s["md"] + s["ns"]).sum())
    if speakers != NATIONAL_SPEAKERS:
        raise SystemExit(f"CHECK 3 failed: {speakers:,} speakers, IBGE publishes {NATIONAL_SPEAKERS:,}")
    print(f"check 3 ok: {speakers:,} speakers aged 2+; {int(s['two'].sum()):,} named two languages, "
          f"{int(s['three'].sum()):,} three")

    # 4. setores
    se = read_setores()
    se["geo_id"] = se["setor"].str[:7]
    ind = se.groupby("geo_id")["indig"].sum()
    chk = pd.concat([ind.rename("setor_all_ages"), s["total"].rename("sidra_2plus"),
                     (s["total"] - s["none"]).rename("speakers")], axis=1).fillna(0)
    under = chk[chk["setor_all_ages"] < chk["sidra_2plus"]]
    nopeople = chk[(chk["speakers"] > 0) & (chk["setor_all_ages"] == 0)]
    print(f"check 4: setor V01690 {int(se['indig'].sum()):,} indigenous people (all ages) in "
          f"{len(se):,} setores; SIDRA 2+ {int(s['total'].sum()):,}; "
          f"{len(under)} municípios where setores hold fewer than SIDRA's 2+; "
          f"{len(nopeople)} with speakers but no indigenous people in setores")
    # Not a failure: IBGE suppresses ("X") a setor's indigenous count where it would identify
    # people, so the setor file holds less than the município tables. countries/br.py spreads
    # each município's shortfall over its setores by population (see _BrWeighter there).
    print(f"  {int(se['suppressed'].sum()):,} setores suppressed (X); shortfall against SIDRA 2+ "
          f"{int((chk['sidra_2plus'] - chk['setor_all_ages']).clip(lower=0).sum()):,} people")
    if len(under):
        print(under.assign(gap=under["sidra_2plus"] - under["setor_all_ages"]).sort_values("gap").tail(5))

    NORM.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    t.assign(geo_level="municipio")[["geo_id", "geo_level", "geo_name", "uf", "code",
                                     "source_category", "count"]].to_csv(NORM / "br.csv", index=False)
    st.to_csv(NORM / "br_status.csv", index=False)
    se[["setor", "indig"]].to_csv(PROC / "br_setor_indigenous.csv", index=False)
    print("wrote br.csv, br_status.csv, br_setor_indigenous.csv")


if __name__ == "__main__":
    main()
