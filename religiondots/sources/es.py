"""Spain — two populations, two sources, one exact partition.

Writes data/normalized/es.csv: religion x 52 provinces, as PEOPLE.

Usage:
    python sources/es.py --fetch     # CIS microdata, INE, Pew, UCIDE  (~400 MB)
    python sources/es.py             # rebuild es.csv from data/raw/es/

THE FINDING THIS COUNTRY IS BUILT ON. Spain has no religion census — INE has never asked —
and the obvious substitute is the CIS barómetro, which asks every single month and has done
since the 1970s. Pooled over three years that is ~140,000 respondents with a province code,
which is enough for all 52 provinces. But CIS's `NACIONALIDAD` variable takes exactly two
values, "la nacionalidad española" and "la nacionalidad española y otra", and that is the
whole story: **the barómetro does not sample foreign nationals at all.** 7.4 million people,
15% of Spain and almost all of its religious variety, are outside the frame rather than
undersampled in it.

Which is why CIS's "creyente de otra religión" reads 3.2% while UCIDE alone puts Muslims at
5% of residents — the two numbers are about different populations and they reconcile once
you notice. 3.2% of 42.4M citizens is ~1.36M; UCIDE's estimate of Spanish-CITIZEN Muslims is
1.09M and fits inside it. So Spain is drawn as two halves that partition the country exactly:

    Spanish citizens   42.4M   CIS barómetro, pooled, self-identification, per province
    foreign nationals   7.4M   INE by province x nationality, x Pew's country composition

Neither is a subset of the other, nothing is counted twice, and the sum is INE's own
population. spec §3.1 is satisfied WITHIN each half and the halves are never added inside a
category without saying so — every row carries `tier`, and the foreign half is `modelled`
throughout.

THE FOUR SOURCES, and what each one is load-bearing for:

  CIS barómetro       magnitude and geography for 86% of the country. Free microdata, no
                      key, no registration. The site's HTML is behind a bot wall and the
                      document paths are not — see the URL note in _cis_url().
  INE ECP + Padrón    the population each half is scaled to, and the nationality mix inside
                      the foreign half. Keyless PC-Axis and a keyless JSON API.
  Pew 2020            religious composition of each origin country. Seven families; the
                      Christian and Muslim splits are hand-authored in taxonomy/es_origin.py.
  UCIDE               the ONLY thing that splits CIS's other-religion cell. Province-level
                      Spanish-citizen Muslims, from a confessional federation's own annual
                      study. spec §14.9 is what permits it.

WHAT IS DELIBERATELY NOT DONE. No attempt is made to place a religion inside a province.
Nothing in Spain measures that — the Observatorio del Pluralismo Religioso's 7,756 geocoded
non-Catholic places of worship are the only candidate and they are a §4.4 location layer, not
a weight — so every node in a province is spread over its municipios by population, exactly
as sources/ru.py does with Kontur hexes, and a Muslim dot sits where the province's people
are rather than where its Muslims are. countries.py's `note` says so.
"""

import argparse
import io
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
import zipfile

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
RAW = os.path.join(ROOT, "data", "raw", "es")
OUT = os.path.join(ROOT, "data", "normalized", "es.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "es_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# --- CIS ------------------------------------------------------------------------------
# Every barómetro is one "estudio" with a four-digit number. Pool from here up: 3390 is
# roughly mid-2023, so the window is about three years. Longer would add sample and blur a
# real trend (the Catholic share fell from 58% to 55% over it); shorter would leave Soria and
# Teruel with a few hundred respondents. RESCALE_MONTHS re-levels the pooled structure onto
# the most recent national shares, spec §3.4.
CIS_FROM, CIS_TO = 3390, 3600
CIS_MIN_N = 1500
CIS_MIN_PROVINCES = 45
RESCALE_RECENT = 12          # most recent N qualifying studies define the national margin

# --- INE ------------------------------------------------------------------------------
# Detailed foreign nationality by province. The Padrón series that carries it ends at 2022;
# the current ECP publishes province x Spanish/foreign only. spec §3.4's rule applies:
# structure from the detailed source, totals from the recent one.
INE_NAT_PX = ("https://www.ine.es/jaxi/files/_px/es/px/t20/e245/p08/l0/03005.px")
INE_NAT_YEAR = 2022
INE_POP_TABLE = 82104        # provinces x age x Spanish/foreign x sex, current quarter

PEW_ZIP = ("https://www.pewresearch.org/wp-content/uploads/sites/20/2025/06/"
           "Religious-Composition-2010-2020-dataset.zip")
UCIDE_PDF = "https://ucide.org/wp-content/uploads/2024/02/estademograf23.pdf"

# INE's aggregate rows in the nationality axis. Everything else is a leaf and the leaves
# partition TOTAL EXTRANJEROS exactly — asserted in _foreign_half().
INE_NAT_AGGREGATES = {
    "TOTAL EXTRANJEROS", "EUROPA", "UNIÓN EUROPEA", "UE(15)", "UE(25)", "UE(27)", "UE(28)",
    "UE(27_2020)", "EUROPA NO UE(28)", "EUROPA NO UE(27_2020)", "ÁFRICA", "AMÉRICA",
    "AMÉRICA CENTRAL Y CARIBE", "AMÉRICA DEL NORTE", "AMÉRICA DEL SUR", "ASIA", "OCEANÍA",
}


# =======================================================================================
# fetch
# =======================================================================================

def _get(url, dest, note=""):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"  {os.path.basename(dest)}  {os.path.getsize(dest):,} bytes {note}")
    return True


def _cis_url(n):
    """CIS microdata lives at /documents/d/guest/<name>, and there are TWO names.

    Studies up to about 3500 are `MD<n>`; later ones are `MD<n>-zip`, which is Liferay
    rendering the filename `MD<n>.zip` with the dot turned into a dash. Neither is
    discoverable from the site, because cis.es puts its catalogue and study pages behind a
    BunkerWeb bot wall that answers a plain fetch with a 15KB challenge page — while leaving
    the document paths themselves completely open. §9s's KOSIS wall, inverted: there the
    metadata was open and the data walled.
    """
    return [f"https://www.cis.es/documents/d/guest/MD{n}-zip",
            f"https://www.cis.es/documents/d/guest/MD{n}"]


def fetch():
    os.makedirs(RAW, exist_ok=True)
    print("CIS barómetros…")
    manifest = {}
    mpath = os.path.join(RAW, "cis_manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath))
    for n in range(CIS_FROM, CIS_TO + 1):
        dest = os.path.join(RAW, "cis", f"MD{n}.zip")
        if str(n) in manifest and os.path.exists(dest):
            continue
        for url in _cis_url(n):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=300) as r:
                    if "zip" not in r.headers.get("Content-Type", ""):
                        continue
                    blob = r.read()
            except Exception:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(blob)
            manifest[str(n)] = {"url": url, "size": len(blob)}
            print(f"  {n} {len(blob):>9,}")
            break
        time.sleep(0.1)
    json.dump(manifest, open(mpath, "w"), indent=0)
    print(f"  {len(manifest)} studies on disk")

    print("INE…")
    _get(INE_NAT_PX, os.path.join(RAW, "ine_03005.px"),
         "(foreign residents by province x nationality)")
    url = (f"https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/{INE_POP_TABLE}"
           f"?nult=1&det=2")
    _get(url, os.path.join(RAW, "ine_pop.json"), "(province x Spanish/foreign, current)")

    print("Pew…")
    _get(PEW_ZIP, os.path.join(RAW, "pew.zip"), "(religious composition by country 2020)")

    print("UCIDE…")
    _get(UCIDE_PDF, os.path.join(RAW, "ucide.pdf"), "(Muslim population by province)")


# =======================================================================================
# CIS: the Spanish-citizen half
# =======================================================================================

def _cis_tables(zpath):
    """(num dataframe, ES syntax text) from one MD zip, or (None, None).

    Two layouts, both live: older studies carry `<n>_num.csv` at the top level, newer ones
    nest both CSVs inside `<n>_csv.zip`. The .sav is present in every one and is not used —
    there is no pyreadstat wheel for this interpreter, and the CSV is the same matrix.
    """
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        num = [n for n in names if n.lower().endswith("_num.csv")]
        syn = [n for n in names if re.fullmatch(r"ES\d+", os.path.basename(n), re.I)]
        text = z.read(syn[0]).decode("iso-8859-1", "replace") if syn else ""
        if num:
            return pd.read_csv(io.BytesIO(z.read(num[0])), sep=";", dtype=str,
                               encoding="utf-8-sig", low_memory=False), text
        inner = [n for n in names if n.lower().endswith("_csv.zip")]
        if not inner:
            return None, text
        with zipfile.ZipFile(io.BytesIO(z.read(inner[0]))) as z2:
            num = [n for n in z2.namelist() if n.lower().endswith("_num.csv")]
            if not num:
                return None, text
            return pd.read_csv(io.BytesIO(z2.read(num[0])), sep=";", dtype=str,
                               encoding="utf-8-sig", low_memory=False), text


def _col(df, name):
    """CIS CSV headers are `NAME: human label`; match on the name half."""
    for c in df.columns:
        if c.split(":")[0].strip().upper() == name:
            return c
    return None


def _religion_labels(text):
    """{code: label} for RELIGION, read out of a study's SPSS syntax file."""
    m = re.search(r"/?RELIGION\s+((?:\s*\d+\s*'[^']*')+)", text)
    if not m:
        return None
    return {int(c): " ".join(lab.split())
            for c, lab in re.findall(r"(\d+)\s*'([^']*)'", m.group(1))}


def _pool_cis():
    """Pooled weighted respondents per (province, RELIGION code), plus the recent margin."""
    import es2026
    cdir = os.path.join(RAW, "cis")
    if not os.path.isdir(cdir):
        sys.exit("no data/raw/es/cis — run `python sources/es.py --fetch` first")
    used, skipped, rejected = [], [], []
    frames = []
    for zpath in sorted(os.listdir(cdir)):
        if not zpath.endswith(".zip"):
            continue
        study = int(re.search(r"\d+", zpath).group())
        try:
            df, text = _cis_tables(os.path.join(cdir, zpath))
        except Exception as e:
            skipped.append((study, f"unreadable: {e}"))
            continue
        if df is None:
            skipped.append((study, "no _num.csv"))
            continue
        prov, rel = _col(df, "PROV"), _col(df, "RELIGION")
        if prov is None or rel is None:
            skipped.append((study, "no PROV or no RELIGION"))
            continue
        if len(df) < CIS_MIN_N:
            skipped.append((study, f"n={len(df)}"))
            continue
        d = pd.DataFrame({
            "prov": pd.to_numeric(df[prov], errors="coerce"),
            "code": pd.to_numeric(df[rel], errors="coerce"),
        })
        wcol = _col(df, "PESO") or _col(df, "PESOCCAA")
        d["w"] = (pd.to_numeric(df[wcol].str.replace(",", "."), errors="coerce").fillna(1.0)
                  if wcol else 1.0)
        d = d[d["prov"].between(1, 52)
              & d["code"].isin(sorted({1, 2, 3, 4, 5, 6} | es2026.NC_CODES))]
        d["nc"] = d["code"].isin(es2026.NC_CODES)
        if d["prov"].nunique() < CIS_MIN_PROVINCES:
            skipped.append((study, f"{d['prov'].nunique()} provinces"))
            continue
        labels = _religion_labels(text)
        if labels is None:
            skipped.append((study, "no RELIGION in syntax"))
            continue
        why = es2026.accepts(labels)
        if why:
            rejected.append((study, why))
            continue
        d["study"] = study
        frames.append(d)
        used.append(study)

    if rejected:
        print(f"  REJECTED {len(rejected)} studies whose RELIGION is a different question: "
              + "; ".join(f"{s} ({w})" for s, w in rejected))
    if not frames:
        sys.exit("!! no usable CIS studies found")
    pool = pd.concat(frames, ignore_index=True)
    print(f"  pooled {len(used)} studies ({min(used)}–{max(used)}), "
          f"{len(pool):,} respondents, {pool['prov'].nunique()} provinces")
    if skipped:
        print(f"  skipped {len(skipped)}: "
              + ", ".join(f"{s}({r})" for s, r in skipped[:6])
              + (" …" if len(skipped) > 6 else ""))
    recent = sorted(used)[-RESCALE_RECENT:]
    return pool, recent


def _nc_share(pool):
    """Weighted share of Spanish citizens who refuse the question, per province.

    spec §3.5 marks non-response rather than filling it, so the citizen population each
    province's shares are applied to is reduced by this. Russia does the same with Arena's
    "difficult to answer" and draws 94.5% of its people; Spain's refusal rate is far lower
    and the effect is about a percentage point.
    """
    nc = pool[pool["nc"]].groupby("prov")["w"].sum()
    allw = pool.groupby("prov")["w"].sum()
    return (nc / allw).reindex(sorted(allw.index)).fillna(0.0)


def _shares_by_province(pool, recent):
    """Province x code shares from the whole pool, re-levelled onto the recent margin.

    spec §3.4, and it is the same operation §9d used for New Zealand: STRUCTURE from the
    source with the geography, TOTALS from the source that is current. Here both come from
    CIS, three years apart — the pool is what makes the small provinces estimable and the
    last twelve waves are what makes the national figure current. One-margin IPF, which
    converges in a handful of passes because there is only one margin to hit.
    """
    tab = (pool[~pool["nc"]].groupby(["prov", "code"])["w"].sum()
           .unstack(fill_value=0.0))
    # N.C. (code 7 or 9, depending on the study) is already dropped at read time —
    # spec §3.5, see es2026.EXCLUDED and NC_CODES.
    tab = tab.reindex(columns=[1, 2, 3, 4, 5, 6], fill_value=0.0)
    share = tab.div(tab.sum(axis=1), axis=0)

    rec = pool[pool["study"].isin(recent) & ~pool["nc"]]
    margin = rec.groupby("code")["w"].sum()
    margin = (margin / margin.sum()).reindex(share.columns).fillna(0.0)

    w = tab.sum(axis=1)                              # province weight, for the margin
    for _ in range(50):
        cur = share.mul(w, axis=0).sum() / w.sum()
        adj = (margin / cur.replace(0, pd.NA)).fillna(1.0)
        share = share.mul(adj, axis=1)
        share = share.div(share.sum(axis=1), axis=0)
        if float((cur - margin).abs().max()) < 1e-12:
            break
    print("  national shares after re-levelling: "
          + "  ".join(f"{c}:{100 * v:.1f}%" for c, v in
                      (share.mul(w, axis=0).sum() / w.sum()).items()))
    return share


# =======================================================================================
# INE
# =======================================================================================

def read_px(path):
    """Minimal PC-Axis reader — enough for INE's t20/e245 tables and nothing more."""
    raw = open(path, "rb").read().decode("iso-8859-15", errors="replace")
    head, data = raw.split("DATA=", 1)

    def axis(kw):
        m = re.search(kw + r"=(.*?);", head, re.S)
        return re.findall(r'"(.*?)"', m.group(1)) if m else []

    dims = axis("STUB") + axis("HEADING")
    levels = []
    for d in dims:
        m = re.search(r'VALUES\("' + re.escape(d) + r'"\)=(.*?);', head, re.S)
        levels.append(re.findall(r'"(.*?)"', m.group(1)))
    nums = data.replace(";", "").split()
    n = 1
    for lv in levels:
        n *= len(lv)
    if len(nums) != n:
        sys.exit(f"!! {os.path.basename(path)}: {len(nums)} cells for a {n}-cell cube")
    idx = pd.MultiIndex.from_product(levels, names=dims)
    vals = [float(x.replace(",", ".")) if x not in ("..", ".", "-") else float("nan")
            for x in nums]
    return pd.Series(vals, index=idx, name="value").reset_index()


def _province_population():
    """Current Spanish-national and foreign population per province, from ECP.

    The API returns one series per (province, age group, nationality, sex) with its name as a
    dotted string; `Todas las edades` and `Total` (sex) are the cells wanted. Province names
    are matched to codes through the nationality px file's own axis, which carries the code
    and the name in one string ("28 Madrid"), so no external crosswalk is needed.
    """
    rows = json.load(open(os.path.join(RAW, "ine_pop.json"), encoding="utf-8"))
    out = {}
    period = None
    for s in rows:
        parts = [p.strip() for p in (s.get("Nombre") or "").split(".")]
        if len(parts) < 4:
            continue
        prov, age, nat, sex = parts[0], parts[1], parts[2], parts[3]
        if age != "Todas las edades" or sex != "Total":
            continue
        if nat not in ("Española", "Extranjera", "Total"):
            continue
        data = s.get("Data") or []
        if not data:
            continue
        period = data[0].get("NombrePeriodo", period)
        out.setdefault(prov, {})[nat] = float(data[0]["Valor"])
    print(f"  INE population as of {period}")
    return out, period


def _by_code(pop):
    """{province code: {'Española': n, 'Extranjera': n, 'Total': n}} from ECP's names."""
    lookup = {_norm(v): k for k, v in PROV_NAMES.items()}
    out, unmatched = {}, []
    for label, d in pop.items():
        if label == "Total Nacional":
            continue
        code = lookup.get(_norm(label))
        if code is None:
            unmatched.append(label)
            continue
        out[code] = d
    missing = sorted(set(PROV_NAMES) - set(out))
    if missing or unmatched:
        sys.exit(f"!! province join failed — no ECP row for {missing}, "
                 f"unmatched ECP labels {unmatched}")
    return out


def _norm(s):
    """Order-free, accent-free key for a province name.

    INE writes the same province three ways across its own products — "Illes Balears" in the
    px axis, "Balears, Illes" in the ECP series names, "Rioja (La)" against "Rioja, La" — so
    matching on the string fails and matching on a SORTED TOKEN SET does not. The article is
    kept as a token rather than dropped, because dropping it would collide Palmas (Las) with
    nothing but is the kind of shortcut that collides something eventually.
    """
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    toks = [x for x in re.split(r"[^a-z0-9]+", s) if x]
    return " ".join(sorted(toks))


# =======================================================================================
# UCIDE
# =======================================================================================

PROV_NAMES = {
    "01": "Araba/Álava", "02": "Albacete", "03": "Alicante/Alacant", "04": "Almería",
    "05": "Ávila", "06": "Badajoz", "07": "Illes Balears", "08": "Barcelona",
    "09": "Burgos", "10": "Cáceres", "11": "Cádiz", "12": "Castellón/Castelló",
    "13": "Ciudad Real", "14": "Córdoba", "15": "Coruña (A)", "16": "Cuenca",
    "17": "Girona", "18": "Granada", "19": "Guadalajara", "20": "Gipuzkoa",
    "21": "Huelva", "22": "Huesca", "23": "Jaén", "24": "León", "25": "Lleida",
    "26": "Rioja (La)", "27": "Lugo", "28": "Madrid", "29": "Málaga", "30": "Murcia",
    "31": "Navarra", "32": "Ourense", "33": "Asturias", "34": "Palencia",
    "35": "Palmas (Las)", "36": "Pontevedra", "37": "Salamanca",
    "38": "Santa Cruz de Tenerife", "39": "Cantabria", "40": "Segovia", "41": "Sevilla",
    "42": "Soria", "43": "Tarragona", "44": "Teruel", "45": "Toledo",
    "46": "Valencia/València", "47": "Valladolid", "48": "Bizkaia", "49": "Zamora",
    "50": "Zaragoza", "51": "Ceuta", "52": "Melilla",
}

# UCIDE's spellings, where they differ from INE's above.
UCIDE_ALIAS = {
    "álava": "01", "alava": "01", "alicante": "03", "alacant": "03",
    "baleares": "07", "islas baleares": "07", "castellón": "12", "castello": "12",
    "la coruña": "15", "coruña": "15", "a coruña": "15", "gerona": "17",
    "guipúzcoa": "20", "guipuzcoa": "20", "lérida": "25", "lerida": "25",
    "la rioja": "26", "orense": "32", "las palmas": "35",
    "tenerife": "38", "santa cruz de tenerife": "38", "valencia": "46",
    "vizcaya": "48", "bizkaia": "48", "asturias": "33", "cantabria": "39",
    "navarra": "31", "murcia": "30", "madrid": "28", "barcelona": "08",
}


def _ucide_spanish_muslims():
    """{province code: Spanish-CITIZEN Muslims}, from UCIDE's province table.

    The table is Autonomías | Provincias | Municipios | Extranjeros | Españoles | Totales,
    with the autonomous-community subtotal interleaved into the FIRST province row of each
    community — so a first row carries seven numbers and every other row four. Reading it as
    a token stream and keying on the province name is what makes that tractable; the check
    that it worked is that the 52 provincial `Totales` sum to UCIDE's own national figure,
    which is asserted by the caller.

    Returns {} if PyMuPDF is unavailable or the layout has moved — Spain then draws with
    CIS's other-religion cell whole, and countries.py's note says which happened.
    """
    path = os.path.join(RAW, "ucide.pdf")
    if not os.path.exists(path):
        return {}
    try:
        import fitz
    except ImportError:
        print("  !! PyMuPDF not installed — UCIDE split skipped")
        return {}
    doc = fitz.open(path)
    text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
    toks = [t.strip() for t in text.splitlines() if t.strip()]

    by_key = {_norm(v): k for k, v in PROV_NAMES.items()}
    by_key.update({_norm(k): v for k, v in UCIDE_ALIAS.items()})
    # UCIDE writes several provinces in their Castilian forms. _norm's sorted-token key
    # already reconciles "Coruña (A)" with "La Coruña" once the alias table goes through it,
    # which is the bug that cost a whole province the first time round: an alias table keyed
    # on raw lowercase silently stops matching the moment the normaliser changes.
    for code, name in PROV_NAMES.items():
        by_key.setdefault(_norm(re.sub(r"\s*\(.*\)", "", name).split("/")[0]), code)

    num = re.compile(r"^\d{1,3}(?:\.\d{3})*$|^\d+$")
    found, i = {}, 0
    while i < len(toks):
        code = by_key.get(_norm(toks[i]))
        if code is None or code in found:
            i += 1
            continue
        nums = []
        j = i + 1
        while j < len(toks) and len(nums) < 7 and num.match(toks[j]):
            nums.append(int(toks[j].replace(".", "")))
            j += 1
        # nums[0] is the municipality count. A community's first province then carries
        # (prov, ccaa) pairs for each of extranjeros / españoles / totales; every other
        # province carries the three bare numbers.
        if len(nums) >= 7:
            esp = nums[3]
        elif len(nums) >= 4:
            esp = nums[2]
        else:
            i += 1
            continue
        found[code] = esp
        i = j
    if len(found) < 52:
        print(f"  !! UCIDE: parsed {len(found)}/52 provinces — split skipped. Missing "
              f"{sorted(set(PROV_NAMES) - set(found))}")
        return {}
    # The report states its own national figure for hispanomusulmanes in the text
    # ("Total de hispanomusulmanes 1.085.593"), which is a check the table cannot fake:
    # a token-stream parser that grabbed a neighbouring column would miss it by millions.
    total = sum(found.values())
    stated = re.search(r"Total de hispanomusulmanes\s+([\d.]+)", text)
    if stated:
        want = int(stated.group(1).replace(".", ""))
        if abs(total - want) > 1:
            print(f"  !! UCIDE: 52 provinces sum to {total:,} but the report says "
                  f"{want:,} — split skipped")
            return {}
        print(f"  parse checks against the report's own total: {total:,}")
    return found


# =======================================================================================
# build
# =======================================================================================

def _foreign_half(pop, px):
    """[unit, node, count] for foreign nationals: INE x Pew x es_origin.py."""
    import es_origin

    pew = None
    with zipfile.ZipFile(os.path.join(RAW, "pew.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[(pew["Year"] == 2020) & (pew["Level"] == 1)].set_index("Country")
    fams = ["Christians", "Muslims", "Religiously_unaffiliated", "Buddhists", "Hindus",
            "Jews", "Other_religions"]

    nat = px[(px["Sexo"] == "Ambos sexos") & (px["Periodo"] == str(INE_NAT_YEAR))]
    nat = nat[~nat["Nacionalidad"].isin(INE_NAT_AGGREGATES)]
    nat = nat[nat["Provincias"] != "TOTAL ESPAÑA"].copy()
    nat["unit"] = nat["Provincias"].str.extract(r"^\s*(\d{2})")[0]
    nat = nat[nat["unit"].notna()]

    # spec §8.1: check the leaves partition the published total before using them.
    tot = px[(px["Sexo"] == "Ambos sexos") & (px["Periodo"] == str(INE_NAT_YEAR))
             & (px["Provincias"] == "TOTAL ESPAÑA")
             & (px["Nacionalidad"] == "TOTAL EXTRANJEROS")]["value"].iloc[0]
    leaf = nat["value"].sum()
    print(f"  INE {INE_NAT_YEAR}: {len(nat['Nacionalidad'].unique())} nationalities, "
          f"leaves {leaf:,.0f} vs published total {tot:,.0f} "
          f"({100 * (leaf - tot) / tot:+.3f}%)")
    if abs(leaf - tot) / tot > 0.001:
        sys.exit("!! INE nationality leaves do not partition the total")

    # Composition per nationality, once.
    comp, unmapped = {}, []
    for name in sorted(nat["Nacionalidad"].unique()):
        try:
            pn = es_origin.pew_name(name)
        except KeyError:
            unmapped.append(name)
            continue
        if pn is None:
            row = None                      # a residual or a microstate; see es_origin
        elif pn in pew.index:
            row = {f: float(pew.loc[pn, f]) for f in fams}
        else:
            unmapped.append(f"{name} (no Pew row for {pn})")
            continue
        comp[name] = es_origin.composition(name, row)
    if unmapped:
        sys.exit(f"!! {len(unmapped)} nationalities have no composition: {unmapped[:8]}")

    cur = pop          # already keyed by province code

    rows = []
    for code, g in nat.groupby("unit"):
        base = g["value"].sum()
        if base <= 0:
            continue
        scale = cur[code]["Extranjera"] / base
        acc = {}
        for name, v in zip(g["Nacionalidad"], g["value"]):
            if v <= 0:
                continue
            for node, share in comp[name].items():
                acc[node] = acc.get(node, 0.0) + v * scale * share
        for node, c in acc.items():
            rows.append((code, node, c))
    df = pd.DataFrame(rows, columns=["geo_id", "node", "count"])
    df["tier"] = "modelled"
    print(f"  foreign half: {df['count'].sum():,.0f} people, {df['node'].nunique()} nodes")
    return df


def build():
    import es2026

    print("CIS…")
    pool, recent = _pool_cis()
    share = _shares_by_province(pool, recent)
    nc = _nc_share(pool)
    print(f"  refusals: {100 * float((nc * 1).mean()):.2f}% mean over provinces, "
          f"{100 * float(pool['nc'].mul(pool['w']).sum() / pool['w'].sum()):.2f}% national")

    print("INE…")
    px = read_px(os.path.join(RAW, "ine_03005.px"))
    pop, period = _province_population()

    cur = _by_code(pop)
    esp_total = sum(v["Española"] for v in cur.values())
    ext_total = sum(v["Extranjera"] for v in cur.values())
    print(f"  {esp_total:,.0f} Spanish nationals + {ext_total:,.0f} foreign "
          f"= {esp_total + ext_total:,.0f}")

    print("UCIDE…")
    ucide = _ucide_spanish_muslims()
    if ucide:
        print(f"  parsed 52 provinces, {sum(ucide.values()):,} Spanish-citizen Muslims")

    # ---- Spanish-citizen half. Written in the house shape — one row per (unit, SOURCE
    # CATEGORY) with the category as CIS words — so tools/check_mapping.py can run on it and
    # taxonomy/es2026.py stays the only place a CIS answer becomes a node.
    rows, capped = [], []
    for code in sorted(PROV_NAMES):
        p = int(code)
        if p not in share.index:
            sys.exit(f"!! province {code} has no CIS respondents at all")
        n_esp = cur[code]["Española"] * (1.0 - float(nc.get(p, 0.0)))
        other = share.loc[p, 3] * n_esp
        muslim = min(float(ucide.get(code, 0)), other) if ucide else 0.0
        if ucide and ucide[code] > other:
            capped.append((code, ucide[code], other))
        for c in (1, 2, 4, 5, 6):
            rows.append((code, PROV_NAMES[code], es2026.CANONICAL[c],
                         share.loc[p, c] * n_esp, ""))
        # The UCIDE split of code 3, recorded as two rows so the operation is legible in the
        # file rather than only in this script: spec §3.1 permits an outside source to SPLIT
        # a category and never to add to it, and these two still sum to CIS's own cell.
        if muslim > 0:
            rows.append((code, PROV_NAMES[code],
                         "Creyente de otra religión — Muslim (UCIDE split)", muslim,
                         "UCIDE 2023 Spanish-citizen Muslims, capped at CIS's cell"))
        if other - muslim > 0:
            rows.append((code, PROV_NAMES[code], "Creyente de otra religión",
                         other - muslim, "residual after the UCIDE split"))
    esp = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count",
                                      "note"])
    esp["geo_level"] = "province"
    esp["basis"] = "self_id"
    esp["year"] = 2026
    esp["source_id"] = "cis_barometros_pooled"
    esp = esp[["geo_id", "geo_level", "geo_name", "source_category", "count", "basis",
               "year", "source_id", "note"]]
    print(f"  Spanish half: {esp['count'].sum():,.0f} people")
    if capped:
        print(f"  !! UCIDE exceeds CIS's other-religion cell in {len(capped)} provinces, "
              f"capped: " + ", ".join(f"{c}({u:,}>{o:,.0f})" for c, u, o in capped[:6]))

    ext = _foreign_half(cur, px)
    ext["geo_level"] = "province"
    ext["geo_name"] = ext["geo_id"].map(PROV_NAMES)
    ext["basis"] = "nationality_derived"
    ext["year"] = INE_NAT_YEAR
    ext["source_id"] = "ine_padron_x_pew2020"
    ext = ext[["geo_id", "geo_level", "geo_name", "node", "count", "tier", "basis", "year",
               "source_id"]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    esp.to_csv(OUT, index=False)
    ext.to_csv(OUT_FOREIGN, index=False)

    drawn = esp["count"].sum() + ext["count"].sum()
    counted = esp_total + ext_total
    print(f"\nwrote {OUT}  ({len(esp):,} rows, "
          f"{esp['source_category'].nunique()} source categories)")
    print(f"wrote {OUT_FOREIGN}  ({len(ext):,} rows, {ext['node'].nunique()} nodes) — "
          f"NODES, not categories, because a nationality resolves to a DISTRIBUTION over "
          f"religions rather than to one, which resolve() cannot express")
    print(f"drawn {drawn:,.0f} of {counted:,.0f} — {100 * drawn / counted:.2f}%. "
          f"The foreign half is drawn whole; the citizen half is "
          f"{100 * esp['count'].sum() / esp_total:.2f}% of citizens, the rest being CIS's "
          f"refusals, which spec §3.5 marks rather than fills.")

    both = pd.concat([
        esp.assign(node=esp["source_category"].map(es2026.resolve))[["node", "count"]],
        ext[["node", "count"]]], ignore_index=True)
    if both["node"].isna().any():
        sys.exit(f"!! unresolved source categories: "
                 f"{sorted(set(esp['source_category']) - set(es2026.MAP))}")
    top = both.groupby("node")["count"].sum().sort_values(ascending=False)
    print("\nnational totals:")
    for node, c in top.head(18).items():
        print(f"  {c:>12,.0f}  {100 * c / drawn:5.2f}%  {node}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
