"""Brazil — IBGE Censo Demografico, religion, at municipio.

Reads (or fetches) data/raw/br/ and writes data/normalized/br.csv.

Brazil is the cleanest example in the project of spec.md 3.4 -- structure from the
detailed source, totals from the recent one -- because IBGE published the two halves
twelve years apart and has said it may never publish them together:

  * Table 2094, Censo 2010.  65 religion categories, down to MUNICIPIO.  Named
    Pentecostal and mission-Protestant denominations, Umbanda split from Candomble,
    Judaism / Islam / Buddhism / Jehovah's Witnesses / LDS each its own row.
    Universe: the whole resident population, 190,755,799.

  * Table 9537, Censo 2022.  NINE categories, down to municipio.  "Evangelicas" is
    one lump of 47.4M and "Outras religiosidades" is one lump of 7.1M.
    Universe: persons aged 10 or over, 176,600,150.

Neither table alone answers R2.  2010 has the categories and is fifteen years stale;
2022 has the totals and no denominations.  Both are written to the normalized file,
kept apart by `year` and `source_id`, and NOT summed -- their universes differ (see
"Universe" below) as well as their vintages.

IBGE stated on release (6 June 2025) that the evangelical denominational breakdown for
2022 is withheld over data quality and that it is still evaluating whether it can be
published at all.  So the 2010 structure is not a stopgap pending a better table; it is
currently the only municipal-level denominational data Brazil has.

CATEGORY TREE.  Classification 133 is a nested list, not a partition, and the nesting is
carried in the LABEL rather than in any code -- "Evangelicas de origem pentecostal -
Igreja Assembleia de Deus" is a child of "Evangelicas de origem pentecostal", which is a
child of "Evangelicas".  CATEGORY_PARENT below states the tree explicitly; `level` and
`parent_category` go into the note column so allocate.py can use it (spec 3.10).
Summing every row of a municipio triple-counts.

Universe: 2010 counts everybody; 2022 counts only persons aged 10+.  IBGE did not ask
religion of under-10s in 2022, so the 2022 total is 176.6M against a population of
203.1M.  That is not a not-stated residual and must not be drawn as one.

BOTH YEARS ARE SAMPLE TABULATIONS AND DO NOT SUM ACROSS GEOGRAPHIC LEVELS.  Measured
2026-09-02: summing SIDRA's own municipal rows gives a national figure up to 34 people
away from SIDRA's own national row for the same table and category -- 2010 Catolica
123,280,184 against a published 123,280,172, 2022 Evangelicas 47,417,990 against
47,418,024.  IBGE expands the sample independently at each level, so the two are
different estimates of the same quantity rather than a sum and its parts.  Under 1 part
in 3,000,000, and the same species as Canada's base-5 rounding (spec 3.8), but a
reconciliation written to demand equality fails on every category.  The 2010 GRAND
TOTAL is a universe count rather than an expansion and is exact, so it is checked
exactly and is what would catch a genuinely missing municipio.

Usage:
    python sources/br.py --fetch    download raw JSON from the SIDRA API (54 requests)
    python sources/br.py            normalise from data/raw/br/
"""

import csv
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "br")
OUT = os.path.join(ROOT, "data", "normalized", "br.csv")

BASIS = "self_id"  # census self-declaration, both years

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

API = "https://apisidra.ibge.gov.br/values"

# The 27 federative units.  n6 (municipio) is requested one UF at a time: the whole
# country in one call is ~370,000 values and the API truncates rather than erroring.
UFS = ["11", "12", "13", "14", "15", "16", "17", "21", "22", "23", "24", "25", "26",
       "27", "28", "29", "31", "32", "33", "35", "41", "42", "43", "50", "51", "52", "53"]

# t=table, v=variable, and the fixed classification filters that reduce each table to
# "religion by place, everyone".  2094 is cross-tabulated by colour/race, so c86/0 takes
# the Total column; 9537 is cross-tabulated by sex and age group, so c2/6794 and
# c58/95253 take theirs.  Getting these wrong returns a plausible-looking file that is
# one demographic slice.
TABLES = {
    2010: dict(t="2094", v="93", p="2010", fixed="c86/0", cls="c133",
               source_id="br_censo_2010",
               note="IBGE SIDRA table 2094 (Censo 2010, amostra); universe = whole "
                    "resident population"),
    2022: dict(t="9537", v="140", p="2022", fixed="c2/6794/c58/95253", cls="c133",
               source_id="br_censo_2022",
               note="IBGE SIDRA table 9537 (Censo 2022, amostra); universe = persons "
                    "aged 10 or over"),
}

# Categories that are structurally absent at every level and every year we request --
# they belong to the 2000 census's version of classification 133, which shares the same
# classification id.  Kept here by name so that a future "..." in a category NOT on this
# list is treated as a finding rather than as noise.
KNOWN_ABSENT = {
    "100408", "100416", "100417", "100418", "100419", "95266",
    "100420", "100421", "100422", "95276",
}

# The explicit nesting of classification 133.  Key = category code, value = the code of
# its parent, or None for a top-level category.  IBGE encodes this only in the label
# text, so it is restated here rather than parsed out of a string.
CATEGORY_PARENT = {
    "95263": None,      # Catolica Apostolica Romana
    "100430": None,     # Catolica Apostolica Brasileira
    "2803": None,       # Catolica Ortodoxa
    "95277": None,      # Evangelicas
    "95264": "95277",   # Evangelicas de Missao
    "100403": "95264",  # ... Igreja Evangelica Luterana
    "100404": "95264",  # ... Igreja Evangelica Presbiteriana
    "100405": "95264",  # ... Igreja Evangelica Metodista
    "99741": "95264",   # ... Igreja Evangelica Batista
    "100406": "95264",  # ... Igreja Evangelica Congregacional
    "100407": "95264",  # ... Igreja Evangelica Adventista
    "99743": "95264",   # ... outras
    "95265": "95277",   # Evangelicas de origem pentecostal
    "100409": "95265",  # ... Igreja Assembleia de Deus
    "99746": "95265",   # ... Igreja Congregacao Crista do Brasil
    "100410": "95265",  # ... Igreja o Brasil para Cristo
    "100411": "95265",  # ... Igreja Evangelho Quadrangular
    "99745": "95265",   # ... Igreja Universal do Reino de Deus
    "100412": "95265",  # ... Igreja Casa da Bencao
    "100413": "95265",  # ... Igreja Deus e Amor
    "100414": "95265",  # ... Igreja Maranata
    "100415": "95265",  # ... Igreja Nova Vida
    "12881": "95265",   # ... Evangelica renovada nao determinada
    "12882": "95265",   # ... Comunidade Evangelica
    "99748": "95265",   # ... outras
    "121096": "95277",  # Evangelica nao determinada
    "12891": None,      # Outras religiosidades cristas
    "100423": None,     # Igreja de Jesus Cristo dos Santos dos Ultimos Dias
    "2824": None,       # Testemunhas de Jeova
    "95267": None,      # Espiritualista
    "2826": None,       # Espirita
    "2827": None,       # Umbanda e Candomble
    "2829": "2827",     # Umbanda
    "2828": "2827",     # Candomble
    "12883": "2827",    # Outras declaracoes de religiosidades afrobrasileira
    "100424": None,     # Judaismo
    "100425": None,     # Hinduismo
    "95269": None,      # Budismo
    "100427": None,     # Novas religioes orientais
    "100428": "100427",  # ... Igreja Messianica Mundial
    "100429": "100427",  # ... Outras novas religioes orientais
    "95270": None,      # Outras religioes orientais
    "100426": None,     # Islamismo
    "95273": None,      # Tradicoes esotericas
    "95274": None,      # Tradicoes indigenas
    "95275": None,      # Outras religiosidades
    "2836": None,       # Sem religiao
    "12884": "2836",    # ... Sem religiao
    "12885": "2836",    # ... Ateu
    "12886": "2836",    # ... Agnostico
    "12887": None,      # Nao determinada e multiplo pertencimento
    "12888": "12887",   # ... Religiosidade nao determinada ou mal definida
    "12889": "12887",   # ... Declaracao de multipla religiosidade
    "12890": None,      # Nao sabe
    "2837": None,       # Sem declaracao
}

TOTAL_CODES = {"0", "95278"}
TOTAL_NOTE = "universe total, not a religion category"


def raw_path(year, uf):
    return os.path.join(RAW, f"t{TABLES[year]['t']}_uf{uf}.json")


def read_url(url, timeout=300):
    """GET returning bytes, transparently gunzipping.

    servicodados (the metadata host) gzips unconditionally and ignores
    `Accept-Encoding: identity`, while apisidra does not compress at all.  urllib does
    not decompress either way, so the magic number is checked rather than the header --
    the symptom otherwise is a UnicodeDecodeError on byte 0x8b that reads like a
    charset problem and is not one.
    """
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip, identity"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        body = fh.read()
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return body


def get(url):
    """One SIDRA call, with retries, returning the parsed array of rows.

    sources.md 5a: a 200 is not a download.  A SIDRA error arrives as a 200 carrying a
    short JSON object rather than an array, so the shape is checked, not the status.
    """
    for attempt in range(4):
        try:
            body = read_url(url)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 400:               # too many values -- caller splits
                raise
            if attempt == 3:
                raise
            print(f"    retry after {exc}")
            time.sleep(5 * (attempt + 1))
        except Exception as exc:              # noqa: BLE001 - retry anything else
            if attempt == 3:
                raise
            print(f"    retry after {exc}")
            time.sleep(5 * (attempt + 1))
    data = json.loads(body)
    if not isinstance(data, list) or len(data) < 2:
        raise SystemExit(f"{url}\n  returned {body[:300]!r}")
    return data


def category_codes(table):
    """Classification 133's category codes, read from the table's own metadata.

    Read rather than hardcoded because the list differs between tables -- 2094 carries
    the full 2010 tree, 9537 only nine categories -- and because a category IBGE adds
    later should arrive on its own.
    """
    meta = json.loads(read_url(
        f"https://servicodados.ibge.gov.br/api/v3/agregados/{table}/metadados",
        timeout=120))
    cls = next(c for c in meta["classificacoes"] if c["id"] == 133)
    return [str(c["id"]) for c in cls["categorias"]]


def fetch_uf(spec, uf, codes):
    """All rows for one UF, splitting the category list when the API refuses.

    SIDRA caps a single response at ~50,000 values and answers 400 -- not a partial
    result -- when a request would exceed it.  Minas Gerais has 853 municipios, so
    853 x 66 categories = 56,298 trips the cap while every other state fits.  Rather
    than hardcode which states are too big, ask for everything and halve the category
    list on refusal.  Only the header of the first chunk is kept.
    """
    joined = ",".join(codes)
    url = (f"{API}/t/{spec['t']}/n6/in%20n3%20{uf}/v/{spec['v']}"
           f"/p/{spec['p']}/{spec['fixed']}/{spec['cls']}/{joined}")
    try:
        return get(url)
    except urllib.error.HTTPError as exc:
        if exc.code != 400 or len(codes) == 1:
            raise
        mid = len(codes) // 2
        print(f"    UF {uf}: 400 on {len(codes)} categories, splitting")
        left = fetch_uf(spec, uf, codes[:mid])
        right = fetch_uf(spec, uf, codes[mid:])
        return left + right[1:]


def fetch():
    """27 UFs x 2 tables, each UF written to its own raw file."""
    os.makedirs(RAW, exist_ok=True)
    for year, spec in TABLES.items():
        codes = category_codes(spec["t"])
        print(f"{year}: table {spec['t']}, {len(codes)} categories")
        for uf in UFS:
            dest = raw_path(year, uf)
            if os.path.exists(dest) and os.path.getsize(dest) > 2000:
                continue
            data = fetch_uf(spec, uf, codes)
            with open(dest, "w", encoding="utf-8") as out:
                json.dump(data, out, ensure_ascii=False)
            print(f"  {year} UF {uf}: {len(data) - 1:,} values")


def parse_value(raw):
    """IBGE sentinels.  Only a real number and '-' (a true zero) become a row.

    '-'   zero
    '..'  not applicable
    '...' not available at this level -- see KNOWN_ABSENT
    'X'   suppressed to avoid identifying an informant
    """
    raw = raw.strip()
    if raw == "-":
        return 0
    if raw in ("..", "...", "X"):
        return None
    return int(raw)


def read_year(year):
    spec = TABLES[year]
    rows, absent, suppressed = [], set(), 0
    for uf in UFS:
        path = raw_path(year, uf)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run with --fetch first")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        # The header row names the dimensions; find which D-columns hold municipio and
        # religion rather than assuming positions, since the two tables differ.
        head = data[0]
        d_mun = next(k for k, v in head.items()
                     if k.endswith("C") and v.startswith("Munic"))[:-1]
        d_rel = next(k for k, v in head.items()
                     if k.endswith("C") and v.startswith("Religi"))[:-1]
        for r in data[1:]:
            code = r[d_rel + "C"]
            value = parse_value(r["V"])
            if value is None:
                if r["V"].strip() == "X":
                    suppressed += 1
                else:
                    absent.add(code)
                continue
            is_total = code in TOTAL_CODES
            note = spec["note"]
            if is_total:
                note += "; " + TOTAL_NOTE
            else:
                parent = CATEGORY_PARENT.get(code)
                depth = 0
                walk = code
                while CATEGORY_PARENT.get(walk):
                    walk = CATEGORY_PARENT[walk]
                    depth += 1
                note += f"; code={code}; level={depth}; parent={parent or ''}"
            rows.append({
                "geo_id": r[d_mun + "C"],
                "geo_level": "municipio",
                "geo_name": r[d_mun + "N"],
                "source_category": r[d_rel + "N"],
                "count": value,
                "basis": BASIS,
                "year": year,
                "source_id": spec["source_id"],
                "note": note,
            })
    return rows, absent, suppressed


# Both censuses are SAMPLE tabulations, and IBGE expands the sample independently at
# each geographic level.  So SIDRA's own national row does not equal the sum of SIDRA's
# own municipal rows -- measured 2026-09-02 at up to 34 people on figures of 10^8, under
# 1 part in 3,000,000.  This is the same species as Canada's base-5 rounding (spec 3.8):
# parent and child disagree by construction, and a reconciliation that demands equality
# is testing the wrong thing.  Equality IS required of the 2010 grand total, which is a
# universe count rather than an expansion, and it holds exactly.
TOLERANCE_ABS = 100
TOLERANCE_REL = 1e-6


def close(got, want):
    return abs(got - want) <= max(TOLERANCE_ABS, want * TOLERANCE_REL)


def check(rows):
    """Reconcile each year against IBGE's own published national figures."""
    published = {
        2010: {
            "Total": 190755799,
            "Católica Apostólica Romana": 123280172,
            "Evangélicas": 42275440,
            "Espírita": 3848876,
            "Umbanda e Candomblé": 588797,
            "Sem religião": 15335510,
        },
        2022: {
            "Total": 176600150,
            "Católica Apostólica Romana": 100216153,
            "Evangélicas": 47418024,
            "Espírita": 3257455,
            "Umbanda e Candomblé": 1849824,
            "Sem religião": 16385342,
        },
    }
    ok = True
    for year, want in published.items():
        print(f"\nnational reconciliation, {year}  "
              f"(sum of municipios vs IBGE's published national figure)")
        for cat, target in want.items():
            got = sum(r["count"] for r in rows
                      if r["year"] == year and r["source_category"] == cat)
            good = close(got, target)
            ok &= good
            drift = got - target
            print(f"  {'OK ' if good else 'BAD'} {cat:<28} {got:>12,}  "
                  f"(published {target:,}, drift {drift:+,})")

    # The 2010 grand total is a universe count, not a sample expansion, so unlike every
    # category above it must be exact.  If this ever drifts, a municipio is missing.
    got = sum(r["count"] for r in rows
              if r["year"] == 2010 and r["source_category"] == "Total")
    good = got == 190755799
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} 2010 grand total is exact: {got:,}")

    # The tree must nest: every parent >= the sum of its children, and where the
    # children are exhaustive it is equal.  Checked on the one branch IBGE documents
    # as exhaustive, Umbanda e Candomble = Umbanda + Candomble + outras afrobrasileira.
    def nat(year, cat):
        return sum(r["count"] for r in rows
                   if r["year"] == year and r["source_category"] == cat)

    parent = nat(2010, "Umbanda e Candomblé")
    kids = (nat(2010, "Umbanda") + nat(2010, "Candomblé")
            + nat(2010, "Outras declarações de religiosidades afrobrasileira"))
    good = close(kids, parent)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} 2010 Umbanda e Candomble {parent:,} "
          f"== children {kids:,} (drift {kids - parent:+,})")

    for year in (2010, 2022):
        sub = [r for r in rows if r["year"] == year]
        units = {r["geo_id"] for r in sub}
        cats = {r["source_category"] for r in sub}
        print(f"\n  {year}: {len(sub):,} rows, {len(units):,} municipios, "
              f"{len(cats)} categories")
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows = []
    for year in TABLES:
        got, absent, suppressed = read_year(year)
        rows += got
        unexpected = absent - KNOWN_ABSENT
        print(f"{year}: {len(got):,} rows; {len(absent)} categories absent "
              f"({len(unexpected)} unexpected); {suppressed} suppressed cells")
        if unexpected:
            print("  UNEXPECTED absent categories:", sorted(unexpected))
    check(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
