"""Japan — Japanese General Social Surveys, the published national religion table, one unit.

    python sources/jp.py --fetch     JGSS's XXRL page + the Kontur extract, ~16 MB
    python sources/jp.py             normalise

Writes:
    data/normalized/jp.csv           one row per JGSS religion code, geo_level `country`

Then `sources/jp_grid.py` (the prefecture placement layer) and `sources/jp_alloc.py` (these
national figures spread over the 47 prefectures). Until the second build on 2026-09-14 this
module also wrote a one-unit hex layer to the same path jp_grid.py now writes, so its
geometry() is gone: running both would put the national layer back over the prefecture one.

**THE BASELINE ANITA RULED FOR ON 2026-09-14** (`ask/answered/011`): national
self-identification first, JGSS's six regional blocks where the blocks differ, then anything
that allocates below them. This module is the first of the three. The other two are
`sources/jp_alloc.py`, after Anita's second ruling the same day (ask 014): JGSS publishes no
religion-by-block table, so the block level comes from JGSS-2015's one published block figure,
and the prefecture pattern from NHK's 1996 survey, which she accepted for allocating.

**THE SOURCE IS A WEB PAGE, NOT A DATASET.** `jgss.daishodai.ac.jp/surveys/table/XXRL.html`
prints the variable XXRL (信仰する宗教/家の宗教, "the religion you believe in / your family's
religion") for all eighteen waves 2000-2024N, unweighted counts, recoded in 2025 to a
four-digit list of 164 codes that names the Buddhist schools, Soka Gakkai, Rissho Kosei-kai,
Tenrikyo, Catholic and Protestant separately. The recode itself is distributed only through
JGSSDDS on application; the marginal table is public.

**THE QUESTION HAS THREE ANSWERS AND TWO OF THEM LEAD TO XXRL.** DORL asks "do you believe in a
religion?": yes / "I do not believe personally but my family has a religion" / no. Both of the
first two are then asked to name it. Code 8888 非該当 is everyone who answered no, so it is
`unaffiliated`, and a respondent who named a family religion is drawn under that religion.
Iwai (JGSS Research Center, Pew 2017) puts JGSS-2015 at 9.2% / 21.2% / 68.6%, and notes that the
first two together are close to ISM's "has personal religious faith" (28%), which is the basis
every Japanese attitude series reports.

**CODE 6000 IS A SUBTOTAL, NOT A CATEGORY.** Its label is 上記のうち、回答が5ケース未満の宗教カテゴ
リ, "of the above, categories with fewer than five cases". Summing it double counts. Dropped
here, and the drop is proven rather than assumed: every wave's rows without it reproduce the
page's own two 計 rows exactly.

**THE POOL IS THE FOUR MOST RECENT WAVES**, 2021H, 2022H, 2023D, 2024N: 10,612 respondents aged
20-89, stratified two-stage random samples over the same six blocks and four city-size strata,
all self-administered (留置). Not weighted, because the page publishes counts only. The
no-religion share drifts up across the series (64.7% pooled 2000-2012, 69.0% 2015-2018G, 71.7%
here), so pooling further back would buy sample and spend currency (§14.16).
"""

import csv
import gzip
import os
import re
import shutil
import ssl
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "jp")

XXRL_URL = "https://jgss.daishodai.ac.jp/surveys/table/XXRL.html"
XXRL = "jgss_XXRL.html"
WAVES = ["2000", "2001", "2002", "2003", "2005", "2006", "2008", "2010", "2012", "2015",
         "2017", "2017G", "2018", "2018G", "2021H", "2022H", "2023D", "2024N"]
POOL = ["2021H", "2022H", "2023D", "2024N"]
YEAR = 2024

KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
              "kontur_population_JP_20231101.gpkg.gz")
KONTUR_VINTAGE = "2023-11-01"
POP_2024 = 123_802_000       # 人口推計 2024-10-01, 第2表 総人口, for the Kontur sanity ratio only

SUBTOTAL = 6000
NONE_CODE = 8888

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Pinned from the page as read on 2026-09-14 (更新日 2025-05-25). A revision fails the build.
EXPECTED_POOL_N = 10_612
EXPECTED_POOL_NONE = 7_607
EXPECTED_CODES = 164

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have {os.path.basename(dest)}")
        return
    body = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                  timeout=300, context=_ctx()).read()
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(body)
    os.replace(tmp, dest)                                   # [[reference_wb_truncates]]
    print(f"  got {os.path.basename(dest)} ({len(body):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(XXRL_URL, os.path.join(RAW, XXRL))
    gz = os.path.join(RAW, "kontur_population_JP_20231101.gpkg.gz")
    _get(KONTUR_URL, gz)
    gpkg = gz[:-3]
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 0):
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)


def read_xxrl():
    """The page's one table -> ({code: (label, [18 counts])}, [the two 計 rows]).

    Every data row is `<code> | <label> | 18 counts`. The two 計 rows have an empty code cell;
    the larger is every respondent, the smaller everyone who was asked XXRL.
    """
    path = os.path.join(RAW, XXRL)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run with --fetch")
    html = open(path, encoding="utf-8").read()
    if "XXRL" not in html or "信仰する宗教" not in html:
        raise SystemExit("jp: the page is not the XXRL table any more")
    codes, totals = {}, []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        cells = [re.sub(r"<[^>]+>|&nbsp;", "", c).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]
        if len(cells) < 2 + len(WAVES):
            continue
        nums = cells[2:2 + len(WAVES)]
        if not all(re.fullmatch(r"\d+", n.replace(",", "")) for n in nums):
            continue
        vals = [int(n.replace(",", "")) for n in nums]
        if re.fullmatch(r"\d{4}", cells[0]):
            if cells[0] in codes:
                raise SystemExit(f"jp: code {cells[0]} appears twice")
            codes[int(cells[0])] = (cells[1], vals)
        elif cells[1] == "計":
            totals.append(vals)
    if len(totals) != 2:
        raise SystemExit(f"jp: expected two 計 rows, found {len(totals)}")
    return codes, totals


def normalise():
    codes, totals = read_xxrl()
    if len(codes) != EXPECTED_CODES:
        raise SystemExit(f"jp: {len(codes)} codes, expected {EXPECTED_CODES}")
    if SUBTOTAL not in codes or NONE_CODE not in codes:
        raise SystemExit("jp: code 6000 or 8888 missing")
    everyone = [max(a, b) for a, b in zip(*totals)]
    asked = [min(a, b) for a, b in zip(*totals)]

    # The proof that 6000 is a subtotal: without it, every wave closes on both 計 rows.
    for i, w in enumerate(WAVES):
        n = sum(v[i] for c, (_, v) in codes.items() if c != SUBTOTAL)
        a = sum(v[i] for c, (_, v) in codes.items() if c not in (SUBTOTAL, NONE_CODE))
        if n != everyone[i] or a != asked[i]:
            raise SystemExit(f"jp: wave {w} does not close without code 6000: rows {n} vs 計 "
                             f"{everyone[i]}, asked {a} vs 計 {asked[i]}")
    print(f"  {len(codes)} codes over {len(WAVES)} waves; every wave closes on its 計 rows "
          "with code 6000 left out")

    idx = [WAVES.index(w) for w in POOL]
    pooled = {c: (lab, sum(v[i] for i in idx)) for c, (lab, v) in codes.items() if c != SUBTOTAL}
    n = sum(k for _, k in pooled.values())
    none = pooled[NONE_CODE][1]
    if n != EXPECTED_POOL_N or none != EXPECTED_POOL_NONE:
        raise SystemExit(f"jp: pool {POOL} is {n:,} respondents / {none:,} none; this build "
                         f"expects {EXPECTED_POOL_N:,} / {EXPECTED_POOL_NONE:,}")
    print(f"  pool {'+'.join(POOL)}: {n:,} respondents, "
          + ", ".join(f"{w} {everyone[WAVES.index(w)]:,}" for w in POOL))

    rows = []
    for c, (lab, k) in sorted(pooled.items(), key=lambda kv: -kv[1][1]):
        if k == 0:
            continue                      # 55 codes nobody in the pool chose
        rows.append(dict(geo_id="JP", geo_level="country", geo_name="Japan",
                         source_category=f"{c} {lab}", count=k, basis="self_id", year=YEAR,
                         source_id="jgss_xxrl_2021h_2024n",
                         note="JGSS Research Center, XXRL 信仰する宗教/家の宗教（本人）, "
                              "unweighted counts pooled over JGSS-2021H, 2022H, 2023D, 2024N"))
    for r in rows[:14]:
        print(f"      {r['source_category'][:34]:<36}{r['count']:>6,}  {100 * r['count'] / n:5.2f}%")
    print(f"      ... {len(rows)} codes with anyone in them")

    out = os.path.join(ROOT, "data", "normalized", "jp.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)


def main():
    if "--fetch" in sys.argv:
        fetch()
    normalise()
    print("  next: python sources/jp_grid.py, then python sources/jp_alloc.py")


if __name__ == "__main__":
    main()
