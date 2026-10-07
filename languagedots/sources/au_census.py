"""Australia, Census 2021, language used at home -> data/normalized/au.csv (+ au_units.csv).

    python sources/au_census.py --fetch      # download, scrape QuickStats (resumable), normalise
    python sources/au_census.py              # normalise from data/raw/au only

THE QUESTION. "Does the person use a language other than English at home?" with tick boxes and a
write-in; one answer, the language used most often (LANP). It is a home-use question, not mother
tongue. Asked of everyone; 5.7% did not answer ("Not stated", not drawn). Place of usual
residence, overseas visitors excluded. Coded to the ASCL 2016, ~430 four-digit languages.

THREE TABLES OF THE ONE CENSUS, because no open table has both the full classification and
small areas (TableBuilder does, behind a registered login, not used):

  sa2      General Community Profile table G13, all 2,472 SA2s (~10,000 people each): English
           only, 35 named languages, and five remainders: "Other Chinese", "Other Indo-Aryan",
           "Other Southeast Asian Austronesian", "Australian Indigenous Languages" (all of them
           as one) and "Other". Read from religiondots' copy of the SA2 DataPack when present.
  state    Cultural diversity data summary 2021, Table 5: every four-digit language by state
           and territory (Other Territories only in the national total). Non-zero rows only.
  sa2_top  QuickStats, one page per SA2: "Language used at home, top responses (other than
           English)", the SA2's five largest languages at four-digit detail. This is what puts
           Warlpiri in the Tanami and Hazaraghi in Dandenong instead of spreading them across a
           state. Scraped once into quickstats_sa2.jsonl (resumable, ~2,470 pages).

countries/au.py combines them; this file only normalises and checks.

CHECKS (must pass):
  1. G13 per SA2: the leaves (English only, 35 languages, 5 remainders, not stated) sum to the
     SA2's total within ABS's perturbation (each cell is randomly adjusted): within 30 people or
     2% in every SA2 of 500+, and 0.1% nationally.
  2. The remainder buckets are what they claim: per state, G13's SA2 sum of each named
     language and each remainder equals Table 5's sum of that bucket's four-digit members,
     within 2.5% or 500 people for every cell over 5,000 (different tables, separately
     perturbed; the first bar, 1%, failed on named languages too, see the assertion).
  3. A second table per unit: every QuickStats top-five row that names a G13 language agrees
     with G13's cell for that SA2: within 10 people or 3% in 99% of rows, nothing over 15%.
  4. QuickStats covers every SA2 with people who use a language other than English, apart from
     the 18 no-boundary pseudo SA2s.

SOURCES. ABS, CC BY 4.0. GCP DataPack (2021_GCP_SA2_for_AUS_short-header.zip, release R2);
"Cultural diversity data summary.xlsx" (released 28 June 2022); "Language classification.xlsx"
(Census dictionary 2021, LANP); QuickStats pages abs.gov.au/census/find-census-data/quickstats/
2021/<SA2 code> (robots.txt allows them). Generic browser User-Agent.
"""
import argparse
import json
import os
import re
import socket
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutTimeout
from html import unescape
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "au"
OUT = HERE / "data" / "normalized" / "au.csv"
UNITS = HERE / "data" / "normalized" / "au_units.csv"
RD_ZIP = HERE.parent / "religiondots" / "data" / "raw" / "au" / "2021_GCP_SA2_for_AUS_short-header.zip"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
ABS = "https://www.abs.gov.au"
FILES = {
    "2021_GCP_SA2_for_AUS_short-header.zip":
        "/census/find-census-data/datapacks/download/2021_GCP_SA2_for_AUS_short-header.zip",
    "Cultural diversity data summary.xlsx":
        "/statistics/people/people-and-communities/cultural-diversity-census/2021/"
        "Cultural%20diversity%20data%20summary.xlsx",
    "Language classification.xlsx":
        "/census/guide-census-data/census-dictionary/2021/variables-topic/cultural-diversity/"
        "language-used-home-lanp/Language%20classification.xlsx",
}
QS = RAW / "quickstats_sa2.jsonl"
socket.setdefaulttimeout(60)
STATES = ["1", "2", "3", "4", "5", "6", "7", "8"]   # Table 5's columns, NSW..ACT

# G13 column stem -> the four-digit ASCL label it holds (checked against Table 5, check 2)
G13_NAMED = {
    "Afrikaans": "Afrikaans", "Arabic": "Arabic", "CL_Canton": "Cantonese",
    "CL_Mandarin": "Mandarin", "Croatian": "Croatian", "French": "French", "German": "German",
    "Greek": "Greek", "IAL_Bengali": "Bengali", "IAL_Guj": "Gujarati", "IAL_Hindi": "Hindi",
    "IAL_Nepali": "Nepali", "IAL_Punjabi": "Punjabi", "IAL_Sinhal": "Sinhalese",
    "IAL_Urdu": "Urdu", "Italian": "Italian", "Japan": "Japanese", "Khmer": "Khmer",
    "Korean": "Korean", "Macedon": "Macedonian", "Malayalam": "Malayalam",
    "Persian_ED": "Persian (excluding Dari)", "Polish": "Polish", "Portuguese": "Portuguese",
    "Russian": "Russian", "Samoan": "Samoan", "Serbian": "Serbian", "SAL_Filipin": "Filipino",
    "SAL_Indon": "Indonesian", "SAL_Tagalog": "Tagalog", "Spanish": "Spanish", "Tamil": "Tamil",
    "Thai": "Thai", "Turkish": "Turkish", "Vietnamese": "Vietnamese",
}
# G13 remainders: column stem -> (G13 label, ASCL narrow group prefix it is the rest of)
G13_REST = {
    "CL_Oth": ("Other Chinese languages", "71"),
    "IAL_Oth": ("Other Indo-Aryan languages", "52"),
    "SAL_Oth": ("Other Southeast Asian Austronesian languages", "65"),
    "AIndLng": ("Australian Indigenous languages", "8"),
    "Oth": ("Other languages", ""),       # everything else; see bucket_of()
}


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, b""
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"{url}: {err}")


def fetch_files():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        dest = RAW / name
        if name.endswith(".zip") and RD_ZIP.exists():
            print(f"  {name}: using religiondots' copy, {RD_ZIP}")
            continue
        if dest.exists():
            continue
        st, body = get(ABS + path)
        if st != 200:
            raise SystemExit(f"{name}: HTTP {st}")
        dest.write_bytes(body)
        print(f"  {name}: {len(body):,} bytes")


ROW = re.compile(r'<th scope="row" class="firstCol">(.*?)</th>\s*<td>([\d,]+)</td>', re.S)


def parse_quickstats(html):
    i = html.find("Language used at home, top responses (other than English)")
    if i < 0:
        return None
    j = html.find("</table>", i)
    rows = [(unescape(re.sub(r"<.*?>", "", a)).strip(), int(b.replace(",", "")))
            for a, b in ROW.findall(html[i:j])]
    return rows


def fetch_quickstats(codes, workers=4):
    done = set()
    if QS.exists():
        for line in QS.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["sa2"])
    todo = [c for c in codes if c not in done]
    print(f"  QuickStats: {len(done):,} on disk, {len(todo):,} to fetch")
    if not todo:
        return

    def one(code):
        st, body = get(f"{ABS}/census/find-census-data/quickstats/2021/{code}")
        time.sleep(0.3)
        rows = parse_quickstats(body.decode("utf-8", "replace")) if st == 200 else None
        return {"sa2": code, "status": st, "rows": rows}

    n = 0
    f = open(QS, "a", encoding="utf-8")
    ex = ThreadPoolExecutor(workers)
    futs = [ex.submit(one, c) for c in todo]
    try:
        # a first run hung for minutes on one stuck connection: give up on a stall instead,
        # and let a re-run pick up where this stopped
        for fu in as_completed(futs, timeout=180):
            rec = fu.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f"    {n:,}/{len(todo):,}", flush=True)
    except FutTimeout:
        f.close()
        print(f"  QuickStats stalled after {n:,} pages; run --fetch again to resume", flush=True)
        os._exit(2)
    f.close()
    ex.shutdown()


def ascl():
    """4-digit code -> label, from the census dictionary's classification workbook."""
    x = pd.read_excel(RAW / "Language classification.xlsx", header=None, dtype=str)
    out = {}
    for _, r in x.iterrows():
        v = [str(c).strip() for c in r if pd.notna(c) and str(c).strip()]
        if len(v) == 2 and re.fullmatch(r"\d{4}", v[0]):
            out[v[0]] = v[1]
    return out


def bucket_of(code, label):
    """Which G13 cell a four-digit language falls in."""
    if label == "English":
        return "English only"
    if label in G13_NAMED.values():
        return label
    for stem, (g13, pre) in G13_REST.items():
        if pre and code.startswith(pre):
            return g13
    return G13_REST["Oth"][0]


def read_g13():
    z = zipfile.ZipFile(RD_ZIP if RD_ZIP.exists() else RAW / "2021_GCP_SA2_for_AUS_short-header.zip")
    parts = []
    for n in sorted(z.namelist()):
        if re.search(r"2021Census_G13[A-E]_AUST_SA2\.csv$", n):
            parts.append(pd.read_csv(z.open(n), dtype={"SA2_CODE_2021": str}, na_values=[".."])
                         .set_index("SA2_CODE_2021"))
    g = pd.concat(parts, axis=1)
    cols = {"PSEO_Tot": "English only", "P_LUatH_NS_Tot": "Not stated"}
    cols.update({f"POL_{k}_Tot": v for k, v in G13_NAMED.items()})
    cols.update({f"POL_{k}_Tot": v[0] for k, v in G13_REST.items()})
    missing = [c for c in cols if c not in g.columns]
    if missing:
        raise SystemExit(f"G13 columns missing: {missing}")
    leaves = g[list(cols)].rename(columns=cols).fillna(0)
    # G13's "Other" INCLUDES the Australian Indigenous languages, which the table also shows as
    # a cell of their own (Yarrabah: Other 1,958, Indigenous 1,961, all other languages 1,966).
    # The metadata does not say so; check 1 and check 2 fail without this line and pass with it.
    # Perturbed separately, so the difference can dip below zero by a few people: clipped.
    oth = G13_REST["Oth"][0]
    leaves[oth] = (leaves[oth] - leaves[G13_REST["AIndLng"][0]]).clip(lower=0)
    checks = g[["P_Tot_Tot", "POL_Tot_Tot", "POL_CL_Tot_Tot", "POL_IAL_Tot_Tot",
                "POL_SAL_Tot_Tot"]].fillna(0)
    return leaves, checks


def read_table5(codes):
    x = pd.read_excel(RAW / "Cultural diversity data summary.xlsx", sheet_name="Table 5",
                      header=None)
    lab2code = {}
    for c, lab in codes.items():
        lab2code.setdefault(lab, c)
    rows = []
    for _, r in x.iterrows():
        v = [c for c in r.tolist() if pd.notna(c) and str(c).strip() != ""]
        if len(v) != 10 or not isinstance(v[0], str):
            continue
        lab = v[0].strip()
        if lab == "Total":
            continue
        code = lab2code.get(lab)
        if code is None and lab not in ("Not stated",):
            raise SystemExit(f"Table 5 label not in the classification: {lab!r}")
        rows.append([lab, code] + [int(n) for n in v[1:]])
    t = pd.DataFrame(rows, columns=["label", "code"] + STATES + ["AUS"])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.fetch:
        fetch_files()
    codes = ascl()
    g13, gchk = read_g13()
    pseudo = [c for c in g13.index if c[1:] in ("97979799", "99999499")]
    if a.fetch:
        fetch_quickstats([c for c in g13.index if c not in pseudo], a.workers)
    t5 = read_table5(codes)

    # ---- check 1: G13 leaves sum to each SA2's total ----
    s = g13.sum(axis=1)
    tot = gchk["P_Tot_Tot"]
    d = (s - tot).abs()
    bad = d[(tot >= 500) & (d > 30) & (d > 0.02 * tot)]
    print(f"check 1: G13 leaves vs SA2 total, worst {d.max():.0f} people; national "
          f"{s.sum():,.0f} vs {tot.sum():,.0f} ({s.sum() / tot.sum() - 1:+.5f})")
    assert bad.empty, f"G13 leaves off their totals: {bad.head()}"
    assert abs(s.sum() / tot.sum() - 1) < 0.001

    # ---- check 2: the buckets, per state ----
    t5 = t5[t5["label"] != "Not stated"].copy()
    t5["bucket"] = [bucket_of(c, lab) for c, lab in zip(t5["code"], t5["label"])]
    tb = t5.groupby("bucket")[STATES].sum()
    gb = g13.drop(columns="Not stated").groupby(g13.index.str[0]).sum().T
    worst, n, fails = 0.0, 0, []
    for b in tb.index:
        for st in STATES:
            want, have = tb.at[b, st], gb.at[b, st]
            if max(want, have) > 5000:
                n += 1
                r = abs(have / want - 1)
                worst = max(worst, r)
                print(f"    {b[:40]:40} state {st}: G13 {have:>10,.0f}  Table 5 {want:>10,.0f}  "
                      f"{have / want - 1:+.4f}")
                # first bar 1%: failed on 8 cells, plain named languages among them (Polish in
                # Queensland -2.4%, 137 people), so it is the two tables' perturbation, not the
                # buckets. NT "Other" is -8.5% (456 people): G13's Other minus its Indigenous
                # cell, both perturbed by up to ~90 in the big remote SA2s (read_g13)
                if r >= 0.025 and abs(have - want) >= 500:
                    fails.append(f"{b} in state {st}")
    print(f"check 2: {n} bucket x state cells over 5,000, worst {worst:.4f}")
    assert not fails, f"check 2: {fails}"

    # ---- QuickStats ----
    qs = []
    status = {}
    for line in QS.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        status[rec["sa2"]] = rec["status"]
        for lab, cnt in rec["rows"] or []:
            qs.append((rec["sa2"], lab, cnt))
    qs = pd.DataFrame(qs, columns=["sa2", "label", "count"]).drop_duplicates()
    extra = {"English only used at home", "Households where a non-English language is used"}
    lang = qs[~qs["label"].isin(extra)].copy()
    unknown = sorted(set(lang["label"]) - set(codes.values()))
    assert not unknown, f"QuickStats labels not in the classification: {unknown}"
    nonen = g13.drop(columns=["English only", "Not stated"]).sum(axis=1)
    missing = [c for c in g13.index if c not in pseudo and nonen[c] > 0 and c not in status]
    no_table = [c for c, st in status.items() if st == 200 and nonen.get(c, 0) > 0
                and not (lang["sa2"] == c).any()]
    print(f"check 4: QuickStats for {len(status):,} SA2s; {len(missing)} with non-English "
          f"speakers not fetched, {len(no_table)} fetched without a language table")
    assert not missing, missing[:10]

    # ---- check 3: QuickStats rows that G13 also names ----
    named = lang[lang["label"].isin(G13_NAMED.values())].copy()
    named["g13"] = [g13.at[c, lab] for c, lab in zip(named["sa2"], named["label"])]
    dd = (named["count"] - named["g13"]).abs()
    ok = (dd <= 10) | (dd <= 0.03 * named["g13"])
    rel = dd / named["g13"].clip(lower=1)
    print(f"check 3: {len(named):,} QuickStats rows name a G13 language; {ok.mean():.4f} within "
          f"10 people or 3%, worst {rel.max():.3f} ({dd.max():.0f} people)")
    assert ok.mean() >= 0.99 and (rel[named["g13"] >= 100] <= 0.15).all()

    # ---- write ----
    out = []
    long = g13.drop(columns="Not stated").stack().rename("count").reset_index()
    long.columns = ["geo_id", "source_category", "count"]
    out.append(long.assign(geo_level="sa2"))
    lt = lang.rename(columns={"sa2": "geo_id", "label": "source_category"})
    out.append(lt.assign(geo_level="sa2_top"))
    st = t5.melt(id_vars=["label", "code", "bucket"], value_vars=STATES, var_name="geo_id",
                 value_name="count").rename(columns={"label": "source_category"})
    out.append(st.assign(geo_level="state"))
    nat = t5[["label", "code", "bucket", "AUS"]].rename(columns={"label": "source_category",
                                                                  "AUS": "count"})
    out.append(nat.assign(geo_level="country", geo_id="AUS"))
    df = pd.concat(out, ignore_index=True)
    code_of = {v: k for k, v in codes.items()}
    df["code"] = df["code"].fillna(df["source_category"].map(code_of))
    df = df[df["count"] > 0]
    df[["geo_level", "geo_id", "source_category", "code", "bucket", "count"]].to_csv(
        OUT, index=False)
    units = pd.DataFrame({"geo_id": g13.index, "state": g13.index.str[0],
                          "total": gchk["P_Tot_Tot"].values,
                          "not_stated": g13["Not stated"].values,
                          "pseudo": g13.index.isin(pseudo)})
    units.to_csv(UNITS, index=False)
    print(f"wrote {OUT}: {len(df):,} rows; {UNITS}: {len(units):,} SA2s")
    print(f"  not stated {g13['Not stated'].sum():,.0f} of {tot.sum():,.0f} "
          f"({g13['Not stated'].sum() / tot.sum():.4f}); in the 18 pseudo SA2s "
          f"{tot[pseudo].sum():,.0f}")


if __name__ == "__main__":
    main()
