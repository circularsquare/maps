"""Malaysia, the rework of 2026-10-19 (session eca0185b-my): three tables that split what the
2020 ethnic-group build (sources/my_census.py) left lumped. Writes, under data/normalized/:

    my_chinese_shares.csv   state, dialect node, share of the state's Chinese (2000 census, DOSM
                            tabulation by dialect group, as printed in Voon 2024)
    my_indian_mix.csv       node, share of Indians (sub-group table, see INDIANS)
    my_noncitizen_mix.csv   group, nationality, nat_share, node, share: what non-citizens speak

    python sources/my_extra.py

countries/my.py reads these and does the placement. Every number's source is in sources/my.md.
Needs no network: Voon's Table 2 is typed in below from the paper (see the check that follows it).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "taxonomy"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

NORM = ROOT / "data" / "normalized"

# ---------------------------------------------------------------------------------------------
# 1. Chinese dialect groups by state, 2000
# Voon Phin-Keong, "The Hakkas of Malaysia to 1970: Population, Livelihood, and Culture",
# Malaysian Journal of Chinese Studies 13(1), 2024, 37-59 (CC BY 4.0), Table 2, "Population and
# Percentage Share of Seven Major Dialect Groups by State, 2000": unpublished tabulation by the
# Department of Statistics, the 2000 census, dialect identity self-reported by households. The
# paper's note 2: the tabulation totals 330,000 (5.7%) fewer Chinese than the published reports.
# Columns: Hakka, Hokkien, Cantonese, Teochew, Hainan, Foochow, Kwongsai; (population in 10,000,
# share of the state's Chinese in %). The shares are used; the counts are the check.
# ---------------------------------------------------------------------------------------------
GROUPS = ["hakka", "hokkien", "cantonese", "teochew", "hainanese", "foochow", "kwongsai"]
NODE = {
    "hakka": "sinotibetan.sinitic.hakka",
    "hokkien": "sinotibetan.sinitic.min_nan",
    "cantonese": "sinotibetan.sinitic.cantonese",
    "teochew": "sinotibetan.sinitic.teochew",
    "hainanese": "sinotibetan.sinitic.hainanese",
    "foochow": "sinotibetan.sinitic.min_dong",
    "kwongsai": "sinotibetan.sinitic.kwongsai",
    "other": "sinotibetan.sinitic",      # Henghua, Hockchia, other Chinese: the group node, unnamed
}
# state code (DOSM: 01 Johor ... 16 Putrajaya) -> name as Voon prints it, counts, shares
VOON = {
    "10": ("Selangor", [20.4, 50.6, 28.9, 5.8, 2.7, 0.9, 0.4], [17.6, 43.5, 24.9, 5.0, 2.3, 0.8, 0.4]),
    "01": ("Johor", [14.0, 41.5, 9.7, 9.1, 2.8, 1.1, 0.8], [16.9, 50.3, 11.8, 11.0, 3.4, 1.4, 1.0]),
    "07": ("Pulau Pinang", [4.0, 29.8, 6.5, 12.3, 1.1, 0.3, 0.1], [7.2, 54.2, 11.7, 22.3, 2.0, 0.6, 0.1]),
    "14": ("Kuala Lumpur", [10.9, 19.0, 18.4, 2.3, 1.5, 0.5, 0.2], [18.5, 34.7, 33.7, 4.1, 2.8, 0.9, 0.4]),
    "08": ("Perak", [13.4, 15.4, 20.2, 6.0, 1.0, 2.8, 1.2], [21.6, 24.9, 32.6, 9.7, 1.6, 4.6, 1.9]),
    "02": ("Kedah", [2.4, 10.1, 2.5, 6.2, 0.4, 0.5, 0.1], [10.6, 45.1, 11.0, 27.4, 1.6, 0.6, 0.3]),
    "04": ("Melaka", [3.2, 8.1, 1.7, 1.1, 1.1, 0.1, 0.1], [19.0, 47.7, 10.0, 6.5, 6.5, 0.5, 0.3]),
    "13": ("Sarawak", [16.2, 6.9, 2.9, 3.8, 0.8, 17.8, 0.04], [31.5, 13.5, 5.7, 7.4, 1.5, 34.8, 0.1]),
    "05": ("Negeri Sembilan", [6.2, 6.4, 5.5, 0.4, 0.8, 0.5, 0.2], [30.1, 30.8, 26.7, 2.1, 3.8, 2.6, 1.1]),
    "06": ("Pahang", [3.8, 5.9, 6.2, 1.1, 0.8, 0.3, 2.0], [18.1, 28.5, 29.9, 5.2, 3.6, 1.5, 9.5]),
    "12": ("Sabah", [14.8, 3.4, 3.1, 1.2, 0.7, 0.5, 0.05], [58.0, 13.3, 12.3, 4.5, 2.6, 2.0, 0.2]),
    "03": ("Kelantan", [0.4, 3.0, 0.6, 0.1, 0.2, 0.01, 0.02], [8.0, 67.0, 12.5, 2.2, 3.5, 0.3, 0.5]),
    "11": ("Terengganu", [0.2, 1.1, 0.4, 0.1, 0.3, 0.01, 0.01], [9.1, 47.6, 17.0, 4.2, 13.0, 0.6, 0.6]),
    "09": ("Perlis", [0.4, 0.9, 0.2, 0.3, 0.04, 0.01, 0.03], [21.0, 48.0, 9.5, 14.7, 2.3, 0.3, 1.7]),
}
VOON_TOTAL = [109.3, 202.1, 106.8, 49.7, 14.1, 25.2, 5.2]     # printed totals, 10,000s
# Federal Territories the table has no row for take their neighbour's shares: Labuan (was Sabah's)
# takes Sabah's, Putrajaya (carved from Selangor in 2001) Selangor's.
SHARE_FROM = {"15": "12", "16": "10"}


def chinese():
    rows = []
    for st, (name, cnt, pct) in VOON.items():
        s = sum(pct)
        if not 88 < s < 100.01:
            raise SystemExit(f"my_extra: {name} shares add to {s:.1f}%")
        # the counts check the shares: each group's count over its share is the state's Chinese,
        # and the seven must agree. Counts have two figures (0.4 = 4,000 people), so groups under
        # 1% are too coarse to test. Voon's KL Hakka count (10.9) is the one that disagrees.
        impl = [c / (p / 100) for c, p in zip(cnt, pct) if c >= 1.5]
        lo, hi = (min(impl), max(impl)) if impl else (1, 1)
        if hi / lo > 1.12:
            print(f"   note: {name}: counts over shares range {lo:.1f} to {hi:.1f} (x10,000); "
                  "shares used")
        for g, p in zip(GROUPS, pct):
            rows.append(dict(state=st, dialect=g, node=NODE[g], share=p / 100,
                             voon_count_10k=cnt[GROUPS.index(g)]))
        rows.append(dict(state=st, dialect="other", node=NODE["other"], share=(100 - s) / 100,
                         voon_count_10k=None))
    # national check: counts add to the printed totals within one unit of rounding per state
    for j, g in enumerate(GROUPS):
        tot = sum(v[1][j] for v in VOON.values())
        print(f"   {g:<10} Table 2 rows add to {tot:6.1f}, printed total {VOON_TOTAL[j]:6.1f}")
    out = pd.DataFrame(rows)
    out.to_csv(NORM / "my_chinese_shares.csv", index=False)
    print(f"1. my_chinese_shares.csv: {len(VOON)} states x 8 groups")
    return out


# ---------------------------------------------------------------------------------------------
# 2. Indians: sub-group table
# P. Bala, "Malaysian Indians", Worldmark Encyclopedia of Cultures and Daily Life (encyclopedia.com),
# the table of Indian sub-groups printed under its population line "(2004)": population, share.
# It is a tabulation of the census's Indian sub-groups (the same 1970-2000 coding as the Chinese
# dialect groups: its total, 1,593,777, is 4% under the 2000 census's 1.66M Indians, like Voon's
# 5.7% gap); its own source line is not printed, so it is used as a published share, not as a count.
# ---------------------------------------------------------------------------------------------
INDIANS = [
    # (label, people, node)
    ("Indian Tamil", 1_396_480, "dravidian.southern.tamil"),
    ("Sri Lankan Tamil", 8_735, "dravidian.southern.tamil"),
    ("Malayali", 35_809, "dravidian.southern.malayalam"),
    ("Telugu", 38_993, "dravidian.southcentral.telugu"),
    ("Sikh", 33_231, "indoeuropean.indoaryan.northwestern.punjabi"),
    ("Punjabi", 23_147, "indoeuropean.indoaryan.northwestern.punjabi"),
    ("Other Indian", 41_477, "other"),
    ("Bangladeshi", 2_951, "indoeuropean.indoaryan.eastern.bengali"),
    ("Singhalese", 1_641, "indoeuropean.indoaryan.sinhala"),
    ("Pakistani", 11_313, "PK"),     # filled from Pakistan's mix below
]
INDIANS_TOTAL = 1_593_777


def indians(mix, back):
    tot = sum(p for _, p, _ in INDIANS)
    if tot != INDIANS_TOTAL:
        raise SystemExit(f"my_extra: Indian sub-groups add to {tot:,}, expected {INDIANS_TOTAL:,}")
    acc = {}
    for lab, p, node in INDIANS:
        m = {back(n): s for n, s in mix("PK", "my").items()} if node == "PK" else {node: 1.0}
        for n, s in m.items():
            acc[n] = acc.get(n, 0.0) + p * s / tot
    out = pd.DataFrame(sorted(acc.items(), key=lambda kv: -kv[1]), columns=["node", "share"])
    out.to_csv(NORM / "my_indian_mix.csv", index=False)
    print(f"2. my_indian_mix.csv: {len(out)} nodes; Tamil {acc['dravidian.southern.tamil']:.1%}")
    return out


# ---------------------------------------------------------------------------------------------
# 3. Non-citizens: nationality, then language
# DOSM's 2020 district counts are the census's; no census table gives their nationality (searched,
# sources/my.md §2). Nationality shares: temporary-pass (PLKS) foreign workers through June 2018,
# Malaysia's Ministry of Human Resources report (mohr.gov.my, BPA Report): Indonesia 39%, Nepal
# 22%, Bangladesh 17%, India 6%, Myanmar 6% (the other 10% are many small countries, left out and
# the five scaled up). Sabah takes only Indonesian and Filipino workers by rule, and its census
# non-citizens are mostly both: split 50/45 with 5% the other nationalities (my.md §6).
# ---------------------------------------------------------------------------------------------
DEFAULT_NAT = {"ID": 39, "NP": 22, "BD": 17, "IN": 6, "MM": 6}
SABAH_NAT = {"ID": 50, "PH": 45, "REST": 5}


def noncitizens(mix, back):
    import sa_census as sa
    d, _ = sa.india_mix()
    tn = sa.mix_from(d[d["geo_name"] == "TAMIL NADU"])       # written ids (in2011)

    def drawn(iso):
        return {back(n): s for n, s in mix(iso, "my").items()}

    langs = {iso: drawn(iso) for iso in ("ID", "NP", "BD", "MM", "PH")}
    langs["IN"] = tn         # uncited call: Indian workers in Malaysia taken as Tamil Nadu's mix
    # Sabah's Indonesians: Bugis, Toraja, Flores (Lamaholot, Manggarai) named by Wan Shawaluddin
    # et al. 2010; the shares are a judgement, the rest at Indonesia's own mix
    sab_id = {"austronesian.buginese": 0.50, "austronesian.toraja": 0.10,
              "austronesian.lamaholot": 0.05, "austronesian.manggarai": 0.05}
    for n, s in langs["ID"].items():
        sab_id[n] = sab_id.get(n, 0.0) + 0.30 * s
    # Sabah's Filipinos: Bajau (Sama) bigger than Suluk (Tausug), the same paper; 10% at the
    # Philippines' own mix (Christian Filipinos, many from the north)
    sab_ph = {"austronesian.sama_bajaw.sama": 0.50, "austronesian.philippine.tausug": 0.40}
    for n, s in langs["PH"].items():
        sab_ph[n] = sab_ph.get(n, 0.0) + 0.10 * s
    rows = []

    def put(group, nat, share, m):
        t = sum(m.values())
        for n, s in m.items():
            rows.append(dict(group=group, nationality=nat, nat_share=share, node=n, share=s / t))

    tot = sum(DEFAULT_NAT.values())
    for nat, v in DEFAULT_NAT.items():
        put("default", nat, v / tot, langs[nat])
    put("sabah", "ID", SABAH_NAT["ID"] / 100, sab_id)
    put("sabah", "PH", SABAH_NAT["PH"] / 100, sab_ph)
    # the other 5%: the default nationalities without Indonesia
    rest = {k: v for k, v in DEFAULT_NAT.items() if k != "ID"}
    rt = sum(rest.values())
    for nat, v in rest.items():
        put("sabah", nat, SABAH_NAT["REST"] / 100 * v / rt, langs[nat])
    out = pd.DataFrame(rows)
    for g, df in out.groupby("group"):
        s = df.drop_duplicates("nationality")["nat_share"].sum()
        if abs(s - 1) > 1e-9:
            raise SystemExit(f"my_extra: {g} nationality shares add to {s}")
    out.to_csv(NORM / "my_noncitizen_mix.csv", index=False)
    print(f"3. my_noncitizen_mix.csv: {len(out)} rows, {out['node'].nunique()} nodes")
    return out


# the written id of a drawn one (taxonomy/regroup.py moves written ids; origin_mix returns drawn)
def written_ids():
    import re
    from regroup import move
    tx = ROOT / "taxonomy"
    ids = set()
    for f in [tx / "tree.txt", *sorted((tx / "tree.d").glob("*.txt"))]:
        text = re.sub(r"# --- origin_mix borrowed nodes.*?# --- end origin_mix borrowed nodes ---",
                      "", f.read_text(encoding="utf-8"), flags=re.S)
        for ln in text.splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "|" in ln:
                ids.add(ln.split("|")[0].strip())
    inv = {}
    for i in ids:
        inv.setdefault(move(i), set()).add(i)

    def back(node):
        if node in ids:
            return node
        w = inv.get(node, set())
        if len(w) != 1:
            raise SystemExit(f"no single written id for drawn {node!r}: {sorted(w)}")
        return next(iter(w))
    return back


def main():
    from origin_mix import mix
    back = written_ids()
    chinese()
    indians(mix, back)
    noncitizens(mix, back)


if __name__ == "__main__":
    main()
