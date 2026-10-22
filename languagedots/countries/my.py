# Malaysia. Census 2020 ethnic group by district (OpenDOSM) and sub-ethnic group by state (state
# volumes' Table 5), read as language under AGENT_BRIEF §2's ethnicity rule (taxonomy/my2020.py).
# Each state's sub-ethnic counts are spread over its districts inside the district's Bumiputera
# count, following the district's 2020 Malay / other-Bumiputera split and, in Sabah and Sarawak,
# the 2010 census's district split of the named groups (sources/my_census.py). On religiondots'
# Kontur hexes for the same 160 districts. The record is sources/my.md.
from _shared import *  # noqa: F401,F403

import numpy as np


def _ipf(seed, rows, cols, n=200):
    """Scale `seed` (districts x groups) to the district totals `rows` and group totals `cols`."""
    m = seed.astype(float).copy()
    for _ in range(n):
        r = m.sum(axis=1)
        m *= np.divide(rows, r, out=np.zeros_like(rows), where=r > 0)[:, None]
        c = m.sum(axis=0)
        m *= np.divide(cols, c, out=np.zeros_like(cols), where=c > 0)[None, :]
    return m


def _weights(units, names, home, out_w):
    """Soft seed weights for `units` (geo ids): 1 in the `home` district names, `out_w` elsewhere."""
    missing = set(home) - set(names.values())
    if missing:
        raise SystemExit(f"my: placement names unknown districts {missing}")
    return np.array([1.0 if names[u] in home else out_w for u in units])


def _chinese(wide, names):
    """Each state's Chinese, split by the 2000 census's dialect shares (my_chinese_shares.csv),
    placed on the state's districts by iterative proportional fitting (district Chinese counts
    exactly, the state's group totals exactly), RETAIN of each group on its dialect and the
    rest on Mandarin. The group the census columns do not name stays on `sinitic`."""
    import my2020
    sh = pd.read_csv(NORM / "my_chinese_shares.csv", dtype={"state": str})
    groups = ["hokkien", "hakka", "cantonese", "teochew", "foochow", "hainanese", "kwongsai",
              "other"]
    parts = []
    for st2 in sorted({u[4:6] for u in wide.index}):
        units = [u for u in wide.index if u[4:6] == st2]
        src = {"15": "12", "16": "10"}.get(st2, st2)       # Labuan, Putrajaya: no row in the table
        s = sh[sh["state"] == src].set_index("dialect")
        rows = wide.loc[units, "Chinese"].to_numpy(float)
        cols = np.array([s.loc[g, "share"] for g in groups]) * rows.sum()
        seed = np.zeros((len(units), len(groups)))
        for j, g in enumerate(groups):
            home, out_w = my2020.DIALECT_PLACE.get((src, g), ([], 1.0))
            seed[:, j] = rows * (_weights(units, names, home, out_w) if home else 1.0)
        m = _ipf(seed, rows, cols) if rows.sum() > 0 else seed
        if rows.sum() > 0 and np.abs(m.sum(axis=1) - rows).max() > 1:
            raise SystemExit(f"my: Chinese in state {st2} did not converge")
        for j, g in enumerate(groups):
            node = s.loc[g, "node"]
            parts.append(pd.DataFrame({"unit": units, "node": node,
                                       "count": m[:, j] * my2020.RETAIN, "tier": "modelled"}))
        parts.append(pd.DataFrame({"unit": units, "node": "sinotibetan.sinitic.mandarin",
                                   "count": rows * (1 - my2020.RETAIN), "tier": "modelled"}))
    return pd.concat(parts, ignore_index=True)


def _noncitizens(wide, names):
    """Non-citizens: the district's census count, on nationalities' languages. Outside Sabah the
    default national mix; in Sabah the Indonesian/Filipino split fitted to districts
    (my2020.SABAH_HOME). Every row `derived`."""
    import my2020
    mx = pd.read_csv(NORM / "my_noncitizen_mix.csv")
    parts = []
    for st2 in sorted({u[4:6] for u in wide.index}):
        units = [u for u in wide.index if u[4:6] == st2]
        n = wide.loc[units, "Non-citizens"].to_numpy(float)
        grp = mx[mx["group"] == ("sabah" if st2 == "12" else "default")]
        nats = list(grp.drop_duplicates("nationality")["nationality"])
        nat_share = grp.drop_duplicates("nationality").set_index("nationality")["nat_share"]
        if st2 == "12":
            seed = np.zeros((len(units), len(nats)))
            for j, nat in enumerate(nats):
                home = my2020.SABAH_HOME.get(nat)
                seed[:, j] = n * (_weights(units, names, home, my2020.SABAH_OUT) if home else 1.0)
            cols = np.array([nat_share[k] for k in nats]) * n.sum()
            m = _ipf(seed, n, cols)
            if np.abs(m.sum(axis=1) - n).max() > 1:
                raise SystemExit("my: Sabah non-citizens did not converge")
        else:
            m = np.outer(n, [nat_share[k] for k in nats])
        for j, nat in enumerate(nats):
            g = grp[grp["nationality"] == nat]
            for node, share in zip(g["node"], g["share"]):
                parts.append(pd.DataFrame({"unit": units, "node": node,
                                           "count": m[:, j] * share, "tier": "derived"}))
    return pd.concat(parts, ignore_index=True)


def _counts():
    import my2020
    df = pd.read_csv(NORM / "my.csv")
    dist = df[(df["level"] == "district") & (df["year"] == 2020)]
    wide = dist.pivot(index="geo_id", columns="source_category", values="count")
    names = dict(zip(dist["geo_id"], dist["geo_name"]))
    state = df[df["level"] == "state"]
    seed10 = df[df["level"] == "seed2010"].pivot(index="geo_id", columns="source_category",
                                                 values="count")
    unknown = sorted(set(state["source_category"]) - set(my2020.NAMES))
    if unknown:
        raise SystemExit(f"my.csv groups with no mapping: {unknown}")
    out = []
    # Others: the district's own count
    out.append(pd.DataFrame({"unit": wide.index, "node": my2020.NAMES["Others"],
                             "count": wide["Others"].to_numpy(float), "tier": "derived"}))
    # Chinese: dialect groups of the 2000 census by state, placed on districts, 53% keeping the
    # dialect and the rest on Mandarin (sources/my.md §3)
    out.append(_chinese(wide, names))
    # Indians: the sub-group shares, flat over districts (sources/my.md §4)
    im = pd.read_csv(NORM / "my_indian_mix.csv")
    for node, share in zip(im["node"], im["share"]):
        out.append(pd.DataFrame({"unit": wide.index, "node": node,
                                 "count": wide["Indians"].to_numpy(float) * share,
                                 "tier": "modelled"}))
    # Non-citizens: the census's district counts on their nationalities' languages (§6)
    out.append(_noncitizens(wide, names))
    # Bumiputera: each state's Table 5 groups spread over its districts' Bumiputera
    for st, g in state.groupby("geo_id"):
        st2 = st[4:6]
        g = g[~g["source_category"].isin(["Cina Chinese", "India Indians", "Lain-lain Others"])]
        labels = g["source_category"].tolist()
        cols = g["count"].to_numpy(float)
        units = [u for u in wide.index if u[4:6] == st2]
        malay = wide.loc[units, "Malay"].to_numpy(float)
        other = wide.loc[units, "Other Bumiputera"].to_numpy(float)
        rows = malay + other
        seed = np.zeros((len(units), len(labels)))
        for j, lab in enumerate(labels):
            if st2 in my2020.SEED:
                col = my2020.SEED[st2].get(lab, "Other Bumiputera")
                s10 = seed10.loc[units]
                share = s10[col] / s10[["Malay", "Kadazan Dusun", "Bajau", "Murut",
                                        "Iban", "Bidayuh", "Melanau",
                                        "Other Bumiputera"]].sum(axis=1, min_count=1)
                seed[:, j] = rows * share.fillna(0).to_numpy(float)
            else:
                seed[:, j] = malay if lab == "Melayu Malay" else other
            home = my2020.HOME.get(st2, {}).get(lab)
            if home:
                missing = set(home) - set(names.values())
                if missing:
                    raise SystemExit(f"my: HOME names unknown districts {missing}")
                keep = np.array([names[u] in home for u in units])
                base = seed[:, j] if seed[:, j][keep].sum() > 0 else rows
                seed[:, j] = np.where(keep, base, 0)
            if seed[:, j].sum() == 0 and cols[j] > 0:
                seed[:, j] = rows          # no pattern anywhere in the state: follow Bumiputera
        # the state's Table 5 and its districts' Bumiputera differ by OpenDOSM's rounding
        cols = cols * rows.sum() / cols.sum()
        m = _ipf(seed, rows, cols)
        err = np.abs(m.sum(axis=1) - rows).max()
        if err > 1:
            raise SystemExit(f"my: state {st2} did not converge (district off by {err:.1f})")
        for j, lab in enumerate(labels):
            tier = "derived" if (lab == "Melayu Malay" and st2 not in my2020.SEED) else "modelled"
            out.append(pd.DataFrame({"unit": units, "node": my2020.resolve(lab),
                                     "count": m[:, j], "tier": tier}))
    res = pd.concat(out, ignore_index=True)
    drawn = wide[["Malay", "Other Bumiputera", "Chinese", "Indians", "Others",
                  "Non-citizens"]].sum().sum()
    if abs(res["count"].sum() - drawn) > 1000:
        raise SystemExit(f"my: drew {res['count'].sum():,.0f} of {drawn:,.0f} people")
    res = res[res["count"] > 0]
    return res.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Malaysia",
    source=("Population and Housing Census of Malaysia 2020: ethnic group and non-citizens by "
            "administrative district (DOSM, OpenDOSM) and sub-ethnic group by state (state "
            "volumes, Table 5); Census 2010, Table 11.1 and 12.1 (Sabah and Sarawak, by "
            "district); Census 2000 Chinese dialect groups by state (DOSM tabulation, Voon "
            "2024)"),
    how=("census, 2020, ethnic group, each drawn as its language; Chinese split by the 2000 "
         "census's dialect groups with 53% keeping the dialect and the rest drawn as Mandarin; "
         "Indians split by sub-group; non-citizens drawn on the languages of their "
         "nationalities; Bumiputera sub-groups counted by state and spread over its districts"),
    parts=[
        dict(covers="Chinese citizens",
             source="2020 census count by district, split by the 2000 census's dialect groups "
                    "by state; 53% keep the dialect (Carstens 2014), the rest drawn as Mandarin",
             people=6_891_900),
        dict(covers="Indian citizens",
             source="2020 census count by district, split by sub-group shares",
             people=1_997_900),
        dict(covers="Non-citizens",
             source="2020 census count by district, on the languages of their nationalities",
             people=2_690_600),
        dict(covers="Malays, indigenous groups and others",
             source="2020 census, ethnic group by district; sub-ethnic group by state, spread "
                    "over districts (the 2010 census's pattern in Sabah and Sarawak)",
             rest=True),
    ],
    grain="160 administrative districts, 203,000 people on average; sub-ethnic groups by state",
    view=[99.3, 0.5, 119.5, 7.6],
    counts=_counts,
    mappings=["my2020"],
    place=RD_GEO / "my" / "my_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Malaysia's census asks ethnic group, not language, so this map reads each group as "
        "its language. Nothing measures how many in each group still speak it, and younger "
        "Kadazan-Dusun families in Sabah often speak Sabah Malay at home, so the smaller "
        "languages are likely drawn too large. Chinese Malaysians are split by dialect group "
        "using the 2000 census, the last to count them, by state. Only 53% are drawn on their "
        "dialect, the share of respondents in one 2014 study who spoke a dialect with their "
        "parents, and the rest as Mandarin. Henghua, Hockchia and other small groups, 4.5% of "
        "Chinese, are drawn as Chinese. Indians are split by sub-group, 88% Tamil. The "
        "indigenous groups of Sabah, Sarawak and the Orang Asli are published by state only, "
        "and are placed on districts by the 2020 and 2010 censuses' district counts. "
        "Non-citizens, 8% of the population and a quarter of Sabah's, are counted by district "
        "but not by nationality. They are drawn on the languages of the nationalities that "
        "hold most work passes (Indonesia, Nepal, Bangladesh), and in Sabah on Indonesian and "
        "Filipino migrant languages such as Bugis, Sama and Tausug."),
)
