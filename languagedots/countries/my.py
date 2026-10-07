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
    # Chinese, Indians, Others: the district's own counts
    for col in ("Chinese", "Others"):
        out.append(pd.DataFrame({"unit": wide.index, "node": my2020.NAMES[col],
                                 "count": wide[col].to_numpy(float), "tier": "derived"}))
    for node, share in my2020.INDIAN:
        out.append(pd.DataFrame({"unit": wide.index, "node": node,
                                 "count": wide["Indians"].to_numpy(float) * share,
                                 "tier": "derived"}))
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
    citizens = wide[["Malay", "Other Bumiputera", "Chinese", "Indians", "Others"]].sum().sum()
    if abs(res["count"].sum() - citizens) > 1000:
        raise SystemExit(f"my: drew {res['count'].sum():,.0f} of {citizens:,.0f} citizens")
    res = res[res["count"] > 0]
    return res.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Malaysia",
    source=("Population and Housing Census of Malaysia 2020: ethnic group by administrative "
            "district (DOSM, OpenDOSM) and sub-ethnic group by state (state volumes, Table 5); "
            "Census 2010, Table 11.1 and 12.1 (Sabah and Sarawak, by district)"),
    how=("census, 2020, ethnic group, each drawn as its language (Indians 80% Tamil); "
         "Bumiputera sub-groups counted by state and spread over its districts"),
    parts=[
        dict(covers="Chinese, Indians, others, and Malays outside Sabah and Sarawak",
             source="2020 census, ethnic group by district, drawn as its language",
             people=25_206_132),
        dict(covers="Indigenous groups, and Malays in Sabah and Sarawak",
             source="2020 census, sub-ethnic group by state, spread over districts (the 2010 "
                    "census's pattern in Sabah and Sarawak)",
             rest=True),
    ],
    grain="160 administrative districts, 203,000 people on average; sub-ethnic groups by state",
    gap="2.69 million non-citizens (8.3%), whom the census gives no ethnic group",
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
        "languages are likely drawn too large. Chinese Malaysians are drawn as Chinese without "
        "naming Hokkien, Hakka, Cantonese or the others, which no recent census counts. "
        "Indians are drawn 80% Tamil. The indigenous groups of Sabah, Sarawak and the Orang "
        "Asli are published by state only, and are placed on districts by the 2020 and 2010 "
        "censuses' district counts. Non-citizens, 8% of the population and a quarter of "
        "Sabah's, are not drawn."),
)
