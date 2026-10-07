# Australia. Census 2021, language used at home (sources/au_census.py): G13 by SA2 for English
# and 35 named languages; the five remainders shared out over their four-digit members using
# each SA2's QuickStats top five and the state tables. Placed on religiondots' SA1 polygons,
# read-only, an SA2's dots shared equally over its SA1s (built to ~400 people each).
from _shared import *  # noqa: F401,F403
import importlib.util
import numpy as np

SA1_SHP = RD_GEO / "au" / "SA1_2021_AUST_GDA2020" / "SA1_2021_AUST_GDA2020.shp"
SA2_SHP = RD_GEO / "au" / "SA2_2021_AUST_GDA2020" / "SA2_2021_AUST_GDA2020.shp"
KERNEL_KM = 30.0     # how far a language's measured presence pulls its unplaced speakers
LAMBDA = 0.9         # share of the seed that follows that pull; the rest is even


def _src():
    p = ROOT / "sources" / "au_census.py"
    spec = importlib.util.spec_from_file_location("ld_au_census", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _centroids():
    import geopandas as gpd
    g = gpd.read_file(SA2_SHP)
    g = g[~(g.geometry.isna() | g.geometry.is_empty)].to_crs(3577)
    c = g.geometry.representative_point()
    return pd.DataFrame({"x": c.x.values / 1000, "y": c.y.values / 1000},
                        index=g["SA2_CODE21"].astype(str).values)


def _ipf(seed, rows, cols, cap, iters=60):
    """Rake seed (u x l) to row sums `rows` and column sums `cols` (scaled to sum(rows)),
    never above `cap`. Ends on the rows, so each SA2's remainder is met exactly unless every
    cell in it is capped; what cannot be placed is returned per row."""
    x = seed.copy()
    cols = cols * rows.sum() / cols.sum()
    for _ in range(iters):
        rs = x.sum(1)
        x *= np.divide(rows, rs, out=np.zeros_like(rows), where=rs > 0)[:, None]
        np.minimum(x, cap, out=x)
        cs = x.sum(0)
        x *= np.divide(cols, cs, out=np.zeros_like(cols), where=cs > 0)[None, :]
        np.minimum(x, cap, out=x)
    rs = x.sum(1)
    x *= np.divide(rows, rs, out=np.zeros_like(rows), where=rs > 0)[:, None]
    np.minimum(x, cap, out=x)
    return x, rows - x.sum(1)


def _counts():
    import au2021
    src = _src()
    t = pd.read_csv(NORM / "au.csv", dtype={"geo_id": str, "code": str})
    u = pd.read_csv(NORM / "au_units.csv", dtype={"geo_id": str, "state": str})
    pseudo = set(u.loc[u["pseudo"], "geo_id"])
    t = t[~t["geo_id"].isin(pseudo)]
    state_of = dict(zip(u["geo_id"], u["state"]))
    rest = {lab for lab, _ in src.G13_REST.values()}

    g13 = t[t.geo_level == "sa2"]
    named = g13[~g13["source_category"].isin(rest)]
    out = [named[["geo_id", "source_category", "count"]].assign(tier="measured")]

    # ---- the remainders ----
    top = t[t.geo_level == "sa2_top"].copy()
    top["bucket"] = [src.bucket_of(c, lab) for c, lab in zip(top["code"], top["source_category"])]
    n_listed = top.groupby("geo_id").size()
    fifth = top.groupby("geo_id")["count"].min()          # anything unlisted is at most this
    st = t[t.geo_level == "state"].copy()
    nat = t[t.geo_level == "country"].set_index("source_category")["count"]
    # Other Territories (Christmas, Cocos, Jervis Bay, Norfolk) have no column in Table 5:
    # Australia minus the eight states
    ot = (nat - st.groupby("source_category")["count"].sum().reindex(nat.index).fillna(0)).clip(lower=0)
    ot = ot[ot > 0].rename("count").reset_index()
    lab_bucket = dict(zip(t.loc[t.geo_level == "country", "source_category"],
                          t.loc[t.geo_level == "country", "bucket"]))
    ot = ot.assign(geo_id="9", bucket=ot["source_category"].map(lab_bucket))
    st = pd.concat([st[["geo_id", "source_category", "bucket", "count"]], ot], ignore_index=True)

    cen = _centroids()
    stats = {"measured": 0.0, "derived": 0.0, "unplaced": 0.0, "scaled": 0}
    for (s, b), grp in g13[g13["source_category"].isin(rest)].groupby(
            [g13["geo_id"].map(state_of), "source_category"]):
        B = grp.set_index("geo_id")["count"]
        B = B[B > 0]
        if B.empty:
            continue
        q = top[(top["bucket"] == b) & top["geo_id"].isin(B.index)]
        q = q.pivot_table(index="geo_id", columns="source_category", values="count",
                          aggfunc="sum").reindex(B.index).fillna(0)
        over = q.sum(1) > B
        if over.any():                        # separately perturbed tables: trim to G13's cell
            stats["scaled"] += int(over.sum())
            q.loc[over] = q.loc[over].mul(B[over] / q.loc[over].sum(1), axis=0)
        qm = q.stack()
        qm = qm[qm > 0].rename("count").reset_index()
        out.append(qm.assign(tier="measured"))
        stats["measured"] += qm["count"].sum()
        R = (B - q.sum(1)).clip(lower=0)
        R = R[R > 0.5]
        if R.empty:
            continue
        comp = st[(st["geo_id"] == s) & (st["bucket"] == b)].set_index("source_category")["count"]
        comp = (comp - q.sum(0).reindex(comp.index).fillna(0)).clip(lower=0)
        comp = comp[comp > 0]
        if comp.empty:                         # the state publishes no member left over
            out.append(pd.DataFrame({"geo_id": R.index, "source_category": b,
                                     "count": R.values, "tier": "derived"}))
            stats["derived"] += R.sum()
            continue
        langs = comp.index
        # seed: where the language is measured nearby (QuickStats, the same state and bucket),
        # mixed with an even share
        allq = top[(top["bucket"] == b) & (top["geo_id"].map(state_of) == s)]
        allq = allq.pivot_table(index="geo_id", columns="source_category", values="count",
                                aggfunc="sum").reindex(columns=langs).fillna(0)
        cu = cen.reindex(R.index)
        cv = cen.reindex(allq.index)
        ok_u = cu.notna().all(1).to_numpy()
        seed = np.full((len(R), len(langs)), 1.0 / len(R))
        if len(allq) and ok_u.all():
            d = np.hypot(cu["x"].to_numpy()[:, None] - cv["x"].to_numpy()[None, :],
                         cu["y"].to_numpy()[:, None] - cv["y"].to_numpy()[None, :])
            k = np.exp(-d / KERNEL_KM) @ allq.to_numpy()            # u x l
            ks = k.sum(0)
            has = ks > 0
            k[:, has] /= ks[has]
            seed[:, has] = (1 - LAMBDA) / len(R) + LAMBDA * k[:, has]
        # cap: a language missing from an SA2's top-five list is at most the fifth entry; a list
        # shorter than five is complete, so nothing unlisted is there at all
        listed = top[top["geo_id"].isin(R.index)].pivot_table(
            index="geo_id", columns="source_category", values="count", aggfunc="sum")
        listed = listed.reindex(index=R.index, columns=langs)
        capv = np.where(n_listed.reindex(R.index).fillna(0).to_numpy() >= 5,
                        fifth.reindex(R.index).fillna(np.inf).to_numpy(), 0.0)
        cap = np.where(listed.notna().to_numpy(), np.inf, capv[:, None])
        x, left = _ipf(seed, R.to_numpy(dtype=float), comp.to_numpy(dtype=float), cap)
        m = pd.DataFrame(x, index=R.index, columns=langs).stack()
        m = m[m > 0].rename("count").reset_index()
        out.append(m.assign(tier="derived"))
        stats["derived"] += m["count"].sum()
        if (left > 1e-6).any():
            lf = pd.DataFrame({"geo_id": R.index, "source_category": b, "count": left,
                               "tier": "derived"})
            out.append(lf[lf["count"] > 1e-6])
            stats["unplaced"] += left.sum()

    df = pd.concat(out, ignore_index=True)
    df.columns = ["geo_id", "source_category", "count", "tier"]
    unknown = sorted(set(df["source_category"]) - set(au2021.NAMES) - au2021.NOT_DRAWN)
    if unknown:
        raise SystemExit(f"au: labels with no mapping: {unknown}")
    dropped = df[df["source_category"].isin(au2021.NOT_DRAWN)]["count"].sum()
    df = df[~df["source_category"].isin(au2021.NOT_DRAWN)]
    df["node"] = df["source_category"].map(au2021.resolve)
    df = df.rename(columns={"geo_id": "unit"})
    out = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    out = out[out["count"] > 0]
    print(f"  au: remainders, {stats['measured']:,.0f} people named by QuickStats in their "
          f"own SA2 (measured; {stats['scaled']} SA2 cells trimmed to G13), "
          f"{stats['derived']:,.0f} shared out by state and proximity (derived), of which "
          f"{stats['unplaced']:,.0f} left on the remainder's group node; "
          f"{dropped:,.0f} answers naming no language not drawn")
    print(f"  au: {out['count'].sum():,.0f} people drawn, derived "
          f"{out.loc[out.tier == 'derived', 'count'].sum():,.0f}")
    return out


ENTRY = dict(
    name="Australia",
    source="Census of Population and Housing 2021: General Community Profile G13, QuickStats "
           "and the Cultural diversity data summary (Australian Bureau of Statistics)",
    how="census, 2021, language used at home",
    parts=[dict(covers="Everyone", source="2021 census, language used at home", rest=True)],
    grain="2,417 statistical areas (SA2), 10,500 people on average",
    gap="the 1.44 million (5.7%) who did not answer, 42,000 whose answer named no language "
        "(\"inadequately described\", \"non-verbal\"), and 53,000 people offshore, on ships "
        "or with no usual address",
    view=[112.0, -44.0, 154.5, -9.5],
    counts=_counts,
    mappings=["au2021"],
    place=SA1_SHP,
    place_unit=lambda g: g["SA2_CODE21"].astype(str),
    # SA1s are built to about 400 people, so equal shares per SA1 are already a population
    # weighting (religiondots sources/au_geo.md §4 measured it)
    place_weight=None,
    note_public=(
        "The census asks which language other than English each person uses at home, so "
        "this map shows home language, not mother tongue. The small-area table names English "
        "and 35 other languages; everyone else is grouped as other Chinese, other Indo-Aryan, "
        "other Southeast Asian Austronesian, Australian Indigenous or other. Those groups are "
        "split into their languages using the five largest languages the Bureau publishes for "
        "each area, which puts about 70% of Indigenous-language speakers in their own area, "
        "and the rest follows each state's totals for each language, placed towards where "
        "that language is named nearby. Those shared-out dots are inferred, not counted."),
)
