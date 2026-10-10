# Canada. Census of Population 2021, mother tongue by dissemination area (sources/ca_census.py),
# placed on religiondots' 2021 cartographic DA polygons, read-only, one polygon per unit.
from _shared import *  # noqa: F401,F403
import numpy as np

DA_SHP = RD_GEO / "ca" / "da" / "lda_000b21a_e.shp"
EN_CID, FR_CID = 396, 397


def _spread(target, have, resid, key):
    """Share each (parent, category) deficit, target minus the children's sum, over the parent's
    children in proportion to `resid` (a child's people who are in its total but in no cell it
    publishes). Returns rows [child, cid, count] of the added people."""
    d = (target - have.reindex(target.index).fillna(0)).clip(lower=0)
    d = d[d > 0].rename("deficit").reset_index()               # parent, cid, deficit
    r = resid.rename("w").reset_index()                          # child, parent, w
    wsum = r.groupby("parent")["w"].transform("sum")
    r = r[(r["w"] > 0) & (wsum > 0)]
    r = r.assign(share=r["w"] / r.groupby("parent")["w"].transform("sum"))
    m = d.merge(r[["parent", key, "share"]], on="parent", how="inner")
    m["count"] = m["deficit"] * m["share"]
    lost = d["deficit"].sum() - m["count"].sum()
    return m[[key, "cid", "count"]], float(d["deficit"].sum()), float(lost)


def _counts():
    import ca2021
    t = pd.read_csv(NORM / "ca.csv", dtype={"geo_id": str, "csd": str})
    u = pd.read_csv(NORM / "ca_units.csv", dtype={"geo_id": str, "csd": str}).set_index("geo_id")
    labels = t.drop_duplicates("cid").set_index("cid")["source_category"]
    multi = set(ca2021.MULTIPLE)
    unmapped = sorted(set(labels) - set(ca2021.NAMES) - multi)
    if unmapped:
        raise SystemExit(f"ca: labels with no mapping: {unmapped}")

    # ---- 1. put back the people the dissemination areas' small cells lose ----
    # StatCan's DA cells drop small counts: the DAs' cells hold 0.974 of their own totals and a
    # median 0.75 of each language's national figure (Swedish 0.31), while the provinces' cells
    # add up to the country exactly. Each language's shortfall is taken from the level above
    # (province -> subdivision -> DA) and shared over the children in proportion to their
    # residual: total minus the cells published. The people are the census's; only where inside
    # the parent they sit is inferred, so these rows are `derived`.
    prov = t[t.geo_level == "province"].assign(parent=lambda x: x.geo_id.str[9:11])
    csd = t[t.geo_level == "csd"].assign(parent=lambda x: x.geo_id.str[9:11])
    da = t[t.geo_level == "da"].rename(columns={"csd": "parent"})
    cu = u[u.geo_level == "csd"]
    du = u[u.geo_level == "da"]
    c_res = (cu["total"] - csd.groupby("geo_id")["count"].sum().reindex(cu.index).fillna(0)
             ).clip(lower=0).fillna(0)
    c_res = pd.DataFrame({"parent": cu.index.str[9:11], "w": c_res}).rename_axis("csd")
    add1, want1, lost1 = _spread(prov.set_index(["parent", "cid"])["count"],
                                 csd.groupby(["parent", "cid"])["count"].sum(),
                                 c_res.set_index("parent", append=True)["w"], "csd")
    c_target = (csd.set_index(["geo_id", "cid"])["count"]
                .add(add1.set_index(["csd", "cid"])["count"].rename_axis(["geo_id", "cid"]),
                     fill_value=0))
    c_target.index.names = ["parent", "cid"]
    # a suppressed DA (no total) keeps its published population as its residual
    d_tot = du["total"].fillna(du["pop"]).fillna(0)
    d_res = (d_tot - da.groupby("geo_id")["count"].sum().reindex(du.index).fillna(0)).clip(lower=0)
    d_res = pd.DataFrame({"parent": du["csd"], "w": d_res}).rename_axis("da")
    add2, want2, lost2 = _spread(c_target, da.groupby(["parent", "cid"])["count"].sum(),
                                 d_res.set_index("parent", append=True)["w"], "da")
    print(f"  ca: small-cell shortfall put back: {want1:,.0f} people province->subdivision "
          f"({lost1:,.0f} with nowhere to go), {want2:,.0f} subdivision->DA ({lost2:,.0f})")

    cells = pd.concat([
        da[["geo_id", "cid", "count"]].assign(tier="measured"),
        add2.rename(columns={"da": "geo_id"}).assign(tier="derived")], ignore_index=True)
    cells["label"] = cells["cid"].map(labels)
    nat = t[t.geo_level == "country"].set_index("cid")["count"]
    s = cells.groupby("cid")["count"].sum() / nat
    big = nat[nat >= 5000].index
    print(f"  ca: DA sums / Canada, {len(big)} categories of 5,000+, after: min {s[big].min():.3f}, "
          f"median {s[big].median():.3f}, max {s[big].max():.3f}")

    # ---- 2. single responses ----
    single = cells[~cells["label"].isin(multi)].copy()
    single["node"] = single["label"].map(ca2021.NAMES)

    # ---- 3. multiple responses: 1/k of a person to each language named ----
    mult = cells[cells["label"].isin(multi)]
    rows, nonoff = [], []
    for lab, shares in ca2021.MULTIPLE.items():
        m = mult[mult["label"] == lab]
        for node, f in shares.items():
            part = m[["geo_id", "count", "tier"]].assign(count=m["count"] * f)
            (nonoff if node == ca2021.NONOFFICIAL else rows).append(part.assign(node=node))
    official = pd.concat(rows, ignore_index=True)
    no = pd.concat(nonoff, ignore_index=True).groupby("geo_id")["count"].sum()

    # The non-official part names no language: share it over the DA's own single-response
    # non-official languages; where the DA has none, its subdivision's; then its province's.
    # Inferred, so `derived` (the viewer can leave these out).
    so = single[~single["cid"].isin([EN_CID, FR_CID])]
    mix_da = so.groupby(["geo_id", "node"])["count"].sum()
    mix_csd = t[(t.geo_level == "csd") & ~t.cid.isin([EN_CID, FR_CID, *range(719, 724)])]
    mix_csd = mix_csd.assign(node=mix_csd["source_category"].map(ca2021.NAMES)).groupby(
        ["geo_id", "node"])["count"].sum()
    mix_pr = t[(t.geo_level == "province") & ~t.cid.isin([EN_CID, FR_CID, *range(719, 724)])]
    mix_pr = mix_pr.assign(node=mix_pr["source_category"].map(ca2021.NAMES)).groupby(
        ["geo_id", "node"])["count"].sum()
    pr_id = {g[-2:]: g for g in u[u.geo_level == "province"].index}

    def share_out(mix, need, key_of):
        mix = mix.rename("c").reset_index()
        mix = mix.assign(f=mix["c"] / mix.groupby("geo_id")["c"].transform("sum"))
        k = pd.DataFrame({"geo_id": need.index, "key": need.index.map(key_of), "need": need.values})
        m = k.merge(mix.rename(columns={"geo_id": "key"}), on="key", how="inner")
        return m.assign(count=m["need"] * m["f"])[["geo_id", "node", "count"]], set(k["geo_id"]) - set(m["geo_id"])

    a, left = share_out(mix_da, no, lambda g: g)
    b, left = share_out(mix_csd, no[no.index.isin(left)], lambda g: du.at[g, "csd"])
    c, left = share_out(mix_pr, no[no.index.isin(left)], lambda g: pr_id[g[9:11]])
    if left:
        raise SystemExit(f"ca: {len(left)} DAs' non-official multiple answers found no mix")
    print(f"  ca: multiple answers' non-official part {no.sum():,.0f}: "
          f"{a['count'].sum():,.0f} on the DA's own languages, {b['count'].sum():,.0f} on its "
          f"subdivision's, {c['count'].sum():,.0f} on its province's")
    inferred = pd.concat([a, b, c], ignore_index=True).assign(tier="derived")

    out = pd.concat([single[["geo_id", "node", "count", "tier"]],
                     official[["geo_id", "node", "count", "tier"]], inferred], ignore_index=True)
    out = out.rename(columns={"geo_id": "unit"})
    out = out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    out = out[out["count"] > 0]
    total = du["total"].sum()
    print(f"  ca: {out['count'].sum():,.0f} people drawn against DA totals {total:,.0f} "
          f"({out['count'].sum() / total:.4f}); derived {out.loc[out.tier == 'derived', 'count'].sum():,.0f}")
    return out


ENTRY = dict(
    name="Canada",
    source="Census of Population 2021, Census Profile 98-401-X2021006, mother tongue "
           "(Statistics Canada)",
    how="census, 2021, mother tongue; someone who gave two or three is shared equally "
        "between them",
    parts=[dict(covers="Everyone", source="2021 census, mother tongue", rest=True)],
    grain="57,936 dissemination areas, 630 people on average",
    gap="institutional residents, 371,000 (1.0%), and the 63 Indian reserves and settlements "
        "the census could not fully count, Kahnawake, Akwesasne and Six Nations among them, "
        "for which nothing is published",
    view=[-128.0, 42.0, -55.0, 58.0],
    counts=_counts,
    mappings=["ca2021"],
    # the dissemination areas (DA_SHP) cut by Kontur hexes, so a rural DA's dots follow where its
    # people live instead of spreading over all its land (sources/kontur_cut.py; 2026-10-08)
    place=GEO / "ca" / "ca_konturcut.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asks everyone the first language they learned at home in childhood and "
        "still understand. About 4% gave more than one; each such person is split equally "
        "between the languages they gave. When one of those is \"a non-official language\" the "
        "census does not say which, so that share follows the other languages spoken in the "
        "same neighbourhood. Statistics Canada leaves very small counts out of its "
        "neighbourhood tables; those people are put back from the municipality's figures, "
        "following the neighbourhoods' unreported remainders. Census counts for 63 First "
        "Nations reserves, among them Kahnawake, Akwesasne and Six Nations, were not "
        "published, so those communities have no dots."),
)
