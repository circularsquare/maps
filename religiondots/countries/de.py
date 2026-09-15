# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _DeGridWeighter:
    """Place a German dot on where that religion actually is, not on where people are.

    spec §8.2 splits placement from magnitude, and everywhere else on this map the
    placement weight is a PROXY — an equal share per unit that was engineered to a
    population target, or, for the US, a demographic model fitted to guess a
    denomination's position inside a county (§8.4). Germany needs neither: destatis
    publishes the same three categories on the 1km INSPIRE grid, so the weight for
    `christianity.catholic` inside Munich is Munich's own per-cell Catholic count.

    That makes this the only country here whose within-unit placement is MEASURED. It is
    not a model and carries no fitted parameter, so §7's confidence machinery has nothing
    to mark: a Catholic dot in Neukölln is there because the register put Catholics in
    that square kilometre.

    Falls back to cell population, and then to equal shares, wherever the node has no
    column or sums to zero in that unit — the latter happens where the Cell-Key
    perturbation zeroed a small category across every cell of a small Gemeinde while the
    Gemeinde table still shows a few people (sources/de.md §3).
    """

    COLUMN = {
        "christianity.catholic": "kath",
        "christianity.protestant": "ev",
        "unrecorded": "son",
        # EVERYTHING BELOW IS THE `Sonstige` COLUMN AND NOT THE RELIGION'S OWN, and the
        # paragraphs above do not describe any of it. destatis publishes a grid for its
        # three categories and for nothing else; these nodes are carved out of the third
        # one — Jews by ZWST's count (de_zwst.py), the rest by the ESS model (de_ess.py) —
        # so `son` is the column they were counted in rather than a column that counts
        # them.
        #
        # It is used because inside a city it is a far better locator than total
        # population: `son` is where the register has no church for people, which is
        # where the Muslim, Orthodox and migrant-descended populations actually are. In
        # Neukölln it is most of the neighbourhood and in a Bavarian village it is a
        # tenth of it. But it is a PROXY — the "placement is measured" claim above is
        # true of the three Zensus categories and of nothing else here.
        "judaism": "son",
        "islam": "son",
        "christianity.orthodox": "son",
        "christianity.evangelical": "son",
        "christianity": "son",
        "other.de": "son",
    }
    # christianity.protestant is NOT in PROXY even though the ESS split adds a modelled
    # slice to it: the node's own `ev` column is the right weight for the measured 19.1M
    # that dominates it, and `weights()` is asked per (unit, node) and cannot see tier.
    PROXY = {"judaism", "islam", "christianity.orthodox",
             "christianity.evangelical", "christianity", "other.de"}

    # --- Route B: citizenship sharpens two of those six ---------------------------------
    # `son` is half of Germany and contains every secular German, so on its own it spreads
    # Muslim dots evenly across a city. destatis publishes citizenship on the SAME 1km
    # cells (sources/de_grid.py), and Turkish and Bosnian passports are a far sharper
    # locator: 1.5 million people whose position inside Berlin is Neukölln and Kreuzberg
    # rather than Zehlendorf.
    #
    # IT CANNOT BE USED ALONE, and the reason is the whole design. Citizenship is not
    # origin — most of Germany's Turkish-descended population holds German passports and
    # is invisible to this column — so placing every Muslim dot on it would assert that
    # naturalised families live exactly where non-naturalised ones do, and would put
    # nobody at all in the many cells where the community is entirely German-citizen.
    #
    # SO THE BLEND COMES FROM THE UNIT'S OWN ARITHMETIC, which is `_ItWeighter`'s move
    # (Italy, §8.4a) with the ratio computed rather than chosen. In each Gemeinde the
    # signal accounts for C people out of the node's N: the citizenship vector gets C/N
    # of the weight and `son` keeps the rest. Where the signal is absent it is pure `son`
    # and nothing changes; where it accounts for the whole node it is pure citizenship.
    # No constant is picked anywhere, so there is nothing here to tune.
    CITIZENSHIP = {"islam": "isl_ctz", "christianity.orthodox": "orth_ctz"}

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.col = {k: place[v].to_numpy(dtype=float) for k, v in
                    ((n, c) for n, c in self.COLUMN.items() if c in place.columns)}
        self.ctz = {k: place[v].to_numpy(dtype=float) for k, v in
                    ((n, c) for n, c in self.CITIZENSHIP.items() if c in place.columns)}
        self.n_measured = 0
        self.n_proxy = 0
        self.n_pop = 0
        self.n_uniform = 0
        # per node: [rows sharpened, sum of the blend fraction, rows where it capped at 1]
        self.ctz_stat = {n: [0, 0.0, 0] for n in self.CITIZENSHIP}

    def weights(self, node, idx, count, plain=False):
        col = self.col.get(node)
        if col is not None:
            w = col[idx]
            if w.sum() > 0:
                if node in self.PROXY:
                    w = self._sharpen(node, idx, count, w)
                    self.n_proxy += 1
                else:
                    self.n_measured += 1
                return w
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def _sharpen(self, node, idx, count, base):
        """Blend the `son` weight with the citizenship signal, in the unit's own ratio."""
        sig = self.ctz.get(node)
        if sig is None or not count:
            return base
        c = sig[idx]
        total = c.sum()
        if total <= 0:
            return base
        raw = float(total) / float(count)
        f = min(1.0, raw)                  # how much of the node the signal accounts for
        s = self.ctz_stat[node]
        s[0] += 1
        s[1] += f
        s[2] += raw >= 1.0
        return (c / total) * f + (base / base.sum()) * (1.0 - f)

    def summary(self):
        proxy = (f", {self.n_proxy:,} on the `Sonstige` column they were carved out of "
                 f"(a proxy and not that religion's own count)" if self.n_proxy else "")
        bits = []
        for node, (n, tot, cap) in self.ctz_stat.items():
            if not n:
                continue
            # A node capped almost everywhere is NOT a healthy blend: it means the
            # measured citizenship count exceeds the modelled magnitude across the
            # country, so the weight is citizenship alone. Said out loud rather than
            # buried, because it is also evidence about the MODEL — see sources/de.md §10.
            bits.append(f"{node} {n:,} rows at {100 * tot / n:.0f}% citizenship"
                        + (f", CAPPED in {100 * cap / n:.0f}% of them" if cap else ""))
        ctz = (" — sharpened by the citizenship grid: " + "; ".join(bits)) if bits else ""
        return (f"{self.n_measured:,} (unit, node) rows placed on that religion's OWN 1km "
                f"grid counts{proxy}{ctz}, {self.n_pop:,} on cell population where the "
                f"category is zero across the unit's cells, {self.n_uniform:,} on equal "
                f"shares (sources/de_grid.py)")


def _de_place_weight(place):
    """countries.py hook. `place` is the 1km grid GeoDataFrame scatter.py has read."""
    if "kath" not in place.columns:
        print("  !! de_grid_1km.gpkg has no religion columns — run sources/de_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _DeGridWeighter(place)


def _de_counts():
    """Zensus 2022 at Gemeinde: three categories on 10,786 units.

    The shallowest country on the map and the only one where that is a property of the
    instrument. Zensus 2022 asks nothing about religion; the figures are read off the
    Melderegister, which records membership of the two churches that levy church tax, so
    `basis` is `roll` (spec §3.1) and 51.8% of the country lands on one node.

    No allocation step, and none is possible: destatis publishes at no coarser AND no
    finer CATEGORY than this. What it does publish finer is geography — the same three
    numbers on a 100m grid — which is a placement upgrade rather than a detail upgrade.

    ONE level. de.csv carries `country` alongside `gemeinde`, which is the same 82.7
    million people counted twice.

    Every category is positive somewhere and the three partition each Gemeinde, so unlike
    Czechia there are explicit zeros to strip: 178 cells across the file are the true-zero
    dash, mostly Catholics in East German villages.
    """
    from de2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "de.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "gemeinde"].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]

    df = _de_add_zwst(df)
    df = _de_split_ess(df)

    df["congregations"] = 0
    if "tier" not in df.columns:
        return df[["unit", "node", "count", "congregations"]]
    return df[["unit", "node", "count", "congregations", "tier"]]


def _de_add_zwst(df):
    """Carve the Jewish communities out of `unrecorded` — sources/de_zwst.py, de.md §7.

    THE ONLY PART OF GERMANY'S 42.8 MILLION GREY CELL THAT A SECOND SOURCE CAN COUNT
    RATHER THAN ESTIMATE. ZWST publishes membership per community and de_zwst.py seats
    each one in a Gemeinde, so this is a `roll` meeting a `roll` (spec §3.1) and Germany
    stays `measured` throughout — unlike the ESS split scouted in de.md §6, which would
    spend that.

    SUBTRACT BEFORE ADDING. These people are inside `Sonstige, keine, ohne Angabe` today,
    because Jewish communities are public-law religious societies in most Laender and the
    Melderegister lumps every body that is not the two big churches into that one cell.
    Adding a judaism row without taking it back out of `unrecorded` would give Germany
    87,934 people it does not have and break the unit totals.

    The subtraction cannot go negative: de_zwst.py asserts every seat has room, and the
    tightest is Straubing at 4.5% of its `unrecorded`. It is asserted again here anyway,
    because that check lives in a file this one does not run.
    """
    path = HERE / "data" / "normalized" / "de_zwst.csv"
    if not path.exists():
        print("  !! de_zwst.csv missing — run sources/de_zwst.py; Germany's Jewish "
              "communities stay inside `unrecorded` (sources/de.md §7)")
        return df

    z = pd.read_csv(path, dtype={"geo_id": str})
    members = z.groupby("geo_id")["count"].sum()

    seats = df["unit"].isin(members.index)
    grey = df["node"] == "unrecorded"
    take = df.loc[seats & grey, "unit"].map(members)

    if len(take) != len(members):
        missing = sorted(set(members.index) - set(df.loc[grey, "unit"]))
        raise SystemExit(f"de_zwst: {len(missing)} seats have no `unrecorded` row to "
                         f"subtract from, e.g. {missing[:5]} — did de.csv change vintage?")
    short = df.loc[take.index, "count"] - take
    if (short < 0).any():
        bad = df.loc[short[short < 0].index, "unit"].tolist()
        raise SystemExit(f"de_zwst: members exceed `unrecorded` in {bad} — the seat join "
                         "is wrong, or the two sources are different vintages")
    df.loc[take.index, "count"] = short

    add = pd.DataFrame({"unit": members.index, "node": "judaism",
                        "count": members.to_numpy()})
    print(f"  {members.sum():,} ZWST members moved from `unrecorded` to `judaism` "
          f"in {len(members)} Gemeinden (sources/de_zwst.py)")
    return pd.concat([df, add], ignore_index=True)


def _de_split_ess(df):
    """Split what is left of `unrecorded` by each Bundesland's ESS composition.

    spec §3.4's SPLIT and §14.10's fractional-share model, and everything about it is in
    sources/de_ess.py. In one line: the register counted 42.8 million people into a cell
    it cannot see inside, ESS asked 13,643 Germans which denomination they belong to, and
    the register's own Catholic and Protestant totals are what makes the two comparable
    — ESS reproduces them to within 0.4 and 0.8 points without being told either.

    THE SPLIT IS PARTIAL ON PURPOSE, and this is the design decision. Only the ~19% of the
    residual that ESS assigns to a NAMED RELIGION is drawn. The 41% who told ESS they
    belong to no religion stay in `unrecorded`, where the register put them, because
    (a) `unrecorded` is a measured cell and replacing it with a modelled `unaffiliated`
    trades a count for an estimate, and (b) §14.12's rule is that ancestry-shaped cells
    model well and attitude-shaped ones badly — Islam and Orthodoxy in Germany are close
    to functions of descent, and non-belief is not. So Germany stays about 90% measured
    and `inferred dots: not shown` returns it to nearly the map it is today.

    Everything here is `modelled` and not `derived`, on §7b's test: the tiers are about
    whether anybody was counted, and nobody counted Germany's Muslims at any level. Note
    what that means for §7's control — the viewer REMOVES modelled dots rather than
    rolling them up, so these do not fall back to `unrecorded`, they go.
    """
    path = HERE / "data" / "normalized" / "de_ess.csv"
    if not path.exists():
        print("  !! de_ess.csv missing — run sources/de_ess.py; Germany's Muslims, "
              "Orthodox and free churches stay inside `unrecorded` (sources/de.md §6)")
        return df

    from de2022 import resolve

    e = pd.read_csv(path, dtype={"geo_id": str})
    e["node"] = e["source_category"].map(resolve)
    if e["node"].isna().any():
        bad = sorted(e.loc[e["node"].isna(), "source_category"].unique())
        raise SystemExit(f"de_ess: unmapped ESS categories {bad} — add them to "
                         "taxonomy/de2022.py")
    # one share per (Land, node): two ESS answers can share a node (other.de)
    share = e.groupby(["geo_id", "node"])["count"].sum()

    # the residual stops being whole people the moment it is split, and leaving `count`
    # as int64 makes the subtraction below a silent pandas dtype error
    df["count"] = df["count"].astype(float)

    grey = df["node"] == "unrecorded"
    land = df.loc[grey, "unit"].str[:2]
    bucket = df.loc[grey, "count"].to_numpy(dtype=float)

    frames, taken = [], pd.Series(0.0, index=df.index[grey])
    for (gid, node), s in share.items():
        m = (land == gid).to_numpy()
        if not m.any():
            raise SystemExit(f"de_ess: no Gemeinde has Land prefix {gid!r}")
        n = bucket * m * float(s)
        taken += pd.Series(n, index=taken.index)
        keep = n > 0
        frames.append(pd.DataFrame({
            "unit": df.loc[grey, "unit"].to_numpy()[keep],
            "node": node, "count": n[keep], "tier": "modelled"}))

    if (taken > df.loc[grey, "count"]).any():
        raise SystemExit("de_ess: a Land's shares take more than its residual")
    df.loc[grey, "count"] = df.loc[grey, "count"] - taken

    add = pd.concat(frames, ignore_index=True)
    add = add.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    df["tier"] = "measured"
    out = pd.concat([df, add], ignore_index=True)

    n = add["count"].sum()
    print(f"  {n:,.0f} people ({100 * n / df['count'].sum():.1f}% of Germany) split out "
          f"of `unrecorded` into {add['node'].nunique()} religions by Bundesland, "
          f"MODELLED (sources/de_ess.py)")
    return out


ENTRY = {
    "de": dict(
        name="Germany",
        source="Zensus 2022, Sonderauswertung Religionszugehörigkeit (Destatis)",
        basis="administrative register (church-tax records), not a question",
        view=[5.4, 47.0, 15.6, 55.3],
        note_public=(
            "Nobody was asked. The 2022 census carries no religion question at all, and "
            "these figures are read off the population register, which records church "
            "membership because it determines church tax. So the data can see the two "
            "churches that levy it and nothing else: 51.8% of Germany was one grey "
            "category holding everyone the register has no religious body for — Muslims, "
            "Orthodox Christians, the free churches, the Old Catholics and everyone who "
            "belongs to nothing, indistinguishable from each other, because the register "
            "never knew. That is not a judgement about who counts; it is the shape of the "
            "instrument. Two things have been pulled back out of that cell, and they are "
            "not equally solid. The 87,934 members of Germany's 105 Jewish communities "
            "are COUNTED, by their own welfare organisation, and placed at each "
            "community's seat — affiliated membership rather than everyone Jewish, and a "
            "community serves a whole region from one address, so read those dots as "
            "where the communities are. The Muslim, Orthodox, free-church and other dots "
            "are MODELLED: nobody counts them anywhere in Germany, so the European Social "
            "Survey's answers from 13,643 Germans are used to divide the grey cell, one "
            "composition per federal state. Two independent instruments agreeing is what "
            "licenses that — the survey finds 24.7% Catholic and 23.9% Protestant where "
            "the register counts 25.1% and 23.1%, without being told either. The Muslim "
            "dots are the weakest thing on this page and they are too few: the survey "
            "implies about 3.5 million where the federal migration office estimates 5.5, "
            "because a German-language household survey of adults misses recent arrivals "
            "and a young population. The figure is left as the survey reports it rather "
            "than scaled to fit. The 41% who told the survey they belong to no religion "
            "are deliberately NOT drawn as non-religious and stay in the grey: the "
            "register genuinely counted them into it, and where somebody stands on belief "
            "is the last thing a model like this can see. What survives untouched is the "
            "confessional map itself, at 10,786 municipalities: Catholic Bavaria, the "
            "Rhineland and the Saarland against a Protestant north, a boundary largely "
            "settled in the sixteenth century and still legible village by village. The "
            "sharpest line is not that one. In the former East, 81% belong to no church, "
            "against 45% in the West, and the Eichsfeld — a Catholic enclave that stayed "
            "Catholic through forty years of the GDR — still reads at 80% against a "
            "Thuringia of 74% none."),
        how="church-tax registers; nobody was asked",
        grain="municipalities, 7,700 people on average",
        counts=_de_counts,
        # Gemeinden are the COUNT layer; the 1km INSPIRE grid is the PLACEMENT layer, and
        # Germany is the one country on this map where §8.2's approximation is dropped
        # outright rather than bounded (sources/de_grid.py).
        #
        # It had to be. Gemeinden are historical units, not units engineered to a
        # population target, so §8.2's usual trick does not apply at all: they run from
        # Dierfeld's 9 people to Berlin's 3,596,999 in ONE polygon, and 78 of them hold
        # 31.6% of the country. The median is 1,797, finer than a Polish gmina, so two
        # thirds of Germany was already drawn well — and the other third was drawn as
        # city-sized blobs, with Neukölln and Zehlendorf identical. Unlike Czechia's
        # Prague and Estonia's Tallinn there is no district-level religion table to swap
        # in.
        #
        # What replaced it is not a finer proxy, it is the same measurement at 1km:
        # destatis publishes THE SAME THREE CATEGORIES per grid cell, so `place_weight`
        # weights each religion by its OWN count in each square kilometre. Berlin goes
        # from 1 polygon to 799 cells, Hamburg to 655, Munich to 305. The US needs a
        # fitted demographic model to do this (§8.4); Germany just reads it.
        #
        # 34 Gemeinden holding 8,199 people (0.0099%) are too small for any 1km centre to
        # land inside them and carry their own polygon as a single cell, so the layer
        # covers all 10,786 units and nothing is unplaceable.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "de" / "de_grid_1km.gpkg",
        place_unit=lambda g: g["ars"].astype(str),
        place_weight=_de_place_weight,
        note="Basis is `roll`, not `self_id`: this is a register of church-tax "
             "liability, so it is comparable with the United States and NOT with any "
             "census that asks the person (spec §3.1). Counts carry Cell-Key "
             "perturbation, so categories do not sum exactly to the published "
             "population — by 174 people in 82.7 million (sources/de.md §3). Judaism is "
             "a SECOND SOURCE on the same basis: ZWST's 2025 membership statistic, "
             "87,934 people in 105 communities seated in 96 Gemeinden and subtracted out "
             "of `unrecorded` rather than added to it (sources/de.md §7). A roll counts "
             "the institution and not the member (§3.6), and these are regional "
             "catchments — Düsseldorf's 6,371 covers much of the lower Rhine. Islam, "
             "Orthodoxy, the free churches and other.de are a THIRD source and the only "
             "modelled part of the country: ESS `rlgdnade` pooled over rounds 6-11 "
             "splits the residual by Bundesland (§3.4, §14.10), 9.7% of Germany, "
             "`modelled` in §7 and therefore removed rather than rolled up when inferred "
             "dots are hidden. The non-religious are NOT split out and stay measured in "
             "`unrecorded` — §14.12's ancestry/attitude rule (sources/de_ess.py). Those "
             "dots are PLACED on destatis' citizenship grid, which shares cell ids with "
             "the religion one: Turkish and Bosnian passports for Islam, Greek, Romanian, "
             "Russian and Ukrainian for Orthodoxy, blended with `son` in each Gemeinde's "
             "own ratio (sources/de.md §10). Placement only, never a magnitude (§8.2).",
    ),
}
