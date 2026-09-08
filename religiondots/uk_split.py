"""spec.md §3.5a inside Christianity — give England's 26.2 million Christians a denomination.

WHAT IT DOES, in one sentence: the census says how many Christians live in each Output Area,
the English Church Census 2005 says which denominations are where, the British Election
Study says how many people belong to each, and this file multiplies the three together.

WHY. **England and Wales publish no Christian denomination at any geography** (sources/uk.md
§2). Every neighbour splits its Christians at least two ways — Scotland three, Northern
Ireland twenty-six, Ireland fourteen, France, Germany and the Netherlands at least two — and
England is the flat patch on the map. Anita, 2026-09-07: "england just looks different from
everywhere else in europe cuz everyone else breaks down christianity at least a little bit."

THE THREE INPUTS AND WHAT EACH IS ALLOWED TO DECIDE:

  sources/uk.py       TS030 at Output Area. **Every magnitude.** 26,167,899 Christians in
                      England, 188,880 areas, and this file never changes a single one of
                      them — it only says what fraction of each is Anglican.
  sources/uk_ecc.py   the English Church Census 2005. **Placement only.** Which counties and
                      which settlement types each denomination is concentrated in. Its
                      levels are attendance and are thrown away; see below.
  sources/uk_bes.py   the British Election Study, wave 21. **The national mix.** How many
                      English adults belong to each denomination, which is the thing
                      attendance cannot tell you.

WHY THE CHURCH CENSUS CANNOT SUPPLY THE MIX, WHICH IS THE WHOLE REASON THIS FILE IS SHAPED
LIKE THIS. Only about 12% of England's Christians were in church on any given Sunday in 2005,
and the ones who were are not a random twelfth. Catholics attend far more per head than
Anglicans, so the two sources disagree violently about the same country:

      leg           church census (attendance)   BES (identity)   ratio applied here
      anglican                31.3%                  64.6%             x2.06
      catholic                34.1%                  18.2%             x0.53

Reading denominational shares off attendance would have drawn an England with more practising
Catholics than Anglicans, which is true of churchgoing and false of the population. §3.1
forbids the addition; what it permits, and what this is, is using one basis to SPLIT another.

THE ARITHMETIC, per Output Area:

    cell        the OA's ECC county x its RUC21 settlement class (urban or rural)
    shape       for each leg, that cell's share of the leg's England-wide weight
    national    the leg's share of England's Christians, from BES
    weight      shape x national, normalised across legs within the cell
    dots        the OA's census Christian count x weight

**`shape` IS BUILT WITHIN A LEG AND NEVER ACROSS LEGS, AND THAT IS NOT A STYLISTIC CHOICE.**
uk_ecc.py's docstring has the measurement: the settlement code is missing for 37% of churches
and the missingness is denominational — 67% of Anglicans carry one against 53% of Catholics,
36% of Pentecostals and 8% of Orthodox. Reading a cell's mix straight off the coded churches
would make every English conurbation about twelve points more Anglican and ten points less
Catholic than the census found it, from a missing-data pattern rather than from anything
about English religion. Taking each leg's own distribution across its own coded churches
cancels the coding rate, because it appears in the numerator and denominator of the same leg.

THE FALLBACK LADDER, which is br_rescale.py's: **county x settlement, then county, then
England.** A leg with fewer than MIN_CHURCHES churches in a cell has no usable settlement
signal there and falls back to its county total; a leg absent from a county falls back to its
national share. `--report` prints how much of each leg lands on each rung.

ORTHODOXY IS NOT PLACED BY THE CHURCH CENSUS AND CANNOT BE. 49 Orthodox churches responded, a
7% rate because 8 May 2005 fell the Sunday after Orthodox Easter and many were shut, and
**zero of the 94 cells reach three churches.** Meanwhile BES puts Orthodoxy at 2.2% of
England's Christians, about 476,000 adults, larger than the Baptists — Romanian, Bulgarian,
Ukrainian and Greek migration almost all of which postdates the census. A 2005 church count
is not merely thin here, it is describing a different country. Anita's call, 2026-09-07:
place it by census country of birth instead. See `orthodox_shape()`.

WHAT RESTS ON ASSUMPTIONS, AND HOW BIG EACH ONE IS. Both are stated rather than corrected,
and both are Anita's calls of 2026-09-07:

  * **The unclassified Christians, 26.6%.** BES classifies 73.4% of the census's adult
    Christians; the rest say "Christian" to a census and will not pick a denomination on a
    survey. Applying BES's ratios to everyone assumes those 5.8 million divide like the
    people who did pick. Untestable here, and the largest single assumption on this map's
    England. sources/uk_bes.py has the argument for both directions.
  * **The children, 15.9%.** BES is 18+, the census is everyone, and 4.17M child Christians
    get their denomination from adults. Chosen over leaving them on a bare `christianity`
    node: "id rather not have a bare christianity group in england, i feel like that'd be
    confusing." Measured cost: giving children their parents' generation's split instead
    would move the England anchor by at most 1.4 points, on `anglican`.

TIER. Every row this file emits is `derived` (§7), not `modelled`: unlike us_rebase.py's
state-level Pew figures, the coarse totals here were COUNTED — the census counted the
Christians in every Output Area and the church census counted the churches in every county.
What is inferred is only which of them is which. The `orthodox` rows are the exception and
are `modelled`, because their placement comes from a proxy variable rather than from any
count of Orthodox people.

Run: python uk_split.py --report   print the shape, the ladder and the checks, write nothing
     python uk_split.py            -> data/normalized/uk_split.csv
"""
import argparse
import csv
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw", "uk")
NORM = os.path.join(HERE, "data", "normalized")
OUT = os.path.join(NORM, "uk_split.csv")

OA_RUC = os.path.join(RAW, "oa_ruc.csv")      # OA21CD -> RUC21CD, RUC21NM
OA_LAD = os.path.join(RAW, "oa_lad.csv")      # OA21CD -> LAD23CD, LAD23NM
OA_MSOA = os.path.join(RAW, "oa_msoa.csv")    # OA21CD -> MSOA21CD, for the Orthodox proxy

MIN_CHURCHES = 3        # below this a cell has no usable settlement signal for a leg

# RUC21 is six categories: three settlement sizes crossed with proximity to a major town.
# Only the urban/rural line is used -- see uk_ecc.py on why the settlement tier is gone.
RURAL_PREFIX = "R"

# ---------------------------------------------------------------------------------------
# The English Church Census's 47 counties, rebuilt from April 2023 local authority
# districts.
#
# **THIS TABLE EXISTS BECAUSE THE ECC'S GEOGRAPHY WAS ABOLISHED BEFORE THE CENSUS WAS
# TAKEN.** `cntycde` is the 1974-1996 county set with two later edits: London is split Inner
# and Outer, and Herefordshire appears under its post-1998 unitary name. So **Avon, Cleveland
# and Hereford UA and Worcester are all present in 2005 data**, and Avon and Cleveland had
# been gone for nine years. Joining ECC counties to any modern county geography by name
# loses them silently, which is sources/uk.md §4's trap pointing the other way.
#
# Rebuilt rather than approximated, because the successors are exact unions:
#   Avon                       Bristol + Bath and North East Somerset + North Somerset
#                              + South Gloucestershire  (so Somerset and Gloucestershire
#                              below must EXCLUDE those, which is why Somerset is one row)
#   Cleveland                  Hartlepool + Middlesbrough + Redcar and Cleveland
#                              + Stockton-on-Tees  (excluded from Durham and North Yorkshire)
#   Hereford UA and Worcester  Herefordshire + the six Worcestershire districts
#   East Yorkshire (combined)  East Riding of Yorkshire + Kingston upon Hull, Humberside's
#                              north bank; its south bank (North Lincolnshire, North East
#                              Lincolnshire) goes to Lincolnshire, where the ECC puts it
#   Leicestershire             includes Rutland, a county again since 1997 and part of
#                              Leicestershire in 1974-96. The ECC has no Rutland row.
#
# INNER VS OUTER LONDON uses ONS's statistical definition, 14 boroughs including the City,
# Haringey and Newham. Brierley does not say which he used and the alternative 12-borough
# reading would move Haringey and Newham across; both are defensible and the ONS one is the
# published standard. It matters: Inner London is 913 of the census's churches.
ECC_COUNTY = {
    "Avon": ("Bath and North East Somerset", "Bristol", "North Somerset",
             "South Gloucestershire"),
    "Bedfordshire": ("Bedford", "Central Bedfordshire", "Luton"),
    "Berkshire": ("Bracknell Forest", "Reading", "Slough", "West Berkshire",
                  "Windsor and Maidenhead", "Wokingham"),
    "Buckinghamshire": ("Buckinghamshire", "Milton Keynes"),
    "Cambridgeshire": ("Cambridge", "East Cambridgeshire", "Fenland", "Huntingdonshire",
                       "Peterborough", "South Cambridgeshire"),
    "Cheshire": ("Cheshire East", "Cheshire West and Chester", "Halton", "Warrington"),
    "Cleveland": ("Hartlepool", "Middlesbrough", "Redcar and Cleveland",
                  "Stockton-on-Tees"),
    "Cornwall": ("Cornwall", "Isles of Scilly"),
    "Cumbria": ("Cumberland", "Westmorland and Furness"),
    "Derbyshire": ("Amber Valley", "Bolsover", "Chesterfield", "Derby", "Derbyshire Dales",
                   "Erewash", "High Peak", "North East Derbyshire", "South Derbyshire"),
    "Devon": ("East Devon", "Exeter", "Mid Devon", "North Devon", "Plymouth", "South Hams",
              "Teignbridge", "Torbay", "Torridge", "West Devon"),
    "Dorset": ("Bournemouth, Christchurch and Poole", "Dorset"),
    "Durham": ("County Durham", "Darlington"),
    "East Sussex": ("Brighton and Hove", "Eastbourne", "Hastings", "Lewes", "Rother",
                    "Wealden"),
    "East Yorkshire (combined)": ("East Riding of Yorkshire", "Kingston upon Hull"),
    "Essex": ("Basildon", "Braintree", "Brentwood", "Castle Point", "Chelmsford",
              "Colchester", "Epping Forest", "Harlow", "Maldon", "Rochford",
              "Southend-on-Sea", "Tendring", "Thurrock", "Uttlesford"),
    "Gloucestershire": ("Cheltenham", "Cotswold", "Forest of Dean", "Gloucester", "Stroud",
                        "Tewkesbury"),
    "Greater Manchester": ("Bolton", "Bury", "Manchester", "Oldham", "Rochdale", "Salford",
                           "Stockport", "Tameside", "Trafford", "Wigan"),
    "Hampshire": ("Basingstoke and Deane", "East Hampshire", "Eastleigh", "Fareham",
                  "Gosport", "Hart", "Havant", "New Forest", "Portsmouth", "Rushmoor",
                  "Southampton", "Test Valley", "Winchester"),
    "Hereford UA and Worcester": ("Bromsgrove", "Herefordshire", "Malvern Hills",
                                  "Redditch", "Worcester", "Wychavon", "Wyre Forest"),
    "Hertfordshire": ("Broxbourne", "Dacorum", "East Hertfordshire", "Hertsmere",
                      "North Hertfordshire", "St Albans", "Stevenage", "Three Rivers",
                      "Watford", "Welwyn Hatfield"),
    "Inner London": ("Camden", "City of London", "Hackney", "Hammersmith and Fulham",
                     "Haringey", "Islington", "Kensington and Chelsea", "Lambeth",
                     "Lewisham", "Newham", "Southwark", "Tower Hamlets", "Wandsworth",
                     "Westminster"),
    "Isle of Wight": ("Isle of Wight",),
    "Kent": ("Ashford", "Canterbury", "Dartford", "Dover", "Folkestone and Hythe",
             "Gravesham", "Maidstone", "Medway", "Sevenoaks", "Swale", "Thanet",
             "Tonbridge and Malling", "Tunbridge Wells"),
    "Lancashire": ("Blackburn with Darwen", "Blackpool", "Burnley", "Chorley", "Fylde",
                   "Hyndburn", "Lancaster", "Pendle", "Preston", "Ribble Valley",
                   "Rossendale", "South Ribble", "West Lancashire", "Wyre"),
    "Leicestershire": ("Blaby", "Charnwood", "Harborough", "Hinckley and Bosworth",
                       "Leicester", "Melton", "North West Leicestershire",
                       "Oadby and Wigston", "Rutland"),
    "Lincolnshire": ("Boston", "East Lindsey", "Lincoln", "North East Lincolnshire",
                     "North Kesteven", "North Lincolnshire", "South Holland",
                     "South Kesteven", "West Lindsey"),
    "Merseyside": ("Knowsley", "Liverpool", "Sefton", "St. Helens", "Wirral"),
    "Norfolk": ("Breckland", "Broadland", "Great Yarmouth",
                "King's Lynn and West Norfolk", "North Norfolk", "Norwich",
                "South Norfolk"),
    "North Yorkshire": ("North Yorkshire", "York"),
    "Northamptonshire": ("North Northamptonshire", "West Northamptonshire"),
    "Northumberland": ("Northumberland",),
    "Nottinghamshire": ("Ashfield", "Bassetlaw", "Broxtowe", "Gedling", "Mansfield",
                        "Newark and Sherwood", "Nottingham", "Rushcliffe"),
    "Outer London": ("Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
                     "Croydon", "Ealing", "Enfield", "Greenwich", "Harrow", "Havering",
                     "Hillingdon", "Hounslow", "Kingston upon Thames", "Merton",
                     "Redbridge", "Richmond upon Thames", "Sutton", "Waltham Forest"),
    "Oxfordshire": ("Cherwell", "Oxford", "South Oxfordshire", "Vale of White Horse",
                    "West Oxfordshire"),
    "Shropshire": ("Shropshire", "Telford and Wrekin"),
    "Somerset": ("Somerset",),
    "South Yorkshire": ("Barnsley", "Doncaster", "Rotherham", "Sheffield"),
    "Staffordshire": ("Cannock Chase", "East Staffordshire", "Lichfield",
                      "Newcastle-under-Lyme", "South Staffordshire", "Stafford",
                      "Staffordshire Moorlands", "Stoke-on-Trent", "Tamworth"),
    "Suffolk": ("Babergh", "East Suffolk", "Ipswich", "Mid Suffolk", "West Suffolk"),
    "Surrey": ("Elmbridge", "Epsom and Ewell", "Guildford", "Mole Valley",
               "Reigate and Banstead", "Runnymede", "Spelthorne", "Surrey Heath",
               "Tandridge", "Waverley", "Woking"),
    "Tyne & Wear": ("Gateshead", "Newcastle upon Tyne", "North Tyneside", "South Tyneside",
                    "Sunderland"),
    "Warwickshire": ("North Warwickshire", "Nuneaton and Bedworth", "Rugby",
                     "Stratford-on-Avon", "Warwick"),
    "West Midlands": ("Birmingham", "Coventry", "Dudley", "Sandwell", "Solihull",
                      "Walsall", "Wolverhampton"),
    "West Sussex": ("Adur", "Arun", "Chichester", "Crawley", "Horsham", "Mid Sussex",
                    "Worthing"),
    "West Yorkshire": ("Bradford", "Calderdale", "Kirklees", "Leeds", "Wakefield"),
    "Wiltshire": ("Swindon", "Wiltshire"),
}

# Greenwich is in Outer London above and in ONS's Inner London in some publications; ONS's
# own statistical Inner/Outer split places it in Outer. Left there deliberately.


def lad_to_county():
    """District name -> ECC county, with the table validated against itself."""
    out = {}
    for county, lads in ECC_COUNTY.items():
        for lad in lads:
            if lad in out:
                raise SystemExit(f"district in two ECC counties: {lad}")
            out[lad] = county
    return out


def geography():
    """(frame of OA21CD -> county, settlement; diagnostics). England only."""
    for path in (OA_RUC, OA_LAD):
        if not os.path.exists(path):
            sys.exit(f"missing {path}\n  see sources/uk_ecc.md for the fetch")
    ruc = pd.read_csv(OA_RUC)
    lad = pd.read_csv(OA_LAD)
    msoa = pd.read_csv(OA_MSOA)
    oa = lad.merge(ruc, on="OA21CD", how="outer", indicator=True)
    oa = oa.merge(msoa, on="OA21CD", how="left")

    diag = {"oa_rows": len(oa),
            "oa_only_in_lad": int((oa["_merge"] == "left_only").sum()),
            "oa_only_in_ruc": int((oa["_merge"] == "right_only").sum())}
    oa = oa[oa["_merge"] == "both"].drop(columns="_merge")

    eng = oa[oa["LAD23CD"].str.startswith("E")].copy()
    diag["oa_england"] = len(eng)
    diag["oa_wales"] = len(oa) - len(eng)

    mapping = lad_to_county()
    eng["county"] = eng["LAD23NM"].map(mapping)
    missing = sorted(eng.loc[eng["county"].isna(), "LAD23NM"].unique())
    diag["districts_unmapped"] = missing
    diag["districts_mapped"] = int(eng["LAD23NM"].nunique() - len(missing))

    known = set(eng["LAD23NM"].unique())
    diag["table_districts_not_in_lookup"] = sorted(set(mapping) - known)

    eng["settlement"] = eng["RUC21CD"].str.startswith(RURAL_PREFIX).map(
        {True: "rural", False: "urban"})
    return eng, diag


def check_geography(eng, diag):
    fails = 0
    print("geography: 2021 Output Areas -> ECC county x settlement")
    for k in ("oa_rows", "oa_only_in_lad", "oa_only_in_ruc", "oa_england", "oa_wales",
              "districts_mapped"):
        print(f"  {k:32s} {diag[k]:,}")
    if diag["oa_only_in_lad"] or diag["oa_only_in_ruc"]:
        print("  ! the two ONS lookups disagree about which Output Areas exist")
        fails += 1
    if diag["districts_unmapped"]:
        print(f"  ! {len(diag['districts_unmapped'])} districts have no ECC county:")
        for d in diag["districts_unmapped"]:
            print(f"      {d}")
        fails += 1
    if diag["table_districts_not_in_lookup"]:
        print("  ! districts named in ECC_COUNTY that ONS does not have "
              "(a typo, or a boundary change):")
        for d in diag["table_districts_not_in_lookup"]:
            print(f"      {d}")
        fails += 1
    if not fails:
        counts = eng.groupby(["county", "settlement"]).size().unstack(fill_value=0)
        print(f"  counties covered                 {len(counts)}")
        print(f"  cells with no urban OAs          {(counts.get('urban', 0) == 0).sum()}")
        print(f"  cells with no rural OAs          {(counts.get('rural', 0) == 0).sum()}")
        print(f"  Output Areas rural               {100 * (eng.settlement == 'rural').mean():.1f}%")
    return fails


def christians_by_oa(eng):
    """Every English Output Area with its county, settlement, MSOA and Christian count.

    Read from uk.csv's TS030 rather than uk_ew_allocated.csv: allocation left `Christian`
    exactly as the census published it (`derivation=exact_single_child`), so the two agree
    to the last person and this file is a tenth the size.
    """
    uk = pd.read_csv(os.path.join(NORM, "uk.csv"),
                     usecols=["geo_id", "geo_level", "source_category", "count",
                              "source_id"],
                     dtype={"geo_id": str}, low_memory=False)
    chr_ = uk[(uk["source_id"] == "uk_ew_census_2021")
              & (uk["geo_level"] == "output_area")
              & (uk["source_category"] == "Christian")]
    oa = eng[["OA21CD", "county", "settlement", "MSOA21CD"]].merge(
        chr_[["geo_id", "count"]], left_on="OA21CD", right_on="geo_id", how="left")
    # a missing row is a zero, not unknown (sources/uk.md §9)
    oa["christians"] = oa["count"].fillna(0.0)
    return oa.drop(columns=["geo_id", "count"])


def shape(ecc, churches):
    """Placement weight per cell per leg = churches now x mean congregation in 2005.

    **NEITHER SOURCE CAN DO THIS ALONE, AND THE REASON IS A BUG THIS FILE ONCE HAD.**
    Placing denominations by the church census's attendance TOTALS drew a Merseyside that
    was 72% Anglican and 16% Catholic. Its response rate is not national: measured against
    the Church of England's own register it runs 26% in Norfolk and 91% in Merseyside,
    because ten dioceses sent spreadsheets and the rest answered a postal survey. A total
    carries that; a mean congregation size does not, because losing half a county's chapels
    changes how many you saw and not how big they were.

    So the count of churches comes from a current register (sources/uk_churches.py) and only
    the congregation size comes from 2005. The ladder is on the SIZE, which is the only part
    that can be thin:

        cell     that county's urban or rural mean for that leg, if it rests on
                 MIN_CHURCHES or more churches
        county   that county's mean for the leg
        England  the leg's national mean

    Returns (county, settlement, leg, weight, rung).
    """
    cty = ecc[ecc["geo_level"] == "county"][["geo_id", "leg", "count", "churches"]]
    cty = cty.rename(columns={"geo_id": "county", "count": "county_mean",
                              "churches": "county_n"})
    cell = ecc[ecc["geo_level"] == "county_settlement"].copy()
    cell[["county", "settlement"]] = cell["geo_id"].str.split("|", expand=True)
    cell = cell[["county", "settlement", "leg", "count", "churches"]].rename(
        columns={"count": "cell_mean", "churches": "cell_n"})
    nat = (ecc[ecc["geo_level"] == "county"]
           .assign(att=lambda d: d["count"] * d["churches"])
           .groupby("leg").apply(lambda g: g["att"].sum() / g["churches"].sum(),
                                 include_groups=False)
           .rename("national_mean").reset_index())

    g = churches.merge(cell, on=["county", "settlement", "leg"], how="left")
    g = g.merge(cty, on=["county", "leg"], how="left").merge(nat, on="leg", how="left")

    use_cell = g["cell_n"].fillna(0) >= MIN_CHURCHES
    use_cty = (~use_cell) & (g["county_n"].fillna(0) >= MIN_CHURCHES)
    g["mean"] = g["national_mean"]
    g.loc[use_cty, "mean"] = g.loc[use_cty, "county_mean"]
    g.loc[use_cell, "mean"] = g.loc[use_cell, "cell_mean"]
    g["rung"] = "England"
    g.loc[use_cty, "rung"] = "county"
    g.loc[use_cell, "rung"] = "cell"

    g["weight"] = g["n_churches"] * g["mean"]
    return g[["county", "settlement", "leg", "weight", "rung", "n_churches", "mean"]]


def propensities(oa, shp, orthodox):
    """One frame of Output Areas x legs, each cell a RATE rather than a count.

    **THE LEGS DO NOT SHARE A PLACEMENT GEOGRAPHY, AND THEY should not have to.** The five
    register legs are placed at county x settlement, 92 cells, because that is as fine as a
    2005 congregation size goes. Orthodoxy is placed at MSOA, 6,856 units, because its proxy
    is a census variable and the census publishes it that fine (sources/uk_orthodox.py).
    Expressing both as a rate per Christian rather than as a count puts them on one axis:

        propensity(OA, leg) = that leg's placement weight in whatever area places it
                              / the Christians living in that same area

    so a leg concentrated in a populous area is not thereby made to look larger, and two
    legs measured over different areas can still be compared inside one Output Area. The
    rake then scales each leg to its national line. `unsplit` is flat by construction.
    """
    cells = oa.groupby(["county", "settlement"], as_index=False)["christians"].sum()
    cells = cells.rename(columns={"christians": "cell_christians"})
    reg = shp.pivot_table(index=["county", "settlement"], columns="leg",
                          values="weight", aggfunc="sum").fillna(0.0).reset_index()
    reg = reg.merge(cells, on=["county", "settlement"], how="left")
    reg_legs = [c for c in reg.columns
                if c not in ("county", "settlement", "cell_christians")]
    for leg in reg_legs:
        reg[leg] = reg[leg] / reg["cell_christians"].replace(0, pd.NA)
    out = oa.merge(reg.drop(columns="cell_christians"),
                   on=["county", "settlement"], how="left")

    if orthodox is not None:
        msoa_chr = oa.groupby("MSOA21CD", as_index=False)["christians"].sum()
        msoa_chr = msoa_chr.rename(columns={"christians": "msoa_christians"})
        o = orthodox.merge(msoa_chr, on="MSOA21CD", how="right")
        o["orthodox"] = (o["weight"].fillna(0.0)
                         / o["msoa_christians"].replace(0, pd.NA))
        out = out.merge(o[["MSOA21CD", "orthodox"]], on="MSOA21CD", how="left")
        reg_legs = reg_legs + ["orthodox"]

    out["unsplit"] = 1.0
    legs = reg_legs + ["unsplit"]
    out[legs] = out[legs].fillna(0.0)
    return out, legs


def rake(oa, legs, anchor, rounds=300):
    """Per-Output-Area leg weights whose England-wide totals match the anchor.

    A per-area normalisation does not preserve national shares, so the per-leg correction k
    is solved for rather than assumed: iterate until the population-weighted national result
    reproduces the British Election Study. Same fitting br_rescale.py does per município,
    run once over England.

    **`unsplit` IS A LEG AND IT HAS NO GEOGRAPHY ON PURPOSE.** Pentecostal, New church and
    `other` are 5.8% of England's Christians and no source available places them --
    OpenStreetMap has 355 Pentecostal churches in all of England, and `newchurch` has no OSM
    tag at all. Ethnic group was considered as a proxy the way country of birth was used for
    Orthodoxy and rejected (Anita, 2026-09-08): Black African in England is heavily Anglican
    and Catholic as well as Pentecostal, so it would place "where the Black-majority
    congregations are" rather than where the Pentecostals are. So they are carried as one
    flat leg proportional to Christians, neither deleted nor pushed into a denomination they
    are not, and the panel says so.
    """
    target = pd.Series({l: anchor[l] for l in legs}, dtype=float)
    target = target / target.sum()
    k = pd.Series(1.0, index=legs)
    err = float("nan")
    w = None
    for _ in range(rounds):
        raw = oa[legs].mul(k, axis=1)
        w = raw.div(raw.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0.0)
        got = w.mul(oa["christians"], axis=0).sum()
        got = got / got.sum()
        err = float((got - target).abs().max())
        if err < 1e-10:
            break
        k = k * (target / got.replace(0, pd.NA)).fillna(1.0)
    return w, k, err


# Leg -> the category string emitted, which taxonomy/uk2021.py resolves to a node.
#
# **THERE IS NO ROW FOR THE UNPLACED 8%, AND THAT IS THE POINT.** Pentecostal, New church,
# Orthodox and `other` have an anchor but no geography yet, so instead of inventing a
# category for them this file writes back a REDUCED `Christian` row -- the census's own
# category, its own node, still `measured`, just smaller. Those people stay exactly where
# the census put them and are described exactly as the census described them, which is the
# honest reading of "we have not worked out which denomination these are yet". When the
# census-proxy placement lands the remainder shrinks again, and nothing else has to change.
CATEGORY = {
    "anglican": "Christian: Anglican",
    "catholic": "Christian: Roman Catholic",
    "methodist": "Christian: Methodist",
    "baptist": "Christian: Baptist",
    "reformed": "Christian: Reformed",
    "orthodox": "Christian: Orthodox",
}
# Orthodoxy is the one leg placed by a proxy rather than by a count of churches, so it is
# `modelled` where the others are `derived` (spec §7). sources/uk_orthodox.py has the case.
TIER = {"orthodox": "modelled"}
TIER_DEFAULT = "derived"
NOTE_BY_LEG = {
    "orthodox": ("level=leaf; derivation=proxy; structure_geo=msoa; "
                 "structure=uk_en_orthodox_proxy_2021; anchor=uk_en_bes_w21; "
                 "parent_column=Christian"),
}
REMAINDER = "Christian"

OUT_COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
               "basis", "year", "source_id", "tier", "note"]
SOURCE_ID = "uk_ew_census_2021"      # the magnitudes are still the census's
BASIS = "self_id"


def emit(oa, w, path=OUT):
    """Per-Output-Area rows: the placed denominations plus the reduced `Christian` remainder."""
    legs = [c for c in CATEGORY if c in w.columns]
    default_note = ("level=leaf; derivation=split; structure_geo=county_settlement; "
                    "structure=uk_en_churches_2026 x uk_en_ecc_2005; "
                    "anchor=uk_en_bes_w21; parent_column=Christian")
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        out.writeheader()
        placed_total = w[legs].sum(axis=1)
        for leg in legs:
            people = (oa["christians"].to_numpy() * w[leg].to_numpy()).round(4)
            keep = people > 0
            for gid, val in zip(oa.loc[keep, "OA21CD"], people[keep]):
                out.writerow({"geo_id": gid, "geo_level": "output_area", "geo_name": gid,
                              "source_category": CATEGORY[leg], "count": val,
                              "basis": BASIS, "year": 2021, "source_id": SOURCE_ID,
                              "tier": TIER.get(leg, TIER_DEFAULT),
                              "note": NOTE_BY_LEG.get(leg, default_note)})
                n += 1
        rest = (oa["christians"].to_numpy() * (1.0 - placed_total.to_numpy())).round(4)
        keep = rest > 0
        for gid, val in zip(oa.loc[keep, "OA21CD"], rest[keep]):
            out.writerow({"geo_id": gid, "geo_level": "output_area", "geo_name": gid,
                          "source_category": REMAINDER, "count": val,
                          "basis": BASIS, "year": 2021, "source_id": SOURCE_ID,
                          "tier": "measured",
                          "note": "level=leaf; cat=Christian; derivation=exact_single_child;"
                                  " parent_column=Christian; denomination not placed"})
            n += 1
    return n


def load_inputs():
    ecc = pd.read_csv(os.path.join(NORM, "uk_ecc.csv"), dtype={"geo_id": str})
    ecc = ecc.rename(columns={"source_category": "leg"})
    ecc["churches"] = ecc["note"].str.extract(r"over (\d+) churches").astype(float)

    ch = pd.read_csv(os.path.join(NORM, "uk_churches.csv"), dtype={"geo_id": str})
    ch = ch.rename(columns={"source_category": "leg", "count": "n_churches"})
    ch[["county", "settlement"]] = ch["geo_id"].str.split("|", expand=True)
    ch = ch[["county", "settlement", "leg", "n_churches"]]

    orth = None
    p = os.path.join(NORM, "uk_orthodox.csv")
    if os.path.exists(p):
        orth = pd.read_csv(p, dtype={"geo_id": str})
        orth = orth.rename(columns={"geo_id": "MSOA21CD", "count": "weight"})
        orth = orth[["MSOA21CD", "weight"]]

    bes = pd.read_csv(os.path.join(NORM, "uk_bes.csv"))
    anchor = dict(zip(bes["source_category"], bes["count"].astype(float)))
    placed = sorted(ch["leg"].unique()) + (["orthodox"] if orth is not None else [])
    anchor["unsplit"] = sum(v for k, v in anchor.items() if k not in placed)
    return ecc, ch, orth, anchor, placed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    eng, diag = geography()
    if check_geography(eng, diag):
        return 1

    ecc, ch, orth, anchor, placed = load_inputs()
    shp = shape(ecc, ch)
    oa = christians_by_oa(eng)
    oa, legs = propensities(oa, shp, orth)
    w, k, err = rake(oa, legs, anchor)

    total = oa["christians"].sum()
    print(f"\nEngland's Christians: {total:,.0f} over {len(oa):,} Output Areas")
    print(f"placed legs: {', '.join(placed)}")
    tot_anchor = sum(anchor[l] for l in legs)
    print(f"unplaced, carried flat as `unsplit`: "
          f"{100*anchor['unsplit']/tot_anchor:.1f}%")
    print(f"rake converged to {err:.1e}")

    print("\n  leg          churches   mean cong.   BES anchor   ladder cell/county/England")
    for leg in sorted(legs, key=lambda l: -anchor[l]):
        s = shp[shp["leg"] == leg]
        rungs = s["rung"].value_counts()
        cong = (s["weight"].sum() / s["n_churches"].sum()) if len(s) else float("nan")
        print(f"  {leg:12s} {s['n_churches'].sum():9,.0f} {cong:11.0f} "
              f"{100*anchor[leg]/tot_anchor:10.1f}%   "
              f"{rungs.get('cell', 0):5d}/{rungs.get('county', 0):6d}/{rungs.get('England', 0):7d}")

    people = w.mul(oa["christians"], axis=0)
    people["county"] = oa["county"].to_numpy()
    by_cty = people.groupby("county").sum(numeric_only=True)
    by_cty = 100 * by_cty.div(by_cty.sum(axis=1), axis=0)
    show = [c for c in ("anglican", "catholic", "methodist", "baptist", "reformed",
                        "orthodox", "unsplit") if c in by_cty.columns]
    print("\n  share of each county's Christians")
    order = ["Merseyside", "Lancashire", "Greater Manchester", "Tyne & Wear", "Durham",
             "Inner London", "Outer London", "West Midlands", "Cornwall", "Devon",
             "Norfolk", "Lincolnshire", "North Yorkshire", "Avon", "Cambridgeshire"]
    print(by_cty.loc[[c for c in order if c in by_cty.index], show].round(1).to_string())

    if args.report:
        return 0

    n = emit(oa, w)
    written = pd.read_csv(OUT, usecols=["source_category", "count"])
    drawn = written["count"].sum()
    print(f"\nwrote {n:,} rows -> {OUT}")
    print(f"  people written {drawn:,.0f} against {total:,.0f} census Christians "
          f"({drawn - total:+,.0f})")
    if abs(drawn - total) > 1.0:
        print("  ! the split does not conserve England's Christians")
        return 1
    print("\n  England as drawn")
    got = written.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for cat, v in got.items():
        print(f"    {cat:28s} {v:12,.0f}  {100*v/drawn:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
