# Venezuela. Censo 2011: each indigenous person's pueblo per parroquia (sources/ve_censo.py), with
# the share who speak their pueblo's language MODELLED from INE's published state shares (ask 007,
# Anita 2026-10-05: option A). Placed on Kontur hexes keyed to parroquias (sources/ve_geo.py).
# sources/ve.md is the record.
from _shared import *  # noqa: F401,F403

SPANISH = "indoeuropean.romance.spanish"

# INE, POBLACION-INDIGENA-CENSO-2011.pdf p. 34, "Hablantes de su idioma", 2011: the share of each
# state's indigenous people who speak their pueblo's language. The only eight INE prints.
STATE_SHARE = {"02": 0.752, "03": 0.171, "04": 0.758, "07": 0.873, "10": 0.894, "16": 0.333,
               "19": 0.025, "23": 0.675}
# p. 33: nationally 10.2% speak only their pueblo's language and 54.1% it and Spanish
NATIONAL_SHARE = 0.102 + 0.541


def model(verbose=False):
    """Rows (geo_id, code, node, count, tier) after the model, and a per-state summary.

    Within each of STATE_SHARE's states, the speakers INE's share implies (share x the state's
    indigenous people whose pueblo is known) are spread over the people of pueblos whose language
    is alive, at one rate for the state; the rest of those people, and everyone of a pueblo whose
    language is gone (ve2011.EXTINCT), are drawn as Spanish. The seventeen other states take one
    rate together: the national 64.3% turned into a rate over living-language people the same
    way, nationally. Pueblo not declared (998) is not drawn.
    """
    import ve2011
    df = pd.read_csv(NORM / "ve.csv", dtype={"geo_id": str})
    df["node"] = df["code"].map(ve2011.resolve)
    unmapped = sorted(set(df["code"]) - set(ve2011.CODES))
    if unmapped:
        raise SystemExit(f"ve: census codes with no entry in ve2011.CODES: {unmapped}")
    df["state"] = df["geo_id"].str[2:4]
    ind = df[df["code"] < 1000]
    known = ind[ind["code"] != 998]
    living = known[known["node"] != SPANISH]
    k_s = known.groupby("state")["count"].sum()
    l_s = living.groupby("state")["count"].sum()

    rate, summary = {}, []
    for s, share in STATE_SHARE.items():
        target = share * k_s[s]
        rate[s] = min(1.0, target / l_s[s])
        summary.append((s, share, int(k_s[s]), int(l_s[s]), target, rate[s]))
    # The seventeen other states: INE's national share, turned into a rate over living-language
    # people the same way (national speakers over national living-language people, 0.708). A
    # residual (national speakers less the eight states') comes out negative, -2,330: the eight
    # shares and the national one do not add up exactly, so it cannot be used.
    rest = [s for s in k_s.index if s not in STATE_SHARE]
    l_rest = l_s.reindex(rest).fillna(0).sum()
    r_rest = NATIONAL_SHARE * k_s.sum() / l_s.sum()
    target_rest = r_rest * l_rest
    for s in rest:
        rate[s] = r_rest
    summary.append(("rest", None, int(k_s.reindex(rest).sum()), int(l_rest), target_rest, r_rest))
    if verbose:
        for s, share, k, lv, t, r in summary:
            print(f"  {s:>4}  share {'' if share is None else f'{share:.3f}':>5}  known pueblo "
                  f"{k:>7,}  living-language {lv:>7,}  speakers {t:>9,.0f}  rate {r:.3f}")

    out = []
    for r in df.itertuples():
        if r.node is None or r.code == 998:
            continue
        if r.code >= 1000:
            out.append((r.geo_id, r.code, SPANISH, float(r.count), "derived"))
        elif r.node == SPANISH:
            out.append((r.geo_id, r.code, SPANISH, float(r.count), "modelled"))
        else:
            f = rate[r.state]
            out.append((r.geo_id, r.code, r.node, r.count * f, "modelled"))
            if f < 1:
                out.append((r.geo_id, r.code, SPANISH, r.count * (1 - f), "modelled"))
    rows = pd.DataFrame(out, columns=["geo_id", "code", "node", "count", "tier"])
    drawn = rows["count"].sum()
    expect = df["count"].sum() - df.loc[df["code"] == 998, "count"].sum()
    if abs(drawn - expect) > 0.5:
        raise SystemExit(f"ve: model draws {drawn:,.1f} people, expected {expect:,}")
    return rows, summary


def _counts():
    rows, _ = model()
    lut = pd.read_csv(GEO / "ve" / "ve_units.csv", dtype=str)
    rows["unit"] = rows["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if rows["unit"].isna().any():
        raise SystemExit(f"ve: {rows['unit'].isna().sum()} rows with no unit; re-run "
                         "sources/ve_geo.py")
    rows = rows.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return _immigrants(rows, lut)


# 2026-10-05 (session edd42a8c-lats, sources/ve.md "Immigrant languages"): the foreign-born by
# country of birth per parroquia (sources/ve_immig.py) on their origin's languages (origin_mix),
# retained at France's TeO2 rate (sources/latam_immig.py), taken out of the `derived` Spanish
# remainder (the born-abroad and non-indigenous rows), never the modelled indigenous rows.
def _immigrants(rows, lut):
    import sys
    sys.path.insert(0, str(ROOT / "sources"))
    import latam_immig
    if latam_immig.active():          # an origin's home mix for another country's build
        return rows
    imm = pd.read_csv(NORM / "ve_immig.csv", dtype={"geo_id": str}, keep_default_na=False)
    imm["unit"] = imm["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if imm["unit"].isna().any():
        raise SystemExit("ve: immigrant rows with no unit")
    imm = imm.groupby(["unit", "iso"], as_index=False)["count"].sum()
    lang = latam_immig.immigrant_languages(imm, SPANISH, "ve")
    out, _ = latam_immig.fold_into(rows, lang, SPANISH, from_tier="derived")
    return out


ENTRY = dict(
    name="Venezuela",
    source="Censo Nacional de Población y Vivienda 2011 (INE), tabulated on INE's REDATAM "
           "server, with INE's published shares of indigenous people who speak their people's "
           "language",
    how="census, 2011, indigenous people by people, speakers modelled from published state "
        "shares; people born abroad drawn by their birth country's languages; everyone else "
        "drawn as Spanish",
    parts=[
        dict(covers="Indigenous languages",
             source="2011 census, indigenous people by people; speakers modelled from INE's "
                    "published state shares",
             people=466_498),
        dict(covers="People born abroad, languages other than Spanish",
             source="2011 census, country of birth, drawn on that country's languages; a "
                    "quarter to a third moved to Spanish by France's TeO2 survey",
             people=103_967),
        dict(covers="Everyone else", source="drawn as Spanish", rest=True),
    ],
    grain="1,111 parroquias and merged municipios, 24,500 people on average",
    gap="15,236 indigenous people who did not say which people they belong to",
    view=[-73.4, 0.6, -59.7, 12.3],
    counts=_counts,
    mappings=["ve2011"],
    place=GEO / "ve" / "ve_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2011 census asked each indigenous person which people they belong to and which "
        "languages they speak, but INE has published the language answers only as national "
        "and state percentages. So these dots are a model. Each person is drawn where the census "
        "counted them, on the language of their people, and in each state only as many are "
        "drawn as speakers as INE's figure for that state allows (75% in Amazonas, 2.5% in "
        "Sucre). Within a state every people gets the same rate, so the map cannot show that "
        "one people keeps its language better than its neighbours. The rest, and peoples "
        "whose language is no longer spoken such as the Añú, are drawn as Spanish speakers. "
        "People born abroad (1.2 million, most of them Colombians) are drawn by the languages "
        "of their birth country. Since no Venezuelan survey asks which language immigrants "
        "speak at home, a quarter to a third of those from a country with another language "
        "are drawn as Spanish speakers, the share of immigrants in France who speak only "
        "French with their children."),
)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    rows, _ = model(verbose=True)
    nat = rows.groupby(["node", "tier"])["count"].sum().sort_values(ascending=False)
    print(nat.head(40).round(0).to_string())
