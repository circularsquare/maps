# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _fr_place_weight(place):
    """France's 34,476 communes, weighted by commune population.

    The same weighter Greece uses, and it earns its keep harder here than anywhere else on
    the map. France's counting geography is 21 anciennes régions at 3.10M people each — the
    coarsest here — so 1,642 communes per counted unit is all that stands between this and
    twenty-one flat blobs.

    **And it is a population weight, not a religion one, which in France costs something
    nameable.** Île-de-France is drawn as one composition over 12.2M people, so its 11.4%
    Muslim share spreads across the whole region in proportion to where anyone lives. The
    real geography — Seine-Saint-Denis against Yvelines — is invisible, and no source on this
    map can supply it. sources/fr.md says so in the terms §8.2 asks for.
    """
    if "pop" not in place.columns:
        print("  !! fr_lau.gpkg has no `pop` column — run sources/fr_geo.py")
        return None
    if not all(c in place.columns for c in ("french", "foreign", "unit")):
        print("  !! fr_lau.gpkg lacks the nationality columns — run "
              "sources/fr_geo.py --fetch then sources/fr_geo.py")
        return _GrLauWeighter(place)
    # §9as, and France is where it does the most work: 26 régions of 2.6M people is the
    # coarsest counting geography on this map, so the placement weight is most of what makes
    # the country look like a country. INSEE's RP 2021 TD_NAT1 gives French and foreign
    # nationals per commune at the same code GISCO carries. The five overseas régions are
    # drawn from Pew as whole units and are largely outside TD_NAT1; since 2026-09-14 their
    # rows in fr_lau.gpkg are Kontur 400 m hexes rather than communes (fr_geo.py `_dom_hexes`),
    # carrying the hex population as both `pop` and `french`, so they are placed by where
    # people live. See `_ItWeighter`.
    from fr2024 import resolve
    return _ItWeighter(place, _foreign_share("fr", resolve, "nuts3"),
                       citizen_col="french", foreign_col="foreign",
                       citizen_label="French", place_label="commune")


def _fr_counts():
    """France at NUTS 2: 21 anciennes régions, from two halves of one census table.

    France has never asked about religion in a census and is barred by law from doing so.
    The two populations are Greece's (§9z), both larger:

      * **French citizens, 60.3M.** ESS rounds 5-11 pooled and restricted to
        `ctzcntr = Yes` — 12,678 respondents over 21 régions, about 600 each, the same
        order as Greece's regions and Russia's federal subjects.
      * **Foreign residents, 4.74M.** Eurostat's 2021 census table `cens_21ctz_r3`, 200
        named citizenships at NUTS 3 covering **100.00%** of the foreign population,
        crossed with Pew's composition for each origin country.

    **Both come out of the same census table**, which publishes `NAT` and `FOR` next to the
    named citizenships, so the halves partition by construction rather than by
    reconciliation.

      * **The five overseas régions, 2.22M.** Pew's own 2020 country estimates, one per
        territory, at the one geography where a Pew country row is also a NUTS 2 unit — so
        nothing is downscaled and §14.3's resolution rule holds by identity. Basis
        `estimate`, which is §3.1's own word for a Pew figure. Added 2026-09-07.

    **NO CELL IS AUTHORED, which is what separates this from Greece.** Greece needed two —
    the Thracian minority ESS is blind to, and Mount Athos. France needs none: the survey
    finds Alsace's Protestants, Île-de-France's Muslims and Jews and the Mediterranean's
    Muslims unaided, and where it is weak (the banlieues) no published régional figure
    exists to substitute. Inventing one is spec §14.4's first prohibition.

    **99.30% of France is drawn on 26 units, and the missing 0.51% is Corsica.** ESS's frame
    is metropolitan and excludes Corsica and the overseas régions; Pew covers the five
    overseas territories and not Corsica, which is part of metropolitan France and has no
    ISO code of its own. Nothing is borrowed for it — a metropolitan mixture would draw the
    national average and say nothing true — and §6.12's coverage wash says which.
    """
    from fr2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "fr.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts3"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"fr.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "fr_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts3"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey is not a count of anybody and a nationality model is not either, so nothing
    # here is `measured` and §7's inferred-dots mode empties the country completely.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "fr": dict(
        name="France",
        name_in="France",
        source="ESS rounds 5–11 (citizens) + Eurostat census 2021 × Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        view=[-5.4, 42.2, 8.4, 51.2],
        note_public=(
            "**France has never asked, and is forbidden by law from asking.** Collecting "
            "religion in official statistics is tightly restricted here, so there is no "
            "census figure to draw and there never will be. What there is instead is a "
            "survey that asks the question directly: the European Social Survey has put it "
            "to French residents in seven rounds since 2010, and pooling them gives 12,678 "
            "citizens with a région attached. The 4.9 million people living in France on "
            "another country's passport are counted separately, from the 2021 census's own "
            "record of who lives where and where they are from. "
            "**Half of France reports no religion — 48.8%, the largest answer on the map "
            "here and larger than Catholicism.** That share has not moved in fourteen "
            "years: it was 52.6% in 2010 and 53.5% in 2023–24, with no trend in between. "
            "What has moved is what the other half is. Catholicism is **38.0%** and falling "
            "about a point every five years; Islam is **8.1%** and rising. "
            "**Alsace is the exception to everything and the survey finds it unaided.** "
            "10.4% Protestant against 2.2% nationally — the Lutheran and Reformed churches "
            "of the one part of France where the 1801 Concordat never lapsed, where the "
            "state still pays clergy and religion is taught in public schools. It is also "
            "10.0% Muslim, so the most Protestant région in France is very nearly its "
            "second most Muslim one. "
            "**Islam's geography is the cities and the industrial north-east.** "
            "Île-de-France **16.4%**, Provence-Alpes-Côte d'Azur **11.6%**, Alsace 10.0%, "
            "Franche-Comté 9.4% and Rhône-Alpes 8.9% — Paris, Marseille, Strasbourg, "
            "Sochaux and Lyon, which is to say where the car plants and the ports were — "
            "against **1.8% in Poitou-Charentes** and 2.4% in Bretagne. "
            "**And the déchristianisé west and centre are still there.** Poitou-Charentes "
            "is **63.8%** no religion, Centre-Val de Loire 57.9% and Picardie 58.0%, "
            "against Alsace at 38.0% and Lorraine at 42.7%. That is roughly the map "
            "Gabriel Le Bras and Fernand Boulard drew from Mass attendance in the 1940s and "
            "1950s, sixty years and one collapse in practice later. "
            "**Jews are 0.53% of France and 1.7% of Île-de-France**, the largest Jewish "
            "population in Europe and the third largest anywhere, and its concentration in "
            "and around Paris is the sharpest of any group here after Alsace's "
            "Protestants. "
            "**What the sources cannot do, and the first one is the big one.** The survey "
            "offers denominations, not positions: there is no atheist or agnostic box, so "
            "everyone who says they belong to nothing lands in a single category and "
            "**France — of all countries — has nothing on the `secular` node at all.** The "
            "immigrant half counts people as their country of origin's religion, an upper "
            "bound that cannot see conversion, lapse or anyone who stopped practising after "
            "arriving. France's Buddhists, the largest community in Europe, are drawn at "
            "0.13% against Pew's 0.71%, because a Buddhist with a French passport has "
            "nowhere to go on the form. "
            "**The two halves are counted at different grains, and it matters most in the "
            "Paris region.** Foreign residents are counted by *département* — 94 of them — "
            "so Île-de-France is drawn as eight units rather than one: **Seine-Saint-Denis "
            "comes out 21.5% Muslim against Seine-et-Marne's 13.9%**, where until recently "
            "the whole region showed a single 16.3%. French citizens are still counted by "
            "*région*, because the survey has nothing finer, so **every département inside "
            "one région shares its citizens' composition** and the real spread is wider "
            "than what is drawn. Below the département nothing is measured at all: within "
            "Paris the twenty arrondissements are one number. "
            "**Within a région, though, the dots are not scattered blindly.** People "
            "counted as foreign nationals are placed where foreign nationals actually "
            "live, commune by commune, and French nationals where French nationals live — "
            "and those are very different maps: **64.5% of France's foreign residents live "
            "in cities against 36.1% of its citizens.** So a Muslim or Buddhist dot sits in "
            "a town rather than in the countryside around it. That is a statement about "
            "where a population lives, not about where a religion is: it cannot tell one "
            "commune from its neighbour, and inside a city it says nothing at all. "
            "**And the five overseas régions are a different country on this map.** They are "
            "outside the survey's frame and are drawn instead from Pew's own estimate for "
            "each territory — which works here because each one *is* a single statistical "
            "region, so nothing is being guessed at a finer grain than it was published. "
            "**Mayotte is 98.8% Muslim**, the only French département that is, and it is "
            "absent from the European census table altogether, so its people are on this map "
            "only because a second source counted them. **La Réunion is 4.5% Hindu and 4.2% "
            "Muslim** — the Malbar and the Zarabe, descended from indentured Tamil labourers "
            "and Gujarati traders — which makes it the most religiously mixed part of "
            "France. **Guyane is 9.2% traditional religion**, the Maroon communities of the "
            "Maroni and the Amerindian peoples of the interior. And where metropolitan France "
            "reports 48.8% no religion, Guadeloupe and Martinique report 2.5% and 2.7%. "
            "**The overseas régions are drawn coarser than the mainland, and that is the "
            "instrument and not the place**: Pew publishes seven broad families per "
            "territory, so their Christianity is one undivided colour where metropolitan "
            "France's is split into Catholic, Protestant and Orthodox. Martinique and "
            "Guadeloupe are overwhelmingly Catholic in every account of them; no source "
            "publishes the split, so this map does not draw it. "
            "**0.51% of France is not drawn**: Corsica, which the survey does not sample and "
            "which Pew does not publish separately because it is not a territory."),
        how="survey, 12,678 people; foreign residents by nationality",
        grain="departments for foreign residents (680,000 people); regions for French citizens",
        counts=_fr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "fr" / "fr_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_fr_place_weight,
        note="**The country spec §14.3 was written against, drawn — and the paragraph that "
             "excluded it was excluding a different route.** §14.3 uses France by name for "
             "the move this map does not make: *\"estimating religion there would mean "
             "inventing the magnitude as well as the location, most plausibly from "
             "surnames, origin or nationality\"*. That was true of every route known when it "
             "was written, and ESS is none of them: it asks French residents which religion "
             "they belong to and publishes it by région, so 92.7% of the people here come "
             "from a survey that asked them. spec §14.11 is the amendment; §14.10, decided "
             "the same day, is what permits the other 7.3%. "
             "**`sources.md` §11l closed France on a geography ESS does not use.** Its "
             "people-per-unit table judged the country on the 13 post-2016 régions — 5.26M "
             "each, \"fails badly\" — and ESS carries the **21 anciennes régions**: "
             "NUTS-2010 `FR10`/`FR21`…`FR82` in rounds 5–7, NUTS-2016 `FR10`/`FRB0`…`FRL0` "
             "in rounds 8–11, a clean 1:1 recode. 3.10M per unit rather than 5.26M. **A "
             "scouting note that rejects a country on a number should say where the number "
             "came from**, because this one was assumed rather than read off the source, "
             "and it cost the country a day short of a fortnight. "
             "**Seven rounds pool where Greece got three, and the category list does not "
             "collapse.** Every round offers and uses at least ten denominations, so "
             "nothing blinks out the way `Islam` does in Greek round 11. What pooling costs "
             "instead is time: rounds 5–11 span 2010–2024. The drift is smaller than that "
             "sounds — \"no religion\" is flat across the whole window — but Islam among "
             "citizens rises 4.6% → 6.8%, which is naturalisation rather than conversion, "
             "so the pooled figure understates the present by about a point. "
             "**NO CELL IS AUTHORED, and that is the difference from Greece.** Greece "
             "needed the Thracian minority put back by hand because a Greek-language sample "
             "reaches none of it. France's sample is not blind anywhere comparable: it "
             "finds Alsace's Protestants, Île-de-France's Muslims and Jews and the "
             "Mediterranean's Muslims on its own. Where it is weak — the banlieues — there "
             "is no published régional figure to substitute and inventing one is §14.4's "
             "first prohibition. "
             "**The foreign half is better than Greece's and the check is looser.** "
             "`cens_21ctz_r3`'s 200 named citizenships cover **100.00%** of France's "
             "4.74M foreign residents, against Greece's 99.84%, so the unnamed-remainder "
             "rescale is a no-op. Portugal, Algeria and Morocco are a third of it. The "
             "cross-check per §14.10: Muslims come out at **8.04%** of the drawn population "
             "against Pew's own independent 2020 estimate of **9.10%** — from Eurostat "
             "counts, Pew origin compositions and an ESS survey, none of which is Pew's "
             "France row. Greece landed at 5.08 against 5.12; this is looser, the "
             "categories are eight times larger, and the gap runs the direction a "
             "self-identification survey always runs against a composite estimate. Nothing "
             "is tuned to close it. "
             "**The foreign half IS drawn at the 94 départements as of 2026-09-08, and this "
             "paragraph used to say the opposite.** It declined them on the reasoning this "
             "file gives for Greece — mixing would put the sharper geography on the half "
             "with the weaker claim to it, making the most-inferred part of France also the "
             "most precise-looking part of it. **Italy (§9as) showed that rule has an "
             "unstated premise: that the fine half is the small half.** France's foreign "
             "half is 4.74M people and holds most of what a religion map of this country is "
             "for, and the price of the old rule was the thing §8 named as the single "
             "biggest defect here — that the map could say Île-de-France is 16% Muslim and "
             "nothing whatever about Seine-Saint-Denis. It now says **21.5% against "
             "Seine-et-Marne's 13.9%**, and the eight départements of the Paris region "
             "separate. The basis did not change and neither did the model; Eurostat "
             "publishes at NUTS 3, so §14.3's *never model finer than the source publishes* "
             "is satisfied by the source rather than by an argument. **The citizen half "
             "stays at the région** — ESS has nothing finer — so the drawn spread is "
             "narrower than the real one, and every citizen row's `note` names the région "
             "its composition came from. "
             "**The overseas régions are a THIRD instrument, added 2026-09-07, and the "
             "reason it is allowed is geometric rather than statistical.** The first build "
             "left them undrawn because ESS's frame is metropolitan and borrowing a "
             "metropolitan composition for Martinique would be a false statement rather than "
             "an honest silence. Pew turns out to publish **all five as separate countries** "
             "in the same file `origin_religion.py` was already reading — and **each DOM is "
             "exactly one NUTS 2 unit**, so a Pew country row IS a unit row and nothing is "
             "downscaled at all. §14.3's *never model finer than the source publishes* is "
             "satisfied by identity, which is the cleanest case of it on the map. Basis "
             "`estimate`, §3.1's own word for a Pew figure, and the foreign half is "
             "deliberately not run over these units because Pew's estimate already covers "
             "every resident whatever passport they hold. "
             "**The magnitudes were checked before they were used**: Pew's own populations "
             "land at 0.98-1.01× the 2021 census for the four units the census carries, "
             "which is independent confirmation from a source that is not the census, and "
             "`fr.py` asserts the band rather than reporting it. **Mayotte has no census row "
             "at all** — declared in the Eurostat geo dimension, no values — so it is Pew's "
             "on both shares and total, and France's one overwhelmingly Muslim département "
             "is on this map only because a second source counted it. `fr.py` now asserts "
             "the census row stays empty for the opposite reason to before: a row appearing "
             "would let the foreign half reach the unit and double-count against Pew. "
             "**Corsica is the one thing left undrawn, and it was looked for.** ESS carries "
             "709 variables and exactly one geography (`region`, plus `regunit` and "
             "`domicil`), so no round reaches it; Pew publishes the DOM because they have "
             "ISO codes and not Corsica because it is metropolitan France. What remains is a "
             "national survey with ~10 Corsican respondents, or the Annuario Pontificio's "
             "diocese of Ajaccio — and that is a `roll` against a `self_id` map, which §3.1 "
             "forbids and which would draw Corsica far more Catholic than the mainland "
             "purely as an artefact. 0.51%, and §6.12's wash marks it. "
             "**Boundaries were free and the trap that bit Greece did not fire.** GISCO's "
             "LAU bundle ships all 34,966 communes and the workbook mapping each to its "
             "département; 34,966 matched both ways, zero either side. Greece lost 644 "
             "codes to Excel stripping leading zeros and France has the same exposure — "
             "every commune in départements 01–09 — and escapes it because **Corsica's "
             "codes are `2A001` and `2B033`**, and one alphanumeric value forces the whole "
             "column to be read as text. `sources/fr_geo.py` guards it anyway, because a "
             "vintage without Corsica would put it straight back.",
    ),
}
