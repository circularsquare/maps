# td — parked 2026-09-14, session `f95259a4-td`

*Parked on a §14 question, not for context: `ask/017-td`. Resume only once Anita has ruled
(the ask moves to `ask/answered/`). If she says leave Chad off, delete `sources/td*`,
`taxonomy/td2009.py`, `data/raw/td/`, `data/normalized/td.csv`, `data/geo/td/`, this file, and
the queue row's first sentence.*

## Last COMMANDS.txt step completed

**Step 3, with step 2 done: checkpoint B plus the mapping.** `data/normalized/td.csv` reconciles,
`data/geo/td/` is built, `taxonomy/td2009.py` is written. Step 4 (the `countries.py` entry) was
deliberately NOT started, so the tree is not in the registered-but-not-built state.
`python kontur_cap.py td` was run: no block at the limit, no rows needed.

## What is on disk

All trustworthy, none a stub:

- `data/raw/td/rgph2_etat_structures.pdf` (the source, Wayback), `rgph2_questionnaire.pdf`,
  `rgph2_resultats_sous_prefecture.pdf` (a scan), `rgph2_manuel_agent.pdf` (the post-enumeration
  survey's manual, not needed), `tcd_admin_boundaries.shp.zip` and `shp/`
- `data/geo/kontur/kontur_population_TD_20231101.gpkg(.gz)`
- `sources/td.py` -> `data/normalized/td.csv` (113 rows, 22 régions, 10,941,682)
- `sources/td_geo.py` -> `data/geo/td/td_regions.gpkg`, `td_hexes.gpkg` (93,946 hexes),
  `td_lookup.csv`
- `taxonomy/td2009.py` (MAP, REVIEW, COLUMNS; `other.td` is referenced and does not exist yet)
- `sources/td.md` (the record), spec §12 *A census older than its boundary file needs an area
  check*, the queue row, `ask/017-td`

## What I was about to do

If the ruling is to draw it:

1. `python tools/claim.py take td --id <sid>`
2. Add `other.td` to `taxonomy/branches.py` after the `other.gw` block, then
   `python taxonomy/build_tree.py`. Drafted:

   ```python
       # --- added <date> with Chad.
       ("other.td",
        "Other religion (Chad)",
        "INSEED's `Autres religions` in the RGPH2 2009, the fifth box on a card that offers "
        "animist, Catholic, Muslim, Protestant, other and none: **56,657 people, 0.52% of the "
        "censused population.**\n\n"
        "The census report does not say what is in it. 91% of it is in the seven southern "
        "régions: 3.6% of Mandoul, 1.4% of Mayo Kebbi Ouest, 1.1% of Moyen Chari."),
   ```
   Then drop the "does NOT exist yet" paragraph from `taxonomy/td2009.py`'s docstring.
3. `countries.py`, re-read immediately before and after editing. Adapters beside `_gn_*`:

   ```python
   def _td_place_weight(place):
       """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

       N'Djaména is 436 km2 holding 951,418 people in 2009 and Tibesti 213,590 km2 holding
       21,303; the Sahelian régions have their people along wadis and the lake
       (sources/td_geo.py).
       """
       return _kontur_place_weight(place, "td_hexes.gpkg", "sources/td_geo.py")


   def _td_counts():
       """Chad RGPH2 2009 at région: 6 nodes on 22 units, every row `measured` and may ring.

       Tableau 5.07's shares raked to its région populations and Tableau 5.06's national counts
       (sources/td.py). The universe is the censused population, 10,941,682, collective
       households and refugee camps included; the 98,191 estimated are in no table.
       """
       from td2009 import resolve

       df = pd.read_csv(HERE / "data" / "normalized" / "td.csv",
                        low_memory=False, keep_default_na=False, na_values=[""])
       df["count"] = df["count"].astype(int)
       if df["geo_id"].nunique() != 22:
           raise SystemExit(f"{df['geo_id'].nunique()} régions in td.csv, expected 22 -- re-run "
                            "sources/td.py")
       df["node"] = df["source_category"].map(resolve)
       unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
       if unmapped:
           raise SystemExit(f"td.csv categories with no node: {unmapped}")
       df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
       df["congregations"] = 0
       df["may_ring"] = True
       df["tier"] = "measured"
       return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]
   ```

   The entry, beside `"gn"`; `note_public` kept short on purpose, and cut further if anything in
   it is on the fence:

   ```python
       "td": dict(
           name="Chad",
           source="RGPH2 2009, État et structures de la population (INSEED), Tableau 5.07",
           basis="self-identification, censused population",
           note_public=(
               "**Chad's 2009 census publishes religion for its 22 régions, and the country "
               "divides north and south.** Twelve of the fourteen northern and central régions "
               "are **98%** Muslim or more. The seven southern régions hold 90% of Chad's "
               "Catholics, 89% of its Protestants and 94% of its animists, and Mayo Kebbi Est is "
               "**32.0%** animist. N'Djaména is 70.7% Muslim and 27.9% Christian."),
           how="census, 2009",
           grain="régions, 500,000 people on average",
           gap="0.89%: 98,191 people in parts of Sila and Tibesti that enumerators could not "
               "reach, whose number was estimated",
           gap_share=0.008894,
           counts=_td_counts,
           units=None,
           unit_key=None,
           place=HERE / "data" / "geo" / "td" / "td_hexes.gpkg",
           place_unit=lambda g: g["unit"].astype(str),
           place_weight=_td_place_weight,
           note="RGPH2 2009 structure volume, Tableau 5.07 (région x religion, % to one "
                "decimal, with populations), only on INSEED's retired jdownloads store via "
                "Wayback. Raked to 5.07's région populations and 5.06's national counts. The "
                "form's B12 codes match the columns and animists had their own box. Régions "
                "rebuilt from COD-AB départements: Ennedi Est + Ouest, and Djourf Al Ahmar and "
                "Abdi moved back to Sila and Ouaddaï. sources/td.md has the record.",
       ),
   ```
4. `python tools/check_mapping.py td`, `python tools/check_md.py`, `python tools/gap_share.py td`
   (the gap is people never in a table, so the tool should find nothing and 0.008894 stays).
5. `scatter.py --country td` at 1:1,000 and `--dot-value 10000`, `coverage.py`,
   `python tools/build_tail.py --id <sid>`, `python tools/built_countries.py --check`.
6. A §9-series section in `sources.md` (check the headings first), the queue row to Drawn, delete
   this file.

## The one thing that will bite you

**`sources/td_geo.py` builds the régions from COD's ADM2, not ADM1, on purpose.** COD-AB 2025 has
Djourf Al Ahmar in Ouaddaï and Abdi in Sila; in 2009 it was the other way round. With ADM1 every
name joins and every total closes, and Sila is drawn 31% too small. The area assertion catches a
revert. Do not "simplify" it back.

## Everything else

All written into `sources/td.md` (construction, checks, questionnaire, the footnote misprint,
boundaries, §14 evidence) and `ask/017-td`. One placement caveat not acted on: Sila's
uncounted people are all in Kimiti département (Moudeïna and Tissi sous-préfectures had nobody
censused), but COD stops at départements, so Sila's dots still fall there.
