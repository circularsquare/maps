# Israeli settlements in the West Bank (xs): record

Drawn 2026-10-05 (session edd42a8c-xs). Not a country: religiondots' `xs` entry (its
`sources/xs.md`, Anita's ruling on its ask 028), same units and same line. CBS Social Survey 2021
native language (adults, Jews and others) applied to the CBS 2022 census population of the 267
units beyond the Green Line, West Bank settlements and East Jerusalem: 723,899 people, 9 nodes,
rows `modelled`. 718 dots on religiondots' Kontur pieces of those units (`xs_places.gpkg`).

Files: `sources/xs_social.py`, `taxonomy/xs2021.py`, `taxonomy/tree.d/xs.txt`, `countries/xs.py`,
`data/normalized/xs.csv`. Survey tables are Israel's cache (`data/raw/il/`, `sources/il_social.py`
fetches them); nothing new downloaded.

## Who is in it

Religiondots' `data/normalized/xs.csv`, read-only: Jews 695,648 and the register's Others 28,251
on the 267 units of `data/geo/il/dropped_units.json`, which Israel's entry (`il`) refuses. East
Jerusalem (locality 3000) 237,240 on 78 units. The Muslims and Christians in the same units
(373,257) are on Palestine's entry, whose census counts East Jerusalem. Asserted in
`xs_social.py` and `countries/xs.py`: every unit is a dropped unit; total 723,899. Nobody is drawn
twice.

## Method

All of it is Israel's model (`sources/il.md`), on different cells:

- **Survey cell.** Everyone is in the survey's "Jews and others" group. Units outside Jerusalem
  take sub-district 71, Judea and Samaria (215,496 adults; its own district, so the shrink to
  district does nothing): Hebrew 70.5%, Russian 7.3%, English 7.7%, Yiddish 4.9%, French 3.1%,
  Spanish 2.4%, Amharic 1.3%, other 2.6%. **East Jerusalem takes Jerusalem (11)** instead,
  because CBS files it in the Jerusalem sub-district, not Judea and Samaria; shrunk towards the
  Jerusalem district as on Israel's entry (Hebrew 68.8%, English 7.9%, Russian 5.5%, French 5.0%,
  Yiddish 4.9%, other 5.1%, Arabic 0.9%).
- **Children**: sa2022 `age0_19_pcnt` per area (99.6% of units matched directly; 47.7% under 20,
  population-weighted) at the national 20-24 shares of Jews and others.
- **Across units**, raked (IPF, 200 rounds) inside each cell from the origin profile, as
  `il_social.py`; Yiddish weighted by religiondots' `Jews [Ultra-religious]` rows. Unit totals
  and cell language totals held, both asserted.
- **Inside a unit**: religiondots' Kontur pieces, `pop_weight` (the same weight religiondots took
  off Palestine's placement for these people).

Result: Hebrew 76.2%, Russian 7.0%, English 5.5%, Yiddish 3.5%, French 2.7%, other 2.3%, Spanish
1.3%, Amharic 1.1%, Arabic 0.4%. Spot checks: Ariel Russian 24%, Efrat English 21%, Modi'in Illit
Yiddish 10%, Beitar Illit Yiddish 9%, Ma'ale Adumim Russian 10%. Scatter: 718 dots; 5,899 people
(0.81%) under one dot.

## Calls someone might reverse

- East Jerusalem's Israelis on the Jerusalem sub-district's shares rather than Judea and
  Samaria's (the survey's own filing).
- Children at Israel-wide 20-24 shares, although half the population here is under 20.
- Yiddish in Modi'in Illit and Beitar Illit only ~10%: the cell total is the survey's 4.9% of
  adults, and the raking can only move it between units.
- No `territory=False` flag: languagedots' viewer has no country shapes or Auto pick by shape, so
  `xs` is a plain picker entry. If the viewer gains one, this entry needs it (religiondots §1).
- `il` and `ps` text updated to point here instead of saying the settlers are not drawn.

## Terms

CBS table generator: public, no login. CBS 2022 census figures and Kontur placement via
religiondots, read-only; Kontur CC BY 4.0.
