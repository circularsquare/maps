# 039 — cd: DR Congo: religion drawn at all 26 provinces, the eastern conflict provinces included

Summary: DR Congo is built from Enquete 1-2-3 household heads at 26 provinces, Nord-Kivu, Sud-Kivu, Ituri, Tanganyika and Maniema at the same grain, as ask 018 did for Burkina and Mali. Keep, coarsen the east, or hold?

*Filed 2026-09-15 by session `cb8b206e-cd`. Anita's call; nothing is waiting on it.*

## What I did

Built DR Congo at its 26 provinces from the Enquête 1-2-3 household heads (2005 and 2012, the U.S. Census Bureau's tabulation on HDX), every province at the same grain, the eastern conflict provinces included, with no coarser tier there. Nothing finer than a province is drawn anywhere. It is waiting for the supervisor's build tail, which will publish it unless `cd` is held.

## What it costs to reverse

Hold it: take `cd` out of `ORDER` in `countries.py` before the tail, a one-line edit. Coarsen the east: merge the chosen provinces into one unit in `sources/cd.py` and `sources/cd_geo.py` and re-scatter, about an hour.

## Why it is yours rather than mine

AGENT_BRIEF §3's first bar, spec §14: at what resolution a country is drawn where armed groups attack people partly for their religion. The supervisor's brief for this country said to flag it and not settle it.

## The detail

- **The east as drawn** (shares of people, by the religion of their household head): Nord-Kivu 45.9% Catholic, 40.4% Protestant, 1.9% Muslim; Ituri 73.7% Catholic, 18.8% Protestant, 0.5% Muslim; Sud-Kivu 48.7% Catholic, 36.4% Protestant, 4.9% Muslim; Tanganyika 33.7% Catholic, 39.2% Protestant, 0.8% Muslim; Maniema 16.4% Muslim, the country's highest.
- **The units are large**: Nord-Kivu 9.3 million people on COD-PS 2024, Sud-Kivu 8.0M, Ituri 4.6M, Tanganyika 4.6M, Maniema 3.2M. The survey also has 164 districts (median 174 heads); they are not drawn, because nothing without the microdata can test them, so the statistical ceiling and the §14 choice land on the same grain.
- **The concern** is the Allied Democratic Forces in Beni and Lubero (Nord-Kivu) and in Irumu and Mambasa (Ituri), who have attacked churches and Christian villages, and the reverse risk for the small Muslim minority there. The map says at province level what any account of the region says: that Ituri and the Kivus are overwhelmingly Christian and Muslims a small minority. It does not place anyone below the province.
- **Precedent**: ask 018 (Burkina Faso at 45 provinces and Mali at 20 régions, "its fine to draw at province / region level"), ask 017 (Chad at 22 régions despite attacks on Christians). Those rulings cover those countries only, which is why this is here.
- **The data are old for the east**: 2005 and 2012 fieldwork. Of the 16 districts with no sampled household, Nyiragongo, Shabunda and Idjwi are in the Kivus; their provinces take the shares of the districts that were sampled.
- **Options**: (a) keep as built; (b) merge Nord-Kivu, Sud-Kivu and Ituri into one unit (loses little: their mixes are alike, Ituri more Catholic); (c) draw Muslims at the national share in those three and keep the rest; (d) hold the country until a DHS file replaces the source.
- **Separately, an optional download (added by the supervisor):** the drawn level is dated. DHS 2023-24 puts women at 23.7% Catholic and 39.1% in non-denominational churches, against 35.3% Catholic and 22.3% other Christian as drawn from 2005-12. Replacing it needs a free DHS Program registration (dhsprogram.com, a short project description) and two files from the DR Congo 2023-24 survey: the women's recode `CDIR81FL` and the men's recode `CDMR81FL` (Stata or flat). If you are willing to register, put them in `data/raw/cd/`; if not, DR Congo stays on the older survey.
