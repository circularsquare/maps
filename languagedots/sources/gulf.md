# The Gulf states: the home-mix method (ae, kw, qa, om, bh)

Session edd42a8c-gulf, 2026-10-05. No Gulf state asks anyone a language. Each was built the way
Saudi Arabia was (`sources/sa.md`), under Anita's 2026-10-05 ruling for countries with no language
question (AGENT_BRIEF §2): citizens on the national Arabic variety, everyone else by nationality,
each nationality on its home country's language or home mix. **Every row `derived`.** Shared code:
`sources/gulf_mix.py` (imports `sa_census.py`'s pieces); one build per country,
`sources/<cc>_build.py`, and a short record per country, `sources/<cc>.md`.

## 1. Arabic varieties for citizens: one Gulf Arabic node, Oman and Bahrain's Shia apart

- **`afroasiatic.gulf_arabic`** (Glottolog gulf1241, whose countries list is AE BH IQ IR KW OM QA
  SA) for the citizens of the UAE, Kuwait, Qatar and Bahrain. One node shared, not four: Glottolog
  has no Emirati, Kuwaiti or Qatari language, only dialects of Gulf Arabic (it lists Bahraini Gulf
  Arabic, bahr1247, as one), and four near-identical colours on adjacent coasts would claim a
  difference nothing measures.
- **Saudi Arabia keeps `saudi_arabic`** as sa built it (its citizens speak Najdi and Hijazi
  mostly; the Eastern Province's Gulf Arabic is inside it). Not mine to merge; a later merge of
  Saudi Arabic into Gulf Arabic would be a mapping edit in `sa_census.py`.
- **Oman's citizens on `afroasiatic.omani_arabic`** (oman1239), which Glottolog keeps as a language
  apart from Gulf Arabic. Dhofari Arabic (dhof1235) and the Modern South Arabian languages of
  Dhofar are not split out: no source counts them (Saudi Arabia's no-guessing rule).
- **Bahrain's Shia Baharna on `afroasiatic.baharna_arabic`** (baha1259), see `sources/bh.md`.
- A citizen of one Gulf state living in another is drawn on their state's node
  (`gulf_mix.OVERRIDE`). UN DESA counts by birthplace where it can, so its "Kuwait" origin in the
  UAE (63,192) holds Kuwait-born non-Kuwaitis too; small, left.

## 2. Nationalities

- **India by state**, as Saudi Arabia, with the Keralite share for each destination from the
  Kerala Migration Survey 2023, Table 3.7 (`data/raw/gulf/KMS-2023-Report.pdf`, p.32, read):
  UAE 38.6%, Saudi Arabia 16.9, Qatar 9.1, Oman 6.4, Kuwait 5.8, Bahrain 3.7% of 2,154,275
  emigrants. The rest of each country's Indians at the MEA clearance mix 2011-17 (all ECR
  destinations together; nothing splits it by destination).
- **Pakistan** at the BEOE province mix 2019-21 (all destinations), as Saudi Arabia.
- **Origins on this map with 20,000+ people** in the country take their drawn home mix
  (`gulf_mix.HOME_MIN`, `DRAWN`), with sa's 1% cut. Smaller Arab origins on this map (Sudan
  or Chad under 20,000) take their drawn mix's largest language (Sudanese Arabic, Shuwa Arabic)
  rather than COUNTRY_LANG's bare "Arabic" or a twenty-language mix for a few hundred people.
  Others on `fr_build.COUNTRY_LANG`'s language; Yemen, Egypt, Iraq and the Levant on sa's Arabic
  nodes. Unnamed Arab nationalities (Oman's "Other Arabs") on `afroasiatic.arabic`.
- **Myanmar** takes Myanmar's drawn mix outside Saudi Arabia (sa's Rohingya call rests on
  GASTAT's Myanmar nationals being the Rohingya; Oman's are domestic workers, 99% women).
- **UN DESA's unnamed `Others`** on `other`: a residual of unnamed nationalities, so an unnamed
  language, the narrowest honest node (religiondots spread it at the named mix; a language map
  should not invent which languages they speak).

## 3. Calls someone might reverse

- One shared Gulf Arabic node (vs one per country, or plain `arabic`).
- DESA `Others` on `other` rather than spread.
- Each Gulf country's Indians at the same ECR state mix; only the Keralite share differs.
- Bahrain: Shia citizens on Baharna Arabic by religiondots' modelled sect split (`sources/bh.md`).
- Qatar: citizens estimated (no total published) from the 10+ count (`sources/qa.md`).
- Kuwait's Bidoon, Bahrain's Ajam and every non-Arabic-speaking citizen group: named in the
  records, not on the map; no table counts them.
