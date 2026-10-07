# Portugal (pt): the record

Drawn 2026-10-05 by session edd42a8c-pt. 10,343,066 people (Censos 2021), 3,092 freguesias,
123 languages, 10,322 dots. Every row is `derived`. Portuguese 98.1%.

Build: `python sources/pt_censos.py --fetch` (INE indicator 0012294 -> `data/raw/pt/`, 60 MB),
`python sources/pt_censos.py` (writes `data/normalized/pt.csv`). Mapping `taxonomy/pt2021.py`
(reuses `fr2023.NAMES`), nodes `taxonomy/tree.d/pt.txt`, entry `countries/pt.py`.

## 1. Is there a language question?

- **Censos 2021**: no language item.
- **INE ICOT 2023** (Inquérito às Condições de Vida, Origens e Trajetórias, ages 18-74,
  35,035 dwellings): asks languages spoken until 15, now at home, with partner, with children,
  each "besides Portuguese" and "1st language". Published (presentation, INE attachment
  look_parentBoui=660879843) only nationally and in three groups: European languages, PALOP
  languages or dialects, other. 486.4k spoke another language at home until 15; 661.7k
  speak Portuguese and another at home now. No region, no origin breakdown, no share of
  immigrants speaking only Portuguese. Not usable for counts or for retention. The destaque
  page (ine.pt destaque 625453018) refused connections from WebFetch; its tables were not seen.
  Microdata may exist on request (not tried; gated).

So Anita's 2026-10-05 rich-country ruling applies (AGENT_BRIEF §2): Portuguese, regional
languages from surveys, immigrant languages by a citizenship proxy, all derived.

## 2. Immigrants: Censos 2021 nationality by freguesia

INE indicator **0012294** ("População residente por Local de residência à data dos Censos
[2021] (NUTS - 2024), Sexo, Grupo etário e Nacionalidade"), totals of sex and age. Every level
from Portugal to the 3,092 freguesias; freguesia codes are religiondots' `kod`.

**This replaces the SEF/AIMA plan in the brief.** The census counts every resident (AIMA counts
permits), in every freguesia (AIMA's tables are not published below municipality; not fetched), in the
same year and universe as the population it is subtracted from. The cost is vintage: foreign
residents with permits were about 0.7M in 2021 (SEF) and over 1.5M in 2024 (AIMA), against the
census's 542k foreign nationals. Said in `note_public`. Scaling to AIMA would change counts and
mix vintages; not done.

**Table traps** (asserted on every unit by the script):
- "1OP Outros países - Europa" is NOT a remainder: it is all of non-EU Europe (60,456 =
  Eurostat's EUR_NEU), named countries included. "ZZ Outros" is the non-EU Europe remainder
  (2,311 = 1OP less GB, UA, MD, RU, CH). The other continents' "2OP".."5OP" are true remainders.
- "Estrangeira" (542,165) excludes the 149 stateless: T = PT + EST + APA.

**Remainders** (ZZ, 2OP-5OP; 24,207 people) are split into countries INE does not name by
Eurostat's census 2021 citizenship by NUTS 3 (`cens_21ctz_r3`, religiondots'
`data/raw/fr/cens_21ctz_r3_fr.json`, read-only; the same census), the freguesia's NUTS 3 2021
from GISCO's `EU-27-LAU-2021-NUTS-2021.xlsx` (all 3,092 freguesias listed). National shares
where the NUTS 3 has none. Stateless go on `other`.

Checks: Portuguese + foreign + stateless = total on all 3,439 units; freguesias sum to Portugal
for every nationality; INE's 52 named nationalities equal Eurostat's national counts exactly
(largest gap 0); drawn rows sum to each freguesia's total (gap 0).

**Nationality -> language**: France's list (`fr_build.COUNTRY_LANG`), with Portugal's
overrides (`pt_censos.COUNTRY_LANG`):
- Brazil, Angola, Mozambique, São Tomé -> Portuguese (Angola's 2014 census: 71% speak
  Portuguese at home; Mozambicans and Santomeans in Portugal are largely urban and long settled;
  São Tomé's own census has Portuguese spoken by nearly everyone). France puts Mozambique on
  Emakhuwa; for emigrants to Portugal that would be wrong.
- Cape Verde -> Kabuverdianu, Guinea-Bissau -> Guinea-Bissau Kriol.
- India -> Punjabi (France and Spain: Hindi). Indian immigration to Portugal since 2015 is
  mostly the Alentejo and Algarve farm workforce from Punjab. No figure behind it.
- Ukraine 70/30 Ukrainian/Russian, Switzerland 65/25/10, Belgium 60/40, Canada 75/25, as Spain.
- China -> Sinitic group (much of it is Qingtian Wu; the source names a country).
- Pakistan Punjabi, Bangladesh Bengali, Nepal Nepali, Venezuela Spanish (many are
  Luso-Venezuelans; no figure to split by).

Portuguese citizens are all Portuguese: naturalised immigrants and their children are not
separable in this table.

## 3. Retention (the share speaking only the host language)

ICOT gives none (§1), so France's TeO2 shares are used, as the brief allows:
`fr_build.TEO_FRENCH` by `fr_build.teo_region` (sources/fr.md §2b; INSEE IMMFRA23-F18, share
not speaking the origin language with their children). Read here as "speaks only Portuguese".
Sahel (Guinea-Bissau) 47.6%, rest of Africa (Cape Verde) 47.8%, Europe 32.6%, other EU 33.2%,
Spain-Italy 37.6%, Asia 29.9%, China 33.7%, Americas 22.6%. Moved onto Portuguese: 103,326.
Largest before -> after: English 33,638 -> 23,379, Spanish 28,530 -> 20,148, Kabuverdianu
27,144 -> 14,183, Punjabi 18,223 -> 12,771, Chinese 16,631 -> 11,031, Kriol 15,298 -> 8,013.

Caveats: TeO measures immigrants in France with children; applied here to all foreign
nationals, retirees and children included. British and other northern European retirees in the
Algarve likely keep English more than this; recent South Asian workers probably more too.

## 4. Mirandese

Xosé-Henrique Costas (Universidade de Vigo), "Presente e Futuro da Língua Mirandesa" / "Usos,
Atitudes i Cumpetencias Lhenguísticas de la Populacon Mirandesa", 350 interviews, March 2020,
Miranda do Douro municipality: about 3,500 know it, about **1,500 use it regularly**; under-18
use 2%; used in families only in the municipality's rural (northern) freguesias (DN, 2023
report of the study; Wikipedia gives 1,000 "common users" from the same study; 1,500 is the
study's own figure as reported). 1,500 regular users is the closest thing to a home language.

Drawn: 1,500 on Miranda do Douro's 12 freguesias other than the town (4,399 people, 34.1%),
by population; the same 34.1% in Vimioso's Vilar Seco and Caçarelhos-Angueira (Angueira merged
in 2013), the two villages outside Miranda the sources name: 140 more. Total 1,640. Taken from
Portuguese. The 34% rate in Vimioso is an assumption; no survey covers it.

Not drawn: Barranquenho (Barrancos) and Minderico (Minde): no count. Galician-Portuguese border
speech is Portuguese.

## 5. Geography

religiondots' `data/geo/pt/pt_freguesias.gpkg` (GISCO LAU 2021, `kod` = INE's code, 3,092 of
3,092), read-only; one polygon per counted unit, so dots are uniform inside a freguesia.
Scatter: 10,322 dots on 2,294 polygons, 102 rings, 21,066 people (0.20%) under one dot per
language.

## 6. Colours

Mirandese hand-set magenta (0.60 0.15 330); its generated colour sat beside Portuguese's blue.
pl.txt defines the node bare; Poland draws a handful at most.

## 7. Room for improvement

- ICOT 2023 tables or microdata by origin would replace France's retention shares.
- A newer nationality-by-municipality source matched to a newer population base (AIMA 2024
  with INE's 2024 estimates) would catch the post-2021 immigration.
- Kontur hexes inside the large Alentejo freguesias (up to 863 km²) would place dots where
  people live; the Odemira farm workers would then cluster.
- A survey of Mirandese for Vimioso; Indian immigrants' languages (Punjabi vs Hindi, Gujarati).
