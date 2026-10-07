# Equatorial Guinea (gq): record

Drawn 2026-10-05 (session edd42a8c-mono4). 1,225,377 people (2015 census), national shares
drawn on 7 provinces, 6 nodes, every row `derived`. 1,221 dots at 1:1000.

```
python sources/gq_dhs.py
python taxonomy/build.py
python tools/check_country.py gq
python scatter.py --country gq
```

Files: `sources/gq_dhs.py`, `taxonomy/gq2011.py`, `taxonomy/tree.d/gq.txt`, `countries/gq.py`,
`data/normalized/gq.csv`, `data/raw/gq/dhs2011_FR271.pdf`. Province populations and hexes from
religiondots (`data/geo/gq/`, read-only).

## 1. What exists

- No census language question (2015 census: none found; coverage sweep: no Afrobarometer, no
  IPUMS). No language question in any open survey.
- **DHS 2011 (EDSGE-I, report FR271)**, Cuadro 3.1 (pdf p.60): ethnicity of women (3,575) and men
  (1,557) aged 15-49, weighted, **national only**; region is a separate row block (insular,
  continental), never crossed with ethnicity. Microdata would give the cross but DHS
  registration is off (needs an institution). Built under the ethnicity ruling (§2, tier D).
- UN DESA's migrant stock for the country is 95% "Others" (220,112 of 230,618 in 2020), so it
  cannot name the foreigners' origins.

## 2. How the counts are made

National share per group = mean of women's and men's percentages, "Sin información" (0.1/0.3)
dropped: Fang 77.27%, Bubi 9.66%, Ndowe 2.75%, Annobonese 2.75%, Bisio 0.80%, foreign 6.16%,
other 0.60%. Times the census's 1,225,377. The read is asserted against pinned values and each
sex's column sums to 100 +-0.2.

**Retention:** no source gives how many of each group speak Spanish (or Fang, for minorities in
the cities) at home. Not applied; said in the note.

## 3. Placement across provinces (calls someone might reverse)

The DHS's unit is the whole country, so spreading its national counts over the provinces moves
no count (AGENT_BRIEF §4.4). Homeland, not population, decides which provinces:

| group | provinces | basis |
|---|---|---|
| Bubi | Bioko Norte, Bioko Sur, by population | the island's own people |
| Annobonese | all of Annobón (5,314); the other 28,434 in Bioko Norte and Litoral by population | the island holds a sixth of them; the rest live in Malabo and Bata |
| Ndowe, Bisio | Litoral | the mainland coast (Bata, Mbini, Kogo; Bisio north to Rio Campo) |
| foreign, other | every province but Annobón, by population | nothing places them |
| Fang | the rest of each province | |

Result: Bioko Norte 35% Bubi, 54% Fang; Litoral 77% Fang, 9% Ndowe. Without this the Bubi would
be drawn on the mainland and the Fang on Annobón. If reversed: one national mix per province.

Other calls: "Ndowe" is one leaf (Kombe, Benga and relatives), because the DHS names the people,
not a language. "Extranjero" (6.2%, men 8.2%) on `other`, as nothing names their nationality.

## 4. Room for improvement

A language table, or the DHS 2011 microdata's ethnicity by province, would replace the homeland
placement. A nationality table for foreigners would give their languages.
