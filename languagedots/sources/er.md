# Eritrea (er): record

Drawn 2026-10-05 (session edd42a8c-mono4). 3,291,271 people (UN WPP 2024 for 2020), national
shares drawn on 6 zobas, 10 nodes, every row `derived`. 3,287 dots at 1:1000.

```
python sources/er_ephs.py
python taxonomy/build.py
python tools/check_country.py er
python scatter.py --country er
```

Files: `sources/er_ephs.py`, `taxonomy/er2010.py`, `taxonomy/tree.d/er.txt`, `countries/er.py`,
`data/normalized/er.csv`. Read-only from religiondots: the EPHS 2010 report
(`data/raw/er/ephs2010_final_report_v4.pdf`), `data/geo/er/er_lookup.csv` and `er_cells.gpkg`
(Meta 2020 cells scaled to the zobas; religiondots' `sources/er.md` for both).

## 1. What exists

- No census, ever. No Afrobarometer, Arab Barometer, WVS; no DHS microdata.
- EDHS 1995 and 2002 and **EPHS 2010** print ethnicity in their background table (EPHS Table
  3-1, pdf p.70: women 15-49, "Women ALL" 30,224 weighted; men 15-49, 4,299), **nationally
  only**; the zoba block is never crossed with it. EPHS 2010 is the newest: the source, under the
  ethnicity ruling (§2, tier D).

## 2. How the counts are made

Share = mean of women's and men's percentages: Tigrinya 63.9%, Tigre 21.4%, Saho 4.6%, Bilen
3.0%, Nara 2.9%, Afar 2.0%, Kunama 1.1%, Hedareb 0.9%, Rashaida 0.1%, other 0.2%. Times the
WPP 2020 total. Read asserted against pinned values.

**Retention:** no source gives home-language use by ethnic group (many Bilen also speak Tigre and
Tigrinya; Hedareb speak Beja, many also Tigre). Not applied.

## 3. Placement across zobas (calls someone might reverse)

The survey's unit is the whole country, so dividing its counts between zobas moves no count
(AGENT_BRIEF §4.4). By homeland, Tigrinya the residual in each zoba:

| group | zobas |
|---|---|
| Afar | Debubawi Keih Bahri up to 90% of it (`CAP`); the 19,000 left in Semenawi Keih Bahri |
| Bilen | Anseba (Keren) |
| Kunama, Nara, Hedareb | Gash-Barka |
| Rashaida | Semenawi Keih Bahri |
| Saho | Debub and Semenawi Keih Bahri, by population |
| Tigre | Anseba, Semenawi Keih Bahri, Gash-Barka, by population |
| other | every zoba, by population |

Result: Maekel 99.8% Tigrinya, Debub 88%, Anseba 36% (Tigre 44%, Bilen 20%), Gash-Barka 35%
(Tigre 44%, Nara 13%), Semenawi Keih Bahri 38%, Debubawi Keih Bahri 10% (Afar 90%). The splits
inside mixed zobas are by population, not measured. Religiondots' zoba populations come from
the survey's own sample proportions, which put only 50,000 in Debubawi Keih Bahri, fewer than
the survey's Afar (64,000); hence the overflow.

Other calls: Hedareb drawn as Beja (their language); Rashaida on `afroasiatic.saudi_arabic`
(Hijazi Arabic sits there in sa.txt); Nara a new leaf (nara1262) beside Kunama. Tigre hand
coloured (gold) apart from Tigrinya's red.

## 4. Room for improvement

EPHS or EDHS microdata (ethnicity by zoba) would replace the homeland placement; a language
question would replace ethnicity.
