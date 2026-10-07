# Gambia: 2013 census ethnic group by LGA, read as language through Afrobarometer R7 mother tongues

Drawn 2026-10-05 (session edd42a8c-wafr). 1,857,181 people (2013 PHC), 8 LGAs, 11 nodes, every
row `derived`. 1,853 dots.

```
python sources/wafr_afro.py gm    # respondents from religiondots' Afrobarometer .sav (read-only)
python sources/gm_census.py
python taxonomy/build.py
python tools/check_country.py gm
python scatter.py --country gm
```

## 1. Sources

- **2013 PHC**, GBoS, *Spatial Distribution of Population and Socio-Cultural Characteristics*
  (religiondots' copy, `religiondots/data/raw/gm/census_2013_spatial_distribution_report.pdf`,
  read-only). Annex G, the Both-sexes table of each LGA (G.10 Banjul p. 59 ... G.61 Basse p.
  110): total, non-Gambians, not stated, and ten ethnic groups, counts. Ethnicity was asked of
  Gambians only. No language question. Checks: each LGA's parts sum to its total; totals equal
  religiondots' 2013 LGA figures (1,857,181); every share within 0.05 pt of Table 3.2.
- **Afrobarometer R7** (2018): 1,200 adults, ethnic group and Q2A "mother tongue". The Gambia
  is in R7-R9 only; R8-R9 ask "language spoken in home". CLEAR Global's Gambia layer is
  Afrobarometer-based, so not a second source. The 2024 census has preliminary totals only.

## 2. Retention and the lingua-franca question

R7 vectors (weighted; English answers, 20, all from English interviews, first moved to the
respondent's group's language): Mandinka/Jahanka 428 respondents, 93% Mandinka, 2% Jahanka; Fula
270, 95%; Wolof 158, 92%; Jola 158, 92%; Serahule 78, 86% (13% Mandinka); Serer 30, 57% (30%
Wolof). Kept whole (under 30 respondents): Manjago 22, Bambara 12, Other 29, Aku 0.

National %, by reading (`RETENTION_ROUNDS`):

| | no move | R7 mother tongue (drawn) | R8-R9 home |
|---|---|---|---|
| Mandinka | 34.4 | 34.2 | 38.2 |
| Fula | 23.9 | 23.9 | 19.7 |
| Wolof | 14.8 | 15.5 | 20.7 |
| Jola | 10.6 | 10.1 | 8.8 |
| Soninke | 8.1 | 7.0 | 6.3 |
| Serer | 3.1 | 2.4 | 1.3 |

R7 itself: 254 of 1,200 give a home language different from their mother tongue, mostly Fula,
Jola and Serer speakers naming Wolof or Mandinka. The home-language rounds are ask 018's other
reading; not used.

Drawn per LGA (top): Banjul Wolof 27%, Mandinka 23%, Fula 21%; Kanifing Mandinka 32%, Fula
18%, Wolof 16%, Jola 15%; Brikama Mandinka 39%, Fula 20%, Jola 18%; Mansakonko Mandinka 53%,
Fula 32%; Kerewan Wolof 32%, Mandinka 30%, Fula 22%; Kuntaur Fula 40%, Wolof 31%; Janjanbureh
Fula 40%, Mandinka 25%, Wolof 24%; Basse Serahule 33%, Mandinka 33%, Fula 29%.

## 3. Calls someone might reverse

- National retention vectors applied in every LGA (R7's regions are the old divisions; 1,200
  respondents would give a few dozen per group per LGA).
- Serer moved on 30 respondents (57% retention), at the MIN_N line.
- Non-Gambians (6%) drawn on their LGA's Gambian mix; most are Senegalese, whose languages
  (Wolof, Pulaar, Mandinka) are the same.
- Jola/Karoninka on Jola-Fonyi's leaf; Creole/Aku on Krio.

## 4. Room for improvement

A district-level ethnic table (GBoS publishes 43 districts for some items) would sharpen the
grain; a Gambian language question would replace the survey's retention.

## Terms

GBoS publication quoted. Afrobarometer: free download, citation requested. Glottolog CC BY;
Kontur CC BY 4.0.
