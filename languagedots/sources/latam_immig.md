# Immigrant languages in the indigenous-question Latin American maps

Session edd42a8c-lats, 2026-10-05. Anita's priority: non-indigenous minority languages in Latin
America. `sources/latam_immig.py` turns foreign-born by country of birth per unit into languages
(`origin_mix.mix(iso, cc)`; the host language's share stays; the rest retained at France's TeO2
rate for the origin's region, `fr_build.TEO2`; the remainder onto the host language) and takes
them out of the country's `derived` host-language remainder. Where the census already measured
the same language (Guarani, Quechua, Aymara among indigenous speakers), only the estimate above
the measured count is added. A recursion guard (`active()`) stops ar <-> cl home-mix loops.

| cc | source | added |
|---|---|---|
| ar | Censo 2022 PAISNAC per departamento | ~460,000 (Paraguayan Guarani +240k, Portuguese 56k, Italian 38k, Quechua +33k) |
| cl | Censo 2024 D4 workbook, 13 birthplace groups per comuna | 124,799 (Haitian Creole 59k, Portuguese 16k) |
| co | CNPV 2018 PA3_PAIS_NAC per municipio | 37,750 (English 17k) |
| ve | Censo 2011 ENCUALPAIS per parroquia | 103,981 (Portuguese 28k, Italian 15k, Arabic 12k) |
| br | Censo 2022 SIDRA 10157 naturalised + foreign per município (1,009,330) x each UF's SISMIGRA active-registration nationality mix | 690,252 (Spanish 472k, Haitian Creole 47k, English 18k, Japanese 18k, Mandarin 14k) |
| ec | Censo 2022 migration tabulado: foreign-born per canton (1.1) x province country mix (7); northern rule (DROP_ROOTS, HISPANIC whole) | 23,114 (English 11k, Italian 2k) |

Venezuelans are Spanish in ar cl co br pe ec (uncited override, origin_mix.md §2b).

## Session 5d7dac7e-br, 2026-10-06

- **br** done: immigrants as in the table (sources/br.md, "Immigrant languages"), plus settled
  communities as `modelled` estimates placed by homeland: Hunsrik 1,173,256 (RS, SC, SW
  Paraná; Altenhofen, Morello et al. 2018), Talian 585,919 (RS; BIRS 1990 via the same book),
  Pomerano 120,000 (ES; IPOL 2014). sources/br.md, "Settled communities".
- **ec** done (sources/ec.md, "Immigrant languages").
- **py, bo**: the census's German is drawn as Plautdietsch in the Mennonite colony districts
  (py, nine districts) and outside the department capitals (bo). sources/py.md, bo.md.
- **Helpers merged**: `sources/latam_imm.py` (mx cr pa hn ni) is folded into
  `sources/latam_immig.py` (`spread`, `unit_rows`, `HISPANIC`, `DROP_ROOTS` kept; a
  `drop_roots` option on `immigrant_languages`). Every caller's national counts compared before
  and after: unchanged except Mexico (Armenian +2.8, Georgian +1.4 people) and Panama (Spanish
  -0.8), from the Caucasus' TeO2 block; not re-scattered (under one dot).

## Not done (as of 2026-10-05; br and ec since done, above)

- **br**: the 2022 census has nationality per município (SIDRA 10157: estrangeiros,
  naturalizados) but no country of birth yet; 1991 is the last census table with countries
  (SIDRA 1626). SISMIGRA-ATIVOS (Polícia Federal active registrations, monthly, by country) is
  behind a UnB SharePoint login: https://unbbr-my.sharepoint.com/:u:/g/personal/obmigra_unb_br/IQD9ZTLjQ5vES4goztUjjisjAZhI7AjrR5Tz2K0sAgnH7Nk
  Hunsrik, Pomeranian, Talian, Japanese: no speaker counts found yet.
- **pe bo py**: not indigenous-only. Peru 2017 and Bolivia 2024 ask everyone's mother tongue;
  Paraguay 2002 asks every household's language (Portuguese 124k, German 36k, Japanese, Korean
  already drawn). Left as built. Plautdietsch: py and bo record Mennonites as German; a split
  by colony municipality would be a place-dependent relabel, not done.
- **ec**: languages spoken (multi-answer); foreign languages folded to Spanish by ruling;
  immigrants not yet added.
