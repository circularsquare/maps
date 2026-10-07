# West and Central Africa — language coverage

24 countries. Swept 2026-10-03/04 using 18 of 22 WebSearch calls, plus IPUMS variable pages, the HDX API, and PDF text extraction.

**Overall:** this region is thin, as expected, but better than feared in the Sahel and Upper Guinea. Seven censuses ask a usable single-language question: Senegal 2023, Mali 2009, Burkina Faso 2019, Guinea 2014, Sierra Leone 2015, Côte d'Ivoire 2021 and CAR 2003. Mauritania asked mother tongue in 2013 but withheld the results. Guinea-Bissau, São Tomé and Chad ask multi-response "languages spoken". Ghana, Liberia, Togo, Gambia and Benin have ethnicity only. Nigeria, DR Congo and Cameroon have nothing.

**Best sources:**
- **CAR (USCB HDX):** 2003 census "langue couramment parlée", counts down to 177 communes, CC BY. This is the finest official count in the region.
- **Mali 2009 Tableau S-3:** language spoken crossed with mother tongue, by region (PDF).
- **CLEAR Global language datasets on HDX (CC BY-SA)** — the surprise of the sweep. They give admin-1 and admin-2 proportions computed from IPUMS census samples for Mali (cercles), Senegal, Guinea (prefectures), Sierra Leone (districts) and Benin (77 communes). They also cover DR Congo at territory level, from the 2016 CAID administrative reports. For Nigeria, Niger, Ghana, Burkina Faso, Gambia, Cabo Verde and Congo they are based on Afrobarometer. Two catches: they are proportions, not counts, and the census-based ones come from gated IPUMS microdata.
- **Afrobarometer:** direct CSV/SAV download with no registration (for example Mali R10). It covers 21 of the 24 countries and asks home language, with a region variable. It does not cover CAR, Equatorial Guinea or DR Congo.

**Biggest gaps:**
- Nigeria: no census language data since 1963. Afrobarometer is the only open stand-in.
- DR Congo: no census since 1984. CAID territory estimates are the stand-in.
- Cameroon: CLEAR's "2005 census" file turns out to be official-language/literacy data, not first language.
- Equatorial Guinea, Gabon, Congo and Cabo Verde were not checked (budget). They are rows marked `guess`.
- Niger 2012: the census report shows no language or ethnicity.

**Caveats:**
- Côte d'Ivoire 2021 publishes "most spoken language" nationally only, by urban/rural and age, on data.gouv.ci (Licence Ouverte).
- Burkina Faso 2019 publishes main language only at region level, by urban/rural.
- Senegal 2023 publishes first language most often spoken nationally; regional reports were not checked.
- Benin: IPUMS has only ethnicity, so CLEAR's "language" file may be a crosswalk from ethnicity. Verify before using it.
