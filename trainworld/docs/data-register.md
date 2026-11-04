# Data register by country

What each country has for every demand input, what the game uses, and which procedure (archetype)
its packs follow (T-501).

**Reading it.** One row per country, the inputs split over four tables. A cell is a status mark,
then the source and its grain:

- **U** used by the country's packs
- **N** usable, not used yet: the task that would use it
- **G** gated: exists, but behind a key, an application, a licence or a bot wall (why)
- **?** said to exist, not opened yet (task where one is worth it)
- **skip** usable, judged not worth it (why)
- **-** none found

Sources' URLs and the measurements behind each choice live in the country's note (last column of
the first table); the rules themselves are in demand-worldwide.md (per country) and
demand-sources.md (the US). **Updating it**: a city task that uses, finds or rules out a source changes that cell, filing a
P3 task for anything marked N. A new country gets a row in every table before its first build.
Countries on global sources only (archetype D, no country module) share one row each in the
Population, Jobs, Commuting, Travel surveys and Destinations tables where every cell would read
the same; split a country out when it gets a source of its own. A pack's status ("shipped" is in
the registry, "parked" failed the runner's gate, "held" waits on Anita) is the plan file's as of
2026-10-22 (T-940); `world_run.py status` is current. Calibration counts gathered for T-509 are
loose checks only since T-636 (Anita, 2026-10-17): no country constant comes from them.

## Countries and their procedure

| Country | Packs | Archetype | Notes |
|---|---|---|---|
| United States | hand-built nyc phl bos wdc bal pvd nhv spf and the zone wor; the rest of the country and Puerto Rico built (T-1004, 2026-10-24: the metro rows, Hartford's grown by Windham, New London and Litchfield in T-1113, and the county zones, on the tier 0 runner with the `usa` module, `pipeline/usa.py` (T-1016): TIGER county outlines, blocks less institutions, LODES jobs and sectors, ACS work from home by county, the US curve and state factor; Puerto Rico its own country, outline, people and work from home from the module, jobs, curve and region tier 0) | A | docs/demand-sources.md, notes/T-142.md, notes/T-1004.md |
| Canada | tor; mon van cag edm shipped (metro rows with their remnants, tier 0 with the `canada` module's population by dissemination area and everyday destinations by census tract of work, T-564) and 46 zones of census subdivisions within reach of cities (T-1191's rule; the zones take global sources and the zone curve, no module; the north dead space), built in T-1239 and T-1377: 45 shipped, chw and six with their drive factor relaxed; lel parked (T-1376); tor re-solved with the group | A (tor); D in effect for the plan rows | notes/T-277.md, notes/T-1191.md, notes/T-1239.md; OSM extracts of 2026-10-26 (Geofabrik's Canadian provinces, British Columbia by its admin-region sub-extracts) |
| England (UK) | lon man lee bir not new she liv por; the rest of England as zones with Wales, Scotland and Northern Ireland, 56 in all in the registry (T-788: ITL3 units grown within ITL2, replanned 2026-10-22; Leicester inside Leicestershire's zone; wil's extracts by hand) | A | notes/T-288.md (T-360 part); loose checks (gathered for T-509): U TfL NUMBAT 2024 station entries and line boarders, U ORR Table 1410 station usage 2024-25, U Census 2011 QS701EW travel to work by MSOA (2021 TS061 skip: lockdown), U GB-wide GTFS (aubin.app, Transitous) for National Rail; G Network Rail and RDG timetable feeds (login); the eight plan packs (man lee bir not new she liv por, all shipped) use the sources of the other tables here, with a borrowed commute curve (T-314) and no Heathrow share; OSM extracts of 2026-10-17 (notes/T-726.md) |
| France | par; lil mar lyo bor tou (plan rows, shipped); nic parked (T-769); the rest of metropolitan France as 80 zones (T-779; Creuse and Lozère dead space, Alpes-Maritimes out with nic, overseas départements not zoned). The zones take global sources only and the zone curve, not the France module (OSM extracts 2026-10-20); the module's sources (MOBPRO flows, Filosofi squares) are usable for them later (T-785) | A, by commune (par and the plan metros); D for the zones | notes/T-289.md; loose checks (gathered for T-509): U IDFM GTFS, U IDFM validations per ZdC per day, U INSEE RP 2022 transit share to work by commune (no rail split); G EGT 2018-20 result tables (omnil.fr 403 to scripts); the six plan packs use the other tables' sources with a borrowed commute curve and no Charles de Gaulle share; OSM extracts of 2026-10-19 (notes/T-726.md) |
| Germany | ber dor fra col ham han mun stu; mah nur dus (plan rows, shipped); the rest of the country as 102 zones (T-788: Kreise grown within their Regierungsbezirk to 0.3M, the last 9 to 0.5M; no dead space) | A, by Gemeinde | notes/T-290.md; the ten plan packs use the sources of the other tables here (Zensus grid, Pendlerrechnung jobs) with a borrowed commute curve; OSM extracts of 2026-10-17 (dor fra col ham han mun) and 2026-10-19 (stu mah nur dus) (notes/T-726.md) |
| Japan | tyo osa ngo fuk spk sdj hij, 38 zones | A | notes/T-420.md, T-432, T-463; loose checks (gathered for T-509): U KSJ S12 station counts, U 2020 census table 18 rail share by municipality of residence (e-Stat e18.xlsx), no country rail constant since T-636; U KSJ N02-24 lines and stations with MLIT passing counts (japanriders' segments) for the whole-country real network, a stress test only (T-737) |
| Brazil | sao | B | notes/T-458.md |
| China | sha; T-897's Jiangsu pilot: suz nan chz hua xuz yac nat zha (plan rows, shipped) and 30 zones (geoBoundaries ADM3 counties grown within their province to 1M, no dead space); the other provinces planned in waves (notes/T-897.md) | C, curve from a mean (sha); D in effect for the Jiangsu rows (`china.plan_hooks` gives the metros the township census from T-1150 and the zones from T-1185, nothing else beyond Shanghai's tables: the borrowed curve for the metros, the zone curve for the zones) | notes/T-291.md, T-897; OSM extracts `asia/china/jiangsu` (xuz and pex also shandong), batch 20261022-093304 |
| South Korea | seo; bus daj gwn shipped (metro rows with their remnants, tier 0: `korea.plan_hooks` gives none) and 18 of 19 zones of whole si, gun and gu (T-1191, built T-1238); dae and too shipped in T-1262 (too with its top-zone jobs gate relaxed at 25.0%) | C, curve from a mean (seo); D in effect for the plan rows | notes/T-457.md, notes/T-1191.md; loose checks (gathered for T-509): U seoulriders' timetable, card gate and KRIC station counts; G census commuting mode by si-gun-gu (KOSIS) |
| Vietnam | the whole country from the world plan (T-1738): hoc hao can ann (metro rows with their remnants, tier 0, no module) and 91 zones of geoBoundaries ADM2 districts (the pre-2025 map) on the default zone rule; 8 mountain districts dead space; zones global sources and the zone curve; 93 of 95 shipped (can, and seven zones with drive_factor_median, relaxed; lao's browser school check, cba's top cell), muo and cxx parked (T-1744) | D | notes/T-1738.md; OSM extract `asia/vietnam` of 2026-10-28 |
| Australia | the cities and the corridors between from the world plan (T-1857): mel syd bri per (metro rows; Sydney grown by hand by Central Coast, Campbelltown, Camden and Wollondilly) and 24 zones of geoBoundaries ADM2 LGAs on Canada's rule (100,000 people within 150 km; the interior, north-west and Cape York dead space, Whyalla and the scattered Unincorporated NT too), Adelaide made one zone by hand (the cap had split it); every row takes the `australia` module's population by SA1, the rest tier 0; all 28 shipped (daw with its floodplain water points relaxed) | C (population by SA1) | notes/T-1857.md; OSM extracts `australia-oceania/australia/*` of 2026-10-30 |
| Thailand | the whole country from the world plan (T-1738): ban Bangkok (metro row with its remnants, tier 0, no module) and 66 zones of geoBoundaries ADM2 amphoe on the default zone rule; 35 hill amphoe dead space (0.77M people); zones global sources and the zone curve; all 67 shipped; five zones' origins moved by hand to their towns (T-1762) | D | notes/T-1738.md; OSM extract `asia/thailand` of 2026-10-28 |
| Singapore | sin, one metro row of 48 geoBoundaries ADM2 subzones (T-1738), tier 0, shipped; the Johor Bahru border at the pair factor 0.5 (borders.json, T-1738: about 170,000 a day each way in the joint solve) | D | notes/T-1738.md; OSM extract `asia/malaysia-singapore-brunei` of 2026-10-29 |
| Cambodia | the country from the world plan (T-1738): phn Phnom Penh (metro row with 2 remnants, tier 0, no module) and 20 zones of geoBoundaries ADM2 districts on the default zone rule; 32 districts dead space (0.83M); 17 rows shipped (bnl with its top-zone jobs share relaxed), the four north-west zones tpe anv phm slk built on the 2019 census by district (T-2000; anv shown as Samraong, phm Bavel, slk Pailin) | C (admin_totals, the north-west) | notes/T-1738.md 11; OSM extract `asia/cambodia` of 2026-11-02 |
| Malaysia | the country from the world plan (T-1738): kua Kuala Lumpur and joo Johor Bahru (metro rows with their remnants, tier 0, no module) and 27 zones of geoBoundaries ADM3 mukim grouped by district (`group_level`, as Korea; far parts of dissolved same-named mukim split, `GB_SPLIT_FAR`); 200 mukim dead space (1.38M, the interiors); all 29 shipped; Malaysia-Singapore border factor 0.5; kua's origin moved by hand to the centre (Masjid Jamek, T-2036) | D | notes/T-1738.md 11; OSM extract `asia/malaysia-singapore-brunei` of 2026-11-02 |
| Indonesia | the country from the world plan (T-1738): 22 metro rows (jak Jakarta 37.55M, sur Surabaya, tan Tangerang, bad Bandung, yog Yogyakarta ... pea Pekalongan; tier 0, no module) and 183 zones of geoBoundaries ADM2 kabupaten (rule `IDN`: `contiguous`, density floor 10 a km2); 23 units dead space (1.9M: Kalimantan's far interior, Papua's southern swamps, Sabang and Anambas); Papua's yai puc pai mmi mru biu maw rebuilt on the 2020 census by kabupaten (T-2038; puc's Mulia cell relaxed), set built after T-2039's parks fix | C (admin_totals, rebuilt rows) | notes/T-1738.md 12; OSM extracts `asia/indonesia/{java,sumatra,nusa-tenggara,sulawesi,kalimantan,papua,maluku}` of 2026-11-02 |
| Philippines | the country from the world plan (T-1738): mai Manila 29.9M, ceb Cebu City, ang Angeles, dav Davao City (metro rows with their remnants, tier 0, no module) and 99 zones of geoBoundaries ADM3 municipalities (rule `PHL`: `contiguous`); 29 units dead space (0.24M); 102 shipped (one-town zones' top-zone jobs share, Makati's top job cell and Davao City's FUA ratio relaxed); pti (Sulu, Tawi-Tawi, Basilan), mrw lbx ctb built on the 2020 census by municipality (T-2053; pti short of its units' census by the shore clip, T-2099); Manila's and Angeles's origins moved to their centres | C (admin_totals, rebuilt rows) | notes/T-1738.md 13; OSM extract `asia/philippines` of 2026-11-03 |
| New Zealand | the country from the world plan (T-1738): auc Auckland 1.89M (metro row with its remnants: Rodney, Hauraki, Thames-Coromandel; tier 0, no module) and 5 zones of geoBoundaries ADM3 wards grouped by territorial authority (rule `NZL`: no density floor, `contiguous`, `join_km` 250): Hamilton (with Tauranga, Rotorua, Gisborne, Hawke's Bay), Wellington (with Palmerston North, Whanganui, Taranaki), Christchurch (with Nelson, Marlborough, the West Coast), Dunedin (Otago, Southland), Whangārei (Northland); the Chatham Islands dead space; all 6 shipped, no relaxations, then rebuilt on 2023 census SA2 totals (T-2077); realms North Island and South Island | C (admin_totals) | notes/T-1738.md 14; OSM extract `australia-oceania/new-zealand` of 2026-11-03 |
| Brunei | gad, one zone of the whole country (21 geoBoundaries ADM2 mukim, 7 empty ones dead space), shipped with Malaysia (T-1738) | D | notes/T-1738.md 11; OSM extract `asia/malaysia-singapore-brunei` of 2026-11-02 |
| India | mum; the whole country from the world plan (T-2111): 73 metro rows (ned New Delhi 36.7M, kol Kolkata 26.5M, ben chn haj hyd ...; GHS-UCDB counts north Bihar's and eastern Uttar Pradesh's village plains and Kerala's coast as urban centres: haj maj mrh meh bib gya, koz koc koa pon thi) and 709 zones of geoBoundaries ADM3 sub-districts (rule `IND`: floor 1M, cap 2.5M, `contiguous`); 149 units dead space (1.53M: Ladakh, Arunachal, the Thar); all 782 shipped (relaxed: the UCDB rows' FUA ratio, ned's and kol's top job cell, one-town zones' jobs, 13 hill and Sundarbans drive factors, kdk, cpu; bwl and bmn origins moved) | C, curve from bands (mum); C for the plan rows with the census hook off (`india.plan_hooks`, T-2078), D in effect | notes/T-292.md, notes/T-2111.md; OSM extracts `asia/india/*-zone` of 2026-11-03 |
| Nigeria | lag owe aba abo | D | notes/T-459.md; owe aba abo (plan rows, shipped) share Lagos's country module and its informal jobs blend; OSM extract of 2026-10-17 (notes/T-726.md) |
| Spain | mad bar val sev mal; the whole country as 40 zones (T-723: alc cat mur cad grn cod alr jae hue mll eiv zar biz gip ara asr tee gra laz aco poe our lug trr gir lle tod cid alb gul nav baj cax vaa len bug sla lar mea ceu); 9 provinces under 20 a km² are dead space | D | global sources only; zones on the zone curve (T-723); OSM extracts 2026-10-17 (mad bar), 2026-10-19 (val sev mal), 2026-10-20 (the zones). Not used yet: the transport ministry's mobile-phone trip tables and INE's 1 km census grid (notes/T-719.md; T-771) |
| Italy | mil rom nap tur bea (shipped); the whole country as 97 zones (T-779: every other province, Isernia merged into Campobasso; no dead space) | D | global sources only; zones on the zone curve (T-723); OSM extracts 2026-10-17 (metros), 2026-10-20 (the zones). National sources not looked for |
| Austria | vie (shipped); the rest of the country as 8 zones (T-788; Lungau dead space) | D | global sources only; OSM extract 2026-10-17 |
| Belgium | bru ant lie (shipped); the rest of the country as 11 zones (T-788: arrondissements grown within their province to 0.3M) | D | global sources only; OSM extract 2026-10-17 |
| Greece | ath the (shipped); the whole country as 31 zones (T-786: NUTS-3 units, small islands onto their region's largest zone, Evrytania and Fokida dead space) | D | global sources only; zones on the zone curve (T-723); OSM extracts 2026-10-17 (metros), 2026-10-20 (the zones). National sources not looked for |
| Poland | war kat kra poz lod (shipped); the rest of the country as 38 zones (T-788) | D | global sources only; OSM extracts 2026-10-17 (war kat), 2026-10-19 (kra poz lod) |
| Hungary | bud (shipped); the rest of the country as 8 zones (T-788) | D | global sources only; OSM extract 2026-10-17 |
| Netherlands | rot (shipped); ams (shipped); the rest of the country as 20 zones (T-788: COROP areas grown within their province to 0.3M) | D | global sources only; OSM extract 2026-10-16 for rot (a hand run, no download log: date from the cache file's time), 2026-10-19 for ams |
| Portugal | lis pot (shipped); 13 of 16 zones (T-786: NUTS-3 units with Madeira and the Azores; the Alentejo's four, Beira Baixa and Terras de Trás-os-Montes dead space); meo lez doo parked on their outlines (T-787) | D | global sources only; zones on the zone curve (T-723); OSM extracts 2026-10-17 (metros), 2026-10-20 (the zones). National sources not looked for |
| Russia | mos sai vor (shipped); 13 more rows held (Anita 2026-10-19; Ukraine's four and Minsk held too, no row until built) | D | global sources only (geoBoundaries ADM2 units); OSM extracts 2026-10-19 (central and north-western federal districts) |
| Sweden | sto (shipped); 5 zones (T-788; the north and the rural län dead space, listed in `world_plan.json`) | D | global sources only; OSM extract 2026-10-19 (sto) |
| Romania | buc (shipped); 24 zones (T-788; Tulcea dead space) | D | global sources only; OSM extract 2026-10-19 (buc) |
| Denmark | cop (shipped); 5 zones (T-788) | D | global sources only; OSM extract 2026-10-19 (cop) |
| Switzerland | zur (shipped); 8 zones (T-788) | D | global sources only; OSM extract 2026-10-19 (zur) |
| Serbia | beg (shipped); 5 zones (T-788) | D | global sources only; OSM extract 2026-10-19 (beg) |
| Norway | osl (shipped); 5 zones (T-788, replanned 2026-10-22; the north and the rural fylker dead space, listed in `world_plan.json`) | D | global sources only; OSM extract 2026-10-19 (osl) |
| Finland | hel (shipped); 4 zones (T-788, replanned 2026-10-22; the rural regions dead space, listed in `world_plan.json`) | D | global sources only; OSM extract 2026-10-19 (hel) |
| Bulgaria | sof (shipped); 6 zones (T-788) | D | global sources only; OSM extract 2026-10-19 (sof) |
| Czechia | pra (shipped); 9 zones (T-788) | D | global sources only; OSM extract 2026-10-19 (pra) |
| Ireland | dub (shipped); the rest of the island's south as 7 zones (T-788: NUTS-3 regions; OSM's counties put Sligo in the West's zone) | D | global sources only; OSM extract 2026-10-19 |
| Albania | tir (shipped); 2 zones (T-788; Gjirokastër dead space) | D | global sources only; OSM extract 2026-10-19 (tir) |
| Slovakia, Slovenia, Croatia, Bosnia and Herzegovina, Montenegro, North Macedonia, Kosovo, Moldova, Estonia, Latvia, Lithuania, Iceland, Malta, Cyprus, Luxembourg | zones only (T-788), shipped: SK 8, SI 2, HR 5, BA 6, ME 1, MK 3, XK 2, MD 4, EE 2, LV 1, LT 3, IS 1, MT 1, CY 1, LU 1 (lum); NUTS-3 units, geoBoundaries for Bosnia (ADM3) and Moldova (ADM1); units under 20 a km² dead space (Lika-Senj, many small Bosnian units, three of Estonia's five, three of Latvia's, Utena, Iceland outside the capital region; `world_plan.json` lists them) | D | global sources only; zones on the zone curve (T-723); OSM extracts from 2026-10-20 (Luxembourg's that day; the rest in the batch logs, notes/T-726.md) |
| Wales | car (shipped); zones in T-788 (England's row) | D | global sources only; England's travel rates, national no-car census (T-562); OSM extract 2026-10-17 |
| Scotland | gla (shipped); zones in T-788 (England's row; Highlands and Islands dead space) | D | as Wales; OSM extract 2026-10-17 |
| Northern Ireland | bel (shipped); zones in T-788 (England's row) | D | as Wales; OSM extract 2026-10-17 (about 08:20, the Ireland and Northern Ireland extract that the runner found already on disk) |

## Population and geography

OSM completeness c is T-456's: OSM errand and leisure places per 1,000 residents over 5 and 3,
capped at 1 (errands / leisure). It decides how far the everyday fallback trusts OSM.

| Country | Population | Boundaries | OSM completeness c |
|---|---|---|---|
| United States | U 2020 census blocks, institutions off (PL 94-171 P5); planned rows read the block .dbf only, each block's people spread by GHSL over the cells nearest its internal point (`usa`) | U TIGER counties | not needed (LODES sectors); OSM about half of employer shops |
| Canada | U 2021 dissemination areas (98-10-0015-01): tor, and the planned metro rows through `canada.plan_population` (DAs whose point is inside the outline, GHSL inside each); zones GHSL; N dissemination blocks, T-282 | U Esri Canada copies of StatCan files; G official files (www12 bot check), T-282 | 0.94 / 0.92 |
| England | U Census 2021 by LSOA (TS001), GHSL inside; skip output areas (LSOA already changed little, T-360) | U ONS ITL3 and LAD polygons | 0.98 / 1 (BRES in use) |
| France | U legal population by commune (geo.api.gouv.fr); N Filosofi 200 m squares, T-384 | U geo.api.gouv.fr communes | 1 / 1 |
| Germany | U Zensus 2022 100 m grid scaled to Gemeinde totals (VG250-EW) | U BKG VG250-EW | 1 / 1 |
| Japan | U 2020 census 500 m mesh | U KSJ N03 municipalities | Tokyo 0.56 / 0.68; other packs about 0.33-1 |
| Brazil | U IBGE 2022 census tracts | U IBGE tracts dissolved | 0.24 / 0.17 |
| China | U sha and every planned mainland metro row: 2020 census by township, the ASPECT 100 m grid (40,718 townships' 2020 counts, figshare 27323106) summed over 2018 township outlines in helper1m's table, copied to `data/raw/tier0/china_townships/townships.fgb`; GHSL's residential volume inside each (tier 0's bounded GHSL in cells outside any township); sha then held to its bulletin's district totals (T-1150; built packs take it only when rebuilt, until then sha by district and the metros GHSL alone); U the zones the same through their `china` module (T-1185), from their rebuild with T-1189's replan, GHSL alone until then; N 2020 census by county (github.com/leiii/census, MIT; Xinjiang from hongheiku), T-897 | U OSM admin_level 6; skip geoBoundaries ADM2 (1:4M); world plan units geoBoundaries ADM3 (holds Taiwan and Arunachal: clip, T-897) | 0.07 / 0.13 (Shanghai); Henan 0.01 / 0.01, Guangdong 0.05 / 0.07 (T-897) |
| South Korea | U resident registration by si-gun-gu (MOIS, open CSV) for seo; N the same for the plan rows, T-1243 (each city's provinces and renames mapped by hand); skip eup-myeon-dong (same file, little gain at 2 km); G 2020 census table (KOSIS: no account without a Korean ID; browser download only) | U OSM relations by Korean name | 0.51 / 0.93 |
| Australia | U ABS 2021 census G01 persons by SA1 (geo.abs.gov.au's `ABS_2021_Census_G01_SA1` layer, G01 on the ASGS Edition 3 SA1 polygons, CC BY 4.0; `australia.plan_population`, GHSL inside each SA1); N mesh blocks (counts only, no gain at 2 km); skip the DataPacks (same figures; abs.gov.au down for maintenance on the day) | U OSM relations matched to geoBoundaries ADM2 LGAs | not recorded |
| Vietnam | U GHSL alone; ? GSO 2019 census population by district (the completed-results volume; not opened, T-1738) | U OSM relations matched to geoBoundaries ADM2 districts | not recorded |
| Thailand | U GHSL alone; ? NSO 2020 registration and 2010 census by amphoe, not looked for | U OSM relations matched to geoBoundaries ADM2 amphoe | not recorded |
| Singapore | U GHSL alone; ? SingStat population by planning area, not looked for | U OSM relations matched to geoBoundaries ADM2 subzones | not recorded |
| Cambodia | U GHSL, scaled in the north-west (Banteay Meanchey, Battambang, Preah Vihear, Oddar Meanchey, Pailin: GHSL 3-10x the census in the Thai border districts) to the 2019 census by district from citypopulation.de (household population, NIS codes = COD-AB KHM ADM2 pcodes; `tools/country/cambodia_census.py` reads its admin page, each district equal to its communes) on COD-AB KHM ADM2 (2018; local copy religiondots/data/raw/kh): config `KH` of `pipeline/admin_totals.py`, partial (`only`), T-2000; the rest of the country on GHSL, since COD-AB lacks five districts made since and Tbong Khmum is renumbered (T-2107); N 2019 census by province with polygons (Open Development Cambodia `final_census2019_en`, CC BY-SA; WFS timed out to scripts); no open district table found (NIS microdata a 10% sample behind a download form, T-2075) | U OSM relations matched to geoBoundaries ADM2 districts | not recorded |
| Malaysia | U GHSL alone; ? DOSM 2020 census mukim totals in releases, no open mukim polygons found | U OSM relations matched to geoBoundaries ADM3 mukim (same-named mukim dissolved into one feature in the file: split, `world_plan.GB_SPLIT_FAR`), Sabah ADM2 | not recorded |
| Brunei | U GHSL alone; not looked for | U OSM relations matched to geoBoundaries ADM2 mukim | not recorded |
| Indonesia | U GHSL alone, about 2x the census across Papua (1.3-2.2x by zone, Yalimo 4.7x) and 1.2-1.7x in Riau (T-1738 12); N 2020 census (Long Form, mid-2022) by kabupaten with polygons, USCB's Indonesia gdb (ADM2 layer) and xlsx (Age-Sex) on HDX (CC BY 4.0, BPS code = COD-AB IDN pcode), in `data/raw/admin_totals/ID/`: config `ID` of `pipeline/admin_totals.py` (join checked: 514 units, 34 provinces exact; T-2084), U for Papua's seven zones rebuilt on it (yai puc pai mmi mru biu maw, T-2038; zone rule `IDN` names the module, so any row rebuilt later takes it), the rest still GHSL (T-2106); desa totals with polygons not found open | U OSM relations matched to geoBoundaries ADM2 kabupaten (far islands of a kabupaten split off, `world_plan.GB_SPLIT_FAR`) | not recorded |
| Philippines | U GHSL alone: 1.0-1.5x the 2020 census by province, but 0.1-0.55x in Muslim Mindanao (Basilan 0.10x, Tawi-Tawi 0.24x, Sulu 0.29x, Lanao del Sur 0.55x; T-2053); N 2020 census by barangay and municipality with polygons, USCB's Philippines gdb on HDX (CC BY 4.0, 10-digit PSGC; read in place from religiondots/data/raw/ph/philippines_uscb.gdb.zip): config `PH` of `pipeline/admin_totals.py` by municipality and city (ADM3, 1,634; barangays are often under a cell; join checked, 116 tabulation provinces exact; T-2084), U for pti mrw lbx ctb (T-2053; zone rule `PHL` names the module), the rest still GHSL (T-2106); skip COD-PS (88 plain provinces, double counts the HUCs) | U OSM relations matched to geoBoundaries ADM3 municipalities | not recorded |
| India | U Census 2011 by sub-district, grown by district to 2020 (mum); U GHSL alone for the plan rows; skip the same census by district (PCA, NADA catalog 6191) as shares within each state: 2.6% of people moved, mostly growth since 2011 where GHSL is the better figure (T-2078, T-2111; config `IN` of `pipeline/admin_totals.py` ready, one line in `india.plan_hooks` to switch on); 2027 census not out (enumeration February 2027, no results announced, T-2075) | U OSM admin_level 5 and 6 by name; skip Datameet 2011 districts with `censuscode` (CC BY 4.0, joined 640 of 640, clipped to the de facto border), T-2078; G SHRUG polygons (non-commercial, checks only) | 0.04 / 0.05 |
| Nigeria | U GHSL alone; - no credible count (census 2006); ? GRID3 100 m estimates, T-507 | U OSM LGAs | 0.02 / 0.01 (owe aba abo the same) |
| Spain | U GHSL alone (GHS-POP E2020 R2023A 100 m); N INE Census 2021 population on the 1 km Eurostat grid, found, not opened (T-719, T-771) | U OSM admin relations matched to NUTS-3 units (`osm_units.py`) | 0.81-0.95 / 0.94-1 |
| Italy | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 0.66-1 / 0.62-1 (Naples lowest) |
| Austria | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Belgium | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Greece | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 0.98-1 |
| Poland | U GHSL alone; N Eurostat's Census 2021 1 km grid, T-2083 | U OSM relations matched to NUTS-3 units | 1 / 0.65-1 (Katowice, Lodz lowest) |
| Hungary | U GHSL alone; N Eurostat's Census 2021 1 km grid, T-2083 | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Netherlands | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Portugal | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Russia | U GHSL alone; ? Rosstat's 2021 census (VPN-2021) by municipal district and okrug on rosstat.gov.ru, not opened (T-2297; it needs a polygon file with OKTMO codes, which geoBoundaries ADM2 lacks) | U OSM relations matched to geoBoundaries ADM2 units | 1 / 0.50-0.82 |
| Sweden | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 0.81 / 1 |
| Romania | U GHSL alone; N Eurostat's Census 2021 1 km grid, T-2083 | U OSM relations matched to NUTS-3 units | 0.99 / 0.64 |
| Denmark | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Switzerland | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Serbia | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 0.77 |
| Norway | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 0.93 / 0.99 |
| Finland | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Bulgaria | U GHSL alone; N Eurostat's Census 2021 1 km grid, T-2083 | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Czechia | U GHSL alone; N Eurostat's Census 2021 1 km grid, T-2083 | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Ireland | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 1 |
| Albania | U GHSL alone; ? not looked for | U OSM relations matched to NUTS-3 units | 1 / 0.98 |
| Wales | U GHSL alone; ? Census 2021 by LSOA, as England's packs use, not opened for Wales | U OSM relations matched to ONS ITL3 units | 1 / 1 |
| Scotland | U GHSL alone; ? Census 2022 not looked for | U OSM relations matched to ONS ITL3 units | 0.79 / 0.87 |
| Northern Ireland | U GHSL alone; ? NISRA Census 2021 not looked for | U OSM relations matched to ONS ITL3 units | 0.81 / 0.85 |
| Slovakia, Slovenia, Croatia, Bosnia and Herzegovina, Montenegro, North Macedonia, Kosovo, Moldova, Estonia, Latvia, Lithuania, Iceland, Malta, Cyprus, Luxembourg (zones only) | U GHSL alone; N for the EU members Eurostat's Census 2021 1 km grid, T-2083 (T-2075); others not looked for | U OSM relations matched to NUTS-3 units (geoBoundaries ADM3 in Bosnia, ADM1 in Moldova) | not recorded |
| New Zealand | U GHSL scaled to the 2023 census usually resident population by SA2 (Stats NZ ArcGIS services, CC BY 4.0; the clipped layer's local copy religiondots/data/geo/nz/sa2_2023_clipped.geojson, 2,311 SA2s): config `NZ` of `pipeline/admin_totals.py`, all six rows rebuilt (T-2077; GHSL alone was 0.86-1.09x by zone, 0.68-1.41x by TA); N SA1 (about 150 people) in the same services, not needed at SA2's 2,000 | U OSM relations matched to geoBoundaries ADM3 wards; N the same services' SA1/SA2 polygons | not recorded |
| Turkey | U ADNKS register 31 Dec 2022 by ilce plus COD-PS's table of Syrians under temporary protection by ilce (COD-PS TUR, HDX, CC BY-IGO; 88.78M), the earthquake provinces Malatya, Hatay, Kahramanmaras and Adiyaman scaled to their 2023 register totals (citypopulation.de's admin page; x0.91-0.95), `admin_totals` config TR (`tools/country/turkey_census.py`, T-2079); skip TUIK's portals (no table API); - mahalle polygons (none open) | U COD-AB TUR ADM2 ilce by pcode (973 of 973) for the totals; U geoBoundaries ADM2 ilce for the plan units, matched to OSM relations | - |
| Mexico (not built) | N 2020 census by municipio, locality (ITER) and urban AGEB (INEGI open terms), T-2080 (T-2075) | N INEGI Marco Geoestadistico 2020 by CVEGEO | - |
| Israel (not built; Middle East pick) | N 2022 census by statistical area (CBS, data.gov.il), T-2082 (T-2075) | N CBS's own statistical-area polygons | - |
| Saudi Arabia (not built) | ? 2022 census by region only (COD-PS SAU, 13 regions); GHSL alone would do (T-2075) | - COD-AB to region only | - |
| Ukraine (held) | - nothing open and current (census 2001; COD-PS UKR 2026 is marked restricted) | ? COD-AB UKR (hromada) | - |
| Belarus | U GHSL alone; ? 2019 census by raion in Belstat PDF bulletins (T-2075, not taken: GHSL is sound in Belarus and a PDF table is past the playbook's hour) | U OSM relations matched to geoBoundaries ADM2 raions; ? COD-AB BLR (raion) | - |

## Jobs

| Country | Jobs at work | Jobs by industry (everyday destinations) | Residents by industry |
|---|---|---|---|
| United States | U LODES WAC, block (planned rows: the newest year from 2023 back to 2016, Michigan 2021, Alaska 2016; none for Puerto Rico, tier 0's proxy) | U LODES WAC sectors, block | N LODES RAC sectors, block, T-500 |
| Canada | U 2021 census workers by tract of work (98-10-0498-01), lockdown at-home workers put back | U 98-10-0492-01, tract x sector (leisure; errands fail the concentration test) | ? census industry by tract of residence, T-500 |
| England | U BRES 2024, LSOA | U BRES by LSOA x SIC division | N Census 2021 industry by LSOA of residence (TS060), T-500 |
| France | U census flows (MOBPRO 2022) summed by commune of work | N Flores 2021, commune x A38 (trade mixed with wholesale), T-503 | ? MOBPRO detail file, T-500 |
| Germany | U Pendlerrechnung 2024 by Gemeinde of work (Berlin one unit; - no Bezirk table found; a suppressed count, "-", counts as 0, T-613) | - Regionaldatenbank 13111 has 4 groups by Gemeinde, too coarse | ? not looked for |
| Japan | U 2021 economic census 500 m mesh, scaled to census workers by municipality of work (public service left out, T-461) | U the same mesh by major industry, 9-1B splits by municipality | N 2020 census industry by municipality of residence, T-500 |
| Brazil | U OD 2017 survey, workers by zone of first job | ? survey's job sector, not checked | ? survey's job sector, not checked |
| China | U sha: 5th economic census by district (legal-person units; district tables only in some cities' bulletins); the Jiangsu rows the tier 0 proxy (row below); N its province totals (national bulletin no. 7, table 7-2), T-897 | ? economic census by district x sector, T-503 | - |
| South Korea | tier 0 proxy; G Census on Establishments by si-gun-gu (KOSIS: no account without a Korean ID; browser download only), T-504 | G the same census by sector, T-504 | - |
| India | tier 0 proxy (population 0.5); N 6th Economic Census 2013 by ward, T-388 | - | - |
| Australia | tier 0 proxy; skip 2021 census place of work (taken in the Sydney, Melbourne and Canberra lockdowns, home workers coded at home); N 2016 census Working Population Profile by destination zone (DataPacks, CC BY 4.0), P3 task to file (T-1857) | U tier 0 | - |
| Vietnam | tier 0 proxy (`plan_config.job_blend` from the ILO vulnerable-employment share); ? GSO economic census 2021 by district not looked for | U tier 0 | - |
| Thailand, Singapore, Cambodia, Malaysia, Brunei, Indonesia, Philippines | tier 0 proxy (`plan_config.job_blend`); not looked for | U tier 0 | - |
| Nigeria | tier 0 proxy (population 0.5, land cap) | - | - |
| New Zealand | U tier 0 proxy (`plan_config.job_blend`); N Business Demography employee count by SA2 (Stats NZ ArcGIS), T-2077; ? 2023 census travel to work by SA2 (datafinder, free account) | ? not looked for | ? not looked for |
| Mexico (not built) | N INEGI DENUE, every economic unit with coordinates and an employee band, T-2081; ? Censos Economicos 2019 by AGEB; ? census 2020 extended-questionnaire municipio of work | ? DENUE by SCIAN sector | ? not looked for |
| Taiwan | tier 0 proxy, its jobs per resident and share aged 0-14 from DGBAS's 2024 manpower survey and the Ministry of the Interior's end-2024 registration (`tier0.NATIONAL`: the World Bank has no Taiwan), T-897; ? jobs by township not looked for | U tier 0 | - |
| Spain, Italy, Austria, Belgium, Greece, Poland, Hungary, Netherlands, Portugal, Russia, Belarus, Turkey, Sweden, Romania, Denmark, Switzerland, Serbia, Norway, Finland, Bulgaria, Czechia, Ireland, Albania, Wales, Scotland, Northern Ireland, the zones-only countries (Population above), China's Jiangsu rows | tier 0 proxy: population x jobs per resident (World Bank employment ratio, 2025), spread by GHSL built volume, OSM job places and population in the weights `plan_config.job_blend` sets from the ILO vulnerable-employment share (30/50/20 in most, Albania 21/33/46); ? national jobs by small area not looked for | U tier 0: OSM shops and services, restaurants, bars, venues and hotels, blended with the jobs at completeness c | ? not looked for |

## Commuting

Work at home: people whose workplace is their home, every working day. Telework: part-week home
days of people with a workplace. Several countries have one figure for both.

| Country | Commute flows (the curve) | Travel survey | Work at home | Telework |
|---|---|---|---|---|
| United States | U LODES OD, fitted (planned rows take the same curve) | U NHTS 2017 and 2022: weekend factors, medical rates, long trips | U ACS by tract and industry; planned rows ACS 2019-23 B08301 over the row's counties, one share (Puerto Rico too) | U in ACS's figure |
| Canada | U 2021 census flows between municipalities (98-10-0459-01, lockdown), fitted; N 2016 flows for a refit, T-282 | ? Toronto's TTS, not checked | U LFS May 2024, one share (24.7%) | U in the LFS share |
| England | U 2011 census flows (WU01EW), MSOA, fitted; 2021 flows lockdown, checks only | G National Travel Survey microdata (UK Data Service licence), not checked | - (Census 2021 lockdown) | U ONS OPN 2024, one share (0.27) |
| France | U MOBPRO 2022 commune flows, fitted | ? Ile-de-France EGT, not checked | U Eurostat LFS usually at home | U LFS sometimes (0.12 in all) |
| Germany | U Pendleratlas, ten largest flows per Gemeinde, fitted | G MiD 2017 microdata (application), not checked | U Eurostat LFS usually at home | U LFS sometimes (0.10 in all) |
| Japan | U 2020 census table 4, municipality flows, fitted; joint gravity across packs | U 2018 Tokyo person trip survey press tables (everyday rates, Travel surveys below) | U census table 4, share by municipality of work | U MLIT 2024 telework survey by region |
| Brazil | - no census flows; U OD 2017 home-to-job distances, fitted | U Metro OD 2017 microdata; its everyday rates are T-450's evidence | U survey (9.1%) | in the survey |
| China | - no open flows; U sha: published mean commute (10.2 km, Shanghai survey via press); U the Jiangsu rows: the borrowed curve (metros), the zone curve (zones); N CAUPD-Baidu report's means for about 45 cities, T-897 | U the survey's mean only | U 0 | U 0 |
| South Korea | G 2020 census commuting by si-gun-gu (KOSIS: no account without a Korean ID; browser download only), T-504; U mobile-data mean commute (Statistics Korea, 20.4 km) | ? KTDB household survey, not checked | U 0 (no figure checked) | U 0 |
| India | U Census 2011 B-28 distance bands by district, both parameters fitted | - | U B-28 "no travel" (20.1%) | - |
| Australia | G 2021 census journey to work by SA2 (TableBuilder, free account); U the borrowed curve (T-314) for the metros, the zone curve (T-723) for the zones | ? state household travel surveys (Sydney HTS, VISTA), not checked | U 0; ? ABS working arrangements (36% regularly at home, August 2023), not a daily share | U 0 |
| Vietnam | ? census migration and commuting tables not looked for; U the borrowed curve (T-314) for the metros, the zone curve (T-723) for the zones | ? not looked for | U 0 | U 0 |
| Thailand, Singapore | ? not looked for; U the borrowed curve for ban and sin, the zone curve for Thailand's zones | ? not looked for | U 0 | U 0 |
| Cambodia, Malaysia, Brunei | ? not looked for; U the borrowed curve for phn, kua and joo, the zone curve for the zones; Johor Bahru to Singapore about 250,000-400,000 workers a day (news estimates), the border factor 0.5 (borders.json) | ? not looked for | U 0 | U 0 |
| Indonesia | ? BPS commuter survey (SAKERNAS, Jabodetabek commuter survey 2019) not looked for; U the borrowed curve for the 22 metros, the zone curve for the zones; borders with Malaysia (Kalimantan) and Singapore (Batam) at the national 0.05 | ? not looked for | U 0 | U 0 |
| Philippines | ? PSA census place of work not looked for; U the borrowed curve for mai ceb ang dav, the zone curve for the zones; no land border (Sabah within 50 km of Tawi-Tawi, national 0.05) | ? MUCEP 2014 (Metro Manila, JICA) not looked for | U 0 | U 0 |
| Nigeria | - none; U the borrowed curve (T-314) | - (one 2010 paper's mean trip, too weak to fit) | U 0 | U 0 |
| Spain | N the transport ministry's "Estudio de movilidad con Big Data" (daily trips between 3,792 districts with home and work ends, open CSV, licence not read; T-719, T-771); U the borrowed curve (T-314) for the metros, the zone curve (T-723) for the zones | ? not looked for | U 0; N Eurostat LFS lfsa_ehomp, T-765 | U 0, as work at home |
| Italy, Austria, Belgium, Greece, Poland, Hungary, Netherlands, Portugal, Russia, Belarus, Turkey, Sweden, Romania, Denmark, Switzerland, Serbia, Norway, Finland, Bulgaria, Czechia, Ireland, Albania, Wales, Scotland, Northern Ireland, the zones-only countries (Population above) | ? national flows not looked for; U the borrowed curve (T-314) for the metros, the zone curve (T-723) for the zones | ? not looked for | U 0; N Eurostat LFS lfsa_ehomp where it has the country (as Germany's and France's), ONS OPN 0.27 for the three UK nations, T-765 | U 0, as work at home |

## Car ownership

Not used since T-854 (Anita, 2026-10-21): demand reads no car ownership; dense homes pay parking at
home instead (demand-mode.md 4.3). The rows below are what `pipeline/nocar.py` wrote into the packs
as `nocar` (people living in a household with no car, per cell, T-511), kept on disk unread, should
the input come back; countries added since (the zones-only countries, China's Jiangsu rows) have no
row here. Where only cars per household are known, households without a car came from
one curve fitted on US tracts and English LSOAs. Every country had a figure (T-562): the rows below,
then the three rule rows at the end in that order. No open world register exists: WHO's GHO counts
all registered vehicles with motorcycles (2016, one total), the World Bank's IRF series stopped
about 2011, OICA's vehicles-in-use pages are gone (404) with no licence stated.

| Country | Homes without a car |
|---|---|
| United States | U ACS 2019-23 B08201 by tract, people in no-vehicle households (nearest populated block's tract) |
| Canada | U StatCan light-duty registrations 2023 per 1,000 x census household size, national; - census asks no vehicle question |
| England | U Census 2021 TS045 by LSOA (households, made people) |
| France | U Eurostat passenger cars per 1,000 (2024) x household size, national; N INSEE census housing base by commune or IRIS, T-512 |
| Germany | U Eurostat passenger cars per 1,000 (2024) x household size, national |
| Japan | U AIRIA private cars per household by prefecture (March 2025); - no small-area figure found |
| Brazil | U Senatran automóveis per 1,000 (2024) x household size, national; ? OD 2017 survey's car ownership, T-512 |
| China | U Ministry of Public Security cars (2024) per 1,000 x household size, national; ? Shanghai yearbook cars per 100 households, T-512 |
| South Korea | U MOLIT passenger cars per 1,000 (2024) x household size, national; G KOSIS small areas, T-504 |
| India | U MoRTH cars, jeeps and taxis per 1,000 (2022) x household size, national |
| Nigeria | U NBS private vehicles per 1,000 (2018) x household size, national; N DHS 2024 by state, T-570 |
| Wales | U Census 2021 TS045 by LSOA (Nomis, households, made people) |
| Scotland | U Census 2022 UV405 by data zone 2022 (NRS topic tables zip; maps.gov.scot boundaries; OGL) |
| Northern Ireland | U Census 2021 car or van availability by data zone 2021 (NISRA ArcGIS layer, counts people in households) |
| Spain | U Eurostat rule 2: 550 cars per 1,000 x 2.5 people per household (2025) |
| Italy | U Eurostat rule 2: 709 x 2.2 |
| Austria | U Eurostat rule 2: 574 x 2.2 |
| Belgium | U Eurostat rule 2: 513 x 2.2 |
| Greece | U Eurostat rule 2: 597 x 2.4 |
| Poland | U Eurostat rule 2: 639 x 2.8 |
| Hungary | U Eurostat rule 2: 461 x 2.3 |
| Netherlands | U Eurostat rule 2: 519 x 2.1 |
| Portugal | U Eurostat rule 2: 561 x 2.4 |
| Russia | U rule 4: World Bank GDP per head PPP (42,358 in 2025) by the logit fit; no register or survey found |
| Sweden | U Eurostat rule 2: 475 x 2.0 |
| Romania | U Eurostat rule 2: 461 x 2.5 |
| Denmark | U Eurostat rule 2: 501 x 1.9 |
| Switzerland | U Eurostat rule 2: 543 x 2.2 (household size 2024) |
| Serbia | U Eurostat rule 2: 394 x 2.6 |
| Norway | U Eurostat rule 2: 524 x 2.0 |
| Finland | U Eurostat rule 2: 671 x 1.9 |
| Bulgaria | U Eurostat rule 2: 507 x 2.2 |
| Czechia | U Eurostat rule 2: 621 x 2.3 |
| Ireland | U Eurostat rule 2: 475 x 2.7 |
| Albania | U rule 1: DHS AL2017DHS, 37.0% of households possess a private car |
| Hong Kong, Singapore, Taiwan | U Transport Department, LTA, Highway Bureau private cars per 1,000 x household size, national (the GDP rule cannot hold for city states; Taiwan has no World Bank figures) |
| Rule 1: DHS countries | U DHS API (open, no registration) HC_TRNS_H_CAR, households possessing a private car, national, latest survey from 2015 (rule 3: from 2010), 51 planned countries; N the same by region, T-570 |
| Rule 2: Eurostat countries | U road_eqs_carhab cars per 1,000 x ilc_lvph01 household size, latest year (CC BY 4.0); 20 planned countries |
| Rule 4: everything else | U households with a car from World Bank GDP per head PPP (NY.GDP.PCAP.PP.KD), logit fit (notes/t562_fit.py); no GDP (Cuba, North Korea, Venezuela): the World Bank income group's median; 46 planned countries, 116 rows; N census or register figures for the larger ones, T-571 (Mexico's INEGI 2020 car availability by municipality among them) |

## Travel surveys

Everyday outing rates by purpose (T-509 stage 4, `pipeline/everyday_rates.py`); a country without
one takes the surveys' median. Trip lengths are one worldwide set (T-637), so a survey's mean
lengths are not used; its share of trips under 2 km is the check (docs/demand-worldwide.md).

| Country | Rates | Lengths |
|---|---|---|
| United States | U NHTS 2017 microdata, weekday legs by purpose (on disk); N NHTS 2022 | U NHTS New York CBSA legs by length band, the check behind the worldwide set (T-637) |
| Canada | - (TTS is Toronto-only, not opened); worldwide | worldwide |
| England (UK) | U NTS0504b weekday trips by purpose, 2015-19; G NTS and LTDS microdata (UKDS) | U LTDS 2024/25 London residents' trips by length band (Travel in London 2025), the check (T-637); skip NTS0413 mean errand lengths |
| France | U EMP 2019 weekday motifs table (SDES); N EMP microdata (open, media/5052) for medical; G EGT 2018-20 (omnil.fr 403) | skip EMP declared km (means; T-637) |
| Germany | U MiD 2017 Ergebnisbericht by Zweck, made weekday; G MiD tables tool (login) | skip SrV 2018 Berlin Steckbrief (means; T-637) |
| Japan | U Tokyo PT 2018 press tables (private split); U national survey 2021 (MLIT Excel) as a check; G Tokyo PT tabulations (application form, login) | skip Tokyo welfare bureau 2021 straight-line km by purpose (means; T-637) |
| Brazil | U Sao Paulo OD 2017 microdata | skip the same (means; T-637) |
| China | -; worldwide | worldwide |
| South Korea | U KTDB 2016 national O/D report, capital-area sample counts; G KTDB 2021 survey (account) | skip Gyeonggi transport DB 2016 bands (T-637) |
| India | -; worldwide | worldwide |
| Nigeria | -; worldwide | worldwide |
| Vietnam | ? not looked for; worldwide (the surveys' median) | worldwide |
| Australia | ? Sydney HTS and Victoria's VISTA, not checked; worldwide | worldwide |
| Thailand, Singapore | ? not looked for (Singapore's LTA Household Interview Travel Survey publishes summaries only); worldwide | worldwide |
| Cambodia, Malaysia, Brunei, Indonesia, Philippines | ? not looked for; worldwide | worldwide |
| Spain, Italy, Austria, Belgium, Greece, Poland, Hungary, Netherlands, Portugal, Russia, Belarus, Turkey, Sweden, Romania, Denmark, Switzerland, Serbia, Norway, Finland, Bulgaria, Czechia, Ireland, Albania, the zones-only countries (Population above) | ? not looked for; worldwide (the surveys' median) | worldwide |
| Wales, Scotland, Northern Ireland | U England's rates, NTS0504b (country GB) | worldwide |

## Destinations

| Country | Schools | Universities | Medical | Airports |
|---|---|---|---|---|
| United States | U NCES CCD public schools (planned rows OSM until T-1019); N NCES PSS private, T-142 | U IPEDS (planned rows OSM until T-1019) | U OSM (relations skipped, T-203) | U FAA enplanements, DB1B connecting share (planned rows Wikidata until T-1019) |
| Canada | U Ontario school file (per province) | U COU and Ontario college files (per province); N partner campuses, T-282 | U OSM | U StatCan 23-10-0253-01; connecting by judgement |
| England | U GIAS | N HESA Table 1 by provider, T-382 | U OSM | U Wikidata; Heathrow 0.7 by judgement |
| France | N ministry directory with pupils, T-385 | ? ministry students by site, T-505 | U OSM | U Wikidata; CDG 0.75 by judgement |
| Germany | U tier 0 OSM (school files per Land) | ? Destatis 21311 by institution, T-505 | U OSM | U Wikidata |
| Japan | U KSJ P29 points, mesh children 0-14 | U tier 0 (short in some zones, T-462) | U OSM, beds almost never tagged (sized by rule, T-387) | U Wikidata; Narita 0.8 by judgement |
| Brazil | U tier 0 OSM; ? INEP school census, T-506 | U tier 0 | U OSM; ? CNES beds by establishment, T-506 | U GRU via the press, Congonhas Wikidata |
| China | U tier 0 OSM | U tier 0 | U OSM, one hospital with beds (sized by rule, T-387) | U sha: CAAC bulletin via the press; the Jiangsu rows tier 0 (OurAirports, Wikidata patronage); N the bulletin itself, every airport (2025 edition: 270), T-897 |
| South Korea | U tier 0 OSM | U tier 0 (3 of 217 sites with Wikidata counts) | U OSM, no beds tagged (sized by rule, T-387) | U Wikidata; Incheon 0.85 by judgement |
| India | U tier 0 OSM; G UDISE+ enrolment (behind the portal) | U tier 0 | U OSM, one hospital with beds (sized by rule, T-387) | U the operator's total via the press |
| Vietnam | U tier 0 OSM (pupils = population x World Bank 0-14 share x 0.9) | U tier 0 | U OSM, beds by the World Bank rule below | U OurAirports sites with Wikidata patronage, O&D share 1 |
| Australia | U tier 0 OSM; ? ACARA school locations with enrolments, not looked for | U tier 0 | U OSM, beds by the World Bank rule below | U OurAirports sites with Wikidata patronage, O&D share 1 |
| Thailand, Singapore | U tier 0 OSM | U tier 0 | U OSM, beds by the World Bank rule below | U OurAirports sites with Wikidata patronage, O&D share 1 (Changi and Suvarnabhumi are hubs; no connecting share set) |
| Cambodia, Malaysia, Brunei | U tier 0 OSM | U tier 0 | U OSM, beds by the World Bank rule below | U OurAirports sites with Wikidata patronage, O&D share 1 (KLIA is a hub; no connecting share set) |
| Indonesia, Philippines | U tier 0 OSM | U tier 0 | U OSM, beds by the World Bank rule below | U OurAirports sites with Wikidata patronage, O&D share 1 |
| Nigeria | U tier 0 OSM; ? GRID3 schools, T-507 | U tier 0 (World Bank rate from 2011) | U OSM, no beds; ? GRID3 health facilities, T-507 | U FAAN via the press |
| Spain, Italy, Austria, Belgium, Greece, Poland, Hungary, Netherlands, Portugal, Russia, Belarus, Turkey, Sweden, Romania, Denmark, Switzerland, Serbia, Norway, Finland, Bulgaria, Czechia, Ireland, Albania, Wales, Scotland, Northern Ireland, the zones-only countries (Population above) | U tier 0 OSM (pupils = population x World Bank 0-14 share x 0.9); ? national school registers not looked for | U tier 0 (Wikidata students where tagged, scaled to the World Bank tertiary rate) | U OSM, beds by the World Bank rule below | U OurAirports sites in the pack with Wikidata patronage, O&D share 1 (Tirana: none inside the pack) |

Everywhere: places and parks from OSM, water from OSM (TIGER and StatCan in the US and Canada),
world market from the packs. U hospital beds per 1,000 people from the World Bank (SH.MED.BEDS.ZS,
CC BY 4.0, the country's latest year, cached in `pipeline/hospital_beds.json`; none for Taiwan,
which takes the world figure) scale every pack's hospital beds (T-387). U attractions from
Wikidata through QLever (CC0; visitors P1174 where given; one world file,
`pipeline/attractions_json.py`, T-263): figures for 1,273 of 12,682 places, most in the US and
Western Europe (China 21, Japan 24, India 19, Nigeria none); the rest sized by a budget shared by
U Wikipedia page views (Wikimedia's monthly pageview dump, September 2026, CC0; English plus the
country's editions; the per-article APIs allow a client naming no contact 10 requests a minute) and
U the World Bank's international arrivals (ST.INT.ARVL) and population, 2019 (CC BY 4.0; Taiwan's
from UNWTO and its Ministry of the Interior), T-1766, cached in `data/work/t1766/`; N national
tourism boards' visitor tables, never looked for. ? Overture Places as a second place source where c is under 0.3
(Brazil, China, India, Nigeria), T-508. Tier 0 alone covers any input marked tier 0 above; its rules are in
demand-worldwide.md "Tier 0".

## Every country: the world plan

Which metros get a pack and which units each holds (`pipeline/world_plan.py`, T-536; the rule in
world-run.md "The world plan"). Downloads in `data/raw/worldplan/`, fetched when missing.

| Input | Source | Terms |
|---|---|---|
| Which cities | U GHSL Urban Centre Database R2024A, 2025 population (11,422 centres), from `citybrowser/data/` | EC reuse with acknowledgement |
| Commuting areas | U GHS-FUA R2019A (`GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0`, 9,031 areas, 2015 epoch; still the latest release, checked 2026-10-16) | "Reuse of this data is authorised with proper acknowledgment of the source" (GHSL datasets page; EC reuse policy) |
| People | U GHS-POP E2025 1 km, Mollweide (R2023A) | as above |
| Units, Europe | U NUTS-3 2024 (Eurostat GISCO, 1:3M; EU, EFTA, candidates); U ONS ITL3 January 2025 for the UK | NUTS: "© EuroGeographics for the administrative boundaries"; non-commercial and non-transferable (read in T-924), so NUTS decides grouping and names only and no NUTS polygon ships: where OSM leaves a NUTS-3 unit uncovered, `osm_units.py` fills it from geoBoundaries (T-925) |
| Outline fills, Europe | U geoBoundaries gbOpen units mostly inside the NUTS-3 unit, the finest level with an open licence (T-925): DE ADM3 (BKG, dl-de/by-2-0), ES ADM3 (IGN, CC BY 4.0; ADM2's INE licence left out), NL ADM2 (CC0), FI ADM3 (ODbL), GR ADM3 (CC0), PT ADM3 (CC0); other countries by the same licence test (`FILL_LICENCES`) when a rebuild needs one | as listed; credited in Settings > Data sources |
| Units, elsewhere | U geoBoundaries gbOpen, simplified, ADM1-4 (the level per country is in `world_plan.json` `unit_sources`, with each file's licence: CC BY, ODbL or public domain) | per country; GADM ruled out (non-commercial) |
| Country of a centre | U Natural Earth admin 0, 1:10M | public domain |
| Extracts | U Geofabrik `index-v1.json`; fetched as `-latest` when a region is read and deleted after, so a row's extract date is its download day (batch logs under `artifacts/trainworld/world_run/`; list in notes/T-726.md) | ODbL (OSM) |
| Time zones | U timezonefinder 6.6 (offline, timezone-boundary-builder) | MIT; data ODbL |
| Outline of a planned city | U OSM `boundary=administrative` relations from the row's Geofabrik extracts, matched to the plan's units (`osm_units.py`, T-537) | ODbL |
| Informal work (jobs blend) | U ILO modelled vulnerable employment, World Bank SL.EMP.VULN.ZS (T-537); - informal employment SL.ISV.IFRM.ZS, archived by the World Bank | CC BY 4.0 |
| Terrain classes (track prices) | U GEDTM30 v1.2 at 30 m (OpenGeoHub, `s3.opengeohub.org/global/dtm/v1.2/`), read once on AWS into `data/work/terrain-fine/` (`terrain_fine.py`, T-1854); bare earth, but towers and raised roads remain (tower fields cut with G GHSL GHS-BUILT-V E2020 at 30 arc seconds, `terrain_fine.py`, T-1881; then capped, track-and-service.md 6.4); the 240 m file for `market_drives.py`. Ruled out for the towers (T-1881): FABDEM (CC BY-NC-SA), Copernicus GLO-30 (a surface model), GEDTM30's own std layer and GHSL's building heights (neither tells Midtown from a built hill) | CC BY 4.0 (GEDTM30, GHSL); the class files ship outside the ODbL packs |
| Water between zones | U the world's coarse water mask (`pipeline/world_water.py`, T-1285): OSM coastline water polygons (osmdata.openstreetmap.de, the same file as the packs' sea) and Natural Earth 1:10m lakes, at 1/60 degree | ODbL (OSM); Natural Earth public domain |

## Archetypes

Four procedures are in use. Each names what it needs; everything else comes from tier 0.

- **A. Census flows.** Needs a commute flow table between small units (municipalities or finer)
  and jobs at work by the same or finer units. Jobs from the table (or an employer register), the
  curve fitted on the flows with the pack's own margins. Grain varies: blocks, tracts, LSOAs or
  500 m meshes (US, Canada, England, Japan), communes or Gemeinden (France, Germany, where a big
  city is one or a few units and tier 0's proxy shapes the jobs inside it). Worked examples:
  Toronto, London, Paris, Berlin, Tokyo.
- **B. Travel survey.** Needs a household travel survey's microdata with home and job zones. Jobs
  by survey zone scaled to the census, work at home from the survey, both curve parameters fitted
  to home-to-job distances. Population from census small areas. Worked example: Sao Paulo.
- **C. District totals.** Needs population (and ideally jobs) by district from a census or
  register, and a published commute figure: a distance-band table (fit both parameters, India) or
  a mean or median (power 1.4, length fitted; China, South Korea). Without a jobs table, tier 0's
  proxy with a larger population share where work is informal. Worked examples: Shanghai, Mumbai,
  Seoul.
- **D. Global only.** Nothing national is credible or open. GHSL population unscaled, the jobs
  proxy with the land cap and its blend set from the ILO vulnerable-employment share (T-537;
  Lagos's population share 0.5), a borrowed curve (`"curve": "borrowed"`, `tier0.borrowed_curve`,
  T-314; zones `borrowed_zone`, T-723), work at home 0. Worked example: Lagos.

**Deciding for a new country.** Fill its row in every table first (an hour of searching at most;
look for mirrors when an official server walls, and never take a source from memory). Then take
the first that holds:

1. A commute flow table between units no bigger than municipalities, open: **A**. Check the
   jobs column next: jobs at work by small area keep A's full grain; otherwise jobs from the flows'
   destination totals (Paris).
2. A household travel survey's microdata for the metro, open: **B** (even if the census is good:
   one file gives jobs, work at home and the curve).
3. Population by district from a census or register within about ten years, and some published
   commute figure: **C**. Take jobs by district if any table has them (China's economic census).
4. Otherwise **D**.

Then work down the columns: residents and jobs by industry, schools, universities, medical and
airports each take the national source if it is U-grade elsewhere, else tier 0. Follow the
worked example's paragraph in adding-a-city-by-hand.md "Outside the US" for the chain of commands.
A gated source does not change the archetype until it is in hand; it goes in its cell with the
task that would use it.
