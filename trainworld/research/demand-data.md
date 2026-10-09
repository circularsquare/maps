# Demand data by country

Research report, 2026-10-08. Nothing downloaded. "Unverified" = not confirmed in this pass,
mostly from memory; check before building on it.

## By country (jobs / OD / grain / access)

- **USA**: jobs LODES 8.4 WAC; OD LODES OD; 2020 census block; 2002-2023 (2023 lacks Alaska
  and Michigan); free CSV, no registration. lehd.ces.census.gov/data/#lodes
- **UK (England & Wales)**: jobs from Census 2021 workplace (WP) and workday (WD) population
  tables down to output area. OD flows on Nomis, mostly local-authority level, some at MSOA;
  OA-level flows are "safeguarded" at the UK Data Service (End User Licence, usually an
  institutional account). Nomis free. nomisweb.co.uk;
  ons.gov.uk/census/aboutcensus/censusproducts/origindestinationflowdata. Scotland (2022)
  unverified; Northern Ireland has an OD page at NISRA.
- **Japan**: jobs from Economic Census for Business Activity 2021 mesh statistics, 1 km and
  500 m, workers by industry, free on e-Stat. OD: 2020 census place-of-work tables,
  municipality to municipality, free on e-Stat (unverified this pass, known to exist).
  Metropolitan person-trip surveys (Tokyo PT) are finer (unverified).
- **South Korea**: jobs grid on SGIS (100 m to 1 km) by data request, probably needs a Korean
  identity login (unverified). OD: 2020 census commuting at si/gun/gu on KOSIS and the Seoul
  portal, free. KTDB national passenger OD by application (foreigner access unverified).
- **France**: OD INSEE base flux mobilités professionnelles 2022, commune to commune
  (Paris/Lyon/Marseille by arrondissement), plus an individual-level Parquet file, free,
  Licence Ouverte. insee.fr/fr/statistiques/8582949. Jobs by commune; geolocated SIRENE
  establishments with headcount bands can split below commune.
- **Germany**: OD from the Federal Employment Agency Pendleratlas (Kreis and Gemeinde,
  socially insured employees) and the statistical offices' Pendleratlas (all commuters,
  Gemeinde); bulk files not confirmed. Jobs by Gemeinde. Zensus 2022 100 m grid is residents
  only. statistik.arbeitsagentur.de
- **Netherlands**: OD CBS StatLine 83628NED, jobs by home and work municipality, 2014-2020,
  free. Neighbourhood jobs unverified; LISA register is paid.
- **Canada**: OD 2021 census table 98-10-0459-01, CSD to CSD, free. Jobs by place of work
  below CSD unverified.
- **Australia**: jobs by destination zone (DZN) from 2021 census place of work; Working
  Population Profile DataPacks probably free (unverified). OD home SA2 to work DZN with mode,
  via ABS TableBuilder (free registration).
- **Spain**: MITMA phone-based mobility study: daily OD at municipality, district (~3,700
  zones) and urban-area level, monthly since 2022, activity tags allow home-to-work. Open
  data, licence assumed CC BY (unverified).
- **Italy**: OD ISTAT 2021 commuting matrix, commune to commune, released 2025, free. Jobs from
  the 2011 industry census by census section (very fine), free.
- **China**: no official small-area jobs or OD. Phone-company data is commercial.
- **Taiwan**: jobs by statistical area via SEGIS and data.gov.tw; OD 2020 census commuting at
  township level. Both unverified.
- **EU-wide**: JRC ENACT-POP R2020A, 1 km day and night population grids for 2011, monthly;
  free. GEOSTAT 2021 1 km counts employed people by home, not work. No pan-EU commuting OD.

## Global jobs proxies

- **GHSL GHS-BUILT-S R2023A** (JRC): has a non-residential (NRES) component; 100 m grid,
  1975-2030 in 5-year steps; free with credit. GHS-BUILT-V (volume) has NRES too, likely the
  better proxy. A 10 m version with NRES (2018) exists.
- **LandScan Global**: 1 km ambient (24-hour) population, CC BY 4.0.
- No proper global employment grid.
- **Synthetic OD**: Tsinghua global commuting OD (arXiv 2505.17111, 2025), model-generated
  flows for 1,625 cities in 179 countries. Dataset licence, location and grain unverified.
  A possible stand-in tier 2 anywhere, or a check on our own gravity fit.
- OSM POIs, non-residential building footprints, Overture Places, Foursquare OS Places: can
  split GHSL NRES by type (not checked this pass).

## Intercity all-mode travel

- **Japan**: MLIT Inter-regional Passenger Flow Survey (Kansen Ryokaku Junryudo Chosa), every 5
  years since 1990. Trips crossing a prefecture line, all modes (air, rail, ferry, bus, car),
  true origin to final destination, purpose, weekday and holiday. Zones: 47 prefectures, 207
  daily-life zones. Summary files free; trip-level by request. 2015 is the latest listed; 2021
  release unverified. mlit.go.jp/sogoseisaku/soukou/sogoseisaku_soukou_fr_000016.html
- **USA**: NextGen NHTS passenger OD (FHWA), trips between 583 zones (MSAs plus rest of each
  state), 2020-2022, yearly and some monthly, from location data. Free. nhts.ornl.gov/od.
  DB1B is air only.
- **Germany**: Verkehrsverflechtungsprognose 2030, all-mode OD at Kreis level, base 2010, said
  to be public CSV (partly verified).
- **EU**: ETISplus / TRANS-TOOLS NUTS-3 passenger OD (around 2010), no current open download
  found. ESPON country-to-country by mode. JRC NUTS-2 travel time and cost matrices (2020) for
  impedance only.
- **Global**: nothing all-mode; air seat capacity is the usual stand-in.

## Which tier each country can reach

1. Tier 2, fine grain: USA (block), UK (MSOA public; OA with registration), Spain (MITMA
   districts).
2. Tier 2, municipal grain, split within by gravity: France, Italy, Japan, Germany,
   Netherlands, Canada, Australia, Korea; Taiwan unverified.
3. Tier 1 with good small-area jobs: Japan (500 m mesh), UK (OA), Australia (DZN), Italy
   (2011 sections), EU (ENACT 1 km).
4. Tier 0 with GHSL NRES as the jobs proxy: China and everywhere else.
5. Intercity all-mode OD: Japan (207 zones), USA (583 zones), Germany (Kreis, 2010).
