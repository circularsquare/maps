// T-924: the data sources the published packs carry and the notices their licences ask for
// (notes/T-924.md has the table). Closed by default at the foot of Settings.

import { t } from "../i18n";

// the licence notices stay in English: it is the binding text (docs/localization.md)
const LINES = [
  <>Map data © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>, under the Open Database License.</>, // i18n-ignore
  <>Basemap © <a href="https://openmaptiles.org/" target="_blank" rel="noopener">OpenMapTiles</a>, served by <a href="https://openfreemap.org/" target="_blank" rel="noopener">OpenFreeMap</a>. Rail lines from OpenStreetMap and national rail registers.</>, // i18n-ignore
  <>© EuroGeographics for the administrative boundaries.</>, // i18n-ignore
  <>Population and built-up areas from GHSL, © European Union 1995-2025, and Kontur Population (CC BY 4.0).</>, // i18n-ignore
  <>Boundaries from <a href="https://www.geoboundaries.org/" target="_blank" rel="noopener">geoBoundaries</a> (Spain's from the Instituto Geográfico Nacional, CC BY 4.0), Natural Earth and the Office for National Statistics. Contains OS data © Crown copyright and database right 2025.</>, // i18n-ignore
  <>Contains public sector information licensed under the Open Government Licence v3.0 (ONS, Nomis, Department for Education, National Records of Scotland, NISRA) and the Open Government Licence (Ontario).</>, // i18n-ignore
  <>Powered by TfL Open Data. Contains Network Rail Infrastructure Limited data, used under the Open Government Licence v3.0.</>, // i18n-ignore
  <>Adapted from Statistics Canada, 2021 Census. This does not constitute an endorsement by Statistics Canada of this product.</>, // i18n-ignore
  <>Australian Bureau of Statistics, 2021 Census and statistical areas (CC BY 4.0).</>, // i18n-ignore
  <>US Census Bureau, NCES, FAA and the Bureau of Transportation Statistics.</>, // i18n-ignore
  <>New York subway, LIRR and Metro-North timetables from MTA open data, and PATH timetables from the Port Authority of New York and New Jersey. The game is not affiliated with or endorsed by the MTA or the Port Authority.</>, // i18n-ignore
  <>Paris timetables © Île-de-France Mobilités, under the <a href="https://opendatacommons.org/licenses/odbl/" target="_blank" rel="noopener">Open Database License</a>.</>, // i18n-ignore
  <>China township populations from the ASPECT population grid (Ju et al. 2025, Scientific Data), CC BY 4.0.</>, // i18n-ignore
  <>INSEE and geo.api.gouv.fr (Licence Ouverte 2.0). © Statistisches Bundesamt (Destatis), Zensus 2022. © <a href="https://www.bkg.bund.de/" target="_blank" rel="noopener">BKG</a> 2025, <a href="https://www.govdata.de/dl-de/by-2-0" target="_blank" rel="noopener">dl-de/by-2-0</a>. Pendleratlas of the statistical offices of the Federation and the Länder.</>, // i18n-ignore
  <>Processed from Japan's 2020 census and 2021 economic census (Statistics Bureau of Japan, e-Stat) and National Land Numerical Information (MLIT).</>, // i18n-ignore
  <>IBGE. Metrô de São Paulo origin and destination survey 2017. Ministry of the Interior and Safety (Korea). Census of India 2011. Shanghai Municipal Statistics Bureau.</>, // i18n-ignore
  <>World Bank (CC BY 4.0), Maddison Project Database 2023 (CC BY 4.0), Eurostat, Wikidata and OurAirports.</>, // i18n-ignore
  <>Large lakes from Natural Earth. Terrain classes from GEDTM30 (OpenGeoHub, CC BY 4.0). Hill shading from <a href="https://github.com/tilezen/joerd/blob/master/docs/attribution.md" target="_blank" rel="noopener">Terrain Tiles</a> on AWS Open Data.</>, // i18n-ignore
  <>Fonts Zen Maru Gothic, Noto Sans and Nunito, under the SIL Open Font License 1.1.</>, // i18n-ignore
  <>Finger snap recording by freesound_community, cork pop by Universfield, wood knock by rohhsadotcom and magazine insert by freesounds-4u, all from <a href="https://pixabay.com/" target="_blank" rel="noopener">Pixabay</a>.</>, // i18n-ignore
  <>All data was adapted for this game.</>, // i18n-ignore
  <>The game's data files are under the <a href="https://opendatacommons.org/licenses/odbl/" target="_blank" rel="noopener">Open Database License (ODbL)</a>, as the OpenStreetMap data they come from is.</>, // i18n-ignore
];

export function Credits() {
  return (
    <details class="credits">
      <summary class="h">{t("credits.title")}</summary>
      {LINES.map((l) => <p>{l}</p>)}
    </details>
  );
}
