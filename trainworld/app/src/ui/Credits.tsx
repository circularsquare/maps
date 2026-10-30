// T-924: the data sources the published packs carry and the notices their licences ask for
// (notes/T-924.md has the table). Closed by default at the foot of Settings.

const LINES = [
  <>Map data © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>, under the Open Database License.</>,
  <>Basemap © <a href="https://openmaptiles.org/" target="_blank" rel="noopener">OpenMapTiles</a>, served by <a href="https://openfreemap.org/" target="_blank" rel="noopener">OpenFreeMap</a>. Rail lines from OpenStreetMap and national rail registers.</>,
  <>© EuroGeographics for the administrative boundaries.</>,
  <>Population and built-up areas from GHSL, © European Union 1995-2025, and Kontur Population (CC BY 4.0).</>,
  <>Boundaries from <a href="https://www.geoboundaries.org/" target="_blank" rel="noopener">geoBoundaries</a> (Spain's from the Instituto Geográfico Nacional, CC BY 4.0), Natural Earth and the Office for National Statistics. Contains OS data © Crown copyright and database right 2025.</>,
  <>Contains public sector information licensed under the Open Government Licence v3.0 (ONS, Nomis, Department for Education, National Records of Scotland, NISRA) and the Open Government Licence (Ontario).</>,
  <>Powered by TfL Open Data. Contains Network Rail Infrastructure Limited data, used under the Open Government Licence v3.0.</>,
  <>Adapted from Statistics Canada, 2021 Census. This does not constitute an endorsement by Statistics Canada of this product.</>,
  <>Australian Bureau of Statistics, 2021 Census and statistical areas (CC BY 4.0).</>,
  <>US Census Bureau, NCES, FAA and the Bureau of Transportation Statistics.</>,
  <>New York subway, LIRR and Metro-North timetables from MTA open data, and PATH timetables from the Port Authority of New York and New Jersey. The game is not affiliated with or endorsed by the MTA or the Port Authority.</>,
  <>Paris timetables © Île-de-France Mobilités, under the <a href="https://opendatacommons.org/licenses/odbl/" target="_blank" rel="noopener">Open Database License</a>.</>,
  <>China township populations from the ASPECT population grid (Ju et al. 2025, Scientific Data), CC BY 4.0.</>,
  <>INSEE and geo.api.gouv.fr (Licence Ouverte 2.0). © Statistisches Bundesamt (Destatis), Zensus 2022. © <a href="https://www.bkg.bund.de/" target="_blank" rel="noopener">BKG</a> 2025, <a href="https://www.govdata.de/dl-de/by-2-0" target="_blank" rel="noopener">dl-de/by-2-0</a>. Pendleratlas of the statistical offices of the Federation and the Länder.</>,
  <>Processed from Japan's 2020 census and 2021 economic census (Statistics Bureau of Japan, e-Stat) and National Land Numerical Information (MLIT).</>,
  <>IBGE. Metrô de São Paulo origin and destination survey 2017. Ministry of the Interior and Safety (Korea). Census of India 2011. Shanghai Municipal Statistics Bureau.</>,
  <>World Bank (CC BY 4.0), Maddison Project Database 2023 (CC BY 4.0), Eurostat, Wikidata and OurAirports.</>,
  <>Large lakes from Natural Earth. Terrain classes from GEDTM30 (OpenGeoHub, CC BY 4.0). Hill shading from <a href="https://github.com/tilezen/joerd/blob/master/docs/attribution.md" target="_blank" rel="noopener">Terrain Tiles</a> on AWS Open Data.</>,
  <>Fonts Zen Maru Gothic, Noto Sans and Nunito, under the SIL Open Font License 1.1.</>,
  <>Finger snap recording by freesound_community, cork pop by Universfield, wood knock by rohhsadotcom and magazine insert by freesounds-4u, all from <a href="https://pixabay.com/" target="_blank" rel="noopener">Pixabay</a>.</>,
  <>All data was adapted for this game.</>,
  <>The game's data files are under the <a href="https://opendatacommons.org/licenses/odbl/" target="_blank" rel="noopener">Open Database License (ODbL)</a>, as the OpenStreetMap data they come from is.</>,
];

export function Credits() {
  return (
    <details class="credits">
      <summary class="h">Data sources</summary>
      {LINES.map((l) => <p>{l}</p>)}
    </details>
  );
}
