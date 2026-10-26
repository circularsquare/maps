// T-924: the data sources the published packs carry and the notices their licences ask for
// (notes/T-924.md has the table). Closed by default at the foot of Settings.

const LINES = [
  <>Map data © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>, under the Open Database License.</>,
  <>Basemap © <a href="https://openmaptiles.org/" target="_blank" rel="noopener">OpenMapTiles</a>, served by <a href="https://openfreemap.org/" target="_blank" rel="noopener">OpenFreeMap</a>. Rail lines from OpenStreetMap and national rail registers.</>,
  <>© EuroGeographics for the administrative boundaries.</>,
  <>Population and built-up areas from GHSL, © European Union 1995-2025, and Kontur Population (CC BY 4.0).</>,
  <>Boundaries from <a href="https://www.geoboundaries.org/" target="_blank" rel="noopener">geoBoundaries</a>, Natural Earth and the Office for National Statistics. Contains OS data © Crown copyright and database right 2025.</>,
  <>Contains public sector information licensed under the Open Government Licence v3.0 (ONS, Nomis, Department for Education, National Records of Scotland, NISRA) and the Open Government Licence (Ontario).</>,
  <>Adapted from Statistics Canada, 2021 Census. This does not constitute an endorsement by Statistics Canada of this product.</>,
  <>US Census Bureau, NCES, FAA and the Bureau of Transportation Statistics.</>,
  <>INSEE and geo.api.gouv.fr (Licence Ouverte 2.0). © Statistisches Bundesamt (Destatis), Zensus 2022. © <a href="https://www.bkg.bund.de/" target="_blank" rel="noopener">BKG</a> 2025, <a href="https://www.govdata.de/dl-de/by-2-0" target="_blank" rel="noopener">dl-de/by-2-0</a>. Pendleratlas of the statistical offices of the Federation and the Länder.</>,
  <>Processed from Japan's 2020 census and 2021 economic census (Statistics Bureau of Japan, e-Stat) and National Land Numerical Information (MLIT).</>,
  <>IBGE. Metrô de São Paulo origin and destination survey 2017. Ministry of the Interior and Safety (Korea). Census of India 2011. Shanghai Municipal Statistics Bureau.</>,
  <>World Bank (CC BY 4.0), Maddison Project Database 2023 (CC BY 4.0), Eurostat, Wikidata and OurAirports.</>,
  <>Terrain classes from GEDTM30 (OpenGeoHub, CC BY 4.0). Hill shading from <a href="https://github.com/tilezen/joerd/blob/master/docs/attribution.md" target="_blank" rel="noopener">Terrain Tiles</a> on AWS Open Data.</>,
  <>Fonts Zen Maru Gothic, Noto Sans and Nunito, under the SIL Open Font License 1.1.</>,
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
