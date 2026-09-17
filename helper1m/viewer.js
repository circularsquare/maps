// helper1m viewer — country-agnostic, reads countries.json + countries/<id>/meta.json
// then loads the current admin level's geojson and lets the user click / shift-click.

const CURRENT_YEAR = new Date().getFullYear();
// Fallback years for the detail panel history table when a country's meta.json
// doesn't name its own (India's five-yearly projections were the first case).
const DISPLAY_YEARS = [2011, 2026];
// A split level is one geojson per adm1 region, loaded only for what's on
// screen. Above this many regions in view we load nothing — the point of
// splitting is that the whole level never gets fetched at once.
const SPLIT_MAX_GROUPS = 4;
const state = {
  country: null,   // {id, name, meta}
  level: null,     // int
  featuresByCode: new Map(),
  selection: [],   // [{code, level, name, raw_pop, est_pop}]
  groups: [],      // [{code, name}] — adm1 regions (states), for the state filter
  shownGroups: null,  // Set of group codes currently visible
  hoverCode: null,    // feature under the cursor
  activeCode: null,   // feature clicked for the detail panel
  overlayGeo: {},     // {1: adm1, 2: adm2} geojson — coarser borders drawn as context
  groupBounds: {},    // {groupCode: [minx, miny, maxx, maxy]} — for split loading
  splitCache: new Map(),  // "level/group" -> features, so re-ticking doesn't refetch
  loadedSplitKey: null,   // which set of groups is currently rendered
  loading: false,         // mid country switch — don't save or lazy-load yet
  comp: null,             // countries/<id>/composition.json, when the country has one
  compOn: true,           // draw the composition pies
  compSize: 1,            // pie size multiplier, off the slider
  compHidden: new Set(),  // group keys switched off in the composition legend
};

// Saved view lives here; declared up top because the sidebar width is read from
// it before the map exists (see the saved-view section for the rest).
const STORE_KEY = "helper1m:view";

// Sidebar width is draggable. Below NARROW the panels tighten up so a sidebar
// about a third of the default still reads.
const SIDEBAR_DEFAULT = 320;
const SIDEBAR_MIN = 100;
const SIDEBAR_NARROW = 200;

// Elevation basemap. Terrain Tiles on AWS Open Data: global, keyless,
// terrarium-encoded, and served with CORS — which matters, because
// updateRelief() reads these same tiles pixel by pixel to find the range on
// screen. The ramp runs blue (lowest on screen) through green, yellow, orange
// and red to white (highest). The top of the scale gets more than an even
// share: a scene's mountains are a long thin tail of elevations, so red to
// white spans 45% of the range and the crowded lower ground keeps the finer
// steps. Colours between anchors are mixed in OKLab so no pair of neighbours
// goes muddy on the way.
const DEM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
// [position along the on-screen range, colour]
const RELIEF_ANCHORS = [
  [0.00, "#2b5ea7"],   // blue
  [0.14, "#3f9b4f"],   // green
  [0.28, "#f2d33a"],   // yellow
  [0.42, "#f08c28"],   // orange
  [0.55, "#d7301f"],   // red
  [1.00, "#ffffff"],   // white
];
const RELIEF_STEPS = 24;
const BASEMAPS = ["streets", "topo", "relief"];

function setSidebarWidth(px) {
  const max = Math.max(SIDEBAR_MIN, Math.min(640, window.innerWidth - 200));
  const w = Math.round(Math.min(max, Math.max(SIDEBAR_MIN, px)));
  const el = document.getElementById("sidebar");
  el.style.width = `${w}px`;
  el.classList.toggle("narrow", w < SIDEBAR_NARROW);
  return w;
}

// Before the map is built, so it initialises at the right size with no flash.
setSidebarWidth(Number(readStore().sidebarWidth) || SIDEBAR_DEFAULT);

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors",
      },
      topo: {
        type: "raster",
        tiles: [
          "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
          "https://b.tile.opentopomap.org/{z}/{x}/{y}.png",
          "https://c.tile.opentopomap.org/{z}/{x}/{y}.png",
        ],
        tileSize: 256,
        maxzoom: 17,
        attribution: "© OpenTopoMap (CC-BY-SA)",
      },
      dem: {
        type: "raster-dem",
        tiles: [DEM_URL],
        encoding: "terrarium",
        tileSize: 256,
        maxzoom: 15,
        attribution: "Terrain Tiles (Mapzen, AWS Open Data)",
      },
    },
    layers: [
      { id: "osm", type: "raster", source: "osm", paint: { "raster-opacity": 0.55 } },
      { id: "topo", type: "raster", source: "topo",
        layout: { visibility: "none" }, paint: { "raster-opacity": 0.8 } },
      { id: "relief", type: "color-relief", source: "dem",
        layout: { visibility: "none" },
        paint: { "color-relief-color": reliefColorExpr(0, 3000) } },
    ],
  },
  center: [0, 0],
  zoom: 2,
  // North stays up and the map stays flat. maxPitch pins the tilt even if
  // something calls easeTo with a pitch.
  dragRotate: false,
  pitchWithRotate: false,
  touchPitch: false,
  maxPitch: 0,
});

// Shift-click is our multi-select; free it from MapLibre's box-zoom handler.
map.boxZoom.disable();
// The constructor flags miss two paths: pinch-rotate on a trackpad or phone,
// and shift+arrow on the keyboard. Pinch-zoom and the arrow-key pan both stay.
map.touchZoomRotate.disableRotation();
map.keyboard.disableRotation();

// ---- data loading ----

// Always bypass the HTTP cache. python -m http.server sends only Last-Modified
// (no ETag/Cache-Control) and mishandles If-Modified-Since, so a rebuilt
// geojson can keep serving stale bytes via a 304 even on a hard refresh.
function fetchNoCache(url) {
  return fetch(url, { cache: "no-store" });
}

async function loadCountriesIndex() {
  const res = await fetchNoCache("countries.json");
  return (await res.json()).countries;
}

async function loadCountry(id) {
  const meta = await (await fetchNoCache(`countries/${id}/meta.json`)).json();
  teardown();   // drop the previous country's sources/layers
  state.loading = true;
  state.country = { id, name: meta.name, meta };
  await loadOverlays();
  loadGroups();
  await loadComposition();

  // Pick up where this country was left. Usually one region at the finest
  // level, so restoring the camera too is what makes it actually continue.
  const saved = loadView(id);
  if (saved && saved.groups) {
    const known = new Set(state.groups.map(g => g.code));
    const ticked = new Set(saved.groups.filter(c => known.has(c)));
    if (ticked.size) state.shownGroups = ticked;
  }
  renderGroupFilter();
  renderLevelRadios();
  if (saved && saved.center) {
    map.jumpTo({ center: saved.center, zoom: saved.zoom });
  } else {
    map.flyTo({ center: meta.center, zoom: meta.zoom, duration: 0 });
  }
  const levels = meta.admin_levels.map(l => l.level);
  const lvl = saved && levels.includes(saved.level) ? saved.level : levels[0];
  state.loading = false;
  await setLevel(lvl);
  saveView();
}

// ---- saved view ----

// Which regions are ticked, which level, and where the map is — per country,
// so a refresh drops you back into the province you were working through.
// STORE_KEY is declared at the top of the file.

function readStore() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); }
  catch (e) { return {}; }   // private window, disabled storage, bad JSON
}

function loadView(countryId) {
  const all = readStore();
  return (all.countries && all.countries[countryId]) || null;
}

function saveView() {
  if (!state.country || state.loading) return;
  try {
    const all = readStore();
    all.countries = all.countries || {};
    all.countries[state.country.id] = {
      level: state.level,
      groups: state.shownGroups ? [...state.shownGroups] : null,
      center: map.getCenter().toArray(),
      zoom: map.getZoom(),
      comp: state.comp ? { on: state.compOn, size: state.compSize,
                           hidden: [...state.compHidden] } : undefined,
    };
    all.last = state.country.id;
    all.basemap = currentBasemap();
    delete all.topo;   // the old checkbox, replaced by basemap
    all.opacity = document.getElementById("fill-opacity").value;
    all.sidebarWidth = parseInt(document.getElementById("sidebar").style.width, 10);
    localStorage.setItem(STORE_KEY, JSON.stringify(all));
  } catch (e) { /* nothing here is worth breaking the page over */ }
}

map.on("moveend", saveView);

function levelCfg(level = state.level) {
  return state.country.meta.admin_levels.find(l => l.level === level);
}

async function setLevel(level) {
  state.level = level;
  state.loadedSplitKey = null;
  document.querySelectorAll("#level-radios input").forEach(r => {
    r.checked = parseInt(r.value) === level;
  });
  const cfg = levelCfg(level);
  if (cfg.split) return loadSplitLevel();

  const url = `countries/${state.country.id}/${cfg.file}`;
  let geo;
  try {
    geo = await (await fetchNoCache(url)).json();
  } catch (e) {
    showInfo(`<div class="empty">No data for level ${level}. Run the build script.</div>`);
    return;
  }
  state.featuresByCode.clear();
  for (const f of geo.features) state.featuresByCode.set(f.properties.code, f);
  renderLayer(geo);
}

// ---- split levels ----

// Which adm1 regions overlap the current viewport, of the ones ticked. Keeps a
// 43,655-township level usable by never holding more than a few provinces.
function groupsInView() {
  const b = map.getBounds();
  const [w, s, e, n] = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
  return state.groups
    .map(g => g.code)
    .filter(code => !state.shownGroups || state.shownGroups.has(code))
    .filter(code => {
      const bb = state.groupBounds[code];
      return bb && bb[0] <= e && bb[2] >= w && bb[1] <= n && bb[3] >= s;
    });
}

async function loadSplitLevel() {
  const cfg = levelCfg();
  if (!cfg || !cfg.split) return;
  const wanted = groupsInView();
  const key = wanted.join(",");
  if (key === state.loadedSplitKey) return;

  if (!wanted.length || wanted.length > SPLIT_MAX_GROUPS) {
    state.loadedSplitKey = key;
    state.featuresByCode.clear();
    renderLayer({ type: "FeatureCollection", features: [] });
    const label = cfg.label.toLowerCase();
    showInfo(`<div class="empty">Zoom in to load ${label}s — this level is
      stored per ${groupLabel()} and loads what's on screen, up to
      ${SPLIT_MAX_GROUPS} at a time.</div>`);
    return;
  }

  const parts = await Promise.all(wanted.map(async code => {
    const cacheKey = `${state.level}/${code}`;
    if (!state.splitCache.has(cacheKey)) {
      const url = `countries/${state.country.id}/${cfg.dir}/${code}.geojson`;
      try {
        state.splitCache.set(cacheKey, (await (await fetchNoCache(url)).json()).features);
      } catch (e) {
        state.splitCache.set(cacheKey, []);
      }
    }
    return state.splitCache.get(cacheKey);
  }));

  // The viewport may have moved on while those were in flight.
  if (groupsInView().join(",") !== key) return loadSplitLevel();

  const features = parts.flat();
  state.loadedSplitKey = key;
  state.featuresByCode.clear();
  for (const f of features) state.featuresByCode.set(f.properties.code, f);
  renderLayer({ type: "FeatureCollection", features });
  // Clear the zoom-in prompt once there is something to click.
  if (document.querySelector("#info-panel .empty")) {
    showInfo(`<div class="empty">Click a region for its population.</div>`);
  }
}

function groupLabel() {
  const adm1 = state.country.meta.admin_levels.find(l => l.level === 1);
  return adm1 ? adm1.label.toLowerCase() : "region";
}

// Reload on idle rather than on every move — setData on a big source backs up.
map.on("idle", () => {
  if (state.loading || !state.country) return;
  if (levelCfg() && levelCfg().split) loadSplitLevel();
});

// ---- state filter ----

function loadGroups() {
  state.groups = [];
  state.shownGroups = null;
  state.groupBounds = {};
  state.splitCache.clear();
  const geo = state.overlayGeo[1];   // adm1 — the state list
  if (!geo) return;
  state.groups = geo.features
    .map(f => ({ code: f.properties.code, name: f.properties.name }))
    .sort((a, b) => a.name.localeCompare(b.name));
  state.shownGroups = new Set(state.groups.map(g => g.code));
  for (const f of geo.features) state.groupBounds[f.properties.code] = bbox(f.geometry);
}

function bbox(geometry) {
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  const walk = (coords) => {
    if (typeof coords[0] === "number") {
      if (coords[0] < minx) minx = coords[0];
      if (coords[0] > maxx) maxx = coords[0];
      if (coords[1] < miny) miny = coords[1];
      if (coords[1] > maxy) maxy = coords[1];
      return;
    }
    for (const c of coords) walk(c);
  };
  walk(geometry.coordinates);
  return [minx, miny, maxx, maxy];
}

function renderGroupFilter() {
  const panel = document.getElementById("group-panel");
  const host = document.getElementById("group-list");
  if (!state.groups.length) { panel.style.display = "none"; return; }
  panel.style.display = "";
  // The filter is by adm1, so name it after whatever the country calls that.
  // Only pluralise a single word — India's "State / UT" reads worse with an s.
  const label = (state.country.meta.admin_levels.find(l => l.level === 1)
    || {}).label || "Region";
  document.getElementById("group-panel-label").textContent =
    /[\s/]/.test(label) ? label : label + "s";
  host.innerHTML = "";
  for (const g of state.groups) {
    const on = state.shownGroups.has(g.code) ? "checked" : "";
    host.insertAdjacentHTML("beforeend",
      `<label title="${escapeHtml(g.name)}"><input type="checkbox" value="${escapeHtml(g.code)}" ${on}>` +
      `<span>${escapeHtml(shortGroupName(g.name))}</span></label>`);
  }
}

// The list is already headed by the division type, so "Anhui Province" can read
// "Anhui" and the list stays legible in a narrow sidebar. Full name on hover.
const GROUP_TYPE_WORDS =
  /\s+(Special Administrative Region|Autonomous Region|Municipality|Province)$/;
function shortGroupName(name) {
  return name.replace(GROUP_TYPE_WORDS, "") || name;
}

// MapLibre filter: show a feature when its group is checked. Features with no
// group (a country not built with one) always show.
function groupFilterExpr() {
  if (!state.shownGroups || state.shownGroups.size === state.groups.length) return null;
  return ["any",
    ["!", ["has", "group"]],
    ["in", ["get", "group"], ["literal", [...state.shownGroups]]]];
}

function applyGroupFilter() {
  const expr = groupFilterExpr();
  for (const id of [FILL, LINE, HL, ...overlayLayerIds()]) {
    if (map.getLayer(id)) map.setFilter(id, expr);
  }
  // A split level holds only the ticked regions, so unticking one has to drop
  // its features rather than just hide them.
  if (state.country && levelCfg() && levelCfg().split) loadSplitLevel();
}

// ---- map layer ----

const SRC = "admin";
const FILL = "admin-fill";
const LINE = "admin-line";
const HL = "admin-highlight";

// Coarser levels drawn as context, heavier the coarser they are, so the
// hierarchy reads at a glance: province thicker than prefecture thicker than
// county thicker than the level you are working in. Each is fetched only once
// the current level is deeper than it, which keeps county borders (10 MB in
// China) off the initial load.
const OVERLAY_LEVELS = [1, 2, 3];
// Thickness carries the ranking; no coarser border is ever paler than a finer
// one, or the thin working-level lines read as the more important boundary.
const OVERLAY_STYLE = {
  1: { color: "#000", width: 2.2 },
  2: { color: "#111", width: 1.3 },
  3: { color: "#222", width: 0.8 },
};
const overlayId = lvl => `ov-${lvl}`;

// Fill opacity for the density choropleth, driven by the sidebar slider so the
// colours can be laid over any basemap at whatever strength reads.
// Selected regions sit a little above whatever that is.
function fillOpacityExpr() {
  const base = (parseInt(document.getElementById("fill-opacity").value, 10) || 0) / 100;
  return ["case",
    ["boolean", ["feature-state", "selected"], false], Math.min(1, base + 0.2),
    base];
}

// Pointer handlers — registered once. They key off the layer id, so they keep
// working when the layer is torn down and rebuilt on a country switch.
map.on("mousemove", FILL, (e) => {
  if (!e.features.length) return;
  map.getCanvas().style.cursor = "pointer";
  const code = e.features[0].properties.code;
  setHover(code);
  showTip(code, e.originalEvent.clientX, e.originalEvent.clientY);
});
map.on("mouseleave", FILL, () => {
  map.getCanvas().style.cursor = "";
  setHover(null);
  hideTip();
});
map.on("click", FILL, onFeatureClick);

// A plain click on empty space (no division) clears the multi-selection.
map.on("click", (e) => {
  if (e.originalEvent.shiftKey || !map.getLayer(FILL)) return;
  if (!map.queryRenderedFeatures(e.point, { layers: [FILL] }).length) clearSelection();
});

// ---- boundary overlays (coarser levels drawn as context) ----

// adm1 is fetched eagerly — the group filter and the split-level loader both
// need its geometry. The rest wait until a level below them is in use.
async function loadOverlays() {
  state.overlayGeo = {};
  await fetchOverlay(1);
}

async function fetchOverlay(lvl) {
  if (lvl in state.overlayGeo) return state.overlayGeo[lvl];
  state.overlayGeo[lvl] = null;
  const cfg = levelCfg(lvl);
  if (!cfg || !cfg.file) return null;
  try {
    state.overlayGeo[lvl] =
      await (await fetchNoCache(`countries/${state.country.id}/${cfg.file}`)).json();
  } catch (e) { /* level not built */ }
  return state.overlayGeo[lvl];
}

function addOverlay(lvl, geo) {
  const id = overlayId(lvl);
  if (!geo || map.getSource(id)) return;
  map.addSource(id, { type: "geojson", data: geo });
  map.addLayer({
    id: id + "-line", type: "line", source: id,
    layout: { visibility: "none" },
    paint: {
      "line-color": OVERLAY_STYLE[lvl].color,
      "line-width": OVERLAY_STYLE[lvl].width,
    },
  }, map.getLayer(HL) ? HL : undefined);   // under the highlight, over the fill
  map.setFilter(id + "-line", groupFilterExpr());
}

// A coarser level's borders show whenever the working level is below it.
async function updateOverlayVisibility() {
  for (const lvl of OVERLAY_LEVELS) {
    const show = state.level > lvl;
    const id = overlayId(lvl) + "-line";
    if (show && !map.getLayer(id)) addOverlay(lvl, await fetchOverlay(lvl));
    if (map.getLayer(id))
      map.setLayoutProperty(id, "visibility", show ? "visible" : "none");
  }
}

function overlayLayerIds() {
  return OVERLAY_LEVELS.map(lvl => overlayId(lvl) + "-line");
}

function teardown() {
  for (const id of [FILL, LINE, HL, ...overlayLayerIds()]) {
    if (map.getLayer(id)) map.removeLayer(id);
  }
  for (const id of [SRC, ...OVERLAY_LEVELS.map(overlayId)]) {
    if (map.getSource(id)) map.removeSource(id);
  }
  state.comp = null;
  document.getElementById("comp-panel").hidden = true;
}

// ---- basemap ----

function currentBasemap() {
  const el = document.querySelector("#basemap-radios input:checked");
  return el ? el.value : "streets";
}

// Swaps what sits under the density fill. The fill itself stays at whatever
// the slider says, so either can be read through the other.
function applyBasemap() {
  const base = currentBasemap();
  map.setLayoutProperty("osm", "visibility", base === "streets" ? "visible" : "none");
  map.setLayoutProperty("topo", "visibility", base === "topo" ? "visible" : "none");
  map.setLayoutProperty("relief", "visibility", base === "relief" ? "visible" : "none");
  document.getElementById("relief-legend").hidden = base !== "relief";
  if (base === "relief") updateRelief();
}

function applyFillOpacity() {
  const el = document.getElementById("fill-opacity");
  document.getElementById("fill-opacity-value").textContent = `${el.value}%`;
  if (map.getLayer(FILL))
    map.setPaintProperty(FILL, "fill-opacity", fillOpacityExpr());
}

// ---- elevation basemap ----

// The relief's colours are stretched over whatever elevations are on screen,
// so a plain and a mountain range each get the whole ramp. MapLibre colours the
// DEM on the GPU but will not say what range it saw, so the range is measured
// here from the same tiles, one zoom coarser than the view and at most a dozen
// of them. Re-measured on idle, and pushed to the layer only when it changes.
const demTiles = new Map();   // "z/x/y" -> Promise of a Float32Array, or null
let reliefApplied = "";       // "lo,hi" last pushed to the layer
let reliefRequest = 0;        // lets a measurement the camera has moved past drop out

map.on("idle", () => {
  if (currentBasemap() === "relief") updateRelief();
});

function demTile(z, x, y) {
  const key = `${z}/${x}/${y}`;
  if (!demTiles.has(key)) {
    demTiles.set(key, (async () => {
      try {
        const url = DEM_URL.replace("{z}", z).replace("{x}", x).replace("{y}", y);
        const blob = await (await fetch(url)).blob();
        // No colour management: the channels encode elevation, not a colour.
        const bmp = await createImageBitmap(blob,
          { colorSpaceConversion: "none", premultiplyAlpha: "none" });
        const canvas = new OffscreenCanvas(256, 256);
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(bmp, 0, 0, 256, 256);
        const px = ctx.getImageData(0, 0, 256, 256).data;
        const elev = new Float32Array(256 * 256);
        for (let i = 0; i < elev.length; i++) {
          elev[i] = px[i * 4] * 256 + px[i * 4 + 1] + px[i * 4 + 2] / 256 - 32768;
        }
        return elev;
      } catch (e) {
        return null;
      }
    })());
    // Bounded; the browser's own cache still holds the PNGs if one is evicted.
    if (demTiles.size > 128) demTiles.delete(demTiles.keys().next().value);
  }
  return demTiles.get(key);
}

function mercatorY(lat) {
  const s = Math.sin(Math.max(-85.0511, Math.min(85.0511, lat)) * Math.PI / 180);
  return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
}

async function updateRelief() {
  const req = ++reliefRequest;
  const b = map.getBounds();
  let z = Math.max(0, Math.min(15, Math.floor(map.getZoom()) - 1));
  let x0, x1, y0, y1;
  for (;;) {
    const n = 2 ** z;
    x0 = (b.getWest() + 180) / 360 * n;
    x1 = (b.getEast() + 180) / 360 * n;
    y0 = mercatorY(b.getNorth()) * n;
    y1 = mercatorY(b.getSouth()) * n;
    const count = (Math.floor(x1) - Math.floor(x0) + 1) * (Math.floor(y1) - Math.floor(y0) + 1);
    if (count <= 12 || z === 0) break;
    z--;
  }
  const n = 2 ** z;
  const jobs = [];
  for (let ty = Math.max(0, Math.floor(y0)); ty <= Math.min(n - 1, Math.floor(y1)); ty++) {
    for (let tx = Math.floor(x0); tx <= Math.floor(x1); tx++) {
      jobs.push(demTile(z, ((tx % n) + n) % n, ty).then(elev => ({ tx, ty, elev })));
    }
  }
  const tiles = await Promise.all(jobs);
  if (req !== reliefRequest || currentBasemap() !== "relief") return;

  // Every 4th pixel that falls inside the view.
  const STEP = 4;
  const land = [];
  const all = [];
  for (const { tx, ty, elev } of tiles) {
    if (!elev) continue;
    for (let iy = STEP / 2; iy < 256; iy += STEP) {
      const gy = ty + iy / 256;
      if (gy < y0 || gy > y1) continue;
      for (let ix = STEP / 2; ix < 256; ix += STEP) {
        const gx = tx + ix / 256;
        if (gx < x0 || gx > x1) continue;
        const v = elev[iy * 256 + ix];
        all.push(v);
        if (v > 0) land.push(v);
      }
    }
  }
  // Stretch over land where there is enough of it, or a coastal view gets
  // squashed by a seafloor thousands of metres down. The sea then sits at the
  // blue end alongside the lowest land. Percentiles rather than min and max,
  // so one spike or pit does not set the scale.
  const sample = land.length >= all.length * 0.05 ? land : all;
  if (!sample.length) return;
  sample.sort((a, b2) => a - b2);
  const pick = q => sample[Math.min(sample.length - 1, Math.floor(q * sample.length))];
  let lo = Math.floor(pick(0.02) / 10) * 10;
  let hi = Math.ceil(pick(0.98) / 10) * 10;
  if (hi - lo < 20) {
    const mid = (lo + hi) / 2;
    lo = mid - 10;
    hi = mid + 10;
  }
  const key = `${lo},${hi}`;
  if (key === reliefApplied) return;
  reliefApplied = key;
  map.setPaintProperty("relief", "color-relief-color", reliefColorExpr(lo, hi));
  document.getElementById("relief-min").textContent = `${lo.toLocaleString()} m`;
  document.getElementById("relief-max").textContent = `${hi.toLocaleString()} m`;
}

// Positions along the range to put a stop at: every anchor, so each colour
// lands exactly where it is placed, plus an even grid so the OKLab mixing
// between anchors survives MapLibre's straight-line interpolation.
function reliefStops() {
  const ts = [...new Set([
    ...RELIEF_ANCHORS.map(([t]) => t),
    ...Array.from({ length: RELIEF_STEPS + 1 }, (_, i) => i / RELIEF_STEPS),
  ])].sort((a, b) => a - b);
  // interpolate needs strictly ascending inputs; drop float near-duplicates.
  return ts.filter((t, i) => i === 0 || t - ts[i - 1] > 1e-6);
}

function reliefColorExpr(lo, hi) {
  const stops = [];
  for (const t of reliefStops()) stops.push(lo + (hi - lo) * t, reliefColor(t));
  return ["interpolate", ["linear"], ["elevation"], ...stops];
}

function reliefGradientCss() {
  const parts = reliefStops().map(t => `${reliefColor(t)} ${(100 * t).toFixed(1)}%`);
  return `linear-gradient(to right, ${parts.join(", ")})`;
}

// t in 0..1 along the positioned anchors, mixed in OKLab.
function reliefColor(t) {
  let seg = 0;
  while (seg < RELIEF_ANCHORS.length - 2 && t > RELIEF_ANCHORS[seg + 1][0]) seg++;
  const [t0, hex0] = RELIEF_ANCHORS[seg];
  const [t1, hex1] = RELIEF_ANCHORS[seg + 1];
  const f = Math.max(0, Math.min(1, (t - t0) / (t1 - t0)));
  const a = toOklab(hexToRgb(hex0));
  const c = toOklab(hexToRgb(hex1));
  const [r, g, bl] = fromOklab(a.map((v, i) => v + (c[i] - v) * f));
  return `rgb(${r},${g},${bl})`;
}

function hexToRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function srgbToLinear(c) {
  c /= 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(c) {
  const v = c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
  return Math.round(Math.max(0, Math.min(1, v)) * 255);
}

function toOklab(rgb) {
  const [R, G, B] = rgb.map(srgbToLinear);
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return [
    0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
  ];
}

function fromOklab([L, A, B]) {
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
  const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
  const s = (L - 0.0894841775 * A - 1.2914855480 * B) ** 3;
  return [
    linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s),
  ];
}

function renderLayer(geojson) {
  // Density (people / km²) — recomputed on every load so it survives level switches.
  for (const f of geojson.features) {
    const latest = latestPop(f.properties.populations);
    f.properties.density = (latest && f.properties.area_km2)
      ? latest.pop / f.properties.area_km2 : 0;
  }
  state.hoverCode = null;
  hideTip();
  if (map.getSource(SRC)) {
    map.getSource(SRC).setData(geojson);
    applyGroupFilter();
    reapplyFeatureStates();
    updateOverlayVisibility();
    return;
  }
  map.addSource(SRC, { type: "geojson", data: geojson, promoteId: "code" });

  // Density-based fill (log scale). People / km². Falls back to flat color when no pop.
  map.addLayer({
    id: FILL, type: "fill", source: SRC,
    paint: {
      "fill-color": [
        "case",
        ["==", ["coalesce", ["get", "density"], 0], 0], "#cccccc",
        ["interpolate", ["linear"], ["log10", ["get", "density"]],
          0,     "#3b6fb5",   //      1 /km²  — blue
          1,     "#1b7a3d",   //     10 /km²  — dark green
          1.845, "#9ace48",   //     70 /km²  — light green
          2.602, "#f4e01f",   //    400 /km²  — yellow
          3,     "#f5901e",   //   1000 /km²  — orange
          3.204, "#e02020",   //   1600 /km²  — red
          3.699, "#ff45e0",   //   5000 /km²  — pink
          4.079, "#8e24aa",   //  12000 /km²  — purple
          4.602, "#2233cc",   //  40000 /km²  — blue
        ],
      ],
      "fill-opacity": fillOpacityExpr(),
    },
  });
  map.addLayer({
    id: LINE, type: "line", source: SRC,
    paint: { "line-color": "#333", "line-width": 0.4 },
  });
  // Highlight layer on top — driven by feature-state, so hover is a cheap
  // setFeatureState instead of a per-mousemove setFilter (which queued and lagged).
  map.addLayer({
    id: HL, type: "line", source: SRC,
    paint: {
      "line-color": ["case",
        ["boolean", ["feature-state", "hover"], false], "#ff8c00",
        ["boolean", ["feature-state", "selected"], false], "#1d4ed8",
        ["boolean", ["feature-state", "active"], false], "#dc2626",
        "#000"],
      "line-width": ["case",
        ["boolean", ["feature-state", "hover"], false], 3,
        ["boolean", ["feature-state", "selected"], false], 2.5,
        ["boolean", ["feature-state", "active"], false], 2.5,
        0],
    },
  });

  applyGroupFilter();
  reapplyFeatureStates();
  updateOverlayVisibility();
  applyBasemap();
  applyFillOpacity();
}

// ---- interactions ----

function onFeatureClick(e) {
  if (!e.features.length) return;
  const code = e.features[0].properties.code;
  const feature = state.featuresByCode.get(code);
  if (e.originalEvent.shiftKey) {
    toggleSelection(feature);
  } else {
    clearSelection();
    setActive(code);
    showFeatureInfo(feature);
  }
}

// Hover / active highlight via feature-state (source has promoteId "code").
function setHover(code) {
  if (code === state.hoverCode) return;
  if (state.hoverCode != null)
    map.setFeatureState({ source: SRC, id: state.hoverCode }, { hover: false });
  state.hoverCode = code;
  if (code != null)
    map.setFeatureState({ source: SRC, id: code }, { hover: true });
}

function setActive(code) {
  if (state.activeCode != null && state.activeCode !== code)
    map.setFeatureState({ source: SRC, id: state.activeCode }, { active: false });
  state.activeCode = code;
  map.setFeatureState({ source: SRC, id: code }, { active: true });
}

// setData drops feature-state; restore highlights for features still present.
function reapplyFeatureStates() {
  for (const s of state.selection) {
    if (state.featuresByCode.has(s.code))
      map.setFeatureState({ source: SRC, id: s.code }, { selected: true });
  }
  if (state.activeCode && state.featuresByCode.has(state.activeCode))
    map.setFeatureState({ source: SRC, id: state.activeCode }, { active: true });
}

// ---- hover tooltip ----

// Name, the divisions it sits inside, and population, under the cursor. It
// reads the feature from featuresByCode rather than the event, because
// MapLibre hands event features back with nested properties like populations
// flattened to JSON strings. The body only re-renders when the region changes;
// within one it just follows.
const tip = document.getElementById("hover-tip");
let tipCode = null;
let tipComplete = true;   // false if an ancestor level was still loading last render

// code -> properties for one coarser level, built once per loaded geojson. Keyed
// on the geojson object, so a country switch (new overlays) drops it for free.
const ancestorIndexes = new WeakMap();

function ancestorIndex(lvl) {
  const geo = state.overlayGeo[lvl];
  if (!geo) return null;
  let idx = ancestorIndexes.get(geo);
  if (!idx) {
    idx = new Map(geo.features.map(f => [String(f.properties.code), f.properties]));
    ancestorIndexes.set(geo, idx);
  }
  return idx;
}

// Parent, grandparent and so on up to adm1, nearest first, by walking
// parent_code through the coarser levels' borders. Those are already loaded
// whenever the working level is below them; if one is still in flight the walk
// stops short and says so, and the tip fills in on the next move.
function ancestors(props, level) {
  const names = [];
  let parent = props.parent_code;
  for (let lvl = level - 1; lvl >= 1 && parent != null; lvl--) {
    const idx = ancestorIndex(lvl);
    const up = idx && idx.get(String(parent));
    if (!up) return { names, complete: false };
    names.push(up.name);
    parent = up.parent_code;
  }
  return { names, complete: true };
}

function showTip(code, x, y) {
  const feature = state.featuresByCode.get(code);
  if (!feature) return hideTip();
  if (code !== tipCode || !tipComplete) {
    const p = feature.properties;
    const chain = ancestors(p, state.level);
    tipCode = code;
    tipComplete = chain.complete;
    const est = estimate(p.populations);
    const raw = latestPop(p.populations);
    let html = `<div class="tip-name">${escapeHtml(p.name)}` +
      (p.name_cn ? ` <span class="tip-cn">${escapeHtml(p.name_cn)}</span>` : "") +
      `</div>`;
    for (const name of chain.names) html += `<div class="tip-parent">${escapeHtml(name)}</div>`;
    if (est) {
      html += `<div class="tip-pop">${fmt(est.pop)} <span class="tip-note">est. ${CURRENT_YEAR}</span></div>`;
    } else if (raw) {
      html += `<div class="tip-pop">${fmt(raw.pop)} <span class="tip-note">${raw.year}</span></div>`;
    }
    html += compTipHtml(code);
    tip.innerHTML = html;
  }
  tip.hidden = false;
  // Below and right of the cursor, flipped when that would run off the window.
  const pad = 14;
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  tip.style.left = `${x + pad + w > window.innerWidth ? x - pad - w : x + pad}px`;
  tip.style.top = `${y + pad + h > window.innerHeight ? y - pad - h : y + pad}px`;
}

function hideTip() {
  tip.hidden = true;
  tipCode = null;
}

function showFeatureInfo(feature) {
  const p = feature.properties;
  const pops = p.populations || {};
  const est = estimate(pops);
  const shown = state.country.meta.display_years || DISPLAY_YEARS;
  const years = Object.keys(pops).map(Number)
    .filter(y => shown.includes(y)).sort((a, b) => a - b);

  let html = `<div class="name">${escapeHtml(p.name)}</div>`;
  if (p.name_cn) html += `<div class="parent">${escapeHtml(p.name_cn)}</div>`;
  if (p.parent_name) html += `<div class="parent">${escapeHtml(p.parent_name)}</div>`;
  if (est) {
    html += `<div class="estimate">${fmt(est.pop)}</div>`;
    html += `<div class="estimate-note">est. ${CURRENT_YEAR}, linear from ${est.from_year} → ${est.to_year}</div>`;
    if (est.flags.length) {
      html += `<div class="flag">⚠ ${est.flags.join("; ")}</div>`;
    }
  } else {
    html += `<div class="empty">No population data.</div>`;
  }
  if (years.length) {
    html += `<table class="history">`;
    for (const y of years) html += `<tr><td>${y}</td><td>${fmt(pops[y])}</td></tr>`;
    html += `</table>`;
  }
  if (p.area_km2) html += `<div class="parent">${fmt(p.area_km2)} km²</div>`;
  showInfo(html);
}

function showInfo(html) {
  document.getElementById("info-panel").innerHTML = html;
}

// ---- composition pies ----

// A country can ship countries/<id>/composition.json: for each admin unit, at
// whichever levels its source actually reaches, how its people divide between a
// fixed set of groups, and the mean position of those people. The viewer draws
// one pie per unit over the shape. Everything else is already here — the
// shapes, the level switch, the region filter, the hover — so this only has to
// draw the pies and add its rows to the tooltip. A level the file has nothing
// for simply draws nothing.
//
// MapLibre has no wedge, so the pies go on a canvas over the map rather than in
// the layer stack. Each is rendered once into an offscreen sprite keyed by unit,
// whole-pixel radius and legend state, so panning redraws a few hundred
// drawImage calls instead of a few thousand arcs.

const compCanvas = document.getElementById("comp-canvas");
const compCtx = compCanvas.getContext("2d");
const compSprites = new Map();
let compGen = 0;   // bumped when the legend changes, which retires every sprite

// Radius for a unit of the level's median size, against zoom. Every other unit
// is sqrt(its people / that median) times this, so a level of provinces and a
// level of counties both come out readable without a per-level setting.
const PIE_R = [[0, 2.5], [3, 5], [5, 8], [7, 13], [9, 19], [12, 28]];
const PIE_MIN_R = 2.5, PIE_MAX_R = 90;
// A group thinner than this folds away and the rest scale back up to fill the
// circle: at a 3 px radius the edge would otherwise be all slivers.
const PIE_MIN_WEDGE = 1.2;   // degrees
const PIE_EDGE = "rgba(255,255,255,0.85)";

function interpStops(stops, x) {
  let i = 0;
  while (i < stops.length - 2 && x > stops[i + 1][0]) i++;
  const [x0, v0] = stops[i], [x1, v1] = stops[i + 1];
  const t = Math.max(0, Math.min(1, (x - x0) / (x1 - x0)));
  return v0 + t * (v1 - v0);
}

async function loadComposition() {
  state.comp = null;
  compSprites.clear();
  const file = state.country.meta.composition;
  if (file) {
    try {
      state.comp = await (await fetchNoCache(
        `countries/${state.country.id}/${file}`)).json();
    } catch (e) {
      state.comp = null;   // declared but not built yet; the panel just stays hidden
    }
  }
  const saved = (loadView(state.country.id) || {}).comp || {};
  state.compOn = saved.on !== false;
  state.compSize = Number(saved.size) > 0 ? Number(saved.size) : 1;
  state.compHidden = new Set(saved.hidden || []);
  const slider = document.getElementById("comp-size");
  slider.value = state.compSize;
  document.getElementById("comp-size-value").textContent = state.compSize.toFixed(1) + "×";
  renderCompPanel();
}

// Codes biggest first, so a small unit is painted over the city beside it and
// not under it, plus the level's median size. Cached on the loaded document.
function compOrder(lvl) {
  const doc = state.comp;
  doc._order = doc._order || {};
  if (!doc._order[lvl]) {
    const units = doc.levels[lvl];
    const codes = Object.keys(units).sort((a, b) => units[b].t - units[a].t);
    doc._order[lvl] = codes;
    doc._ref = doc._ref || {};
    doc._ref[lvl] = units[codes[Math.floor(codes.length / 2)]].t || 1;
  }
  return doc._order[lvl];
}

function compUnits() {
  if (!state.comp || !state.compOn || state.level == null) return null;
  return state.comp.levels[String(state.level)] || null;
}

// The part of a unit's breakdown still switched on, cached until the legend moves.
function compVisible(u) {
  if (u._gen === compGen) return u;
  const w = [];
  let sum = 0;
  for (let i = 0; i < u.g.length; i++) {
    const g = state.comp.groups[u.g[i]];
    if (state.compHidden.has(g.key)) continue;
    w.push([g.color, u.k[i]]);
    sum += u.k[i];
  }
  u._w = w;
  u._sum = sum;
  u._gen = compGen;
  return u;
}

function compSprite(code, u, r) {
  const key = `${code}:${r}:${compGen}`;
  const got = compSprites.get(key);
  if (got) return got;
  if (compSprites.size > 6000) compSprites.clear();
  const dpr = window.devicePixelRatio || 1;
  const d = 2 * (r + 1);
  const c = document.createElement("canvas");
  c.width = c.height = Math.ceil(d * dpr);
  const x = c.getContext("2d");
  x.scale(c.width / d, c.height / d);
  let keep = u._w.filter(([, k]) => k / u._sum * 360 >= PIE_MIN_WEDGE);
  if (!keep.length) keep = [u._w[0]];
  const kept = keep.reduce((a, [, k]) => a + k, 0);
  let a0 = -Math.PI / 2;
  for (const [color, k] of keep) {
    const a1 = a0 + k / kept * 2 * Math.PI;
    x.beginPath();
    x.moveTo(r + 1, r + 1);
    x.arc(r + 1, r + 1, r, a0, a1);
    x.closePath();
    x.fillStyle = color;
    x.fill();
    a0 = a1;
  }
  if (r > 2.2) {
    x.beginPath();
    x.arc(r + 1, r + 1, r, 0, 2 * Math.PI);
    x.strokeStyle = PIE_EDGE;
    x.lineWidth = 1;
    x.stroke();
  }
  compSprites.set(key, c);
  return c;
}

// Pie area follows the unit's own population, the number the rest of the tool
// works in, not the composition file's total. The two disagree where the census
// counts a development zone apart from the district it stands in, and the
// population panel is the one that has to be believed.
function unitPop(props) {
  const est = estimate(props.populations);
  if (est) return est.pop;
  const raw = latestPop(props.populations);
  return raw ? raw.pop : 0;
}

function drawPies() {
  const rect = map.getCanvas().getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const style = compCanvas.style;
  style.left = `${rect.left}px`;
  style.top = `${rect.top}px`;
  style.width = `${rect.width}px`;
  style.height = `${rect.height}px`;
  const W = Math.round(rect.width * dpr), H = Math.round(rect.height * dpr);
  if (compCanvas.width !== W || compCanvas.height !== H) {
    compCanvas.width = W;
    compCanvas.height = H;
  }
  compCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  compCtx.clearRect(0, 0, rect.width, rect.height);
  const units = compUnits();
  if (!units) return;
  const lvl = String(state.level);
  const ref = (compOrder(lvl), state.comp._ref[lvl]);
  const base = interpStops(PIE_R, map.getZoom()) * state.compSize;
  const filtered = state.shownGroups && state.groups.length
    && state.shownGroups.size !== state.groups.length;
  for (const code of compOrder(lvl)) {
    const f = state.featuresByCode.get(code);
    if (!f) continue;   // not loaded: a split level holds only the ticked regions
    const p = f.properties;
    if (filtered && p.group != null && !state.shownGroups.has(p.group)) continue;
    const u = units[code];
    const r = Math.min(PIE_MAX_R,
      Math.max(PIE_MIN_R, base * Math.sqrt((unitPop(p) || u.t) / ref)));
    const pt = map.project([u.x, u.y]);
    if (pt.x < -r || pt.y < -r || pt.x > rect.width + r || pt.y > rect.height + r) continue;
    if (!compVisible(u)._sum) continue;
    const img = compSprite(code, u, Math.max(2, Math.round(r)));
    const d = img.width / dpr;
    compCtx.drawImage(img, pt.x - d / 2, pt.y - d / 2, d, d);
  }
}

map.on("render", drawPies);

function compPct(v) {
  const p = v * 100;
  if (p >= 10) return `${p.toFixed(0)}%`;
  if (p >= 0.1) return `${p.toFixed(1)}%`;
  return "<0.1%";
}

// The rows the hover tooltip gets under the population. Percentages are of the
// unit's whole population as the composition source counted it, so they add up
// whatever is switched off in the legend.
function compTipHtml(code) {
  const units = compUnits();
  const u = units && units[code];
  if (!u) return "";
  let html = '<div class="tip-comp">';
  let shown = 0, more = 0;
  for (let i = 0; i < u.g.length; i++) {
    const g = state.comp.groups[u.g[i]];
    if (state.compHidden.has(g.key)) continue;
    if (shown >= 5) { more++; continue; }
    shown++;
    html += `<div><span class="sw" style="background:${g.color}"></span>` +
      `<span>${escapeHtml(g.en)}</span>` +
      `<span class="pc">${compPct(u.k[i] / u.t)}</span></div>`;
  }
  if (more) html += `<div class="tip-note">+ ${more} more</div>`;
  return `${html}</div>`;
}

function compRefresh() {
  compGen++;
  hideTip();
  drawPies();
}

function renderCompPanel() {
  const panel = document.getElementById("comp-panel");
  const doc = state.comp;
  if (!doc) { panel.hidden = true; return; }
  panel.hidden = false;
  document.getElementById("comp-toggle").checked = state.compOn;
  document.getElementById("comp-label").textContent = doc.label || "Composition";

  const levels = Object.keys(doc.levels).map(Number).sort((a, b) => a - b);
  const labels = levels.map(l =>
    (state.country.meta.admin_levels.find(x => x.level === l) || {}).label || `level ${l}`);
  document.getElementById("comp-note").textContent =
    `${doc.year ? doc.year + ", " : ""}${labels.join(", ").toLowerCase()}`;

  // Country-wide totals, from the coarsest level the file has, for the order of
  // the list and the share beside each name.
  if (!doc._nat) {
    const top = doc.levels[String(levels[0])];
    const tot = new Array(doc.groups.length).fill(0);
    for (const u of Object.values(top))
      u.g.forEach((g, i) => { tot[g] += u.k[i]; });
    doc._nat = tot;
    doc._natTotal = tot.reduce((a, b) => a + b, 0) || 1;
  }
  const host = document.getElementById("comp-list");
  host.innerHTML = "";
  const order = doc.groups.map((g, i) => i)
    .filter(i => doc._nat[i] > 0)
    .sort((a, b) => doc._nat[b] - doc._nat[a]);
  for (const i of order) {
    const g = doc.groups[i];
    const off = state.compHidden.has(g.key) ? " off" : "";
    host.insertAdjacentHTML("beforeend",
      `<div class="row${off}" data-key="${escapeHtml(g.key)}" ` +
      `title="${escapeHtml(g.cn || g.en)}">` +
      `<span class="sw" style="background:${g.color}"></span>` +
      `<span class="nm">${escapeHtml(g.en)}</span>` +
      `<span class="pc">${compPct(doc._nat[i] / doc._natTotal)}</span></div>`);
  }
}

document.getElementById("comp-toggle").addEventListener("change", (e) => {
  state.compOn = e.target.checked;
  compRefresh();
  saveView();
});
document.getElementById("comp-size").addEventListener("input", (e) => {
  state.compSize = Number(e.target.value);
  document.getElementById("comp-size-value").textContent = state.compSize.toFixed(1) + "×";
  drawPies();
  saveView();
});
document.getElementById("comp-list").addEventListener("click", (e) => {
  const row = e.target.closest(".row");
  if (!row || !state.comp) return;
  const key = row.dataset.key;
  // Shift or alt on a row isolates it, the way the region list's all/none do in bulk.
  if (e.shiftKey || e.altKey) {
    state.compHidden = new Set(state.comp.groups.map(g => g.key).filter(k => k !== key));
  } else if (state.compHidden.has(key)) {
    state.compHidden.delete(key);
  } else {
    state.compHidden.add(key);
  }
  for (const el of document.querySelectorAll("#comp-list .row"))
    el.classList.toggle("off", state.compHidden.has(el.dataset.key));
  compRefresh();
  saveView();
});
document.getElementById("comp-all").addEventListener("click", () => {
  state.compHidden.clear();
  renderCompPanel();
  compRefresh();
  saveView();
});
document.getElementById("comp-none").addEventListener("click", () => {
  if (!state.comp) return;
  state.compHidden = new Set(state.comp.groups.map(g => g.key));
  renderCompPanel();
  compRefresh();
  saveView();
});

// ---- population math ----

function latestPop(pops) {
  if (!pops) return null;
  const years = Object.keys(pops).map(Number).sort((a, b) => b - a);
  if (!years.length) return null;
  return { year: years[0], pop: pops[years[0]] };
}

function estimate(pops) {
  if (!pops) return null;
  const years = Object.keys(pops).map(Number).sort((a, b) => a - b);
  if (years.length < 2) return null;
  const [y1, y2] = [years[years.length - 2], years[years.length - 1]];
  const [p1, p2] = [pops[y1], pops[y2]];
  const slope = (p2 - p1) / (y2 - y1);
  const est = Math.round(p2 + slope * (CURRENT_YEAR - y2));

  // Heuristic flags for suspicious data.
  const flags = [];
  if (years.length >= 3) {
    const y0 = years[years.length - 3];
    const p0 = pops[y0];
    const slope_prev = (p1 - p0) / (y1 - y0);
    // Says what happened, not why. A flip can be a boundary change, but in
    // China it is usually the real turn from growth to decline after 2020.
    if (Math.sign(slope) !== Math.sign(slope_prev) && Math.abs(slope) > 0 && Math.abs(slope_prev) > 0) {
      flags.push("direction changed between periods");
    } else if (Math.abs(slope) > 3 * Math.max(Math.abs(slope_prev), 1)) {
      flags.push("growth rate shifted sharply");
    }
  }
  return { pop: est, from_year: y1, to_year: y2, flags };
}

// ---- selection / calculator ----

function toggleSelection(feature) {
  const code = feature.properties.code;
  const existing = state.selection.findIndex(s => s.code === code);
  if (existing >= 0) {
    state.selection.splice(existing, 1);
    map.setFeatureState({ source: SRC, id: code }, { selected: false });
  } else {
    const est = estimate(feature.properties.populations);
    const raw = latestPop(feature.properties.populations);
    state.selection.push({
      code,
      name: feature.properties.name,
      raw_year: raw?.year, raw_pop: raw?.pop,
      est_pop: est?.pop,
    });
    map.setFeatureState({ source: SRC, id: code }, { selected: true });
  }
  renderSelection();
}

function renderSelection() {
  const countEl = document.getElementById("selection-count");
  const summaryEl = document.getElementById("selection-summary");
  const listEl = document.getElementById("selection-list");

  countEl.textContent = state.selection.length;
  listEl.innerHTML = "";
  if (!state.selection.length) {
    summaryEl.className = "empty";
    summaryEl.innerHTML = "Shift-click regions to add.";
    return;
  }
  // With only one year there is nothing to extrapolate from, so the counted
  // figure stands in — otherwise the running sum reads zero.
  const sumEst = state.selection.reduce((a, s) => a + (s.est_pop ?? s.raw_pop ?? 0), 0);
  const sumRaw = state.selection.reduce((a, s) => a + (s.raw_pop || 0), 0);
  const anyEst = state.selection.some(s => s.est_pop != null);
  summaryEl.className = "";
  summaryEl.innerHTML =
    `<div class="sum">${fmt(sumEst)}</div>` +
    `<div class="estimate-note">${anyEst ? `estimated ${CURRENT_YEAR}  ·  ` : ""}` +
    `raw sum ${fmt(sumRaw)}</div>`;
  for (const s of state.selection) {
    const li = document.createElement("li");
    li.innerHTML =
      `<span>${escapeHtml(s.name)}</span>` +
      `<span>${fmt(s.est_pop || s.raw_pop || 0)} <button class="sel-remove" data-code="${s.code}">×</button></span>`;
    listEl.appendChild(li);
  }
}

function clearSelection() {
  for (const s of state.selection) {
    map.setFeatureState({ source: SRC, id: s.code }, { selected: false });
  }
  state.selection = [];
  renderSelection();
}

document.getElementById("clear-selection").addEventListener("click", clearSelection);

document.getElementById("selection-list").addEventListener("click", (e) => {
  const code = e.target?.dataset?.code;
  if (!code) return;
  const feature = state.featuresByCode.get(code);
  if (feature) toggleSelection(feature);
});

document.getElementById("group-list").addEventListener("change", (e) => {
  if (e.target.type !== "checkbox") return;
  if (e.target.checked) state.shownGroups.add(e.target.value);
  else state.shownGroups.delete(e.target.value);
  applyGroupFilter();
  saveView();
});
document.getElementById("groups-all").addEventListener("click", () => {
  state.shownGroups = new Set(state.groups.map(g => g.code));
  renderGroupFilter();
  applyGroupFilter();
  saveView();
});
document.getElementById("groups-none").addEventListener("click", () => {
  state.shownGroups = new Set();
  renderGroupFilter();
  applyGroupFilter();
  saveView();
});

document.getElementById("basemap-radios").addEventListener("change", () => {
  applyBasemap();
  saveView();
});
document.getElementById("fill-opacity").addEventListener("input", applyFillOpacity);
// Save on release rather than on every pixel of the drag.
document.getElementById("fill-opacity").addEventListener("change", saveView);

// ---- sidebar resize ----

// Drag the sidebar's edge; double-click it to go back to the default. The map
// canvas only follows the new width after map.resize(), which is batched to one
// call a frame while dragging.
(() => {
  const handle = document.getElementById("sidebar-resizer");
  let dragging = false;
  let queued = false;
  const resizeMap = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; map.resize(); });
  };
  const stop = () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    document.body.classList.remove("resizing");
    map.resize();
    saveView();
  };
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    dragging = true;
    handle.classList.add("dragging");
    document.body.classList.add("resizing");
    handle.setPointerCapture(e.pointerId);
  });
  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    setSidebarWidth(e.clientX);
    resizeMap();
  });
  handle.addEventListener("pointerup", stop);
  handle.addEventListener("pointercancel", stop);
  handle.addEventListener("lostpointercapture", stop);
  handle.addEventListener("dblclick", () => {
    setSidebarWidth(SIDEBAR_DEFAULT);
    map.resize();
    saveView();
  });
  // A narrower window can leave the saved width too wide for it.
  window.addEventListener("resize", () => {
    setSidebarWidth(parseInt(document.getElementById("sidebar").style.width, 10));
    map.resize();
  });
})();

// ---- hotkeys ----

// 1-4 pick the admin level, where the country has that level. Ignored while
// typing into a control, and with a modifier held so browser shortcuts still work.
document.addEventListener("keydown", async (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat || !state.country) return;
  const tag = e.target.tagName;
  if (tag === "SELECT" || tag === "TEXTAREA" ||
      (tag === "INPUT" && !["checkbox", "radio", "range"].includes(e.target.type))) return;
  const level = parseInt(e.key, 10);
  if (!levelCfg(level) || level === state.level) return;
  e.preventDefault();
  await setLevel(level);
  saveView();
});

// ---- UI wiring ----

function renderLevelRadios() {
  const host = document.getElementById("level-radios");
  host.innerHTML = "";
  for (const l of state.country.meta.admin_levels) {
    const id = `level-${l.level}`;
    host.insertAdjacentHTML("beforeend",
      `<label title="Hotkey ${l.level}"><input type="radio" name="level" value="${l.level}" id="${id}"> ${escapeHtml(l.label)}</label>`);
  }
}

// Registered once. It used to be added inside renderLevelRadios, which runs on
// every country switch, so each switch stacked another copy of this handler and
// one click loaded the level several times over.
document.getElementById("level-radios").addEventListener("change", async (e) => {
  if (e.target.name !== "level") return;
  await setLevel(parseInt(e.target.value));
  saveView();
});

async function renderCountrySelector(countries) {
  const sel = document.getElementById("country-select");
  sel.innerHTML = "";
  for (const c of countries) {
    sel.insertAdjacentHTML("beforeend", `<option value="${c.id}">${escapeHtml(c.name)}</option>`);
  }
  sel.addEventListener("change", () => loadCountry(sel.value));
}

// ---- utils ----

function fmt(n) {
  if (n == null || !isFinite(n)) return "—";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return Math.round(n).toString();
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- boot ----

(async () => {
  const countries = await loadCountriesIndex();
  await renderCountrySelector(countries);
  if (!countries.length) return;

  // Basemap settings are global rather than per country, so they go on first.
  // A view saved before the elevation option existed has a topo checkbox value.
  const saved = readStore();
  const basemap = BASEMAPS.includes(saved.basemap) ? saved.basemap
    : saved.topo === true ? "topo" : "streets";
  document.querySelector(`#basemap-radios input[value="${basemap}"]`).checked = true;
  if (saved.opacity != null)
    document.getElementById("fill-opacity").value = saved.opacity;
  document.getElementById("relief-ramp").style.background = reliefGradientCss();
  applyBasemap();
  applyFillOpacity();

  const start = countries.some(c => c.id === saved.last) ? saved.last : countries[0].id;
  document.getElementById("country-select").value = start;
  await loadCountry(start);
})();
