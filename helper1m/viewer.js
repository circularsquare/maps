// helper1m viewer — country-agnostic, reads countries.json + countries/<id>/meta.json
// then loads the current admin level's geojson and lets the user click / shift-click.

const CURRENT_YEAR = new Date().getFullYear();
const DISPLAY_YEARS = [2011, 2026];  // years shown in the detail panel history table
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
};

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
    },
    layers: [
      { id: "osm", type: "raster", source: "osm", paint: { "raster-opacity": 0.55 } },
      { id: "topo", type: "raster", source: "topo",
        layout: { visibility: "none" }, paint: { "raster-opacity": 0.8 } },
    ],
  },
  center: [0, 0],
  zoom: 2,
});

// Shift-click is our multi-select; free it from MapLibre's box-zoom handler.
map.boxZoom.disable();

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
  state.country = { id, name: meta.name, meta };
  map.flyTo({ center: meta.center, zoom: meta.zoom, duration: 0 });
  await loadOverlays();
  loadGroups();
  renderGroupFilter();
  renderLevelRadios();
  // default to the finest level with a geojson already built
  const lvl = meta.admin_levels.find(l => l).level;
  await setLevel(lvl);
}

async function setLevel(level) {
  state.level = level;
  const cfg = state.country.meta.admin_levels.find(l => l.level === level);
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
  document.querySelectorAll("#level-radios input").forEach(r => {
    r.checked = parseInt(r.value) === level;
  });
}

// ---- state filter ----

function loadGroups() {
  state.groups = [];
  state.shownGroups = null;
  const geo = state.overlayGeo[1];   // adm1 — the state list
  if (!geo) return;
  state.groups = geo.features
    .map(f => ({ code: f.properties.code, name: f.properties.name }))
    .sort((a, b) => a.name.localeCompare(b.name));
  state.shownGroups = new Set(state.groups.map(g => g.code));
}

function renderGroupFilter() {
  const panel = document.getElementById("group-panel");
  const host = document.getElementById("group-list");
  if (!state.groups.length) { panel.style.display = "none"; return; }
  panel.style.display = "";
  host.innerHTML = "";
  for (const g of state.groups) {
    const on = state.shownGroups.has(g.code) ? "checked" : "";
    host.insertAdjacentHTML("beforeend",
      `<label><input type="checkbox" value="${escapeHtml(g.code)}" ${on}>${escapeHtml(g.name)}</label>`);
  }
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
  for (const id of [FILL, LINE, HL, OV_DIST + "-line", OV_STATE + "-line"]) {
    if (map.getLayer(id)) map.setFilter(id, expr);
  }
}

// ---- map layer ----

const SRC = "admin";
const FILL = "admin-fill";
const LINE = "admin-line";
const HL = "admin-highlight";
const OV_STATE = "ov-state";      // overlay source: state borders
const OV_DIST = "ov-district";    // overlay source: district borders

// Fill opacity for the density choropleth (set to 0 in topographic mode, so
// regions stay clickable but their colours don't fight the topo basemap).
const FILL_OPACITY = ["case",
  ["boolean", ["feature-state", "selected"], false], 0.75,
  0.55];

// Pointer handlers — registered once. They key off the layer id, so they keep
// working when the layer is torn down and rebuilt on a country switch.
map.on("mousemove", FILL, (e) => {
  if (!e.features.length) return;
  map.getCanvas().style.cursor = "pointer";
  setHover(e.features[0].properties.code);
});
map.on("mouseleave", FILL, () => {
  map.getCanvas().style.cursor = "";
  setHover(null);
});
map.on("click", FILL, onFeatureClick);

// A plain click on empty space (no division) clears the multi-selection.
map.on("click", (e) => {
  if (e.originalEvent.shiftKey || !map.getLayer(FILL)) return;
  if (!map.queryRenderedFeatures(e.point, { layers: [FILL] }).length) clearSelection();
});

// ---- boundary overlays (coarser levels drawn as context) ----

async function loadOverlays() {
  state.overlayGeo = {};
  for (const lvl of [1, 2]) {
    const cfg = state.country.meta.admin_levels.find(l => l.level === lvl);
    if (!cfg) continue;
    try {
      state.overlayGeo[lvl] =
        await (await fetchNoCache(`countries/${state.country.id}/${cfg.file}`)).json();
    } catch (e) { /* level not built */ }
  }
}

function addOverlay(srcId, geo, color, width) {
  if (!geo || map.getSource(srcId)) return;
  map.addSource(srcId, { type: "geojson", data: geo });
  map.addLayer({
    id: srcId + "-line", type: "line", source: srcId,
    layout: { visibility: "none" },
    paint: { "line-color": color, "line-width": width },
  });
}

// State borders show below state level; district borders below district level.
function updateOverlayVisibility() {
  const vis = (id, show) => {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", show ? "visible" : "none");
  };
  vis(OV_STATE + "-line", state.level >= 2);
  vis(OV_DIST + "-line", state.level >= 3);
}

function teardown() {
  for (const id of [FILL, LINE, HL, OV_DIST + "-line", OV_STATE + "-line"]) {
    if (map.getLayer(id)) map.removeLayer(id);
  }
  for (const id of [SRC, OV_DIST, OV_STATE]) {
    if (map.getSource(id)) map.removeSource(id);
  }
}

// Topographic mode swaps the basemap and hides the density fill (regions stay
// clickable — an opacity-0 fill still receives events).
function applyBasemap() {
  const topo = document.getElementById("topo-toggle").checked;
  map.setLayoutProperty("topo", "visibility", topo ? "visible" : "none");
  map.setLayoutProperty("osm", "visibility", topo ? "none" : "visible");
  if (map.getLayer(FILL))
    map.setPaintProperty(FILL, "fill-opacity", topo ? 0 : FILL_OPACITY);
}

function renderLayer(geojson) {
  // Density (people / km²) — recomputed on every load so it survives level switches.
  for (const f of geojson.features) {
    const latest = latestPop(f.properties.populations);
    f.properties.density = (latest && f.properties.area_km2)
      ? latest.pop / f.properties.area_km2 : 0;
  }
  state.hoverCode = null;
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
      "fill-opacity": FILL_OPACITY,
    },
  });
  map.addLayer({
    id: LINE, type: "line", source: SRC,
    paint: { "line-color": "#333", "line-width": 0.4 },
  });
  // Coarser-level borders as context — created hidden, toggled by level.
  addOverlay(OV_DIST, state.overlayGeo[2], "#222", 1.0);
  addOverlay(OV_STATE, state.overlayGeo[1], "#000", 1.4);
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

function showFeatureInfo(feature) {
  const p = feature.properties;
  const pops = p.populations || {};
  const est = estimate(pops);
  const years = Object.keys(pops).map(Number)
    .filter(y => DISPLAY_YEARS.includes(y)).sort((a, b) => a - b);

  let html = `<div class="name">${escapeHtml(p.name)}</div>`;
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
    if (Math.sign(slope) !== Math.sign(slope_prev) && Math.abs(slope) > 0 && Math.abs(slope_prev) > 0) {
      flags.push("sign flipped between periods (boundary change?)");
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
  const sumEst = state.selection.reduce((a, s) => a + (s.est_pop || 0), 0);
  const sumRaw = state.selection.reduce((a, s) => a + (s.raw_pop || 0), 0);
  summaryEl.className = "";
  summaryEl.innerHTML =
    `<div class="sum">${fmt(sumEst)}</div>` +
    `<div class="estimate-note">estimated ${CURRENT_YEAR}  ·  raw sum ${fmt(sumRaw)}</div>`;
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
});
document.getElementById("groups-all").addEventListener("click", () => {
  state.shownGroups = new Set(state.groups.map(g => g.code));
  renderGroupFilter();
  applyGroupFilter();
});
document.getElementById("groups-none").addEventListener("click", () => {
  state.shownGroups = new Set();
  renderGroupFilter();
  applyGroupFilter();
});

document.getElementById("topo-toggle").addEventListener("change", applyBasemap);

// ---- UI wiring ----

function renderLevelRadios() {
  const host = document.getElementById("level-radios");
  host.innerHTML = "";
  for (const l of state.country.meta.admin_levels) {
    const id = `level-${l.level}`;
    host.insertAdjacentHTML("beforeend",
      `<label><input type="radio" name="level" value="${l.level}" id="${id}"> ${escapeHtml(l.label)}</label>`);
  }
  host.addEventListener("change", (e) => {
    if (e.target.name === "level") setLevel(parseInt(e.target.value));
  });
}

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
  if (countries.length) await loadCountry(countries[0].id);
})();
