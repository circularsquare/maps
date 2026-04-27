// helper1m viewer — country-agnostic, reads countries.json + countries/<id>/meta.json
// then loads the current admin level's geojson and lets the user click / shift-click.

const CURRENT_YEAR = new Date().getFullYear();
const state = {
  country: null,   // {id, name, meta}
  level: null,     // int
  featuresByCode: new Map(),
  selection: [],   // [{code, level, name, raw_pop, est_pop}]
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
    },
    layers: [{ id: "osm", type: "raster", source: "osm", paint: { "raster-opacity": 0.55 } }],
  },
  center: [0, 0],
  zoom: 2,
});

// ---- data loading ----

async function loadCountriesIndex() {
  const res = await fetch("countries.json");
  return (await res.json()).countries;
}

async function loadCountry(id) {
  const meta = await (await fetch(`countries/${id}/meta.json`)).json();
  state.country = { id, name: meta.name, meta };
  map.flyTo({ center: meta.center, zoom: meta.zoom, duration: 0 });
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
    geo = await (await fetch(url)).json();
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

// ---- map layer ----

const SRC = "admin";
const FILL = "admin-fill";
const LINE = "admin-line";
const HOVER = "admin-hover";

function renderLayer(geojson) {
  if (map.getSource(SRC)) {
    map.getSource(SRC).setData(geojson);
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
          0, "#ffffcc",
          1, "#a1dab4",
          2, "#41b6c4",
          3, "#2c7fb8",
          4, "#253494",
        ],
      ],
      "fill-opacity": [
        "case",
        ["boolean", ["feature-state", "selected"], false], 0.75,
        0.55,
      ],
    },
  });
  map.addLayer({
    id: LINE, type: "line", source: SRC,
    paint: { "line-color": "#333", "line-width": 0.4 },
  });
  map.addLayer({
    id: HOVER, type: "line", source: SRC,
    paint: { "line-color": "#e94", "line-width": 2 },
    filter: ["==", ["get", "code"], ""],
  });

  // Compute density on client-side (avoids baking stale density into geojson).
  // Mutates feature properties of the loaded source.
  for (const f of geojson.features) {
    const latest = latestPop(f.properties.populations);
    if (latest && f.properties.area_km2) {
      f.properties.density = latest.pop / f.properties.area_km2;
    } else {
      f.properties.density = 0;
    }
  }
  map.getSource(SRC).setData(geojson);

  map.on("mousemove", FILL, (e) => {
    if (!e.features.length) return;
    map.getCanvas().style.cursor = "pointer";
    map.setFilter(HOVER, ["==", ["get", "code"], e.features[0].properties.code]);
  });
  map.on("mouseleave", FILL, () => {
    map.getCanvas().style.cursor = "";
    map.setFilter(HOVER, ["==", ["get", "code"], ""]);
  });
  map.on("click", FILL, onFeatureClick);
}

// ---- interactions ----

function onFeatureClick(e) {
  if (!e.features.length) return;
  const code = e.features[0].properties.code;
  const feature = state.featuresByCode.get(code);
  if (e.originalEvent.shiftKey) {
    toggleSelection(feature);
  } else {
    showFeatureInfo(feature);
  }
}

function showFeatureInfo(feature) {
  const p = feature.properties;
  const pops = p.populations || {};
  const est = estimate(pops);
  const years = Object.keys(pops).map(Number).sort((a, b) => a - b);

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

document.getElementById("clear-selection").addEventListener("click", () => {
  for (const s of state.selection) {
    map.setFeatureState({ source: SRC, id: s.code }, { selected: false });
  }
  state.selection = [];
  renderSelection();
});

document.getElementById("selection-list").addEventListener("click", (e) => {
  const code = e.target?.dataset?.code;
  if (!code) return;
  const feature = state.featuresByCode.get(code);
  if (feature) toggleSelection(feature);
});

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
