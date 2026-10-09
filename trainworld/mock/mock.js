// trainworld UI mock (T-009). Static fake data; nothing here is game code.

const root = document.documentElement;
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

// ---------- mock switcher state: URL beats saved beats default ----------

const STORE = "trainworld-mock";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch { saved = {}; }
const params = new URLSearchParams(location.search);
const look = {
  set: params.get("set") || saved.set || "a",
  font: params.get("font") || saved.font || "zen",
};

function applyLook() {
  root.dataset.set = look.set;
  root.dataset.font = look.font;
  for (const k of ["set", "font"])
    $$(`#mockbar [data-${k}]`).forEach(b => b.classList.toggle("on", b.dataset[k] === look[k]));
  try { localStorage.setItem(STORE, JSON.stringify(look)); } catch { /* private window */ }
}
applyLook();

$("#mockbar").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  for (const k of ["set", "font"]) if (b.dataset[k]) look[k] = b.dataset[k];
  applyLook();
  renderAll();
  paintMap();
});

const css = name => getComputedStyle(root).getPropertyValue(name).trim();
const lineHex = i => css(`--l${i}`);

function mix(a, b, t) {
  const p = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const A = p(a), B = p(b);
  return "#" + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, "0")).join("");
}
// Trains: a darker shade of the line's own hue (as in the T-002 spike), with a white edge.
function darker(hex) {
  let [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2, d = mx - mn;
  let h = 0, s = d ? d / (1 - Math.abs(2 * l - 1)) : 0;
  if (d) h = mx === r ? ((g - b) / d + 6) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  const L = l * 0.6, S = Math.min(1, s * 0.9);
  const C = (1 - Math.abs(2 * L - 1)) * S, X = C * (1 - Math.abs(h % 2 - 1)), m = L - C / 2;
  const [R, G, B] = [[C, X, 0], [X, C, 0], [0, C, X], [0, X, C], [X, 0, C], [C, 0, X]][Math.floor(h) % 6];
  return "#" + [R, G, B].map(v => Math.round((v + m) * 255).toString(16).padStart(2, "0")).join("");
}

// ---------- formatting ----------

const k = n => n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 1 : 2).replace(/\.?0+$/, "") + "M"
            : Math.round(n / 1000) + "k";
const money = n => {
  const a = Math.abs(n), s = n < 0 ? "−" : "";
  if (a >= 1e9) return s + "$" + (a / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return s + "$" + (a / 1e6).toFixed(2) + "M";
  return s + "$" + Math.round(a / 1000) + "k";
};
const headway = tph => {
  if (!tph) return "none";
  const m = 60 / tph;
  return (Number.isInteger(m) ? m : m.toFixed(1)) + " min";
};
const needed = (line, tph) => Math.ceil(line.rtt * tph / 60);
const runCost = line => {   // service hours: peak 6, shoulder 8, off-peak 6; $1,250 a train-hour
  const h = [6, 8, 6];
  return line.tph.reduce((s, t, i) => s + needed(line, t) * h[i], 0) * 1250;
};

// ---------- state ----------

let sel = LINES[0];
let tab = "lines";
let level = "-1";
let sameAllDay = false;

const linesAt = id => LINES.filter(l => l.stops.includes(id));
// Letter colour on a line chip: dark on light lines (yellow), white on the rest.
function inkOn(hex) {
  const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
  return 0.299 * r + 0.587 * g + 0.114 * b > 0.62 ? "#2b2b2b" : "#ffffff";
}
const chip = (l, cls = "") => {
  const c = lineHex(l.colour);
  return `<span class="chip ${cls}" style="--c:${c};--ct:${inkOn(c)}">${l.id}</span>`;
};

const ICON = {
  minus: `<svg viewBox="0 0 16 16"><path d="M4 8h8"/></svg>`,
  plus: `<svg viewBox="0 0 16 16"><path d="M4 8h8M8 4v8"/></svg>`,
  pencil: `<svg viewBox="0 0 16 16"><path d="M10.5 3l2.5 2.5L6 12.5H3.5V10z"/></svg>`,
};
const stepper = (key, value, wide = false, disabled = false) =>
  `<span class="stepper${wide ? " wide" : ""}">` +
  `<button class="btn" data-step="${key}" data-d="-1"${disabled ? " disabled" : ""}>${ICON.minus}</button>` +
  `<span class="v">${value}</span>` +
  `<button class="btn" data-step="${key}" data-d="1"${disabled ? " disabled" : ""}>${ICON.plus}</button></span>`;

// ---------- dock panes ----------

function renderLinesPane() {
  const total = LINES.reduce((s, l) => s + l.riders, 0);
  $("#pane-lines").innerHTML =
    `<ul class="rows lines-list">` +
    LINES.map(l => `<li data-line="${l.id}" class="${l === sel ? "on" : ""}">${chip(l)}` +
      `<span class="grow">${l.name}</span>` +
      `<span class="val">${k(l.riders)}</span>` +
      `<span class="val muted" style="width:68px">${needed(l, l.tph[0])} trains</span></li>`).join("") +
    `</ul><div class="pane-foot"><span class="muted">${k(total)} riders a day</span>` +
    `<button class="btn text">${ICON.plus}New line</button></div>`;
}

function renderStationsPane() {
  const tot = {};
  for (const l of LINES) l.stops.forEach((s, i) => tot[s] = (tot[s] || 0) + l.board[i] * 1000);
  const ids = Object.keys(tot).sort((a, b) => tot[b] - tot[a]);
  $("#pane-stations").innerHTML =
    `<ul class="rows">` + ids.slice(0, 8).map(id =>
      `<li><span class="grow">${STATIONS[id][0]}</span>` +
      linesAt(id).map(l => chip(l, "sm")).join("") +
      `<span class="val" style="width:44px">${k(tot[id])}</span></li>`).join("") +
    `</ul><div class="pane-foot"><span class="muted">${ids.length} stations</span>` +
    `<button class="btn text">${ICON.plus}New station</button></div>`;
}

function renderMoneyPane() {
  const net = MONEY.rows.reduce((s, r) => s + r[1], 0);
  $("#pane-money").innerHTML =
    `<div class="h">Each day<span class="right muted">cash ${money(MONEY.cash)}</span></div>` +
    `<ul class="rows">` + MONEY.rows.map(([n, v]) =>
      `<li><span class="grow">${n}</span><span class="val">${money(v)}</span></li>`).join("") +
    `<li><b class="grow">Left over</b><b class="val">${money(net)}</b></li></ul>`;
}

function renderCityPane() {
  const max = Math.max(...CITY.periods.map(p => p[1]));
  $("#pane-city").innerHTML =
    `<ul class="rows"><li><span class="grow">People</span><span class="val">${CITY.population}</span></li>` +
    `<li><span class="grow">Jobs</span><span class="val">${CITY.jobs}</span></li></ul>` +
    `<div class="h">Trips by train</div><ul class="rows">` +
    CITY.periods.map(([n, v]) =>
      `<li><span style="width:96px">${n}</span><span class="grow"><i class="seg seg-train" style="display:inline-block;height:8px;width:${v / max * 100}%"></i></span>` +
      `<span class="val" style="width:44px">${v.toFixed(1)}%</span></li>`).join("") + `</ul>`;
}

function renderBuildPane() {
  $("#pane-build").innerHTML =
    `<div class="h">Track per km</div><ul class="rows">` +
    ["3", "2", "1", "0", "-1", "-2", "-3"].map(lv =>
      `<li class="${lv === level ? "on" : ""}"><span style="width:28px" class="num">${lv > 0 ? "+" + lv : lv.replace("-", "−")}</span>` +
      `<span class="grow">${LEVEL_NAME[lv]}</span><span class="val">$${LEVEL_COST[lv]}M</span></li>`).join("") +
    `</ul><div class="pane-foot"><span class="muted">Station $60M. Track over water costs 5 times as much.</span></div>`;
}

// ---------- inspector: the selected line ----------

function renderInspector() {
  const l = sel, c = lineHex(l.colour);
  const periods = ["Peak", "Shoulder", "Off-peak"];
  $("#inspector").innerHTML =
    `<div class="title">${chip(l, "lg")}<span class="name">${l.name}</span>` +
    `<button class="swatch" data-colour style="--c:${c}" title="Line colour"></button>` +
    `<button class="btn icon" title="Rename">${ICON.pencil}</button></div>` +
    `<div class="sub muted">${l.stops.length} stops, ${l.km} km</div>` +

    `<div class="h" style="margin-top:16px">Schedule</div>` +
    `<table class="sched"><tr><th></th><th>Trains an hour</th><th>Headway</th><th>Trains</th></tr>` +
    periods.map((p, i) => {
      const off = sameAllDay && i > 0;
      return `<tr class="${off ? "off" : ""}"><td>${p}</td><td>${stepper("tph" + i, l.tph[i], false, off)}</td>` +
        `<td>${headway(l.tph[i])}</td><td>${needed(l, l.tph[i])}</td></tr>`;
    }).join("") + `</table>` +
    `<label class="check"><input type="checkbox" id="same"${sameAllDay ? " checked" : ""}>Same all day</label>` +
    `<ul class="rows" style="margin-top:4px"><li><span class="grow">Round trip</span><span class="val">${l.rtt} min</span></li>` +
    `<li><span class="grow">Running cost a day</span><span class="val">${money(runCost(l))}</span></li></ul>` +

    `<div class="h">Riders</div><ul class="rows">` +
    `<li><span class="grow">Riders a day</span><span class="val">${k(l.riders)}</span></li>` +
    `<li><span class="grow">Fullest train at peak</span><span class="val">${l.fullest}%</span></li>` +
    `<li><span class="grow">Fare</span>${stepper("fare", "$" + l.fare.toFixed(2), true)}</li>` +
    `<li><span class="grow">Fares a day</span><span class="val">${money(l.riders * l.fare)}</span></li></ul>` +

    `<div class="h">Stops<span class="right muted small">boardings a day</span></div>` +
    `<ul class="stops" style="--c:${c}">` + l.stops.map((s, i) => {
      const others = linesAt(s).filter(o => o !== l);
      return `<li class="${others.length ? "xfer" : ""}"><span class="grow">${STATIONS[s][0]}</span>` +
        others.map(o => chip(o, "sm")).join("") +
        `<span class="val num" style="width:40px;text-align:right">${l.board[i]}k</span></li>`;
    }).join("") + `</ul>`;
}

function renderAll() {
  renderLinesPane(); renderStationsPane(); renderMoneyPane(); renderCityPane(); renderBuildPane();
  renderInspector();
}
renderAll();

// ---------- dock events ----------

$(".tabs").addEventListener("click", e => {
  const b = e.target.closest(".tab[data-tab]");
  if (!b) return;
  tab = b.dataset.tab;
  $$(".tab[data-tab]").forEach(t => t.classList.toggle("on", t === b));
  $$(".pane").forEach(p => p.hidden = p.id !== "pane-" + tab);
});

$("#pane-lines").addEventListener("click", e => {
  const li = e.target.closest("[data-line]");
  if (li) select(LINES.find(l => l.id === li.dataset.line));
});

function select(l) {
  sel = l;
  sameAllDay = l.tph[0] === l.tph[1] && l.tph[1] === l.tph[2];
  renderLinesPane(); renderInspector(); drawNetwork();
}

$("#inspector").addEventListener("click", e => {
  const st = e.target.closest("[data-step]");
  if (st && !st.disabled) {
    const d = +st.dataset.d, key = st.dataset.step;
    if (key === "fare") sel.fare = Math.max(0, Math.round((sel.fare + d * 0.1) * 100) / 100);
    else {
      const i = +key.slice(3);
      sel.tph[i] = Math.max(0, Math.min(30, sel.tph[i] + d));
      if (sameAllDay) sel.tph = [sel.tph[0], sel.tph[0], sel.tph[0]];
      drawNetwork();
    }
    renderInspector(); renderLinesPane();
    return;
  }
  const sw = e.target.closest("[data-colour]");
  if (sw) { sel.colour = sel.colour % 8 + 1; renderAll(); drawNetwork(); }   // mock: cycles; the game gets a picker
});
$("#inspector").addEventListener("change", e => {
  if (e.target.id !== "same") return;
  sameAllDay = e.target.checked;
  if (sameAllDay) sel.tph = [sel.tph[0], sel.tph[0], sel.tph[0]];
  renderInspector(); renderLinesPane(); drawNetwork();
});

// ---------- tools ----------

$("#tools").addEventListener("click", e => {
  const t = e.target.closest("[data-tool]");
  if (t) $$("[data-tool]").forEach(b => b.classList.toggle("on", b === t));
  const lv = e.target.closest("[data-level]");
  if (lv) {
    level = lv.dataset.level;
    $$("[data-level]").forEach(b => b.classList.toggle("on", b === lv));
    $(".level-cost").textContent = `${LEVEL_NAME[level]} at $${LEVEL_COST[level]}M per km`;
    renderBuildPane();
  }
});
$(".level-cost").textContent = `${LEVEL_NAME[level]} at $${LEVEL_COST[level]}M per km`;

// ---------- clock ----------

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
let minute = 8 * 60 + 42, day = 38, speed = 1, timer = null;
function periodName(h) {
  if (h >= 6 && h < 10) return "Morning peak";
  if (h >= 10 && h < 16) return "Midday";
  if (h >= 16 && h < 19) return "Evening peak";
  if (h >= 19 && h < 23) return "Evening";
  return "Night";
}
function drawClock() {
  const h = Math.floor(minute / 60), m = minute % 60;
  $(".time").textContent = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  $(".day").textContent = `${DAYS[day % 7]}, day ${day}`;
  $(".period").textContent = periodName(h);
}
function setSpeed(s) {
  speed = s;
  $$("[data-speed]").forEach(b => b.classList.toggle("on", +b.dataset.speed === s));
  clearInterval(timer);
  timer = s && !document.hidden ? setInterval(() => {
    minute += speed;
    if (minute >= 1440) { minute -= 1440; day++; }
    drawClock();
  }, 1000) : null;
}
$(".speed").addEventListener("click", e => {
  const b = e.target.closest("[data-speed]");
  if (b) setSpeed(+b.dataset.speed);
});
document.addEventListener("visibilitychange", () => setSpeed(speed));
if (params.get("paused")) setSpeed(0); else setSpeed(1);
drawClock();

// ---------- map ----------

const LAT0 = 40.75, KX = Math.cos(LAT0 * Math.PI / 180), M_PER_DEG = 111320;
const toXY = ([lng, lat]) => [lng * KX, lat];
const toLL = ([x, y]) => [x / KX, y];

// Centripetal Catmull-Rom through the stations, so lines curve the way built track would.
function smooth(pts, n = 14) {
  const P = pts.map(toXY);
  const ext = (a, b) => [2 * a[0] - b[0], 2 * a[1] - b[1]];
  const Q = [ext(P[0], P[1]), ...P, ext(P[P.length - 1], P[P.length - 2])];
  const out = [];
  const tj = (ti, a, b) => ti + Math.pow(Math.hypot(b[0] - a[0], b[1] - a[1]), 0.5);
  for (let i = 1; i < Q.length - 2; i++) {
    const [p0, p1, p2, p3] = [Q[i - 1], Q[i], Q[i + 1], Q[i + 2]];
    const t0 = 0, t1 = tj(t0, p0, p1), t2 = tj(t1, p1, p2), t3 = tj(t2, p2, p3);
    const L = (a, b, ta, tb, t) => [0, 1].map(j => (tb - t) / (tb - ta) * a[j] + (t - ta) / (tb - ta) * b[j]);
    for (let s = 0; s < n; s++) {
      const t = t1 + (t2 - t1) * s / n;
      const A1 = L(p0, p1, t0, t1, t), A2 = L(p1, p2, t1, t2, t), A3 = L(p2, p3, t2, t3, t);
      const B1 = L(A1, A2, t0, t2, t), B2 = L(A2, A3, t1, t3, t);
      out.push(L(B1, B2, t1, t2, t));
    }
  }
  out.push(P[P.length - 1]);
  return out;
}

// A stretch of the path from distance a to b (metres), for a train capsule.
function stretch(xy, cum, a, b) {
  const at = d => {
    let i = 1;
    while (i < cum.length - 1 && cum[i] < d) i++;
    const f = (d - cum[i - 1]) / (cum[i] - cum[i - 1] || 1);
    return [xy[i - 1][0] + (xy[i][0] - xy[i - 1][0]) * f, xy[i - 1][1] + (xy[i][1] - xy[i - 1][1]) * f];
  };
  const pts = [at(a)];
  for (let i = 0; i < cum.length; i++) if (cum[i] > a && cum[i] < b) pts.push(xy[i]);
  pts.push(at(b));
  return pts.map(toLL);
}

const PATHS = {};
for (const l of LINES) {
  const xy = smooth(l.stops.map(s => STATIONS[s].slice(1)));
  const cum = [0];
  for (let i = 1; i < xy.length; i++)
    cum.push(cum[i - 1] + Math.hypot(xy[i][0] - xy[i - 1][0], xy[i][1] - xy[i - 1][1]) * M_PER_DEG);
  PATHS[l.id] = { xy, cum, len: cum[cum.length - 1] };
}

// Which side of the line the selected line's station names go on.
const LABEL_SIDE = { B: ["right", [-9, 0]], F: ["bottom", [0, -8]], C: ["left", [9, 0]],
                     A: ["top", [0, 8]], H: ["bottom", [0, -8]] };

function networkData() {
  const lines = [], trains = [], stations = [];
  for (const l of LINES) {
    const c = lineHex(l.colour), p = PATHS[l.id], on = l === sel;
    lines.push({ type: "Feature", properties: { id: l.id, c, sel: on },
                 geometry: { type: "LineString", coordinates: p.xy.map(toLL) } });
    // Trains at the peak count, spread around the out-and-back loop. 300 m capsules.
    const n = needed(l, l.tph[0]), CAP = 300;
    for (let i = 0; i < n; i++) {
      let d = ((i + 0.37 * (i % 2)) / n) * 2 * p.len;
      if (d > p.len) d = 2 * p.len - d;
      const a = Math.max(0, Math.min(p.len - CAP, d - CAP / 2));
      trains.push({ type: "Feature", properties: { c: darker(c), sel: on },
                    geometry: { type: "LineString", coordinates: stretch(p.xy, p.cum, a, a + CAP) } });
    }
  }
  for (const [id, [name, lng, lat]] of Object.entries(STATIONS))
    stations.push({ type: "Feature", properties: { id, xfer: linesAt(id).length > 1 },
                    geometry: { type: "Point", coordinates: [lng, lat] } });
  return { lines, trains, stations };
}

const bounds = Object.values(STATIONS).reduce((b, [, lng, lat]) =>
  [[Math.min(b[0][0], lng), Math.min(b[0][1], lat)], [Math.max(b[1][0], lng), Math.max(b[1][1], lat)]],
  [[180, 90], [-180, -90]]);

const map = new maplibregl.Map({
  container: "map",
  style: "https://tiles.openfreemap.org/styles/positron",
  bounds, fitBoundsOptions: { padding: { top: 28, bottom: 60, left: 28, right: 28 } },
  attributionControl: { compact: true },
  dragRotate: false, pitchWithRotate: false,
});
map.touchZoomRotate.disableRotation();
window.mockMap = map;

const ready = new Promise(r => map.on("load", r));

let styleReady = false;
function paintMap() {
  if (!styleReady) return;
  const land = css("--map-land"), water = css("--map-water"), park = css("--map-park");
  const label = css("--map-label"), text = css("--net-ink");
  const set = (id, prop, v) => { if (map.getLayer(id)) map.setPaintProperty(id, prop, v); };
  set("background", "background-color", land);
  set("park", "fill-color", park);
  set("landcover_wood", "fill-color", park);
  set("water", "fill-color", water);
  set("waterway", "line-color", water);
  set("landuse_residential", "fill-color", mix(land, text, 0.03));
  set("building", "fill-color", mix(land, text, 0.04));
  set("building", "fill-outline-color", mix(land, text, 0.08));
  set("road_area_pier", "fill-color", land);
  set("road_pier", "line-color", land);
  set("highway_minor", "line-color", mix(land, text, 0.07));
  set("highway_path", "line-color", mix(land, text, 0.05));
  for (const id of ["highway_major_casing", "highway_motorway_casing", "highway_motorway_bridge_casing",
                    "tunnel_motorway_casing"]) set(id, "line-color", mix(land, text, 0.13));
  for (const id of ["boundary_2", "boundary_3", "boundary_disputed"]) set(id, "line-color", mix(land, text, 0.25));
  for (const layer of map.getStyle().layers) {
    if (layer.type === "symbol") {
      try {
        map.setPaintProperty(layer.id, "text-color", label);
        map.setPaintProperty(layer.id, "text-halo-color", land);
        map.setPaintProperty(layer.id, "text-halo-width", 1.2);
      } catch { /* icon-only layer */ }
    }
  }
  
  drawNetwork();
}

map.on("style.load", () => {
  styleReady = true;
  // The player builds the rail: hide the basemap's own railways and road shields.
  for (const layer of map.getStyle().layers)
    if (/^railway|shield/.test(layer.id)) map.setLayoutProperty(layer.id, "visibility", "none");
  paintMap();
});

ready.then(() => {
  // Start with the attribution folded to its button.
  document.querySelector(".maplibregl-ctrl-attrib")?.classList.remove("maplibregl-compact-show");
  const empty = { type: "FeatureCollection", features: [] };
  for (const s of ["lines", "trains", "stations"]) map.addSource(s, { type: "geojson", data: empty });
  const ifSel = (a, b) => ["case", ["get", "sel"], a, b];
  map.addLayer({ id: "line-casing", type: "line", source: "lines",
    layout: { "line-join": "round", "line-cap": "round" },
    paint: { "line-color": "#ffffff",
             "line-width": ["interpolate", ["linear"], ["zoom"], 9, ifSel(6.5, 5.5), 12, ifSel(12, 10), 15, ifSel(18, 15)] } });
  map.addLayer({ id: "line", type: "line", source: "lines",
    layout: { "line-join": "round", "line-cap": "round" },
    paint: { "line-color": ["get", "c"],
             "line-width": ["interpolate", ["linear"], ["zoom"], 9, ifSel(4.5, 3.5), 12, ifSel(9, 7.5), 15, ifSel(14, 12)] } });
  map.addLayer({ id: "stations", type: "circle", source: "stations",
    paint: { "circle-color": "#ffffff", "circle-stroke-color": css("--net-ink"),
             "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 9, 1.2, 13, 2],
             "circle-radius": ["interpolate", ["linear"], ["zoom"],
                               9, ["case", ["get", "xfer"], 3.5, 2.6],
                               12, ["case", ["get", "xfer"], 6.2, 4.8],
                               15, ["case", ["get", "xfer"], 9, 7]] } });
  map.addLayer({ id: "train-edge", type: "line", source: "trains",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": "#ffffff",
             "line-width": ["interpolate", ["linear"], ["zoom"], 9, 5, 12, 8, 15, 12] } });
  map.addLayer({ id: "train", type: "line", source: "trains",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": ["get", "c"],
             "line-width": ["interpolate", ["linear"], ["zoom"], 9, 3, 12, 5.5, 15, 9] } });

  map.on("click", e => {
    const f = map.queryRenderedFeatures([[e.point.x - 6, e.point.y - 6], [e.point.x + 6, e.point.y + 6]],
                                        { layers: ["line", "train"] })[0];
    if (f && f.properties.id) select(LINES.find(l => l.id === f.properties.id));
  });
  map.on("mousemove", e => {
    const hit = map.queryRenderedFeatures([[e.point.x - 6, e.point.y - 6], [e.point.x + 6, e.point.y + 6]],
                                          { layers: ["line"] }).length;
    map.getCanvas().style.cursor = hit ? "pointer" : "";
  });
  paintMap();
});

let labels = [];
function drawNetwork() {
  if (!map.getSource || !map.getSource("lines")) return;
  const d = networkData();
  map.getSource("lines").setData({ type: "FeatureCollection", features: d.lines });
  map.getSource("trains").setData({ type: "FeatureCollection", features: d.trains });
  map.getSource("stations").setData({ type: "FeatureCollection", features: d.stations });
  labels.forEach(m => m.remove());
  const [anchor, offset] = LABEL_SIDE[sel.id];
  labels = sel.stops.map(s => {
    const el = document.createElement("div");
    el.className = "stn-label";
    el.textContent = STATIONS[s][0];
    return new maplibregl.Marker({ element: el, anchor, offset }).setLngLat(STATIONS[s].slice(1)).addTo(map);
  });
}
