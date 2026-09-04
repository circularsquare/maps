// The bubble layer: projection, draw loop, density thinning, hit-testing.
// Emits hover/click via callbacks; knows nothing about cards or editing.
//
// MapLibre owns pan, zoom and the pointer (see basemap.js). This file owns a
// transparent canvas glued on top of it. The bubbles are NOT a MapLibre layer,
// and that is the point: density thinning, the curation-aware colours and the
// "largest dot under the cursor wins" hit test are all ours, and a style swap
// has nothing to re-add afterwards.

import { state, mercY } from './data.js';
import { map as gl, vpW, vpH, worldFitZoom } from './basemap.js';

const CELL = 10;        // target cell size for the occupancy test, px
const BIG_R = 3.2;      // dots this size or larger are never suppressed
const MAX_DRAW = 26000; // ceiling so a pathological view still stays smooth

// How much dot radius grows with zoom. At scale 1 this is 1.0 either way, so
// the zoomed-out view is unchanged; a lower exponent makes zoomed-IN dots much
// smaller, which is what separates them once you are down at city level. 0.55
// made a zoom of 100x swell dots 12.6x; 0.30 makes it 4x.
const ZOOM_EXP = 0.30;

const mercX = lon => (lon + 180) / 360;
const worldPx = () => 512 * Math.pow(2, gl.getZoom());

// The same three numbers the old hand-rolled pan/zoom kept in a mutable object,
// now read straight off MapLibre's transform so the two can never drift. Kept
// under the old names and the old meaning -- scale 1 is one world copy across
// the viewport -- because the URL hash, flyTo() and the radius curve are all
// written in those terms.
export const view = {
  get x() { return mercX(gl.getCenter().lng); },
  get y() { return mercY(gl.getCenter().lat); },
  get scale() { return worldPx() / Math.min(vpW(), vpH() * 2); },
};

let cv, ctx, W = 0, H = 0, DPR = 1;
const occ = new Set();
let vis, visX, visY, visR, visN = 0;
let onHover = () => {}, onClick = () => {}, onCounts = () => {};
let deleted = new Set();
// Which kind codes are drawn. Everything except plain cities is OFF by default:
// three near-identical bubbles for "New York" is noise, and a county, a ward or
// an upazila is not a place anyone is from. See kinds.py.
let showKinds = new Set([0]);

export const sx = nx => (nx - view.x) * worldPx() + vpW() / 2;
export const sy = ny => (ny - view.y) * worldPx() + vpH() / 2;
export const unproject = (px, py) => {
  const k = worldPx();
  return { nx: view.x + (px - vpW() / 2) / k, ny: view.y + (py - vpH() / 2) / k };
};
export const toLatLon = (nx, ny) => {
  const lon = nx * 360 - 180;
  const t = Math.PI * (1 - 2 * ny);
  const lat = (Math.atan(Math.sinh(t))) * 180 / Math.PI;
  return { lat, lon };
};

export function init(canvas, handlers) {
  cv = canvas;
  ctx = cv.getContext('2d');
  onHover = handlers.onHover || onHover;
  onClick = handlers.onClick || onClick;
  onCounts = handlers.onCounts || onCounts;
  vis = new Int32Array(MAX_DRAW);
  visX = new Float32Array(MAX_DRAW);
  visY = new Float32Array(MAX_DRAW);
  visR = new Float32Array(MAX_DRAW);
  addEventListener('resize', resize);
  wire();
  resize();
}

let drawPending = false;
export function scheduleDraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => { drawPending = false; draw(); });
}

export function markDeleted(key) { deleted.add(key); draw(); }

export function setKinds(codes) { showKinds = new Set(codes); draw(); }
export function kindsShown() { return showKinds; }

export function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = vpW(); H = vpH();
  cv.width = W * DPR; cv.height = H * DPR;
  cv.style.width = W + 'px'; cv.style.height = H + 'px';
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  gl.setMinZoom(worldFitZoom());   // keep the cap tied to the current width
  draw();
}

export function draw() {
  if (!state.n) return;
  W = vpW(); H = vpH();
  // Transparent, not filled: the basemap is underneath now, and a water-coloured
  // rect over it would be an opaque sheet hiding the thing we just added.
  ctx.clearRect(0, 0, W, H);

  const c = gl.getCenter();
  const ws = worldPx(), cxn = mercX(c.lng), cyn = mercY(c.lat);
  const k = ws;                        // world width in px
  const scale = ws / Math.min(W, H * 2);

  // --- pass 1: density thinning.
  //
  // Cities are sorted population-descending, so claiming a cell means the
  // largest city in that cell wins and the rest are skipped.
  //
  // The grid is anchored in WORLD space, not screen space. A screen-space grid
  // slides under the data as you pan, so cell membership keeps changing and the
  // selection shimmers. Quantising the cell size to a power of two makes the
  // grid identical for any pan at a given zoom — it only changes in discrete
  // steps as you zoom, which reads as detail appearing rather than as noise.
  const zk = Math.max(0.35, Math.pow(scale, ZOOM_EXP));
  const level = Math.max(0, Math.round(Math.log2(Math.max(k / CELL, 1))));
  const g = Math.pow(2, level);        // cells per world unit, per axis
  occ.clear();
  let nvis = 0;
  for (let i = 0; i < state.n; i++) {
    if (!showKinds.has(state.kind[i])) continue;
    if (deleted.has(state.keys[i])) continue;
    const r = state.r[i] * zk;
    if (r < 0.45) continue;
    const Y = (state.ny[i] - cyn) * ws + H / 2;
    if (Y < -30 || Y > H + 30) continue;
    // MapLibre repeats the world east and west of the prime copy, so a city can
    // be on screen in more than one of them near the antimeridian. Solve for
    // which copies land on screen rather than looping a fixed range: at the
    // minimum zoom one world already fills the viewport, so this is almost
    // always exactly one iteration.
    const X0 = (state.nx[i] - cxn) * ws + W / 2;
    const c0 = Math.ceil((-30 - X0) / ws), c1 = Math.floor((W + 30 - X0) / ws);
    for (let cp = c0; cp <= c1; cp++) {
      if (r < BIG_R) {
        // Copy index is part of the key, or the west copy of a city would be
        // suppressed by its own east copy having claimed the world cell.
        const cell = (((state.ny[i] * g) | 0) * g + ((state.nx[i] * g) | 0)) * 4
                   + (cp & 3);
        if (occ.has(cell)) continue;
        occ.add(cell);
      }
      vis[nvis] = i; visX[nvis] = X0 + cp * ws; visY[nvis] = Y; visR[nvis] = r;
      if (++nvis >= MAX_DRAW) break;
    }
    if (nvis >= MAX_DRAW) break;
  }
  visN = nvis;

  // --- pass 2: draw. Untouched first so curated ones sit on top.
  ctx.globalAlpha = 0.85;
  for (const wantTouched of [0, 1]) {
    for (let j = 0; j < visN; j++) {
      const i = vis[j];
      if (state.touched[i] !== wantTouched) continue;
      ctx.beginPath();
      ctx.arc(visX[j], visY[j], visR[j], 0, 6.2832);
      ctx.fillStyle = state.col[i];
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
  onCounts(visN, state.n);
}

function pick(px, py) {
  let best = -1, bestD = 1e9;
  for (let j = 0; j < visN; j++) {
    const d = Math.hypot(visX[j] - px, visY[j] - py);
    if (d < Math.max(5, visR[j]) && d < bestD) { bestD = d; best = j; }
  }
  return best < 0 ? -1 : vis[best];
}

function wire() {
  // Glued to MapLibre by redrawing on its 'render' event, which is the only way
  // to stay in step with the transform mid-animation. A tiled basemap also
  // fires 'render' for every tile fade-in, and draw() walks 61k cities, so
  // those extra frames would be pure waste -- the transform is not moving
  // during a fade. Skip when the view is identical to the last one drawn.
  let lastView = '';
  const redraw = () => {
    const t = gl.transform;
    const sig = `${t.center.lng},${t.center.lat},${t.zoom},${t.width},${t.height}`;
    if (sig === lastView) return;
    lastView = sig;
    draw();
  };
  gl.on('move', redraw);
  gl.on('render', redraw);

  // MapLibre's own click, so taps work on touch and a pan never counts as a
  // click. edit.js reads clientX/clientY off the event for the create and move
  // tools, so it gets the underlying DOM event rather than MapLibre's wrapper.
  gl.on('click', e => onClick(pick(e.point.x, e.point.y), e.originalEvent));

  const el = gl.getCanvasContainer();
  el.addEventListener('mousemove', e => {
    if (gl.isMoving()) { onHover(-1, e); return; }
    const rect = cv.getBoundingClientRect();
    const i = pick(e.clientX - rect.left, e.clientY - rect.top);
    // The cursor used to be pure CSS on the canvas. It cannot be any more:
    // MapLibre writes an inline cursor on its own canvas while dragging, and an
    // inline style beats a stylesheet. `data-tool` is set by edit.js and stays
    // the contract for what the pointer means.
    const tool = document.body.dataset.tool;
    gl.getCanvas().style.cursor = tool ? 'copy' : (i >= 0 ? 'pointer' : '');
    onHover(i, e);
  });
  el.addEventListener('mouseleave', e => onHover(-1, e));
}

export function reset() {
  gl.jumpTo({ center: [0, 20], zoom: worldFitZoom() });
}

// Instant, not animated -- this is how you land on a search result, and an
// arc-and-swoop across the world is a wait, not a feature.
export function flyTo(lat, lon, scale = 220) {
  const ws = scale * Math.min(vpW(), vpH() * 2);
  gl.jumpTo({ center: [lon, lat], zoom: Math.log2(ws / 512) });
}
