// Canvas map: projection, draw loop, density thinning, pan/zoom, hit-testing.
// Emits hover/click via callbacks; knows nothing about cards or editing.

import { state, mercY } from './data.js';
import * as tiles from './tiles.js';

const CELL = 10;        // target cell size for the occupancy test, px
const BIG_R = 3.2;      // dots this size or larger are never suppressed
const MAX_DRAW = 26000; // ceiling so a pathological view still stays smooth

// How much dot radius grows with zoom. At scale 1 this is 1.0 either way, so
// the zoomed-out view is unchanged; a lower exponent makes zoomed-IN dots much
// smaller, which is what separates them once you are down at city level. 0.55
// made a zoom of 100x swell dots 12.6x; 0.30 makes it 4x.
const ZOOM_EXP = 0.30;

export const view = { scale: 1, x: 0.5, y: 0.5 };

let cv, ctx, W = 0, H = 0, DPR = 1;
const occ = new Set();
let vis, visX, visY, visR, visN = 0;
let onHover = () => {}, onClick = () => {}, onCounts = () => {};
let deleted = new Set();
// Which kind codes are drawn. Aggregates (metro/urban areas) and admin
// duplicates are OFF by default: three near-identical bubbles for "New York"
// is noise, not information.
let showKinds = new Set([0]);

export const sx = nx => (nx - view.x) * view.scale * Math.min(W, H * 2) + W / 2;
export const sy = ny => (ny - view.y) * view.scale * Math.min(W, H * 2) + H / 2;
export const unproject = (px, py) => {
  const k = view.scale * Math.min(W, H * 2);
  return { nx: view.x + (px - W / 2) / k, ny: view.y + (py - H / 2) / k };
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
  Object.assign({ onHover, onClick }, handlers);
  onHover = handlers.onHover || onHover;
  onClick = handlers.onClick || onClick;
  onCounts = handlers.onCounts || onCounts;
  vis = new Int32Array(MAX_DRAW);
  visX = new Float32Array(MAX_DRAW);
  visY = new Float32Array(MAX_DRAW);
  visR = new Float32Array(MAX_DRAW);
  addEventListener('resize', resize);
  // Tiles arrive one at a time and a screenful is ~30 of them. Redrawing per
  // arrival is 30 full redraws in one frame; coalescing makes it one.
  tiles.setOnLoad(scheduleDraw);
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
  DPR = window.devicePixelRatio || 1;
  W = innerWidth; H = innerHeight;
  cv.width = W * DPR; cv.height = H * DPR;
  cv.style.width = W + 'px'; cv.style.height = H + 'px';
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  draw();
}

// The fixed-scale ne_50m coastline. Still the ONLY ground when tiles are off,
// and the fallback under them: it is local, so it paints instantly and works
// with no network, which is what stops a slow or absent tile fetch from
// showing bare water.
function drawLand(css) {
  ctx.fillStyle = css.getPropertyValue('--land').trim() || '#f5f3ee';
  ctx.strokeStyle = css.getPropertyValue('--coast').trim() || '#c9d3dc';
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  for (const f of state.land.features) {
    const polys = f.geometry.type === 'Polygon'
      ? [f.geometry.coordinates] : f.geometry.coordinates;
    for (const poly of polys) for (const ring of poly) {
      for (let i = 0; i < ring.length; i++) {
        const X = sx((ring[i][0] + 180) / 360), Y = sy(mercY(ring[i][1]));
        i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
      }
      ctx.closePath();
    }
  }
  ctx.fill();
  ctx.stroke();
}

export function draw() {
  if (!state.n) return;
  const k = view.scale * Math.min(W, H * 2);
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue('--water').trim() || '#dbe7f0';
  ctx.fillRect(0, 0, W, H);

  // Once every visible tile is decoded the geojson is painted over completely,
  // so drawing it is ~50k invisible path ops per frame — and this runs on every
  // mousemove of a drag. Skipping it there is most of the cost of the layer.
  if (!tiles.covered(k, W, H, sx, sy)) drawLand(css);
  tiles.draw(ctx, k, W, H, sx, sy, DPR);

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
  const zk = Math.max(0.35, Math.pow(view.scale, ZOOM_EXP));
  const level = Math.max(0, Math.round(Math.log2(Math.max(k / CELL, 1))));
  const g = Math.pow(2, level);        // cells per world unit, per axis
  occ.clear();
  let nvis = 0;
  for (let i = 0; i < state.n; i++) {
    if (!showKinds.has(state.kind[i])) continue;
    if (deleted.has(state.keys[i])) continue;
    const r = state.r[i] * zk;
    if (r < 0.45) continue;
    const X = sx(state.nx[i]), Y = sy(state.ny[i]);
    if (X < -30 || X > W + 30 || Y < -30 || Y > H + 30) continue;
    if (r < BIG_R) {
      const cell = ((state.ny[i] * g) | 0) * g + ((state.nx[i] * g) | 0);
      if (occ.has(cell)) continue;
      occ.add(cell);
    }
    vis[nvis] = i; visX[nvis] = X; visY[nvis] = Y; visR[nvis] = r;
    if (++nvis >= MAX_DRAW) break;
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
  let drag = null, moved = false;
  cv.addEventListener('mousedown', e => {
    drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
    moved = false;
  });
  addEventListener('mouseup', e => {
    if (drag && !moved) onClick(pick(e.clientX, e.clientY), e);
    drag = null;
  });
  addEventListener('mousemove', e => {
    if (drag) {
      if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) moved = true;
      const k = view.scale * Math.min(W, H * 2);
      view.x = drag.vx - (e.clientX - drag.x) / k;
      view.y = drag.vy - (e.clientY - drag.y) / k;
      onHover(-1, e);
      draw();
      return;
    }
    onHover(pick(e.clientX, e.clientY), e);
  });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const k = view.scale * Math.min(W, H * 2);
    const mx = view.x + (e.clientX - W / 2) / k, my = view.y + (e.clientY - H / 2) / k;
    view.scale = Math.max(1, Math.min(4000, view.scale * Math.exp(-e.deltaY * 0.0016)));
    const k2 = view.scale * Math.min(W, H * 2);
    view.x = mx - (e.clientX - W / 2) / k2;
    view.y = my - (e.clientY - H / 2) / k2;
    draw();
    onHover(pick(e.clientX, e.clientY), e);
  }, { passive: false });
}

export function reset() { view.scale = 1; view.x = 0.5; view.y = 0.5; draw(); }

export function flyTo(lat, lon, scale = 220) {
  view.x = (lon + 180) / 360;
  view.y = mercY(lat);
  view.scale = scale;
  draw();
}
