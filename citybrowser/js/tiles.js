// Raster XYZ tile basemap.
//
// The map's own projection IS the tile grid. data.js stores
// nx = (lon+180)/360 and ny = mercY(lat), which is normalised Web Mercator, so
// tile (z,X,Y) covers exactly nx in [X/2^z,(X+1)/2^z] and ny in [Y/2^z,(Y+1)/2^z].
// There is nothing to reproject — that is the whole reason a tile layer is a
// hundred lines here instead of a rewrite.
//
// RASTER, not vector. Vector tiles mean MapLibre, and MapLibre wants to own the
// canvas — which would take the draw loop, the density thinning and the
// hit-testing with it. Raster is one drawImage per tile into the canvas map.js
// already owns, and every one of those stays untouched.
//
// The geojson land layer stays underneath as the offline/loading fallback, so
// the map never shows blank ground and still works with no network at all.

const TILE = 256;          // CSS px per tile. @2x images still occupy 256 CSS px.
const MAX_INFLIGHT = 16;   // a fast zoom would otherwise queue hundreds of GETs
const CACHE_MAX = 800;     // LRU. ~50 MB of decoded RGBA at @2x.
const PARENT_DEPTH = 5;    // how far up to look for a stand-in while loading
const RETRY_MS = 20000;    // a failed tile is worth one more try, not a hammer

// Attribution is a licence condition on all of these, not decoration.
export const SOURCES = {
  off: { label: 'Off (coastline only)', attrib: '' },
  light_nolabels: {
    label: 'Light — no labels',
    url: (z, x, y, r) => `https://${sub(x, y)}.basemaps.cartocdn.com/light_nolabels/${z}/${x}/${y}${r}.png`,
    maxZoom: 20,
    attrib: '&copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap</a> &middot; &copy; <a href="https://carto.com/attributions">CARTO</a>',
  },
  light_all: {
    label: 'Light — with labels',
    url: (z, x, y, r) => `https://${sub(x, y)}.basemaps.cartocdn.com/light_all/${z}/${x}/${y}${r}.png`,
    maxZoom: 20,
    attrib: '&copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap</a> &middot; &copy; <a href="https://carto.com/attributions">CARTO</a>',
  },
  voyager_nolabels: {
    label: 'Voyager — no labels',
    url: (z, x, y, r) => `https://${sub(x, y)}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/${z}/${x}/${y}${r}.png`,
    maxZoom: 20,
    attrib: '&copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap</a> &middot; &copy; <a href="https://carto.com/attributions">CARTO</a>',
  },
  esri_gray: {
    label: 'Esri light gray',
    url: (z, x, y) => `https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/${z}/${y}/${x}`,
    maxZoom: 16,
    retina: false,
    attrib: 'Tiles &copy; <a href="https://www.esri.com/">Esri</a>',
  },
};

const sub = (x, y) => 'abcd'[(x + y) & 3];

let sourceId = 'off';
let onLoad = () => {};
// Insertion-ordered, so the first key is the least recently used.
const cache = new Map();
let inflight = 0;

export function setSource(id) {
  if (!SOURCES[id]) id = 'off';
  if (id === sourceId) return;
  sourceId = id;
  cache.clear();          // keys are z/x/y, so a stale source would collide
  inflight = 0;
  onLoad();
}

export function source() { return sourceId; }
export function enabled() { return sourceId !== 'off'; }
export function attribution() { return SOURCES[sourceId].attrib || ''; }
export function setOnLoad(fn) { onLoad = fn; }

function get(key, url) {
  let t = cache.get(key);
  if (t) {
    if (t.state === 'error' && performance.now() > t.retryAt) cache.delete(key);
    else { cache.delete(key); cache.set(key, t); return t; }   // LRU touch
  }
  if (inflight >= MAX_INFLIGHT) return null;   // the next draw will ask again
  const img = new Image();
  t = { img, state: 'loading', retryAt: 0 };
  cache.set(key, t);
  inflight++;
  img.onload = () => { t.state = 'ok'; inflight--; onLoad(); };
  img.onerror = () => {
    t.state = 'error'; t.retryAt = performance.now() + RETRY_MS;
    inflight--;
  };
  img.src = url;
  while (cache.size > CACHE_MAX) {
    const oldest = cache.keys().next().value;
    if (oldest === key) break;
    cache.delete(oldest);
  }
  return t;
}

// Zoom level whose tiles are closest to their native size on screen.
// k is world-width in CSS px, so 2^z tiles of TILE px span it when 2^z*TILE = k.
function levelFor(k) {
  const max = SOURCES[sourceId].maxZoom ?? 19;
  return Math.max(0, Math.min(max, Math.round(Math.log2(k / TILE))));
}

// sx(nx) = (nx - view.x)*k + W/2, so the inverse is straight algebra. Derived
// from the projection functions themselves rather than from a second copy of
// the view state, so the two can never drift apart.
const inv = (px, k, proj) => (px - proj(0)) / k;

// Visible tile range at level z. Deliberately does NOT wrap in longitude: the
// city dots do not repeat, so a repeating basemap would show land with no
// cities on it and read as a map bug.
function range(z, k, W, H, sx, sy) {
  const n = 1 << z;
  return {
    n,
    x0: Math.max(0, Math.floor(inv(0, k, sx) * n)),
    x1: Math.min(n - 1, Math.floor(inv(W, k, sx) * n)),
    y0: Math.max(0, Math.floor(inv(0, k, sy) * n)),
    y1: Math.min(n - 1, Math.floor(inv(H, k, sy) * n)),
  };
}

/**
 * Is every visible tile already decoded at the target level?
 * map.js uses this to skip the geojson land pass — with full tile coverage the
 * geojson is invisible anyway, and it is ~50k path ops per frame during a pan.
 */
export function covered(k, W, H, sx, sy) {
  if (!enabled()) return false;
  const z = levelFor(k);
  const { x0, x1, y0, y1 } = range(z, k, W, H, sx, sy);
  for (let x = x0; x <= x1; x++) {
    for (let y = y0; y <= y1; y++) {
      const t = cache.get(`${z}/${x}/${y}`);
      if (!t || t.state !== 'ok') return false;
    }
  }
  return true;
}

export function draw(ctx, k, W, H, sx, sy, dpr) {
  if (!enabled()) return;
  const src = SOURCES[sourceId];
  const r = (src.retina !== false && dpr > 1) ? '@2x' : '';
  const z = levelFor(k);
  const { n, x0, x1, y0, y1 } = range(z, k, W, H, sx, sy);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  for (let x = x0; x <= x1; x++) {
    for (let y = y0; y <= y1; y++) {
      // Rounding the SHARED edge of adjacent tiles, rather than rounding a
      // position and a width independently, is what keeps hairline seams out
      // of the fill. Both neighbours agree on the boundary by construction.
      const dx = Math.round(sx(x / n)), dw = Math.round(sx((x + 1) / n)) - dx;
      const dy = Math.round(sy(y / n)), dh = Math.round(sy((y + 1) / n)) - dy;
      if (dw <= 0 || dh <= 0) continue;

      const t = get(`${z}/${x}/${y}`, src.url(z, x, y, r));
      if (t && t.state === 'ok') {
        ctx.drawImage(t.img, dx, dy, dw, dh);
        continue;
      }
      // Not here yet: blow up whatever ancestor we already hold. This is what
      // makes a zoom read as "sharpening" instead of "blank, then a flash".
      for (let d = 1; d <= PARENT_DEPTH && (z - d) >= 0; d++) {
        const pz = z - d, pxi = x >> d, pyi = y >> d;
        const p = cache.get(`${pz}/${pxi}/${pyi}`);
        if (!p || p.state !== 'ok') continue;
        const f = 1 << d;
        const sw = p.img.width / f, sh = p.img.height / f;
        ctx.drawImage(p.img, (x - (pxi << d)) * sw, (y - (pyi << d)) * sh, sw, sh,
                      dx, dy, dw, dh);
        break;
      }
    }
  }

  // Prefetch the parent level so the NEXT zoom-out already has a stand-in.
  // Cheap (a quarter as many tiles) and it removes the blank flash entirely.
  if (z > 0 && inflight < MAX_INFLIGHT / 2) {
    const pz = z - 1, pn = 1 << pz;
    for (let x = Math.max(0, x0 >> 1); x <= Math.min(pn - 1, x1 >> 1); x++) {
      for (let y = Math.max(0, y0 >> 1); y <= Math.min(pn - 1, y1 >> 1); y++) {
        get(`${pz}/${x}/${y}`, src.url(pz, x, y, r));
      }
    }
  }
}
