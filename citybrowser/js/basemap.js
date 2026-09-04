// The ground under the bubbles: a real vector basemap, via MapLibre.
//
// This replaces a hand-rolled raster XYZ layer (js/tiles.js, deleted) that drew
// PNG tiles into the same canvas as the bubbles, with a fixed-scale Natural
// Earth coastline underneath as the offline fallback. That worked, but it owned
// a tile cache, an LRU, an inflight cap, a parent-tile stand-in and a seam-free
// rounding rule -- all of it reimplementing a tile client, and none of it able
// to fix the thing TODO.md actually complained about, which is that the
// coastline is coarse when you zoom in. A vector style is sharp at every zoom
// and drops both the tile client and ne_50m_land.geojson (1.6 MB off the load).
//
// Same source and the same version pin as cityhistory, so the two maps behave
// identically and a MapLibre upgrade is one decision, not two.
//
// MapLibre owns pan/zoom and the pointer now. The bubbles stay on their own
// canvas on top (see map.js), which is what keeps the density thinning, the
// curation colours and the hit-testing ours.

const KEY = 'citybrowser.basemap';
const DEFAULT = 'positron';

// Light styles only, deliberately: the whole palette in css/app.css is a light
// one, and a dark basemap under dark-on-light bubbles reads as a bug.
// Attribution rides along inside each style's own sources, which is why there
// is no attrib string here any more -- AttributionControl reads it from the
// style, so it can never drift out of date with what is actually drawn.
const OFM = 'https://tiles.openfreemap.org/styles/';

// `labels: false` strips the style's symbol layers after it loads. OpenFreeMap
// ships no labelless variant, and this map draws its own labels-that-matter as
// bubbles -- a basemap that also names every country and sea competes with
// them. This is the default, and it is what the old raster default
// ("light_nolabels") was too.
export const STYLES = {
  positron: { label: 'Light — no labels', url: OFM + 'positron', labels: false },
  positron_labels: { label: 'Light — with labels', url: OFM + 'positron', labels: true },
  bright: { label: 'Bright — with labels', url: OFM + 'bright', labels: true },
  liberty: { label: 'Liberty — detailed', url: OFM + 'liberty', labels: true },
  off: { label: 'Off (no basemap)', url: null },
};

// A style with no layers at all: the GL canvas paints nothing and the page's
// --water background shows through. This is the honest "no ground" state, and
// it is also what you get to look at if the network is gone -- the bubbles are
// local and still draw.
const EMPTY = { version: 8, sources: {}, layers: [] };

const styleOf = id => STYLES[id]?.url ?? EMPTY;

// ?basemap=bright beats the stored choice, matching ?city= and ?q=. Same reason
// as those: it makes the layer checkable without driving the UI.
function initialId() {
  const q = new URLSearchParams(location.search).get('basemap');
  const id = q || localStorage.getItem(KEY) || DEFAULT;
  return STYLES[id] ? id : DEFAULT;
}

let current = initialId();
export const styleId = () => current;

// Zooming out stops where one world copy fills the viewport width. Any further
// is repeated worlds and a lot of empty ocean -- and it is also exactly where
// map.js's `view.scale` is 1, which is the scale the bubble radii were tuned
// against.
const box = document.getElementById('map');
export const vpW = () => box.clientWidth || innerWidth;
export const vpH = () => box.clientHeight || innerHeight;
export const worldFitZoom = () => Math.log2(vpW() / 512);

export const map = new maplibregl.Map({
  container: 'map',
  style: styleOf(current),
  center: [0, 20], zoom: worldFitZoom(), minZoom: worldFitZoom(), maxZoom: 16,
  renderWorldCopies: true,
  attributionControl: false,
  // map.js projects the bubbles with a flat affine -- north-up, no perspective
  // -- so it cannot follow a rotated or tilted map, and right-drag would spin
  // the basemap out from under bubbles that stayed put. Turn the gestures off
  // rather than reimplement the projection: matching MapLibre's perspective
  // per frame for 60k bubbles is exactly the per-point map.project() cost the
  // affine exists to avoid.
  dragRotate: false, pitchWithRotate: false, touchPitch: false, maxPitch: 0,
});
map.touchZoomRotate.disableRotation();
// No NavigationControl on purpose. The map had no zoom buttons before this
// change -- wheel, pinch and double-click were the whole story -- and the
// bottom-right corner is already spoken for by #status. Adding chrome would be
// a visible regression dressed up as a feature.
//
// Attribution is a licence condition, not decoration. Compact so it does not
// run along the bottom edge into #status.
map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right');
// compact:true still renders it OPEN on first load, collapsing only once the
// "i" is clicked, so collapse it up front.
map.once('load', () => document.querySelector('.maplibregl-ctrl-attrib')
                         ?.classList.remove('maplibregl-compact-show'));

// Every style gets repainted into the app's own palette, so the ground under
// the bubbles is the same light blue and warm off-white the panels were drawn
// against rather than each style's idea of grey. The colours are READ FROM THE
// CSS, so css/app.css stays the one place they are defined.
//
// Positron's `water` fill and its `background` are the two layers that matter;
// everything else in these styles sits on top of the background and inherits
// the change. Runs on 'style.load', which fires for the first style and for
// every setStyle() after it.
function tune() {
  if (!STYLES[current]?.url) return;
  const css = getComputedStyle(document.documentElement);
  const water = css.getPropertyValue('--water').trim() || '#d8e6f2';
  const land = css.getPropertyValue('--land').trim() || '#f6f4ef';
  if (map.getLayer('water')) map.setPaintProperty('water', 'fill-color', water);
  if (map.getLayer('background')) {
    map.setPaintProperty('background', 'background-color', land);
  }
  if (STYLES[current].labels === false) {
    for (const l of map.getStyle().layers) {
      if (l.type === 'symbol') map.removeLayer(l.id);
    }
  }
}
map.on('style.load', tune);

// Everything here is an ES module, so nothing is reachable from the console or
// from a headless probe without this. Both are how this page gets checked.
window.__gl = map;

export function setStyle(id) {
  if (!STYLES[id]) id = DEFAULT;
  current = id;
  localStorage.setItem(KEY, id);
  // The bubbles live on their own canvas, so a style swap has nothing to
  // re-add afterwards -- which is the other half of why they are not a
  // MapLibre layer.
  map.setStyle(styleOf(id));
}
