// Wiring only. Each module owns its own concern; this file connects them.

import { load, state } from './data.js';
import * as map from './map.js';
import * as card from './card.js';
import * as edit from './edit.js';
import { popRampCSS } from './colors.js';
import { KINDS } from './data.js';
import * as search from './search.js';
import * as basemap from './basemap.js';
import * as review from './review.js';
import { popColor, fade, rgb } from './colors.js';

const $ = id => document.getElementById(id);

const cardEl = $('card'), editEl = $('edit');
let hoverIdx = -1;

$('ramp').style.background = popRampCSS();

edit.init(editEl, {
  onChange: () => {
    const t = state.rec.filter(c => c._touched).length;
    $('touched').textContent = t.toLocaleString();
  },
});

function onHover(i, e) {
  if (i === hoverIdx && i < 0) return;
  hoverIdx = i;
  if (i < 0) { cardEl.style.display = 'none'; return; }
  cardEl.innerHTML = card.render(state.rec[i], state.keys[i]);
  cardEl.style.display = 'block';
  card.place(cardEl, e.clientX, e.clientY);
}

async function onClick(i, e) {
  await edit.handleMapClick(i, e);
}

// --- search ---------------------------------------------------------------
const qEl = $('q'), resEl = $('results');
let hits = [], sel = -1;

function renderResults() {
  if (!hits.length) { resEl.style.display = 'none'; return; }
  resEl.innerHTML = hits.map((i, n) => {
    const c = state.rec[i];
    const col = rgb(c._touched ? popColor(c.pop) : fade(popColor(c.pop)));
    const sub = [c.adminName, c.kind !== 'city' ? c.kind : ''].filter(Boolean).join(' · ');
    return `<div class="res${n === sel ? ' sel' : ''}" data-n="${n}">
      <span class="rdot" style="background:${col}"></span>
      <b>${(c.name || '').replace(/[&<>]/g, '')}</b>
      <span class="rsub">${(sub || '').replace(/[&<>]/g, '')}</span>
      <span class="rpop">${c.pop ? Number(c.pop).toLocaleString() : ''}</span>
    </div>`;
  }).join('');
  resEl.style.display = 'block';
  resEl.querySelectorAll('.res').forEach(el => {
    el.onclick = () => go(hits[+el.dataset.n]);
  });
}

function go(i) {
  const c = state.rec[i];
  map.flyTo(c.lat, c.lon, 600);
  resEl.style.display = 'none';
  qEl.blur();
  edit.open(i);
}

let searchTimer = null;
qEl.addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    hits = search.search(qEl.value, 12, map.kindsShown());
    sel = hits.length ? 0 : -1;
    renderResults();
  }, 90);
});
qEl.addEventListener('keydown', e => {
  if (e.key === 'ArrowDown') { sel = Math.min(sel + 1, hits.length - 1); renderResults(); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { sel = Math.max(sel - 1, 0); renderResults(); e.preventDefault(); }
  else if (e.key === 'Enter' && sel >= 0) { go(hits[sel]); }
  else if (e.key === 'Escape') { qEl.value = ''; resEl.style.display = 'none'; qEl.blur(); }
});
// "/" focuses search, the way every tool that expects typing does.
addEventListener('keydown', e => {
  if (e.key === '/' && document.activeElement !== qEl) { e.preventDefault(); qEl.focus(); }
});

// View state in the URL hash: #lat,lon,scale. Deep-links a city, survives a
// reload during curation, and makes "look at this one" shareable.
function applyHash() {
  const m = /^#(-?[\d.]+),(-?[\d.]+),([\d.]+)$/.exec(location.hash);
  if (m) map.flyTo(parseFloat(m[1]), parseFloat(m[2]), parseFloat(m[3]));
}
let hashTimer = null;
function saveHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const { lat, lon } = map.toLatLon(map.view.x, map.view.y);
    history.replaceState(null, '',
      `#${lat.toFixed(4)},${lon.toFixed(4)},${map.view.scale.toFixed(1)}`);
  }, 250);
}

load().then(() => {
  $('count').textContent = state.n.toLocaleString() + ' cities';
  const t = state.rec.filter(c => c._touched).length;
  $('touched').textContent = t.toLocaleString();
  // Per-kind counts in the settings panel, so the toggles say how much they hide
  const tally = {};
  for (const c of state.rec) tally[c.kind || 'city'] = (tally[c.kind || 'city'] || 0) + 1;
  KINDS.forEach(k => {
    const el = $('n-' + k);
    if (el) el.textContent = (tally[k] || 0).toLocaleString();
    // A toggle for a kind nothing is classified as reads as a broken control,
    // so hide the row rather than show it at zero. `admin` is the live case:
    // US counties moved to `rural`, which is decided first, and nothing is
    // left that claims only to be a county seat.
    const row = $('row-' + k);
    if (row && k !== 'city') row.hidden = !tally[k];
  });
  map.init($('c'), {
    onHover, onClick,
    onCounts: (shown) => {
      $('shown').textContent = shown.toLocaleString() + ' shown';
      saveHash();
    },
  });
  review.init($('reviewpanel'), { onDone: reviewCount });
  reviewCount();
  applyKinds(chosen);
  // ?q=... runs a search on load. Handy for linking someone straight to a city,
  // and it makes the search path testable without driving a real keyboard.
  // ?city=Q60 opens a city's panel directly — deep-linkable, and it means the
  // panel can be verified without driving a real mouse.
  const c0 = new URLSearchParams(location.search).get('city');
  if (c0 && state.index.has(c0)) {
    const i = state.index.get(c0);
    map.flyTo(state.rec[i].lat, state.rec[i].lon, 600);
    edit.open(i);
  }
  // ?card=Q60 pins the hover card open. Same reason as ?q= and ?city=: the card
  // is otherwise reachable only by a real mouse, so nothing about it — layout,
  // overflow, a missing field — can be checked without one.
  const k0 = new URLSearchParams(location.search).get('card');
  if (k0 && state.index.has(k0)) {
    const i = state.index.get(k0);
    map.flyTo(state.rec[i].lat, state.rec[i].lon, 400);
    cardEl.innerHTML = card.render(state.rec[i], k0);
    cardEl.style.display = 'block';
    card.place(cardEl, innerWidth / 2, 90);
  }
  // ?review=1 opens the queue. Same rationale as ?card= and ?q=: the panel is
  // otherwise only reachable through the settings menu and a keypress.
  if (new URLSearchParams(location.search).get('review')) review.open();
  const q0 = new URLSearchParams(location.search).get('q');
  if (q0) {
    qEl.value = q0;
    hits = search.search(q0, 12, map.kindsShown());
    sel = hits.length ? 0 : -1;
    renderResults();
  }
  applyHash();
  addEventListener('hashchange', applyHash);
}).catch(err => {
  $('status').textContent = 'failed to load: ' + err.message;
});

$('reset').onclick = () => map.reset();

// --- settings: which kinds of point are drawn.
// Persisted, because "hide metro areas" is a standing preference, not a
// per-session one — having to re-tick it every reload would be its own bug.
const SET_KEY = 'citybrowser.kinds';
function loadKinds() {
  try {
    const v = JSON.parse(localStorage.getItem(SET_KEY));
    if (Array.isArray(v) && v.length) return new Set(v);
  } catch (e) { /* corrupt or absent -> default */ }
  return new Set([0]);            // cities only
}
function applyKinds(set) {
  map.setKinds([...set]);
  localStorage.setItem(SET_KEY, JSON.stringify([...set]));
  KINDS.forEach((k, i) => {
    const el = $('k-' + k);
    if (el) el.checked = set.has(i);
  });
}
const chosen = loadKinds();

$('settings').onclick = () => {
  const open = $('setpanel').style.display !== 'block';
  $('setpanel').style.display = open ? 'block' : 'none';
  $('settings').classList.toggle('active', open);
};
KINDS.forEach((k, i) => {
  const el = $('k-' + k);
  if (!el) return;
  el.onchange = () => {
    el.checked ? chosen.add(i) : chosen.delete(i);
    if (!chosen.size) { chosen.add(0); }   // never hide everything
    applyKinds(chosen);
  };
});

// --- settings: which basemap style.
// The choice, its persistence and the ?basemap= override all live in
// basemap.js, which owns the map — this is only the control that drives it.
const bmEl = $('basemap');
bmEl.innerHTML = Object.entries(basemap.STYLES)
  .map(([id, s]) => `<option value="${id}">${s.label}</option>`).join('');
bmEl.value = basemap.styleId();
bmEl.onchange = () => {
  basemap.setStyle(bmEl.value);
  bmEl.value = basemap.styleId();   // snaps back if the id was not a real one
};

// --- the GHS match review queue, reached from settings.
// Kept out of the main bar deliberately: it is a batch job you sit down to do,
// not something to trip over while curating a city.
function reviewCount() {
  const n = review.pending().length;
  const el = $('nreview');
  if (el) el.textContent = n ? `(${n.toLocaleString()})` : '(none left)';
  return n;
}
$('review').onclick = () => {
  if (review.isOpen()) { review.close(); return; }
  $('setpanel').style.display = 'none';
  $('settings').classList.remove('active');
  review.open();
};

$('create').onclick = () => {
  const t = edit.setTool('create');
  $('create').classList.toggle('active', t === 'create');
};

// Export is NOT part of the edit loop — the map reads base + overrides directly,
// so edits are live on reload without it. This only writes a standalone
// cities.json for publishing the map read-only.
$('rebuild').onclick = async () => {
  $('rebuild').textContent = '...';
  try {
    const r = await edit.rebuild();
    $('status').textContent =
      `exported cities.json — ${r.cities.toLocaleString()} cities` +
      (r.stale ? `, ${r.stale} stale` : '');
  } catch (err) {
    $('status').textContent = 'export failed: ' + err.message;
  }
  $('rebuild').textContent = 'Export';
};
