// The detail panel: what a city is, where to read about it, and — in edit mode —
// the fields and tools for curating it.
//
// This module and serve.py's write API are two halves of one contract; change
// one and you change the other. See SCHEMA.md.
//
// Everything writes a FIELD-LEVEL patch. We never send a whole record, because
// overrides.json has to be able to tell "Anita corrected this" from "this
// happened to be the value at the time".
//
// The panel opens on click in BOTH modes, not just edit mode. The hover card
// can never hold a link — it follows the cursor and vanishes — so the Wikipedia
// link, which is where the hand-written facts actually come from, needs
// somewhere stable to live.

import { state, refresh, add, KIND_LABEL } from './data.js';
import * as map from './map.js';

let panel, current = null;
// Edit mode is ALWAYS ON. This is a curation tool that happens to render a map,
// not a map with an editing feature — having to arm it every session was pure
// friction. setMode is kept so the read-only path still exists for a published
// build, but nothing in the UI turns it off.
let mode = true;
export let tool = null;          // null | 'move' | 'create'
let onChange = () => {};

const LIST_FIELDS = ['altNames', 'facts'];
const TEXT_FIELDS = [
  ['name', 'Name'],
  ['altNames', 'Alt names (one per line, max 3)'],
  ['facts', 'Facts (one per line, max 3)'],
  ['languages', 'Languages (one per line)'],
];

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export function init(el, handlers = {}) {
  panel = el;
  onChange = handlers.onChange || onChange;
}

export function enabled() { return mode; }

export function setMode(on) {
  mode = on;
  document.body.classList.toggle('editing', mode);
  if (!mode) tool = null;
  if (current) open(current.i);       // re-render with/without edit controls
  return mode;
}

export function setTool(t) {
  tool = (tool === t) ? null : t;
  document.body.dataset.tool = tool || '';
  return tool;
}

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${method} ${path} -> ${res.status}`);
  return res.json();
}

async function patch(key, field, value) {
  await api('PATCH', `/api/city/${encodeURIComponent(key)}`, { field, value });
  const i = state.index.get(key);
  if (i != null) {
    const rec = state.rec[i];
    const empty = value === null || value === '' ||
                  (Array.isArray(value) && !value.length);
    if (empty) {
      delete rec[field];
      rec._edited = (rec._edited || []).filter(f => f !== field);
      if (!rec._edited.length) delete rec._touched;
    } else {
      rec[field] = value;
      rec._edited = [...new Set([...(rec._edited || []), field])];
      rec._touched = true;
    }
    refresh(key);
  }
  onChange(key);
  map.draw();
}

const fmtN = n => (n == null ? null : Number(n).toLocaleString());

function stats(c) {
  // Read-only facts, so the panel is a real detail view rather than a form.
  // Missing values say so explicitly — a reference card that hides its gaps
  // looks finished when it is not, which is the opposite of what this is for.
  const gap = '<span class="missing">&mdash;</span>';
  const rows = [
    ['Population', fmtN(c.pop) || gap],
    ['GDP / capita', c.gdpPc
      ? `$${fmtN(c.gdpPc)} <em class="src">${esc(c.gdpSrc || '')} ${c.gdpYear || ''}</em>`
      : gap],
    ['Altitude', c.elev != null ? `${fmtN(c.elev)} m` : gap],
    ['Coordinates', (c.lat != null && c.lon != null)
      ? `${Math.abs(c.lat).toFixed(3)}&deg;${c.lat < 0 ? 'S' : 'N'} ` +
        `${Math.abs(c.lon).toFixed(3)}&deg;${c.lon < 0 ? 'W' : 'E'}` : gap],
    ['Country', esc(c.countryName || '') || gap],
  ];
  return `<dl class="estats">${rows.map(([k, v]) =>
    `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>`;
}

function links(c, key) {
  const out = [];
  if (c.wiki) {
    const u = 'https://en.wikipedia.org/wiki/' + encodeURIComponent(c.wiki.replace(/ /g, '_'));
    out.push(`<a class="btn wide" href="${u}" target="_blank" rel="noopener">Wikipedia &#8599;</a>`);
  }
  if (key.startsWith('Q')) {
    out.push(`<a class="btn" href="https://www.wikidata.org/wiki/${esc(key)}"
              target="_blank" rel="noopener" title="Wikidata">WD</a>`);
  }
  return out.length ? `<div class="elinks">${out.join('')}</div>` : '';
}

function altPicker(c) {
  // base.json carries only a few candidates (they feed search); the full list
  // arrives from /api/alts and is cached on the record.
  const cands = c._altsFull || c.altCandidates || [];
  if (!cands.length) return '';
  const chosen = new Set((c.altNames || []).map(s => s.toLowerCase()));
  return `<div class="apick">
    <span class="alab">Suggested alt names &mdash; click to add</span>
    ${cands.map(([lang, t]) => `
      <button class="acand${chosen.has(t.toLowerCase()) ? ' on' : ''}"
              data-alt="${esc(t)}"><em>${esc(lang)}</em>${esc(t)}</button>`).join('')}
  </div>`;
}

function langPicker(c) {
  // Country-level seed offered as clickable chips. Same pattern as alt names:
  // suggest, never assert. A city's languages are not its country's languages,
  // so nothing here is written until it is clicked.
  const cands = c.languageCandidates || [];
  if (!cands.length) return '';
  const chosen = new Set((c.languages || []).map(s => s.toLowerCase()));
  return `<div class="apick">
    <span class="alab">From country &mdash; click to add</span>
    ${cands.map(t => `
      <button class="acand lang${chosen.has(t.toLowerCase()) ? ' on' : ''}"
              data-lang="${esc(t)}">${esc(t)}</button>`).join('')}
  </div>`;
}

export function open(i) {
  if (i < 0) return;
  current = { i, key: state.keys[i] };
  const c = state.rec[i];
  const key = current.key;

  const readOnly = `
    <div class="ehead">
      <strong>${esc(c.name || key)}</strong>
      <button class="btn" id="e-close" title="Close">&times;</button>
    </div>
    <div class="esub">${esc(c.adminName || '')}</div>
    ${(c.typeNames || []).length
      ? `<div class="kinds">${(c.kind && KIND_LABEL[c.kind])
          ? `<span class="kindtag">${KIND_LABEL[c.kind]}</span>` : ''
        }${c.typeNames.map(t => `<span class="ktype">${esc(t)}</span>`).join('')}</div>`
      : ''}
    ${links(c, key)}
    ${stats(c)}`;

  if (!mode) {
    panel.innerHTML = readOnly +
      `<div class="ehint">Turn on Edit to curate this city.</div>`;
    panel.style.display = 'block';
    panel.querySelector('#e-close').onclick = close;
    return;
  }

  panel.innerHTML = readOnly + `
    ${TEXT_FIELDS.map(([f, label]) => {
      const v = c[f];
      const text = Array.isArray(v) ? v.join('\n') : (v ?? '');
      const edited = (c._edited || []).includes(f);
      const stale = (c._stale || []).includes(f);
      return `<label class="efield${edited ? ' edited' : ''}">
        <span>${label}${stale ? ' <em class="warn-i">source changed</em>' : ''}</span>
        ${f === 'name'
          ? `<input data-f="${f}" value="${esc(text)}">`
          : `<textarea data-f="${f}" rows="3">${esc(text)}</textarea>`}
      </label>${f === 'altNames' ? altPicker(c)
                : f === 'languages' ? langPicker(c) : ''}`;
    }).join('')}
    <div class="etools">
      <button class="btn wide" id="e-move">Move</button>
      <button class="btn wide" id="e-del">Delete</button>
    </div>
    <div class="ehint">Changes save when you leave a field.</div>`;
  panel.style.display = 'block';

  panel.querySelector('#e-close').onclick = close;
  panel.querySelectorAll('[data-f]').forEach(el => {
    el.onblur = () => commit(el.dataset.f, el.value);
  });
  // Pull the full candidate list once per city, then re-render the picker.
  if (mode && !c._altsFull && key.startsWith('Q')) {
    fetch(`/api/alts/${encodeURIComponent(key)}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!d || !d.alt || !d.alt.length) { c._altsFull = c.altCandidates || []; return; }
        c._altsFull = d.alt;
        if (current && current.key === key) open(current.i);
      })
      .catch(() => { c._altsFull = c.altCandidates || []; });
  }
  panel.querySelectorAll('.acand').forEach(b => {
    b.onclick = () => {
      const t = b.dataset.alt;
      const cur = state.rec[current.i].altNames || [];
      const has = cur.some(x => x.toLowerCase() === t.toLowerCase());
      const next = has ? cur.filter(x => x.toLowerCase() !== t.toLowerCase())
                       : [...cur, t].slice(0, 3);
      patch(current.key, 'altNames', next.length ? next : null)
        .then(() => open(current.i))
        .catch(err => alert(err.message));
    };
  });
  panel.querySelectorAll('.acand.lang').forEach(b => {
    b.onclick = () => {
      const t = b.dataset.lang;
      const cur = state.rec[current.i].languages || [];
      const has = cur.some(x => x.toLowerCase() === t.toLowerCase());
      const next = has ? cur.filter(x => x.toLowerCase() !== t.toLowerCase())
                       : [...cur, t];
      patch(current.key, 'languages', next.length ? next : null)
        .then(() => open(current.i))
        .catch(err => alert(err.message));
    };
  });
  panel.querySelector('#e-move').onclick = () => {
    setTool('move');
    panel.querySelector('#e-move').classList.toggle('active', tool === 'move');
  };
  panel.querySelector('#e-del').onclick = async () => {
    if (!confirm(`Delete ${state.rec[current.i].name || current.key}?`)) return;
    await api('DELETE', `/api/city/${encodeURIComponent(current.key)}`);
    map.markDeleted(current.key);
    close();
  };
}

function commit(f, raw) {
  raw = raw.trim();
  let value;
  if (f === 'name') value = raw || null;
  else {
    const lines = raw.split('\n').map(s => s.trim()).filter(Boolean);
    value = lines.length ? (LIST_FIELDS.includes(f) ? lines.slice(0, 3) : lines) : null;
  }
  const before = state.rec[current.i][f];
  if (JSON.stringify(before ?? null) === JSON.stringify(value)) return;
  patch(current.key, f, value).catch(err => alert(err.message));
}

export function close() {
  current = null;
  if (panel) panel.style.display = 'none';
}

// --- tools driven from a map click -----------------------------------------

export async function handleMapClick(i, ev) {
  if (mode && tool === 'create') {
    const { nx, ny } = map.unproject(ev.clientX, ev.clientY);
    const { lat, lon } = map.toLatLon(nx, ny);
    const name = prompt('Name of the new city?');
    if (!name) return true;
    const res = await api('POST', '/api/city', { lat, lon, name });
    const rec = { name, lat, lon, pop: null, kind: 'city',
                  _touched: true, _created: true, _edited: [] };
    const idx = add(res.key, rec);
    setTool(null);
    map.draw();
    open(idx);
    return true;
  }

  if (mode && tool === 'move' && current) {
    const { nx, ny } = map.unproject(ev.clientX, ev.clientY);
    const { lat, lon } = map.toLatLon(nx, ny);
    await patch(current.key, 'lat', lat);
    await patch(current.key, 'lon', lon);
    state.nx[current.i] = nx; state.ny[current.i] = ny;
    setTool(null);
    map.draw();
    open(current.i);
    return true;
  }

  if (i >= 0) { open(i); return true; }
  close();
  return false;
}

export async function rebuild() {
  return api('POST', '/api/build');
}
