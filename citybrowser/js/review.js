// The GHS match review queue.
//
// match_ghs.py cannot decide two classes of case and deliberately does not try:
//
//   near   + low   an urban centre nobody claims, and the best nearby city.
//                  Conakry offered the blob GHS calls "Coyah".
//   centre + low   the names agree but the populations are 3x+ apart. Either
//                  GHS merged (New Delhi's blob is 31M against a 250k council)
//                  or GHS fragmented (its "Fort Worth" is a 102k sliver of a
//                  918k city).
//
// Both need a person. This is where that happens: one item at a time, the map
// flown to the city so the geography is visible, keyboard-driven because there
// are ~2,800 of them and a queue that needs the mouse will never be finished.
//
// Decisions are ordinary field overrides, so they land in overrides.json and
// survive a refetch like every other correction. They are exempt from
// `_touched` — see REVIEW_FIELDS in data.js.

import { state } from './data.js';
import * as map from './map.js';
import { subtitle } from './card.js';

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = n => (n == null ? '—' : Number(n).toLocaleString());

let el = null, onDone = () => {};
let queue = [], at = 0;
// Session-only. A skip means "not now", not "decided" — persisting it would
// quietly shrink the queue with no record of why.
const skipped = new Set();

/** Every city still awaiting a decision. */
export function pending() {
  const out = [];
  for (let i = 0; i < state.n; i++) {
    const c = state.rec[i];
    if (c.ghsConf !== 'low' || c.ghs == null) continue;
    if (skipped.has(state.keys[i])) continue;
    // Already decided: the review wrote one of these fields.
    if ((c._edited || []).some(f => f === 'ghs' || f === 'ghsConf')) continue;
    out.push(i);
  }
  return out;
}

export function init(node, handlers = {}) {
  el = node;
  onDone = handlers.onDone || onDone;
  addEventListener('keydown', key);
}

export function isOpen() { return el && el.style.display === 'block'; }

export function open() {
  queue = pending();
  at = 0;
  // Biggest first — a wrong match on Conakry matters more than on a village,
  // and it front-loads the cases where judgement is easiest.
  queue.sort((a, b) => (state.rec[b].pop || 0) - (state.rec[a].pop || 0));
  el.style.display = 'block';
  render();
}

export function close() {
  el.style.display = 'none';
  onDone();
}

function key(e) {
  if (!isOpen()) return;
  if (e.target && /^(INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === 'escape') { close(); e.preventDefault(); }
  else if (k === 'a' || k === 'y') { decide(true); e.preventDefault(); }
  else if (k === 'r' || k === 'n') { decide(false); e.preventDefault(); }
  else if (k === 's' || k === 'arrowright') { skip(); e.preventDefault(); }
}

function current() {
  return at < queue.length ? queue[at] : -1;
}

function advance() {
  at++;
  render();
}

function skip() {
  const i = current();
  if (i >= 0) skipped.add(state.keys[i]);
  advance();
}

async function patch(key, field, value) {
  const r = await fetch('/api/city/' + key, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ field, value }),
  });
  if (!r.ok) throw new Error(`${field}: ${r.status}`);
  return r.json();
}

async function decide(accept) {
  const i = current();
  if (i < 0) return;
  const key = state.keys[i], c = state.rec[i];
  const before = { ghsConf: c.ghsConf, ghsCentre: c.ghsCentre };

  // A rejection is `ghsConf = "none"`, and ONLY that.
  //
  // The obvious version — also null out `ghs` — is a trap: in the PATCH API
  // `value: null` means "clear this override and revert to base", not "set the
  // field to null" (serve.py do_PATCH). Sending it would delete the rejection
  // instead of recording it, and the item would reappear in the queue forever.
  //
  // So `ghs` keeps pointing at the centre that was considered, which is worth
  // having anyway — it is the record of what was rejected. `ghsConf: "none"` is
  // what everything downstream reads, so card.js and data.js gate on that.
  const edited = new Set(c._edited || []);
  edited.add('ghsConf');
  // Optimistic, then rolled back on failure — a queue that waits for a round
  // trip before showing the next item is a queue nobody finishes.
  c.ghsConf = accept ? 'high' : 'none';
  if (!accept) c.ghsCentre = undefined;
  c._edited = [...edited].sort();
  advance();

  try {
    await patch(key, 'ghsConf', accept ? 'high' : 'none');
  } catch (err) {
    Object.assign(c, before);
    c._edited = (c._edited || []).filter(f => f !== 'ghsConf');
    if (c._edited.length === 0) delete c._edited;
    const s = document.getElementById('status');
    if (s) s.textContent = 'review save failed: ' + err.message;
  }
}

function render() {
  const i = current();
  if (i < 0) {
    el.innerHTML = `<div class="rvhead"><b>Review matches</b>
        <button class="btn rvx" data-act="close">&times;</button></div>
      <div class="rvdone">Nothing left in the queue.<br>
        <em>${skipped.size ? skipped.size.toLocaleString() +
          ' skipped this session — reopen to see them again.' : 'All decided.'}</em>
      </div>`;
    wire();
    return;
  }
  const c = state.rec[i], k = state.keys[i], g = c.ghsCentre || {};
  const centre = c.ghsRole === 'centre';
  const ratio = g.pop ? (c.pop || 0) / g.pop : 0;

  map.flyTo(c.lat, c.lon, centre ? 300 : 160);

  // The two cases are asked as different QUESTIONS, because they are. One is
  // "is this the same place?", the other "is this blob a sane description of
  // this city?" — and answering the second as though it were the first is how
  // a 102k "Fort Worth" gets confirmed.
  const ask = centre
    ? `GHS names this centre <b>${esc(g.name)}</b> too, but its population is
       <b>${ratio >= 1 ? (ratio).toFixed(1) + '&times; smaller' :
            (1 / ratio).toFixed(1) + '&times; larger'}</b>.
       Is it still the same place?`
    : `This urban centre has no city claiming it. Is <b>${esc(c.name)}</b> it?`;

  el.innerHTML = `
    <div class="rvhead">
      <b>Review matches</b>
      <span class="rvn">${(at + 1).toLocaleString()} of ${queue.length.toLocaleString()}</span>
      <button class="btn rvx" data-act="close">&times;</button>
    </div>
    <div class="rvcity">
      <h3>${esc(c.name)}</h3>
      <div class="rvsub">${esc(subtitle(c))}</div>
      <div class="rvpop">${fmt(c.pop)}<em>city</em></div>
    </div>
    <div class="rvask">${ask}</div>
    <div class="rvcentre">
      <div class="rvcname">${esc(g.name || '—')}${
        g.nMembers > 1 ? ` <em>&middot; ${g.nMembers} places</em>` : ''}</div>
      <dl class="uc">
        <dt>Contains</dt><dd>${fmt(g.pop)}</dd>
        <dt>Area</dt><dd>${fmt(g.area)} km&sup2;</dd>
        <dt>Distance</dt><dd>${c.ghsDistKm} km</dd>
      </dl>
      ${g.members && g.members.length > 1
        ? `<div class="umem">${g.members.slice(0, 8).map(esc).join(' &middot; ')}${
            g.nMembers > 8 ? ` &middot; +${g.nMembers - 8}` : ''}</div>` : ''}
    </div>
    <div class="rvacts">
      <button class="btn wide" data-act="accept">Accept</button>
      <button class="btn wide" data-act="reject">Reject</button>
      <button class="btn" data-act="skip">Skip</button>
    </div>
    <div class="rvhint"><b>a</b> accept &middot; <b>r</b> reject &middot;
      <b>s</b> skip &middot; <b>esc</b> close &middot; ${esc(k)}</div>`;
  wire();
}

function wire() {
  el.querySelectorAll('[data-act]').forEach(b => {
    b.onclick = () => ({
      close, skip,
      accept: () => decide(true),
      reject: () => decide(false),
    })[b.dataset.act]();
  });
}
