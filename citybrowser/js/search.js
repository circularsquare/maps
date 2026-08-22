// Search. With 61,866 cities the only way to reach one was panning the globe,
// which made curation impractical — this is the difference between a map you
// look at and one you can work in.
//
// The index includes alt-name candidates, so "Köln", "Cologne" and
// "Constantinople" all find their city. Matching is on a normalised form
// (lowercase, diacritics stripped) so "Sao Paulo" finds "São Paulo".
//
// WORD BOUNDARIES ARE PRESERVED, deliberately. A first version stripped spaces
// and ranked only "starts with" above "contains" — so searching "york" returned
// York County, York Township and York urban area, and NOT New York, which was
// buried with the mid-word matches. Matching a whole word is nearly as strong a
// signal as matching the start.

import { state } from './data.js';

let idx = null;          // index -> normalised haystack (words separated by ' ')

export function norm(s) {
  return (s || '')
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')       // strip combining diacritics
    .toLowerCase()
    .replace(/[^a-z0-9　-鿿]+/g, ' ')
    .trim();
}

function build() {
  idx = new Array(state.n);
  for (let i = 0; i < state.n; i++) {
    const c = state.rec[i];
    const parts = [norm(c.name)];
    for (const a of (c.altNames || [])) parts.push(norm(a));
    const alts = c.altCandidates || [];
    for (let k = 0; k < alts.length && k < 8; k++) parts.push(norm(alts[k][1]));
    idx[i] = parts.join(' | ');
  }
}

/**
 * Ranked matches, best first.
 *   0  the query starts a WORD    ("york" -> York, New York, Yorkville)
 *   1  the query matches mid-word ("york" -> Little Yorkshire)
 * Population breaks ties.
 *
 * Only two ranks on purpose. Ranking "starts the whole name" above "starts a
 * word" buried New York (8.8M) beneath a dozen York Counties and York
 * Townships, because the limit filled with rank-0 hits before any rank-1 was
 * considered. Starting a word is a strong enough signal on its own; size
 * decides the rest.
 *
 * `kinds` limits results to the kinds currently drawn — offering to fly to a
 * metro area that the map is hiding is just confusing.
 */
export function search(qRaw, limit = 12, kinds = null) {
  if (!idx) build();
  const q = norm(qRaw);
  if (q.length < 2) return [];
  const out = [];
  for (let i = 0; i < state.n; i++) {
    if (kinds && !kinds.has(state.kind[i])) continue;
    const h = idx[i];
    const pos = h.indexOf(q);
    if (pos < 0) continue;
    const rank = (pos === 0 || h[pos - 1] === ' ') ? 0 : 1;
    out.push([rank, -(state.rec[i].pop || 0), i]);
  }
  out.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return out.slice(0, limit).map(([, , i]) => i);
}

export function invalidate() { idx = null; }
