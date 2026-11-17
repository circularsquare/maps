// The trainworld board (T-3391, T-3392). Reads data.json, which build.py writes, and the inbox through
// serve.py's /twboard/_inbox; notes/T-3391.md has the formats.
'use strict';

const PRI_RANK = { P1: 1, P2: 2, P3: 3, P4: 4 };
const PRIS = ['P1', 'P2', 'P3', 'P4'];
const POLL_MS = 15000;
const $ = (sel) => document.querySelector(sel);

let DATA = null;
let HIST = null;            // history.json: the open-task counts per hour (history.py), for the graph
let HIST_BY = 'area';       // the graph stacks by area or by priority
let HIST_OPEN = true;
try { HIST_BY = localStorage.getItem('twb-histby') || 'area'; HIST_OPEN = localStorage.getItem('twb-histopen') !== '0'; } catch (e) { /* not remembered */ }
let SHOWP4 = false;          // P4 (Someday) tasks are left out of every open count unless this is on
try { SHOWP4 = localStorage.getItem('twb-showp4') === '1'; } catch (e) { /* not remembered */ }
/** a task the board counts and lists: all but parked P4 ones, unless Show P4 is on or its panel is the one open */
function counted(t) { return !!t && (SHOWP4 || !t.p4 || (panel && panel.kind === 'line' && panel.task === t.id)); }
let VIEW = null;            // the lines as the board shows them: data.json plus her queued moves
let INBOX = { ok: false, pending: [], pads: [], last_at: null, sent_at: null };
let panel = null;           // {kind: 'line', slug} | {kind: 'queue'} | {kind: 'pad'} | {kind: 'ack'}
const drafts = {};          // text typed in the panel, by field key, so a re-render keeps it
const opened = new Set();   // task comment boxes that are open
let confirm = null;         // {key, action}: a select change waiting for Queue change or Cancel
const withdrawn = new Set();
let REQ = null;             // requests.json: player request groups, their reports, pasted items
let view = 'board';         // 'board' | 'requests' | 'changelog'
const CL = { data: null, ok: true, status: {}, timers: {}, conflict: {}, menu: null }; // the changelog (T-3424); menu: the item whose area menu is open (T-3466)
const expanded = new Set(); // request groups showing their reports
let lastGroup = null;       // the group last opened, kept in the URL hash
let groupForm = null;       // {gid, kind: 'accept' | 'merge' | 'dismiss' | 'comment'}
const notesOpen = new Set(); // task rows showing their notes
const toolsOpen = new Set(); // task rows showing their priority and line selects
let flashTask = null;       // the task a link just landed on, highlighted briefly
let flashUntil = 0;
let ARCHIVE = null;         // archive.json, loaded the first time a closed task is opened
let ARCHIVED = new Set();   // closed task numbers, from data.json
const NOTES_CACHE = {};     // notes name -> {state: 'loading' | 'ok' | 'missing', text}
let hashApplied = false;    // the URL hash is read once data.json is in (a task id needs it)

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k === 'value') e.value = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid);
  return e;
}
const colour = (key) => `--c: var(--s-${key})`;

const SVG_NS = 'http://www.w3.org/2000/svg';
// small line icons, 16 px box, drawn in currentColor
const ICONS = {
  need: [['circle', { cx: 8, cy: 8, r: 7, fill: 'currentColor', stroke: 'none' }],
    ['path', { d: 'M8 4.2v4.6', style: 'stroke: var(--icon-on, #fff)', 'stroke-width': 1.9 }],
    ['circle', { cx: 8, cy: 11.4, r: 1.05, style: 'fill: var(--icon-on, #fff)', stroke: 'none' }]],
  context: [['rect', { x: 3, y: 3, width: 10, height: 10, rx: 1.5 }]],
  goal: [['path', { d: 'M2.5 8h10M9 4.5 12.5 8 9 11.5' }]],
  other: [['circle', { cx: 3.5, cy: 8, r: 1.3, fill: 'currentColor', stroke: 'none' }],
    ['circle', { cx: 8, cy: 8, r: 1.3, fill: 'currentColor', stroke: 'none' }],
    ['circle', { cx: 12.5, cy: 8, r: 1.3, fill: 'currentColor', stroke: 'none' }]],
  eye: [['path', { d: 'M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z' }], ['circle', { cx: 8, cy: 8, r: 2 }]],
  'eye-off': [['path', { d: 'M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z' }], ['circle', { cx: 8, cy: 8, r: 2 }],
    ['path', { d: 'M2.5 13.5 13.5 2.5' }]],
  star: [['path', { d: 'M8 2l1.8 3.7 4 .6-2.9 2.8.7 4L8 11.2 4.4 13.1l.7-4L2.2 6.3l4-.6z' }]],
  undo: [['path', { d: 'M5.5 3.5 2.5 6.5l3 3' }], ['path', { d: 'M2.5 6.5h7a3.5 3.5 0 0 1 0 7H6' }]],
  grip: [['circle', { cx: 5.5, cy: 3.5, r: 1.1 }], ['circle', { cx: 10.5, cy: 3.5, r: 1.1 }], ['circle', { cx: 5.5, cy: 8, r: 1.1 }],
    ['circle', { cx: 10.5, cy: 8, r: 1.1 }], ['circle', { cx: 5.5, cy: 12.5, r: 1.1 }], ['circle', { cx: 10.5, cy: 12.5, r: 1.1 }]],
  chev: [['path', { d: 'M6 4l4 4-4 4' }]],
  // the changelog's areas (T-3466), as the game draws them (app/src/ui/widgets.tsx, Changelog.tsx)
  'a-track': [['path', { d: 'M2.5 10.5l8-8M5.5 13.5l8-8M3 9l4 4M6 6l4 4M9 3l4 4' }]],
  'a-demand': [['circle', { class: 'bubble', cx: 8, cy: 8, r: 5.5 }]],
  'a-scheduling': [['path', { d: 'M1.5 2.5h13M1.5 13.5h13M2 13.5l4-11M7.5 2.5l4 11M10 13.5l4-11' }]],
  'a-fares': [['rect', { x: 1.5, y: 3.5, width: 10, height: 6.5 }], ['circle', { cx: 6.5, cy: 6.75, r: 1.4 }],
    ['circle', { class: 'bg', cx: 11.5, cy: 11.5, r: 3 }], ['circle', { cx: 11.5, cy: 11.5, r: 1.2 }]],
  'a-schematic': [['path', { d: 'M1.5 4l4-1.5 5 2 4-1.5v9.5l-4 1.5-5-2-4 1.5zM5.5 2.5V12M10.5 4.5V14' }]],
  'a-community': [['circle', { cx: 8, cy: 5.2, r: 2.9 }], ['path', { d: 'M2.4 14.2c.6-3 2.9-4.6 5.6-4.6s5 1.6 5.6 4.6' }]],
  'a-performance': [['rect', { class: 'screen', x: 1.5, y: 2.5, width: 13, height: 8.5, rx: 1 }], ['path', { d: 'M8 11v2.5M5 13.8h6' }]],
  'a-other': [['circle', { cx: 8, cy: 8, r: 2, fill: 'currentColor', stroke: 'none' }]],
  'a-bug': [['path', { d: 'M6.2 4.8a1.8 1.8 0 013.6 0' }], ['rect', { x: 4.8, y: 4.8, width: 6.4, height: 9.2, rx: 3.2 }],
    ['path', { d: 'M8 7.6V14M4.8 8.6H2.4M4.8 11.6l-2.2 1.5M11.2 8.6h2.4M11.2 11.6l2.2 1.5M5.3 6.3L3.4 4.6M10.7 6.3l1.9-1.7' }]],
};
function icon(name) {
  const s = document.createElementNS(SVG_NS, 'svg');
  for (const [k, v] of Object.entries({ viewBox: '0 0 16 16', width: 14, height: 14, fill: 'none', stroke: 'currentColor',
    'stroke-width': 1.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true', class: `ic ic-${name}` })) s.setAttribute(k, v);
  for (const [tag, attrs] of ICONS[name]) {
    const e = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    s.append(e);
  }
  return s;
}

/** a priority as a small pill in its colour (P1 strongest) */
function priBadge(p) {
  return p && PRI_RANK[p] ? el('span', { class: `pri ${p.toLowerCase()}` }, p) : el('span', { class: 'pri none' }, 'no priority');
}
/** text with its P1..P4 shown as pills, the rest as rich text */
function withPri(text) {
  return String(text).split(/\b(P[1-4])\b/).flatMap((part, i) => (i % 2 ? [priBadge(part)] : [].concat(rich(part))));
}
/** a priority select tinted in the chosen priority's colour */
function tintPri(sel) {
  const set = () => { sel.dataset.pri = sel.value; };
  set();
  sel.addEventListener('change', set);
  return sel;
}

function ago(iso) {
  const t = Date.parse(iso);
  if (!isFinite(t)) return '';
  const m = Math.max(0, Math.round((Date.now() - t) / 60000));
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  if (h < 48) return m % 60 ? `${h} h ${m % 60} min` : `${h} h`;
  return `${Math.floor(h / 24)} days`;
}
function clock(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return '';
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function day(s) {
  const d = new Date(s + 'T12:00:00');
  return isNaN(d) ? s : d.toLocaleDateString([], { day: 'numeric', month: 'short' });
}

// ---------- loading ----------

async function rebuild() {
  const btn = $('#refresh');
  btn.disabled = true;
  setBuilt('Rebuilding');
  let failed = null;
  try {
    const r = await fetch('_run/rebuild', { method: 'POST', headers: { 'X-Maps-Run': '1' } });
    if (r.ok) {
      let s = await r.json();
      const t0 = Date.now();
      while (s.running && Date.now() - t0 < 30000) {
        await new Promise((res) => setTimeout(res, 250));
        s = await (await fetch('_run/rebuild', { cache: 'no-store' })).json();
      }
      if (s.running) failed = 'rebuild is taking long, showing the last one';
      else if (s.code) failed = `rebuild failed: ${s.line}`;
    }
    // a 404 is a serve.py started before the rebuild action existed: show the data as it is
  } catch (e) {
    failed = 'server not reachable';
  }
  await fetchInbox();
  await load(failed);
  btn.disabled = false;
}

async function load(failed) {
  try {
    const r = await fetch(`data.json?t=${Date.now()}`, { cache: 'no-store' });
    DATA = await r.json();
  } catch (e) {
    setBuilt('No board data yet', true);
    return;
  }
  try {
    const r = await fetch(`requests.json?t=${Date.now()}`, { cache: 'no-store' });
    if (r.ok) REQ = await r.json();
  } catch (e) { /* a build from before stage 3: no requests yet */ }
  try {
    const r = await fetch(`history.json?t=${Date.now()}`, { cache: 'no-store' });
    HIST = r.ok ? await r.json() : null;
  } catch (e) { HIST = null; }
  ARCHIVED = new Set(DATA.archived || []);
  ARCHIVE = null;
  for (const k of Object.keys(NOTES_CACHE)) if (NOTES_CACHE[k].state !== 'loading') loadNotes(k, true);
  const first = !hashApplied;
  if (first) { hashApplied = true; applyHash(); }
  if (panel && panel.kind === 'closed') loadArchive();
  setBuilt(failed ? `Built ${clock(DATA.built)}, ${failed}` : `Built ${clock(DATA.built)}`, !!failed);
  fillUpdates();
  render();
  if (first && panel && panel.task) {
    const row = $(`#panel [data-task="${panel.task}"]`);
    if (row) row.scrollIntoView({ block: 'center' });
  }
}

function setBuilt(text, bad) {
  const b = $('#built');
  b.textContent = text;
  b.classList.toggle('bad', !!bad);
}

function fillUpdates() {
  const sel = $('#update');
  if (sel.options.length > 1) return;
  for (const u of DATA.updates) sel.append(el('option', { value: u.key }, u.name));
}

// ---------- the inbox ----------

function padsSig(pads) { return (pads || []).map((p) => p.id + (p.handled ? 'h' : '')).join(); }

/** true when the pending list or the sent pads changed */
async function fetchInbox() {
  const before = INBOX.pending.map((a) => a.id).join();
  const beforePads = padsSig(INBOX.pads);
  try {
    const r = await fetch('_inbox', { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    INBOX = { ok: true, ...(await r.json()) };
    // a pad-send has its own list under the scratch pad; it is not a queued change
    INBOX.pending = INBOX.pending.filter((a) => a.type !== 'pad-send');
    INBOX.pads = INBOX.pads || [];
  } catch (e) {
    INBOX = { ok: false, pending: [], pads: [], last_at: null, sent_at: null };
  }
  return INBOX.pending.map((a) => a.id).join() !== before || padsSig(INBOX.pads) !== beforePads;
}

async function post(obj) {
  try {
    const r = await fetch('_inbox', {
      method: 'POST', headers: { 'X-Maps-Run': '1', 'Content-Type': 'application/json' }, body: JSON.stringify(obj),
    });
    if (!r.ok) throw new Error(r.status);
  } catch (e) {
    setBuilt(`Not saved: ${e.message === '413' ? 'too long' : 'server not reachable'}`, true);
    return false;
  }
  await fetchInbox();
  render();
  return true;
}

const queue = post;

async function withdraw(id) {
  withdrawn.add(id);
  await post({ handled: id, by: 'anita', note: 'withdrawn' });
}

/** pending actions that touch a line (itself or one of its tasks) or a task */
function pendingFor(slug, taskId) {
  return INBOX.pending.filter((a) => (taskId ? a.task === taskId : a.line === slug || (a.task && DATA.tasks[a.task] && DATA.tasks[a.task].line === slug)));
}

function sent() {
  return INBOX.sent_at && (!INBOX.last_at || INBOX.sent_at >= INBOX.last_at);
}

// her queued reorders and area moves, laid over data.json so the board shows where she put things
function buildView() {
  const lines = DATA.lines.map((l) => ({ ...l }));
  const bySlug = new Map(lines.map((l) => [l.slug, l]));
  let order = lines.map((l) => l.slug);
  for (const a of INBOX.pending) {
    const l = bySlug.get(a.line);
    if (!l) continue;
    if (a.type === 'area' && a.to) {
      l.area = a.to;
      order = order.filter((s) => s !== a.line).concat(a.line);
    } else if (a.type === 'reorder' && Array.isArray(a.order)) {
      const want = a.order.filter((s) => bySlug.has(s) && bySlug.get(s).area === l.area);
      const slots = order.map((s, i) => (want.includes(s) ? i : -1)).filter((i) => i >= 0);
      if (slots.length !== want.length) continue;
      slots.forEach((i, k) => { order[i] = want[k]; });
    }
  }
  if (SHOWP4) {
    for (const l of lines) {
      const counts = Object.fromEntries(DATA.columns.map((c) => [c.key, 0]));
      for (const id of l.tasks) counts[DATA.tasks[id].status] += 1;
      l.counts = counts;
      l.open = l.tasks.length;
      l.status = ['progress', 'you', 'high', 'low', 'after'].find((k) => counts[k]) || (l.done_recent ? 'done' : null);
    }
  }
  VIEW = order.map((s) => bySlug.get(s));
}

function lineName(slug) {
  const l = DATA.lines.find((x) => x.slug === slug);
  return l ? l.name : slug;
}
function areaName(slug) {
  const a = DATA.areas.find((x) => x.slug === slug);
  return a ? a.name : slug;
}

function askText(id) {
  const a = (DATA.asks || []).find((x) => x.id === id);
  const t = a ? a.title || a.text : id;
  return t.length > 70 ? t.slice(0, 70) + '...' : t;
}

function describe(a) {
  switch (a.type) {
    case 'reorder': return `Move ${lineName(a.line)} ${a.dir === 'down' ? 'down' : 'up'}`;
    case 'area': return `Move ${lineName(a.line)} to ${areaName(a.to)}`;
    case 'priority': return `Change ${a.task} to ${a.to}`;
    case 'move-task': return `Move ${a.task} to ${lineName(a.to)}`;
    case 'answer': return `Answer on ${a.task}: ${a.text}`;
    case 'ask-answer': return a.dismiss && !a.text ? `Dismiss ask: ${askText(a.ask)}` : `${a.dismiss ? 'Answer and dismiss' : 'Answer'} ask: ${askText(a.ask)}: ${a.text}`;
    case 'text-ack': return a.text ? `Text acknowledged: ${a.text}` : 'Text acknowledged';
    case 'add': {
      const where = [a.area ? areaName(a.area) : '', a.line ? lineName(a.line) : ''].filter(Boolean).join(', ');
      return `New item${where ? ` (${where})` : ''}: ${a.text}`;
    }
    case 'pad-send': return `Sent for filing: ${a.text}`;
    case 'comment': return `Comment on ${a.group ? groupTitle(a.group) : a.task || lineName(a.line)}: ${a.text}`;
    case 'accept': return `Accept ${groupTitle(a.group)} as ${a.priority} in ${a.line === 'new' ? 'a new line' : lineName(a.line)} (${areaName(a.area)})`;
    case 'merge': return `Merge ${groupTitle(a.group)} into ${a.task || lineName(a.line)}`;
    case 'dismiss': return `Dismiss ${groupTitle(a.group)}`;
    case 'paste': return `Report from ${a.source || 'elsewhere'}: ${a.text.length > 120 ? a.text.slice(0, 120) + '...' : a.text}`;
    default: return a.type || 'Action';
  }
}

function queuedList(actions) {
  return el('div', { class: 'queued-box' }, actions.map((a) => el('div', { class: 'queued-item', 'data-action': a.id },
    el('div', { class: 'what' }, a.type === 'priority' || a.type === 'accept' ? withPri(describe(a)) : rich(describe(a))),
    a.comment ? el('div', { class: 'why' }, a.comment) : null,
    el('div', { class: 'when' }, `Queued ${clock(a.at)} · `,
      el('button', { class: 'link', type: 'button', onclick: () => withdraw(a.id) }, 'Withdraw')))));
}

// ---------- filtering ----------

function query() { return $('#search').value.trim().toLowerCase(); }

function taskHits(t, q) {
  return t.id.toLowerCase().includes(q) || t.text.toLowerCase().includes(q)
    || Object.values(t.fields || {}).some((v) => v.toLowerCase().includes(q));
}

/** null when the line is hidden; else the number of its tasks that match the search (or -1: no search) */
function lineMatch(line, q, update) {
  if (!line.status) return null;
  if (update && line.update !== update) return null;
  if (!q) return -1;
  const own = line.name.toLowerCase().includes(q) || line.about.toLowerCase().includes(q) || line.slug.includes(q);
  const n = line.tasks.filter((id) => counted(DATA.tasks[id]) && taskHits(DATA.tasks[id], q)).length;
  if (!own && !n) return null;
  return n;
}

// ---------- rendering ----------

function render() {
  if (!DATA) return;
  buildView();
  renderTop();
  const req = view === 'requests';
  const cl = view === 'changelog';
  for (const id of ['#strip', '#history', '#grid-wrap', '#update']) $(id).hidden = req || cl;
  $('#requests').hidden = !req;
  $('#changelog').hidden = !cl;
  $('#search').placeholder = req ? 'Search requests' : cl ? 'Search the changelog' : 'Search lines and tasks';
  $('#v-board').setAttribute('aria-pressed', String(!req && !cl));
  $('#v-requests').setAttribute('aria-pressed', String(req));
  $('#v-changelog').setAttribute('aria-pressed', String(cl));
  if (cl) renderChangelog();
  else if (req) renderRequests();
  else {
    renderStrip();
    renderHistory();
    renderGrid();
  }
  renderWarnings();
  renderPanel();
}

function renderTop() {
  const n = INBOX.pending.length;
  for (const id of ['#pad', '#ack', '#queued', '#send']) $(id).hidden = !INBOX.ok;
  $('#queued').hidden = !INBOX.ok || !n;
  $('#queued').textContent = `${n} queued`;
  const send = $('#send');
  send.textContent = n && sent() ? 'Sent' : 'Send now';
  send.disabled = !n || !!sent();
  $('#nosave').hidden = INBOX.ok;
}

function renderStrip() {
  const live = DATA.live;
  const strip = $('#strip');
  strip.replaceChildren();
  const row = (label, ...kids) => el('div', { class: 'srow' }, el('span', { class: 'slab' }, label), el('div', { class: 'sitems' }, ...kids));

  // agents on the same task doing the same thing share one item
  const groups = [];
  for (const a of live.agents) {
    const doing = a.doing || a.what || '';
    const g = groups.find((x) => x.task === a.task && x.doing === doing && x.tree === a.tree);
    if (g) g.names.push(a.name);
    else groups.push({ task: a.task, doing, tree: a.tree, started: a.started, names: [a.name] });
  }
  strip.append(row(live.orchestrator ? `Agents, ${live.orchestrator}` : 'Agents',
    groups.length ? groups.map((g) => el('span', { class: 'sitem agent' },
      el('b', {}, g.names.join(', ')), ' ', taskLink(g.task),
      g.doing ? el('span', { class: 'doing' }, ' ', rich(g.doing)) : null,
      g.tree ? el('span', { class: 'time' }, ` in ${g.tree}`) : null,
      g.started ? el('span', { class: 'time' }, ` ${ago(g.started)}`) : null))
      : el('span', { class: 'quiet' }, 'None running')));

  const wt = DATA.worktrees || [];
  if (wt.length) {
    strip.append(row('Worktrees', wt.map((w) => el('span', { class: 'sitem' },
      el('b', {}, w.main ? 'main' : w.name), ' ', el('span', { class: 'mono' }, w.branch),
      w.date ? el('span', { class: 'time' }, ` ${ago(w.date)}`) : null))));
  }

  const b = live.builds || {};
  const builds = [['master', DATA.head || b.master], ['8800', b['8800']], ['live', b.live]].filter(([, v]) => v)
    .map(([k, v]) => el('span', { class: 'sitem' }, el('span', { class: 'time' }, `${k} `), el('span', { class: 'mono' }, v)));
  const c = DATA.cores;
  const claims = (c && c.claims) || [];
  const runs = live.runs.map((r) => el('span', { class: 'sitem' }, el('b', {}, r.name), r.doing ? ` ${r.doing}` : '',
    el('span', { class: 'time' }, ` ${ago(r.started)}`)));
  strip.append(el('div', { class: 'srow' },
    el('span', { class: 'slab' }, 'Builds'), el('div', { class: 'sitems' }, builds,
      el('span', { class: 'slab inline' }, 'Long runs'), runs.length ? runs : el('span', { class: 'sitem quiet' }, 'none'),
      el('span', { class: 'slab inline' }, 'Cores'),
      c && c.cap != null ? el('span', { class: 'sitem' }, `${c.claimed} of ${c.cap} claimed`) : el('span', { class: 'sitem quiet' }, 'ledger not readable'),
      claims.map((x) => el('span', { class: 'sitem' }, el('b', {}, x.name), ` ${x.cores}`, el('span', { class: 'time' }, ` ${x.age}`))))));
}

// ---------- open tasks over time ----------

const SVGNS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs, ...kids) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  for (const c of kids) if (c != null) n.append(c);
  return n;
}

function renderHistory() {
  const box = $('#history');
  box.replaceChildren();
  const rows = HIST && HIST.rows;
  if (!rows || rows.length < 2) { box.hidden = true; return; }
  box.hidden = false;
  const last = rows[rows.length - 1];
  const t1 = Date.parse(last.t);
  const week = rows.filter((r) => Date.parse(r.t) >= t1 - 7 * 864e5);
  const sum = (since, k) => rows.filter((r) => Date.parse(r.t) > t1 - since).reduce((a, r) => a + (r[k] || 0), 0);
  const first = week[0];
  const delta = last.n - first.n;
  const toggle = (key, label) => el('button', { type: 'button', 'aria-pressed': String(HIST_BY === key), onclick: () => {
    HIST_BY = key;
    try { localStorage.setItem('twb-histby', key); } catch (e) { /* not remembered */ }
    renderHistory();
  } }, label);
  box.append(el('div', { class: 'hhead' },
    el('b', {}, 'Open tasks'),
    el('span', { class: 'hnow' }, String(last.n)),
    el('span', { class: 'hmeta' }, `${delta > 0 ? '+' : ''}${delta} in 7 days`),
    el('span', { class: 'hmeta' }, `opened ${sum(864e5, 'opened')}, closed ${sum(864e5, 'closed')} in 24 hours`),
    el('span', { class: 'hmeta' }, `opened ${sum(7 * 864e5, 'opened')}, closed ${sum(7 * 864e5, 'closed')} in 7 days`),
    el('span', { class: 'hsw' }, toggle('area', 'By area'), toggle('pri', 'By priority')),
    el('button', { type: 'button', onclick: () => {
      HIST_OPEN = !HIST_OPEN;
      try { localStorage.setItem('twb-histopen', HIST_OPEN ? '1' : '0'); } catch (e) { /* not remembered */ }
      renderHistory();
    } }, HIST_OPEN ? 'Hide' : 'Show')));
  if (!HIST_OPEN) return;

  const byArea = HIST_BY === 'area';
  const series = byArea
    ? HIST.areas.map((a) => ({ key: a.slug, name: a.name, colour: `var(--a-${a.slug})` }))
    : [['P1', 'P1'], ['P2', 'P2'], ['P3', 'P3'], ['P4', 'P4'], ['none', 'No priority']].map(([k, n]) => ({ key: k, name: n, colour: k === 'none' ? 'var(--p-none)' : `var(--${k.toLowerCase()})` }));
  const field = byArea ? 'area' : 'pri';
  const W = 720, H = 130, L = 30, R = 34, T = 8, B = 16;
  const t0 = t1 - 7 * 864e5;
  const x = (t) => L + ((t - t0) / (t1 - t0)) * (W - L - R);
  const max = Math.max(...week.map((r) => r.n));
  const step = max > 300 ? 100 : 50;
  const top = Math.ceil(max / step) * step;
  const y = (v) => T + (1 - v / top) * (H - T - B);
  const s = svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Open tasks over the last 7 days' });
  for (let v = 0; v <= top; v += step) {
    s.append(svg('line', { class: 'grid', x1: L, x2: W - R, y1: y(v), y2: y(v) }));
    s.append(svg('text', { x: L - 4, y: y(v) + 3, 'text-anchor': 'end' }, String(v)));
  }
  // stacks, bottom series first
  const acc = week.map(() => 0);
  for (const sr of series) {
    const lo = acc.slice();
    week.forEach((r, i) => { acc[i] += (r[field] && r[field][sr.key]) || 0; });
    if (acc.every((v, i) => v === lo[i])) { sr.total = 0; continue; }
    const up = week.map((r, i) => `${x(Date.parse(r.t)).toFixed(1)},${y(acc[i]).toFixed(1)}`);
    const dn = week.map((r, i) => `${x(Date.parse(r.t)).toFixed(1)},${y(lo[i]).toFixed(1)}`).reverse();
    s.append(svg('polygon', { points: up.concat(dn).join(' '), fill: sr.colour }));
  }
  s.append(svg('polyline', { class: 'tot', points: week.map((r) => `${x(Date.parse(r.t)).toFixed(1)},${y(r.n).toFixed(1)}`).join(' ') }));
  s.append(svg('text', { class: 'totl', x: x(t1) + 4, y: y(last.n) + 3 }, String(last.n)));
  // a tick at each local midnight
  const d = new Date(t0); d.setHours(24, 0, 0, 0);
  for (; d.getTime() <= t1; d.setDate(d.getDate() + 1)) {
    const px = x(d.getTime());
    s.append(svg('line', { class: 'grid', x1: px, x2: px, y1: H - B, y2: H - B + 3 }));
    s.append(svg('text', { x: px, y: H - 3, 'text-anchor': 'middle' }, d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric' })));
  }
  const keyRows = series.filter((sr) => sr.total !== 0 && (last[field] && last[field][sr.key])).reverse().map((sr) => el('span', {},
    el('i', { style: `background: ${sr.colour}` }), sr.name, el('em', {}, String((last[field] && last[field][sr.key]) || 0))));
  box.append(el('div', { class: 'hbody' }, s, el('div', { class: 'hkey' }, keyRows)));
}

function taskLink(text) {
  const id = (String(text || '').match(/T-\d+/) || [])[0];
  if (!id || !knownTask(id)) return text || '';
  return el('button', { class: 'link', type: 'button', onclick: () => goTask(id) }, text);
}

function showTask(id) { goTask(id); }

function renderGrid() {
  const q = query();
  const update = $('#update').value;
  const grid = $('#grid');
  grid.replaceChildren();

  const shown = new Map();
  for (const line of VIEW) {
    const m = lineMatch(line, q, update);
    if (m !== null) shown.set(line.slug, m);
  }
  const perCol = {};
  perCol.upnext = upnextTasks(q, update).length;
  for (const line of VIEW) if (shown.has(line.slug)) perCol[line.status] = (perCol[line.status] || 0) + 1;
  const asks = (DATA.asks || []).filter((a) => !q || `${a.title || ''} ${a.text}`.toLowerCase().includes(q));
  if (asks.length && !update) perCol.you = (perCol.you || 0) + asks.length;

  const last = DATA.columns.length;   // columns 0 (the area names) .. last; a drag edge on the right of all but the last
  grid.append(el('div', { class: 'colhead', style: 'background: var(--bg)' }, colGrip(0)));
  DATA.columns.forEach((col, i) => {
    grid.append(el('div', { class: 'colhead', style: colour(col.key) },
      col.name, perCol[col.key] ? el('span', { class: 'n' }, ` ${perCol[col.key]}`) : null, i + 1 < last ? colGrip(i + 1) : null));
  });
  if (asks.length && !update) {
    grid.append(el('div', { class: 'rowhead' }, 'Asks, not tasks', colGrip(0)));
    for (const [i, col] of DATA.columns.entries()) {
      const cell = el('div', { class: 'cell', 'data-area': 'asks', 'data-status': col.key }, i + 1 < last ? colGrip(i + 1) : null);
      if (col.key === 'you') for (const a of asks) cell.append(askCard(a));
      grid.append(cell);
    }
  }
  for (const area of DATA.areas) {
    grid.append(el('div', { class: 'rowhead' }, area.name, colGrip(0)));
    for (const [i, col] of DATA.columns.entries()) {
      const cell = el('div', { class: 'cell', 'data-area': area.slug, 'data-status': col.key }, i + 1 < last ? colGrip(i + 1) : null);
      if (col.key === 'upnext') {
        for (const t of upnextTasks(q, update)) if (upnextArea(t) === area.slug) cell.append(taskCard(t));
        grid.append(cell);
        continue;
      }
      for (const line of VIEW) {
        if (line.area !== area.slug || line.status !== col.key || !shown.has(line.slug)) continue;
        cell.append(card(line, shown.get(line.slug)));
      }
      grid.append(cell);
    }
  }
  applyColWidths();
}

// Column widths: dragging the line between two columns moves width from one to the other, so the board keeps its
// width. Kept as fr weights (the widths in px when she let go) in localStorage, so they follow the window's size.
const COLW_KEY = 'twboard.colw';
let COLW = null;
try { const w = JSON.parse(localStorage.getItem(COLW_KEY)); if (Array.isArray(w) && w.every((x) => x > 0)) COLW = w; } catch (e) { /* default widths */ }
function applyColWidths() {
  const grid = $('#grid');
  const ok = COLW && DATA && COLW.length === DATA.columns.length + 1;
  grid.style.gridTemplateColumns = ok ? COLW.map((w) => `minmax(0, ${w}fr)`).join(' ') : '';
}
function colGrip(i) {
  return el('div', { class: 'col-edge', title: 'Drag to resize, double-click to reset', ondblclick: () => {
    COLW = null;
    try { localStorage.removeItem(COLW_KEY); } catch (e) { /* not remembered */ }
    applyColWidths();
  }, onpointerdown: (e) => {
    e.preventDefault();
    const grid = $('#grid');
    const heads = [...grid.querySelectorAll('.colhead')];
    const start = heads.map((h) => h.getBoundingClientRect().width);
    const x0 = e.clientX;
    const MIN = i === 0 ? 70 : 90;
    const edge = e.currentTarget;
    edge.setPointerCapture(e.pointerId);
    document.body.classList.add('col-dragging');
    const move = (ev) => {
      const pair = start[i] + start[i + 1];
      const a = Math.max(MIN, Math.min(pair - 90, start[i] + ev.clientX - x0));
      const w = start.slice();
      w[i] = a;
      w[i + 1] = pair - a;
      COLW = w.map((x) => Math.round(x));
      applyColWidths();
    };
    const up = () => {
      edge.removeEventListener('pointermove', move);
      edge.removeEventListener('pointerup', up);
      document.body.classList.remove('col-dragging');
      try { if (COLW) localStorage.setItem(COLW_KEY, JSON.stringify(COLW)); } catch (err) { /* not remembered */ }
    };
    edge.addEventListener('pointermove', move);
    edge.addEventListener('pointerup', up);
  } });
}

function askCard(a) {
  const isOpen = panel && panel.kind === 'ask' && panel.id === a.id;
  const q = INBOX.pending.filter((p) => p.type === 'ask-answer' && p.ask === a.id).length;
  return el('button', {
    class: 'card ask' + (isOpen ? ' open' : ''), type: 'button', 'data-ask': a.id, style: colour('you'),
    onclick: () => openPanel(isOpen ? null : { kind: 'ask', id: a.id }),
  }, el('div', { class: 'row' }, el('div', { class: 'name' }, a.title || a.text), el('div', { class: 'count' }, a.since ? day(a.since) : '')),
  q ? el('div', { class: 'queued' }, `${q} queued`) : null);
}

function askPanel(id) {
  const a = (DATA.asks || []).find((x) => x.id === id);
  if (!a) return [head('Ask'), el('p', { class: 'meta' }, 'This ask is gone: it was answered or removed.')];
  const k = `ask:${id}`;
  const mine = INBOX.pending.filter((p) => p.type === 'ask-answer' && p.ask === id);
  return [
    head(a.title || 'Ask'),
    el('div', { class: 'meta' }, [a.by ? `From ${a.by}` : '', a.since ? `since ${day(a.since)}` : ''].filter(Boolean).join(', ')),
    el('p', { class: 'about' }, rich(a.text)),
    mine.length ? queuedList(mine) : null,
    INBOX.ok ? el('div', { class: 'compose' },
      textBox(`${k}:answer`, 'Your answer', 2),
      el('button', { class: 'btn', type: 'button', onclick: () => submit(`${k}:answer`, { type: 'ask-answer', ask: id }) }, 'Send answer'),
      el('button', { class: 'btn', type: 'button', onclick: () => submit(`${k}:answer`, { type: 'ask-answer', ask: id, dismiss: true }, true) }, 'Dismiss')) : null,
  ];
}

// The "Up next" queue (notes/board-live.json, kept by tools/tasks.py upnext): one card per task, in queue order.
function upnextTasks(q, update) {
  return (DATA.upnext || []).map((id) => DATA.tasks[id]).filter((t) => t && (!q || taskHits(t, q)) &&
    (!update || (DATA.lines.find((l) => l.slug === t.line) || {}).update === update));
}
function upnextArea(t) { return (DATA.lines.find((l) => l.slug === t.line) || {}).area || 'ops'; }
function taskCard(t) {
  const line = DATA.lines.find((l) => l.slug === t.line);
  return el('button', { class: 'card task', type: 'button', style: colour('upnext'), 'data-task': t.id, onclick: () => goTask(t.id) },
    el('div', { class: 'row' }, el('div', { class: 'name' }, t.id), priBadge(t.pri)),
    el('div', { class: 'ttl' }, t.text),
    line ? el('div', { class: 'sub' }, line.name) : null);
}

function card(line, hits) {
  let count;
  if (line.status === 'done') count = line.done ? `Done ${day(line.done)}` : 'Done';
  else count = hits > 0 ? `${hits} match` : `${line.open} open`;
  // one segment per open task in both bars, in the same order, so each task's status sits right above its priority
  const colRank = Object.fromEntries(DATA.columns.map((c, i) => [c.key, i]));
  const ts = line.tasks.map((id) => DATA.tasks[id]).filter(counted)
    .sort((a, b) => colRank[a.status] - colRank[b.status] || (PRI_RANK[a.pri] || 9) - (PRI_RANK[b.pri] || 9) || num(a.id) - num(b.id));
  const statusBar = el('div', { class: 'bar' }, ts.map((t) => el('span', { style: `background: var(--s-${t.status})` })));
  const priBar = el('div', { class: 'bar pri-bar' }, ts.map((t) => el('span', { style: `background: ${PRI_RANK[t.pri] ? `var(--${t.pri.toLowerCase()})` : 'var(--soft)'}` })));
  const bar = ts.length ? [statusBar, priBar] : null;
  const q = pendingFor(line.slug).length;
  const isOpen = panel && panel.kind === 'line' && panel.slug === line.slug;
  return el('button', {
    class: 'card' + (isOpen ? ' open' : ''), type: 'button', 'data-slug': line.slug,
    style: colour(line.status), onclick: () => openPanel(isOpen ? null : { kind: 'line', slug: line.slug }),
  }, el('div', { class: 'row' }, el('div', { class: 'name' }, line.name), el('div', { class: 'count' }, count)),
  line.open ? bar : null,
  q ? el('div', { class: 'queued' }, `${q} queued`) : null);
}

function openPanel(p) {
  panel = p;
  confirm = null;
  if (p && p.kind === 'line') view = 'board';
  setHash();
  render();
  if (p && p.kind !== 'line') $('#panel').scrollTop = 0;
}

// ---------- the side panel ----------

function renderPanel() {
  const box = $('#panel');
  if (panel && panel.kind === 'pad' && PAD.root && box.contains(PAD.root)) {
    // the pad's own nodes stay put, so typing and focus are not disturbed
    padShowStatus();
    padRenderSent();
    return;
  }
  if (!(panel && panel.kind === 'pad') && PAD.root) padFlush();
  const focus = document.activeElement && box.contains(document.activeElement) ? document.activeElement.dataset.key : null;
  const caret = focus ? [document.activeElement.selectionStart, document.activeElement.selectionEnd] : null;
  const scroll = box.scrollTop;
  let kids = null;
  if (panel && panel.kind === 'line') kids = linePanel(VIEW.find((l) => l.slug === panel.slug));
  else if (panel && panel.kind === 'ask') kids = askPanel(panel.id);
  else if (panel && panel.kind === 'queue') kids = queuePanel();
  else if (panel && panel.kind === 'pad') kids = padPanel();
  else if (panel && panel.kind === 'ack') kids = ackPanel();
  else if (panel && panel.kind === 'closed') kids = closedPanel(panel.id);
  else if (panel && panel.kind === 'notes') kids = notesPanel(panel.name);
  if (!kids) { box.hidden = true; box.replaceChildren(); $('#grip').hidden = true; return; }
  box.replaceChildren(...kids.filter(Boolean));
  box.hidden = false;
  $('#grip').hidden = false;
  box.scrollTop = scroll;
  if (focus) {
    const f = box.querySelector(`[data-key="${CSS.escape(focus)}"]`);
    if (f) {
      f.focus({ preventScroll: true });     // focus() alone scrolls the field into view: the snap she saw
      if (caret && f.setSelectionRange && caret[0] != null) try { f.setSelectionRange(caret[0], caret[1]); } catch (e) { /* a select */ }
    }
  }
  box.scrollTop = scroll;
  applyPendingScroll(box);
}

// Back and Forward return to where she was: the panel's scroll and her open notes are kept in each history entry.
let pendingScroll = null;   // {top, until}: waits for the notes to finish loading, then scrolls
let scrollTimer = null;
function saveScroll() {
  clearTimeout(scrollTimer);
  const box = $('#panel');
  if (!panel || box.hidden) return;
  try { history.replaceState({ sc: box.scrollTop, notes: [...notesOpen] }, ''); } catch (e) { /* no history */ }
}
function applyPendingScroll(box) {
  if (!pendingScroll) return;
  box.scrollTop = pendingScroll.top;
  if (Date.now() > pendingScroll.until || !box.querySelector('.notes.meta')) pendingScroll = null;
}
$('#panel').addEventListener('scroll', () => {
  if (pendingScroll) return;
  clearTimeout(scrollTimer);
  scrollTimer = setTimeout(saveScroll, 200);
});

function head(title) {
  return el('div', { class: 'head' }, el('h2', {}, title),
    el('button', { class: 'close', type: 'button', onclick: () => openPanel(null) }, 'Close'));
}

/** a textarea whose text survives re-renders */
function textBox(key, placeholder, rows = 2) {
  return el('textarea', {
    class: 'input', rows, placeholder, 'data-key': key, value: drafts[key] || '',
    oninput: (e) => { drafts[key] = e.target.value; },
  });
}
function textOf(key) { return (drafts[key] || '').trim(); }

async function submit(key, action, keepEmpty) {
  const text = textOf(key);
  if (!text && !keepEmpty) return;
  if (await queue({ ...action, ...(text ? { text } : {}) })) {
    delete drafts[key];
    render();
  }
}

/** a select that, once changed, asks for an optional comment before queueing */
function changeSelect(key, label, current, options, makeAction) {
  const sel = el('select', {
    class: 'small', 'data-key': key,
    onchange: (e) => {
      if (e.target.value === current) { confirm = null; } else { confirm = { key, action: makeAction(e.target.value) }; }
      renderPanel();
    },
  }, options.map((o) => (o.group
    ? el('optgroup', { label: o.group }, o.options.map((x) => el('option', { value: x.value }, x.label)))
    : el('option', { value: o.value }, o.label))));
  sel.value = confirm && confirm.key === key ? valueOf(confirm.action) : current;
  return el('label', { class: 'field' }, el('span', {}, label), sel);
}
function valueOf(a) { return a.to; }

function confirmRow(key) {
  if (!confirm || confirm.key !== key) return null;
  const ck = `${key}:why`;
  return el('div', { class: 'confirm' },
    el('input', { class: 'input', type: 'text', placeholder: 'Comment (optional)', 'data-key': ck, value: drafts[ck] || '',
      oninput: (e) => { drafts[ck] = e.target.value; },
      onkeydown: (e) => { if (e.key === 'Enter') doConfirm(key); } }),
    el('button', { class: 'btn', type: 'button', onclick: () => doConfirm(key) }, 'Queue change'),
    el('button', { class: 'btn plain', type: 'button', onclick: () => { confirm = null; delete drafts[ck]; renderPanel(); } }, 'Cancel'));
}
async function doConfirm(key) {
  const ck = `${key}:why`;
  const comment = textOf(ck);
  const action = { ...confirm.action, ...(comment ? { comment } : {}) };
  if (await queue(action)) {
    confirm = null;
    delete drafts[ck];
    render();
  }
}

function areaOptions() { return DATA.areas.map((a) => ({ value: a.slug, label: a.name })); }
function lineOptions(extra) {
  const groups = DATA.areas.map((a) => ({
    group: a.name,
    options: VIEW.filter((l) => l.area === a.slug && (l.slug !== 'not-sorted' || l.slug === extra)).map((l) => ({ value: l.slug, label: l.name })),
  }));
  return groups.filter((g) => g.options.length);
}

/** where up and down would take a line: its neighbours in the same cell */
function neighbours(line) {
  const cell = VIEW.filter((l) => l.area === line.area && l.status === line.status);
  const i = cell.findIndex((l) => l.slug === line.slug);
  return { prev: cell[i - 1], next: cell[i + 1] };
}

function move(line, dir) {
  const { prev, next } = neighbours(line);
  const other = dir === 'up' ? prev : next;
  if (!other) return;
  const area = VIEW.filter((l) => l.area === line.area).map((l) => l.slug).filter((s) => s !== line.slug);
  const at = area.indexOf(other.slug) + (dir === 'up' ? 0 : 1);
  area.splice(at, 0, line.slug);
  const action = { type: 'reorder', line: line.slug, area: line.area, dir, order: area };
  action[dir === 'up' ? 'before' : 'after'] = other.slug;
  queue(action);
}

function linePanel(line) {
  if (!line) return null;
  const upd = DATA.updates.find((u) => u.key === line.update);
  const q = query();

  const byStatus = {};
  let parked = 0;
  for (const id of line.tasks) { if (counted(DATA.tasks[id])) (byStatus[DATA.tasks[id].status] ||= []).push(DATA.tasks[id]); else parked += 1; }
  const groups = [];
  for (const col of DATA.columns) {
    const ts = byStatus[col.key];
    if (!ts) continue;
    ts.sort((a, b) => (PRI_RANK[a.pri] || 9) - (PRI_RANK[b.pri] || 9) || num(a.id) - num(b.id));
    groups.push(el('h3', { style: colour(col.key) }, el('span', { class: 'dot' }), col.name, el('span', { class: 'n' }, String(ts.length))));
    for (const t of ts) groups.push(taskRow(t, col.key, q));
  }

  const meta = [areaName(line.area), line.done && !line.open ? `done ${day(line.done)}` : ''].filter(Boolean).join(', ');
  const mine = INBOX.pending.filter((a) => a.line === line.slug && !a.task);
  const { prev, next } = neighbours(line);
  const k = `line:${line.slug}`;
  const editable = INBOX.ok && line.slug !== 'not-sorted';
  return [
    head(line.name),
    el('div', { class: 'meta' }, meta),
    upd ? el('div', { class: 'meta' }, `Part of the ${upd.name.toLowerCase()}`) : null,
    el('p', { class: 'about' }, rich(line.about)),
    parked ? el('div', { class: 'meta' }, `${parked} P4 ${parked === 1 ? 'task' : 'tasks'} hidden (Show P4 in the top bar)`) : null,
    editable ? el('div', { class: 'controls' },
      el('button', { class: 'btn', type: 'button', disabled: !prev, onclick: () => move(line, 'up') }, 'Move up'),
      el('button', { class: 'btn', type: 'button', disabled: !next, onclick: () => move(line, 'down') }, 'Move down'),
      changeSelect(`${k}:area`, 'Area', line.area, areaOptions(), (to) => ({ type: 'area', line: line.slug, from: line.area, to }))) : null,
    editable ? confirmRow(`${k}:area`) : null,
    mine.length ? queuedList(mine) : null,
    INBOX.ok ? el('div', { class: 'compose' },
      textBox(`${k}:comment`, 'Comment on this line', 1),
      el('button', { class: 'btn', type: 'button', onclick: () => submit(`${k}:comment`, { type: 'comment', line: line.slug }) }, 'Add comment')) : null,
    ...(groups.length ? groups : [el('p', { class: 'meta' }, 'No open tasks')]),
  ];
}

function num(id) { return parseInt(id.slice(2), 10); }

/** what a select shows: her latest queued value, else the files' */
function queuedTo(pend, type, current) {
  const last = pend.filter((a) => a.type === type).pop();
  return last ? last.to : current;
}

function taskRow(t, key, q) {
  const notes = [];
  if (t.agent) notes.push([`agent ${t.agent}`]);
  const waits = [...t.waits.map((w) => (w === 'Anita' ? 'you' : w)), ...t.after];
  if (waits.length) {
    const parts = ['waits on '];
    waits.forEach((w, i) => {
      if (i) parts.push(', ');
      if (/^T-\d+$/.test(w)) parts.push(taskLink(w), DATA.tasks[w] ? '' : ' (closed)');
      else parts.push(rich(w));
    });
    notes.push(parts);
  }
  if (t.section) notes.push([`listed under ${t.section}`]);
  const where = el('span', { class: 'where' });
  notes.forEach((n, i) => where.append(...(i ? [' · '] : []), ...n));

  const k = `task:${t.id}`;
  const pend = pendingFor(null, t.id);
  const waitsOnYou = t.waits.includes('Anita') && !pend.some((a) => a.type === 'answer');
  const showing = t.notes && notesOpen.has(t.id);
  if (t.notes) {
    where.append(...(notes.length ? [' · '] : []), el('button', { class: 'link', type: 'button', onclick: () => {
      if (showing) notesOpen.delete(t.id); else notesOpen.add(t.id);
      renderPanel();
    } }, showing ? 'Hide notes' : 'Show notes'));
  }
  // one small line: where it stands, then her controls; the priority and line selects stay folded until she opens them
  const small = el('div', { class: 'controls small' }, where);
  let after = [];
  if (INBOX.ok) {
    const toolsShown = toolsOpen.has(t.id) || !!(confirm && confirm.key.startsWith(`${k}:`));
    const sep = () => (where.childNodes.length ? ' · ' : '');
    if (!opened.has(t.id)) where.append(sep(), el('button', { class: 'link', type: 'button', onclick: () => { opened.add(t.id); renderPanel(); } }, 'Comment'));
    where.append(sep(), el('button', { class: 'link fold' + (toolsShown ? ' open' : ''), type: 'button', 'aria-expanded': String(toolsShown),
      onclick: () => { if (toolsShown) { toolsOpen.delete(t.id); if (confirm && confirm.key.startsWith(`${k}:`)) confirm = null; } else toolsOpen.add(t.id); renderPanel(); } },
    'Priority and line', icon('chev')));
    if (toolsShown) {
      const priOpts = (t.pri ? [] : [{ value: '', label: 'none' }]).concat(PRIS.map((p) => ({ value: p, label: p })));
      const pri = changeSelect(`${k}:pri`, 'Priority', queuedTo(pend, 'priority', t.pri || ''), priOpts, (to) => ({ type: 'priority', task: t.id, from: t.pri, to }));
      tintPri(pri.querySelector('select'));
      small.append(el('span', { class: 'tools' }, pri,
        changeSelect(`${k}:line`, 'Line', queuedTo(pend, 'move-task', t.line), lineOptions(t.line), (to) => ({ type: 'move-task', task: t.id, from: t.line, to }))));
    }
    after = [
      waitsOnYou ? el('div', { class: 'compose' },
        textBox(`${k}:answer`, 'Your answer', 2),
        el('button', { class: 'btn', type: 'button', onclick: () => submit(`${k}:answer`, { type: 'answer', task: t.id }) }, 'Save answer')) : null,
      opened.has(t.id) ? el('div', { class: 'compose' },
        textBox(`${k}:comment`, `Comment on ${t.id}`),
        el('button', { class: 'btn', type: 'button', onclick: async () => { await submit(`${k}:comment`, { type: 'comment', task: t.id, line: t.line }); if (!drafts[`${k}:comment`]) { opened.delete(t.id); renderPanel(); } } }, 'Add comment'),
        el('button', { class: 'btn plain', type: 'button', onclick: () => { opened.delete(t.id); delete drafts[`${k}:comment`]; renderPanel(); } }, 'Cancel')) : null,
    ];
  }
  const flash = flashTask === t.id && Date.now() < flashUntil;
  const pc = t.pri && PRI_RANK[t.pri] ? `; --pc: var(--${t.pri.toLowerCase()})` : '';
  // the row's parts in order; the notes sit right above the answer box so she can read and answer in one place
  return el('div', { class: 'task-row' + (q && taskHits(t, q) ? ' hit' : '') + (flash ? ' flash' : ''), style: colour(key) + pc, 'data-task': t.id },
    el('div', { class: 'top' }, el('b', { class: 'tid' }, t.id), priBadge(t.pri), el('span', { class: t.fields ? 'title' : 'text' }, rich(t.text))),
    taskBody(t),
    small,
    INBOX.ok ? confirmRow(`${k}:pri`) : null,
    INBOX.ok ? confirmRow(`${k}:line`) : null,
    showing ? notesBlock(t.notes, t.id) : null,
    pend.length ? queuedList(pend) : null,
    after);
}

/** a task's fields, each led by an icon in place of its label */
const FIELD_LABELS = [['need', 'Need from Anita'], ['context', 'Context'], ['goal', 'Goal'], ['other', 'Other']];
function taskBody(t) {
  if (!t.fields) return null;
  imgDir = 'trainworld/';
  const rows = FIELD_LABELS.filter(([k]) => t.fields[k]).map(([k, label]) =>
    el('div', { class: `field-row ${k}` }, el('span', { class: 'lab', title: label, 'aria-label': label }, icon(k)), el('span', { class: 'val' }, rich(t.fields[k]))));
  return rows.length ? el('div', { class: 'fields' }, rows) : null;
}

function queuePanel() {
  const list = INBOX.pending;
  return [
    head('Queued'),
    el('p', { class: 'about' }, 'The orchestrator picks these up 3 minutes after your last change, or when you press Send now.'),
    list.length ? queuedList(list) : el('p', { class: 'meta' }, 'Nothing queued'),
  ];
}

// ---------- the scratch pad (T-3396) ----------
// One free-text area, saved on the server as she types (/_pad), sent for filing as a pad-send action.

const PAD_IDLE_MS = 1500;
const PAD_PREVIEW = 200;
const PAD = {
  text: '', version: 0, loaded: false, status: 'loading', reason: '',
  saving: false, timer: null, retry: null, mine: null, root: null, sentList: null,
};
const padOpen = new Set();      // sent items showing their whole text

function padStatusText() {
  if (PAD.status === 'error') return `Not saved: ${PAD.reason}`;
  if (PAD.status === 'saving') return 'Saving';
  if (PAD.status === 'loading') return 'Loading';
  return 'Saved';
}
function padShowStatus() {
  if (!PAD.root) return;
  const s = PAD.root.querySelector('.pad-state');
  s.textContent = padStatusText();
  s.classList.toggle('bad', PAD.status === 'error');
  PAD.root.querySelector('.pad-send').disabled = !PAD.text.trim() || !!PAD.mine;
}

async function padLoad() {
  try {
    const r = await fetch('_pad', { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    const s = await r.json();
    PAD.text = s.text;
    PAD.version = s.version;
    PAD.loaded = true;
    PAD.status = 'saved';
  } catch (e) {
    PAD.status = 'error';
    PAD.reason = 'server not reachable';
  }
  if (PAD.root) PAD.root.querySelector('textarea').value = PAD.text;
  padShowStatus();
}

function padChanged() {
  PAD.status = 'saving';
  padShowStatus();
  clearTimeout(PAD.timer);
  PAD.timer = setTimeout(padSave, PAD_IDLE_MS);
}

/** PUT the text; resolves when it is saved or has failed. `beacon` is for a page that is going away */
async function padSave(beacon) {
  clearTimeout(PAD.timer);
  clearTimeout(PAD.retry);
  PAD.timer = null;
  if (PAD.saving || !PAD.loaded || PAD.mine) return;
  PAD.saving = true;
  const text = PAD.text;
  try {
    const r = await fetch('_pad', {
      method: 'PUT', keepalive: !!beacon && text.length < 30000,
      headers: { 'X-Maps-Run': '1', 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, version: PAD.version }),
    });
    if (r.status === 409) {
      const cur = await r.json();
      if (cur.text !== text) PAD.mine = text;
      PAD.text = cur.text;
      PAD.version = cur.version;
      PAD.status = 'saved';
      if (PAD.root) PAD.root.querySelector('textarea').value = PAD.text;
      padRenderMine();
    } else if (!r.ok) {
      throw new Error(r.status);
    } else {
      PAD.version = (await r.json()).version;
      PAD.status = PAD.text === text ? 'saved' : 'saving';
    }
  } catch (e) {
    PAD.status = 'error';
    PAD.reason = e.message === '413' ? 'too long' : 'server not reachable';
    if (e.message !== '413') PAD.retry = setTimeout(padSave, 5000);
  }
  PAD.saving = false;
  if (PAD.status === 'saving') PAD.timer = setTimeout(padSave, PAD_IDLE_MS);
  padShowStatus();
}

/** save now, if something is waiting */
function padFlush(beacon) {
  if (PAD.timer || PAD.status === 'error') padSave(beacon);
}

/** the "Your unsaved text" box, after another tab saved first */
function padRenderMine() {
  if (!PAD.root) return;
  const box = PAD.root.querySelector('.pad-mine');
  box.hidden = !PAD.mine;
  box.querySelector('textarea').value = PAD.mine || '';
  padShowStatus();
}

async function padSend() {
  const sentText = PAD.text;
  if (!sentText.trim() || PAD.mine) return;
  clearTimeout(PAD.timer);
  PAD.timer = null;
  while (PAD.saving) await new Promise((res) => setTimeout(res, 50));
  if (PAD.status !== 'saved') await padSave();
  if (PAD.status === 'error' || PAD.mine) return;
  if (!(await post({ type: 'pad-send', text: sentText }))) return;
  // what she typed after pressing stays for the next send
  const rest = PAD.text.startsWith(sentText) ? PAD.text.slice(sentText.length).replace(/^\s+/, '') : PAD.text;
  PAD.text = rest;
  PAD.root.querySelector('textarea').value = rest;
  PAD.status = 'saving';
  await padSave();
  padRenderSent();
}

/** the first ~200 characters, cut at a space */
function padPreview(text) {
  if (text.length <= PAD_PREVIEW) return text;
  const cut = text.lastIndexOf(' ', PAD_PREVIEW);
  return text.slice(0, cut > 100 ? cut : PAD_PREVIEW) + '...';
}

function padRenderSent() {
  if (!PAD.sentList) return;
  const pads = (INBOX.pads || []).slice().sort((a, b) => (a.id < b.id ? 1 : -1));
  PAD.sentList.replaceChildren(
    el('h3', {}, 'Sent for filing'),
    ...(pads.length ? pads.map((p) => {
      const long = p.text.length > PAD_PREVIEW;
      const whole = padOpen.has(p.id);
      return el('div', { class: 'pad-item' + (p.handled ? ' done' : ''), 'data-pad': p.id },
        el('div', { class: 'when' }, `Sent ${padStamp(p.at)}`),
        p.handled
          ? el('div', { class: 'note' }, p.handled.note ? rich(p.handled.note) : 'Filed')
          : el('div', { class: 'wait' }, 'Waiting to be filed'),
        el('div', { class: 'sent-text' }, rich(whole ? p.text : padPreview(p.text))),
        long ? el('button', {
          class: 'link', type: 'button',
          onclick: () => { if (whole) padOpen.delete(p.id); else padOpen.add(p.id); padRenderSent(); },
        }, whole ? 'Show less' : 'Show all') : null);
    }) : [el('p', { class: 'meta' }, 'Nothing sent in the last 14 days')]));
}

function padStamp(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return '';
  const today = new Date().toDateString() === d.toDateString();
  return today ? clock(iso) : `${d.toLocaleDateString([], { day: 'numeric', month: 'short' })}, ${clock(iso)}`;
}

function padBuild() {
  const area = el('textarea', {
    class: 'input pad-area', 'data-key': 'pad:text', spellcheck: 'true',
    placeholder: 'Type requests and ideas, as many as you like. Nothing is sent until you press Send for filing.',
    oninput: (e) => { PAD.text = e.target.value; padChanged(); },
    onblur: () => padFlush(),
  });
  area.value = PAD.text;
  const mine = el('div', { class: 'pad-mine', hidden: true },
    el('h3', {}, 'Your unsaved text'),
    el('p', { class: 'meta' }, 'The pad changed in another tab, so it now shows that version. What you typed here is below.'),
    el('textarea', { class: 'input', rows: 6, readonly: true }),
    el('div', { class: 'controls' },
      el('button', { class: 'btn', type: 'button', onclick: () => {
        const add = PAD.mine;
        PAD.mine = null;
        PAD.text = PAD.text ? `${PAD.text.replace(/\s+$/, '')}\n\n${add}` : add;
        area.value = PAD.text;
        padRenderMine();
        padChanged();
      } }, 'Add to the pad'),
      el('button', { class: 'btn plain', type: 'button', onclick: () => { PAD.mine = null; padRenderMine(); } }, 'Discard')));
  PAD.sentList = el('div', { class: 'pad-sent' });
  PAD.root = el('div', { class: 'pad' }, area, mine,
    el('div', { class: 'controls' },
      el('button', { class: 'btn pad-send', type: 'button', onclick: padSend }, 'Send for filing'),
      el('span', { class: 'pad-state' })),
    PAD.sentList);
  return PAD.root;
}

function padPanel() {
  if (!PAD.root) padBuild();
  padShowStatus();
  padRenderMine();
  padRenderSent();
  if (!PAD.loaded || (!PAD.timer && !PAD.saving && PAD.status !== 'error')) padLoad();
  return [head('Scratch pad'), PAD.root];
}

function ackPanel() {
  const k = 'ack';
  return [
    head('Text acknowledged'),
    el('p', { class: 'about' }, 'For text an agent listed for you to check.'),
    el('div', { class: 'compose' }, textBox(`${k}:text`, 'Which text (leave empty for all of it)', 3)),
    el('div', { class: 'controls' },
      el('button', { class: 'btn', type: 'button', onclick: async () => {
        await submit(`${k}:text`, { type: 'text-ack' }, true);
        openPanel({ kind: 'queue' });
      } }, 'Acknowledge')),
  ];
}

function renderWarnings() {
  const w = $('#warnings');
  const list = DATA.warnings || [];
  w.hidden = !list.length;
  if (!list.length) return;
  w.querySelector('summary').textContent = `Problems reading the task files (${list.length})`;
  w.querySelector('ul').replaceChildren(...list.map((s) => el('li', {}, rich(s))));
}

// ---------- player requests ----------

const KIND = { feature: 'Feature', error: 'Bug', feedback: 'Feedback', language: 'Language' };
const STATE = { accepted: 'Accepted', merged: 'Merged', dismissed: 'Dismissed' };
const SOURCES = ['Discord', 'Twitter', 'Other'];

function groupTitle(gid) {
  const g = REQ && REQ.groups.find((x) => x.id === gid);
  return g ? g.title : gid;
}

function hashNow() {
  if (panel && panel.kind === 'line' && panel.task) return `#task/${panel.task}`;
  if (panel && panel.kind === 'closed') return `#task/${panel.id}`;
  if (panel && panel.kind === 'notes') return `#notes/${panel.name}`;
  if (view === 'changelog') return '#changelog';
  if (view === 'requests') return lastGroup ? `#requests/${lastGroup}` : '#requests';
  if (panel && panel.kind === 'line') return `#${panel.slug}`;
  return '';
}

/** push: a jump Back should undo (a link to a task or notes); otherwise the hash is just kept current */
function setHash(push) {
  const h = hashNow();
  if (h === location.hash || (!h && !location.hash)) return;
  const url = h || location.pathname + location.search;
  if (push) history.pushState(null, '', url); else history.replaceState(history.state, '', url);
}

function applyHash() {
  const hash = decodeURIComponent(location.hash.slice(1));
  panel = null;
  view = 'board';
  if (hash === 'changelog') {
    view = 'changelog';
    loadChangelog();
    loadRoadmap();
  } else if (hash === 'requests' || hash.startsWith('requests/')) {
    view = 'requests';
    const g = hash.split('/')[1];
    if (g) { expanded.add(g); lastGroup = g; }
  } else if (hash.startsWith('task/')) {
    const id = hash.slice(5);
    if (DATA && DATA.tasks[id]) { panel = { kind: 'line', slug: DATA.tasks[id].line, task: id }; flash(id); }
    else panel = { kind: 'closed', id };
  } else if (hash.startsWith('notes/')) panel = { kind: 'notes', name: hash.slice(6) };
  else if (hash) panel = { kind: 'line', slug: hash };
  if (panel && panel.kind === 'closed') loadArchive();
}

// ---------- links to tasks and notes, and the notes view ----------

// T-1840, T 1840, T1840 (and T-1840's); notes paths as tasks cite them
const LINK_RE = /(notes\/(?:archive\/[\w-]+\/)?(?:[\w.-]+?\.md|T-\d+\/))|\bT[- ]?(\d{2,5})(?!\d)/g;

function knownTask(id) {
  return !!(DATA && (DATA.tasks[id] || ARCHIVED.has(parseInt(id.slice(2), 10))));
}

// An image path in any text the board shows becomes a thumbnail (served from maps/artifacts by serve.py's /_img).
// "artifacts/..." (also ../artifacts, maps/artifacts, C:\...\artifacts\) is exact; a bare or relative name ("b.png" after
// "artifacts/trainworld/t1191/a.png") is tried in the folder of the last exact one, and stays text if nothing is there.
const IMG_RE = /(?<![\w*.\/\\-])((?:[\w.:-]+[\\/])*?artifacts[\\/])?([\w][\w.-]*(?:[\\/][\w.-]+)*\.(?:png|jpe?g|webp))(?![\w\/\\])/gi;
let imgDir = 'trainworld/';  // where a bare image name is looked for: the folder of the last exact path seen
const DIR_RE = /(?<![\w*.\/\\-])(?:[\w.:-]+[\\/])*?artifacts[\\/]((?:[\w][\w.-]*[\\/])*)(?![\w.-])/gi;
function images(text) {
  const out = [];
  let at = 0;
  // a bare folder path ("artifacts/trainworld/t2625/") sets the folder for the bare names after it
  const dirs = [...text.matchAll(DIR_RE)].map((d) => ({ end: d.index + d[0].length, dir: d[1].replace(/\\/g, '/') }));
  let di = 0;
  for (const m of text.matchAll(IMG_RE)) {
    while (di < dirs.length && dirs[di].end <= m.index) imgDir = dirs[di++].dir;
    const rest = m[2].replace(/\\/g, '/');
    let src;
    if (m[1]) {
      src = rest;
      imgDir = rest.includes('/') ? rest.slice(0, rest.lastIndexOf('/') + 1) : '';
    } else src = imgDir + rest;
    if (m.index > at) out.push(text.slice(at, m.index));
    out.push(thumb(src, m[0]));
    at = m.index + m[0].length;
  }
  while (di < dirs.length) imgDir = dirs[di++].dir;
  if (!out.length) return null;
  if (at < text.length) out.push(text.slice(at));
  return out;
}
// Ctrl+V of a clipboard image into any board textarea: saved by serve.py's /_paste, its path typed in at the caret
document.addEventListener('paste', async (e) => {
  const area = e.target;
  if (!(area instanceof HTMLTextAreaElement) || area.readOnly) return;
  const file = [...(e.clipboardData ? e.clipboardData.items : [])].find((i) => i.kind === 'file' && /^image\/(png|jpeg|webp)$/.test(i.type));
  if (!file) return;
  e.preventDefault();
  const blob = file.getAsFile();
  const note = el('span', { class: 'meta paste-note' }, 'Adding image');
  area.after(note);
  const done = (msg, ms) => { note.textContent = msg; setTimeout(() => note.remove(), ms); };
  try {
    const r = await fetch('_paste', { method: 'POST', headers: { 'X-Maps-Run': '1', 'Content-Type': blob.type }, body: blob });
    if (!r.ok) throw new Error(r.status === 413 ? 'too big (10 MB)' : r.status);
    const { path } = await r.json();
    const at = area.selectionStart, end = area.selectionEnd, v = area.value;
    const pre = at && !/\s$/.test(v.slice(0, at)) ? '\n' : '';
    const ins = pre + path + '\n';
    area.value = v.slice(0, at) + ins + v.slice(end);
    area.selectionStart = area.selectionEnd = at + ins.length;
    area.dispatchEvent(new Event('input', { bubbles: true }));
    done('Image added', 2000);
  } catch (err) {
    done(`Image not added: ${err.message || err}`, 5000);
  }
});
function imgUrl(src) { return '_img/' + src.split('/').map(encodeURIComponent).join('/'); }
/** the path as a link, and a small picture of it at the end of its paragraph (list item, cell, field); both open it
 * large. If the file isn't there, the plain text comes back and no picture shows. */
function thumb(src, text) {
  const open = (e) => { e.stopPropagation(); showImage(src); };
  const link = el('button', { class: 'link img-link', type: 'button', title: `artifacts/${src}`, onclick: open }, text);
  const img = el('img', { src: imgUrl(src), alt: text, loading: 'lazy' });
  const pic = el('button', { class: 'thumb', type: 'button', title: `artifacts/${src}`, onclick: open }, img);
  img.addEventListener('error', () => { pic.remove(); if (link.isConnected || link.parentNode) link.replaceWith(document.createTextNode(text)); });
  // once the surrounding text is built, the picture joins its block's strip
  queueMicrotask(() => {
    const block = link.parentNode && link.parentElement.closest('p, li, td, th, .val, .text, .title, .rtext, .gsummary, .notes, div');
    if (!block) return;
    let strip = [...block.children].find((c) => c.classList.contains('thumbs'));
    if (!strip) {
      strip = el('span', { class: 'thumbs' });
      const sub = block.tagName === 'LI' ? [...block.children].find((c) => c.tagName === 'UL' || c.tagName === 'OL') : null;
      block.insertBefore(strip, sub || null);
    }
    if (!img.complete || img.naturalWidth) strip.append(pic);
  });
  return link;
}
function showImage(src) {
  const close = () => { box.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  const box = el('div', { class: 'lightbox', onclick: close },
    el('img', { src: imgUrl(src), alt: src }),
    el('div', { class: 'lb-cap' }, `artifacts/${src}`));
  document.addEventListener('keydown', onKey);
  document.body.append(box);
}

/** text with task ids and notes paths as links and image paths as thumbnails; anything that resolves to nothing stays plain */
function rich(text) {
  const s = String(text == null ? '' : text);
  if (!DATA) return s;
  const parts = images(s);
  if (parts) return parts.flatMap((p) => (typeof p === 'string' ? richLinks(p) : [p]));
  return richLinks(s);
}
function richLinks(s) {
  const out = [];
  let at = 0;
  for (const m of s.matchAll(LINK_RE)) {
    let node = null;
    if (m[1]) {
      const name = notesName(m[1]);
      if (name) node = el('button', { class: 'link', type: 'button', onclick: () => goNotes(name) }, m[1]);
    } else {
      const id = `T-${parseInt(m[2], 10)}`;
      if (knownTask(id)) node = el('button', { class: 'link', type: 'button', onclick: () => goTask(id) }, m[0]);
    }
    if (!node) continue;
    if (m.index > at) out.push(s.slice(at, m.index));
    out.push(node);
    at = m.index + m[0].length;
  }
  if (at < s.length) out.push(s.slice(at));
  return out;
}

/** "notes/archive/t1000-1499/T-1114.md" or "T-3390.md" -> its key in the notes index, or null */
function notesName(path) {
  const idx = DATA.notes_index || {};
  const p = String(path).replace(/^\.?\/?(notes\/)?/, '').replace(/#.*$/, '');
  const base = p.endsWith('/') ? p.split('/').slice(-2).join('/') : p.split('/').pop();
  return idx[base] ? base : null;
}

function flash(id) {
  flashTask = id;
  flashUntil = Date.now() + 1800;
}

function goTask(id) {
  saveScroll();
  if (DATA.tasks[id]) {
    view = 'board';
    panel = { kind: 'line', slug: DATA.tasks[id].line, task: id };
    confirm = null;
    flash(id);
    setHash(true);
    render();
    const row = $(`#panel [data-task="${id}"]`);
    if (row) row.scrollIntoView({ block: 'center' });
  } else {
    panel = { kind: 'closed', id };
    setHash(true);
    loadArchive();
    render();
    $('#panel').scrollTop = 0;
  }
}

function goNotes(name) {
  saveScroll();
  panel = { kind: 'notes', name };
  setHash(true);
  render();
  $('#panel').scrollTop = 0;
}

async function loadArchive() {
  if (ARCHIVE) return;
  try {
    const r = await fetch(`archive.json?t=${Date.now()}`, { cache: 'no-store' });
    ARCHIVE = r.ok ? await r.json() : {};
  } catch (e) {
    ARCHIVE = {};
  }
  renderPanel();
}

function closedPanel(id) {
  const k = [head(id)];
  if (!ARCHIVE) return k.concat(el('p', { class: 'meta' }, 'Loading'));
  const a = ARCHIVE[id];
  if (!a) return k.concat(el('p', { class: 'meta' }, 'Not in the task files'));
  const m = a.text.match(/\{L:([\w-]+)\}/);
  const line = m && DATA.lines.find((l) => l.slug === m[1]);
  return k.concat(
    el('div', { class: 'meta' }, `Closed, in ${a.file}`, line ? ' · ' : null,
      line ? el('button', { class: 'link', type: 'button', onclick: () => openPanel({ kind: 'line', slug: line.slug }) }, line.name) : null),
    el('div', { class: 'task-row closed', style: colour('done') }, el('div', { class: 'text' }, rich(a.text.replace(/\{(L|waits|after):[^}]*\}\s*/g, '')))),
    a.notes ? el('h3', {}, 'Notes') : null,
    a.notes ? notesBlock(a.notes, id) : null);
}

function notesPanel(name) {
  return [head(name), notesBlock(name, null)];
}

/** the notes file, rendered; a file several tasks share (someday.md) shows only this task's part, with the rest a click away */
function notesBlock(name, taskId) {
  const c = NOTES_CACHE[name];
  if (!c) { loadNotes(name); return el('div', { class: 'notes meta' }, 'Loading notes'); }
  if (c.state === 'loading') return el('div', { class: 'notes meta' }, 'Loading notes');
  if (c.state === 'missing') return el('div', { class: 'notes meta' }, 'Notes file not found');
  let text = c.text;
  let part = false;
  const shared = taskId && !name.startsWith(taskId + '.') && !name.startsWith(taskId + '-') && !name.startsWith(taskId + '/');
  if (shared) {
    const mine = excerpt(text, taskId);
    if (mine) { text = mine; part = true; }
  }
  return el('div', { class: 'notes' },
    el('div', { class: 'notes-src' }, part ? `From ${name} · ` : `${name} · `,
      el('button', { class: 'link', type: 'button', onclick: () => goNotes(name) }, part ? 'Show the whole file' : 'Open on its own')),
    md(text));
}

/** the list item or section about one task in a shared notes file, else null */
function excerpt(text, id) {
  const lines = text.split('\n');
  const rx = new RegExp(`^(\\s*[-*]\\s+|#+\\s+)${id}\\b`);
  const i = lines.findIndex((l) => rx.test(l));
  if (i < 0) return null;
  const headLevel = (lines[i].match(/^(#+)\s/) || [])[1];
  const out = [lines[i]];
  for (let j = i + 1; j < lines.length; j++) {
    const l = lines[j];
    if (headLevel ? new RegExp(`^#{1,${headLevel.length}}\\s`).test(l) : (/^\s*[-*]\s+T-\d+/.test(l) || /^#+\s/.test(l) || (!l.trim() && !/^\s+\S/.test(lines[j + 1] || '')))) break;
    out.push(l);
  }
  return out.join('\n');
}

/** quiet: a refresh that keeps showing the old text, and redraws only if the file changed (a redraw must not move her scroll) */
async function loadNotes(name, quiet) {
  const old = NOTES_CACHE[name];
  if (!quiet) NOTES_CACHE[name] = { state: 'loading' };
  const entry = (DATA.notes_index || {})[name];
  const paths = Array.isArray(entry) ? entry : entry ? [entry] : [];
  try {
    const texts = [];
    for (const p of paths) {
      const r = await fetch(`../twnotes/${p.split('/').map(encodeURIComponent).join('/')}`, { cache: 'no-store' });
      if (!r.ok) throw new Error(r.status);
      const t = await r.text();
      texts.push(paths.length > 1 ? `# ${p.split('/').pop()}\n\n${t}` : t);
    }
    NOTES_CACHE[name] = paths.length ? { state: 'ok', text: texts.join('\n\n') } : { state: 'missing' };
  } catch (e) {
    NOTES_CACHE[name] = quiet && old ? old : { state: 'missing' };
  }
  if (quiet && old && old.state === NOTES_CACHE[name].state && old.text === NOTES_CACHE[name].text) return;
  if (view === 'requests') renderRequests();
  renderPanel();
}

// A small markdown reader for the notes: headings, lists, tables, quotes, code, bold, links. It builds DOM nodes
// (never HTML strings), so nothing in a note can run as markup.
function md(src) {
  imgDir = 'trainworld/';
  const lines = src.replace(/\r/g, '').split('\n');
  const out = [];
  let i = 0;
  const isBlockStart = (l) => /^\s*```/.test(l) || /^#{1,6}\s/.test(l) || /^\s*([-*+]|\d+[.)])\s+/.test(l) || /^\s*>/.test(l) || /^\s*\|/.test(l);
  while (i < lines.length) {
    const l = lines[i];
    if (!l.trim()) { i++; continue; }
    if (/^\s*```/.test(l)) {
      const code = [];
      for (i++; i < lines.length && !/^\s*```/.test(lines[i]); i++) code.push(lines[i]);
      i++;
      out.push(el('pre', {}, el('code', {}, code.join('\n'))));
      continue;
    }
    const h = l.match(/^(#{1,6})\s+(.*)$/);
    if (h) { out.push(el('div', { class: `md-h md-h${Math.min(h[1].length, 3)}` }, inline(h[2]))); i++; continue; }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(l)) { out.push(el('hr')); i++; continue; }
    if (/^\s*\|/.test(l) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const cells = (row) => row.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
      const head = cells(l);
      const body = [];
      for (i += 2; i < lines.length && /^\s*\|/.test(lines[i]); i++) body.push(cells(lines[i]));
      const table = el('table', {},
        el('thead', {}, el('tr', {}, head.map((c) => el('th', { class: c.length <= 24 ? 'short' : null }, inline(c))))),
        el('tbody', {}, body.map((r) => el('tr', {}, r.map((c) => el('td', { class: c.length <= 24 ? 'short' : null }, inline(c)))))));
      resizable(table);
      out.push(el('div', { class: 'md-table' }, table));
      continue;
    }
    if (/^\s*>/.test(l)) {
      const q = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]); i++) q.push(lines[i].replace(/^\s*>\s?/, ''));
      out.push(el('blockquote', {}, md(q.join('\n'))));
      continue;
    }
    if (/^\s*([-*+]|\d+[.)])\s+/.test(l)) {
      const stack = [];
      let last = null;
      for (; i < lines.length; i++) {
        const x = lines[i];
        const m = x.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
        if (m) {
          const ind = m[1].replace(/\t/g, '    ').length;
          const ordered = /\d/.test(m[2]);
          while (stack.length && stack[stack.length - 1].ind > ind) stack.pop();
          if (!stack.length || ind > stack[stack.length - 1].ind) {
            const list = el(ordered ? 'ol' : 'ul');
            if (stack.length && last) last.append(list); else out.push(list);
            stack.push({ ind, list });
          }
          last = el('li', {}, inline(m[3]));
          stack[stack.length - 1].list.append(last);
        } else if (x.trim() && /^\s+\S/.test(x) && last) {
          last.append(' ', ...inline(x.trim()));
        } else if (!x.trim() && /^\s*([-*+]|\d+[.)])\s+|^\s{2,}\S/.test(lines[i + 1] || '')) {
          continue;
        } else break;
      }
      continue;
    }
    const para = [];
    for (; i < lines.length && lines[i].trim() && !(para.length && isBlockStart(lines[i])); i++) para.push(lines[i].trim());
    out.push(el('p', {}, inline(para.join(' '))));
  }
  return out;
}

/** a drag handle on the right edge of every header cell; the first drag fixes all the widths where they are */
function resizable(table) {
  const ths = [...table.tHead.rows[0].cells];
  ths.forEach((th, i) => {
    const grip = el('span', { class: 'col-grip' });
    grip.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!table.classList.contains('fixed')) {
        const w = ths.map((x) => x.offsetWidth);
        ths.forEach((x, j) => { x.style.width = `${w[j]}px`; });
        table.style.width = `${w.reduce((a, b) => a + b, 0)}px`;
        table.classList.add('fixed');
      }
      const x0 = e.clientX;
      const w0 = th.offsetWidth;
      const t0 = table.offsetWidth;
      grip.setPointerCapture(e.pointerId);
      const move = (ev) => {
        const w = Math.max(36, w0 + ev.clientX - x0);
        th.style.width = `${w}px`;
        table.style.width = `${t0 + w - w0}px`;
      };
      const up = () => {
        grip.removeEventListener('pointermove', move);
        grip.removeEventListener('pointerup', up);
      };
      grip.addEventListener('pointermove', move);
      grip.addEventListener('pointerup', up);
    });
    th.append(grip);
  });
}

const INLINE_RE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\[[^\]]+\]\([^)\s]+\))|(\*[^*\s][^*]*\*)/g;

function inline(text) {
  const out = [];
  let at = 0;
  for (const m of text.matchAll(INLINE_RE)) {
    if (m.index > at) out.push(...rich(text.slice(at, m.index)));
    const t = m[0];
    if (m[1]) { const pic = images(t.slice(1, -1)); out.push(pic && pic.length === 1 ? pic[0] : el('code', {}, t.slice(1, -1))); }
    else if (m[2]) out.push(el('b', {}, inline(t.slice(2, -2))));
    else if (m[4]) out.push(el('i', {}, inline(t.slice(1, -1))));
    else {
      const [, label, url] = t.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
      const pic = /^https?:/i.test(url) ? null : images(url);
      if (pic && pic.length === 1) {
        const prev = out[out.length - 1];
        if (typeof prev === 'string' && prev.endsWith('!')) out[out.length - 1] = prev.slice(0, -1);   // ![alt](x.png)
        out.push(pic[0]);
      } else if (/^https?:\/\//i.test(url)) out.push(el('a', { href: url, target: '_blank', rel: 'noopener noreferrer' }, label));
      else if (/\.md(#.*)?$|\/$/.test(url) && notesName(url)) {
        const name = notesName(url);
        out.push(el('button', { class: 'link', type: 'button', onclick: () => goNotes(name) }, label));
      } else out.push(...rich(label));
    }
    at = m.index + t.length;
  }
  if (at < text.length) out.push(...rich(text.slice(at)));
  return out;
}

function showView(v) {
  view = v;
  if (v !== 'board' && panel && panel.kind === 'line') panel = null;
  if (v === 'changelog') { loadChangelog(); loadRoadmap(); }
  setHash();
  render();
  window.scrollTo(0, 0);
}

function padStamp(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return '';
  return d.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
}

function safeLink(url) {
  return /^https?:\/\//i.test(url || '') ? el('a', { href: url, target: '_blank', rel: 'noopener noreferrer' }, url) : (url || null);
}

/** pasted items still waiting for a sort: queued ones, plus handled ones no group holds (not withdrawn) */
function unsortedPastes() {
  const inGroups = new Set(REQ ? REQ.groups.flatMap((g) => g.pasted || []) : []);
  return INBOX.pending.filter((a) => a.type === 'paste' && !inGroups.has(a.id));
}

function groupText(g) {
  const parts = [g.title, g.summary];
  for (const id of g.reports || []) { const r = REQ.reports[id]; if (r) parts.push(r.text, r.name); }
  for (const id of g.pasted || []) { const p = REQ.pastes[id]; if (p) parts.push(p.text); }
  return parts.join(' ').toLowerCase();
}

/** re-render a container without losing the field she is typing in */
function keepFocus(box, fn, anchorSel) {
  const a = document.activeElement;
  const key = a && box.contains(a) ? a.dataset.key : null;
  const caret = key && a.selectionStart != null ? [a.selectionStart, a.selectionEnd] : null;
  // T-3587: keep her place. The first element still on screen stays where it was, even if items came in above it
  const y = window.scrollY;
  let anchor = null;
  if (anchorSel) {
    const el0 = [...box.querySelectorAll(anchorSel)].find((e) => e.getBoundingClientRect().bottom > 0);
    if (el0) anchor = [el0.dataset.item, el0.getBoundingClientRect().top];
  }
  fn();
  const f = key ? box.querySelector(`[data-key="${CSS.escape(key)}"]`) : null;
  if (f) {
    f.focus({ preventScroll: true });
    if (caret) try { f.setSelectionRange(caret[0], caret[1]); } catch (e) { /* a select */ }
  }
  const now = anchor && box.querySelector(`[data-item="${CSS.escape(anchor[0])}"]`);
  window.scrollTo(0, now ? window.scrollY + now.getBoundingClientRect().top - anchor[1] : y);
}

function renderRequests() {
  keepFocus($('#requests'), drawRequests);
}

function drawRequests() {
  const box = $('#requests');
  const q = query();
  const kids = [pasteBox()];
  if (!REQ) {
    kids.push(el('p', { class: 'meta' }, 'No player requests yet'));
    box.replaceChildren(...kids);
    return;
  }
  const show = (g) => !q || groupText(g).includes(q);
  const open = REQ.groups.filter((g) => (g.state || 'open') === 'open' && show(g))
    .sort((a, b) => (b.count || 0) - (a.count || 0) || a.title.localeCompare(b.title));
  const decided = REQ.groups.filter((g) => (g.state || 'open') !== 'open' && show(g))
    .sort((a, b) => String(b.decided_at || '').localeCompare(String(a.decided_at || '')));

  const pastes = unsortedPastes().filter((p) => !q || p.text.toLowerCase().includes(q));
  const reports = (REQ.unsorted_reports || []).map((id) => REQ.reports[id]).filter((r) => r && (!q || r.text.toLowerCase().includes(q)));
  if (pastes.length || reports.length) {
    kids.push(el('h2', { class: 'section' }, 'Not sorted yet', el('span', { class: 'n' }, ` ${pastes.length + reports.length}`)));
    for (const p of pastes) {
      kids.push(el('div', { class: 'group unsorted' }, pasteItem(p),
        el('div', { class: 'rmeta' }, `Queued ${clock(p.at)} · `,
          el('button', { class: 'link', type: 'button', onclick: () => withdraw(p.id) }, 'Withdraw'))));
    }
    for (const r of reports) kids.push(el('div', { class: 'group unsorted' }, reportItem(r)));
  }
  kids.push(el('h2', { class: 'section' }, 'Open', el('span', { class: 'n' }, ` ${open.length}`)));
  for (const g of open) kids.push(groupCard(g, true));
  if (decided.length) {
    kids.push(el('h2', { class: 'section' }, 'Decided', el('span', { class: 'n' }, ` ${decided.length}`)));
    for (const g of decided) kids.push(groupCard(g, false));
  }
  box.replaceChildren(...kids);
}

function pasteBox() {
  const src = drafts['paste:source'] || 'Discord';
  return el('div', { class: 'paste' },
    el('h2', { class: 'section' }, 'Paste a report'),
    textBox('paste:text', 'A bug report or suggestion from Discord, Twitter or elsewhere', 3),
    el('div', { class: 'controls' },
      el('label', { class: 'field' }, el('span', {}, 'From'),
        (() => {
          const s = el('select', { class: 'small', 'data-key': 'paste:source', onchange: (e) => { drafts['paste:source'] = e.target.value; } },
            SOURCES.map((x) => el('option', { value: x }, x)));
          s.value = src;
          return s;
        })()),
      el('input', { class: 'input link-input', type: 'text', placeholder: 'Link (optional)', 'data-key': 'paste:link',
        value: drafts['paste:link'] || '', oninput: (e) => { drafts['paste:link'] = e.target.value; } }),
      el('button', { class: 'btn', type: 'button', disabled: !INBOX.ok, onclick: async () => {
        const link = (drafts['paste:link'] || '').trim();
        const action = { type: 'paste', source: drafts['paste:source'] || 'Discord' };
        if (link) action.link = link;
        await submit('paste:text', action);
        if (!drafts['paste:text']) delete drafts['paste:link'];
        render();
      } }, 'Add report')));
}

function reportItem(r) {
  const meta = [KIND[r.kind] || r.kind, stamp(r.time), r.name, r.lang].filter(Boolean).join(' · ');
  return el('div', { class: 'report' },
    el('div', { class: 'rmeta' }, meta, r.pictures ? ` · ${r.pictures} ${r.pictures === 1 ? 'picture' : 'pictures'}` : ''),
    el('div', { class: 'rtext' }, rich(r.text || '')));
}

function pasteItem(p) {
  const link = safeLink(p.link);
  return el('div', { class: 'report' },
    el('div', { class: 'rmeta' }, [p.source || 'Other', stamp(p.at)].join(' · '), link ? ' · ' : null, link),
    el('div', { class: 'rtext' }, rich(p.text || '')));
}

function matchInfo(g) {
  const m = g.match;
  if (!m || (!m.line && !m.task)) return el('span', {}, 'No line yet');
  const parts = [];
  if (m.line) parts.push(el('button', { class: 'link', type: 'button', onclick: () => openPanel({ kind: 'line', slug: m.line }) }, lineName(m.line)));
  if (m.task) {
    if (parts.length) parts.push(', ');
    parts.push(taskLink(m.task), DATA.tasks[m.task] ? '' : ' (closed)');
  }
  return el('span', {}, 'On the board: ', parts);
}

function groupCard(g, isOpen) {
  const pend = INBOX.pending.filter((a) => a.group === g.id);
  const shown = expanded.has(g.id);
  const n = g.count || (g.reports || []).length + (g.pasted || []).length;
  const reach = g.reach ? `${g.reach[0].toUpperCase()}${g.reach.slice(1)} players${g.reach_why ? `: ${g.reach_why}` : ''}` : '';
  const toggle = () => {
    if (shown) expanded.delete(g.id); else { expanded.add(g.id); lastGroup = g.id; }
    if (shown && lastGroup === g.id) lastGroup = null;
    setHash();
    renderRequests();
  };
  return el('div', { class: 'group' + (isOpen ? '' : ' decided'), 'data-group': g.id },
    el('div', { class: 'ghead' },
      el('button', { class: 'gtitle', type: 'button', onclick: toggle }, g.title),
      el('span', { class: 'gcount' }, isOpen ? `${n} ${n === 1 ? 'report' : 'reports'}`
        : `${STATE[g.state] || g.state}${g.decided_at ? ` ${day(String(g.decided_at).slice(0, 10))}` : ''}, ${n} ${n === 1 ? 'report' : 'reports'}`)),
    el('div', { class: 'gmeta' }, reach || null, reach ? ' · ' : null, matchInfo(g)),
    g.summary ? el('p', { class: 'gsummary' }, rich(g.summary)) : null,
    pend.length ? queuedList(pend) : null,
    el('div', { class: 'controls small' },
      isOpen && INBOX.ok ? [
        formButton(g, 'accept', 'Accept'), formButton(g, 'merge', 'Merge into'),
        formButton(g, 'dismiss', 'Dismiss'), formButton(g, 'comment', 'Comment'),
      ] : null,
      el('button', { class: 'link', type: 'button', onclick: toggle }, shown ? 'Hide reports' : 'Show reports')),
    groupForm && groupForm.gid === g.id ? groupFormBox(g) : null,
    shown ? el('div', { class: 'reports' },
      (g.reports || []).map((id) => REQ.reports[id]).filter(Boolean).map(reportItem),
      (g.pasted || []).map((id) => REQ.pastes[id] || INBOX.pending.find((a) => a.id === id)).filter(Boolean).map(pasteItem)) : null);
}

function formButton(g, kind, label) {
  const on = groupForm && groupForm.gid === g.id && groupForm.kind === kind;
  return el('button', { class: 'btn' + (on ? ' on' : ''), type: 'button', onclick: () => {
    groupForm = on ? null : { gid: g.id, kind };
    renderRequests();
  } }, label);
}

function groupFormBox(g) {
  const k = `group:${g.id}:${groupForm.kind}`;
  const m = g.match || {};
  const close = () => { groupForm = null; renderRequests(); };
  const why = el('input', { class: 'input', type: 'text', placeholder: 'Comment (optional)', 'data-key': `${k}:why`,
    value: drafts[`${k}:why`] || '', oninput: (e) => { drafts[`${k}:why`] = e.target.value; } });
  const send = async (action) => {
    const comment = textOf(`${k}:why`);
    if (await queue({ ...action, group: g.id, ...(comment ? { comment } : {}) })) {
      for (const key of Object.keys(drafts)) if (key.startsWith(k)) delete drafts[key];
      groupForm = null;
      render();
    }
  };
  const cancel = el('button', { class: 'btn plain', type: 'button', onclick: close }, 'Cancel');
  const pick = (key, fallback) => (drafts[`${k}:${key}`] != null ? drafts[`${k}:${key}`] : fallback);
  const sel = (key, options, value, rerender) => {
    const s = el('select', { class: 'small', 'data-key': `${k}:${key}`, onchange: (e) => { drafts[`${k}:${key}`] = e.target.value; if (rerender) renderRequests(); } },
      options.map((o) => (o.group ? el('optgroup', { label: o.group }, o.options.map((x) => el('option', { value: x.value }, x.label)))
        : el('option', { value: o.value }, o.label))));
    s.value = value;
    return s;
  };

  if (groupForm.kind === 'accept') {
    const matchLine = VIEW.find((l) => l.slug === m.line);
    const area = pick('area', matchLine ? matchLine.area : DATA.areas[0].slug);
    const linesHere = VIEW.filter((l) => l.area === area && l.slug !== 'not-sorted');
    const lineVal = pick('line', matchLine && matchLine.area === area ? matchLine.slug : 'new');
    const pri = pick('pri', 'P3');
    return el('div', { class: 'gform' },
      el('div', { class: 'controls' },
        el('label', { class: 'field' }, el('span', {}, 'Area'), sel('area', areaOptions(), area, true)),
        el('label', { class: 'field' }, el('span', {}, 'Line'),
          sel('line', [{ value: 'new', label: 'New line' }].concat(linesHere.map((l) => ({ value: l.slug, label: l.name }))), lineVal)),
        el('label', { class: 'field' }, el('span', {}, 'Priority'), tintPri(sel('pri', PRIS.map((p) => ({ value: p, label: p })), pri)))),
      el('div', { class: 'confirm' }, why,
        el('button', { class: 'btn', type: 'button', onclick: () => send({ type: 'accept', area: pick('area', area), line: pick('line', lineVal), priority: pick('pri', pri) }) }, 'Queue accept'),
        cancel));
  }
  if (groupForm.kind === 'merge') {
    const lineVal = pick('line', m.line || VIEW[0].slug);
    const task = el('input', { class: 'input task-input', type: 'text', placeholder: 'optional', 'data-key': `${k}:task`,
      value: pick('task', m.task || ''), oninput: (e) => { drafts[`${k}:task`] = e.target.value; } });
    return el('div', { class: 'gform' },
      el('div', { class: 'controls' },
        el('label', { class: 'field' }, el('span', {}, 'Line'), sel('line', lineOptions(), lineVal)),
        el('label', { class: 'field' }, el('span', {}, 'Task'), task)),
      el('div', { class: 'confirm' }, why,
        el('button', { class: 'btn', type: 'button', onclick: () => {
          const t = (pick('task', m.task || '') || '').trim().toUpperCase();
          send({ type: 'merge', line: pick('line', lineVal), ...(t ? { task: t } : {}) });
        } }, 'Queue merge'),
        cancel));
  }
  if (groupForm.kind === 'dismiss') {
    return el('div', { class: 'gform' }, el('div', { class: 'confirm' }, why,
      el('button', { class: 'btn', type: 'button', onclick: () => send({ type: 'dismiss' }) }, 'Queue dismiss'), cancel));
  }
  return el('div', { class: 'gform compose' },
    textBox(`${k}:text`, 'Comment on this request'),
    el('div', { class: 'controls' },
      el('button', { class: 'btn', type: 'button', onclick: async () => {
        const text = textOf(`${k}:text`);
        if (!text) return;
        await send({ type: 'comment', text });
      } }, 'Add comment'),
      cancel));
}

// ---------- changelog (T-3424) ----------

const CL_IDLE_MS = 1200;

async function loadChangelog() {
  try {
    const r = await fetch('_changelog', { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    CL.data = await r.json();
    CL.ok = true;
  } catch (e) {
    CL.ok = false;
  }
  if (view === 'changelog') renderChangelog();
}

function clShown(it) { return it.edited || it.text; }

// T-3466: what an item is about, in the game's order (app/src/ui/updates.ts AREAS); the names are the board's only
const CL_AREAS = [['track', 'Track'], ['demand', 'Demand'], ['scheduling', 'Scheduling'], ['fares', 'Fares and money'],
  ['schematic', 'Map and schematic'], ['community', 'Community and accounts'], ['performance', 'Performance'],
  ['other', 'Other (a plain dot)'], ['bug', 'Bug fix of no area']];
const clRank = (it) => { const a = CL_AREAS.findIndex(([k]) => k === it.area); return (a < 0 ? CL_AREAS.length : a) * 2 + (it.fix ? 1 : 0); };

/** T-3591: the board's three sections, top to bottom: highlights, the rest of what is shown, hidden */
const clClass = (it) => (it.hidden ? 2 : it.highlight ? 0 : 1);

/** T-3590: her order is the file's (new items first); highlights first (T-3591), then the other shown items, then hidden
 *  ones; a group's items under it. A hidden group goes down with its items. Within a group, hidden items come after the
 *  shown ones. */
function clOrdered(all) {
  const groups = new Set(all.filter((i) => i.group).map((i) => i.id));
  const top = all.filter((it) => !groups.has(it.parent));
  const kidsOf = (g) => all.filter((c) => c.parent === g.id);
  const stable = (arr) => arr.map((it, n) => [it, n]).sort((a, b) => clClass(a[0]) - clClass(b[0]) || a[1] - b[1]).map(([it]) => it);
  return stable(top).flatMap((it) => (it.group ? [it, ...stable(kidsOf(it)).map((c) => ({ ...c, child: true }))] : [it]));
}

/** the same move changelog.move_item makes: it just before `before` (an id), or last, under the group `under` (an id or
 *  null; undefined keeps the parent) */
function clMoveLocal(id, before, under) {
  const items = CL.data.items;
  const it = items.find((x) => x.id === id);
  if (!it) return;
  if (under !== undefined && (under || undefined) !== it.parent) {
    if (under) {
      it.parent = under;
      delete it.highlight;
      if (items.find((x) => x.id === under).hidden) it.hidden = true;
    } else delete it.parent;
  }
  items.splice(items.indexOf(it), 1);
  items.splice(before == null ? items.length : items.findIndex((x) => x.id === before), 0, it);
}

const clDrag = { id: null, over: null, after: false };

function clDropClear() {
  for (const e of document.querySelectorAll('#changelog .cl-drop-before, #changelog .cl-drop-after, #changelog .cl-drop-into, #changelog .cl-dragging'))
    e.classList.remove('cl-drop-before', 'cl-drop-after', 'cl-drop-into', 'cl-dragging');
}

/** where a drag of the item `id` would land, from the row under the pointer and where on it (`frac`, 0 top to 1
 *  bottom): {parent, before, row, mode} or null. parent is the group it ends up under (null: top level); mode is
 *  'before' or 'after' the row (a line), or 'into' (the group row is lit). On a group row the top quarter drops above
 *  the group, the rest puts the item first in the group. A group never goes inside a group. An item dropped in a
 *  group joins that group's section (hidden when the group is); within a section the order is the file's. */
function clTarget(id, rowId, frac) {
  const all = CL.data.items;
  const it = all.find((x) => x.id === id);
  const row = all.find((x) => x.id === rowId);
  if (!it || !row || row.id === id) return null;
  const isGroup = (x) => !!x.group;
  const grp = (x) => (x.parent && all.some((g) => g.group && g.id === x.parent) ? x.parent : null);
  let parent;
  let mode;
  let at = row;                                   // the row the drop is placed against
  if (isGroup(row) && !isGroup(it) && frac > 0.25) { parent = row.id; mode = 'into'; }
  else if (grp(row)) {
    if (isGroup(it)) { at = all.find((x) => x.id === grp(row)); parent = null; mode = 'after'; if (at.id === id) return null; }
    else { parent = grp(row); mode = frac > 0.5 ? 'after' : 'before'; }
  } else { parent = null; mode = frac > 0.5 ? 'after' : 'before'; }
  const gHidden = parent ? !!all.find((x) => x.id === parent).hidden : false;
  const cls = it.hidden || gHidden ? 2 : parent ? 1 : clClass(it);
  const order = clOrdered(all).filter((x) => grp(x) === parent && x.id !== id);
  let before;
  if (mode === 'into') {
    const first = order.find((x) => clClass(x) === cls);
    before = first ? first.id : null;
    if (!first && order.some((x) => clClass(x) > cls)) before = order.find((x) => clClass(x) > cls).id;
  } else {
    const k = order.findIndex((x) => x.id === at.id);
    if (k < 0 || clClass(order[k]) !== cls) return null;        // highlights, the rest and hidden items keep apart
    const next = order[k + (mode === 'after' ? 1 : 0)];
    before = next && clClass(next) === cls ? next.id : null;
    if (before === null && order.slice(k + 1).some((x) => clClass(x) === cls)) return null;
  }
  return { parent, before, row: at.id, mode };
}

async function clMove(id, before, under) {
  clMoveLocal(id, before, under);
  CL.status[id] = 'Saving';
  renderChangelog();
  try {
    const r = await clPost({ op: 'move', id, before, under });
    if (!r.ok) throw new Error(r.status);
    CL.status[id] = 'Saved';
  } catch (e) {
    CL.status[id] = 'Not saved: could not reorder';
  }
  await loadChangelog();
}

function clAreaIcon(area) {
  return el('span', { class: `cl-area cl-a-${area || 'none'}` }, icon(`a-${area || 'other'}`));
}

async function clPost(body) {
  const r = await fetch('_changelog', {
    method: 'POST', headers: { 'X-Maps-Run': '1', 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!r.ok && r.status !== 409) throw new Error(r.status);
  return r;
}

async function clAddGroup() {
  try {
    const r = await clPost({ op: 'group', text: 'new group', area: 'bug', fix: true });
    const it = await r.json();
    CL.data.items.unshift(it);
    CL.menu = it.id;
  } catch (e) {
    CL.ok = false;
  }
  renderChangelog();
  const a = document.querySelector(`#changelog [data-item="${CL.menu}"] textarea.cl-text`);
  if (a) { a.focus(); a.select(); }
}

async function clDelete(it) {
  try {
    const r = await clPost({ op: 'delete', id: it.id, rev: it.rev });
    if (r.status === 409) clSetItem(await r.json());
    else CL.data.items = CL.data.items.filter((x) => x.id !== it.id);
  } catch (e) {
    CL.status[it.id] = 'Not deleted: server not reachable';
  }
  CL.menu = null;
  renderChangelog();
}

/** the area menu under an item's icon: its area, bug fix or not, and the group it sits under */
function clMenu(it) {
  const groups = CL.data.items.filter((g) => g.group && g.id !== it.id);
  const kids = CL.data.items.some((c) => c.parent === it.id);
  return el('div', { class: 'cl-menu', onclick: (e) => e.stopPropagation() },
    ...CL_AREAS.map(([k, name]) => el('button', { class: `cl-pick${it.area === k ? ' on' : ''}`, type: 'button',
      onclick: () => { CL.menu = null; clSave(it.id, { area: k }); } }, clAreaIcon(k), name)),
    el('label', { class: 'cl-check' }, el('input', { type: 'checkbox', checked: !!it.fix,
      onchange: (e) => clSave(it.id, { fix: e.target.checked }) }), 'Bug fix (lighter text)'),
    it.group ? (kids ? null : el('button', { class: 'btn plain cl-del', type: 'button', onclick: () => clDelete(it) }, 'Delete this group'))
      : el('label', { class: 'cl-under' }, 'Under ', el('select', { class: 'small',
        onchange: (e) => { CL.menu = null; clSave(it.id, { parent: e.target.value || null }); } },
        el('option', { value: '' }, 'no group'),
        ...groups.map((g) => el('option', { value: g.id, selected: it.parent === g.id }, clShown(g))))));
}
function clDraft(it) { const d = drafts[`cl:${it.id}`]; return d === undefined ? clShown(it) : d; }

function renderChangelog() {
  const cols = document.querySelector('#changelog .rm-cols');
  const left = cols ? cols.scrollLeft : 0;
  // T-3587: the poll re-renders every 15 s; nothing changed, nothing to redraw
  const sig = JSON.stringify([CL.data, CL.ok, CL.menu, CL.conflict, CL.status, RM.doc, RM.status, RM.ok, query(), drafts]);
  if (sig === CL.sig && $('#changelog').childElementCount) return;
  CL.sig = sig;
  keepFocus($('#changelog'), () => {
    drawChangelog();
    for (const a of document.querySelectorAll('#changelog textarea.cl-text, #changelog .rm textarea')) clGrow(a);
    const c2 = document.querySelector('#changelog .rm-cols');
    if (c2) c2.scrollLeft = left;
  }, '.cl-item[data-item]');
}

function clGrow(a) {
  a.style.height = 'auto';
  a.style.height = `${a.scrollHeight + 2}px`;
}

function clSetItem(item) {
  const i = CL.data.items.findIndex((x) => x.id === item.id);
  if (i >= 0) CL.data.items[i] = item;
}

async function clSave(id, fields) {
  const it = CL.data && CL.data.items.find((x) => x.id === id);
  if (!it) return;
  const key = `cl:${id}`;
  const sent = drafts[key];
  const body = { id, rev: it.rev, ...fields };
  if (!fields) { delete body.undefined; body.edited = sent; }
  CL.status[id] = 'Saving';
  clStatus(id);
  try {
    const r = await fetch('_changelog', {
      method: 'POST', headers: { 'X-Maps-Run': '1', 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    if (r.status === 409) {
      const cur = await r.json();
      clSetItem(cur);
      CL.conflict[id] = fields ? null : sent;
      delete drafts[key];
      CL.status[id] = '';
    } else if (!r.ok) throw new Error(r.status);
    else {
      const cur = await r.json();
      clSetItem(cur);
      if (drafts[key] === sent) delete drafts[key];
      CL.status[id] = drafts[key] === undefined ? 'Saved' : 'Saving';
    }
  } catch (e) {
    CL.status[id] = `Not saved: ${e.message === '503' ? 'busy, try again' : 'server not reachable'}`;
  }
  if (view === 'changelog') renderChangelog();
}

function clStatus(id) {
  const s = document.querySelector(`#changelog [data-item="${id}"] .cl-state`);
  if (s) s.textContent = CL.status[id] || '';
}

/** T-3585: *word* and **word** are bold in the game; a lone asterisk stays as typed (same rule as app/src/ui/Changelog.tsx) */
function clBold(s) {
  const out = [];
  let at = 0;
  for (const m of String(s).matchAll(/\*\*(?=\S)([^*\n]*?\S)\*\*|\*(?=\S)([^*\n]*?\S)\*/g)) {
    if (m.index > at) out.push(s.slice(at, m.index));
    out.push(el('b', {}, m[1] != null ? m[1] : m[2]));
    at = m.index + m[0].length;
  }
  if (at < s.length) out.push(s.slice(at));
  return out;
}

function clItem(it) {
  const id = it.id;
  const bv = el('div', { class: 'cl-view', 'aria-hidden': 'true' }, ...clBold(clDraft(it)));
  const area = el('textarea', {
    class: 'input cl-text', rows: 1, spellcheck: 'true', 'data-key': `cl:${id}`,
    oninput: (e) => {
      drafts[`cl:${id}`] = e.target.value;
      bv.replaceChildren(...clBold(e.target.value));
      clGrow(e.target);
      CL.status[id] = 'Saving';
      clStatus(id);
      clearTimeout(CL.timers[id]);
      CL.timers[id] = setTimeout(() => clSave(id), CL_IDLE_MS);
    },
    onblur: () => {
      if (drafts[`cl:${id}`] === undefined || !CL.timers[id]) return;
      clearTimeout(CL.timers[id]);
      CL.timers[id] = null;
      clSave(id);
    },
  });
  area.value = clDraft(it);
  const mine = CL.conflict[id];
  const when = new Date(it.at).toLocaleDateString([], { day: 'numeric', month: 'short' });
  const cls = `cl-item${it.highlight ? ' is-highlight' : ''}${it.hidden ? ' is-hidden' : ''}${it.fix ? ' is-fix' : ''}${it.group ? ' is-group' : ''}${it.child ? ' is-child' : ''}`;
  return el('div', { class: cls, 'data-item': id,
    ondragstart: (e) => {
      if (!e.currentTarget.draggable || e.target !== e.currentTarget) { e.preventDefault(); return; }
      clDrag.id = id;
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', id);
      e.currentTarget.classList.add('cl-dragging');
    },
    ondragover: (e) => {
      if (!clDrag.id) return;
      const b = e.currentTarget.getBoundingClientRect();
      const t = clTarget(clDrag.id, id, (e.clientY - b.top) / b.height);
      clDropClear();
      const src = document.querySelector(`#changelog [data-item="${clDrag.id}"]`);
      if (src) src.classList.add('cl-dragging');
      if (!t) return;
      e.preventDefault();
      const row = document.querySelector(`#changelog [data-item="${t.row}"]`);
      if (row) row.classList.add(t.mode === 'into' ? 'cl-drop-into' : t.mode === 'after' ? 'cl-drop-after' : 'cl-drop-before');
    },
    ondrop: (e) => {
      if (!clDrag.id) return;
      e.preventDefault();
      const b = e.currentTarget.getBoundingClientRect();
      const t = clTarget(clDrag.id, id, (e.clientY - b.top) / b.height);
      const moved = clDrag.id;
      clDrag.id = null;
      clDropClear();
      if (t) clMove(moved, t.before, t.parent);
    },
    ondragend: (e) => { clDrag.id = null; e.currentTarget.draggable = false; clDropClear(); } },
    el('div', { class: 'cl-row' },
      el('span', { class: 'cl-grip', 'aria-label': 'Drag to reorder',
        onpointerdown: (e) => { e.currentTarget.closest('.cl-item').draggable = true; },
        onpointerup: (e) => { e.currentTarget.closest('.cl-item').draggable = false; } }, icon('grip')),
      el('span', { class: 'cl-area-cell' },
        el('button', { class: 'icon-btn cl-area-btn', type: 'button', 'aria-expanded': String(CL.menu === id),
          title: `${(CL_AREAS.find(([k]) => k === it.area) || [0, 'No area yet'])[1]}${it.fix ? ', bug fix' : ''}. Click to change`,
          onclick: (e) => { e.stopPropagation(); CL.menu = CL.menu === id ? null : id; renderChangelog(); } }, clAreaIcon(it.area)),
        CL.menu === id ? clMenu(it) : null),
      it.group ? el('span', { class: 'cl-task meta' }, 'group') : it.task ? el('span', { class: 'cl-task' }, rich(it.task)) : el('span', { class: 'cl-task meta' }, 'no task'),
      el('span', { class: 'cl-wrap' }, area, bv),
      el('span', { class: 'cl-by' }, `added by ${it.by || 'an agent'}, ${when}`, it.edited ? ', reworded' : ''),
      el('span', { class: 'cl-state' }, CL.status[id] || ''),
      it.edited ? el('button', { class: 'icon-btn', type: 'button', title: `Use the agent's wording: ${it.text}`, 'aria-label': 'Use the agent\'s wording',
        onclick: () => { delete drafts[`cl:${id}`]; clSave(id, { edited: null }); } }, icon('undo')) : el('span', { class: 'icon-btn ghost' }),
      it.child ? el('span', { class: 'icon-btn ghost' })
        : el('button', { class: 'icon-btn star', type: 'button', 'aria-pressed': String(!!it.highlight),
          'aria-label': it.highlight ? 'Remove from the highlights' : 'Make a highlight',
          onclick: () => clSave(id, { highlight: !it.highlight }) }, icon('star')),
      el('button', { class: 'icon-btn eye', type: 'button', 'aria-pressed': String(!it.hidden),
        title: it.hidden ? 'Hidden from the update. Click to show it' : 'In the update. Click to hide it',
        'aria-label': it.hidden ? 'Show in the update' : 'Hide from the update',
        onclick: () => clSave(id, { hidden: !it.hidden }) }, icon(it.hidden ? 'eye-off' : 'eye'))),
    mine ? el('div', { class: 'cl-conflict' },
      el('p', { class: 'meta' }, 'This item changed while you were editing, so it shows the new wording. What you typed:'),
      el('textarea', { class: 'input', rows: 2, readonly: true }, mine),
      el('div', { class: 'controls' },
        el('button', { class: 'btn', type: 'button', onclick: () => { drafts[`cl:${id}`] = mine; CL.conflict[id] = null; clSave(id); } }, 'Use my wording'),
        el('button', { class: 'btn plain', type: 'button', onclick: () => { CL.conflict[id] = null; renderChangelog(); } }, 'Discard'))) : null);
}

function drawChangelog() {
  const box = $('#changelog');
  if (!CL.data) {
    box.replaceChildren(el('p', { class: 'meta' }, CL.ok ? 'Loading' : 'The changelog can\'t be read until serve.py is restarted'));
    return;
  }
  const q = query();
  const all = CL.data.items;
  const items = clOrdered(all).filter((i) => !q || (i.text + ' ' + (i.edited || '') + ' ' + (i.task || '')).toLowerCase().includes(q));
  const hid = all.filter((i) => i.hidden).length;
  const kids = [
    drawRoadmap(),
    el('div', { class: 'cl-head' },
      el('h2', { class: 'section' }, 'Next update', el('span', { class: 'n' }, ` ${all.length - hid} shown, ${hid} hidden`)),
      el('button', { class: 'btn', type: 'button', onclick: clAddGroup }, 'Add group')),
    el('p', { class: 'meta' }, CL.data.line ? `Start-menu line: ${CL.data.line}` : 'Start-menu line: not written yet'),
  ];
  if (!CL.ok) kids.push(el('p', { class: 'meta' }, 'Not saved: server not reachable'));
  if (!all.length) kids.push(el('p', { class: 'meta' }, 'Nothing here yet'));
  // T-3591: with any highlights, two small labels split the shown items
  const anyHl = all.some((i) => i.highlight && !i.hidden && !i.parent);
  let last = -1;
  for (const it of items) {
    const c = clClass(it);
    if (!it.child && c !== last && anyHl && c < 2) kids.push(el('h3', { class: 'cl-section' }, c === 0 ? 'Highlights' : 'Everything else'));
    if (!it.child) last = c;
    kids.push(clItem(it));
  }
  box.replaceChildren(...kids);
}

// ---------- roadmap (T-3467) ----------
// Planned versions (0.4, 0.5, ...) as columns of short items, on the changelog tab only. Saved whole to
// trainworld/changelog/roadmap.json through serve.py's /_roadmap (the pad's pattern: a PUT names the version it
// read, and a newer file answers 409, which reloads it). Not shown in the game.

const RM = { doc: null, version: 0, ok: true, status: '', timer: null, drag: null };
const RM_IDLE_MS = 1200;
const rmId = (p) => p + Math.random().toString(36).slice(2, 9);

async function loadRoadmap() {
  try {
    const r = await fetch('_roadmap', { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    const s = await r.json();
    RM.version = s.version;
    RM.doc = s.text ? JSON.parse(s.text) : null;
    RM.ok = true;
  } catch (e) {
    RM.ok = false;
  }
  if (!RM.doc || !Array.isArray(RM.doc.versions)) {
    RM.doc = { versions: [{ id: rmId('v'), name: '0.4', items: [] }, { id: rmId('v'), name: '0.5', items: [] }] };
  }
  if (view === 'changelog') renderChangelog();
}

/** save the whole roadmap now, or soon (after she stops typing) */
function rmSave(soon) {
  clearTimeout(RM.timer);
  RM.status = 'Saving';
  rmShowStatus();
  RM.timer = setTimeout(rmPut, soon ? RM_IDLE_MS : 0);
}

async function rmPut() {
  RM.timer = null;
  try {
    const r = await fetch('_roadmap', {
      method: 'PUT', headers: { 'X-Maps-Run': '1', 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: JSON.stringify(RM.doc, null, 2) + String.fromCharCode(10), version: RM.version }),
    });
    if (r.status === 409) {
      const s = await r.json();
      RM.version = s.version;
      RM.doc = JSON.parse(s.text);
      RM.status = 'Changed in another tab, so this shows that one; redo the last change';
      for (const k of Object.keys(drafts)) if (k.startsWith('rm:')) delete drafts[k];
      if (view === 'changelog') renderChangelog();
      return;
    }
    if (!r.ok) throw new Error(r.status);
    RM.version = (await r.json()).version;
    RM.status = 'Saved';
  } catch (e) {
    RM.status = 'Not saved: server not reachable';
  }
  rmShowStatus();
}

function rmShowStatus() {
  const s = document.querySelector('#changelog .rm-state');
  if (s) s.textContent = RM.status;
}

/** where an item is: [version, index] */
function rmFind(id) {
  for (const v of RM.doc.versions) {
    const i = v.items.findIndex((x) => x.id === id);
    if (i >= 0) return [v, i];
  }
  return [null, -1];
}

/** the next version's name: one more in the last part of the last one ("0.5" after "0.4") */
function rmNextName() {
  const last = RM.doc.versions[RM.doc.versions.length - 1];
  const m = last && /^(.*?)(\d+)$/.exec(last.name.trim());
  return m ? m[1] + (Number(m[2]) + 1) : '0.4';
}

function rmText(key, value, onSave, cls) {
  const t = el('textarea', {
    class: `input ${cls}`, rows: 1, spellcheck: 'true', 'data-key': key,
    oninput: (e) => { drafts[key] = e.target.value; clGrow(e.target); onSave(e.target.value); rmSave(true); },
    onblur: () => { delete drafts[key]; if (RM.timer) { clearTimeout(RM.timer); rmPut(); } },
  });
  t.value = drafts[key] !== undefined ? drafts[key] : value;
  return t;
}

/** the index the dragged item would drop at in this item list, from the pointer's height */
function rmDropAt(list, y) {
  const cards = [...list.querySelectorAll(':scope > .rm-card:not(.dragging)')];
  const i = cards.findIndex((c) => { const b = c.getBoundingClientRect(); return y < b.top + b.height / 2; });
  return [cards, i < 0 ? cards.length : i];
}

function rmClearMarks() {
  for (const m of document.querySelectorAll('#changelog .rm-drop-before, #changelog .rm-drop-end')) m.classList.remove('rm-drop-before', 'rm-drop-end');
}

function rmCard(v, it) {
  const card = el('div', { class: 'rm-card', 'data-rm': it.id },
    el('span', { class: 'rm-grip', draggable: 'true', title: 'Drag to move',
      ondragstart: (e) => {
        RM.drag = it.id;
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', it.id);
        e.dataTransfer.setDragImage(card, 12, 12);
        requestAnimationFrame(() => card.classList.add('dragging'));
      },
      ondragend: () => { RM.drag = null; card.classList.remove('dragging'); rmClearMarks(); } }, '⠿'),
    rmText(`rm:i:${it.id}`, it.text, (val) => { it.text = val; }, 'rm-text'),
    el('button', { class: 'icon-btn rm-x', type: 'button', title: 'Delete this item', 'aria-label': 'Delete this item',
      onclick: () => { v.items = v.items.filter((x) => x !== it); rmSave(); renderChangelog(); } }, '×'));
  return card;
}

function rmColumn(v) {
  const list = el('div', { class: 'rm-items',
    ondragover: (e) => {
      if (!RM.drag) return;
      e.preventDefault();
      rmClearMarks();
      const [cards, i] = rmDropAt(list, e.clientY);
      if (cards[i]) cards[i].classList.add('rm-drop-before'); else list.classList.add('rm-drop-end');
    },
    ondragleave: (e) => { if (!list.contains(e.relatedTarget)) rmClearMarks(); },
    ondrop: (e) => {
      e.preventDefault();
      const id = RM.drag;
      rmClearMarks();
      if (!id) return;
      const [, i] = rmDropAt(list, e.clientY);
      const [from, at] = rmFind(id);
      if (!from) return;
      const [it] = from.items.splice(at, 1);
      v.items.splice(i, 0, it);
      RM.drag = null;
      rmSave();
      renderChangelog();
    } },
  ...v.items.map((it) => rmCard(v, it)));
  const key = `rm:add:${v.id}`;
  const add = el('input', { class: 'input rm-add', type: 'text', placeholder: 'Add an item', 'data-key': key,
    oninput: (e) => { drafts[key] = e.target.value; },
    onkeydown: (e) => {
      if (e.key !== 'Enter' || !e.target.value.trim()) return;
      v.items.push({ id: rmId('r'), text: e.target.value.trim() });
      delete drafts[key];
      rmSave();
      renderChangelog();
    } });
  add.value = drafts[key] || '';
  return el('div', { class: 'rm-col', 'data-version': v.id },
    el('div', { class: 'rm-col-head' },
      rmText(`rm:v:${v.id}`, v.name, (val) => { v.name = val; }, 'rm-name'),
      el('button', { class: 'icon-btn rm-x', type: 'button', title: 'Delete this version', 'aria-label': 'Delete this version',
        onclick: () => {
          if (v.items.length && !confirm(`Delete ${v.name} and its ${v.items.length} items?`)) return;
          RM.doc.versions = RM.doc.versions.filter((x) => x !== v);
          rmSave();
          renderChangelog();
        } }, '×')),
    list, add);
}

function drawRoadmap() {
  if (!RM.doc) return el('p', { class: 'meta' }, 'Loading the roadmap');
  return el('div', { class: 'rm' },
    el('div', { class: 'cl-head' },
      el('h2', { class: 'section' }, 'Roadmap', el('span', { class: 'n' }, ' not shown in the game')),
      el('span', { class: 'rm-state meta' }, RM.ok ? RM.status : 'The roadmap can\'t be read until serve.py is restarted'),
      el('button', { class: 'btn', type: 'button', onclick: () => {
        RM.doc.versions.push({ id: rmId('v'), name: rmNextName(), items: [] });
        rmSave();
        renderChangelog();
        const cols = document.querySelector('#changelog .rm-cols');
        if (cols) cols.scrollLeft = cols.scrollWidth;
      } }, 'Add version')),
    el('div', { class: 'rm-cols' }, ...RM.doc.versions.map(rmColumn)));
}

// ---------- wiring ----------

try {
  const w = parseInt(localStorage.getItem('twboard.panelw'), 10);
  if (w >= 360) document.documentElement.style.setProperty('--panel-w', `${Math.min(w, window.innerWidth)}px`);
} catch (e) { /* storage blocked: the default width */ }
$('#grip').addEventListener('pointerdown', (e) => {
  e.preventDefault();
  const grip = $('#grip');
  grip.setPointerCapture(e.pointerId);
  const move = (ev) => {
    const w = Math.round(Math.max(360, Math.min(window.innerWidth - ev.clientX, window.innerWidth * 0.92)));
    document.documentElement.style.setProperty('--panel-w', `${w}px`);
  };
  const up = () => {
    grip.removeEventListener('pointermove', move);
    grip.removeEventListener('pointerup', up);
    const w = parseInt(document.documentElement.style.getPropertyValue('--panel-w'), 10);
    try { if (w) localStorage.setItem('twboard.panelw', String(w)); } catch (err) { /* not remembered */ }
  };
  grip.addEventListener('pointermove', move);
  grip.addEventListener('pointerup', up);
});
$('#search').addEventListener('input', () => render());
$('#v-board').addEventListener('click', () => showView('board'));
$('#v-requests').addEventListener('click', () => showView('requests'));
$('#v-changelog').addEventListener('click', () => showView('changelog'));
document.addEventListener('click', () => { if (CL.menu) { CL.menu = null; if (view === 'changelog') renderChangelog(); } });
$('#update').addEventListener('change', () => DATA && renderGrid());
$('#p4').addEventListener('click', () => {
  SHOWP4 = !SHOWP4;
  try { localStorage.setItem('twb-showp4', SHOWP4 ? '1' : '0'); } catch (e) { /* not remembered */ }
  $('#p4').setAttribute('aria-pressed', String(SHOWP4));
  render();
});
$('#p4').setAttribute('aria-pressed', String(SHOWP4));
$('#refresh').addEventListener('click', rebuild);
$('#pad').addEventListener('click', () => openPanel(panel && panel.kind === 'pad' ? null : { kind: 'pad' }));
$('#ack').addEventListener('click', () => openPanel({ kind: 'ack' }));
$('#queued').addEventListener('click', () => openPanel({ kind: 'queue' }));
$('#send').addEventListener('click', () => post({ type: 'send-now' }));
window.addEventListener('pagehide', () => padFlush(true));
document.addEventListener('visibilitychange', () => { if (document.hidden) padFlush(true); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && panel) openPanel(null);
  // Anita presses Ctrl+S out of habit; edits here save on their own, so the browser's Save as is never wanted.
  if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) e.preventDefault();
});
new ResizeObserver(() => {
  document.documentElement.style.setProperty('--head', `${$('#top').offsetHeight}px`);
}).observe($('#top'));
window.addEventListener('popstate', (e) => {
  if (!DATA) return;
  applyHash();
  const st = e.state;
  if (st && Array.isArray(st.notes)) { notesOpen.clear(); st.notes.forEach((n) => notesOpen.add(n)); }
  const restored = !!(panel && st && typeof st.sc === 'number');
  if (restored) pendingScroll = { top: st.sc, until: Date.now() + 4000 };
  render();
  if (panel && panel.task && !restored) {
    const row = $(`#panel [data-task="${panel.task}"]`);
    if (row) row.scrollIntoView({ block: 'center' });
  }
});
setInterval(() => DATA && renderStrip(), 60000);
// the orchestrator marks actions handled as it applies them; then the files have changed too
setInterval(async () => {
  if (!DATA || document.hidden) return;
  if (view === 'changelog' && !CL.menu && !clDrag.id && !RM.drag && !Object.values(CL.timers).some(Boolean)) loadChangelog();
  if (view === 'changelog' && !RM.drag && !RM.timer && !Object.keys(drafts).some((k) => k.startsWith('rm:'))) loadRoadmap();
  const was = new Set(INBOX.pending.map((a) => a.id));
  const padsWere = new Set(INBOX.pads.filter((p) => p.handled).map((p) => p.id));
  if (!(await fetchInbox())) return;
  const applied = [...was].some((id) => !withdrawn.has(id) && !INBOX.pending.some((a) => a.id === id))
    || INBOX.pads.some((p) => p.handled && !padsWere.has(p.id));
  if (applied) rebuild();
  else render();
}, POLL_MS);
rebuild();
