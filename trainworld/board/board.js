// The trainworld board (T-3391, T-3392). Reads data.json, which build.py writes, and the inbox through
// serve.py's /twboard/_inbox; notes/T-3391.md has the formats.
'use strict';

const PRI_RANK = { P1: 1, P2: 2, P3: 3, P4: 4 };
const PRIS = ['P1', 'P2', 'P3', 'P4'];
const POLL_MS = 15000;
const $ = (sel) => document.querySelector(sel);

let DATA = null;
let VIEW = null;            // the lines as the board shows them: data.json plus her queued moves
let INBOX = { ok: false, pending: [], last_at: null, sent_at: null };
let panel = null;           // {kind: 'line', slug} | {kind: 'queue'} | {kind: 'add'} | {kind: 'ack'}
const drafts = {};          // text typed in the panel, by field key, so a re-render keeps it
const opened = new Set();   // task comment boxes that are open
let confirm = null;         // {key, action}: a select change waiting for Queue change or Cancel
const withdrawn = new Set();
let REQ = null;             // requests.json: player request groups, their reports, pasted items
let view = 'board';         // 'board' | 'requests'
const expanded = new Set(); // request groups showing their reports
let lastGroup = null;       // the group last opened, kept in the URL hash
let groupForm = null;       // {gid, kind: 'accept' | 'merge' | 'dismiss' | 'comment'}
const notesOpen = new Set(); // task rows showing their notes
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
  ARCHIVED = new Set(DATA.archived || []);
  ARCHIVE = null;
  for (const k of Object.keys(NOTES_CACHE)) delete NOTES_CACHE[k];
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

/** true when the pending list changed */
async function fetchInbox() {
  const before = INBOX.pending.map((a) => a.id).join();
  try {
    const r = await fetch('_inbox', { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    INBOX = { ok: true, ...(await r.json()) };
  } catch (e) {
    INBOX = { ok: false, pending: [], last_at: null, sent_at: null };
  }
  return INBOX.pending.map((a) => a.id).join() !== before;
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

function describe(a) {
  switch (a.type) {
    case 'reorder': return `Move ${lineName(a.line)} ${a.dir === 'down' ? 'down' : 'up'}`;
    case 'area': return `Move ${lineName(a.line)} to ${areaName(a.to)}`;
    case 'priority': return `Change ${a.task} to ${a.to}`;
    case 'move-task': return `Move ${a.task} to ${lineName(a.to)}`;
    case 'answer': return `Answer on ${a.task}: ${a.text}`;
    case 'text-ack': return a.text ? `Text acknowledged: ${a.text}` : 'Text acknowledged';
    case 'add': {
      const where = [a.area ? areaName(a.area) : '', a.line ? lineName(a.line) : ''].filter(Boolean).join(', ');
      return `New item${where ? ` (${where})` : ''}: ${a.text}`;
    }
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
    el('div', { class: 'what' }, rich(describe(a))),
    a.comment ? el('div', { class: 'why' }, a.comment) : null,
    el('div', { class: 'when' }, `Queued ${clock(a.at)} · `,
      el('button', { class: 'link', type: 'button', onclick: () => withdraw(a.id) }, 'Withdraw')))));
}

// ---------- filtering ----------

function query() { return $('#search').value.trim().toLowerCase(); }

function taskHits(t, q) {
  return t.id.toLowerCase().includes(q) || t.text.toLowerCase().includes(q);
}

/** null when the line is hidden; else the number of its tasks that match the search (or -1: no search) */
function lineMatch(line, q, update) {
  if (!line.status) return null;
  if (update && line.update !== update) return null;
  if (!q) return -1;
  const own = line.name.toLowerCase().includes(q) || line.about.toLowerCase().includes(q) || line.slug.includes(q);
  const n = line.tasks.filter((id) => taskHits(DATA.tasks[id], q)).length;
  if (!own && !n) return null;
  return n;
}

// ---------- rendering ----------

function render() {
  if (!DATA) return;
  buildView();
  renderTop();
  const req = view === 'requests';
  for (const id of ['#strip', '#grid-wrap', '#update']) $(id).hidden = req;
  $('#requests').hidden = !req;
  $('#search').placeholder = req ? 'Search requests' : 'Search lines and tasks';
  $('#v-board').setAttribute('aria-pressed', String(!req));
  $('#v-requests').setAttribute('aria-pressed', String(req));
  if (req) renderRequests();
  else {
    renderStrip();
    renderGrid();
  }
  renderWarnings();
  renderPanel();
}

function renderTop() {
  const n = INBOX.pending.length;
  for (const id of ['#add', '#ack', '#queued', '#send']) $(id).hidden = !INBOX.ok;
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

  const rows = live.agents.map((a) => el('tr', {},
    el('td', {}, el('b', {}, a.name)),
    el('td', { class: 'id' }, taskLink(a.task)),
    el('td', {}, rich(a.doing || '')),
    el('td', { class: 'time' }, ago(a.started))));
  strip.append(el('div', { class: 'block agents' },
    el('h2', {}, live.orchestrator ? `Agents (orchestrator ${live.orchestrator})` : 'Agents'),
    rows.length ? el('table', {}, rows) : el('div', { class: 'quiet' }, 'None running')));

  const runs = live.runs.map((r) => el('tr', {},
    el('td', {}, el('b', {}, r.name)),
    el('td', {}, r.doing || ''),
    el('td', { class: 'time' }, ago(r.started))));
  strip.append(el('div', { class: 'block' },
    el('h2', {}, 'Long runs'),
    runs.length ? el('table', {}, runs) : el('div', { class: 'quiet' }, 'None')));

  const c = DATA.cores;
  const claims = (c && c.claims) || [];
  strip.append(el('div', { class: 'block' },
    el('h2', {}, 'Cores'),
    c && c.cap != null ? el('div', {}, `${c.claimed} of ${c.cap} claimed`) : el('div', { class: 'quiet' }, 'Ledger not readable'),
    claims.length ? el('table', {}, claims.map((x) => el('tr', {},
      el('td', {}, x.name), el('td', {}, String(x.cores)), el('td', { class: 'time' }, x.age)))) : null));

  const b = live.builds || {};
  const kv = el('div', { class: 'kv' });
  const add = (k, v) => { if (v) kv.append(el('span', {}, k), el('span', { class: 'mono' }, v)); };
  add('master', DATA.head || b.master);
  add('8800', b['8800']);
  add('live', b.live);
  strip.append(el('div', { class: 'block' }, el('h2', {}, 'Builds'), kv));
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
  for (const line of VIEW) if (shown.has(line.slug)) perCol[line.status] = (perCol[line.status] || 0) + 1;

  grid.append(el('div', { class: 'colhead', style: 'background: var(--bg)' }));
  for (const col of DATA.columns) {
    grid.append(el('div', { class: 'colhead', style: colour(col.key) },
      col.name, perCol[col.key] ? el('span', { class: 'n' }, ` ${perCol[col.key]}`) : null));
  }
  for (const area of DATA.areas) {
    grid.append(el('div', { class: 'rowhead' }, area.name));
    for (const col of DATA.columns) {
      const cell = el('div', { class: 'cell', 'data-area': area.slug, 'data-status': col.key });
      for (const line of VIEW) {
        if (line.area !== area.slug || line.status !== col.key || !shown.has(line.slug)) continue;
        cell.append(card(line, shown.get(line.slug)));
      }
      grid.append(cell);
    }
  }
}

function card(line, hits) {
  let count;
  if (line.status === 'done') count = line.done ? `Done ${day(line.done)}` : 'Done';
  else count = hits > 0 ? `${hits} match` : `${line.open} open`;
  const bar = el('div', { class: 'bar' });
  for (const col of DATA.columns) {
    const n = line.counts[col.key];
    if (n) bar.append(el('span', { style: `flex: ${n}; background: var(--s-${col.key})` }));
  }
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
  const focus = document.activeElement && box.contains(document.activeElement) ? document.activeElement.dataset.key : null;
  const caret = focus ? [document.activeElement.selectionStart, document.activeElement.selectionEnd] : null;
  const scroll = box.scrollTop;
  let kids = null;
  if (panel && panel.kind === 'line') kids = linePanel(VIEW.find((l) => l.slug === panel.slug));
  else if (panel && panel.kind === 'queue') kids = queuePanel();
  else if (panel && panel.kind === 'add') kids = addPanel();
  else if (panel && panel.kind === 'ack') kids = ackPanel();
  else if (panel && panel.kind === 'closed') kids = closedPanel(panel.id);
  else if (panel && panel.kind === 'notes') kids = notesPanel(panel.name);
  if (!kids) { box.hidden = true; box.replaceChildren(); return; }
  box.replaceChildren(...kids.filter(Boolean));
  box.hidden = false;
  box.scrollTop = scroll;
  if (focus) {
    const f = box.querySelector(`[data-key="${CSS.escape(focus)}"]`);
    if (f) {
      f.focus();
      if (caret && f.setSelectionRange && caret[0] != null) try { f.setSelectionRange(caret[0], caret[1]); } catch (e) { /* a select */ }
    }
  }
}

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
  for (const id of line.tasks) (byStatus[DATA.tasks[id].status] ||= []).push(DATA.tasks[id]);
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
    editable ? el('div', { class: 'controls' },
      el('button', { class: 'btn', type: 'button', disabled: !prev, onclick: () => move(line, 'up') }, 'Move up'),
      el('button', { class: 'btn', type: 'button', disabled: !next, onclick: () => move(line, 'down') }, 'Move down'),
      changeSelect(`${k}:area`, 'Area', line.area, areaOptions(), (to) => ({ type: 'area', line: line.slug, from: line.area, to }))) : null,
    editable ? confirmRow(`${k}:area`) : null,
    mine.length ? queuedList(mine) : null,
    INBOX.ok ? el('div', { class: 'compose' },
      textBox(`${k}:comment`, 'Comment on this line'),
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
  const where = el('div', { class: 'where' });
  notes.forEach((n, i) => where.append(...(i ? [' · '] : []), ...n));

  const k = `task:${t.id}`;
  const pend = pendingFor(null, t.id);
  const waitsOnYou = t.waits.includes('Anita') && !pend.some((a) => a.type === 'answer');
  let tools = null;
  if (INBOX.ok) {
    const priOpts = (t.pri ? [] : [{ value: '', label: 'none' }]).concat(PRIS.map((p) => ({ value: p, label: p })));
    tools = [
      el('div', { class: 'controls small' },
        changeSelect(`${k}:pri`, 'Priority', queuedTo(pend, 'priority', t.pri || ''), priOpts, (to) => ({ type: 'priority', task: t.id, from: t.pri, to })),
        changeSelect(`${k}:line`, 'Line', queuedTo(pend, 'move-task', t.line), lineOptions(t.line), (to) => ({ type: 'move-task', task: t.id, from: t.line, to })),
        opened.has(t.id) ? null : el('button', { class: 'link', type: 'button', onclick: () => { opened.add(t.id); renderPanel(); } }, 'Comment')),
      confirmRow(`${k}:pri`),
      confirmRow(`${k}:line`),
      waitsOnYou ? el('div', { class: 'compose' },
        textBox(`${k}:answer`, 'Your answer', 3),
        el('button', { class: 'btn', type: 'button', onclick: () => submit(`${k}:answer`, { type: 'answer', task: t.id }) }, 'Save answer')) : null,
      opened.has(t.id) ? el('div', { class: 'compose' },
        textBox(`${k}:comment`, `Comment on ${t.id}`),
        el('button', { class: 'btn', type: 'button', onclick: async () => { await submit(`${k}:comment`, { type: 'comment', task: t.id, line: t.line }); if (!drafts[`${k}:comment`]) { opened.delete(t.id); renderPanel(); } } }, 'Add comment'),
        el('button', { class: 'btn plain', type: 'button', onclick: () => { opened.delete(t.id); delete drafts[`${k}:comment`]; renderPanel(); } }, 'Cancel')) : null,
    ];
  }
  const showing = t.notes && notesOpen.has(t.id);
  if (t.notes) {
    where.append(...(notes.length ? [' · '] : []), el('button', { class: 'link', type: 'button', onclick: () => {
      if (showing) notesOpen.delete(t.id); else notesOpen.add(t.id);
      renderPanel();
    } }, showing ? 'Hide notes' : 'Show notes'));
  }
  const flash = flashTask === t.id && Date.now() < flashUntil;
  // the row's parts in order; the notes sit right above the answer box so she can read and answer in one place
  return el('div', { class: 'task-row' + (q && taskHits(t, q) ? ' hit' : '') + (flash ? ' flash' : ''), style: colour(key), 'data-task': t.id },
    el('div', { class: 'top' }, el('b', {}, t.id), t.pri ? ` ${t.pri}` : ' no priority'),
    taskBody(t),
    where,
    showing ? notesBlock(t.notes, t.id) : null,
    pend.length ? queuedList(pend) : null,
    tools);
}

/** what a task says; tomorrow's task fields (need from Anita, context, goal) go here */
function taskBody(t) {
  return el('div', { class: 'text' }, rich(t.text));
}

function queuePanel() {
  const list = INBOX.pending;
  return [
    head('Queued'),
    el('p', { class: 'about' }, 'The orchestrator picks these up 3 minutes after your last change, or when you press Send now.'),
    list.length ? queuedList(list) : el('p', { class: 'meta' }, 'Nothing queued'),
  ];
}

function addPanel() {
  const k = 'add';
  const area = drafts['add:area'] || '';
  const lineSel = el('select', { class: 'small', 'data-key': 'add:line', onchange: (e) => { drafts['add:line'] = e.target.value; } },
    el('option', { value: '' }, 'No line'),
    lineOptions().filter((g) => !area || g.group === areaName(area)).map((g) => el('optgroup', { label: g.group },
      g.options.map((o) => el('option', { value: o.value }, o.label)))));
  lineSel.value = drafts['add:line'] || '';
  const areaSel = el('select', {
    class: 'small', 'data-key': 'add:area',
    onchange: (e) => { drafts['add:area'] = e.target.value; drafts['add:line'] = ''; renderPanel(); },
  }, el('option', { value: '' }, 'No area'), areaOptions().map((o) => el('option', { value: o.value }, o.label)));
  areaSel.value = area;
  return [
    head('Add item'),
    el('div', { class: 'compose' }, textBox(`${k}:text`, 'What should happen', 5)),
    el('div', { class: 'controls' },
      el('label', { class: 'field' }, el('span', {}, 'Area'), areaSel),
      el('label', { class: 'field' }, el('span', {}, 'Line'), lineSel)),
    el('div', { class: 'controls' },
      el('button', { class: 'btn', type: 'button', onclick: async () => {
        const action = { type: 'add' };
        if (drafts['add:area']) action.area = drafts['add:area'];
        if (drafts['add:line']) action.line = drafts['add:line'];
        if (!textOf(`${k}:text`)) return;
        await submit(`${k}:text`, action);
        delete drafts['add:area'];
        delete drafts['add:line'];
        openPanel({ kind: 'queue' });
      } }, 'Add item')),
  ];
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
  if (view === 'requests') return lastGroup ? `#requests/${lastGroup}` : '#requests';
  if (panel && panel.kind === 'line') return `#${panel.slug}`;
  return '';
}

/** push: a jump Back should undo (a link to a task or notes); otherwise the hash is just kept current */
function setHash(push) {
  const h = hashNow();
  if (h === location.hash || (!h && !location.hash)) return;
  const url = h || location.pathname + location.search;
  if (push) history.pushState(null, '', url); else history.replaceState(null, '', url);
}

function applyHash() {
  const hash = decodeURIComponent(location.hash.slice(1));
  panel = null;
  view = 'board';
  if (hash === 'requests' || hash.startsWith('requests/')) {
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

/** text with task ids and notes paths as links; anything that resolves to nothing stays plain */
function rich(text) {
  const s = String(text == null ? '' : text);
  if (!DATA) return s;
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

async function loadNotes(name) {
  NOTES_CACHE[name] = { state: 'loading' };
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
    NOTES_CACHE[name] = { state: 'missing' };
  }
  if (view === 'requests') renderRequests();
  renderPanel();
}

// A small markdown reader for the notes: headings, lists, tables, quotes, code, bold, links. It builds DOM nodes
// (never HTML strings), so nothing in a note can run as markup.
function md(src) {
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
      out.push(el('div', { class: 'md-table' }, el('table', {},
        el('thead', {}, el('tr', {}, head.map((c) => el('th', {}, inline(c))))),
        el('tbody', {}, body.map((r) => el('tr', {}, r.map((c) => el('td', {}, inline(c)))))))));
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

const INLINE_RE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\[[^\]]+\]\([^)\s]+\))|(\*[^*\s][^*]*\*)/g;

function inline(text) {
  const out = [];
  let at = 0;
  for (const m of text.matchAll(INLINE_RE)) {
    if (m.index > at) out.push(...rich(text.slice(at, m.index)));
    const t = m[0];
    if (m[1]) out.push(el('code', {}, t.slice(1, -1)));
    else if (m[2]) out.push(el('b', {}, inline(t.slice(2, -2))));
    else if (m[4]) out.push(el('i', {}, inline(t.slice(1, -1))));
    else {
      const [, label, url] = t.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
      if (/^https?:\/\//i.test(url)) out.push(el('a', { href: url, target: '_blank', rel: 'noopener noreferrer' }, label));
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
  if (v === 'requests' && panel && panel.kind === 'line') panel = null;
  setHash();
  render();
  window.scrollTo(0, 0);
}

function stamp(iso) {
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
function keepFocus(box, fn) {
  const a = document.activeElement;
  const key = a && box.contains(a) ? a.dataset.key : null;
  const caret = key && a.selectionStart != null ? [a.selectionStart, a.selectionEnd] : null;
  fn();
  if (!key) return;
  const f = box.querySelector(`[data-key="${CSS.escape(key)}"]`);
  if (!f) return;
  f.focus();
  if (caret) try { f.setSelectionRange(caret[0], caret[1]); } catch (e) { /* a select */ }
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
        el('label', { class: 'field' }, el('span', {}, 'Priority'), sel('pri', PRIS.map((p) => ({ value: p, label: p })), pri))),
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

// ---------- wiring ----------

$('#search').addEventListener('input', () => render());
$('#v-board').addEventListener('click', () => showView('board'));
$('#v-requests').addEventListener('click', () => showView('requests'));
$('#update').addEventListener('change', () => DATA && renderGrid());
$('#refresh').addEventListener('click', rebuild);
$('#add').addEventListener('click', () => openPanel({ kind: 'add' }));
$('#ack').addEventListener('click', () => openPanel({ kind: 'ack' }));
$('#queued').addEventListener('click', () => openPanel({ kind: 'queue' }));
$('#send').addEventListener('click', () => post({ type: 'send-now' }));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && panel) openPanel(null);
});
new ResizeObserver(() => {
  document.documentElement.style.setProperty('--head', `${$('#top').offsetHeight}px`);
}).observe($('#top'));
window.addEventListener('popstate', () => {
  if (!DATA) return;
  applyHash();
  render();
  if (panel && panel.task) {
    const row = $(`#panel [data-task="${panel.task}"]`);
    if (row) row.scrollIntoView({ block: 'center' });
  }
});
setInterval(() => DATA && renderStrip(), 60000);
// the orchestrator marks actions handled as it applies them; then the files have changed too
setInterval(async () => {
  if (!DATA || document.hidden) return;
  const was = new Set(INBOX.pending.map((a) => a.id));
  if (!(await fetchInbox())) return;
  const applied = [...was].some((id) => !withdrawn.has(id) && !INBOX.pending.some((a) => a.id === id));
  if (applied) rebuild();
  else render();
}, POLL_MS);
rebuild();
