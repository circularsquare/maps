// T-2683: a name the player set survives edits. A new game with three parallel tracks 40 m apart,
// a station at the west end of each (one complex: platforms within 50 m). The player renames the
// first, then a station is added to the complex and a line is built to the second, which then has
// the most lines; the first keeps its name and the complex still shows it (`complexLead`, what the
// Stations tab, search and diagram take as the complex's name).
// From the maps root:   node trainworld/checks/t2683-keep-names.mjs [url]
import { launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6178/';
const LAT = 40.652, LNG0 = -73.985;
const dLat = (m) => m / 111320, dLng = (m) => m / (111320 * Math.cos((LAT * Math.PI) / 180));
const row = (k) => [[LNG0, LAT + dLat(40 * k)], [LNG0 + dLng(600), LAT + dLat(40 * k)]];

const chrome = await launchChrome({ prefix: 'tw-t2683-', size: [1200, 800] });
const errors = [];
let failed = 0;
const ok = (cond, what) => { console.log(cond ? 'PASS' : 'FAIL', what); if (!cond) failed++; };
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  const r = await page.evaluate(async ([R0, R1, R2]) => {
    const tw = window.tw, c = tw.client, pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = (p) => { const q = tw.lngLatToPack(p[0], p[1]); return [q.x, q.y]; };
    const near = (p) => { const [x, y] = xy(p); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const free = (p) => { const [x, y] = xy(p); return { kind: 'free', x, y, level: -1 }; };
    const stations = () => tw.world.value.save.stations;
    const byName = (n) => stations().find((s) => s.name === n);
    const add = async (p, name) => { const a = await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 100, name }); await pause(250); return a; };
    for (const row of [R0, R1]) { const a = await c.edit({ op: 'route', from: free(row[0]), to: free(row[1]), pis: [], single: false }); await pause(300); if (!a.ok) return { err: 'route', issues: a.issues }; }
    const s = [await add(R0[0], 'Alpha'), await add(R1[0], 'Beta'), await add(R1[1], 'Beta East')];
    if (s.some((x) => !x.ok)) return { err: 'station', issues: s.map((x) => x.issues) };
    await c.edit({ op: 'constructAll' }); await pause(600);
    const a = byName('Alpha');
    if (!a) return { err: 'no Alpha', stations: stations().map((x) => x.name) };
    // the player names the complex's station; then a line gives Beta the most lines
    const rn = await c.edit({ op: 'stationProps', node: a.num, platform: a.length, name: 'Hand Name' }); await pause(300);
    const ln = await c.edit({ op: 'addLine', stops: [near(R1[0]), near(R1[1])], name: '', colour: '#0a7d3b' }); await pause(500);
    // a third track and a station on it, joining the complex
    const t = await c.edit({ op: 'route', from: free(R2[0]), to: free(R2[1]), pis: [], single: false }); await pause(300);
    const d = await add(R2[0], 'Delta');
    await c.edit({ op: 'constructAll' }); await pause(600);
    return { ok: [rn.ok, ln.ok, t.ok, d.ok], issues: [rn, ln, t, d].flatMap((x) => x.issues ?? []), names: stations().map((x) => x.name) };
  }, [row(0), row(1), row(2)]);
  console.log(JSON.stringify(r));
  ok(!r.err && r.ok.every(Boolean), 'the scene builds ' + (r.err ?? r.issues.join(',')));
  ok(r.names?.includes('Hand Name') && r.names.includes('Beta') && r.names.includes('Delta'), 'the player-named station keeps its name after a station and a line are added');
  const lead = await page.evaluate(async () => {
    const mod = await import('/src/game/complexLead.ts').catch((e) => ({ err: String(e) }));
    const cv = await import('/src/game/complexView.ts').catch((e) => ({ err: String(e) }));
    if (!mod.complexLead || !cv.complexOf) return { err: mod.err ?? cv.err ?? 'no complexLead' };
    const w = window.tw.world.value, st = w.save.stations;
    const named = st.find((s) => s.name === 'Hand Name');
    const members = cv.complexOf(named.id);
    const lead = mod.complexLead(st, w.save.lines, members);
    return { shown: lead?.name, members: members.length, flagged: st.filter((s) => s.named).map((s) => s.name), beta: st.find((s) => s.name === 'Beta').id };
  });
  console.log(JSON.stringify(lead));
  ok(lead.members === 3 && lead.flagged?.length === 1 && lead.flagged[0] === 'Hand Name', 'the three stations are one complex and only the typed name is flagged: ' + JSON.stringify(lead));
  ok(lead.shown === 'Hand Name','the complex shows the player\'s name though another member has more lines');
  ok(errors.length === 0, 'no console errors ' + errors.slice(0, 2).join(' | '));
} finally {
  await chrome.close();
}
process.exit(failed ? 1 : 0);
