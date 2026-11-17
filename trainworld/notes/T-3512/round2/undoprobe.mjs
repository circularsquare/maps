// settle2: does an undo take back a train count edit? Load the NYC save, then edit/undo pairs at set gaps, the
// world's trains read after each. env: NAME BASE QS GAP0 (s between the load's final and the first edit)
import { readFileSync, writeFileSync } from 'node:fs';
const LIB = 'file:///C:/Users/anita/projects/maps/.tw-trees/settle/trainworld/checks/';
const { launchChrome, importSave, demandFinal, REAL_SAVE } = await import(LIB + 'lib.mjs');
const SCR = 'C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f3eca8a5-6550-4b65-9245-def721281ba7/scratchpad/settle2/';
const name = process.env.NAME ?? 'probe', BASE = process.env.BASE ?? 'http://localhost:6154/';
const sec = (s) => new Promise((r) => setTimeout(r, s * 1000));
const out = { name, rows: [] };
const { ctx, close } = await launchChrome({ profile: (SCR + 'profile-' + name).replace(/\//g, '\\') });
const T = () => window.tw.world.value.lineStats[0]?.trainsNeeded?.join(',') + ' wv ' + window.tw.world.value.version + ' canUndo ' + window.tw.world.value.canUndo + ' clock ' + JSON.stringify({ running: window.tw.clock?.running?.peek?.() });
try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  const logs = [];
  page.on('console', (m) => { const t = m.text(); if (/undo|edit|AutoTrain|T1071/i.test(t)) logs.push(t.slice(0, 200)); });
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(BASE + '?debug=1&new=1&paused=1' + (process.env.QS ?? '&lod=fine') + '&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await sec(8);
  await importSave(page, readFileSync(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  let prevAll = null;
  const ALL = () => Object.fromEntries(Object.entries(window.tw.world.value.lineStats).map(([k, x]) => [k, x.trainsNeeded.join(',')]));
  const row = async (what) => { const r = [what, await page.evaluate(T)]; const all = await page.evaluate(ALL); const moved = prevAll ? Object.keys(all).filter((k) => all[k] !== prevAll[k]).map((k) => k + ':' + prevAll[k] + '->' + all[k]) : []; prevAll = all; r.push('moved ' + moved.join(' ')); out.rows.push(r); console.log(r.join(' | ')); };
  await row('loaded (before final)');
  await demandFinal(page, { timeout: 3600000, polling: 100, lines: 0 });
  await row('load final');
  const edit = (op) => page.evaluate(async (op) => {
    const tw = window.tw, w = tw.world.value, l = w.save.lines[0];
    if (op === 'undo') return JSON.stringify(await tw.client.edit({ op: 'undo' }));
    const had = w.lineStats[l.id].trainsNeeded;
    return JSON.stringify(await tw.client.edit({ op: 'setTrainCount', line: l.num, trains: had.map((x) => x + 1) }));
  }, op);
  for (const gap of [Number(process.env.GAP0 ?? 0), 10, 0, 10]) {
    await sec(gap);
    await row(`after ${gap} s`);
    await row('edit all +1 ' + (await edit('all')));
    await sec(2);
    await row('2 s after the edit');
    await row('undo ' + (await edit('undo')));
    await sec(2);
    await row('2 s after the undo');
  }
  out.logs = logs.slice(0, 100);
} catch (e) { out.error = String(e.stack ?? e).slice(0, 600); console.log(out.error); } finally {
  writeFileSync(SCR + 'out/' + name + '.json', JSON.stringify(out, null, 1));
  await close();
}
process.exit(0);
