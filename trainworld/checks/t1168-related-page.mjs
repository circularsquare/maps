// T-1168: the clock query `lineRelated` in the page, as the line panel will ask it: a new game
// with a line and its copy made into an express (the copy beside the line on the same track,
// every station a meeting, the express's skipped stations passed), then the real save's 2, 6X and
// 7X against what t1168-related.mjs gets in Node. Run from the maps root on a free port:
//   node trainworld/checks/t1168-related-page.mjs <port>
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, launchChrome, collectErrors, importSave, buildLine, startVite } from './lib.mjs';

const port = Number(process.argv[2] ?? 6168);
const vite = await startVite(port);
const chrome = await launchChrome({ prefix: 'tw-t1168-' });
const errors = [];
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${vite.url}?debug=1&new=1&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });

  const local = await buildLine(page, [[-73.99, 40.70], [-73.98, 40.71], [-73.97, 40.72], [-73.96, 40.73], [-73.95, 40.74]], { name: 'Local' });
  check(local?.ok && local.line !== undefined, 'a five-station local built');
  const r = await page.evaluate(async (line) => {
    const tw = window.tw, c = tw.client;
    const { decodeRelated } = await import('/src/game/relatedLines.ts');
    const copy = await c.edit({ op: 'copyLine', line });
    const stops = tw.world.value.save.lines.find((l) => l.num === line).stops;
    await c.edit({ op: 'setStops', line: copy.line, stops: [stops[0], stops[2], stops[4]] });
    await new Promise((d) => setTimeout(d, 500));
    const a = decodeRelated(await c.query({ what: 'lineRelated', line }));
    const b = decodeRelated(await c.query({ what: 'lineRelated', line: copy.line }));
    return { copy: copy.line, a, b };
  }, local.line);
  const g = r.a[0];
  check(r.a.length === 1 && g.line === r.copy && g.kind === 'beside' && g.shared, `the express beside the local on the same track (${JSON.stringify(r.a.map((x) => [x.line, x.kind, x.shared]))})`);
  check(g && g.meetings.length === 5 && g.meetings.map((m) => m.stopO).join() === '0,-1,1,-1,2', `every station a meeting, the express passing 2 of them (${g?.meetings.map((m) => m.stopO)})`);
  check(g && g.track.length === 1 && g.stopsBefore === 0 && g.stopsAfter === 0 && g.own.length === 0, 'one shared stretch, nothing of its own');
  check(r.b.length === 1 && r.b[0].line === local.line && r.b[0].kind === 'beside', 'and the local beside the express');

  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await new Promise((d) => setTimeout(d, 1000));
  const node = JSON.parse(await readFile(new URL('../../artifacts/trainworld/t1168/related-T-007-real-nyc.json', import.meta.url), 'utf8'));
  for (const name of ['subway 2', 'subway 6X', 'subway 7X']) {
    const got = await page.evaluate(async (name) => {
      const tw = window.tw;
      const { decodeRelated } = await import('/src/game/relatedLines.ts');
      const lines = tw.world.value.save.lines, l = lines.find((x) => x.name === name);
      const t = performance.now();
      const v = await tw.client.query({ what: 'lineRelated', line: l.num });
      const ms = performance.now() - t;
      const byNum = new Map(lines.map((x) => [x.num, x.name]));
      return { ms, groups: decodeRelated(v).map((g) => [byNum.get(g.line), g.kind, g.meetings.length, g.own.length]) };
    }, name);
    const want = node[name].map((g) => g.kind).join();
    check(got.groups.map((g) => g[1]).join() === want, `${name}: ${got.groups.length} groups as in Node, ${got.groups.filter((g) => g[1] === 'beside').map((g) => g[0]).join(', ')} beside (${got.ms.toFixed(0)} ms round trip)`);
  }
  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await chrome.close();
  vite.child.kill();
}
process.exit(fail ? 1 : 0);
