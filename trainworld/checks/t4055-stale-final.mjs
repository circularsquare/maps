// T-4055: an edit's final is the figures of the packs open when it lands. One edit reaching two new zones (Philadelphia
// and Providence from a new New York game) opens them while the load's hold is on, coarse, and the check of levels
// opens them again fine during that edit's solve; the solve's final then came on the coarse packs and the next solve
// was never waited for (Philadelphia line 63,996 riders against 68,341 for the same network solved again).
// Now a pack opening during a solve keeps its network waiting (demandClient `covers`), so the final read here must
// equal a fresh solve of the same network.
//   node trainworld/checks/t4055-stale-final.mjs [url]      (from the maps root; your own Vite, about 1-2 minutes)
// Prints PASS or FAIL lines and exits 1 on failure.
import { openGame } from './lib/play.mjs';
import { buildLine } from './lib.mjs';

const URL_ = process.argv[2] ?? 'http://localhost:6122/';
let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const riders = (page) => page.evaluate(() => {
  const v = window.twDemand.view(), names = new Map(window.tw.world.value.save.lines.map((l) => [l.id, l.name]));
  return { version: v.version, lines: Object.fromEntries([...v.lines].map(([id, l]) => [names.get(id) ?? id, Math.round(l.ridersPerDay ?? 0)])) };
});

const g = await openGame({ url: URL_, city: 'nyc', mode: 'sandbox' });
try {
  const page = g.page;
  // both lines in one go as soon as the game is up (the load's hold still on), no wait for demand between them
  await buildLine(page, [[-75.2, 39.95], [-75.15, 39.96], [-75.1, 39.97]], { name: 'PHL', trains: [6, 4, 3] });
  await buildLine(page, [[-71.45, 41.82], [-71.4, 41.83], [-71.35, 41.84]], { name: 'PVD', trains: [6, 4, 3] });
  await g.waitDemand({ lines: 1, timeout: 170000 });
  const first = await riders(page);
  await page.evaluate(() => window.twDemand.client.resubmit());
  await g.waitDemand({ after: first.version, timeout: 170000 });
  const again = await riders(page);
  for (const name of ['PHL', 'PVD']) {
    const a = first.lines[name] ?? 0, b = again.lines[name] ?? 0;
    check(`${name}'s riders at the edit's final equal a fresh solve's`, a > 0 && Math.abs(a - b) <= 0.005 * b, `${a} against ${b} (v${first.version}, v${again.version})`);
  }
  check('no page errors', g.pageErrors().length === 0, g.pageErrors().slice(0, 3).join(' | '));
} finally {
  await g.close();
}
console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
