// T-3037: hidden achievements. In a new game: the achievements view before; a test corridor (the Orient
// Express's ends swapped for two points on a Manhattan row) met by two lines that share a station, which
// earns it for real (toast, save); then two more posted through the clock client for the view and hovers.
// Also prints the predicates at known places and, with --real, each real save's national shares.
//   node trainworld/checks/t3037-hidden-achievements.mjs [base url, default http://localhost:6161/] [--real]
import { join } from 'node:path';
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { ARTIFACTS, REAL_SAVE, buildLine, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv.find((a) => a.startsWith('http')) ?? 'http://localhost:6161/';
const REAL = process.argv.includes('--real');
const TOKYO = 'C:/Users/anita/Documents/trainworld/Tokyo Seoul Fukuoka Busan (from anita.garden).save';
let failed = 0;
const say = (ok, what) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); };

async function open(ctx, shade = 'light') {
  const p = await ctx.newPage();
  await p.addInitScript((s) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: s, display: { dayShown: 'average' }, v: 2 })); } catch {} }, shade);
  await p.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&track=1&v=${Date.now()}`);
  await p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw.moneyState.value, undefined, { timeout: 180000 });
  return p;
}
const view = async (p, name) => {
  await p.evaluate(async () => { const { openAchievements } = await import('/src/ui/Achievements.tsx'); openAchievements(); });
  await p.waitForTimeout(600);
  const f = join(ARTIFACTS, `t3037-${name}.png`);
  await (await p.$('#dock')).screenshot({ path: f });
  console.log('shot', f);
};

const { ctx, close } = await launchChrome({ prefix: 'tw-t3037-' });
try {
  const p = await open(ctx);
  const errors = [];
  collectErrors(p, errors);
  // predicates at known places
  const preds = await p.evaluate(async () => {
    const a = await import('/src/game/achievements.ts');
    const at = (cc, lng, lat, zone = null) => ({ zone, cc, lng, lat });
    const c = (id) => a.CORRIDORS.find((x) => x.id === id);
    return {
      sapporoHokkaido: c('allJapan').from(at('JP', 141.35, 43.07)), hakodate: c('allJapan').from(at('JP', 140.73, 41.77)),
      aomoriIsHonshu: c('allJapan').from(at('JP', 140.74, 40.82)), omaIsHonshu: c('allJapan').from(at('JP', 140.91, 41.52)),
      takamatsu: c('allJapan').to(at('JP', 134.05, 34.35)), matsuyama: c('allJapan').to(at('JP', 132.77, 33.84)), kochi: c('allJapan').to(at('JP', 133.55, 33.56)),
      hiroshimaNot: c('allJapan').to(at('JP', 132.46, 34.4)), okayamaNot: c('allJapan').to(at('JP', 133.92, 34.66)), fukuokaNot: c('allJapan').to(at('JP', 130.42, 33.59)),
      irkutsk: c('siberian').to(at('RU', 104.28, 52.29)), novosibirskNot: c('siberian').to(at('RU', 82.9, 55.03)), vladivostok: c('siberian').to(at('RU', 131.89, 43.12)),
      berlinWest: c('siberian').from(at('DE', 13.4, 52.5)), warsawNot: c('siberian').from(at('PL', 21, 52.2)),
    };
  });
  const want = { sapporoHokkaido: true, hakodate: true, aomoriIsHonshu: false, omaIsHonshu: false, takamatsu: true, matsuyama: true, kochi: true, hiroshimaNot: false, okayamaNot: false, fukuokaNot: false, irkutsk: true, novosibirskNot: false, vladivostok: true, berlinWest: true, warsawNot: false };
  for (const [k, v] of Object.entries(want)) say(preds[k] === v, `predicate ${k} = ${v}`);

  await view(p, 'before');
  const before = await p.evaluate(() => document.querySelector('.ach-open .num')?.textContent);
  console.log('button before', before);
  const listedBefore = await p.evaluate(() => document.querySelectorAll('.ach-firsts li').length);

  // a corridor between two points on a Manhattan row, so the real path runs
  const x0 = -73.99, dx = 0.02, y = 40.758;
  await p.evaluate(([x0, dx]) => import('/src/game/achievements.ts').then((a) => {
    const c = a.CORRIDORS.find((z) => z.id === 'orient');
    c.from = (w) => w.lng < x0 + dx * 0.5;
    c.to = (w) => w.lng > x0 + dx * 3.5;
  }), [x0, dx]);
  const l1 = await buildLine(p, [0, 1, 2].map((k) => [x0 + k * dx, y]), { name: 'A', trains: [6, 4, 3], wait: { route: 300, station: 150, settle: 300 } });
  await p.waitForTimeout(2500);
  const afterOne = await p.evaluate(() => Object.keys(window.tw.moneyState.value.achievements ?? {}));
  say(!afterOne.includes('orient'), `one line (${l1?.ok}) does not meet a two-line corridor: ${afterOne.join(' ') || 'nothing earned'}`);
  const l2 = await buildLine(p, [2, 3, 4].map((k) => [x0 + k * dx, y + 0.0006 + (k - 2) * 0.002]), { name: 'B', trains: [6, 4, 3], wait: { route: 300, station: 150, settle: 300 } });
  console.log('line B', l2?.ok, l2?.issues);
  await p.waitForFunction(() => !!window.tw.moneyState.value.achievements?.orient, undefined, { timeout: 30000 }).catch(() => {});
  console.log('debug', JSON.stringify(await p.evaluate(async () => {
    const a = await import('/src/game/achievements.ts'), cx = (await import('/src/game/complexView.ts')).complexes.value, w = window.tw.world.value;
    return { lines: w.save.lines.map((l) => [l.name, l.stops, w.lineStats[l.id]?.status]), complexes: cx.members.filter((m) => m.length > 1), figs: [...(a.corridorFigures.value?.firsts ?? [])], stations: w.save.stations.map((s) => [s.id, s.built, s.name, s.lng.toFixed(4)]) };
  })));
  const got = await p.evaluate(() => window.tw.moneyState.value.achievements ?? {});
  say(!!got.orient, `two lines sharing a station earn it: ${JSON.stringify(got.orient)}`);
  await p.waitForTimeout(2500);
  const toast = await p.evaluate(() => document.querySelector('.ach-toast')?.textContent ?? null);
  console.log('toast:', toast);
  say(!!toast && /Orient Express/.test(toast), 'the toast names it');
  await p.screenshot({ path: join(ARTIFACTS, 't3037-toast.png') });

  // a second and third through the clock client
  await p.evaluate(async () => {
    const { postAchievements } = await import('/src/workers/clockClient.ts');
    postAchievements({ allJapan: { of: 'Sapporo', in: ['Kochi'] }, 'nation:JP': { of: 'JP', v: 10.3 } });
  });
  await p.waitForFunction(() => !!window.tw.moneyState.value.achievements?.allJapan, undefined, { timeout: 15000 });
  await p.waitForTimeout(1500);
  await view(p, 'after');
  const after = await p.evaluate(() => document.querySelector('.ach-open .num')?.textContent);
  const listedAfter = await p.evaluate(() => [...document.querySelectorAll('.ach-firsts li')].map((li) => li.textContent));
  console.log('button after', after, 'rows', listedBefore, '->', listedAfter.length, listedAfter.slice(0, 4));
  say(listedAfter.length === listedBefore + 3, "three hidden ones listed once earned");
  const tips = await p.evaluate(() => [...document.querySelectorAll('.ach-firsts li')].slice(0, 3).map((li) => li.getAttribute('data-tip')));
  console.log('hovers', JSON.stringify(tips));
  // T-3038: the report and the counts, against a fake server (the page's own start-up has none)
  await p.evaluate(async () => {
    const m = await import('/src/game/achievementCounts.ts');
    window.sent = [];
    m.startAchievementCounts(async (method, path, body) => {
      window.sent.push([method, path, body]);
      if (method === 'GET') return { players: 200, counts: { orient: 12, allJapan: 3, 'nation:JP': 0, 'cities:2': 150 } };
    });
    m.loadAchievementCounts();
  });
  await p.waitForTimeout(6000);
  const sent = await p.evaluate(() => window.sent);
  const posts = sent.filter((x) => x[0] === 'POST');
  console.log('sent', JSON.stringify(sent));
  say(posts.length === 1 && /^[0-9a-f]{32}$/.test(posts[0][2].b) && Object.keys(posts[0][2]).sort().join() === 'b,ids' && posts[0][2].ids.includes('orient'), 'one post, the browser id and the keys only');
  await p.evaluate(async () => { const { postAchievements } = await import('/src/workers/clockClient.ts'); postAchievements({ ghan: { of: 'A', in: ['B'] } }); });
  await p.waitForTimeout(6000);
  const posts2 = (await p.evaluate(() => window.sent)).filter((x) => x[0] === 'POST');
  say(posts2.length === 2 && posts2[1][2].ids.join() === 'ghan', `a later one sends only what is new: ${JSON.stringify(posts2[1]?.[2])}`);
  await view(p, 'counts');
  console.log('hover with share', JSON.stringify(await p.evaluate(() => [...document.querySelectorAll('.ach-firsts li')].slice(0, 4).map((li) => li.getAttribute('data-tip')))));
  for (const lang of ['ja', 'de']) {
    await p.evaluate(async (l) => { const m = await import('/src/i18n/index.ts'); await m.setLanguage(l); }, lang);
    await p.waitForTimeout(1200);
    console.log(lang, JSON.stringify(await p.evaluate(() => [...document.querySelectorAll('.ach-firsts li')].slice(0, 3).map((li) => li.textContent + ' | ' + li.getAttribute('data-tip')))));
  }
  say(errors.length === 0, `no errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
  await p.close();

  if (REAL) {
    for (const [name, path] of [['nyc', REAL_SAVE], ['tokyo', TOKYO]]) {
      if (!existsSync(path)) continue;
      const q = await open(ctx);
      await importSave(q, await readFile(path));
      await q.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
      await demandFinal(q, { timeout: 1800000, polling: 2000, lines: 0 });
      await q.waitForTimeout(3000);
      const r = await q.evaluate(async () => {
        const a = await import('/src/game/achievements.ts');
        const d = (await import('/src/game/demand.ts')).demandByDay.value[6];
        const cities = await import('/src/game/cities.ts');
        const ls = await import('/src/game/listScope.ts');
        const rail = {};
        const seen = new Set();
        for (const s of window.tw.world.value.save.stations) { const z = ls.stationZone(s); if (z) seen.add(z); }
        for (const z of seen) { const c = cities.cityById(z), v = d.cities.get(z); if (c.country && v?.final) rail[c.country] = (rail[c.country] ?? 0) + v.railTripsPerDay; }
        const shares = Object.fromEntries(a.NATIONS.filter((n) => rail[n.cc]).map((n) => [n.cc, +(100 * rail[n.cc] / (3 * n.people)).toFixed(3)]));
        return { shares, rail: Object.fromEntries(Object.entries(rail).map(([k, v]) => [k, Math.round(v)])), met: [...a.metNow.value.met ? Object.keys(a.metNow.value.met) : []] };
      });
      console.log(name, JSON.stringify(r));
      await q.close();
    }
  }
} finally {
  await close();
}
console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
