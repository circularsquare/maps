// T-2021, T-2679: a country's fare multiplier is a multiple of the reference country's fares, set to the nearest
// 0.01x, so a displayed 1.30x acts as exactly 1.30 times the reference's: a value like 1.2999 or 1.304 stores, charges
// and shows as 1.30, in a game from T-1906 on (multiples of the default stored) and in an older save (the scale
// itself stored), with the US or Japan as the reference. A save without a reference country rounds its scales to
// 0.01x of the default as it loads (`tidyOwn`). A country with none set shows its default multiple of the reference.
//   node trainworld/checks/t2021-fare-multiple.mjs      (from the maps root; no browser, about 5 s)
// Prints PASS or FAIL lines and exits 1 on failure.
import { loadApp } from './lib.mjs';

globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };

const { fareScale, fareFactors, shownScale, storedScale, tidyOwn, incomeScale } = await loadApp(
  "export { fareScale, fareFactors, shownScale, storedScale, tidyOwn, incomeScale } from '@app/src/game/income.ts';");

let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const start = incomeScale('US');
for (const [label, base] of [
  ['a game from T-1906 on (multiples stored)', { '@': start }],
  ['an older save (the scale stored)', {}],
]) {
  for (const ref of ['US', 'JP']) {
    const bad = [];
    const rs = fareScale(ref, base);
    for (const c of ['JP', 'DE', 'IN', 'US', 'MX'].filter((c) => c !== ref)) {
      const at = (v) => ({ ...base, [c]: storedScale(v, c, base, rs) });
      const want = (what, ok) => { if (!ok) bad.push(`${c} ${what}`); };
      const s13 = at(1.3);
      for (const v of [1.2999, 1.304, 1.3049, 1.2951]) {
        const s = at(v);
        want(`${v} stores differently`, s[c] === s13[c]);
        want(`${v} does not show 1.30x`, shownScale(fareScale(c, s), rs) === 1.3);
        want(`${v} gives other fare factors`, same(fareFactors(s), fareFactors(s13)));
      }
      want('1.30x is not exactly 1.30 times the reference', Math.abs(fareScale(c, s13) - 1.3 * rs) < 1e-12);
      want('setting it moves the reference', fareScale(ref, s13) === rs);
    }
    check(`${label}, ${ref} the reference: 1.2999, 1.304, 1.3049 and 1.2951 all act as 1.30`, bad.length === 0, bad.join('; '));
  }
}
{
  const bad = [];
  for (const [label, mk] of [['multiples stored', (m, c) => ({ '@': start, [c]: m })], ['the scale stored', (m, c) => ({ [c]: m * incomeScale(c) })]]) {
    for (const c of ['JP', 'DE', 'IN', 'US']) {
      const r = mk(1.3, c);
      for (const v of [1.2999, 1.304, 1.3049, 1.2951]) {
        const s = mk(v, c);
        if (!same(tidyOwn(c, s[c], s), tidyOwn(c, r[c], r))) bad.push(`${label} ${c} ${v}`);
      }
    }
  }
  check('a save without a reference country loads 1.2999 etc. as 1.30x of the default (both forms)', bad.length === 0, bad.join('; '));
}
{
  // T-2679: a save from before (no reference country) loads its scales through `tidyOwn` and charges what it did,
  // when `fareScale` rounded each multiple to 0.01x of the default as it was used
  const { defaultScale } = await loadApp("export { defaultScale } from '@app/src/game/income.ts';");
  const r2 = (v) => Math.round(v * 100) / 100;
  const before = (c, s) => (s['@'] !== undefined || s['*'] !== undefined) ? Math.min(19.9, Math.max(0.1, r2(s[c]))) * defaultScale(c, s) : Math.min(19.9, Math.max(0.01, r2(s[c] / defaultScale(c, s)))) * defaultScale(c, s);
  const bad = [];
  for (const base of [{ '@': start }, { '@': incomeScale('MX') }, {}, { '*': 1 }]) {
    for (const c of ['MX', 'JP', 'US', 'IN']) {
      for (const v of [0.45, 1.13, 1.2999, 2, 7.3, 0.6]) {
        const saved = { ...base, [c]: v }, loaded = { ...base, [c]: tidyOwn(c, v, saved) };
        if (fareScale(c, loaded) !== before(c, saved)) bad.push(`${JSON.stringify(saved)}: ${fareScale(c, loaded)} not ${before(c, saved)}`);
      }
    }
  }
  check('a save from before with Mexico, Japan, the US or India set charges exactly what it did (every save form)', bad.length === 0, bad.slice(0, 3).join('; '));
}
const s0 = { '@': start }, us = fareScale('US', s0);
const shown = Object.fromEntries(['US', 'MX', 'JP', 'IN'].map((c) => [c, shownScale(fareScale(c, s0), us)]));
check('a US start with none set: US 1.00x, Mexico 0.40x, Japan 0.63x, India 0.17x', same(shown, { US: 1, MX: 0.4, JP: 0.63, IN: 0.17 }), JSON.stringify(shown));
const mx = fareScale('MX', s0), us2 = shownScale(fareScale('US', s0), mx);
check('the same game with Mexico the reference: US 2.52x', us2 === 2.52, String(us2));
const old = {}, oldUs = fareScale('US', old);
check('an older save with none set: US 1.00x, Japan 0.63x', shownScale(fareScale('US', old), oldUs) === 1 && shownScale(fareScale('JP', old), oldUs) === 0.63);

process.exitCode = failed ? 1 : 0;
