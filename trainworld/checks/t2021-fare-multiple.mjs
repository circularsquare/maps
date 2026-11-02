// T-2021: a country's fare multiplier is kept and used to the nearest 0.01x, so a displayed 1.30x acts as
// exactly 1.30: a value like 1.2999 or 1.304 gives the same scale, factor and display as 1.30, in a game
// from T-1906 on (multiples stored) and in an older save (the scale itself stored).
//   node trainworld/checks/t2021-fare-multiple.mjs      (from the maps root; no browser, about 5 s)
// Prints PASS or FAIL lines and exits 1 on failure.
import { loadApp } from './lib.mjs';

globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };

const { fareScale, fareFactors, shownScale, storedScale, defaultScale, tidyOwn, incomeScale } = await loadApp(
  "export { fareScale, fareFactors, shownScale, storedScale, defaultScale, tidyOwn, incomeScale } from '@app/src/game/income.ts';");

let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const start = incomeScale('US');
for (const [label, mk] of [
  ['a game from T-1906 on (multiples stored)', (m, c) => ({ '@': start, [c]: m })],
  ['an older save (the scale stored)', (m, c) => ({ [c]: m * incomeScale(c) })],
]) {
  const bad = [];
  for (const c of ['JP', 'DE', 'IN', 'US']) {
    const ref = mk(1.3, c);
    const want = (what, ok) => { if (!ok) bad.push(`${c} ${what}`); };
    for (const v of [1.2999, 1.304, 1.3049, 1.2951]) {
      const s = mk(v, c);
      want(`${v} charges differently`, fareScale(c, s) === fareScale(c, ref));
      want(`${v} does not show 1.30x`, shownScale(fareScale(c, s), c, s) === 1.3);
      want(`${v} gives other fare factors`, same(fareFactors(s), fareFactors(ref)));
      want(`${v} is kept as something else`, same(tidyOwn(c, s[c], s), tidyOwn(c, ref[c], ref)));
    }
    want('1.30x is not exactly 1.30 times the default', Math.abs(fareScale(c, ref) - 1.3 * defaultScale(c, ref)) < 1e-12);
    want('typing 1.2999 stores other than 1.30', storedScale(1.2999, c, ref) === storedScale(1.3, c, ref));
    want('what is stored reads back other than 1.30x', shownScale(fareScale(c, { ...ref, [c]: storedScale(1.3, c, ref) }), c, ref) === 1.3);
  }
  check(`${label}: 1.2999, 1.304, 1.3049 and 1.2951 all act as 1.30 (JP, DE, IN, US)`, bad.length === 0, bad.join('; '));
}
check('no multiplier set shows 1.00x', shownScale(fareScale('JP', { '@': start }), 'JP', { '@': start }) === 1);
check('an older save with none set shows 1.00x', shownScale(fareScale('JP', {}), 'JP', {}) === 1);

process.exitCode = failed ? 1 : 0;
