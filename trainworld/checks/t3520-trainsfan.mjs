// T-3520: the trainsfan achievement (game/hoursPlayed.ts). A new game on a dev server: with 99.9 hours
// played it is not earned and the hover shows the figure; with the count a few seconds under 100 hours it is
// earned as the count passes, with a toast, and the count survives a reload. Screenshots of the panel go to
// artifacts/trainworld/t3520_*.png.
//   node trainworld/checks/t3520-trainsfan.mjs [base url, default http://localhost:6167/]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.find((a) => a.startsWith('http')) ?? 'http://localhost:6167/';
const KEY = 'trainworld.played';
let failed = 0;
const say = (ok, what) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-t3520-' });
try {
  const errors = [];
  const p = await ctx.newPage();
  collectErrors(p, errors);
  const open = async () => {
    await p.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
    await p.waitForFunction(() => window.tw?.world?.value && window.tw.moneyState.value, undefined, { timeout: 180000 });
  };
  const state = () => p.evaluate(() => ({ earned: Object.keys(window.tw.moneyState.value.achievements ?? {}), stored: Number(localStorage.getItem('trainworld.played')) }));
  const panel = async (name) => {
    await p.evaluate(async () => { const { openAchievements } = await import('/src/ui/Achievements.tsx'); openAchievements(); });
    await p.waitForTimeout(800);
    const row = p.locator('.ach-firsts li', { hasText: 'trainsfan' });
    await row.hover();
    await p.waitForTimeout(700);
    await (await p.$('#dock')).screenshot({ path: join(ARTIFACTS, name) });
    console.log(`shot ${join(ARTIFACTS, name)}`);
  };

  // 99.9 hours: not earned, and the stored count keeps growing
  await open();
  await p.evaluate((k) => localStorage.setItem(k, String(99.9 * 3600)), KEY);
  await open();
  await p.waitForTimeout(12000); // past a save (10 s)
  let s = await state();
  say(!s.earned.includes('trainsfan'), `at 99.9 h trainsfan is not earned (${s.stored.toFixed(0)} s stored)`);
  say(s.stored > 99.9 * 3600 + 5 && s.stored < 100 * 3600, 'the count keeps adding while a game is open');
  await panel('t3520_unearned.png');

  // survives a reload
  await open();
  const afterReload = (await state()).stored;
  say(afterReload >= s.stored - 1, `the count survives a reload (${s.stored.toFixed(0)} -> ${afterReload.toFixed(0)} s)`);

  // a few seconds under 100 hours: earned as it passes
  await p.evaluate((k) => localStorage.setItem(k, String(100 * 3600 - 4)), KEY);
  await open();
  await p.waitForFunction(() => !!window.tw.moneyState.value.achievements?.trainsfan, undefined, { timeout: 30000 }).catch(() => {});
  s = await state();
  say(s.earned.includes('trainsfan'), 'the count passing 100 h earns trainsfan');
  const toast = await p.evaluate(() => [...document.querySelectorAll('.ach-toast')].map((t) => t.textContent));
  say(toast.some((t) => /trainsfan/.test(t)), `with a toast ${JSON.stringify(toast)}`);
  await p.waitForTimeout(500);
  await panel('t3520_earned.png');

  // earned stays after a reload of the count
  say(errors.length === 0, `no errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
  await p.close();
} finally {
  await close();
}
console.log(failed ? `${failed} FAILED` : 'all passed');
process.exitCode = failed ? 1 : 0;
