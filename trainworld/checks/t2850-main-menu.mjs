// T-2850: the Saved games panel's "Save and quit to main menu" button in the New game row. From a started game
// (a slot, not ?new=1) with a line built: the button is there, a click lands on the start menu, and the autosave
// holds the line (the start menu's card for the game counts 1 line).
//   TW_CORES=2 node trainworld/checks/t2850-main-menu.mjs http://localhost:6177/
import { GL, launchChrome, acrossLine, demandFinal } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6177/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t2850-', args: GL, size: [1400, 900] });
try {
  const page = await chrome.ctx.newPage();
  await page.goto(`${BASE}?debug=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .new-game', { timeout: 120000 });
  await page.click('.start-menu .new-game-cities [data-city="nyc"]');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu'), undefined, { timeout: 240000 });
  await demandFinal(page, { strict: false, timeout: 240000 }).catch(() => {});
  const line = await acrossLine(page, [-73.98, 40.75], 2);
  check(line !== null, 'a line is built');
  await page.click('.tab.gear.save');
  await page.waitForSelector('.save-files .save-main-menu', { timeout: 60000 });
  const label = await page.$eval('.save-files .save-main-menu', (b) => b.getAttribute('aria-label'));
  check(label === 'Save and quit to main menu', `the button is in the New game row (${label})`);
  await page.click('.save-files .save-main-menu');
  await page.waitForSelector('.start-menu', { timeout: 60000 });
  check(true, 'a click lands on the start menu');
  await page.waitForTimeout(3000);
  const text = await page.evaluate(() => document.querySelector('.start-menu').innerText);
  console.log(text.replace(/\s+/g, ' ').slice(0, 300));
  check(/\b1 line\b/.test(text), 'the saved game has the line (autosave holds the latest state)');
  // the start menu's own list has no such button
  check(!(await page.$('.save-main-menu')), 'no button on the start menu');
} finally { await chrome.close(); }
process.exitCode = fail ? 1 : 0;
