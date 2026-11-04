// T-1656: the game's hover tag shows above a modal dialog (the saved games panel). Opens the panel,
// hovers a button in it and checks that the tag is shown as a popover, which a modal dialog's top
// layer cannot cover (elementFromPoint cannot tell: the page outside a modal is inert).
//   node trainworld/checks/t1656-tip-over-dialog.mjs <url> [screenshot.png]      (from the maps root)
import { launchChrome } from './lib.mjs';

const [BASE, SHOT] = process.argv.slice(2);
if (!BASE) { console.error('usage: node checks/t1656-tip-over-dialog.mjs <url> [screenshot.png]'); process.exit(2); }
const { ctx, close } = await launchChrome({ prefix: 'tw-tipdlg-', size: [1200, 800] });
let ok = false;
try {
  const page = await ctx.newPage();
  await page.goto(`${BASE}${BASE.includes('?') ? '&' : '?'}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });
  await page.click('.tab.save');
  await page.waitForSelector('dialog.saves-panel[open]');
  const btn = page.locator('dialog.saves-panel [data-tip]').first();
  await btn.hover();
  await page.mouse.move((await btn.boundingBox()).x + 4, (await btn.boundingBox()).y + 4);
  await page.waitForTimeout(300);
  const r = await page.evaluate(() => {
    const tag = document.querySelector('.ui-tag');
    if (!tag || tag.style.display === 'none') return { shown: false };
    const b = tag.getBoundingClientRect();
    return { shown: true, text: tag.textContent, w: Math.round(b.width), popover: tag.matches(':popover-open') };
  });
  console.log(JSON.stringify(r));
  if (SHOT) await page.screenshot({ path: SHOT });
  ok = r.shown && r.popover && r.w > 20;
} finally {
  await close();
}
console.log(`${ok ? 'PASS' : 'FAIL'}  tag above the saved games panel`);
process.exit(ok ? 0 : 1);
