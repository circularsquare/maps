// Screenshot mockup HTML pages to PNG: node shot.mjs <in.html> <out.png> [...pairs]
import { pathToFileURL } from 'node:url';
const lib = await import(pathToFileURL('C:/Users/anita/projects/maps/trainworld/checks/lib.mjs').href);
const { ctx, close } = await lib.launchChrome({ prefix: 'tw-t1168-', size: [1060, 600], args: [] });
try {
  const page = await ctx.newPage();
  const a = process.argv.slice(2);
  for (let i = 0; i + 1 < a.length; i += 2) {
    await page.goto(pathToFileURL(a[i]).href);
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(300);
    await page.screenshot({ path: a[i + 1], fullPage: true });
    console.log('wrote', a[i + 1]);
  }
} finally {
  await close();
}
