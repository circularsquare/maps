// T-3318: zones and the Kansai metro show in Japanese and Chinese (i18n/places.ts, place-names-zones.json).
//   node trainworld/checks/t3318-zone-names.mjs [port]     (needs the app's Vite on that port, dev-nowatch.mjs; default 6173)
import { launchChrome, collectErrors } from './lib.mjs';

const port = process.argv[2] ?? 6173;
const WANT = {
  'zh-Hans': { cau: '常熟', osa: '关西', dor: '鲁尔区' },
  'zh-TW': { cau: '常熟', osa: '關西' },
  ja: { cau: '常熟', osa: '関西', sdj: '仙台' },
  en: { cau: 'Changshu' },
};
const NAMES = { cau: 'Changshu', osa: 'Kansai', dor: 'Ruhr', sdj: 'Sendai' };
const { ctx, close } = await launchChrome({ prefix: 'tw-3318-' });
let failed = 0;
try {
  for (const [lang, want] of Object.entries(WANT)) {
    const page = await ctx.newPage();
    const errors = [];
    collectErrors(page, errors);
    await page.goto(`http://localhost:${port}/?new=1&lang=${lang}`, { waitUntil: 'domcontentloaded' });
    const got = await page.evaluate(async ([ids, names]) => {
      const { tPlace } = await import('/src/i18n/places.ts');
      tPlace('cau', 'x'); // asks for the zones file
      await new Promise((r) => setTimeout(r, 1500));
      return Object.fromEntries(ids.map((id) => [id, tPlace(id, names[id])]));
    }, [Object.keys(want), NAMES]);
    for (const [id, name] of Object.entries(want)) {
      const ok = got[id] === name;
      if (!ok) failed++;
      console.log(`${ok ? 'PASS' : 'FAIL'} ${lang} ${id}: ${got[id]} (want ${name})`);
    }
    if (errors.length) console.log(`      page errors: ${errors.slice(0, 3).join(' | ')}`);
    await page.close();
  }
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
