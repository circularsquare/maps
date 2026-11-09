import { launchChrome, collectErrors } from './lib.mjs';
// T-2798: money in USD, GBP, EUR, RMB, JPY or KRW (Settings, Units and performance, "Currency"): display only, at the
// fixed rates of game/currency.ts. Formats a set of amounts in each currency and in en, ja, zh-Hans and zh-TW; USD reads
// as it always did; yen and won have no decimals; a compact figure stays short; the loss red follows the shown text;
// a typed fare reads back as typed after the game's storage rounding (New York's scale 11.6); the picker changes the
// top bar's cash and is remembered; the game's own cash never moves.
//   node trainworld/checks/t2798-currency.mjs [url]     (from the maps root; default localhost:6121)
// Prints PASS or FAIL lines and exits 1 on failure. About a minute.

const BASE = (process.argv[2] ?? 'http://localhost:6121/').replace(/\/?$/, '/');
const { ctx, close } = await launchChrome({ prefix: 'tw-t2798-', size: [1300, 900] });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${ok ? '' : detail}`); };
const errors = [];

try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.moneyState?.value, undefined, { timeout: 240000 });

  // 1. the formatters, every currency in every language
  const table = await page.evaluate(async () => {
    const f = await import('/src/game/format.ts');
    const { setDisplay } = await import('/src/game/ui.ts');
    const { chooseLanguage } = await import('/src/i18n/index.ts');
    const { CURRENCIES, CURRENCY } = await import('/src/game/currency.ts');
    const AMOUNTS = [0, 499, 500, 999, 12345, -12345, 4.2e6, -893.6e6, 22.43e6, 1.153e9, 11.363e9, 25e9, -1.5e12];
    const out = {};
    for (const lang of ['en', 'ja', 'zh-Hans', 'zh-TW']) {
      await chooseLanguage(lang);
      for (const c of CURRENCIES) {
        setDisplay('currency', c);
        out[`${lang} ${c}`] = {
          money: AMOUNTS.map(f.money), cash: AMOUNTS.map(f.moneyCash), short: AMOUNTS.map(f.moneyShort),
          millions: [0.5, 1.5, 90, 112.5, 1000, 2500].map(f.millions), range: f.millionsRange(10, 55), rangeBig: f.millionsRange(500, 5000),
          price: [0, 0.07, 2.9, 2.92, 22.5, 150].map(f.price), small: [0.04, 0.16, 0.5, 9, 40].map(f.smallMoney),
          neg: AMOUNTS.map(f.moneyNegative), negEdge: [-499 / CURRENCY[c].rate, -500.01 / CURRENCY[c].rate].map(f.moneyNegative), zero: f.money(-499 / CURRENCY[c].rate),
        };
      }
    }
    await chooseLanguage('en');
    setDisplay('currency', 'USD');
    return out;
  });
  const T = (k) => table[k];
  const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  // USD in English as it was before the setting existed
  const u = T('en USD');
  report('USD en: money, cash, short, millions as ever', eq([u.money[4], u.money[6], u.money[8], u.money[9], u.money[1], u.cash[10], u.short[7], u.millions[2], u.millions[1], u.range], ['$12k', '$4.20M', '$22.4M', '$1.15B', '$0', '$11.36B', '−$894M', '$90M', '$1.5M', '$10-55M']), JSON.stringify(u));
  report('USD en: prices and rates as ever', eq([u.price[2], u.price[0], u.small[1], u.small[3]], ['$2.90', '$0.00', '$0.16', '$9']), JSON.stringify([u.price, u.small]));
  report('USD en: loss under $500 is not red, from $500 it is', eq(u.negEdge, [false, true]) && u.zero === '$0', JSON.stringify([u.negEdge, u.zero]));

  // one fare in each currency, at the rates
  const FARE = { USD: '$2.90', GBP: '£2.32', EUR: '€2.61', CNY: '¥20.30', JPY: '¥435', KRW: '₩3,770' };
  report('a $2.90 fare in each currency (rates 1, 0.8, 0.9, 7, 150, 1300)', Object.entries(FARE).every(([c, s]) => T(`en ${c}`).price[2] === s), JSON.stringify(Object.keys(FARE).map((c) => T(`en ${c}`).price[2])));
  report('a fare off the finest step rounds to it (won to ten, yen to one)', T('en KRW').price[3] === '₩3,800' && T('en JPY').price[3] === '¥438' && T('en GBP').price[1] === '£0.06', JSON.stringify([T('en KRW').price[3], T('en JPY').price[3], T('en GBP').price[1]]));
  report('yen and won show no decimals in any price or rate of 10 or more', ['JPY', 'KRW'].every((c) => ['en', 'ja', 'zh-Hans', 'zh-TW'].every((l) => [...T(`${l} ${c}`).price, ...T(`${l} ${c}`).small.slice(2)].every((s) => !/\./.test(s)))), '');
  report('a small yen rate keeps a tenth under 10 ("¥1.5"); a yen cost from 10 is whole', T('en JPY').small[0] === '¥6' && T('en JPY').small[1] === '¥24' && T('en JPY').small[3] === '¥1,350', JSON.stringify(T('en JPY').small));
  report('compact money has its unit and stays short: yen B, T; won T, Q (en)', eq([T('en JPY').money[9], T('en JPY').cash[10], T('en KRW').short[9], T('en KRW').cash[12]], ['¥173B', '¥1.704T', '₩1.50T', '−₩1.95Q']), JSON.stringify([T('en JPY').money[9], T('en JPY').cash[10], T('en KRW').short[9], T('en KRW').cash[12]]));
  report('millions in a currency worth less: "¥13.5B" for $90M, "₩146B" for $112.5M; a pound stays in M', eq([T('en JPY').millions[2], T('en KRW').millions[3], T('en GBP').millions[2], T('en EUR').millions[1]], ['¥13.5B', '₩146B', '£72M', '€1.35M']), JSON.stringify([T('en JPY').millions[2], T('en KRW').millions[3], T('en GBP').millions[2], T('en EUR').millions[1]]));
  report('a range keeps one unit: "¥1.5-8.25B", "£8-44M"', T('en JPY').range === '¥1.5-8.25B' && T('en GBP').range === '£8-44M', JSON.stringify([T('en JPY').range, T('en GBP').range]));

  // every output short enough for the top bar and the map tags
  // (the last amount, -US$1.5 trillion, is left out: in a currency of rate 16000 it is past any game)
  const longest = Object.entries(table).flatMap(([k, v]) => [...v.cash.slice(0, 12), ...v.short.slice(0, 12), ...v.money.slice(0, 12)].map((s) => [k, s])).filter(([, s]) => [...s].length > 10);
  report('no cash, tag or panel figure is longer than 10 characters', longest.length === 0, JSON.stringify(longest.slice(0, 5)));

  // the red follows the text: a loss that shows as zero is not red, in every currency
  const redOk = Object.entries(table).every(([, v]) => v.negEdge[0] === false && v.negEdge[1] === true && /0$/.test(v.zero) && v.money.every((s, i) => (v.neg[i] ? s.startsWith('−') : !s.startsWith('−'))));
  report('negative money is red exactly when it shows a minus, in every currency and language', redOk, '');

  // the languages: 万 and 億 in ja, 万 and 亿 in zh-Hans, 萬 and 億 in zh-TW, with the symbol first
  const jp = T('ja JPY'), us = T('ja USD'), cn = T('zh-Hans CNY'), tw = T('zh-TW KRW');
  report('ja: 万 and 億 with the symbol first', /^¥\d.*[万億]$/.test(jp.money[9]) && /^\$.*[万億]$/.test(us.money[8]) && jp.millions[2].startsWith('¥') && /[億万]/.test(jp.millions[2]), JSON.stringify([jp.money, jp.millions, us.money]));
  report('zh-Hans and zh-TW: 万/亿 and 萬/億', /^¥\d.*[万亿]$/.test(cn.money[9]) && /^₩\d.*[萬億兆]$/.test(tw.money[9]), JSON.stringify([cn.money[9], tw.money[9]]));
  report('ja, zh: yen and won prices have no decimals; the symbol stays first', ['ja JPY', 'zh-Hans KRW', 'zh-TW JPY'].every((k) => T(k).price.slice(1).every((s) => /^[¥₩][\d,]+$/.test(s))), JSON.stringify(T('ja JPY').price));

  // 2. a typed fare reads back as typed, through the game's storage (curve in the baseline's units, 1e-4, times the scale 11.6)
  const drift = await page.evaluate(async () => {
    const f = await import('/src/game/format.ts');
    const { setDisplay } = await import('/src/game/ui.ts');
    const { CURRENCIES, CURRENCY, roundTo } = await import('/src/game/currency.ts');
    const S = 11.6;
    const bad = [];
    for (const c of CURRENCIES) {
      setDisplay('currency', c);
      const sp = CURRENCY[c];
      // a few hundred fares from the finest step up to about 30 USD's worth, each on the currency's own step
      for (let k = 1; k <= 400; k++) {
        const typed = roundTo(k * 0.0731 * sp.rate, sp.grain);
        const usd = f.typedPrice(typed);
        const stored = Math.round((usd / S) * 1e4) / 1e4;
        const back = stored * S;
        const want = sp.symbol + typed.toLocaleString('en-US', { minimumFractionDigits: sp.dp, maximumFractionDigits: sp.dp });
        if (f.price(back) !== want) bad.push(`${c} ${typed}: ${f.price(back)} not ${want}`);
      }
    }
    setDisplay('currency', 'USD');
    return bad;
  });
  report('typed fares read back as typed after storage in every currency (scale 11.6)', drift.length === 0, drift.slice(0, 6).join('; '));

  // 3. the picker, the top bar, the Money tab
  const faresNow = () => page.evaluate(() => JSON.stringify(window.tw.moneyState.value.fares));
  const fares0 = await faresNow();
  await page.click('button.tab[aria-label="Settings"]');
  const sel = page.locator('select[aria-label="Currency"]');
  // T-2860: ten first in relevance order, the rest by code; the default is the browser's region (en-US here)
  const opts = await sel.locator('option').evaluateAll((os) => os.map((o) => o.value));
  const rest = opts.slice(10);
  report('Settings has one Currency select: USD EUR GBP JPY CNY KRW CAD AUD INR MXN first, the rest by code, USD chosen', (await sel.count()) === 1 && eq(opts.slice(0, 10), ['USD', 'EUR', 'GBP', 'JPY', 'CNY', 'KRW', 'CAD', 'AUD', 'INR', 'MXN']) && rest.length >= 25 && eq(rest, [...rest].sort()) && (await sel.inputValue()) === 'USD', JSON.stringify(opts));
  const labels = await sel.locator('option').allTextContents();
  report('picker rows show code and symbol', labels[0] === 'USD $' && labels[3] === 'JPY ¥' && labels.every((s, i) => opts[i] === 'CNY' || s.startsWith(opts[i])), JSON.stringify(labels.slice(0, 12)));
  const dflt = await page.evaluate(async () => {
    const { defaultCurrency } = await import('/src/game/currency.ts');
    return ['es-MX', 'en-GB', 'de-DE', 'es', 'en-US', 'zh-Hant-TW', 'fr-CH', 'pt-BR', 'xx-ZZ'].map((l) => defaultCurrency([l])).concat(defaultCurrency(['es', 'en-AU']), defaultCurrency([]));
  });
  report('default currency from the browser region (es-MX MXN, en-GB GBP, de-DE EUR, plain es USD, zh-Hant-TW TWD, unknown USD)', eq(dflt, ['MXN', 'GBP', 'EUR', 'USD', 'USD', 'TWD', 'CHF', 'BRL', 'USD', 'AUD', 'USD']), JSON.stringify(dflt));
  // a high-rate currency: compact money, a fare and its step
  const big = await page.evaluate(async () => {
    const f = await import('/src/game/format.ts');
    const { setDisplay } = await import('/src/game/ui.ts');
    const { CURRENCY } = await import('/src/game/currency.ts');
    const out = {};
    for (const c of ['VND', 'IDR', 'HUF', 'MXN', 'CHF']) { setDisplay('currency', c); out[c] = { short: f.moneyShort(22.43e6), cash: f.moneyCash(1.153e9), price: f.price(2.9), step: CURRENCY[c].fareStep }; }
    setDisplay('currency', 'USD');
    return out;
  });
  report('high-rate currencies: VND ₫73,000 fare and ₫561B for $22.4M, HUF Ft 1,045 fare, MXN $52.00 (to its 0.5 grain), CHF CHF 2.32', big.VND.price === '₫73,000' && /^₫5\d\dB$/.test(big.VND.short) && big.HUF.price === 'Ft 1,045' && big.MXN.price === '$52.00' && big.CHF.price === 'CHF 2.32' && big.IDR.step === 5000, JSON.stringify(big));
  const topCash = () => page.$eval('.cash', (e) => e.textContent);
  const usdCash = await topCash();
  await sel.selectOption('JPY');
  await page.waitForTimeout(300);
  const jpyCash = await topCash();
  report('picking JPY redraws the top bar in yen', /^¥/.test(jpyCash) && /^\$/.test(usdCash), JSON.stringify([usdCash, jpyCash]));
  report('the game keeps its fares: picking a currency changes nothing it holds', (await faresNow()) === fares0, '');
  report('the pick is remembered in the stored settings', await page.evaluate(() => { try { return JSON.parse(localStorage.getItem('trainworld-ui')).display.currency === 'JPY'; } catch { return false; } }), '');
  report('the hover names the rate', (await page.$eval('select[aria-label="Currency"]', (e) => e.closest('.slider').dataset.tip ?? '')) === 'Fixed rate: 1 USD = 150 JPY', '');

  // the Money tab: type, step, in yen and won; the stepper text, and the curve in the game as typed / rate / scale
  const S = await page.evaluate(async () => { const { fareScale } = await import('/src/game/income.ts'); const { HOME_CITY, cityById } = await import('/src/game/cities.ts'); return fareScale(cityById(HOME_CITY).country, window.tw.moneyState.value.fares.scales); });
  const nodesNow = () => page.evaluate(() => JSON.stringify(window.tw.moneyState.value.fares.nodes));
  const waitChange = (before) => page.waitForFunction((b) => JSON.stringify(window.tw.moneyState.value.fares.nodes) !== b, before, { timeout: 30000 }).catch(() => {});
  const baseText = () => page.$eval('ul.fare-rows > li:nth-child(2) .stepper .v', (e) => e.textContent).catch(() => null);
  await page.click('button.tab[data-guide-tab="money"]');
  await page.waitForSelector('ul.fare-rows > li');
  const baseRow = async () => { const rows = await page.$$eval('ul.fare-rows > li', (ls) => ls.map((l) => l.querySelector('.fr-what')?.textContent ?? '')); return rows.findIndex((r) => /base/i.test(r)) + 1; };
  const bi = await baseRow();
  const baseVal = () => page.$eval(`ul.fare-rows > li:nth-child(${bi}) .stepper .v`, (e) => e.textContent);
  const typeBase = async (text) => { const b = await nodesNow(); await page.click('button[aria-label="Type base fare"]'); await page.fill('input[aria-label="base fare"]', text); await page.keyboard.press('Enter'); await waitChange(b); await page.waitForTimeout(200); };
  const step = async (label) => { const b = await nodesNow(); await page.click(`button[aria-label="${label} base fare"]`); await waitChange(b); await page.waitForTimeout(200); };
  await typeBase('435');
  report('JPY: a typed 435 shows ¥435 and is $2.90 underneath', (await baseVal()) === '¥435' && Math.abs(JSON.parse(await nodesNow())[0][1] * S - 435 / 150) < 1e-3, `${await baseVal()} ${await nodesNow()}`);
  await step('More');
  report('JPY: + steps ¥50 to ¥485', (await baseVal()) === '¥485', await baseVal());
  await step('Less'); await step('Less');
  report('JPY: two steps down, ¥385 with no drift', (await baseVal()) === '¥385', await baseVal());
  await page.click('button.tab[aria-label="Settings"]');
  await page.locator('select[aria-label="Currency"]').selectOption('KRW');
  await page.click('button.tab[data-guide-tab="money"]');
  const kb = await baseVal();
  report('KRW: the same fare (¥385 is $2.57) reads to ten won', kb === '₩3,340', kb);
  await typeBase('3770');
  report('KRW: typed 3770 reads ₩3,770', (await baseVal()) === '₩3,770', await baseVal());
  await step('More');
  report('KRW: + steps ₩500 to ₩4,270', (await baseVal()) === '₩4,270', await baseVal());
  await page.click('button.tab[aria-label="Settings"]');
  await page.locator('select[aria-label="Currency"]').selectOption('EUR');
  await page.click('button.tab[data-guide-tab="money"]');
  await typeBase('2.61');
  report('EUR: typed 2.61 reads €2.61, then + steps €0.25', (await baseVal()) === '€2.61' && (await (async () => { await step('More'); return baseVal(); })()) === '€2.86', await baseVal());
  const ticks = await page.$$eval('svg.fare-graph .fg-label', (ts) => ts.map((x) => x.textContent));
  report('the fare graph axis is in euros', ticks.some((s) => /^€/.test(s)) && !ticks.some((s) => /\$/.test(s)), JSON.stringify(ticks));

  // 4. back to dollars: the game's fares are what they were, and nothing shows a stray symbol
  await page.click('button.tab[aria-label="Settings"]');
  await page.locator('select[aria-label="Currency"]').selectOption('USD');
  await page.click('button.tab[data-guide-tab="money"]');
  const usdBase = await baseVal();
  report('back in USD the base fare reads as dollars and cents', /^\$\d+\.\d\d$/.test(usdBase), usdBase);

  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
const failed = results.filter((x) => !x).length;
console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
