// T-3316: the language list -> starting currency choice. A plain function test, no browser:
//   node trainworld/checks/t3316-currency-default.mjs   (Node 22.18+ strips the .ts types itself)
import { defaultCurrency } from '../app/src/game/currency.ts';

const cases = [
  [['ja'], 'JPY'], [['ko'], 'KRW'], [['zh'], 'CNY'], [['JA'], 'JPY'],
  [['ja-JP'], 'JPY'], [['zh-Hant-TW'], 'TWD'], [['zh-HK'], 'HKD'], [['zh-CN'], 'CNY'], [['ko-KR'], 'KRW'],
  [['ja', 'en-US'], 'JPY'], [['ko', 'en-GB'], 'KRW'], [['en', 'ja'], 'JPY'], [['en-GB', 'ja'], 'GBP'],
  [['es', 'en-AU'], 'AUD'], [['es'], 'USD'], [['de', 'fr'], 'USD'], [['en'], 'USD'], [['xx-ZZ'], 'USD'],
  [['ja', 'ko'], 'JPY'], [[], 'USD'], [[''], 'USD'],
  [['zh-Hant'], 'TWD'], [['zh-Hans'], 'CNY'], [['zh-hant', 'en-US'], 'TWD'],
];
let bad = 0;
for (const [langs, want] of cases) {
  const got = defaultCurrency(langs);
  if (got !== want) { bad++; console.log('FAIL', JSON.stringify(langs), 'want', want, 'got', got); }
}
console.log(bad ? `${bad} failed` : `all ${cases.length} passed`);
process.exit(bad ? 1 : 0);
