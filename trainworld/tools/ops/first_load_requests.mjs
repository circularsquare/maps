// T-3243: requests and bytes of one live first load of /trainsworld/?new=1, per host, and which bytes came from
// GitHub Pages through Cloudflare uncached (cf-cache-status DYNAMIC). Feeds R2_READS_PER_PLAY and
// PAGES_BYTES_PER_NEW_BROWSER in check_resources.py. market.anita.garden is blocked, so the run adds nothing to
// the play stats. Fetches about 25 MB and makes about 33 R2 reads.
//   OUT=<scratch folder> node trainworld/tools/ops/first_load_requests.mjs
import { launchChrome } from '../../checks/lib.mjs';
const OUT = process.env.OUT ?? '.';
const URL_ = process.env.URL ?? 'https://anita.garden/trainsworld/?new=1';
const { ctx, close } = await launchChrome({ prefix: OUT + '/chrome-', size: [1300, 900], args: ['--enable-unsafe-swiftshader'] });
const hosts = {};
const add = (h, k, n) => { hosts[h] ??= { requests: 0, bytes: 0, ranges: 0 }; hosts[h][k] += n; };
await ctx.route('**/*', (route) => new URL(route.request().url()).hostname === 'market.anita.garden' ? route.abort() : route.continue());
const page = await ctx.newPage();
page.on('request', (r) => {
  const h = new URL(r.url()).hostname;
  add(h, 'requests', 1);
  if (r.headers()['range']) add(h, 'ranges', 1);
});
page.on('response', async (r) => {
  const h = new URL(r.url()).hostname;
  try {
    const s = await r.request().sizes();
    add(h, 'bytes', s.responseBodySize + s.responseHeadersSize);
    if (h === 'anita.garden') {
      const c = (await r.allHeaders())['cf-cache-status'] ?? 'none';
      add(`${h} ${c}`, 'bytes', s.responseBodySize);
      add(`${h} ${c}`, 'requests', 1);
    }
  } catch {}
});
try {
  await page.goto(URL_, { waitUntil: 'load', timeout: 120000 });
  await page.waitForTimeout(25000); // counts stopped changing after about 20 s in T-3243's runs
  console.log(JSON.stringify(hosts, null, 1));
} finally {
  await close();
}
