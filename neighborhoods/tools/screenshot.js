/*
 * Screenshot the browser WITH THE MAP DRAWN, and run JS against it first.
 *
 * WHY THIS EXISTS. The map half of index.html used to be unverifiable here: headless
 * Chrome fell back to no WebGL, MapLibre never initialised, and every screenshot came
 * back with a black rectangle where the map should be. Two flags fix it —
 *
 *     --headless=new --enable-unsafe-swiftshader
 *
 * which gives software WebGL, and MapLibre renders normally on it, slowly.
 *
 * The second half of the problem is timing. `chrome --screenshot --virtual-time-budget`
 * captures while MapLibre still has tiles in flight, so the map is black even with WebGL
 * working; virtual time and an async render loop do not mix. So this drives an already-
 * running Chrome over the DevTools protocol instead and waits in REAL time. Node's
 * built-in WebSocket, so there is nothing to install.
 *
 * Start Chrome once (its own --user-data-dir, so it is a separate instance and none of
 * Anita's windows are involved):
 *
 *   "C:/Program Files/Google/Chrome/Application/chrome.exe" --headless=new \
 *     --enable-unsafe-swiftshader --remote-debugging-port=9222 \
 *     --user-data-dir=<some temp dir> --window-size=1400,900 about:blank
 *
 * Then, with `python serve.py` running:
 *
 *   node tools/screenshot.js http://localhost:8766/ out.png
 *   node tools/screenshot.js http://localhost:8766/ out.png 13000 probe.js
 *   node tools/screenshot.js http://localhost:8766/ side.png 12000 "" 360 900 2
 *
 * `probe.js` is evaluated in the page after the wait and its value is printed, so it can
 * both DRIVE the map (`setTheme`, `selectCity`, `map.jumpTo`) and ASSERT against it. An
 * async IIFE returning a string works; the value is awaited. Console output and uncaught
 * exceptions from the page are printed after the capture, which is the only place a
 * MapLibre layer error ever appears.
 *
 * localStorage survives in the profile directory, so the theme persists between runs.
 * A probe that cares should call `setTheme('dark')` rather than clicking the button.
 */
const fs = require('fs');

const [url, out, waitMs = '12000', jsFile, w = '1400', h = '900', scale = '1'] =
  process.argv.slice(2);
const PORT = 9222;

if (!url || !out) {
  console.error('usage: node tools/screenshot.js <url> <out.png> [waitMs] [jsFile] [w h scale]');
  process.exit(2);
}

const rpc = (ws, id, method, params) =>
  new Promise((resolve, reject) => {
    const onMsg = ev => {
      const m = JSON.parse(ev.data);
      if (m.id !== id) return;
      ws.removeEventListener('message', onMsg);
      m.error ? reject(new Error(method + ': ' + m.error.message)) : resolve(m.result);
    };
    ws.addEventListener('message', onMsg);
    ws.send(JSON.stringify({id, method, params}));
  });

const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
  const page = targets.find(t => t.type === 'page');
  if (!page) throw new Error('no page target; is chrome running with --remote-debugging-port?');

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  const logs = [];
  ws.addEventListener('message', ev => {
    const m = JSON.parse(ev.data);
    if (m.method === 'Runtime.consoleAPICalled')
      logs.push(m.params.type + ': ' + m.params.args.map(a => a.value ?? a.description).join(' '));
    if (m.method === 'Runtime.exceptionThrown')
      logs.push('EXCEPTION: ' + (m.params.exceptionDetails.exception?.description ||
                                 m.params.exceptionDetails.text));
  });

  let id = 1;
  await rpc(ws, id++, 'Runtime.enable', {});
  await rpc(ws, id++, 'Page.enable', {});
  await rpc(ws, id++, 'Emulation.setDeviceMetricsOverride',
            {width: +w, height: +h, deviceScaleFactor: +scale, mobile: false});
  await rpc(ws, id++, 'Page.navigate', {url});
  await sleep(+waitMs);

  if (jsFile) {
    const r = await rpc(ws, id++, 'Runtime.evaluate',
      {expression: fs.readFileSync(jsFile, 'utf8'), awaitPromise: true, returnByValue: true});
    console.log('eval ->', JSON.stringify(r.result?.value ?? r.result));
    // The probe usually moves the camera, and the move needs its own tiles.
    await sleep(6000);
  }

  const shot = await rpc(ws, id++, 'Page.captureScreenshot', {format: 'png'});
  fs.writeFileSync(out, Buffer.from(shot.data, 'base64'));
  console.log('wrote', out);
  if (logs.length) console.log('console:\n  ' + logs.slice(0, 40).join('\n  '));
  ws.close();
})().catch(e => { console.error(e.message); process.exit(1); });
