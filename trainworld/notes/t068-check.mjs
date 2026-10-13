import { writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
const context = await browser.newContext({ viewport: { width: 688, height: 953 } });
const errors = [];
const check = (v, message) => { if (!v) throw Error(message); };
async function setup(page, fresh = false) {
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('http://localhost:8800/trainworld/?debug=1&paused=1' + (fresh ? '&new=1' : ''));
  await page.waitForFunction(() => window.tw?.world.value);
  await page.evaluate(() => {
    window.slotState = null;
    window.tw.client.worker.addEventListener('message', e => { if (e.data.kind === 'saveSlots') window.slotState = e.data; });
    window.tw.client.post({ kind: 'saveSlot', action: 'refresh' });
  });
  await page.waitForFunction(() => window.slotState);
}
async function state(page) { return page.evaluate(() => window.slotState); }
async function action(page, action, key, name) {
  await page.evaluate(args => { window.slotState = null; window.tw.client.post({ kind: 'saveSlot', ...args }); }, { action, key, name });
  await page.waitForFunction(() => window.slotState);
  return state(page);
}
async function fare(page, base) {
  const a = await page.evaluate(base => window.tw.client.edit({ op: 'fares', base, perKm: 0.1 }), base);
  check(a.ok, 'Fare fixture edit failed');
}
async function records(page) {
  return page.evaluate(async () => {
    const db = await new Promise((resolve, reject) => { const r = indexedDB.open('anitabuilder', 1); r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); });
    const s = db.transaction('saves', 'readonly').objectStore('saves');
    const req = r => new Promise((resolve, reject) => { r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); });
    const [keys, data] = await Promise.all([req(s.getAllKeys()), req(s.getAll())]);
    const out = {};
    for (let i = 0; i < keys.length; i++) {
      const raw = new Uint8Array(await new Response(new Blob([data[i].bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      const n = new DataView(raw.buffer).getUint32(4, true);
      out[keys[i]] = { name: data[i].name, at: data[i].at, header: JSON.parse(new TextDecoder().decode(raw.slice(8, 8+n))) };
    }
    db.close(); return out;
  });
}
async function flush(page, key, base) {
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction(async ({key, base}) => {
    const d = await new Promise(resolve => { const o = indexedDB.open('anitabuilder', 1); o.onsuccess = () => { const r = o.result.transaction('saves').objectStore('saves').get(key); r.onsuccess = () => resolve(r.result); }; });
    if (!d) return false;
    const raw = new Uint8Array(await new Response(new Blob([d.bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    const n = new DataView(raw.buffer).getUint32(4,true);
    return JSON.parse(new TextDecoder().decode(raw.slice(8,8+n))).fares.base === base;
  }, {key,base});
}
try {
  const first = await context.newPage(); await setup(first, true);
  check((await state(first)).writable, 'First tab must own original slot');
  await fare(first, 1.11); await flush(first, 'autosave', 1.11);
  const second = await context.newPage(); await setup(second);
  check(!(await state(second)).writable, 'Second tab must not overwrite original');
  check(await second.locator('.save-notice').isVisible(), 'Read-only autosave warning must remain visible outside Settings');
  await fare(second, 2.22); await second.evaluate(() => window.tw.client.post({kind:'flush'}));
  await second.waitForTimeout(100);
  check((await records(first)).autosave.header.fares.base === 1.11, 'Second tab overwrote original game');
  const blocked = await action(second,'open','missing'); check(!!blocked.error,'Invalid slot should report an error');
  const alpha = (await action(first,'create',undefined,'Alpha')).active;
  const beta = (await action(second,'create',undefined,'Beta')).active;
  check(alpha !== beta && alpha !== 'autosave', 'Distinct slots not created');
  check((await state(first)).writable && (await state(second)).writable,'Different slots must save concurrently');
  await fare(first,3.33); await flush(first,alpha,3.33);
  await fare(second,4.44); await flush(second,beta,4.44);
  let saved = await records(first);
  check(saved.autosave.header.fares.base === 1.11 && saved[alpha].header.fares.base === 3.33 && saved[beta].header.fares.base === 4.44,'Slots are not isolated');
  const refusal = await action(first,'delete',beta);
  check(!!refusal.error && !!(await records(first))[beta],'Must not delete a game another tab owns');
  await action(first,'rename',alpha,'Renamed Alpha');
  check((await records(first))[alpha].name === 'Renamed Alpha','Rename not persisted');
  await second.waitForFunction(key => window.slotState.slots.some(s => s.key === key && s.name === 'Renamed Alpha'),alpha);
  const opened = await action(second,'open',alpha);
  check(opened.active === alpha && !opened.writable,'Opening an owned slot must be read-only');
  check(await second.evaluate(() => window.tw.moneyState.value.fares.base) === 3.33,'Opened slot did not restore game');
  await fare(second,9.99);
  const protectedOpen = await action(second,'open',beta);
  check(!!protectedOpen.error && protectedOpen.active === alpha,'Switching must not discard unsaved read-only game');
  const rescued = await action(second,'create',undefined,'Rescued copy');
  check(rescued.writable && rescued.active !== alpha,'Read-only game must be rescuable to new slot');
  check((await records(first))[rescued.active].header.fares.base === 9.99,'Rescued copy lost changes');
  await action(second,'open',alpha);
  await first.close(); await second.reload(); await second.waitForFunction(() => window.tw?.world.value);
  await second.evaluate(() => { window.slotState=null; window.tw.client.worker.addEventListener('message',e=>{if(e.data.kind==='saveSlots')window.slotState=e.data}); window.tw.client.post({kind:'saveSlot',action:'refresh'}); });
  await second.waitForFunction(() => window.slotState);
  check((await state(second)).writable && (await state(second)).active === alpha,'Closed tab did not release slot lock / remembered slot was not restored');
  check(await second.evaluate(() => window.tw.moneyState.value.fares.base) === 3.33,'Reload restored stale read-only changes');
  await action(second,'delete',beta); check(!(await records(second))[beta],'Inactive slot deletion failed');
  await second.getByRole('button',{name:'Settings',exact:true}).click().catch(()=>{});
  await second.evaluate(()=>{document.querySelector('#debug').style.display='none';});
  await second.locator('.save-slot-new').first().scrollIntoViewIfNeeded();
  const pane = await second.locator('.pane').evaluate(e=>({width:e.clientWidth,scrollWidth:e.scrollWidth}));
  check(pane.scrollWidth <= pane.width + 1, 'Save slot UI overflows narrow pane');
  await second.screenshot({path:'artifacts/trainworld/T-068-slots.png'});
  saved = await records(second);
  const failureContext = await browser.newContext({viewport:{width:688,height:953}});
  await failureContext.route('**/assets/clock.worker-*.js',async route=>{
    const response=await route.fetch();
    await route.fulfill({response,body:'IDBObjectStore.prototype.put=function(){throw new DOMException("Quota exceeded","QuotaExceededError")};\n'+await response.text()});
  });
  const failurePage=await failureContext.newPage();await setup(failurePage,true);
  await fare(failurePage,7.77);await failurePage.evaluate(()=>window.tw.client.post({kind:'flush'}));
  await failurePage.waitForFunction(()=>window.slotState && !window.slotState.writable && window.slotState.reason.includes('could not save'));
  check(await failurePage.locator('.save-notice').isVisible(),'Quota failure notice must remain visible');
  await action(failurePage,'refresh');
  check(!(await state(failurePage)).writable && (await state(failurePage)).reason.includes('could not save'),'Refreshing metadata must not clear a write-failure notice');
  const downloading=failurePage.waitForEvent('download');await failurePage.evaluate(()=>window.tw.client.post({kind:'export'}));await downloading;
  await failureContext.close();
  check(errors.length===0,errors.join('\n'));
  const result={alpha,beta,rescued:rescued.active,slots:Object.fromEntries(Object.entries(saved).map(([k,v])=>[k,{name:v.name,base:v.header.fares.base}])),errors};
  await writeFile('artifacts/trainworld/notes-outputs/T-068-check.json',JSON.stringify(result,null,2)); console.log(JSON.stringify(result));
} finally { await browser.close(); }
