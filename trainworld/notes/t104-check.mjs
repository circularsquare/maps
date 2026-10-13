import {readFile,writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try {
 const page=await browser.newPage({viewport:{width:688,height:953},acceptDownloads:true}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix);
 const bytes=await readFile('trainworld/T-007-real-nyc.save');
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(bytes));
 await page.waitForFunction(()=>window.tw.world.value.save.lines.length>80);
 const initial=await page.evaluate(()=>{
  const tw=window.tw,l=tw.world.value.save.lines.find(l=>l.name==='njt MNE');if(!l)throw Error('Fixture line missing');window.testLine=l.id;tw.select({kind:'line',line:l.id});
  return {line:l.name,tph:l.tph,counts:tw.world.value.lineStats[l.id].trainsNeeded};
 });await page.waitForTimeout(100);
 if((await page.locator('.sched').innerText()).includes('Trains an hour'))throw Error('Old schedule input remains');
 const ack=await page.evaluate(()=>window.tw.client.edit({op:'setTrainCount',line:Number(window.testLine),trains:[4,2,0]}));
 if(!ack.ok)throw Error(JSON.stringify(ack));
 const sample=()=>page.evaluate(()=>{const w=window.tw.world.value,id=window.testLine,l=w.save.lines.find(l=>l.id===id),s=w.lineStats[id];return {counts:s.trainsNeeded,tph:l.tph,roundTrip:s.roundTripByLevel,residual:['high','medium','low'].map((d,i)=>l.tph[d]*s.roundTripByLevel[i]/3600-s.trainsNeeded[i])};});
 const chosen=await sample();if(JSON.stringify(chosen.counts)!=='[4,2,0]'||chosen.residual.some(r=>Math.abs(r)>0.01))throw Error(JSON.stringify(chosen));
 const text=await page.locator('.sched').innerText();if(/\d\.\d{5}/.test(text))throw Error('Raw float schedule text');
 const version=await page.evaluate(()=>window.tw.world.value.version);
 await page.locator('.sched tr').nth(1).getByRole('button',{name:'More trains on the entire line'}).click();
 await page.waitForFunction(v=>window.tw.world.value.version>v,version);
 const stepped=await sample();if(stepped.counts[0]!==5)throw Error('Stepper did not add one train');
 const downloading=page.waitForEvent('download');await page.evaluate(()=>window.tw.client.post({kind:'export'}));const download=await downloading;
 const saved=await readFile(await download.path());
 await page.evaluate(()=>window.tw.client.edit({op:'setTrainCount',line:Number(window.testLine),trains:[1,1,1]}));
 const beforeImport=await page.evaluate(()=>window.tw.world.value.version);
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(saved));
 await page.waitForFunction(v=>window.tw.world.value.version>v,beforeImport);
 const restored=await sample();if(JSON.stringify(restored.counts)!=='[5,2,0]')throw Error('Count allocation not saved');
 await page.evaluate(()=>{document.querySelector('#debug').style.display='none';window.tw.select({kind:'line',line:window.testLine});});await page.waitForTimeout(100);
 await page.screenshot({path:'artifacts/trainworld/T-104-schedule.png'});
 const layout=await page.locator('#inspector').evaluate(e=>({width:e.clientWidth,scrollWidth:e.scrollWidth}));if(layout.scrollWidth>layout.width+1)throw Error('Inspector overflows');
 if(errors.length)throw Error(errors.join('\n'));
 const result={initial,chosen,stepped,restored,layout,errors};console.log(JSON.stringify(result));await writeFile('artifacts/trainworld/notes-outputs/T-104-check.json',JSON.stringify(result,null,2));
}finally{await browser.close();}
