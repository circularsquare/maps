import {readFile,writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try {
 const page=await browser.newPage({viewport:{width:1280,height:850}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
 await page.waitForFunction(()=>window.tw?.world.value);
 const bytes=await readFile('trainworld/T-007-real-nyc.save');
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(bytes));
 await page.waitForFunction(()=>window.twDemand?.view()?.lines.size>30&&window.twDemand.view().round>=2,null,{timeout:90000});
 const sample=()=>page.evaluate(()=>({version:window.twDemand.view().version,round:window.twDemand.view().round,split:window.twDemand.view().split,rail:window.twDemand.view().railTripsPerDay,fare:window.tw.moneyState.value.fares}));
 const results=[await sample()];
 for(const fare of [{base:0,perKm:0},{base:20,perKm:0},{base:0,perKm:10},{base:1.5,perKm:0.1}]){
  const previous=results.at(-1).version;
  const ack=await page.evaluate(async f=>window.tw.client.edit({op:'fares',...f}),fare);
  if(!ack.ok)throw Error('Fare edit rejected');
  await page.waitForFunction(v=>window.twDemand.view()?.version>v&&window.twDemand.view().round>=2,previous,{timeout:90000});
  results.push(await sample());
 }
 if(!(results[1].rail>results[2].rail&&results[1].rail>results[3].rail&&results[1].rail>results[4].rail))throw Error('Fare increases did not reduce ridership');
 // An unchanged fare should not solve again.
 const count=await page.evaluate(()=>window.twDemand.log.length);
 await page.evaluate(()=>window.tw.client.edit({op:'fares',base:1.5,perKm:0.1}));await page.waitForTimeout(500);
 if(await page.evaluate(()=>window.twDemand.log.length)!==count)throw Error('Unchanged fare unnecessarily solved again');
 if(errors.length)throw Error(errors.join('\n'));
 const result={results,errors};console.log(JSON.stringify(result));await writeFile('trainworld/notes/T-067-check.json',JSON.stringify(result,null,2));
}finally{await browser.close();}
