import {readFile,writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try {
 const page=await browser.newPage({viewport:{width:1100,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
 await page.waitForFunction(()=>window.tw?.world.value&&window.twDemand?.client);
 const bytes=await readFile('trainworld/T-007-real-nyc.save');
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(bytes));
 await page.waitForFunction(()=>window.twDemand.view()?.round===3&&window.twDemand.view().lines.size>100,undefined,{timeout:180000});
 const measure=()=>page.evaluate(()=>{
  const t=window.twDemand,s=t.log.at(-1),d=t.view();
  return {version:d.version,lines:d.lines.size,stations:window.tw.world.value.save.stations.length,cities:[...t.cities()].map(([id,v])=>({id,rail:v.railTripsPerDay,split:v.split})),market:t.intercity(),firstMs:s.firstAt-s.sentAt,refinedMs:s.refinedAt-s.sentAt,setupMs:s.setupMs,periodMs:s.periodMs,memory:s.memBytes,workers:t.start.workers.length,status:t.client.status};
 });
 const legacy=await measure();
 const fixture=await page.evaluate(async()=>{
  const t=window.tw,run=async op=>{const a=await t.client.edit(op);if(!a.ok)throw Error(JSON.stringify({op,a}));return a;};
  const R=6371008.8,rad=Math.PI/180,x=R*(-75.1636+73.985)*rad*Math.cos(40.758*rad),y=R*(39.9526-40.758)*rad;
  await run({op:'route',from:{kind:'free',x,y:y-2000,level:-2},to:{kind:'free',x,y:y+2000,level:-2},pis:[],single:false});
  const edge=t.world.value.edges.at(-1);
  await run({op:'addStation',at:{kind:'node',node:edge.a},platform:200,name:'Philadelphia south'});
  await run({op:'addStation',at:{kind:'node',node:edge.b},platform:200,name:'Philadelphia north'});
  const line=(await run({op:'addLine',stops:[edge.a,edge.b],name:'Philadelphia check',colour:'#17a488'})).line;
  await run({op:'setTrainCount',line,trains:[4,4,2]});await run({op:'constructAll'});return {line,nodes:[edge.a,edge.b]};
 });
 await page.waitForFunction(v=>window.twDemand.view()?.version>v&&window.twDemand.view().round===3,legacy.version,{timeout:180000});
 const combined=await measure();
 if(combined.cities.some(c=>c.rail<=0)||combined.lines!==legacy.lines+1||combined.workers>6)throw Error('Combined real network incorrect '+JSON.stringify(combined));
 // the City tab is gone (T-220): search goes to a city
 await page.getByRole('button',{name:'Search stations, lines and places'}).click();await page.getByRole('combobox',{name:'Search stations, lines and places'}).fill('Philadelphia');await page.keyboard.press('Enter');
 await page.waitForFunction(()=>window.twDemand.activeCity.value==='phl'&&Math.abs(window.tw.map.getCenter().lng+75.1636)<.001);
 const query=await page.evaluate(async()=>{
  const c=await window.twDemand.cityCells(),m=await window.twDemand.cellModes(),s=await window.twDemand.stationRiders(String(window.tw.world.value.save.stations.find(s=>s.name==='Philadelphia north').id));
  return {cells:c.n,modes:m?.homeRail.length,station:s&&{home:s.home.total,work:s.work.total,maxCell:Math.max(...s.home.cells)},worldXY:Array.from(c.xy.slice(0,2))};
 });
 if(query.cells!==110578||query.modes!==query.cells||!query.station||query.station.home<=0||query.station.maxCell>=query.cells)throw Error('Philadelphia catchment failed '+JSON.stringify(query));
 await page.evaluate(()=>{window.tw.display.value={...window.tw.display.value,commuters:true,commuterEnd:'home'};document.querySelector('#debug').style.display='none';});
 await page.waitForTimeout(1500);await page.screenshot({path:'artifacts/trainworld/T-116-philadelphia.png'});
 const gl=await page.evaluate(()=>window.tw.renderer.gl.getError());if(errors.length||gl)throw Error(JSON.stringify({errors,gl}));
 const result={legacy,combined,fixture,query,errors,gl};console.log(JSON.stringify(result));await writeFile('artifacts/trainworld/notes-outputs/T-116-real-browser.json',JSON.stringify(result,null,2));
}finally{await browser.close();}
