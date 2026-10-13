import {readFile,writeFile} from 'node:fs/promises';
import {gunzipSync,gzipSync} from 'node:zlib';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
const waitSolve=async page=>page.waitForFunction(()=>window.twDemand?.view()?.round===3,undefined,{timeout:120000});
try {
 const page=await browser.newPage({viewport:{width:1100,height:900},acceptDownloads:true}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1&demandWorkers=2');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix);
 await waitSolve(page);
 const empty=await page.evaluate(()=>({cities:[...window.twDemand.cities()].map(([id,d])=>[id,d.tripsPerDay,d.railTripsPerDay]),market:window.twDemand.intercity(),origin:window.tw.world.value.save.city}));
 if(empty.cities.length!==2||empty.cities.some(c=>c[1]<=0||c[2]!==0)||empty.market.tripsPerDay<=0||empty.market.railTripsPerDay!==0)throw Error('Empty regional solve failed '+JSON.stringify(empty));
 // the City tab is gone (T-220): search goes to a city
 const goCity=async name=>{await page.getByRole('button',{name:'Search stations, lines and places'}).click();await page.getByRole('combobox',{name:'Search stations, lines and places'}).fill(name);await page.keyboard.press('Enter');};
 await goCity('Philadelphia');
 await page.waitForFunction(()=>window.twDemand.activeCity.value==='phl'&&Math.abs(window.tw.map.getCenter().lng+75.1636)<.001);
 if(await page.locator('#topbar .city').innerText()!=='Philadelphia')throw Error('Focused city header wrong');
 const phl=await page.evaluate(async()=>{const c=await window.twDemand.cityCells(),m=await window.twDemand.cellModes();return {n:c.n,xy:Array.from(c.xy.slice(0,2)),modes:m?.homeRail.length};});
 if(phl.n!==110578||phl.modes!==phl.n||phl.xy[0]>-30000)throw Error('Philadelphia cells not transformed '+JSON.stringify(phl));
 await goCity('New York');
 await page.waitForFunction(()=>window.twDemand.activeCity.value==='nyc'&&Math.abs(window.tw.map.getCenter().lng+73.985)<.001);
 const nyc=await page.evaluate(async()=>{const c=await window.twDemand.cityCells();return {n:c.n,xy:Array.from(c.xy.slice(0,2))};});
 if(nyc.n===phl.n||nyc.n<200000)throw Error('City cells retained old focus');
 // Supply capital in this isolated acceptance fixture, preserving all real save inputs.
 const downloading=page.waitForEvent('download');await page.evaluate(()=>window.tw.client.post({kind:'export'}));
 const raw=gunzipSync(await readFile(await (await downloading).path())),len=raw.readUInt32LE(4),header=JSON.parse(raw.subarray(8,8+len));
 header.cash=100000;const json=Buffer.from(JSON.stringify(header)),prefix=Buffer.alloc(8);prefix.write('TWG1');prefix.writeUInt32LE(json.length,4);
 const funded=gzipSync(Buffer.concat([prefix,json,raw.subarray(8+len)]));
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(funded));
 await page.waitForFunction(()=>window.tw.moneyState.value.cash===100000);
 const fixture=await page.evaluate(async()=>{
  const tw=window.tw,run=async op=>{const a=await tw.client.edit(op);if(!a.ok)throw Error(JSON.stringify({op,a}));return a;};
  const R=6371008.8,rad=Math.PI/180,point=(lon,lat)=>({x:R*(lon+73.985)*rad*Math.cos(40.758*rad),y:R*(lat-40.758)*rad});
  const end=point(-75.1636,39.9526);
  await run({op:'route',from:{kind:'free',x:0,y:0,level:-2},to:{kind:'free',...end,level:-2},pis:[],single:false});
  const original=tw.world.value.edges[0],nodes=[original.a];
  for(const [t,name]of[[.015,'New York Central'],[.625,'Trenton'],[.985,'Philadelphia Central']]) {
   const x=end.x*t,y=end.y*t;
   const edge=tw.world.value.edges.find(e=>{const a=tw.world.value.nodes.get(e.a),b=tw.world.value.nodes.get(e.b);return x<=Math.max(a.x,b.x)+1&&x>=Math.min(a.x,b.x)-1;});
   await run({op:'addStation',at:{kind:'edge',edge:edge.id,x,y},platform:200,name});
   nodes.push(Number(tw.world.value.save.stations.find(s=>s.name===name).id));
  }
  nodes.push(original.b);
  for(const [node,name]of[[nodes[0],'NYC local'],[nodes[4],'Philadelphia local']])await run({op:'addStation',at:{kind:'node',node},platform:200,name});
  const left=(await run({op:'addLine',stops:nodes.slice(0,2),name:'NYC feeder',colour:'#d7263d'})).line;
  const express=(await run({op:'addLine',stops:nodes.slice(1,4),name:'Northeast',colour:'#2277bb'})).line;
  const right=(await run({op:'addLine',stops:nodes.slice(3),name:'Philadelphia feeder',colour:'#17a488'})).line;
  for(const [line,trains]of[[left,[4,4,2]],[express,[12,12,6]],[right,[4,4,2]]])await run({op:'setTrainCount',line,trains});
  await run({op:'constructAll'});window.fixture={nodes,left,express,right,end};return window.fixture;
 });
 const version=await page.evaluate(()=>window.twDemand.view()?.version??0);
 await page.waitForFunction(v=>window.twDemand.view()?.version>v&&window.twDemand.view().round===3,version,{timeout:120000});
 const summarize=()=>page.evaluate(()=>{
  const d=window.twDemand.view(),i=window.twDemand.intercity(),w=window.tw.world.value;
  return {version:d.version,round:d.round,cities:[...window.twDemand.cities()].map(([id,d])=>({id,trips:d.tripsPerDay,rail:d.railTripsPerDay,split:d.split})),market:i,
   lines:[...d.lines].map(([id,l])=>({id,boardings:l.ridersPerDay,intercity:l.intercityRidersPerDay,loads:l.loads.map(a=>Array.from(a)),counts:w.lineStats[id].trainsNeeded,rt:w.lineStats[id].roundTripByLevel,tph:w.save.lines.find(l=>l.id===id).tph})),fares:window.tw.income.value.faresPerDay,setup:window.twDemand.log.at(-1).setupMs,memory:window.twDemand.log.at(-1).memBytes};
 });
 const good=await summarize();
 if(good.cities.some(c=>c.rail<=0)||good.market.railTripsPerDay<=0||good.lines.some(l=>l.intercity<=0)||good.fares<=0)throw Error('Two cities/feeder intercity riders absent '+JSON.stringify(good));
 if(good.lines.some(l=>l.loads.flat().some(v=>!Number.isFinite(v)||v<0)))throw Error('Invalid loads');
 if(good.lines.some(l=>['high','medium','low'].some((k,i)=>Math.abs(l.tph[k]*l.rt[i]/3600-l.counts[i])>.01)))throw Error('Count convergence failed');
 await page.evaluate(()=>window.tw.client.edit({op:'setTrainCount',line:window.fixture.express,trains:[0,0,0]}));
 await page.waitForFunction(v=>window.twDemand.view()?.version>v&&window.twDemand.view().round===3,good.version,{timeout:120000});
 const off=await summarize();
 const cross=m=>m.corridors.filter(c=>c.from!==c.to).reduce((a,c)=>a+c.rail,0);
 if(cross(off.market)>=cross(good.market)*.1)throw Error('Disabled intercity line still carries trips');
 await page.evaluate(()=>window.tw.client.edit({op:'setTrainCount',line:window.fixture.express,trains:[12,12,6]}));
 await page.waitForFunction(v=>window.twDemand.view()?.version>v&&window.twDemand.view().round===3,off.version,{timeout:120000});
 const restored=await summarize();
 if(Math.abs(restored.market.railTripsPerDay-good.market.railTripsPerDay)>1)throw Error('Deterministic restored service differs');
 const download2=page.waitForEvent('download');await page.evaluate(()=>window.tw.client.post({kind:'export'}));
 const saved=await readFile(await (await download2).path()),savedRaw=gunzipSync(saved),savedHeader=JSON.parse(savedRaw.subarray(8,8+savedRaw.readUInt32LE(4)));
 if(savedHeader.region!=='northeast'||savedHeader.cities.length!==2)throw Error('Region metadata absent');
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(saved));
 await page.waitForFunction(()=>window.tw.world.value.save.lines.length===3&&window.tw.world.value.save.stations.length===5);
 // the City tab's Whole region button is gone (T-220): zoom out to every city
 await page.evaluate(()=>window.tw.map.fitBounds([[-77.8,38.4],[-70.7,42.6]],{padding:40,duration:0}));
 await page.waitForTimeout(1000);await page.evaluate(()=>document.querySelector('#debug').style.display='none');
 await page.screenshot({path:'artifacts/trainworld/T-116-northeast.png'});
 const layout=await page.locator('#tabs').evaluate(e=>({width:e.clientWidth,scrollWidth:e.scrollWidth})).catch(()=>null);
 if(errors.length)throw Error(errors.join('\n'));
 const result={empty,phl,nyc,fixture,good,off,restored,savedHeader,layout,errors};
 console.log(JSON.stringify(result));await writeFile('artifacts/trainworld/notes-outputs/T-116-browser.json',JSON.stringify(result,null,2));
}finally{await browser.close();}
