// Four-city integration, isolated profile. Run from maps root after production build.
import { readFile, writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const browser = await chromium.launch({ executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true });
const result=[];
const check=(name,ok,detail)=>{result.push({name,ok:!!ok,detail});if(!ok)throw Error(name+': '+JSON.stringify(detail));};
try {
  const page=await browser.newPage({viewport:{width:1300,height:900}}), errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1&demandWorkers=2');
  await page.waitForFunction(()=>twDemand?.view()?.round===3,undefined,{timeout:180000});
  await page.addStyleTag({content:'#debug-root {display:none}'});
  check('four local populations solve',await page.evaluate(()=>twDemand.cities().size===4));
  check('five-market corridor summary',await page.evaluate(()=>twDemand.intercity().corridors.length===15));
  await page.getByRole('button',{name:'City',exact:true}).click();
  for(const [id,name,center,market] of [['bos','Boston',[-71.0589,42.3601],3],['wdc','Washington, DC',[-77.0365,38.8977],4]]) {
    await page.getByRole('button',{name,exact:true}).click();
    await page.waitForFunction(id=>twDemand.activeCity.value===id,id);
    await page.waitForFunction(()=>!tw.map.isMoving());
    await page.getByRole('group',{name:'Demand on the map'}).getByRole('button',{name:'Homes',exact:true}).click();
    await page.waitForFunction(()=>twViews.overlay.showBubbles && twViews.overlay.discsDrawn>0);
    await page.waitForTimeout(1000);
    const cells=await page.evaluate(async()=>{const c=await twDemand.cityCells();return{n:c.n,first:Array.from(c.xy.slice(0,2)),finite:c.xy.every(Number.isFinite)};});
    check(name+' cells load',cells.n>10000&&cells.finite,cells);
    const chosen=await page.evaluate(()=>{
      const m=tw.map;
      for(let y=80;y<m.getContainer().clientHeight-20;y+=15)for(let x=30;x<m.getContainer().clientWidth-20;x+=15){const s=twViews.click(x,y,false);if(s?.kind==='commuters'&&s.cells.length>10&&[[3,0],[-3,0],[0,3],[0,-3]].every(([dx,dy])=>twViews.click(x+dx,y+dy,false)?.kind==='commuters'))return{x,y,selection:s};}return null;
    });
    check(name+' selectable homes visible',!!chosen,chosen&&{x:chosen.x,y:chosen.y,cells:chosen.selection.cells.length});
    const flow=await page.evaluate(async cells=>{const r=await twDemand.intercitySelection('home',Uint32Array.from(cells));return r&&{matched:r.matched,all:r.starts.all,places:r.starts.places.map(p=>({id:p.place,all:p.all})),finite:r.starts.zones.every(z=>[z.x,z.y,z.all,z.rail].every(Number.isFinite))};},chosen.selection.cells);
    check(name+' longer trips map into all five places',flow&&flow.matched>0.9&&flow.all>0&&flow.finite&&flow.places.some(p=>p.id===market)&&flow.places.some(p=>p.id===(market===3?4:3)),flow);
    const b=await page.locator('#map').boundingBox();
    await page.mouse.click(b.x+chosen.x,b.y+chosen.y);
    check(name+' map click opens its demand inspector',await page.evaluate(()=>tw.selection.value?.kind==='commuters'));
    await page.waitForTimeout(700);
    await page.screenshot({path:'artifacts/trainworld/T-134-'+id+'.png'});
    await page.evaluate(()=>{tw.selection.value=null;});
  }
  const before=await page.evaluate(()=>twDemand.view().version);
  const lines=await page.evaluate(async()=>{
    const t=tw, run=async op=>{const r=await t.client.edit(op);if(!r.ok)throw Error(JSON.stringify({op,r}));return r;},out=[];
    for(const [name,ll] of [['Boston',[-71.0589,42.3601]],['Washington, DC',[-77.0365,38.8977]]]) {
      const p=t.lngLatToPack(...ll);
      await run({op:'route',from:{kind:'free',x:p.x-3000,y:p.y,level:-2},to:{kind:'free',x:p.x+3000,y:p.y,level:-2},pis:[],single:false});
      const e=t.world.value.edges.at(-1);
      await run({op:'addStation',at:{kind:'node',node:e.a},platform:200,name:name+' west'});
      await run({op:'addStation',at:{kind:'node',node:e.b},platform:200,name:name+' east'});
      const line=(await run({op:'addLine',stops:[e.a,e.b],name:name+' test',colour:'#17a488'})).line;
      await run({op:'setTrainCount',line,trains:[4,4,2]});out.push(line);
    }
    await run({op:'constructAll'});return out;
  });
  await page.waitForFunction(v=>twDemand.view()?.version>v&&twDemand.view()?.round===3&&twDemand.view().lines.size===2,before,{timeout:180000});
  const riders=await page.evaluate(()=>[...twDemand.cities()].map(([id,v])=>({id,riders:v.railTripsPerDay}))); 
  check('both new cities earn local train riders', ['bos','wdc'].every(id=>riders.find(r=>r.id===id)?.riders>0),riders);
  const station=await page.evaluate(async()=>{const s=tw.world.value.save.stations.find(s=>s.name==='Washington, DC east');const r=await twDemand.stationRiders(s.id),ic=await twDemand.intercityStation(s.id);return{local:r&&r.home.total,regionalFinite:ic&&ic.starts.places.every(p=>Number.isFinite(p.all)&&Number.isFinite(p.rail))};});
  check('new-city station queries decode',station.local>=0&&station.regionalFinite,station);
  const old=await readFile('trainworld/T-007-real-nyc.save');
  await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;tw.client.post({kind:'import',bytes},[bytes]);},Array.from(old));
  await page.waitForFunction(()=>tw.world.value.save.lines.length===109&&twDemand.view()?.lines.size===109&&twDemand.view()?.round===3,undefined,{timeout:240000});
  check('existing NYC save remains playable',await page.evaluate(()=>tw.frame().kind==='equirect'&&twDemand.cities().size===4&&twDemand.cities().get('nyc').railTripsPerDay>0));
  check('no browser exceptions',errors.length===0,errors);
} finally {await writeFile('trainworld/notes/T-134-browser.json',JSON.stringify(result,null,2));await browser.close();}
console.log(JSON.stringify(result,null,2));
