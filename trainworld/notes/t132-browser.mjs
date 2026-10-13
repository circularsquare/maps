// Isolated profile; captures stay outside trainworld. Run from maps root.
import { readFile, writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
const result = [];
const check = (name, ok, detail) => { result.push({name,ok,detail}); if (!ok) throw new Error(name + ': ' + JSON.stringify(detail)); };
try {
  const page = await browser.newPage({ viewport: { width: 1300, height: 900 } });
  await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1&demandWorkers=3');
  await page.waitForFunction(() => window.tw?.world.value && window.twViews);
  await page.addStyleTag({content:'#debug-root {display:none}'});
  const group = page.getByRole('group', { name: 'Demand on the map' });
  for (const [label, end] of [['Homes','home'], ['Destinations','work'], ['None',null]]) {
    await group.getByRole('button', {name:label, exact:true}).click();
    const state = await page.evaluate(() => ({display:tw.display.value, selection:tw.selection.value}));
    check(label, state.display.commuters === !!end && (!end || state.display.commuterEnd === end) && !state.selection, state);
  }
  await group.getByRole('button', {name:'Destinations',exact:true}).click();
  await page.reload();
  await page.waitForFunction(() => window.tw?.world.value);
  await page.addStyleTag({content:'#debug-root {display:none}'});
  check('persisted demand view', await group.getByRole('button',{name:'Destinations',exact:true}).getAttribute('aria-pressed') === 'true');
  await group.getByRole('button', {name:'None',exact:true}).click();
  await page.waitForFunction(() => tw.map.getLayer('tw-city-boundary-nyc') && tw.map.getLayer('tw-city-boundary-phl'));
  await page.evaluate(() => tw.map.jumpTo({center:[-74.65,40.6],zoom:7.6}));
  await page.waitForFunction(() => tw.map.isSourceLoaded('tw-city-boundary-nyc') && tw.map.isSourceLoaded('tw-city-boundary-phl'));
  const layers = await page.evaluate(() => ['nyc','phl'].map(id => {
    const key='tw-city-boundary-'+id, l=tw.map.getLayer(key); return {id,maxzoom:l.maxzoom,opacity:tw.map.getPaintProperty(key,'line-opacity'),source:tw.map.getSource(key).serialize().data};
  }));
  check('both exact boundary sources and close-zoom hiding', layers.every(l => l.maxzoom===10.5 && (typeof l.source === 'string' ? l.source.endsWith(l.id+'.boundary.geojson') : l.source.features?.length>0)), layers.map(l=>({...l,source:typeof l.source === 'string'?l.source:l.source.type})));
  await page.screenshot({path:'artifacts/trainworld/T-131-regional.png'});
  await page.setViewportSize({width:688,height:900});
  await group.getByRole('button',{name:'Homes',exact:true}).click();
  const bounds=await group.boundingBox(), mapBounds=await page.locator('#map').boundingBox();
  check('selector inside narrow map upper left', bounds.x>=mapBounds.x && bounds.y>=mapBounds.y && bounds.x+bounds.width<=mapBounds.x+mapBounds.width && bounds.y-mapBounds.y<12,{bounds,mapBounds});
  await page.screenshot({path:'artifacts/trainworld/T-132-narrow.png'});
  const restored=await page.evaluate(async()=>{
    const m=tw.map, style=m.getStyle();
    await new Promise(resolve=> { m.once('style.load',resolve); m.setStyle(style, {diff:false}); });
    return ['nyc','phl'].every(id=>!!m.getLayer('tw-city-boundary-'+id));
  });
  check('boundaries survive style reload',restored);
  await page.setViewportSize({width:1300,height:900});
  const bytes=await readFile('trainworld/T-007-real-nyc.save');
  await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;tw.client.post({kind:'import',bytes},[bytes]);},Array.from(bytes));
  await page.waitForFunction(()=>tw.world.value.save.stations.length>100 && twDemand.view()?.round===3 && twDemand.view().lines.size>20,undefined,{timeout:120000});
  await page.evaluate(()=>{tw.map.jumpTo({center:[-73.97,40.75],zoom:11.2});tw.selection.value=null;});
  await group.getByRole('button',{name:'Homes',exact:true}).click();
  await page.waitForFunction(()=>twViews.overlay.showBubbles && twViews.overlay.discsDrawn>0);
  await page.waitForTimeout(800); // allow resized camera buffers and imported network to paint
  const overlap=await page.evaluate(()=>{
    const n=tw.renderer.net;
    for(let i=0;i<n.stationCount;i++){const [x,y]=tw.overlay.toScreen(n.stations[i*3],n.stations[i*3+1]);if(x<20||y<50||x>tw.map.getContainer().clientWidth-20||y>tw.map.getContainer().clientHeight-20)continue;
      if(tw.pick(tw.overlay,x,y,false)?.kind==='station'&&[[0,0],[2,0],[-2,0],[0,2],[0,-2]].every(([dx,dy])=>twViews.click(x+dx,y+dy,false)?.kind==='commuters'))return {x,y,station:n.stationIds[i]};}
    return null;
  });
  check('real station covered by visible bubble found',!!overlap,overlap);
  const mapBox=await page.locator('#map').boundingBox();
  await page.evaluate(()=>tw.map.on('click',e=>window.__lastClick={point:e.point,hover:twViews.hover(e.point.x,e.point.y),demand:twViews.click(e.point.x,e.point.y,false)}));
  const clickOverlap=()=>page.mouse.click(mapBox.x+overlap.x,mapBox.y+overlap.y);
  await clickOverlap();
  const bubbleSelection=await page.evaluate(()=>tw.selection.value);
  check('actual map click selects top bubble over station',bubbleSelection?.kind==='commuters',{selection:bubbleSelection,click:await page.evaluate(()=>window.__lastClick),overlap,mapBox});
  await page.evaluate(()=>{tw.selection.value=null;tw.display.value={...tw.display.value,trains:false};});
  await group.getByRole('button',{name:'None',exact:true}).click();
  await page.waitForTimeout(400);
  await clickOverlap();
  const stationSelection=await page.evaluate(()=>tw.selection.value);
  check('hidden bubble gives station click back',stationSelection?.kind==='station',stationSelection);
  await page.waitForFunction(()=>twViews.overlay.sets.has('stationNear'),undefined,{timeout:60000});
  const catchment=await page.evaluate(()=>{
    for(let y=60;y<tw.map.getContainer().clientHeight-20;y+=8)for(let x=20;x<tw.map.getContainer().clientWidth-20;x+=8){const hover=twViews.hover(x,y);if(hover?.some(s=>s.includes('riders')))return{x,y};}return null;
  });
  check('visible station catchment found',!!catchment,catchment);
  await page.mouse.click(mapBox.x+catchment.x,mapBox.y+catchment.y);
  check('catchment click retains parent station',await page.evaluate(id=>tw.selection.value?.station===id,stationSelection.station));
  await page.evaluate(()=>{tw.selection.value=null;});
  await group.getByRole('button',{name:'Homes',exact:true}).click();
  await page.mouse.move(mapBox.x+overlap.x,mapBox.y+overlap.y);
  await page.mouse.down();
  await page.waitForTimeout(500);
  await page.mouse.move(mapBox.x+overlap.x+60,mapBox.y+overlap.y+60,{steps:5});
  await page.mouse.up();
  check('long press above station starts bubble selection',await page.evaluate(()=>tw.selection.value?.kind==='commuters'));
} finally {
  await writeFile('artifacts/trainworld/notes-outputs/T-132-browser.json',JSON.stringify(result,null,2));
  await browser.close();
}
console.log(JSON.stringify(result,null,2));
