import {readFile,writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try{
 const page=await browser.newPage({viewport:{width:1280,height:850}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&paused=1&new=1');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix,null,{timeout:30000});
 const bytes=await readFile('trainworld/T-007-real-nyc.save');
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(bytes));
 await page.waitForFunction(()=>window.tw.world.value.save.stations.length>800);
 await page.evaluate(async()=>{await document.fonts.ready;window.tw.display.value={...window.tw.display.value,commuters:true,commuterEnd:'work'};document.querySelector('#debug').style.display='none';});
 const sample=()=>page.evaluate(()=>{
  const box=document.querySelector('.stn-labels'),map=document.querySelector('#map').getBoundingClientRect();
  const shown=getComputedStyle(box).display!=='none';
  const rects=[...document.querySelectorAll('.stn-label')].filter(e=>shown&&getComputedStyle(e).visibility!=='hidden').map(e=>e.getBoundingClientRect()).filter(r=>r.right>map.left&&r.left<map.right&&r.bottom>map.top&&r.top<map.bottom);
  let overlaps=0;for(let i=0;i<rects.length;i++)for(let j=0;j<i;j++){const a=rects[i],b=rects[j];if(a.left<b.right-1&&a.right>b.left+1&&a.top<b.bottom-1&&a.bottom>b.top+1)overlaps++;}
  return {zoom:window.tw.map.getZoom(),shown,visible:rects.length,overlaps,font:getComputedStyle(document.querySelector('.stn-label')).fontFamily,shadowMinZoom:window.tw.map.getLayer('tw-station-shadow').minzoom,legend:!!document.querySelector('#commuter-legend')};
 });
 const samples=[];
 for(const zoom of [11.2,11.99,12,12.5,13,14.5,11.5,13]){
  await page.evaluate(z=>window.tw.map.jumpTo({center:[-73.97,40.75],zoom:z}),zoom);await page.waitForTimeout(250);
  const s=await sample();samples.push(s);
  if(zoom<12&&s.visible!==0)throw Error('Station labels visible below cutoff');
  if(zoom>=12&&s.visible===0)throw Error('Labels did not return at closer zoom');
  if(s.overlaps||s.legend||!s.font.includes('Zen Maru Gothic'))throw Error('Labels/legend/font check failed');
  if(zoom===11.2||zoom===12.5||zoom===14.5)await page.screenshot({path:`trainworld/T-101-zoom-${zoom}.png`});
 }
 const cdp=await page.context().newCDPSession(page);await cdp.send('DOM.enable');await cdp.send('CSS.enable');
 const root=await cdp.send('DOM.getDocument');const label=await cdp.send('DOM.querySelector',{nodeId:root.root.nodeId,selector:'.stn-label:not([style*="hidden"])'});
 const fonts=await cdp.send('CSS.getPlatformFontsForNode',{nodeId:label.nodeId});
 if(!fonts.fonts.some(f=>f.familyName==='Zen Maru Gothic'&&f.isCustomFont))throw Error('Actual station glyphs do not use the project font');
 await page.evaluate(()=>window.tw.map.panBy([90,30],{duration:400}));await page.waitForTimeout(500);samples.push(await sample());
 await page.evaluate(()=>window.tw.display.value={...window.tw.display.value,stationNames:false});await page.waitForTimeout(150);
 if((await sample()).shown)throw Error('Station names switch did not hide labels');
 await page.evaluate(()=>window.tw.display.value={...window.tw.display.value,stationNames:true});await page.waitForTimeout(150);
 if(!(await sample()).shown)throw Error('Station names switch did not restore labels');
 await page.evaluate(()=>window.tw.display.value={...window.tw.display.value,commuterEnd:'home'});await page.waitForTimeout(150);
 if(await page.locator('#commuter-legend').count())throw Error('Homes legend reappeared');
 const result={samples,fonts,errors};console.log(JSON.stringify(result));await writeFile('trainworld/notes/T-101-check.json',JSON.stringify(result,null,2));
 if(errors.length)throw Error(errors.join('\n'));
}finally{await browser.close();}
