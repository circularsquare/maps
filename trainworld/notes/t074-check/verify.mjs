import { readFile,writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try {
const page=await browser.newPage({viewport:{width:1360,height:900}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix,null,{timeout:30000});
const bytes=await readFile('trainworld/T-007-real-nyc.save');
await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(bytes));
await page.waitForFunction(()=>window.tw.world.value.save.stations.length>800,null,{timeout:30000});
await page.evaluate(async()=>{
 await document.fonts.ready;
 const tw=window.tw;
 window.labelSamples=[];
 window.verifyLabels=()=>{
  if(getComputedStyle(document.querySelector('.stn-labels')).display==='none')return {max:0,n:0,overlaps:0};
  const w=tw.world.value,lines=new Map();for(const l of w.save.lines)for(const s of new Set(l.stops))lines.set(s,(lines.get(s)||0)+1);
  const stations=[...w.save.stations].sort((a,b)=>(lines.get(b.id)||0)-(lines.get(a.id)||0)||Number(b.built)-Number(a.built));
  const outer=tw.map.getContainer().getBoundingClientRect();let max=0,n=0,overlaps=0;const rects=[];
  document.querySelectorAll('.stn-label').forEach((el,i)=>{
   if(el.style.visibility==='hidden')return;
   const s=stations[i],r=el.getBoundingClientRect();const sin=Math.sin(s.lat*Math.PI/180);
   const p=tw.overlay.toScreen(((s.lng+180)/360-tw.originMerc[0])*40075016.686,(.5-Math.log((1+sin)/(1-sin))/(4*Math.PI)-tw.originMerc[1])*40075016.686);
   const below=Math.abs(Math.cos(s.heading))>Math.abs(Math.sin(s.heading));
   max=Math.max(max,Math.abs(r.left-outer.left-(below?p[0]-r.width/2:p[0]+10)),Math.abs(r.top-outer.top-(below?p[1]+8:p[1]-r.height/2)));
   for(const t of rects)if(r.left<t.right-.99&&r.right>t.left+.99&&r.top<t.bottom-.99&&r.bottom>t.top+.99)overlaps++;
   rects.push(r);n++;
  });return {max,n,overlaps};
 };
 tw.overlay.afterMapFrame.push(()=>window.labelSamples.push(window.verifyLabels()));
 tw.map.jumpTo({center:[-73.97,40.75],zoom:12.2});
});
await page.waitForTimeout(600);
for(const opts of [{offset:[800,400]}, {offset:[-1200,-500]}, {zoom:14.2,bearing:32}, {pitch:45}, {pitch:0,bearing:0,zoom:12.2}]){
 await page.evaluate(o=>{if(o.offset)window.tw.map.panBy(o.offset,{duration:500});else window.tw.map.easeTo({...o,duration:500});},opts);
 await page.waitForTimeout(700);
}
await page.evaluate(()=>{const tw=window.tw,w=tw.world.value;tw.world.value={...w,save:{...w.save,stations:w.save.stations.map((s,i)=>i===0?{...s,name:'Renamed station'}:s)}};});
await page.waitForTimeout(200);
const renamed=await page.locator('.stn-label').filter({hasText:'Renamed station'}).count();
await page.evaluate(()=>{window.tw.display.value={...window.tw.display.value,stationNames:false};window.tw.map.panBy([700,0],{duration:300});});
await page.waitForTimeout(500);
await page.evaluate(()=>{window.tw.display.value={...window.tw.display.value,stationNames:true};});
await page.waitForTimeout(200);
await page.setViewportSize({width:1000,height:750});await page.waitForTimeout(200);
await page.evaluate(()=>window.tw.map.jumpTo({center:[-73.97,40.75],zoom:12.6,pitch:0,bearing:0}));await page.waitForTimeout(300);
const result=await page.evaluate(()=>({stations:window.tw.world.value.save.stations.length,frames:window.labelSamples.length,maxError:Math.max(...window.labelSamples.map(x=>x.max)),maxOverlaps:Math.max(...window.labelSamples.map(x=>x.overlaps)),final:window.verifyLabels()}));
result.renamed=renamed;result.errors=errors;
console.log(JSON.stringify(result));
await page.screenshot({path:'trainworld/notes/t074-check/real-network.png'});
await writeFile('trainworld/notes/t074-check/verification.json',JSON.stringify(result,null,2));
if(result.maxError>1||result.maxOverlaps>0||renamed!==1||errors.length)throw Error('Real-network verification failed');
}finally{await browser.close();}
