import {readFile,writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try {
 const page=await browser.newPage({viewport:{width:1280,height:850}}),errors=[];
 page.on('pageerror',e=>{errors.push(e.message);console.log('PAGE ERROR',e.message);});
 await page.goto('http://localhost:8800/trainworld/?debug=1&paused=1&new=1');
 console.log('Page loaded');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix,null,{timeout:30000});
 console.log('Game ready');
 const bytes=await readFile('trainworld/T-007-real-nyc.save');
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(bytes));
 await page.waitForFunction(()=>document.querySelectorAll('.cap-mark').length>30);
 await page.evaluate(async()=>{await document.fonts.ready;window.tw.display.value={...window.tw.display.value,capacity:true,commuters:false};document.querySelector('#debug').style.display='none';});
 const sample=()=>page.evaluate(()=>{
  const box=document.querySelector('.cap-marks'),shown=getComputedStyle(box).display!=='none';
  const rects=[...box.children].filter(e=>shown&&getComputedStyle(e).visibility!=='hidden').map(e=>e.getBoundingClientRect());
  let overlaps=0;for(let i=0;i<rects.length;i++)for(let j=0;j<i;j++){const a=rects[i],b=rects[j];if(a.left<b.right&&a.right>b.left&&a.top<b.bottom&&a.bottom>b.top)overlaps++;}
  return {zoom:window.tw.map.getZoom(),total:box.children.length,visible:rects.length,overlaps};
 });
 const samples=[];
 for(const zoom of [11.2,11.99,12,12.5,13,14.5,11.5,13]){
  await page.evaluate(z=>window.tw.map.jumpTo({center:[-74,40.75],zoom:z}),zoom);await page.waitForTimeout(300);
  const s=await sample();samples.push(s);
  if(s.overlaps||zoom<12&&s.visible||zoom>=12&&!s.visible)throw Error(JSON.stringify(s));
  if(zoom===12.5)await page.screenshot({path:'trainworld/T-102-delay-tags.png'});
 }
 await page.evaluate(()=>window.tw.map.panBy([130,45],{duration:500}));
 for(let i=0;i<6;i++){await page.waitForTimeout(90);const s=await sample();if(s.overlaps)throw Error('Overlap during pan');}
 await page.evaluate(()=>{
  // The GTFS fixture has busy platforms, but no busy junction. Add a derived UI fixture.
  const w=window.tw.world.value,n=[...w.nodes.values()][0],nodes=new Map(w.nodes);
  nodes.set(n.id,{...n,ports:3});
  const [ox,oy]=window.tw.originMerc,s=Math.sin(n.lat*Math.PI/180);
  const x=((n.lng+180)/360-ox)*40075016.686,y=(0.5-Math.log((1+s)/(1-s))/(4*Math.PI)-oy)*40075016.686;
  const markers=[{kind:4,x,y,rho:1.1,delayS:120},{kind:2,x,y,rho:0.9,delayS:10}];
  window.tw.world.value={...w,nodes,capacity:[markers,markers,markers]};
  window.tw.map.jumpTo({center:[n.lng,n.lat],zoom:17});
 });await page.waitForTimeout(300);
 const link=page.locator('.cap-mark.link:not([style*="hidden"])').first();
 if((await sample()).visible!==1||await link.innerText()!=='+2:00')throw Error('Worst delay did not win collision');
 await link.hover();await page.waitForTimeout(80);
 const tooltip=await page.locator('.demand-tag').filter({hasText:'capacity'}).innerText();
 if(!tooltip.includes('Click to see the junction'))throw Error('Tooltip missing');
 await link.click();await page.waitForTimeout(150);
 if(!await page.locator('#dock').innerText().then(s=>s.includes('Junction')))throw Error('Junction inspector did not open');
 for(const capacity of [false,true]){
  await page.evaluate(on=>window.tw.display.value={...window.tw.display.value,capacity:on},capacity);await page.waitForTimeout(80);
  const s=await sample();if(capacity?!s.visible:s.visible)throw Error('Capacity toggle failed');
 }
 if(errors.length)throw Error(errors.join('\n'));
 const result={samples,tooltip,errors};console.log(JSON.stringify(result));await writeFile('trainworld/notes/T-102-check.json',JSON.stringify(result,null,2));
}finally{await browser.close();}
