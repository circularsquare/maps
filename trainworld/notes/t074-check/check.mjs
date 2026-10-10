import { writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const browser = await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
const system = await browser.newBrowserCDPSession();
const results = [];
for (const old of [true, false, false, true]) {
  const context = await browser.newContext({viewport:{width:1360,height:900}});
  const page = await context.newPage();
  const errors=[]; page.on('pageerror', e=>{errors.push(e.message);console.log('PAGE ERROR',e.message);});
  page.on('requestfailed',r=>console.log('REQUEST FAILED',r.url(),r.failure()));
  await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1'+(old?'&perfOff=labelPan':''));
  console.log('loaded page',old);
  try { await page.waitForFunction(()=>window.tw?.world.value && window.tw.overlay.camMatrix,null,{timeout:20000}); }
  catch(e) { console.log(await page.evaluate(()=>({tw:!!window.tw,world:!!window.tw?.world.value,matrix:!!window.tw?.overlay.camMatrix,body:document.body.innerText})));await browser.close();throw e; }
  await page.evaluate(async()=> {
    await document.fonts.ready;
    const tw=window.tw, w=tw.world.value;
    const stations=Array.from({length:300},(_,i)=>({id:String(10000+i),num:10000+i,name:`S ${i}`,lng:-74.025+(i%20)*.007,lat:40.70+Math.floor(i/20)*.006,x:0,y:0,heading:i%2?0:Math.PI/2,built:true,length:200,level:0}));
    tw.world.value={...w,save:{...w.save,stations,lines:[]}};
    tw.map.jumpTo({center:[-73.96,40.745],zoom:11.8,bearing:0,pitch:0});
    window.labelWrites={plane:0,children:0};
    new MutationObserver(ms=>{for(const m of ms) m.target.classList.contains('stn-label-plane')?window.labelWrites.plane++:window.labelWrites.children++;}).observe(document.querySelector('.stn-label-plane'),{attributes:true,subtree:true,attributeFilter:['style']});
    window.checkLabels=()=>{
      let n=0,max=0;const rect=tw.map.getContainer().getBoundingClientRect();
      for(const el of document.querySelectorAll('.stn-label')){
        if(el.style.visibility==='hidden')continue;
        const i=Number(el.textContent.split(' ').at(-1)),s=stations[i],r=el.getBoundingClientRect(),p=tw.overlay.toScreen((s.lng+180)/360*40075016.686-tw.originMerc[0]*40075016.686,(.5-Math.log((1+Math.sin(s.lat*Math.PI/180))/(1-Math.sin(s.lat*Math.PI/180)))/(4*Math.PI))*40075016.686-tw.originMerc[1]*40075016.686);
        const below=i%2===1;
        const x=below?p[0]-r.width/2:p[0]+10,y=below?p[1]+8:p[1]-r.height/2;
        max=Math.max(max,Math.abs(r.left-rect.left-x),Math.abs(r.top-rect.top-y));n++;
      }return {n,max};
    };
  });
  await page.waitForTimeout(1500);
  const cd = await context.newCDPSession(page); await cd.send('Performance.enable');
  const cpu=async()=> (await system.send('SystemInfo.getProcessInfo')).processInfo.reduce((a,p)=>a+p.cpuTime,0);
  const startCPU=await cpu(),startMetrics=await cd.send('Performance.getMetrics'),start=Date.now();
  await page.evaluate(()=>{window.labelWrites={plane:0,children:0};});
  const rect=await page.locator('#map').boundingBox();
  await page.mouse.move(rect.x+rect.width/2,rect.y+rect.height/2);
  await page.mouse.down();
  for(let i=0;i<240;i++){
    await page.mouse.move(rect.x+rect.width/2+110*Math.sin(i/24),rect.y+rect.height/2+50*Math.sin(i/32));
    await page.waitForTimeout(16);
  }
  const elapsed=Date.now()-start,endCPU=await cpu(),endMetrics=await cd.send('Performance.getMetrics');
  const metric=(m,n)=>m.metrics.find(x=>x.name===n)?.value??0;
  const during=await page.evaluate(()=>({writes:window.labelWrites,alignment:window.checkLabels(),matrix:Array.from(window.tw.overlay.camMatrix),transform:document.querySelector('.stn-label-plane').style.transform}));
  await page.mouse.up();await page.waitForTimeout(400);
  const checks=[];
  for(const opts of [{center:[-73.975,40.75]},{zoom:14},{bearing:28},{pitch:40},{pitch:0,bearing:0},{padding:{left:80,right:10,top:20,bottom:5}}]){
    await page.evaluate(o=>window.tw.map.jumpTo(o),opts);await page.waitForTimeout(200);checks.push(await page.evaluate(()=>window.checkLabels()));
  }
  await page.setViewportSize({width:1120,height:780});await page.waitForTimeout(200);checks.push(await page.evaluate(()=>window.checkLabels()));
  await page.evaluate(()=>{window.tw.display.value={...window.tw.display.value,stationNames:false};});
  const hidden=await page.locator('.stn-labels').evaluate(e=>getComputedStyle(e).display==='none');
  await page.evaluate(()=>{window.tw.display.value={...window.tw.display.value,stationNames:true};});await page.waitForTimeout(200);checks.push(await page.evaluate(()=>window.checkLabels()));
  const result={old,elapsed,cpuPct:(endCPU-startCPU)*100000/elapsed,taskPct:(metric(endMetrics,'TaskDuration')-metric(startMetrics,'TaskDuration'))*100000/elapsed,during,checks,hidden,errors};
  results.push(result);console.log(JSON.stringify(result));
  if(!old) await page.screenshot({path:'trainworld/notes/t074-check/labels.png'});
  if(during.alignment.max>1 || checks.some(c=>c.max>1) || !hidden || errors.length) throw Error('Browser check failed');
  await context.close();
}
await writeFile('trainworld/notes/t074-check/results.json',JSON.stringify(results,null,2));
await browser.close();
