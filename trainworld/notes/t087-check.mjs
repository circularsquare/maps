import {readFile,writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try {
 const page=await browser.newPage({viewport:{width:1280,height:850}}),errors=[];
 page.on('pageerror',e=>{errors.push(e.message);console.log(e.message);});
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix);
 const bytes=await readFile('trainworld/T-007-real-nyc.save');
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(bytes));
 await page.waitForFunction(()=>window.twDemand?.view()?.lines.size>30,null,{timeout:90000});
 await page.evaluate(()=>{
  const r=window.tw.renderer,original=r.setLayers.bind(r);
  r.setLayers=l=>{window.testLayers=l;original(l);};
  window.tw.display.value={...window.tw.display.value,trainLoad:true,trainLoadBasis:'average'};
 });await page.waitForTimeout(100);
 const result=await page.evaluate(()=>{
  const dv=window.twDemand.view(),w=window.tw.world.value,l=w.save.lines.find(l=>dv.lines.get(l.id)?.loads[0]?.[0]>0&&w.lineStats[l.id]?.status==='running');
  const st=w.lineStats[l.id],dm=dv.lines.get(l.id),average=dm.loads[0][0]/(l.tph.high*4),busiest=average*dv.peakHourFactor[0];
  window.testLine=l;
  const gauge=n=>n<=44*st.cars?0.5*n/(44*st.cars):n<=160*st.cars?0.5+0.5*(n-44*st.cars)/(116*st.cars):2;
  const net=window.tw.renderer.net,index=Array.from(net.sampleStroke).findIndex(i=>i>=0&&net.lines.colour[i]===l.num&&net.lines.info[i*4]===0);
  window.testSample=index;
  return {line:l.name,factor:dv.peakHourFactor[0],average,busiest,expectedAverage:gauge(average),expectedBusiest:gauge(busiest),gaugeAverage:window.testLayers.fill[index*2]};
 });
 await page.evaluate(()=>window.tw.display.value={...window.tw.display.value,trainLoadBasis:'busiest'});await page.waitForTimeout(100);
 result.gaugeBusiest=await page.evaluate(()=>window.testLayers.fill[window.testSample*2]);
 if(Math.abs(result.gaugeAverage-result.expectedAverage)>1e-5||Math.abs(result.gaugeBusiest-result.expectedBusiest)>1e-5)throw Error(JSON.stringify(result));
 await page.evaluate(()=>{
  const l=window.testLine,profile=l.num*6,dep=window.tw.clock.now()-60;
  if(!window.tw.tripAt(window.tw.renderer.net,profile,dep,window.tw.clock.now()))throw Error('Train fixture not on route');
  window.tw.select({kind:'train',line:l.id,profile,dep});
 });await page.waitForTimeout(150);
 const inspector=await page.locator('#dock').innerText();
 if(!inspector.includes('Average riders')||!inspector.includes('Busiest hour'))throw Error('Inspector missing both views');
 result.inspector=inspector;
 // The remembered setting survives a fresh load, and the visible control switches it.
 await page.getByRole('button',{name:'Settings',exact:true}).click();
 await page.getByText('Average',{exact:true}).click();
 if(await page.evaluate(()=>window.tw.display.value.trainLoadBasis)!=='average')throw Error('Settings switch failed');
 await page.reload();await page.waitForFunction(()=>window.tw?.display);
 if(await page.evaluate(()=>window.tw.display.value.trainLoadBasis)!=='average')throw Error('Setting not persisted');
 if(errors.length)throw Error(errors.join('\n'));
 result.errors=errors;console.log(JSON.stringify(result));await writeFile('trainworld/notes/T-087-check.json',JSON.stringify(result,null,2));
}finally{await browser.close();}
