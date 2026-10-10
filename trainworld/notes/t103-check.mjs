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
 await page.waitForFunction(()=>window.twDemand?.view()?.lines.size>30&&window.twDemand.view().round>=3,null,{timeout:90000});
 await page.evaluate(()=>{window.tw.map.jumpTo({center:[-74,40.74],zoom:12});document.querySelector('#debug').style.display='none';window.tw.display.value={...window.tw.display.value,capacity:false,stationNames:false};});
 await page.waitForTimeout(300);await page.screenshot({path:'trainworld/T-103-network.png'});
 const actualCars=await page.evaluate(()=>{
  const w=window.tw.world.value,n=window.tw.renderer.net;
  for(const l of w.save.lines)if(n.lineCars[l.num]!==w.lineStats[l.id].cars)throw Error('Car count differs from line');
  return {lines:w.save.lines.length,min:Math.min(...n.lineCars),max:Math.max(...n.lineCars)};
 });
 // A single stopped train on a 90 m radius curve, isolated from worker/save state.
 await page.evaluate(()=>{
  const tw=window.tw,r=tw.renderer,base=r.net,K=1/Math.cos(40.73*Math.PI/180),len=90*Math.PI,step=2,count=Math.ceil(len/step)+1;
  const samples=new Float32Array(count*2),seg=new Float32Array((count-1)*4);
  for(let i=0;i<count;i++){const phi=Math.min(i*step,len)/90-Math.PI/2;const x=90*Math.cos(phi)*K,y=90*Math.sin(phi)*K;samples.set([x,y],i*2);if(i)seg.set([samples[(i-1)*2],samples[(i-1)*2+1],x,y],(i-1)*4);}
  const stroke={seg,count:count-1,colour:new Uint32Array(count-1),edge:new Uint32Array(count-1),level:new Float32Array(count-1),flags:new Float32Array(count-1),dist:new Float32Array(count-1)};
  const meta=new Float32Array(24);for(let p=0;p<6;p++)meta.set([0,1,3600,len],p*4);
  const n={...base,track:stroke,lines:stroke,stationCount:0,stations:new Float32Array(0),stationIds:[],phases:new Float32Array([0,len/2,0,0]),meta,samples,lineTable:new Float32Array([0,count,len,step]),lineCars:new Float32Array([8]),lineColours:new Uint8Array([25,118,210,255]),sampleOff:new Float32Array(count),sampleStroke:new Int32Array(count)};
  r.setNetwork(n);
  const fill=new Float32Array(count*2).fill(0.65);
  const layers={lineWidth:new Float32Array(count-1).fill(1),lineOffset:new Float32Array(count-1),sampleOff:new Float32Array(count),fill,stationSize:new Float32Array(0)};
  r.setLayers(layers);window.fixtureLayers=layers;
  const epoch=Math.floor(tw.clock.now()/3600)*3600;r.setTrips(epoch,999,new Float32Array([0,tw.clock.now()-epoch]));
  const mx=tw.originMerc[0]+90*K/40075016.686;tw.map.jumpTo({center:[mx*360-180,40.73],zoom:18});tw.overlay.invalidate();
 });await page.waitForTimeout(300);
 await page.screenshot({path:'trainworld/T-103-cars.png'});
 const checks=[];
 for(const [zoom,expected] of [[15.49,false],[15.5,true],[18,true]]){
  await page.evaluate(z=>window.tw.map.jumpTo({zoom:z}),zoom);await page.waitForTimeout(150);
  const result=await page.evaluate(()=>({zoom:window.tw.map.getZoom(),detail:window.tw.renderer.detailedTrains,error:window.tw.renderer.gl.getError()}));
  if(result.detail!==expected||result.error)throw Error(JSON.stringify(result));checks.push(result);
 }
 // Click the outermost cars rather than the analytic centre: picking follows the whole consist.
 const picks=await page.evaluate(()=>{
  const tw=window.tw,n=tw.renderer.net,len=n.lineTable[2],K=1/Math.cos(40.73*Math.PI/180),results=[];
  for(const car of [0,3,7]){
   const d=len/2+(car+0.5)*20-80,phi=d/90-Math.PI/2,[x,y]=tw.overlay.toScreen(90*Math.cos(phi)*K,90*Math.sin(phi)*K);
   const hit=tw.pick(tw.overlay,x,y,true);if(hit?.kind!=='train')throw Error('Car not pickable '+car);results.push({car,hit});
  }return results;
 });
 for(const profile of [0,3])for(const fill of [-1,0,0.5,1,2]){
  await page.evaluate(({profile,fill})=>{const tw=window.tw,r=tw.renderer;window.fixtureLayers.fill.fill(fill);r.setLayers(window.fixtureLayers);const epoch=Math.floor(tw.clock.now()/3600)*3600;r.setTrips(epoch,1000+profile+fill,new Float32Array([profile,tw.clock.now()-epoch]));tw.overlay.invalidate();},{profile,fill});await page.waitForTimeout(60);
  if(await page.evaluate(()=>window.tw.renderer.gl.getError()))throw Error('WebGL error');
 }
 if(errors.length)throw Error(errors.join('\n'));
 const result={actualCars,checks,picks,errors};console.log(JSON.stringify(result));await writeFile('trainworld/notes/T-103-check.json',JSON.stringify(result,null,2));
}finally{await browser.close();}
