import {readFile,writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try{
 const page=await browser.newPage({viewport:{width:1280,height:850}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix);
 const bytes=await readFile('trainworld/T-007-real-nyc.save');
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(bytes));
 await page.waitForFunction(()=>window.twDemand?.view()?.lines.size>30&&window.twDemand.view().round>=3&&window.twDemand.view().split.train>1,null,{timeout:90000});
 await page.waitForTimeout(200);
 const result=await page.evaluate(()=>{
  const tw=window.tw,r=tw.renderer,n=r.net,g=r.gl,dv=window.twDemand.view(),size=new Float32Array(n.stationCount*2),K=1/Math.cos(40.73*Math.PI/180);
  g.bindBuffer(g.ARRAY_BUFFER,r.stationSizeBuf);g.getBufferSubData(g.ARRAY_BUFFER,0,size);
  let clustered=0,isolated=0;
  for(let i=0;i<n.stationCount;i++){
   const d=dv.stations.get(n.stationIds[i]);if(!d)continue;
   let near=false;
   for(let j=0;j<n.stationCount;j++)if(j!==i&&(n.stations[i*3]-n.stations[j*3])**2+(n.stations[i*3+1]-n.stations[j*3+1])**2<=(100*K)**2){near=true;break;}
   const expected=Math.sqrt(Math.min(1,Math.max(0,d.boardings+d.alightings)/150000))*(near?.375:1);
   if(Math.abs(size[i*2+1]-expected)>1e-5)throw Error('Unexpected station radius '+n.stationIds[i]);
   if(near)clustered++;else isolated++;
  }
  if(!clustered||!isolated)throw Error('Real network must exercise both station cases');
  for(const l of dv.lines.values())for(const load of l.loads)if([...load].some(v=>!Number.isFinite(v)||v<0))throw Error('Invalid common-line demand loads');
  return {stations:n.stationCount,clustered,isolated,lines:dv.lines.size,round:dv.round,split:dv.split,railTripsPerDay:dv.railTripsPerDay,webglError:g.getError()};
 });
 await page.evaluate(()=>{const tw=window.tw;document.querySelector('#debug').style.display='none';tw.display.value={...tw.display.value,trains:false,commuters:false,stationNames:false};tw.map.jumpTo({center:[-73.989,40.750],zoom:15});});
 await page.waitForTimeout(200);await page.screenshot({path:'artifacts/trainworld/T-094-stations.png'});
 if(errors.length||result.webglError)throw Error(JSON.stringify({errors,result}));
 await writeFile('artifacts/trainworld/notes-outputs/T-094-browser.json',JSON.stringify({result,errors},null,2));console.log(JSON.stringify({result,errors}));
}finally{await browser.close();}
