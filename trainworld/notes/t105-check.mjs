import {readFile,writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try {
 const page=await browser.newPage({viewport:{width:688,height:953}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix);
 if(await page.evaluate(()=>window.tw.display.value.trackColour)!=='traffic')throw Error('Fresh default is not traffic');
 const bytes=await readFile('trainworld/T-007-real-nyc.save');
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);},Array.from(bytes));
 await page.waitForFunction(()=>window.twDemand?.view()?.lines.size>30&&window.twDemand.view().round>=3,null,{timeout:90000});
 await page.evaluate(()=>{const r=window.tw.renderer,original=r.setLayers.bind(r);r.setLayers=l=>{window.testLayers=l;original(l);};window.tw.map.jumpTo({center:[-73.99,40.75],zoom:12.2});document.querySelector('#debug').style.display='none';});
 await page.getByRole('button',{name:'Settings',exact:true}).click();
 const modes=[];
 for(const [label,mode] of [['Line','line'],['Line + traffic thickness','traffic'],['Height','height'],['Max speed','speed']]){
  await page.getByRole('button',{name:label,exact:true}).click();await page.waitForTimeout(180);
  const sample=await page.evaluate(()=>({mode:window.tw.display.value.trackColour,widths:Array.from(new Set(window.testLayers.lineWidth)),slot:window.tw.renderer.slotPx,error:window.tw.renderer.gl.getError(),layout:{width:document.querySelector('.pane').clientWidth,scrollWidth:document.querySelector('.pane').scrollWidth}}));
  if(sample.mode!==mode||sample.error||sample.layout.scrollWidth>sample.layout.width+1)throw Error(JSON.stringify(sample));
  if(mode==='traffic'?!sample.widths.some(w=>w!==1):sample.widths.some(w=>w!==1))throw Error('Wrong width mode');
  if((mode==='height'||mode==='speed')&&sample.slot!==0)throw Error('Colour mode retained line slots');
  modes.push({...sample,widths:sample.widths.slice(0,5)});
 }
 await page.screenshot({path:'trainworld/T-105-speed.png'});
 await page.getByRole('button',{name:'Line',exact:true}).click();await page.reload();await page.waitForFunction(()=>window.tw?.world.value);
 if(await page.evaluate(()=>window.tw.display.value.trackColour)!=='line')throw Error('Plain line mode not persisted');
 const ack=await page.evaluate(()=>window.tw.client.edit({op:'route',from:{kind:'free',x:0,y:0,level:0},to:{kind:'free',x:1000,y:1000,level:0},pis:[{x:1000,y:0,radius:100,level:0}],single:false}));
 if(!ack.ok)throw Error(JSON.stringify(ack));
 const speeds=await page.evaluate(()=>{const n=window.tw.renderer.net,s=Array.from(n.track.speed);return {count:s.length,segments:n.track.count,min:Math.min(...s),max:Math.max(...s)};});
 if(speeds.count!==speeds.segments||Math.abs(speeds.min-Math.sqrt(1.1*100)*3.6)>0.1||Math.abs(speeds.max-160)>0.1)throw Error(JSON.stringify(speeds));
 // Old settings retain their width checkbox choice when migrated.
 const migrations=[];
 for(const lineLoad of [false,true]){
  await page.evaluate(on=>localStorage.setItem('trainworld-ui',JSON.stringify({display:{trackColour:'line',lineLoad:on}})),lineLoad);await page.reload();await page.waitForFunction(()=>window.tw?.display);
  const mode=await page.evaluate(()=>window.tw.display.value.trackColour);if(mode!==(lineLoad?'traffic':'line'))throw Error('Legacy migration failed');migrations.push({lineLoad,mode});
 }
 if(errors.length)throw Error(errors.join('\n'));
 const result={modes,speeds,migrations,errors};console.log(JSON.stringify(result));await writeFile('trainworld/notes/T-105-check.json',JSON.stringify(result,null,2));
}finally{await browser.close();}
