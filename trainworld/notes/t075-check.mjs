import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try{
 const results=[],errors=[];
 for(const candidate of [false,true,true,false,false,true]){
  const page=await browser.newPage({viewport:{width:1280,height:850}});page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&synth=1&lines=500&trains=10000'+(candidate?'':'&perfOff=noDouble'));
  await page.waitForFunction(()=>window.tw?.overlay.camMatrix&&window.tw.renderer.tripCount>1000);
  await page.evaluate(()=>{const tw=window.tw;document.querySelector('#debug').style.display='none';tw.map.jumpTo({center:[-73.98,40.73],zoom:12});tw.clock.setRunning(true);});
  await page.waitForTimeout(300);
  const result=await page.evaluate(async()=>{
   const tw=window.tw,o=tw.overlay,original=o.draw.bind(o);let drawMs=0,draws=0;
   o.draw=()=>{const a=performance.now();original();drawMs+=performance.now()-a;draws++;};
   const m=o.mapFrames,at=performance.now();tw.lagStart();
   tw.map.easeTo({center:[-74.06,40.76],duration:2500,easing:t=>t});
   await new Promise(resolve=>setTimeout(resolve,2600));const elapsed=performance.now()-at;
   o.draw=original;
   return {elapsed,draws,mapFrames:o.mapFrames-m,drawMs,drawsPerSec:draws/elapsed*1000,lag:tw.lagStop(),webglError:tw.renderer.gl.getError()};
  });results.push({candidate,...result});await page.close();
 }
 if(errors.length||results.some(r=>r.webglError))throw Error(JSON.stringify({errors,results}));
 await writeFile('trainworld/notes/T-075-check.json',JSON.stringify({results,errors},null,2));console.log(JSON.stringify({results,errors}));
}finally{await browser.close();}
