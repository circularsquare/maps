import {readFile,writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try{
 const page=await browser.newPage({viewport:{width:1600,height:850}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix);
 const data=Array.from(await readFile('trainworld/T-007-real-nyc.save'));
 await page.evaluate(data=>{const bytes=Uint8Array.from(data).buffer;window.tw.client.post({kind:'import',bytes},[bytes]);document.querySelector('#debug').style.display='none';},data);
 await page.waitForFunction(()=>window.twDemand?.view()?.round>=3,null,{timeout:90000});
 const measurements=[];
 for(const viewport of [1600,1280,1024,880,688,600,500]){
  await page.setViewportSize({width:viewport,height:850});await page.waitForTimeout(150);
  const result=await page.evaluate(()=>{
   const header=document.querySelector('#topbar'),r=header.getBoundingClientRect(),visible=[...header.children].filter(e=>getComputedStyle(e).display!=='none');
   for(const e of visible){const a=e.getBoundingClientRect();if(a.left<r.left-1||a.right>r.right+1||a.top<r.top||a.bottom>r.bottom)throw Error('Outside top bar '+e.className);}
   for(let i=0;i<visible.length;i++)for(let j=i+1;j<visible.length;j++){
    const a=visible[i].getBoundingClientRect(),b=visible[j].getBoundingClientRect();
    if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>1&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1)throw Error('Overlapping '+visible[i].className+' '+visible[j].className);
   }
   const split=document.querySelector('.split'),sr=split.getBoundingClientRect();
   for(const e of split.children)if(getComputedStyle(e).display!=='none'&&e.getBoundingClientRect().right>sr.right+1)throw Error('Split overflows');
   return {mapWidth:r.width,barHeight:r.height,walkingVisible:getComputedStyle(document.querySelector('.split-other')).display!=='none',clockVisible:document.querySelector('.time').getBoundingClientRect().width>0,speedButtons:document.querySelectorAll('.speed button').length};
  });measurements.push({viewport,...result});
  if(viewport===688)await page.screenshot({path:'artifacts/trainworld/T-106-topbar.png'});
 }
 // A wider dock changes layout even when browser width stays fixed.
 await page.setViewportSize({width:1280,height:850});
 await page.evaluate(()=>document.documentElement.style.setProperty('--dock-w','850px'));await page.waitForTimeout(150);
 const resized=await page.evaluate(()=>({mapWidth:document.querySelector('#topbar').getBoundingClientRect().width,barHeight:document.querySelector('#topbar').getBoundingClientRect().height}));
 if(resized.barHeight<70)throw Error('Dock resize did not trigger compact header');
 if(errors.length)throw Error(errors.join('\n'));
 await writeFile('artifacts/trainworld/notes-outputs/T-106-check.json',JSON.stringify({measurements,resized,errors},null,2));console.log(JSON.stringify({measurements,resized,errors}));
}finally{await browser.close();}
