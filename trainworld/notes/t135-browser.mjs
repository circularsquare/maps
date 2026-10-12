// Boston work selection -> all represented cities. Isolated browser, captures outside project.
import { writeFile } from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
const results=[];
const check=(name,ok,detail)=>{results.push({name,ok:!!ok,detail});if(!ok)throw Error(name+': '+JSON.stringify(detail));};
try {
  const page=await browser.newPage({viewport:{width:1300,height:900}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://localhost:8800/trainworld/?new=1&paused=1&debug=1&demandWorkers=2');
  await page.waitForFunction(()=>window.twDemand?.view()?.round===3,undefined,{timeout:180000});
  await page.addStyleTag({content:'#debug-root {display:none}'});
  await page.getByRole('button',{name:'City',exact:true}).click();
  await page.getByRole('button',{name:'Boston',exact:true}).click();
  await page.waitForFunction(()=>!tw.map.isMoving()&&twDemand.activeCity.value==='bos');
  await page.getByRole('group',{name:'Demand on the map'}).getByRole('button',{name:'Destinations',exact:true}).click();
  await page.waitForFunction(()=>twViews.overlay.discsDrawn>0&&twViews.overlay.showBubbles);
  await page.waitForTimeout(500);
  const selected=await page.evaluate(()=>{
    for(let y=120;y<700;y+=10)for(let x=100;x<900;x+=10){const s=twViews.click(x,y,false);if(s?.kind==='commuters'&&s.end==='work'&&s.cells.length>20){tw.selection.value=s;return s;}}return null;
  });
  check('Boston work selection found',!!selected);
  const flows=await page.evaluate(async cells=>{const r=await twDemand.intercitySelection('work',Uint32Array.from(cells));return r&&{all:r.ends.all,zones:r.ends.zones.filter(z=>z.all>0).length,places:r.ends.places.map(p=>({place:p.place,all:p.all,rail:p.rail}))};},selected.cells);
  await page.waitForFunction(()=>twViews.overlay.sets.get('inter')?.count>0);
  check('close view retains every positive far zone',await page.evaluate(n=>twViews.overlay.sets.get('inter').count===n,flows.zones),flows);
  await page.evaluate(()=>tw.map.fitBounds([[-78,38],[-70.3,43.1]],{padding:90,duration:0}));
  await page.waitForFunction(()=>document.querySelectorAll('.intercity-label').length===4);
  check('one bubble per other city, selected Boston omitted',await page.evaluate(()=>twViews.overlay.sets.get('inter').count===4&&!Array.from(document.querySelectorAll('.intercity-label')).some(e=>e.textContent.startsWith('Boston ·'))));
  const labels=await page.locator('.intercity-label').allTextContents();
  check('other city totals labeled', ['New York','Philadelphia','Trenton','Washington, DC'].every(n=>labels.some(l=>l.startsWith(n+' ·'))),labels);
  const radii=await page.evaluate(()=>{
    const v=twViews.overlay,s=v.sets.get('inter'),gl=v.gl,data=new Float32Array(s.count*8);
    gl.bindBuffer(gl.ARRAY_BUFFER,s.buf);gl.getBufferSubData(gl.ARRAY_BUFFER,0,data);
    return Array.from({length:s.count},(_,i)=>({radius:data[i*8+2],colour:Array.from(data.slice(i*8+3,i*8+6))}));
  });
  const external=flows.places.filter(p=>p.place!==3).sort((a,b)=>b.all-a.all);
  check('circle area proportional to total travellers',radii.every((d,i)=>Math.abs((d.radius/radii[0].radius)**2-external[i].all/external[0].all)<1e-6),radii);
  const hover=await page.evaluate(()=>{
    const p=tw.map.project([-77.0365,38.8977]);return twViews.hover(p.x,p.y);
  });
  check('work selection shows incoming Washington demand',hover?.some(l=>l.includes('from Washington, DC')),hover);
  const unchanged=await page.evaluate(()=>tw.selection.value?.kind==='commuters'&&tw.selection.value.end==='work'&&twDemand.activeCity.value==='bos');
  check('regional panning retains Boston selection',unchanged);
  await page.mouse.move(320,870);
  await page.screenshot({path:'artifacts/trainworld/T-135-regional-work.png'});
  await page.evaluate(()=>{tw.display.value={...tw.display.value,intercity:false};});
  check('hide setting clears circles and labels',await page.evaluate(()=>!twViews.overlay.sets.has('inter')&&document.querySelectorAll('.intercity-label').length===0));
  await page.evaluate(()=>{tw.display.value={...tw.display.value,intercity:true};tw.map.jumpTo({center:[-71.0589,42.3601],zoom:11.2});});
  await page.waitForFunction(()=>twViews.overlay.sets.get('inter')?.count>5);
  check('zoom back restores detailed zones',await page.evaluate(n=>twViews.overlay.sets.get('inter').count===n&&document.querySelectorAll('.intercity-label').length===0,flows.zones));
  await page.screenshot({path:'artifacts/trainworld/T-135-close-work.png'});
  check('no browser exceptions',errors.length===0,errors);
} finally {await writeFile('trainworld/notes/T-135-browser.json',JSON.stringify(results,null,2));await browser.close();}
console.log(JSON.stringify(results,null,2));
