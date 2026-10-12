import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try {
 const page=await browser.newPage({viewport:{width:1200,height:850}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix);
 const ids=await page.evaluate(async()=>{
  const tw=window.tw,run=async op=>{const a=await tw.client.edit(op);if(!a.ok)throw Error(JSON.stringify(a));return a;};
  await run({op:'route',from:{kind:'free',x:0,y:0,level:-2},to:{kind:'free',x:6000,y:0,level:-2},pis:[],single:false});
  const edge=tw.world.value.edges[0],A=edge.a,B=edge.b;
  for(const [node,name] of [[A,'A'],[B,'B']])await run({op:'addStation',at:{kind:'node',node},platform:200,name});
  await run({op:'addStation',at:{kind:'edge',edge:edge.id,x:2000,y:0},platform:200,name:'Infill'});
  const I=Number(tw.world.value.save.stations.find(s=>s.name==='Infill').id);
  const next=tw.world.value.edges.find(e=>e.a===B||e.b===B);
  await run({op:'addStation',at:{kind:'edge',edge:next.id,x:4000,y:0},platform:200,name:'Infill 2'});
  const C=Number(tw.world.value.save.stations.find(s=>s.name==='Infill 2').id);
  await run({op:'route',from:{kind:'node',node:A},to:{kind:'free',x:-2000,y:0,level:-2},pis:[],single:false});
  const ext=tw.world.value.edges.find(e=>e.a===A&&e.b!==I||e.b===A&&e.a!==I),E=ext.a===A?ext.b:ext.a;
  await run({op:'addStation',at:{kind:'node',node:E},platform:200,name:'Extension'});
  await run({op:'route',from:{kind:'free',x:10000,y:0,level:-2},to:{kind:'free',x:12000,y:0,level:-2},pis:[],single:false});
  const island=tw.world.value.edges.at(-1).a;
  await run({op:'addStation',at:{kind:'node',node:island},platform:200,name:'Island'});
  const added=await run({op:'addLine',stops:[A,B],name:'Insertion check',colour:'#0055bb'});
  const line=added.line;window.fixture={A,B,I,C,E,island,line};tw.select({kind:'line',line:String(line)});
  const ss=tw.world.value.save.stations;
  tw.map.fitBounds([[Math.min(...ss.map(s=>s.lng)),Math.min(...ss.map(s=>s.lat))],[Math.max(...ss.map(s=>s.lng)),Math.max(...ss.map(s=>s.lat))]],{padding:50,duration:0});
  return window.fixture;
 });
 await page.getByRole('button',{name:'Edit stops',exact:true}).click();
 await page.waitForFunction(()=>window.tw.tools.editingLine===window.fixture.line);
 const click=async station=>{
  const p=await page.evaluate(station=>{
   const tw=window.tw,n=tw.renderer.net,i=Array.from(n.stationIds).findIndex(id=>Number(id)===station);
   const [x,y]=tw.overlay.toScreen(n.stations[i*3],n.stations[i*3+1]),r=tw.map.getContainer().getBoundingClientRect();
   return {x:x+r.left,y:y+r.top};
  },station);
  await page.mouse.click(p.x,p.y);
 };
 const stops=()=>page.evaluate(()=>window.tw.world.value.save.lines.find(l=>Number(l.id)===window.fixture.line).stops.map(Number));
 const wait=async expected=>{await page.waitForFunction(expected=>JSON.stringify(window.tw.world.value.save.lines.find(l=>Number(l.id)===window.fixture.line).stops.map(Number))===JSON.stringify(expected),expected);};
 await click(ids.I);await wait([ids.A,ids.I,ids.B]);
 await click(ids.C);await wait([ids.A,ids.I,ids.C,ids.B]);
 await click(ids.E);await wait([ids.E,ids.A,ids.I,ids.C,ids.B]);
 await click(ids.island);await page.waitForTimeout(150);
 if(JSON.stringify(await stops())!==JSON.stringify([ids.E,ids.A,ids.I,ids.C,ids.B]))throw Error('Disconnected stop modified line');
 // A queued answer must never edit a line after leaving edit-stops mode.
 await page.evaluate(async()=>{
  const tw=window.tw;await tw.client.edit({op:'setStops',line:window.fixture.line,stops:[window.fixture.A,window.fixture.B]});
  const original=tw.client.query.bind(tw.client);
  window.restoreQuery=()=>{tw.client.query=original;};
  tw.client.query=q=>q.what==='insertStop'?new Promise(resolve=>{window.releaseInsertion=async()=>resolve(await original(q));}):original(q);
 });
 await click(ids.I);await page.waitForFunction(()=>window.releaseInsertion);
 await page.evaluate(()=>{window.tw.tools.exit();void window.releaseInsertion();});
 await page.waitForTimeout(100);
 if(JSON.stringify(await stops())!==JSON.stringify([ids.A,ids.B]))throw Error('Stale query edited line');
 await page.evaluate(()=>window.restoreQuery());
 await page.getByRole('button',{name:'Edit stops',exact:true}).click();
 await page.getByRole('button',{name:'Remove stop A',exact:true}).click();await wait([ids.B]);
 await page.getByRole('button',{name:'Remove stop B',exact:true}).click();await wait([]);
 await click(ids.I);await wait([ids.I]);
 await click(ids.B);await wait([ids.I,ids.B]);
 const result={ids,stops:await stops(),errors};if(errors.length)throw Error(errors.join('\n'));
 await writeFile('trainworld/notes/T-112-check.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));
} finally {await browser.close();}
