import { writeFile } from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try {
 const page=await browser.newPage({viewport:{width:688,height:953}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix,null,{timeout:30000});
 await page.getByRole('button',{name:'Build',exact:true}).click();
 if(await page.getByRole('button',{name:'Select',exact:true}).count())throw Error('Select button remains');
 if(await page.locator('#tools button').count())throw Error('Build buttons remain on map');
 await page.locator('[title="Draw track (1)"]').click();
 await page.keyboard.press('Escape');
 if(await page.evaluate(()=>window.tw.tools===null))throw Error('Tools unavailable');
 await page.locator('[title="Draw track (1)"]').click();
 await page.locator('[title="Draw track (1)"]').click();
 if(await page.locator('.tool-pick .on').count())throw Error('Clicking the active tool did not return to selection');
 const fixture=await page.evaluate(async()=>{
  const tw=window.tw,client=tw.client;
  const route=async(x,y,x1,y1,level=0,single=false)=>{
   const a=await client.edit({op:'route',from:{kind:'free',x,y,level},to:{kind:'free',x:x1,y:y1,level},pis:[],single});
   if(!a.ok)throw Error(JSON.stringify(a));
  };
  await route(0,0,1000,0,0);
  await route(0,800,1000,800,-2,true);
  await route(0,-800,1000,-800,1);
  let w=tw.world.value;const first=w.edges[0];
  for(const node of [first.a,first.b]){
   const a=await client.edit({op:'addStation',at:{kind:'node',node},platform:200,name:'Station '+node});if(!a.ok)throw Error(JSON.stringify(a));
  }
  for(let i=0;i<2;i++){
   const a=await client.edit({op:'addLine',stops:[first.a,first.b],name:'Test '+i,colour:i?'#007700':'#cc0044'});if(!a.ok)throw Error(JSON.stringify(a));
  }
  w=tw.world.value;
  return {items:w.money.blueprintItems,total:w.money.blueprintItems.reduce((s,r)=>s+r.cost,0),before:tw.moneyState.value.cash};
 });
 await page.waitForTimeout(300);
 await page.locator('.build-options').evaluate(e=>{e.scrollTop=e.scrollHeight;});
 await page.screenshot({path:'trainworld/T-100-build-narrow.png'});
 const layout=await page.locator('.build-options').evaluate(e=>({width:e.clientWidth,scrollWidth:e.scrollWidth}));
 if(layout.scrollWidth>layout.width+1)throw Error('Build pane overflows horizontally');
 const total=Number(await page.locator('[data-total]').getAttribute('data-total'));
 if(Math.abs(total-fixture.total)>.01)throw Error('UI total differs from worker');
 await page.locator('.build-options').evaluate(e=>{e.scrollTop=0;});
 await page.screenshot({path:'trainworld/T-100-build-controls.png'});
 const handle=await page.locator('.pane-resize').boundingBox();
 await page.mouse.move(handle.x+handle.width/2,handle.y+handle.height/2);
 await page.mouse.down();await page.mouse.move(handle.x+handle.width/2,780,{steps:10});await page.mouse.up();
 await page.screenshot({path:'trainworld/T-100-build-full.png'});
 await page.locator('[title="Undo (Ctrl+Z)"]').click();
 await page.locator('[title="Redo (Ctrl+Y)"]').click();
 await page.getByRole('button',{name:'Construct blueprints'}).click();
 await page.waitForFunction(()=>window.tw.world.value.money.blueprintItems.length===0);
 const charge=await page.evaluate(before=>before-window.tw.moneyState.value.cash,fixture.before);
 if(Math.abs(charge-fixture.total/1e6)>1e-6)throw Error(`Quote ${fixture.total/1e6} differs from charge ${charge}`);
 await page.getByRole('button',{name:'Lines',exact:true}).click();
 if(await page.getByRole('button',{name:'Construct blueprints'}).count())throw Error('Construct shown outside Build');
 await page.getByRole('button',{name:'Build',exact:true}).click();
 await page.screenshot({path:'trainworld/T-100-built.png'});
 const result={fixture,charge,layout,errors};
 console.log(JSON.stringify(result));await writeFile('trainworld/notes/T-100-check.json',JSON.stringify(result,null,2));
 if(errors.length)throw Error(errors.join('\n'));
}finally{await browser.close();}
