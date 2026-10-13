import {writeFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE??'playwright');
const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
try {
 const page=await browser.newPage({viewport:{width:900,height:750}}),errors=[],checks=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
 await page.waitForFunction(()=>window.tw?.world.value&&window.tw.overlay.camMatrix);
 await page.getByRole('button',{name:'Lines',exact:true}).click();
 await page.getByRole('button',{name:'New line',exact:true}).click();
 await page.waitForFunction(()=>document.activeElement?.classList.contains('name-edit'));
 await page.keyboard.press('Enter');
 const modal=page.getByRole('dialog');
 for (const width of [900,400]) {
  await page.setViewportSize({width,height:750});
  await page.getByRole('button',{name:'Delete line',exact:true}).click();
  await modal.waitFor({state:'visible'});
  const box=await modal.boundingBox();
  if(Math.abs(box.x+box.width/2-width/2)>2||Math.abs(box.y+box.height/2-375)>2)throw Error('Dialog not centered '+JSON.stringify(box));
  if(await page.locator(':focus').innerText()!=='Keep line')throw Error('Destructive action initially focused');
  await page.keyboard.press('Shift+Tab');
  if(await page.locator(':focus').innerText()!=='Delete line')throw Error('Focus trap failed');
  await page.keyboard.press('Tab');
  if(await page.locator(':focus').innerText()!=='Keep line')throw Error('Focus did not wrap');
  const newLine=await page.getByRole('button',{name:'New line',exact:true}).boundingBox();
  await page.mouse.click(newLine.x+newLine.width/2,newLine.y+newLine.height/2);
  if(await page.evaluate(()=>window.tw.world.value.save.lines.length)!==1)throw Error('Background interaction created line');
  await page.keyboard.press('1');
  if(await modal.count()!==1)throw Error('Game shortcut dismissed dialog');
  await page.keyboard.press('Escape');
  await modal.waitFor({state:'detached'});
  if(await page.evaluate(()=>window.tw.world.value.save.lines.length)!==1)throw Error('Escape deleted line');
  if(await page.locator(':focus').innerText()!=='Delete line')throw Error('Focus not restored');
  checks.push({width,box});
 }
 await page.getByRole('button',{name:'Delete line',exact:true}).click();
 await page.screenshot({path:'artifacts/trainworld/T-113-confirmation.png'});
 await modal.getByRole('button',{name:'Delete line',exact:true}).click();
 await page.waitForFunction(()=>window.tw.world.value.save.lines.length===0);
 // the City tab is gone (T-220): no City tab, and search goes to a city
 if(await page.getByRole('button',{name:'City',exact:true}).count())throw Error('City tab remains');
 await page.evaluate(()=>window.tw.map.jumpTo({center:[-74.8,40.2],zoom:9}));
 await page.getByRole('button',{name:'Search stations, lines and places'}).click();
 await page.getByRole('combobox',{name:'Search stations, lines and places'}).fill('New York');
 await page.keyboard.press('Enter');
 await page.waitForFunction(()=>Math.abs(window.tw.map.getCenter().lng+73.985)<.001);
 if(errors.length)throw Error(errors.join('\n'));
 const result={checks,confirmed:true,cityNavigation:true,errors};console.log(JSON.stringify(result));
 await writeFile('artifacts/trainworld/notes-outputs/T-113-check.json',JSON.stringify(result,null,2));
}finally{await browser.close();}
