// Run the actual demand worker module with Node's worker-message/wasm-fetch adapters.
// No browser or UI state is controlled by this harness.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {Worker} from 'node:worker_threads';
import {fileURLToPath} from 'node:url';
import {rolldown} from '../app/node_modules/rolldown/dist/index.mjs';
const artifacts=new URL('../../artifacts/trainworld/t066/',import.meta.url);
mkdirSync(artifacts,{recursive:true});
const compiled=new URL('demand-worker.mjs',artifacts), wrapper=new URL('worker-adapter.mjs',artifacts);
const wasm=new URL('../app/src/wasm/demand/sim_bg.wasm',import.meta.url);
const bundle=await rolldown({input:fileURLToPath(new URL('../app/src/workers/demand.worker.ts',import.meta.url)),platform:'browser'});
await bundle.write({file:fileURLToPath(compiled),format:'esm'});await bundle.close();
writeFileSync(wrapper,`import {parentPort} from 'node:worker_threads';
import {readFileSync} from 'node:fs';
globalThis.onmessage=null;
globalThis.postMessage=(data,options={})=>parentPort.postMessage(data,options.transfer??[]);
globalThis.fetch=async()=>new Response(readFileSync(${JSON.stringify(fileURLToPath(wasm))}),{headers:{'Content-Type':'application/wasm'}});
await import(${JSON.stringify(compiled.href)});
parentPort.on('message',data=>globalThis.onmessage({data}));
parentPort.postMessage({kind:'boot'});
`);
const w=new Worker(wrapper),messages=[],waiters=[];
w.on('message',msg=>{messages.push(msg);for(const f of [...waiters])f(msg);});
w.on('error',e=>{console.error(e);process.exitCode=1;});
const next=(predicate)=>new Promise((resolve,reject)=>{
  const existing=messages.find(predicate);if(existing)return resolve(existing);
  const timeout=setTimeout(()=>reject(new Error('worker timed out')),60000);
  const f=m=>{if(predicate(m)){clearTimeout(timeout);waiters.splice(waiters.indexOf(f),1);resolve(m);}};waiters.push(f);
});
await next(m=>m.kind==='boot');
const registry=JSON.parse(readFileSync(new URL('../app/src/game/registry.json',import.meta.url)));
const packs=registry.cities.map(({id})=>({id,header:readFileSync(new URL(`../data/packs/${id}.json`,import.meta.url),'utf8'),bin:Uint8Array.from(readFileSync(new URL(`../data/packs/${id}.bin`,import.meta.url))).buffer}));
w.postMessage({kind:'init',packs,intercity:readFileSync(new URL(`../data/packs/${registry.market.pack}.json`,import.meta.url),'utf8'),periods:[0,1,2,3,4]});
const ready=await next(m=>m.kind==='ready'||m.kind==='failed');assert.equal(ready.kind,'ready',ready.error);
assert.equal(Object.keys(ready.cityInfo).length,9);
const times=new Float32Array(18);for(let lev=0;lev<3;lev++){times[lev*3]=60;times[lev*3+1]=60;times[(lev+3)*3+1]=60;times[(lev+3)*3+2]=60;}
const net={frame:{kind:'equirectangular',lon:-73.985,lat:40.758},stXY:new Float32Array([0,0,4000,0,8000,0]),lineN:new Uint32Array([3]),stops:new Uint32Array([0,1,2]),times,tph:new Float32Array([12,12,12]),cars:new Float32Array([4]),fare:{base:1,perKm:.1},hopKm:new Float32Array([4,4,0,0,4,4])};
w.postMessage({kind:'solve',version:1,net,rounds:1});
await next(m=>m.kind==='done'||m.kind==='failed');
const failure=messages.find(m=>m.kind==='failed');assert(!failure,failure?.error);
const results=messages.filter(m=>m.kind==='result'&&m.summary[0]===1);
assert.equal(results.length,5);
const sums=messages.find(m=>m.kind==='sums'&&m.city==='nyc'&&m.round===1);
assert(sums.everyday?.length>0);
const n=sums.everyday.length/6, cells=new Uint32Array([80000,80001,80002]);
w.postMessage({kind:'purposes',id:1,city:'nyc',version:1,round:1,end:1,cells});
const purpose=await next(m=>m.kind==='purposes'&&m.id===1);
assert.equal(purpose.mask,31);assert.equal(purpose.everyday.length,purpose.names.length*(30+3*n));
assert.deepEqual(purpose.names,['Shopping and errands','Social and leisure','School (metro public)','University (metro IPEDS)']);
const stride=30+3*n;
for(let mode=0;mode<3;mode++){
  let total=0;for(let kind=0;kind<purpose.names.length;kind++)for(let q=0;q<5;q++)total+=purpose.everyday[kind*stride+mode*5+q]+purpose.everyday[kind*stride+15+mode*5+q];
  const bubble=cells.reduce((a,c)=>a+sums.everyday[(mode+3)*n+c],0);
  assert(Math.abs(total-bubble)<1e-4*Math.max(1,bubble));
}
// Other cities keep their commute layer; no fabricated purpose arrays.
w.postMessage({kind:'purposes',id:2,city:'phl',version:1,round:1,end:1,cells:new Uint32Array([0])});
const other=await next(m=>m.kind==='purposes'&&m.id===2);assert.equal(other.everyday,null);assert.equal(other.work.length,30);
const total=results.reduce((a,m)=>a+m.summary[1],0),rail=results.reduce((a,m)=>a+m.summary[2],0);
for(const m of results){const fromLayers=Object.values(m.citySummary).reduce((a,v)=>a+v[1],0)+m.intercitySummary[1];assert(Math.abs(fromLayers-m.summary[1])<1e-6);}
const result={cities:9,periods:5,sharedCrowdingRounds:1,tripLegs:total,railTripLegs:rail,nycPurposeQuery:true,otherCitiesCommuteOnly:true,versionedSnapshotMatchesPurposeQuery:true,wasmMemoryMiB:results.at(-1).memBytes/1024/1024};
writeFileSync(new URL('T-066-worker-check.json',import.meta.url),JSON.stringify(result,null,2));console.log(result);
await w.terminate();
