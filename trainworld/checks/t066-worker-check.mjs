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
globalThis.location={search:'',hash:'',href:'http://localhost/'};
globalThis.postMessage=(data,options={})=>parentPort.postMessage(data,options.transfer??[]);
globalThis.fetch=async()=>new Response(readFileSync(${JSON.stringify(fileURLToPath(wasm))}),{headers:{'Content-Type':'application/wasm'}});
await import(${JSON.stringify(compiled.href)});
parentPort.on('message',data=>globalThis.onmessage({data}));
parentPort.postMessage({kind:'boot'});
`);
const w=new Worker(wrapper),messages=[],waiters=[];
w.on('message',msg=>{messages.push(msg);for(const f of [...waiters])f(msg);});
w.on('error',e=>{console.error(e);process.exit(1);});
const next=(predicate)=>new Promise((resolve,reject)=>{
  const existing=messages.find(predicate);if(existing)return resolve(existing);
  const timeout=setTimeout(()=>reject(new Error('worker timed out; had '+messages.map(m=>m.kind+(m.error?':'+m.error:'')).join(','))),180000);
  const f=m=>{if(predicate(m)){clearTimeout(timeout);waiters.splice(waiters.indexOf(f),1);resolve(m);}};waiters.push(f);
});
await next(m=>m.kind==='boot');
const registry=JSON.parse(readFileSync(new URL('../app/src/game/registry.json',import.meta.url)));
const packs=registry.cities.map(({id})=>({id,header:readFileSync(new URL(`../data/packs/${id}.json`,import.meta.url),'utf8'),bin:Uint8Array.from(readFileSync(new URL(`../data/packs/${id}.bin`,import.meta.url))).buffer}));
w.postMessage({kind:'init',packs,intercity:readFileSync(new URL(`../data/packs/${registry.market.pack}.json`,import.meta.url),'utf8'),periods:[0,1,2,3,4]});
const ready=await next(m=>m.kind==='ready'||m.kind==='failed');assert.equal(ready.kind,'ready',ready.error);
assert.equal(Object.keys(ready.cityInfo).length,registry.cities.length);
const times=new Float32Array(18);for(let lev=0;lev<3;lev++){times[lev*3]=60;times[lev*3+1]=60;times[(lev+3)*3+1]=60;times[(lev+3)*3+2]=60;}
const net={frame:{kind:'equirectangular',lon:-73.985,lat:40.758},stXY:new Float32Array([0,0,4000,0,8000,0]),lineN:new Uint32Array([3]),stops:new Uint32Array([0,1,2]),times,tph:new Float32Array([12,12,12]),cars:new Float32Array([4]),fare:{base:1,perKm:.1},hopKm:new Float32Array([4,4,0,0,4,4])};
w.postMessage({kind:'solve',version:1,net,rounds:1});
await next(m=>m.kind==='done'||m.kind==='failed');
const failure=messages.find(m=>m.kind==='failed');assert(!failure,failure?.error);
// A solve sends the plan, then one cluster result per period for each cluster but the last (T-392),
// then the final: results without `cluster`, one per period and round, then done. The worker
// currently sends the final's rounds twice (the generic round loop runs after the early path), so the
// last result of each period stands for the final.
const finalMsgs=messages.filter(m=>m.kind==='result'&&!m.cluster&&m.version===1&&m.summary[0]===1);
const results=[...new Map(finalMsgs.map(m=>[m.period,m])).values()];
assert.equal(results.length,5);
const plan=messages.find(m=>m.kind==='plan'&&m.version===1);
assert(plan&&plan.solving.length>0,'no plan');
const perCluster=messages.filter(m=>m.kind==='result'&&m.cluster&&m.version===1);
assert.equal(perCluster.length,5*(plan.solving.length-1),'one cluster result per period for each cluster but the last');
// per period the cluster results are disjoint and each one's figures are the final's for its cities;
// with the last cluster (sent only in the final) they cover every city of the final
for(const final of results){
  const mine=perCluster.filter(m=>m.period===final.period);
  const seen=new Set();
  for(const m of mine){
    for(const c of m.cluster.cities){assert(!seen.has(c),`period ${final.period}: ${c} in two clusters`);seen.add(c);assert.deepEqual(m.citySummary[c],final.citySummary[c],`period ${final.period} ${c}`);}
    assert.deepEqual(Object.keys(m.citySummary).sort(),[...m.cluster.cities].sort());
    if(m.cluster.market)assert.deepEqual(m.intercitySummary,final.intercitySummary);else assert.equal(m.intercitySummary.length,0);
    const sum=Object.values(m.citySummary).reduce((a,v)=>a+v[1],0)+(m.cluster.market?m.intercitySummary[1]:0);
    assert(Math.abs(sum-m.summary[1])<1e-6,`period ${final.period}: cluster total ${m.summary[1]} against ${sum}`);
  }
  for(const c of plan.solving.at(-1))assert(!seen.has(c),`period ${final.period}: ${c} is in the last cluster and also sent early`);
  const finalCities=Object.keys(final.citySummary).sort(),covered=[...seen,...plan.solving.at(-1)].sort();
  assert.deepEqual(covered.filter(c=>finalCities.includes(c)),finalCities,`period ${final.period}: clusters cover the final's cities`);
}
assert.equal(messages.find(m=>m.kind==='done'&&m.version===1).clusters?.[1],plan.solving.length);
// Every city has the six everyday purposes since T-142 (New York since T-066/T-140/T-152): for
// each, a selection's purpose/mode/period counts reconcile with its bubble cell sums. The test
// line is in Manhattan, so the other cities' trips walk or drive.
const everydayLegs={};let id=1;
for(const {id:city} of registry.cities){
  const sums=messages.find(m=>m.kind==='sums'&&m.city===city&&m.round===1);
  assert(sums.everyday?.length>0,`${city}: no everyday sums`);
  const n=sums.everyday.length/6;
  const cells=city==='nyc'?new Uint32Array([80000,80001,80002]):new Uint32Array([Math.floor(n*.3),Math.floor(n*.5),Math.floor(n*.5)+1,Math.floor(n*.7)]);
  w.postMessage({kind:'purposes',id,city,version:1,round:1,end:1,cells});
  const purpose=await next(m=>m.kind==='purposes'&&m.id===id);id++;
  assert.equal(purpose.mask,31);
  // six since T-142's airports and medical visits (Springfield's airport purpose is there, at zero)
  assert.deepEqual(purpose.names,['Shopping and errands','Social and leisure','School','University','Airport','Medical'],city);
  assert.equal(purpose.work.length,30);
  // T-285: packed, per kind 30 timing values, m, m far cells, their rail, walk and drive legs
  const at=[];let p=0;
  for(let kind=0;kind<purpose.names.length;kind++){at.push(p);p+=31+4*purpose.everyday[p+30];}
  assert.equal(p,purpose.everyday.length,`${city}: packed layout`);
  for(let mode=0;mode<3;mode++){
    let total=0,far=0;
    for(const o of at){
      for(let q=0;q<5;q++)total+=purpose.everyday[o+mode*5+q]+purpose.everyday[o+15+mode*5+q];
      const m=purpose.everyday[o+30];for(let i=0;i<m;i++)far+=purpose.everyday[o+31+(mode+1)*m+i];
    }
    const bubble=cells.reduce((a,c)=>a+sums.everyday[(mode+3)*n+c],0);
    assert(Math.abs(total-bubble)<1e-4*Math.max(1,bubble),`${city} mode ${mode}: ${total} against ${bubble}`);
    assert(Math.abs(far-total)<1e-4*Math.max(1,total),`${city} mode ${mode}: far end ${far} against ${total}`);
  }
  // the day's everyday trip legs, all modes: each half of the sums (one per trip end) adds up to them
  let legs=0;for(let k=3*n;k<6*n;k++)legs+=sums.everyday[k];everydayLegs[city]=Math.round(legs);
}
const total=results.reduce((a,m)=>a+m.summary[1],0),rail=results.reduce((a,m)=>a+m.summary[2],0);
for(const m of results){const fromLayers=Object.values(m.citySummary).reduce((a,v)=>a+v[1],0)+m.intercitySummary[1];assert(Math.abs(fromLayers-m.summary[1])<1e-6);}
const result={cities:registry.cities.length,periods:5,sharedCrowdingRounds:1,tripLegs:total,railTripLegs:rail,everyCityPurposeQuery:true,everydayLegsPerCity:everydayLegs,versionedSnapshotMatchesPurposeQuery:true,wasmMemoryMiB:results.at(-1).memBytes/1024/1024};
writeFileSync(new URL('../../artifacts/trainworld/notes-outputs/T-066-worker-check.json',import.meta.url),JSON.stringify(result,null,2));console.log(result);
await w.terminate();
