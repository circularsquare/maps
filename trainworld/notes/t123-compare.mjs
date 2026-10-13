import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
const json=async name=>JSON.parse(await readFile(`artifacts/trainworld/notes-outputs/${name}.json`,'utf8'));
const [nativeBefore,nativeAfter,before,after]=await Promise.all([
 json('T-123-before-results'),json('T-123-after-results'),
 json('T-119-browser-t123-before'),json('T-119-browser-t123-final'),
]);
let worst=0,values=0;
for(const [key,v]of Object.entries(nativeBefore)) {
 const a=Array.isArray(v)?v:[v],b=Array.isArray(nativeAfter[key])?nativeAfter[key]:[nativeAfter[key]];
 assert.equal(a.length,b.length,key);
 const scale=Math.max(1e-9,...a.map(Math.abs));
 for(let i=0;i<a.length;i++) {
  if(key.startsWith('sum_')&&i===4)continue; // elapsed milliseconds are not a demand result
  worst=Math.max(worst,Math.abs(a[i]-b[i])/scale);values++;
 }
}
assert(worst<1e-12,`Native result changed: ${worst}`);
const compare=(a,b,path='summary')=>{
 if(typeof a==='number')assert(Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a)),path);
 else if(a&&typeof a==='object'){
  assert.deepEqual(Object.keys(a),Object.keys(b),path);
  for(const k of Object.keys(a))compare(a[k],b[k],`${path}.${k}`);
 }else assert.equal(a,b,path);
};
compare(before.summary,after.summary);
const memory=r=>({openingMB:r.start.workers.reduce((n,w)=>n+w.memBytes/1e6,0),
 combinedMB:r.combined.memMB.reduce((a,b)=>a+b,0)+r.start.workers.at(-1).memBytes/1e6,
 perSolverMB:r.combined.memMB});
const oldMemory=memory(before),newMemory=memory(after);
assert(newMemory.combinedMB<oldMemory.combinedMB*.97,'Need a meaningful full-network memory reduction');
assert(after.combined.firstEstimateMs<before.combined.firstEstimateMs*1.25,'Free-flow regression');
assert(after.combined.refinedMs<before.combined.refinedMs*1.25,'Refinement regression');
const result={nativeValuesCompared:values,nativeWorstRelativeDifference:worst,browserResultsMatch:true,
 before:oldMemory,after:newMemory,savedMB:oldMemory.combinedMB-newMemory.combinedMB,
 reductionPct:100*(1-newMemory.combinedMB/oldMemory.combinedMB),
 firstEstimateMs:[before.combined.firstEstimateMs,after.combined.firstEstimateMs],
 refinedMs:[before.combined.refinedMs,after.combined.refinedMs],
 remoteBasemapUnavailable:after.errors.length>0};
await writeFile('artifacts/trainworld/notes-outputs/T-123-check.json',JSON.stringify(result,null,2)+'\n');console.log(result);
