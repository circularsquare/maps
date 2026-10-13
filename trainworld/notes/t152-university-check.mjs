// Production-WASM optional-purpose/source checks; supervisor runs after the shared build.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import init,{EverydayApi,DemandApi} from '../app/src/wasm/demand/sim.js';
await init({module_or_path:readFileSync(new URL('../app/src/wasm/demand/sim_bg.wasm',import.meta.url))});
const header=readFileSync(new URL('../data/packs/nyc.json',import.meta.url),'utf8'), h=JSON.parse(header);
const bin=readFileSync(new URL('../data/packs/nyc.bin',import.meta.url));
const audit=JSON.parse(readFileSync(new URL('../data/work/nyc.universities.audit.json',import.meta.url)));
const close=(a,b,label)=>assert(Math.abs(a-b)<3e-4*Math.max(1,Math.abs(b)),`${label}: ${a} != ${b}`);
const sum=v=>v.reduce((a,b)=>a+b,0);
const commute=new DemandApi(header,bin),xy=commute.cell_xy(),n=xy.length/2;
const real=new EverydayApi(header,bin);assert.equal(real.purpose_count(),4);
const info=Array.from(real.info());
const oldHeader=JSON.stringify({...h,arrays:h.arrays.filter(a=>!['university_adults','university_enrollment'].includes(a.name))});
const old=new EverydayApi(oldHeader,bin);assert.equal(old.purpose_count(),3);
assert.deepEqual(Array.from(old.info()),info.slice(0,9));old.free();
const campus=audit.institutions.find(r=>r.unitid===193900);assert(campus);
const cx=xy[2*campus.cell],cy=xy[2*campus.cell+1];
const times=new Float32Array(18);
for(let lev=0;lev<3;lev++){times[lev*3]=60;times[lev*3+1]=60;times[(lev+3)*3+1]=60;times[(lev+3)*3+2]=60;}
real.set_network(new Float32Array([cx-4000,cy,cx,cy,cx+4000,cy]),new Uint32Array([3]),new Uint32Array([0,1,2]),times,new Float32Array([12,12,12]),new Float32Array([4]));
real.set_fares(1,.1,new Float32Array([4,4,0,0,4,4]));
for(let q=0;q<5;q++)real.solve(q);
const stride=30+3*n, selection=real.selection(1,new Uint32Array([campus.cell])), modes=real.cell_modes();
const university=selection.subarray(3*stride,4*stride);
close(sum(university.subarray(0,30)),sum(university.subarray(30)),'university timing/far reconciliation');
const outward=sum(university.subarray(0,15)),returning=sum(university.subarray(15,30));
assert(outward>0 && returning>0);
close([0,1,2].reduce((a,m)=>a+university[m*5],0)/outward,.65,'university morning outward');
close([0,1,2].reduce((a,m)=>a+university[15+m*5+2],0)/returning,.42,'university evening return');
for(let m=0;m<3;m++){
 let total=0;for(let kind=0;kind<4;kind++)for(let q=0;q<5;q++)total+=selection[kind*stride+m*5+q]+selection[kind*stride+15+m*5+q];
 close(total,modes[(m+3)*n+campus.cell],'all-purpose bubble reconciliation');
}
assert(info[9]>0 && info[9]<=h.university.enrolled*.6+1);
const all=real.selection(1,Uint32Array.from({length:n},(_,c)=>c));
close(sum(all.subarray(3*stride,3*stride+30)),2*info[9],'university daily budget');
const result={institutions:h.university.institutions,enrollment:h.university.enrolled,onlineExcluded:h.university.exclusively_online_excluded,
 attendanceOutings:info[9],dailyUniversityLegs:2*info[9],walkShare:info[10],meanKm:info[11],campus:campus.name,
 selectedCampusLegs:sum(university.subarray(0,30)),existingGenerationPreserved:true,purposeTimingAndBubbleCountsReconcile:true};
writeFileSync(new URL('../../artifacts/trainworld/notes-outputs/T-152-wasm-check.json',import.meta.url),JSON.stringify(result,null,2));console.log(result);
real.free();commute.free();
