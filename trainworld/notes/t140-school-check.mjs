// School-specific production-WASM conservation and real-source checks.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import init,{EverydayApi,DemandApi} from '../app/src/wasm/demand/sim.js';
await init({module_or_path:readFileSync(new URL('../app/src/wasm/demand/sim_bg.wasm',import.meta.url))});
const header=readFileSync(new URL('../data/packs/nyc.json',import.meta.url),'utf8'),h=JSON.parse(header);
const bin=readFileSync(new URL('../data/packs/nyc.bin',import.meta.url));
const audit=JSON.parse(readFileSync(new URL('../data/work/nyc.schools.audit.json',import.meta.url)));
const close=(a,b,label)=>assert(Math.abs(a-b)<3e-4*Math.max(1,Math.abs(b)),`${label}: ${a} != ${b}`);
const sum=v=>v.reduce((a,b)=>a+b,0);
const commute=new DemandApi(header,bin),xy=commute.cell_xy(),n=xy.length/2;
const real=new EverydayApi(header,bin);assert.equal(real.purpose_count(),h.arrays.some(a=>a.name==='university_enrollment')?4:3);
const info=Array.from(real.info());
const oldHeader=JSON.stringify({...h,arrays:h.arrays.filter(a=>!['school_children','school_enrollment','university_adults','university_enrollment'].includes(a.name))});
const old=new EverydayApi(oldHeader,bin);assert.equal(old.purpose_count(),2);
assert.deepEqual(Array.from(old.info()),info.slice(0,6));old.free();
const school=audit.schools.find(s=>Math.abs(s.lon+73.985)<.02 && Math.abs(s.lat-40.758)<.02);
assert(school);
const cx=xy[2*school.cell],cy=xy[2*school.cell+1];
const times=new Float32Array(18);
for(let lev=0;lev<3;lev++){times[lev*3]=60;times[lev*3+1]=60;times[(lev+3)*3+1]=60;times[(lev+3)*3+2]=60;}
const stations=new Float32Array([cx-4000,cy,cx,cy,cx+4000,cy]);
const network=a=>{a.set_network(stations,new Uint32Array([3]),new Uint32Array([0,1,2]),times,new Float32Array([12,12,12]),new Float32Array([4]));a.set_fares(1,.1,new Float32Array([4,4,0,0,4,4]));};
network(real);for(let q=0;q<5;q++)real.solve(q);
const cells=new Uint32Array([school.cell]),stride=30+3*n,selection=real.selection(1,cells),modes=real.cell_modes();
const v=selection.subarray(2*stride,3*stride);
const schoolLegs=sum(v.subarray(0,30)),farLegs=sum(v.subarray(30));close(schoolLegs,farLegs,'school timing vs far counts');
assert(schoolLegs>0);
const morning=[0,1,2].reduce((a,m)=>a+v[m*5],0),outward=sum(v.subarray(0,15));
const middayReturn=[0,1,2].reduce((a,m)=>a+v[15+m*5+1],0),returning=sum(v.subarray(15,30));
close(morning/outward,.94,'morning share');close(middayReturn/returning,.82,'midday return share');
for(let m=0;m<3;m++){
 let all=0;for(let kind=0;kind<real.purpose_count();kind++)for(let q=0;q<5;q++)all+=selection[kind*stride+m*5+q]+selection[kind*stride+15+m*5+q];
 close(all,modes[(m+3)*n+school.cell],'all-purpose bubble reconciliation');
}
const schoolRail=[0,1,2,3,4].reduce((a,q)=>a+v[q]+v[15+q],0);assert(schoolRail>0);
const allCells=Uint32Array.from({length:n},(_,c)=>c),all=real.selection(1,allCells);
close(sum(all.subarray(2*stride,2*stride+30)),2*info[6],'school daily budget');
assert(info[6] <= h.school.enrolled*.9+1);
// A removed school layer exactly preserves old attraction budgets and existing cell order.
const source=name=>{const a=h.arrays.find(a=>a.name===name);return Array.from(new Float32Array(bin.buffer,bin.byteOffset+a.offset,a.count));};
const children=source('school_children'),enrollment=source('school_enrollment');
assert(enrollment.every((v,c)=>v===0 || audit.schools.some(s=>s.cell===c)));
assert(children.every(v=>Number.isFinite(v) && v>=0));
if(h.school.nces_schools){
 // T-181: NCES alone, five boroughs included (no DOE building input since then)
 assert(h.school.doe_enrolled===undefined && h.school.nces_schools>5000 && h.school.nces_enrolled===h.school.enrolled);
 assert(sum(children)>3000000);
 assert(children.some((v,c)=>v>0 && (xy[2*c]<-40000 || xy[2*c]>50000)));
 assert(audit.schools.some(s=>s.source==='NCES CCD 2023-24'));
}
else assert(children.every((v,c)=>v===0 || (xy[2*c]>-40000 && xy[2*c]<50000)));
const result={matchedBuildings:h.school.matched_buildings,enrollment:h.school.enrolled,sourceEnrollmentCoverage:h.school.location_match_share,attendanceOutings:info[6],dailySchoolTripLegs:2*info[6],schoolWalkFraction:info[7],schoolMeanKm:info[8],selectedBuilding:school.building,selectedSchoolTripLegs:schoolLegs,selectedSchoolRailLegs:schoolRail,morningOutwardShare:morning/outward,middayHomewardShare:middayReturn/returning,existingLayersPreserved:true,purposeAndFarCountsReconcile:true};
writeFileSync(new URL('../../artifacts/trainworld/notes-outputs/T-140-school-check.json',import.meta.url),JSON.stringify(result,null,2));console.log(result);
real.free();commute.free();
