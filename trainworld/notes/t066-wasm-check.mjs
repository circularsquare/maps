// Production WASM checks, independent of browser automation.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import init,{EverydayApi,DemandApi} from '../app/src/wasm/demand/sim.js';
await init({module_or_path:readFileSync(new URL('../app/src/wasm/demand/sim_bg.wasm',import.meta.url))});
const close=(a,b,label)=>assert(Math.abs(a-b)<2e-4*Math.max(1,Math.abs(b)),`${label}: ${a} != ${b}`);
const sum=v=>v.reduce((a,b)=>a+b,0);
function fixture() {
  const arrays=[], parts=[];let offset=0;
  const weights={x_m:[0,4000,8000],y_m:[0,0,0],pop:[1000,2000,500],jobs:[1000,1000,1500],errand_jobs:[100,800,100],leisure_jobs:[100,100,800]};
  for(const [name,v]of Object.entries(weights)) {
    const p=new Float32Array(v);arrays.push({name,per:'cell',dtype:'f32',count:v.length,offset});parts.push(new Uint8Array(p.buffer));offset+=p.byteLength;
  }
  const bin=new Uint8Array(offset);let at=0;for(const p of parts){bin.set(p,at);at+=p.length;}
  return [JSON.stringify({format:1,city:'fixture',cells:3,arrays}),bin];
}
const [h,b]=fixture();
function network(a,tph=[12,12,12]){
  const times=new Float32Array(18);
  for(let level=0;level<3;level++){
    times[level*3]=60;times[level*3+1]=60;
    times[(level+3)*3+1]=60;times[(level+3)*3+2]=60;
  }
  a.set_network(new Float32Array([0,0,4000,0,8000,0]),new Uint32Array([3]),new Uint32Array([0,1,2]),times,new Float32Array(tph),new Float32Array([4]));
  a.set_fares(1,0.1,new Float32Array([4,4,0,0,4,4]));
}
const full=new EverydayApi(h,b),one=new EverydayApi(h,b),two=new EverydayApi(h,b);
for(const a of [full,one,two])network(a);
for(const q of [0,1,2,3,4])full.solve(q);
for(const q of [0,3])one.solve(q);
for(const q of [1,2,4])two.solve(q);
const modes=full.cell_modes(),m1=one.cell_modes(),m2=two.cell_modes();
modes.forEach((v,k)=>close(m1[k]+m2[k],v,'pooled cell modes'));
const picked=new Uint32Array([1]);
const selected=full.selection(1,picked),s1=one.selection(1,picked),s2=two.selection(1,picked);
selected.forEach((v,k)=>close(s1[k]+s2[k],v,'pooled purposes'));
for(let kind=0;kind<2;kind++){
  const o=kind*39;
  close(sum(selected.subarray(o,o+30)),sum(selected.subarray(o+30,o+39)),'timing vs far counts');
}
for(let mode=0;mode<3;mode++){
  const selectedTotal=[0,1].reduce((a,kind)=>a+sum(selected.subarray(kind*39+mode*5,kind*39+mode*5+5))+sum(selected.subarray(kind*39+15+mode*5,kind*39+20+mode*5)),0);
  close(selectedTotal,modes[(mode+3)*3+1],'selected purpose vs bubble');
}
const commute=new DemandApi(h,b);network(commute);
for(let q=0;q<5;q++)commute.solve(q);
const work=commute.purpose_timing(1,picked),workModes=commute.cell_modes(commute.sub_sums());
for(let m=0;m<3;m++)close(sum(work.subarray(m*5,m*5+5))+sum(work.subarray(15+m*5,20+m*5)),2*workModes[(3+m)*3+1],'work timing vs bubble');
const preFare=sum(Array.from({length:5},(_,q)=>full.summary(q)[2]));
full.set_fares(100,10,new Float32Array([4,4,0,0,4,4]));for(let q=0;q<5;q++)full.solve(q);
assert(sum(Array.from({length:5},(_,q)=>full.summary(q)[2]))<preFare/10);
for(const a of [full,one,two,commute])a.free();

const header=readFileSync(new URL('../data/packs/nyc.json',import.meta.url),'utf8');
const bin=readFileSync(new URL('../data/packs/nyc.bin',import.meta.url));
let t=performance.now();const real=new EverydayApi(header,bin);const openMs=performance.now()-t;
const info=Array.from(real.info()), n=JSON.parse(header).cells;
real.set_network(new Float32Array(),new Uint32Array(),new Uint32Array(),new Float32Array(),new Float32Array(),new Float32Array());
for(let q=0;q<5;q++)real.solve(q);
const emptyModes=real.cell_modes(),daily=Array.from({length:5},(_,q)=>Array.from(real.summary(q)));
const total=sum(daily.map(v=>v[1]));
close(sum(emptyModes.subarray(0,3*n)),total,'real home total');
close(sum(emptyModes.subarray(3*n)),total,'real destination total');
assert(daily.every(v=>v[2]===0 && v[3]>=0 && v[3]<=v[1]));
// A small central line: query a mixed-purpose destination and check common loads/fare effects.
network(real);for(let q=0;q<5;q++)real.solve(q);
const city=new DemandApi(header,bin),xy=city.cell_xy(),chosen=[];
for(let c=0;c<n;c++)if(Math.abs(xy[2*c]-4000)<700 && Math.abs(xy[2*c+1])<700)chosen.push(c);
assert(chosen.length>0);
const local=real.selection(1,new Uint32Array(chosen)),realModes=real.cell_modes(),stride=30+3*n;
for(let mode=0;mode<3;mode++){
  let p=0;for(let kind=0;kind<real.purpose_count();kind++)for(let q=0;q<5;q++)p+=local[kind*stride+mode*5+q]+local[kind*stride+15+mode*5+q];
  close(p,chosen.reduce((a,c)=>a+realModes[(mode+3)*n+c],0),'real destination purpose vs bubble');
}
const result={fixture:{pooledModes:true,pooledPurposes:true,workTimingReconciles:true,faresReduceRail:true},nyc:{cells:n,openMs,errandOutings:info[0],errandWalkFraction:info[1],errandMeanKm:info[2],socialOutings:info[3],socialWalkFraction:info[4],socialMeanKm:info[5],commuteMeanKm:JSON.parse(header).gravity.mean_trip_km,dailyNewTripLegs:total,selectedCells:chosen.length,lineRailTrips:sum(Array.from({length:5},(_,q)=>real.summary(q)[2]))}};
writeFileSync(new URL('T-066-wasm-check.json',import.meta.url),JSON.stringify(result,null,2));console.log(result);
real.free();city.free();
