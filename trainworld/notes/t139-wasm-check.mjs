// Actual production WASM query contract and period-pool aggregation; no browser state used.
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import init, {IntercityApi} from '../app/src/wasm/demand/sim.js';
await init({module_or_path:readFileSync(new URL('../app/src/wasm/demand/sim_bg.wasm',import.meta.url))});
const header=JSON.stringify({zone_m:6000,trip_rate:0.015,markets:['NYC','PHL'],sites:[
  [0,0,100000,50000,0],[60000,0,100000,50000,0],[120000,0,100000,50000,1],
]});
const api=periods=>{
  const a=new IntercityApi(header);
  const times=new Float32Array(18);
  for(let lev=0;lev<3;lev++){
    times[lev*3]=1800;times[lev*3+1]=1800;
    times[(3+lev)*3+1]=1800;times[(3+lev)*3+2]=1800;
  }
  a.set_network(new Float32Array([0,0,60000,0,120000,0]),new Uint32Array([3]),new Uint32Array([0,1,2]),times,new Float32Array([6,6,6]),new Float32Array([4]));
  for(const q of periods)assert(a.solve(q));
  return a;
};
const full=api([0,1,2,3,4]), first=api([0,3]), second=api([1,2,4]);
const rows=a=>a.selection_flows(0,new Float32Array([0,0]),new Float32Array([1]));
const all=rows(full), r1=rows(first), r2=rows(second), head=17,nz=3,meta=head+4*nz;
assert.equal(all.length,head+7*nz);
const summed=r1.slice();
for(let k=1;k<meta;k++)summed[k]+=r2[k];
for(let k=0;k<all.length;k++)assert(Math.abs(summed[k]-all[k])<1e-6*Math.max(1,Math.abs(all[k])));
let same=0,other=0;
for(let z=0;z<nz;z++){
  const market=all[meta+3*z+2],trips=all[head+4*z];
  assert([0,1].includes(market));
  if(market===0)same+=trips;else other+=trips;
}
assert(same>0&&other>0);
assert(Math.abs(same-all[1])<1e-6);
assert(Math.abs(other-all[9])<1e-6);
const station=full.station_flows(0),sh=12,sm=sh+2*nz;
assert.equal(station.length,sh+5*nz);
for(let z=0;z<nz;z++)assert.equal(station[sm+3*z+2],all[meta+3*z+2]);
const real=new IntercityApi(readFileSync(new URL('../data/packs/intercity.json',import.meta.url),'utf8'));
const markets=real.market_count(),realRow=real.selection_flows(0,new Float32Array(),new Float32Array()),rh=1+8*markets,rn=(realRow.length-rh)/7;
assert.equal(markets,10);assert.equal(rn,real.info()[1]);
const ids=new Set(Array.from({length:rn},(_,z)=>realRow[rh+4*rn+3*z+2]));
assert.equal(ids.size,10);
const result={synthetic:{sameCityTrips:same,otherCityTrips:other,metadataPreservedAcrossPeriodPool:true,stationMembershipMatches:true},real:{markets,zones:rn,zoneMarketIds:[...ids].sort((a,b)=>a-b)}};
writeFileSync(new URL('../../artifacts/trainworld/notes-outputs/T-139-wasm-check.json',import.meta.url),JSON.stringify(result,null,2));
console.log(result);
full.free();first.free();second.free();real.free();
