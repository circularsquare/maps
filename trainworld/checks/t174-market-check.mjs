// T-174: the regional market in the production WASM, opened from its pack's derived block and
// without it (derived at opening, as before T-174), on the real New York network plus one
// Philadelphia line: open times, and every output the app reads compared.
//   node trainworld/checks/t174-market-check.mjs        (after npm run wasm; writes artifacts/trainworld/notes-outputs/T-174-market-check.json)
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import init,{IntercityApi} from '../app/src/wasm/demand/sim.js';
await init({module_or_path:readFileSync(new URL('../app/src/wasm/demand/sim_bg.wasm',import.meta.url))});
const text=readFileSync(new URL('../data/packs/intercity.json',import.meta.url),'utf8');
// the block is the pack's last key (bin/market_pack.rs); its arrays hold no braces
const bare=text.replace(/,"derived":\{[^}]*\}(?=\}$)/,'');
assert(bare.length<text.length,'the pack has no derived block: run pipeline/intercity.py --derive');
const open=(t)=>{const t0=performance.now();const a=new IntercityApi(t);return [a,performance.now()-t0];};
const [withBlock,openWith]=open(text),[without,openWithout]=open(bare);
assert.equal(withBlock.info()[3],1,'block not taken as current');
assert.equal(without.info()[3],0);
// the real New York network (pipeline/realnet.py) and one Philadelphia line, as the Rust test
const d=JSON.parse(readFileSync(new URL('../data/work/realnet/nyc_real_track.json',import.meta.url),'utf8')).demand;
const xy=[...d.st_xy],lineN=[...d.line_n],stops=[...d.stops],times=[...d.times],tph=[...d.tph],cars=[...d.cars];
const R=6371008.8,rad=Math.PI/180;
const wx=Math.fround(R*(-75.1636+73.985)*rad*Math.cos(40.758*rad)),wy=R*(39.9526-40.758)*rad;
const s0=xy.length/2;
xy.push(wx,Math.fround(wy-2000),wx,Math.fround(wy+2000));lineN.push(2);stops.push(s0,s0+1);
for(let run=0;run<2;run++)for(let k=0;k<3;k++)times.push(...(run===0?[600,0]:[0,600]));
tph.push(4,4,2);cars.push(8);
const f32=Float32Array.from(xy),hop=[];let off=0;
for(const n of lineN){for(let dir=0;dir<2;dir++)for(let i=0;i<n;i++){const k=dir===0?off+i:off+n-1-i,next=dir===0?k+1:k-1;
  hop.push(i+1<n?Math.hypot(f32[2*stops[k]]-f32[2*stops[next]],f32[2*stops[k]+1]-f32[2*stops[next]+1])/1000:0);}off+=n;}
const ms={};
for(const [name,a] of [['withBlock',withBlock],['without',without]]){
  let t=performance.now();
  a.set_network(f32,Uint32Array.from(lineN),Uint32Array.from(stops),Float32Array.from(times),Float32Array.from(tph),Float32Array.from(cars));
  a.set_fares(2.9,0.1,Float32Array.from(hop));
  ms[name+'SetNetwork']=performance.now()-t;t=performance.now();
  for(let q=0;q<5;q++)assert(a.solve(q));
  for(let r=0;r<3;r++)for(let q=0;q<5;q++)assert(a.crowd_with_external(q,a.seg(q)));
  ms[name+'Solve']=performance.now()-t;
}
const sites=JSON.parse(bare).sites,markets=withBlock.market_count();
const sels=[[0,[0,0,1500,-800,-3000,-7000],[1,0.5,1]],[1,[wx,wy,wx+700,wy+900],[1,1]]];
for(let m=0;m<markets;m++){const pts=sites.filter(s=>s[4]===m).filter((_,i)=>i%97===0).slice(0,6);sels.push([m,pts.flatMap(s=>[s[0],s[1]]),pts.map(()=>1)]);}
const outputs=(a)=>{
  const o={info:Array.from(a.info()).slice(0,3)};
  for(let q=0;q<5;q++){
    const s=Array.from(a.summary(q));s[4]=0;o['summary'+q]=s;
    o['corridors'+q]=Array.from(a.corridor_summary(q));
    for(const k of ['seg','board','alight','entries','exits','load_of_crush'])o[k+q]=Array.from(a[k](q));
  }
  for(const st of [0,1,10,100,300,500,s0-1,s0,s0+1])o['station'+st]=Array.from(a.station_flows(st));
  sels.forEach(([m,p,sh],k)=>{o['selection'+k]=Array.from(a.selection_flows(m,Float32Array.from(p),Float32Array.from(sh)));});
  return o;
};
const A=outputs(withBlock),B=outputs(without);
let identical=0,worst=0,worstKey='';const differing=[];
for(const k of Object.keys(B)){
  assert.equal(A[k].length,B[k].length,k);
  const scale=Math.max(1e-12,...B[k].map(Math.abs));
  const diff=Math.max(0,...A[k].map((v,i)=>Math.abs(v-B[k][i])));
  if(diff===0)identical++;else differing.push(k);
  if(diff/scale>worst){worst=diff/scale;worstKey=k;}
}
// float noise only: the block is derived natively, whose exp and powf can differ from the
// browser's in the last bits; f32 outputs then move by an ulp or two (6e-8 each)
assert(worst<1e-5,`outputs differ: ${worstKey} ${worst}`);
const railDay=(o)=>[0,1,2,3,4].reduce((s,q)=>s+o['summary'+q][2],0);
const result={sites:A.info[0],zones:A.info[1],tripsPerDay:A.info[2],pairsServedAndAll:Array.from(withBlock.pair_counts()),
  openMsWithBlock:Math.round(openWith),openMsDerivedAtOpening:Math.round(openWithout),
  ...Object.fromEntries(Object.entries(ms).map(([k,v])=>[k+'Ms',Math.round(v)])),
  railTripsPerDay:railDay(A),outputs:Object.keys(B).length,identical,worstRelativeDifference:worst,worstKey,differing:differing.length};
writeFileSync(new URL('../../artifacts/trainworld/notes-outputs/T-174-market-check.json',import.meta.url),JSON.stringify(result,null,2));console.log(result);
withBlock.free();without.free();
