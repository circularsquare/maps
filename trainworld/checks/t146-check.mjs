// Run from maps: node trainworld/checks/t146-check.mjs
// Station complexes (T-146): grouping by platform gap and the demand-merge rule.
import assert from 'node:assert/strict';
import {loadApp} from './lib.mjs';
const {buildComplexes,membersOf,segmentGap,PLATFORM_GAP_M,DEMAND_MERGE_M}=await loadApp(`
 export * from '@app/src/game/complexes.ts';`);
const near=(a,b,m='')=>assert.ok(Math.abs(a-b)<1e-6,`${m} ${a} != ${b}`);
assert.equal(PLATFORM_GAP_M,50);assert.equal(DEMAND_MERGE_M,100);

// segment distances
near(segmentGap(0,0,10,0, 0,5,10,5),5,'parallel overlapping');
near(segmentGap(0,0,10,0, 13,4,20,4),5,'parallel, end to end diagonal');
near(segmentGap(0,0,10,0, 5,-5,5,5),0,'crossing');
near(segmentGap(0,0,10,0, 15,0,25,0),5,'collinear gap');
near(segmentGap(0,0,0,0, 3,4,3,4),5,'two points');
near(segmentGap(0,0,10,0, 5,3,5,3),3,'point over a segment');

const S=(id,x,y,heading=0,length=200)=>({id,x,y,heading,length});
const groups=c=>c.members.map(m=>m.join(',')).sort();

// side by side 40 m apart: one complex; 60 m apart: two (centres over 100 m apart along the track)
assert.deepEqual(groups(buildComplexes([S('1',0,0),S('2',0,40)])),['1,2']);
assert.deepEqual(groups(buildComplexes([S('1',0,0),S('2',150,60)])),['1','2'],'60 m sideways, 150 m along: apart');
// side by side 60-100 m: the demand model already merges them (centres within 100 m)
assert.deepEqual(groups(buildComplexes([S('1',0,0),S('2',0,90)])),['1,2'],'demand merge rule');
// end to end along one track: gap 30 m between 200 m platforms (centres 230 m apart)
assert.deepEqual(groups(buildComplexes([S('1',0,0),S('2',230,0)])),['1,2']);
assert.deepEqual(groups(buildComplexes([S('1',0,0),S('2',260,0)])),['1','2'],'60 m end gap');
// crossing at right angles (a line over or under another), 120 m from centre to centre
assert.deepEqual(groups(buildComplexes([S('1',0,0),S('2',120,0,Math.PI/2)])),['1,2'],'T crossing: the long platform reaches it');
assert.deepEqual(groups(buildComplexes([S('1',0,0,0,60),S('2',200,0,Math.PI/2,60)])),['1','2'],'short platforms far apart');
// chains: 1-2 and 2-3 linked, 1-3 not
const chain=buildComplexes([S('1',0,0),S('2',0,45),S('3',0,90+45),S('4',5000,0)]);
assert.deepEqual(groups(chain),['1,2,3','4']);
assert.deepEqual(membersOf(chain,'3'),['1','2','3']);
assert.deepEqual(membersOf(chain,'4'),['4']);
assert.deepEqual(membersOf(chain,'missing'),['missing']);
// across grid cell boundaries and negative coordinates
assert.deepEqual(groups(buildComplexes([S('1',-1,-1),S('2',1,30)])),['1,2']);
// many stations: each pair of a parallel row is linked, rows are far apart
const many=[];for(let k=0;k<300;k++){many.push(S(`a${k}`,k*1000,0),S(`b${k}`,k*1000,35));}
const mc=buildComplexes(many);assert.equal(mc.members.length,300);assert.ok(mc.members.every(m=>m.length===2));
// T-197: the player's links join complexes whatever the distance; a link naming a station not in
// the list is ignored; a link inside a complex changes nothing
const far=[S('1',0,0),S('2',0,40),S('3',400,0,Math.PI/2),S('4',3000,0)];
assert.deepEqual(groups(buildComplexes(far,[['2','3']])),['1,2,3','4'],'a link joins two complexes');
assert.deepEqual(groups(buildComplexes(far,[['3','9']])),['1,2','3','4'],'a link to a missing station');
assert.deepEqual(groups(buildComplexes(far,[['1','2']])),['1,2','3','4'],'a link inside a complex');
assert.deepEqual(groups(buildComplexes(far,[['4','3'],['2','3']])),['1,2,3,4'],'chained links');

console.log('t146 ok');
