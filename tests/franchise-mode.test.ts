import test from 'node:test';
import assert from 'node:assert/strict';
import type {Anime,Profile} from '../packages/shared/src/index.js';
import {MODE_WEIGHTS,computeTasteSignals,AniMatchEngine} from '../packages/engine/src/index.js';
import {datasetMissing,titlePrerequisites,sequelHint} from '../apps/api/src/franchiseGate.js';
function anime(id:number,title:string,genres=['Action'],themes=['Military'],artstyles=['retro_artstyle']):Anime{
 return {id,title,titleEnglish:title,type:'TV',year:2022,score:8.8,popularity:500,episodes:12,status:'Finished Airing',airedFrom:'2022-01-01',genres,themes,demographics:[],studios:[],synopsis:'A series about a mysterious world',imageUrl:'',artstyles,visualTags:[],entryId:id,prereqId:null};
}
const bleach=anime(269,'Bleach');
const tybw=anime(41467,'Bleach: Sennen Kessen-hen');
const part2=anime(53998,'Bleach: Sennen Kessen-hen - Ketsubetsu-tan');
const list=[bleach,tybw,part2,anime(800,'Other')];
test('Bleach TYBW is locked without entry-level original even if CSV chain is [41467]',()=>{
 assert.deepEqual(titlePrerequisites(tybw,list),[269]);
 assert.deepEqual(datasetMissing(tybw,list,new Set()).map(r=>r.idMal),[269]);
 assert.deepEqual(datasetMissing(tybw,list,new Set([269])),[]);
});
test('Later Bleach TYBW parts require both original and preceding part',()=>{
 assert.deepEqual(new Set(titlePrerequisites(part2,list)),new Set([269,41467]));
 assert.deepEqual(datasetMissing(part2,list,new Set([269])).map(r=>r.idMal),[41467]);
});
test('Unknown season 2 needs graph verification',()=>{
 assert.equal(sequelHint(anime(420,'Mystery Show Season 2')),true);
});
test('Content, Lapsed and Eye-Candy assign different explicit channel weights',()=>{
 assert.ok(MODE_WEIGHTS.eyecandy.visual>MODE_WEIGHTS.content.visual);
 assert.ok(MODE_WEIGHTS.content.genre>MODE_WEIGHTS.eyecandy.genre);
 const engine=new AniMatchEngine(list);
 const base:Profile={id:'test',displayName:'Explorer',mode:'content',genres:['Action'],themes:['Military'],demographics:[],artstyles:[]};
 const content=computeTasteSignals(tybw,base,[],engine.byId);
 const eyecandy=computeTasteSignals(tybw,{...base,mode:'eyecandy'},[],engine.byId);
 assert.ok(content.score>eyecandy.score,'Missing visual metadata should not masquerade as visual preference evidence');
});
test('Visual signal grows from likes only when anime has a curated style tag',()=>{
 const engine=new AniMatchEngine(list);
 const base:Profile={id:'test',displayName:'Explorer',mode:'eyecandy',genres:[],themes:[],demographics:[],artstyles:[]};
 const before=computeTasteSignals(tybw,base,[],engine.byId);
 const after=computeTasteSignals(tybw,base,[{animeId:269,action:'like'}],engine.byId);
 assert.equal(before.channels[2].value,null);
 assert.ok((after.channels[2].value??0)>0);
});
