import test from 'node:test';import assert from 'node:assert/strict';
import {AniMatchEngine,dot} from '../packages/engine/src/index.js';
import type {Anime} from '../packages/shared/src/index.js';
const make=(id:number,title:string,genres:string[],entryId:number|null=null,prereqId:number|null=null):Anime=>({id,title,titleEnglish:title,type:'TV',year:2020,score:8.4,popularity:50,episodes:12,status:'Finished Airing',airedFrom:'2020-01-01',genres,themes:[],demographics:[],studios:['Studio A'],synopsis:'A story about friendship and adventure in a new world.',imageUrl:'',artstyles:[],visualTags:[],entryId,prereqId});
const data=[make(1,'Alpha', ['Action']),make(2,'Beta',['Romance']),make(3,'Alpha Season 2',['Action'],1,1),make(4,'Gamma',['Fantasy']),make(5,'Delta',['Adventure'])];
const p={genres:['Action'],themes:[],demographics:[],artstyles:[]};
test('scores are finite and vectors have positive similarity with self',()=>{const e=new AniMatchEngine(data);assert.ok(dot(e.vector(1),e.vector(1))>0);assert.ok(e.recommend(p,[]).every(x=>Number.isFinite(x.relevance)))});
test('franchise prerequisite prevents sequels before watching previous part',()=>{const e=new AniMatchEngine(data);const deck=e.recommend(p,[],{size:5,minScore:0});assert.ok(!deck.some(x=>x.anime.id===3));});
test('already swiped titles are excluded',()=>{const e=new AniMatchEngine(data);assert.ok(!e.recommend(p,[{animeId:1,action:'like'}],{minScore:0}).some(x=>x.anime.id===1))});
test('rocchio like changes user preference vector',()=>{const e=new AniMatchEngine(data);const before=e.userVector(p,[]),after=e.userVector(p,[{animeId:2,action:'like'}]);assert.notDeepEqual([...before],[...after])});
test('recommendation deterministic for same inputs',()=>{const e=new AniMatchEngine(data);const x=e.recommend(p,[],{size:3}),y=e.recommend(p,[],{size:3});assert.deepEqual(x,y)});

test('blocks unreleased anime and future premieres',()=>{const future={...make(20,'Future',['Action']),status:'Not yet aired',airedFrom:'2030-01-01'};const e=new AniMatchEngine([...data,future]);assert.ok(!e.recommend(p,[],{size:20,minScore:0}).some(x=>x.anime.id===20))});
test('respects duration preference',()=>{const e=new AniMatchEngine(data);assert.ok(e.recommend({...p,lengthPreference:'one'},[],{size:10,minScore:0}).length===0);assert.ok(e.recommend({...p,lengthPreference:'short'},[],{size:10,minScore:0}).length>0)});
test('pivot produces stable finite scores',()=>{const e=new AniMatchEngine(data);const results=e.recommend(p,[],{size:3,pivot:true});assert.ok(results.every(x=>Number.isFinite(x.relevance)))});

test('rated N/A is never recommended, including pivot and relaxed fallback',()=>{
 const unrated={...make(90,'Unrated Feature',['Action']),score:null};
 const e=new AniMatchEngine([...data,unrated]);
 for(const pivot of [false,true])assert.ok(!e.recommend(p,[],{size:24,pivot}).some(x=>x.anime.id===90));
});
test('1-episode mode is strictly Movies with one episode',()=>{
 const film={...make(91,'Film A',['Action']),type:'Movie',episodes:1};
 const tv={...make(92,'TV Special B',['Action']),type:'TV',episodes:1,score:9.4};
 const e=new AniMatchEngine([...data,film,tv]);
 const deck=e.recommend({...p,lengthPreference:'one'},[],{size:24,pivot:true});
 assert.ok(deck.some(x=>x.anime.id===91));
 assert.ok(deck.every(x=>x.anime.type==='Movie'&&x.anime.episodes===1));
});
test('rating 8+ receives priority over lower scores when preference similarity ties',()=>{
 const hi={...make(93,'Strong Feature',['Action']),score:8.5};
 const lo={...make(94,'Average Feature',['Action']),score:7.5};
 const e=new AniMatchEngine([hi,lo]);
 assert.equal(e.recommend(p,[],{size:2})[0]?.anime.id,93);
});
