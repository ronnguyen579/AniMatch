import 'dotenv/config';
import pg from 'pg';
import {writeFileSync,mkdirSync} from 'node:fs';
import {AniMatchEngine} from '../packages/engine/src/index.js';
import type {Anime,Profile} from '../packages/shared/src/index.js';
const db=new pg.Pool({connectionString:process.env.DATABASE_URL});const raw=await db.query('SELECT * FROM anime ORDER BY id');await db.end();
const anime:Anime[]=raw.rows.map(r=>({id:r.id,title:r.title,titleEnglish:r.title_english,type:r.type,year:r.year,score:r.score,popularity:r.popularity,episodes:r.episodes,status:r.status||'',airedFrom:r.aired_from?String(r.aired_from).slice(0,10):null,genres:r.genres,themes:r.themes,demographics:r.demographics??[],studios:r.studios,synopsis:r.synopsis,imageUrl:r.image_url,artstyles:r.artstyle_tags,visualTags:r.visual_tags,entryId:r.entry_id,prereqId:r.prereq_id}));const engine=new AniMatchEngine(anime);
const profiles=[
 {key:'eye_stable',label:'Eye-candy ổn định',genres:['Action'],artstyles:['retro_artstyle'],change:null},
 {key:'eye_shift',label:'Eye-candy đổi thẩm mỹ',genres:['Action'],artstyles:['retro_artstyle'],change:{artstyles:['monochrome_artstyle']}},
 {key:'content_stable',label:'Content ổn định',genres:['Action','Adventure'],artstyles:[],change:null},
 {key:'content_shift',label:'Content đổi gu',genres:['Action','Adventure'],artstyles:[],change:{genres:['Romance','Slice of Life']}},
 {key:'lapsed_stable',label:'Lapsed giữ gu',genres:['Fantasy','Adventure'],artstyles:[],change:null},
 {key:'lapsed_shift',label:'Lapsed đổi gu',genres:['Fantasy','Adventure'],artstyles:[],change:{genres:['Drama','Romance']}}
] as const;
const seedCount=Math.max(1,Math.min(100,Number(process.env.SEEDS??5))),steps=Math.max(2,Math.min(100,Number(process.env.STEPS??20)));
const csv=(v:unknown)=>'"'+String(v??'').replaceAll('"','""').replaceAll('\n',' ')+'"';
let lines=['scenario,seed,turn,phase,anime_id,title,genres,artstyles,feedback,relevance,content_score,visual_score,target_genres,target_artstyles'];
for(const scenario of profiles)for(let seed=1;seed<=seedCount;seed++){
 let state=seed>>>0;const random=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return state/4294967296};
 const history:{animeId:number;action:'like'|'pass'|'watched'}[]=[];const p:Profile={id:'synthetic',displayName:'Synthetic',mode:scenario.key.startsWith('eye')?'eyecandy':scenario.key.startsWith('lapsed')?'lapsed':'content',genres:[...scenario.genres],themes:[],demographics:[],artstyles:[...scenario.artstyles]};
 for(let t=1;t<=steps;t++){
 const switchTurn=Math.floor(steps/2)+1;const after=t>=switchTurn;const change=scenario.change;if(after&&change){if('genres' in change&&change.genres)p.genres=[...change.genres];if('artstyles' in change&&change.artstyles)p.artstyles=[...change.artstyles]}
 const lapse=scenario.key.startsWith('lapsed')&&t===switchTurn; // No interactions during lapse (simulated pause).
 const out=engine.recommend(p,history,{size:6,visualWeight:scenario.key.startsWith('eye')?0.65:0.2});const first=out[0];if(!first)break;const a=first.anime;
 const targetGenre=a.genres.some(g=>p.genres.includes(g));const targetArt=a.artstyles.some(g=>p.artstyles.includes(g));const pref=scenario.key.startsWith('eye')?(.75*Number(targetArt)+.25*Number(targetGenre)):(.8*Number(targetGenre)+.2*Number(targetArt));const reply=random()<(.12+.80*pref)?'like':'pass';
 lines.push([scenario.label,seed,t,lapse?'QUAY LAI':after&&change?'SAU DOI GU':'TRUOC DOI GU',a.id,a.title,a.genres.join(' | '),a.artstyles.join(' | '),reply,first.relevance,first.contentScore,first.visualScore,p.genres.join(' | '),p.artstyles.join(' | ')].map(csv).join(','));history.push({animeId:a.id,action:reply});
 }}mkdirSync('benchmarks/results',{recursive:true});writeFileSync('benchmarks/results/sequential.csv',lines.join('\n'),'utf8');console.log(`Completed ${profiles.length} scenarios × ${seedCount} seeds × ${steps} turns (maximum).`);console.log('Saved benchmarks/results/sequential.csv');
