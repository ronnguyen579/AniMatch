import { isVisualStyleTag, getAnimeVisualStyles } from '@animatch/shared';
import dotenv from 'dotenv';
dotenv.config({path:fileURLToPath(new URL('../../../.env',import.meta.url))});
import Fastify from 'fastify';
import cors from '@fastify/cors';
import pg from 'pg';
import { createHash, randomBytes, randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { AniMatchEngine, computeTasteSignals, MODE_WEIGHTS } from '@animatch/engine';
import {datasetMissing,sequelHint} from './franchiseGate.js';
import type { Anime, Profile, Action } from '@animatch/shared';
import { z } from 'zod';
import { getAnimeMedia } from './media.js';
import { fetchFranchiseRelations, missingPrequels, type Relation } from './anilist.js';
const db=new pg.Pool({connectionString:process.env.DATABASE_URL, max:10});
const app=Fastify({logger:true,bodyLimit:1024*128});
await app.register(cors,{origin:process.env.WEB_ORIGIN?.split(',')??['http://localhost:5173']});
const genres=z.array(z.string().trim().min(1).max(70)).max(150);
const lengthSchema=z.enum(['all','one','short','medium','long','epic']);
const profileBody=z.object({displayName:z.string().max(60).default('Anime Explorer'), mode:z.enum(['eyecandy','content','lapsed']).default('content'), genres:genres.default([]), themes:genres.default([]), demographics:genres.default([]), artstyles:genres.default([]),lengthPreference:lengthSchema.default('all'),pastGenres:genres.default([]),exploreGenres:genres.default([]),watchedAnimeIds:z.array(z.number().int().positive()).max(100).optional()});
const feedbackBody=z.object({animeId:z.number().int().positive(),action:z.enum(['like','pass','watched'])});
const recomQuery=z.object({size:z.coerce.number().int().min(1).max(24).default(8),lambda:z.coerce.number().min(0).max(1).default(.75)});
const hash=(s:string)=>createHash('sha256').update(s).digest('hex');
const rowAnime=(r:any):Anime=>({id:r.id,title:r.title,titleEnglish:r.title_english,type:r.type,year:r.year,score:r.score,popularity:r.popularity,episodes:r.episodes,status:r.status||'',airedFrom:r.aired_from?String(r.aired_from instanceof Date?r.aired_from.toISOString().slice(0,10):r.aired_from).slice(0,10):null,genres:r.genres||[],themes:r.themes||[],demographics:r.demographics||[],studios:r.studios||[],synopsis:r.synopsis,imageUrl:r.image_url,artstyles:r.artstyle_tags||[],visualTags:r.visual_tags||[],entryId:r.entry_id,prereqId:r.prereq_id});
// Apply the tiny backward-compatible beta migration even if the user skipped db:init.
// A missing profiles.mode column otherwise makes POST /api/profile fail with HTTP 500.
await db.query("ALTER TABLE IF EXISTS anime ADD COLUMN IF NOT EXISTS episodes INTEGER");
await db.query("ALTER TABLE IF EXISTS anime ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT ''");
await db.query("ALTER TABLE IF EXISTS anime ADD COLUMN IF NOT EXISTS aired_from DATE");
await db.query("ALTER TABLE IF EXISTS profiles ADD COLUMN IF NOT EXISTS length_preference TEXT NOT NULL DEFAULT 'all'");
await db.query("ALTER TABLE IF EXISTS profiles ADD COLUMN IF NOT EXISTS past_genres TEXT[] NOT NULL DEFAULT '{}'");
await db.query("ALTER TABLE IF EXISTS profiles ADD COLUMN IF NOT EXISTS explore_genres TEXT[] NOT NULL DEFAULT '{}'");
await db.query("ALTER TABLE IF EXISTS interactions ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'swipe'");
await db.query("CREATE TABLE IF NOT EXISTS reentry_queue (profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,anime_id INTEGER NOT NULL REFERENCES anime(id),eligible_after BIGINT NOT NULL DEFAULT 0,created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY(profile_id,anime_id))");
await db.query("ALTER TABLE IF EXISTS profiles ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'content'");
await db.query("ALTER TABLE IF EXISTS profiles ADD COLUMN IF NOT EXISTS demographics TEXT[] NOT NULL DEFAULT '{}'");
await db.query("ALTER TABLE IF EXISTS anime ADD COLUMN IF NOT EXISTS demographics TEXT[] NOT NULL DEFAULT '{}'");
await db.query("CREATE TABLE IF NOT EXISTS anilist_relation_cache (anime_id INTEGER PRIMARY KEY REFERENCES anime(id) ON DELETE CASCADE, relations JSONB NOT NULL DEFAULT '[]'::jsonb, fetched_at TIMESTAMPTZ NOT NULL DEFAULT now())");
await db.query("CREATE TABLE IF NOT EXISTS external_watched (profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE, mal_id INTEGER NOT NULL CHECK (mal_id>0), title TEXT NOT NULL DEFAULT '', created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY(profile_id,mal_id))");
await db.query("CREATE TABLE IF NOT EXISTS anime_media_cache (anime_id INTEGER PRIMARY KEY REFERENCES anime(id) ON DELETE CASCADE, payload JSONB NOT NULL, fetched_at TIMESTAMPTZ NOT NULL DEFAULT now())");
const rows=await db.query('SELECT * FROM anime ORDER BY id');
const catalog=rows.rows.map(rowAnime);
const engine=new AniMatchEngine(catalog);
app.log.info({anime:catalog.length},'Engine initialized');
app.addHook('onRequest', async(req,reply)=>{if(!req.url.startsWith('/api/')||req.url==='/api/health'||req.url==='/api/options'||req.url.startsWith('/api/onboarding/anime')||/^\/api\/anime\/\d+\/media(?:\?.*)?$/.test(req.url)||req.url==='/api/profile'&&req.method==='POST')return;
 const auth=req.headers.authorization??'';const token=auth.startsWith('Bearer ')?auth.slice(7):'';if(!/^[a-f0-9]{64}$/.test(token)){reply.code(401).send({error:'Missing anonymous session token'});return}
 const result=await db.query('SELECT id,display_name,mode,genres,themes,demographics,artstyles,length_preference,past_genres,explore_genres FROM profiles WHERE token_hash=$1',[hash(token)]);if(!result.rowCount){reply.code(401).send({error:'Session not found'});return} (req as any).profile=result.rows[0];});
const getProfile=(req:any):Profile=>{const x=req.profile;return {id:x.id,displayName:x.display_name,mode:x.mode||'content',genres:x.genres,themes:x.themes,demographics:x.demographics||[],artstyles:x.artstyles,lengthPreference:x.length_preference??'all',pastGenres:x.past_genres??[],exploreGenres:x.explore_genres??[]}};
const history=async(profileId:string)=>{const r=await db.query('SELECT id,anime_id,action,source,created_at FROM interactions WHERE profile_id=$1 ORDER BY id ASC',[profileId]);return r.rows.map(x=>({id:Number(x.id),animeId:x.anime_id,action:x.action as Action,source:x.source,createdAt:x.created_at}))};
const err=(reply:any,e:any)=>{if(e instanceof z.ZodError){reply.code(400).send({error:'Invalid input',details:e.issues});return true}return false};
app.get('/api/health',async()=>({ok:true,animeCount:catalog.length}));
// Explicit error responses make it possible to distinguish a Fastify exception from a Vite proxy error.
app.setErrorHandler((error, request, reply) => {
 app.log.error({err:error,method:request.method,url:request.url},'Request failed');
 const status=typeof error.statusCode==='number'&&error.statusCode>=400&&error.statusCode<500?error.statusCode:500;
 reply.status(status).send({error:status===500?'AniMatch backend encountered an error':error.message,path:request.url,code:(error as any).code||'SERVER_ERROR'});
});

app.post('/api/profile',async(req,reply)=>{try{const body=profileBody.parse(req.body||{});const id=randomUUID();const token=randomBytes(32).toString('hex');const client=await db.connect();try{await client.query('BEGIN');await client.query('INSERT INTO profiles (id,token_hash,display_name,mode,genres,themes,demographics,artstyles,length_preference,past_genres,explore_genres) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)',[id,hash(token),body.displayName,body.mode,body.genres,body.themes,body.demographics,[],body.lengthPreference,body.pastGenres,body.exploreGenres]);if(body.mode==='lapsed')for(const animeId of [...new Set(body.watchedAnimeIds??[])])if(engine.byId.has(animeId))await client.query('INSERT INTO interactions(profile_id,anime_id,action,source) VALUES ($1,$2,$3,$4)',[id,animeId,'watched','onboarding']);await client.query('COMMIT')}catch(e){await client.query('ROLLBACK');throw e}finally{client.release()}reply.code(201).send({token,profile:{id,...body,artstyles:[]}})}catch(e){if(!err(reply,e))throw e}});
app.get('/api/profile',async(req)=>({profile:getProfile(req),history:await history(getProfile(req).id)}));
app.put('/api/profile',async(req,reply)=>{try{const body=profileBody.parse(req.body);const p=getProfile(req);await db.query('UPDATE profiles SET display_name=$1,mode=$2,genres=$3,themes=$4,demographics=$5,artstyles=$6,length_preference=$7,past_genres=$8,explore_genres=$9,updated_at=now() WHERE id=$10',[body.displayName,body.mode,body.genres,body.themes,body.demographics,[],body.lengthPreference,body.pastGenres,body.exploreGenres,p.id]);return {profile:{...p,...body}}}catch(e){if(!err(reply,e))throw e}});
const visualCounts=new Map<string,number>();for(const a of catalog)for(const tag of new Set(a.visualTags))visualCounts.set(tag,(visualCounts.get(tag)??0)+1);
const meaningfulVisual=isVisualStyleTag;
const artstyleOptions=[...new Set([...catalog.flatMap(a=>a.artstyles),...[...visualCounts].filter(([tag,n])=>n>=20&&meaningfulVisual(tag)).map(([tag])=>tag)])].sort();
// Cache OPTIONS at boot so onboarding never recomputes data on each request.
// Defensively ignore malformed legacy rows and keep this endpoint database-free.
const listTags=(key:'genres'|'themes') => [...new Set(catalog.flatMap(a => Array.isArray(a[key]) ? a[key].filter((x):x is string=>typeof x==='string') : []))].sort();
const safeTags=(xs:string[])=>xs.filter(g=>!['Hentai','Erotica','Ecchi'].includes(g));
const demographicsOptions=safeTags([...new Set(catalog.flatMap(a=>a.demographics))]).sort();
const optionsPayload={genres:safeTags(listTags('genres')),themes:safeTags(listTags('themes')),demographics:demographicsOptions,artstyles:artstyleOptions,artstyleCount:artstyleOptions.length};
app.get('/api/options',async()=>optionsPayload);
app.get('/api/onboarding/anime',async(req,reply)=>{const q=z.object({q:z.string().trim().max(90).default('')}).safeParse(req.query);if(!q.success)return reply.code(400).send({error:'Invalid search'});if(q.data.q.length<2)return {items:[]};const text=q.data.q.toLowerCase();return {items:catalog.filter(a=>(a.title+' '+a.titleEnglish).toLowerCase().includes(text)&&a.airedFrom&&a.airedFrom<=new Date().toISOString().slice(0,10)&&!/not yet aired|upcoming/i.test(a.status)).slice(0,14).map(a=>({id:a.id,title:a.title,year:a.year,imageUrl:a.imageUrl,genres:a.genres}))}});

app.get('/api/anime',async(req,reply)=>{const q=z.object({q:z.string().max(120).default(''),genre:z.string().default(''),limit:z.coerce.number().int().min(1).max(100).default(30),page:z.coerce.number().int().min(1).default(1)}).safeParse(req.query);if(!q.success)return reply.code(400).send({error:'Invalid query'});const s=q.data;let list=catalog;if(s.q)list=list.filter(a=>(a.title+' '+a.titleEnglish).toLowerCase().includes(s.q.toLowerCase()));if(s.genre)list=list.filter(a=>a.genres.includes(s.genre));return {total:list.length,items:list.slice((s.page-1)*s.limit,s.page*s.limit)}});
app.get('/api/vibe',async(req)=>{
 const p=getProfile(req),h=await history(p.id);const latest=new Map<number,typeof h[number]>();for(const item of h)latest.set(item.animeId,item);
 const positive=new Map<string,number>(), negative=new Map<string,number>(),tPos=new Map<string,number>(),tNeg=new Map<string,number>(),vPos=new Map<string,number>(),vNeg=new Map<string,number>();
 let likes=0,passes=0;for(const item of latest.values()){
  const a=engine.byId.get(item.animeId);if(!a||item.action==='watched')continue;
  const pos=item.action==='like';if(pos)likes++;else passes++;
  const target=pos?positive:negative,tt=pos?tPos:tNeg,vt=pos?vPos:vNeg;
  for(const g of new Set(a.genres))target.set(g,(target.get(g)??0)+1);
  for(const t of new Set(a.themes))tt.set(t,(tt.get(t)??0)+1);
  for(const tag of new Set(getAnimeVisualStyles(a)))vt.set(tag,(vt.get(tag)??0)+1);
 }
 const toAxes=(p:Map<string,number>,n:Map<string,number>,limit:number)=>[...new Set([...p.keys(),...n.keys()])].map(name=>({name,count:p.get(name)??0,passes:n.get(name)??0,score:Math.max(-5,Math.min(5,(p.get(name)??0)-(n.get(name)??0)*.65))})).filter(a=>a.score>0.05).sort((a,b)=>Math.abs(b.score)-Math.abs(a.score)||a.name.localeCompare(b.name)).slice(0,limit);
 const last=h[h.length-1],a=last?engine.byId.get(last.animeId):null;
 return {genres:toAxes(positive,negative,12),themes:toAxes(tPos,tNeg,12),visual:toAxes(vPos,vNeg,12),visualCoverage:[...latest.values()].filter(x=>x.action==='like'||x.action==='pass').filter(x=>{const a=engine.byId.get(x.animeId);return a&&getAnimeVisualStyles(a).length>0}).length,lastLikeGenres:last?.action==='like'?(a?.genres??[]):[],lastAction:last?.action??null,likes,passes,total:h.length};
});
app.get('/api/anime/:id/media',async(req,reply)=>{
 const id=Number((req.params as any).id),anime=engine.byId.get(id);
 if(!anime)return reply.code(404).send({error:'Anime not found'});
 const rows=await db.query("SELECT payload FROM anime_media_cache WHERE anime_id=$1 AND fetched_at > now() - interval '14 days'",[id]);
 if(rows.rowCount && rows.rows[0]?.payload && (!process.env.YOUTUBE_API_KEY?.trim() || rows.rows[0].payload.trailerSource==='YouTube search #1'))return rows.rows[0].payload;
 const media=await getAnimeMedia(id,anime.imageUrl,anime.titleEnglish||anime.title,anime.year);
 // Persist successful metadata so quota usage does not grow with number of users.
 if((media.trailerYoutubeId||media.source!=='dataset')&&(!process.env.YOUTUBE_API_KEY?.trim()||media.trailerSource==='YouTube search #1'))await db.query('INSERT INTO anime_media_cache (anime_id,payload,fetched_at) VALUES ($1,$2::jsonb,now()) ON CONFLICT (anime_id) DO UPDATE SET payload=excluded.payload,fetched_at=excluded.fetched_at',[id,JSON.stringify(media)]);
 return media;
});
app.get('/api/anime/:id',async(req,reply)=>{const id=Number((req.params as any).id);const item=engine.byId.get(id);return item??reply.code(404).send({error:'Anime not found'})});
// AniList PREQUEL guard is applied to the deck BEFORE it reaches the swipe UI.
// When AniList cannot be reached, the existing dataset guards still apply.
app.get('/api/recommendations',async(req,reply)=>{
 const parsed=recomQuery.safeParse(req.query);if(!parsed.success)return reply.code(400).send({error:'Invalid recommendation options'});
 const p=getProfile(req),h=await history(p.id);const wanted=parsed.data.size;
 const recentActions=h.filter(x=>x.source!=='onboarding').slice(-3).map(x=>x.action);const pivot=recentActions.length===3&&(recentActions.every(x=>x==='pass')||recentActions.every(x=>x==='watched'));
 const cooldown=await db.query('SELECT anime_id,eligible_after FROM reentry_queue WHERE profile_id=$1',[p.id]);
 const blocked=cooldown.rows.filter(x=>h.length<Number(x.eligible_after)).map(x=>Number(x.anime_id));
 const candidates=engine.recommend(p,h,{...parsed.data,pivot,lambda:pivot?0.57:parsed.data.lambda,exclude:blocked,size:Math.max(wanted*5,40),guard:false,visualWeight:p.mode==='eyecandy'?.78:p.mode==='content'?.16:.33,lapsed:p.mode==='lapsed'}).filter(x=>x.anime.score!==null&&Number.isFinite(x.anime.score)&& (p.lengthPreference!=='one'||(x.anime.episodes===1&&/movie/i.test(x.anime.type))));
 // Load the PREQUEL ancestry (not only the immediate parent). Dataset may say
 // Bleach TYBW has chain [41467], so CSV ancestry alone is insufficient.
 const graph=await fetchFranchiseRelations(db,candidates.map(x=>x.anime.id));
 const relations=new Map(graph.relations),verified=new Set(graph.verified);
 let warning=graph.warning;
 const visited=new Set(candidates.map(x=>x.anime.id));
 let frontier=[...relations.values()].flat().filter(r=>r.relationType==='PREQUEL'&&r.idMal&&engine.byId.has(r.idMal)).map(r=>r.idMal!);
 for(let depth=0;depth<5&&frontier.length;depth++){
  const pending=[...new Set(frontier)].filter(id=>!visited.has(id));if(!pending.length)break;
  for(const id of pending)visited.add(id);
  const fetched=await fetchFranchiseRelations(db,pending);
  for(const [id,items] of fetched.relations)relations.set(id,items);
  for(const id of fetched.verified)verified.add(id);
  if(fetched.warning)warning=warning??fetched.warning;
  frontier=[...fetched.relations.values()].flat().filter(r=>r.relationType==='PREQUEL'&&r.idMal&&engine.byId.has(r.idMal)).map(r=>r.idMal!);
 }
 const watched=new Set(h.filter(x=>x.action==='watched').map(x=>x.animeId));
 const external=await db.query('SELECT mal_id FROM external_watched WHERE profile_id=$1',[p.id]);for(const x of external.rows)watched.add(x.mal_id);
 const locked:Array<{animeId:number;title:string;prequels:Relation[]}> = [];
 const eligible=candidates.filter(x=>{
  const missing=new Map<number|string,Relation>();
  const put=(r:Relation)=>missing.set(r.idMal??`unknown:${r.title}`,r);
  // Always apply dataset roots even if AniList responds with an empty list.
  for(const r of datasetMissing(x.anime,catalog,watched))put(r);
  const stack=[x.anime.id],seenIds=new Set<number>();let ancestryUnknown=false;
  while(stack.length){
   const current=stack.pop()!;if(seenIds.has(current))continue;seenIds.add(current);
   if(!verified.has(current)){ancestryUnknown=true;continue;}
   for(const r of (relations.get(current)??[]).filter(r=>r.relationType==='PREQUEL')){
    if(!r.idMal){put(r);continue;}
    if(!watched.has(r.idMal))put(r);
    if(engine.byId.has(r.idMal))stack.push(r.idMal);
   }
  }
  // If AniList is down or mapping absent, explicit sequel signals are
  // blocked rather than incorrectly assumed to be standalone anime.
  if(ancestryUnknown&&sequelHint(x.anime)&&missing.size===0){
   put({idMal:null,title:'Chưa xác minh được chuỗi PREQUEL',relationType:'UNVERIFIED_SEQUEL'});
  }
  if(missing.size){locked.push({animeId:x.anime.id,title:x.anime.title,prequels:[...missing.values()]});return false;}
  return true;
 });
 const result=eligible.slice(0,wanted);
 await db.query('INSERT INTO recommendation_events (profile_id,anime_ids) VALUES ($1,$2)',[p.id,result.map(x=>x.anime.id)]);
 return {items:result,pivot:{active:pivot,reason:pivot?'3 lần liên tiếp Pass hoặc Watched — khám phá mở rộng và Hidden Gems':null},totalHistory:h.length,franchise:{status:warning?'partial':'verified',warning,verified:verified.size,candidates:candidates.length,locked}};
});
// Mark a prerequisite as watched even when MAL ID does not exist in our local catalog.
app.post('/api/franchise/external-watched',async(req,reply)=>{
 const body=z.object({malId:z.number().int().positive(),title:z.string().max(250).default('')}).safeParse(req.body);
 if(!body.success)return reply.code(400).send({error:'Invalid MAL ID'});
 const p=getProfile(req);
 await db.query('INSERT INTO external_watched(profile_id,mal_id,title) VALUES ($1,$2,$3) ON CONFLICT (profile_id,mal_id) DO NOTHING',[p.id,body.data.malId,body.data.title]);
 return {ok:true};
});
// Detailed gate inspection for a specific anime, including prequels absent from local CSV.
app.get('/api/franchise/:id',async(req,reply)=>{
 const id=Number((req.params as any).id),anime=engine.byId.get(id);
 if(!anime)return reply.code(404).send({error:'Anime not found'});
 const p=getProfile(req),hist=await history(p.id);
 const watched=new Set(hist.filter(x=>x.action==='watched').map(x=>x.animeId));
 const external=await db.query('SELECT mal_id FROM external_watched WHERE profile_id=$1',[p.id]);for(const x of external.rows)watched.add(x.mal_id);
 const graph=await fetchFranchiseRelations(db,[id]);
 const prequels=(graph.relations.get(id)??[]).filter(r=>r.relationType==='PREQUEL').map(r=>({...r,watched:!!r.idMal&&watched.has(r.idMal)}));
 const datasetPrerequisites=datasetMissing(anime,catalog,watched).map(r=>({idMal:r.idMal,title:r.title,watched:false}));
 const unknown=!graph.verified.has(id)&&sequelHint(anime);
 return {animeId:id,title:anime.title,source:graph.verified.has(id)?'anilist':'dataset',warning:graph.warning,locked:unknown||prequels.some(x=>!x.watched)||datasetPrerequisites.length>0,prequels,datasetPrerequisites,unverified:unknown};
});
app.post('/api/feedback',async(req,reply)=>{try{const b=feedbackBody.parse(req.body);if(!engine.byId.has(b.animeId))return reply.code(404).send({error:'Anime not found'});const p=getProfile(req);await db.query('DELETE FROM reentry_queue WHERE profile_id=$1 AND anime_id=$2',[p.id,b.animeId]);const r=await db.query('INSERT INTO interactions (profile_id,anime_id,action) VALUES ($1,$2,$3) RETURNING id',[p.id,b.animeId,b.action]);return {ok:true,id:Number(r.rows[0].id)}}catch(e){if(!err(reply,e))throw e}});
app.post('/api/feedback/undo',async(req)=>{const p=getProfile(req);const deleted=await db.query('DELETE FROM interactions WHERE id=(SELECT MAX(id) FROM interactions WHERE profile_id=$1) AND profile_id=$1 RETURNING anime_id,action',[p.id]);return {undone:deleted.rows[0]??null}});
// Tonight pick: deterministic per day and reroll, among LIKED but not already watchlisted.
app.get('/api/library/pick-tonight',async(req,reply)=>{
 const query=z.object({reroll:z.coerce.number().int().min(0).max(1000).default(0)}).safeParse(req.query);
 if(!query.success)return reply.code(400).send({error:'Invalid reroll'});
 const p=getProfile(req),h=await history(p.id);const latest=new Map<number,typeof h[number]>();for(const x of h)latest.set(x.animeId,x);
 const w=await db.query('SELECT anime_id FROM watchlist WHERE profile_id=$1',[p.id]);const excluded=new Set<number>(w.rows.map(x=>Number(x.anime_id)));
 const list=[...latest.values()].filter(x=>x.action==='like'&&!excluded.has(x.animeId)).map(x=>engine.byId.get(x.animeId)).filter((a):a is Anime=>!!a);
 const today=new Date().toISOString().slice(0,10);
 const hashStr=(str:string)=>{let n=2166136261;for(let i=0;i<str.length;i++)n=Math.imul(n^str.charCodeAt(i),16777619);return (n>>>0)/4294967295};
 const ranked=list.map(a=>{
  const signals=computeTasteSignals(a,p,h,engine.byId);
  const quality=Math.min(1,Math.max(0,((a.score??0)-7)/3));
  const suitability=signals.score;
  const discovery=hashStr(today+p.id+a.id);
  return {anime:a,signals,rank:0.50*quality+0.40*suitability+0.10*discovery};
 }).sort((a,b)=>b.rank-a.rank);
 const selected=ranked[query.data.reroll%ranked.length];
 if(!selected)return {anime:null,eligibleCount:0,date:today,reason:'Chưa có anime đã Like và chưa nằm trong Watchlist.',factors:null};
 const {anime,signals}=selected;
 const positive=signals.channels.filter(x=>x.value!==null&&x.value>0).sort((a,b)=>b.weight*(b.value??0)-a.weight*(a.value??0));
 const label:{[key:string]:string}={genre:'Genre',theme:'Themes / mô típ',visual:'Visual style'};
 const details=positive.slice(0,2).map(x=>`${label[x.name]} phù hợp (${Math.round((x.value??0)*100)}% tín hiệu, trọng số ${Math.round(x.weight*100)}%)`);
 const reason=[`Đã Like và chưa thêm vào Watchlist`,...details,`rating ★ ${(anime.score??0).toFixed(2)}/10`, `được chọn từ ${ranked.length} ứng viên`].join(' · ');
 return {anime,eligibleCount:ranked.length,date:today,reason,factors:{mode:p.mode,weights:signals.modeWeights,channels:signals.channels,quality:Number(Math.min(1,Math.max(0,((anime.score??0)-7)/3)).toFixed(3)),discovery:Number(hashStr(today+p.id+anime.id).toFixed(3))}};
});
app.get('/api/library',async(req)=>{const p=getProfile(req),h=await history(p.id);const latest=new Map<number,typeof h[number]>();for(const x of h)latest.set(x.animeId,x);const list=(action:Action)=>[...latest.values()].filter(x=>x.action===action).map(x=>engine.byId.get(x.animeId)).filter(Boolean);
 const w=await db.query('SELECT anime_id FROM watchlist WHERE profile_id=$1',[p.id]);return {history:h,liked:list('like'),passed:list('pass'),watched:list('watched'),watchlist:w.rows.map(x=>engine.byId.get(x.anime_id)).filter(Boolean)};
});
app.post('/api/library/:id/transition',async(req,reply)=>{
 const id=Number((req.params as any).id),p=getProfile(req);if(!engine.byId.has(id))return reply.code(404).send({error:'Anime not found'});
 const parse=z.object({to:z.enum(['requeue','like','pass','watched'])}).safeParse(req.body);if(!parse.success)return reply.code(400).send({error:'Invalid transition'});
 const client=await db.connect();try{
 await client.query('BEGIN');await client.query('DELETE FROM interactions WHERE profile_id=$1 AND anime_id=$2',[p.id,id]);
 if(parse.data.to==='requeue'){
  const count=await client.query('SELECT count(*)::int AS n FROM interactions WHERE profile_id=$1',[p.id]);
  // Re-enter only after at least 3 new swipes, not on next immediate deck.
  await client.query('INSERT INTO reentry_queue(profile_id,anime_id,eligible_after) VALUES($1,$2,$3) ON CONFLICT(profile_id,anime_id) DO UPDATE SET eligible_after=EXCLUDED.eligible_after,created_at=now()',[p.id,id,Number(count.rows[0].n)+3]);
 }else{await client.query('DELETE FROM reentry_queue WHERE profile_id=$1 AND anime_id=$2',[p.id,id]);await client.query('INSERT INTO interactions(profile_id,anime_id,action) VALUES($1,$2,$3)',[p.id,id,parse.data.to])}
 await client.query('COMMIT');return {ok:true,to:parse.data.to};
 }catch(e){await client.query('ROLLBACK');throw e}finally{client.release()}
});
app.post('/api/watchlist/:id',async(req,reply)=>{const id=Number((req.params as any).id);if(!engine.byId.has(id))return reply.code(404).send({error:'Not found'});await db.query('INSERT INTO watchlist (profile_id,anime_id) VALUES ($1,$2) ON CONFLICT DO NOTHING',[getProfile(req).id,id]);return {ok:true}});
app.delete('/api/watchlist/:id',async(req)=>{await db.query('DELETE FROM watchlist WHERE profile_id=$1 AND anime_id=$2',[getProfile(req).id,Number((req.params as any).id)]);return {ok:true}});
const port=Number(process.env.PORT||3001);
const host=process.env.API_HOST||'127.0.0.1';
try {
 await app.listen({port,host});
} catch (error:any) {
 if(error?.code==='EADDRINUSE') {
  app.log.error(`Port ${port} đã được dùng bởi một tiến trình khác. Đóng terminal AniMatch API cũ hoặc kiểm tra: Get-NetTCPConnection -LocalPort ${port} -State Listen`);
 } else {
  app.log.error(error,'AniMatch backend không thể khởi động');
 }
 await db.end();
 process.exitCode=1;
}

