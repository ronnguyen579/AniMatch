import { getAnimeVisualStyles } from '@animatch/shared';
import type { Anime, Action, Profile, Recommendation } from '@animatch/shared';
export type Sparse = Map<string, number>;
export interface Feedback { animeId:number;action:Action;createdAt?:string }
export interface RecommendOptions { size?:number; lambda?:number; visualWeight?:number; exclude?:number[]; minScore?:number; diversity?:boolean; guard?:boolean; lapsed?:boolean; pivot?:boolean }
const token = (s:string) => (s.toLowerCase().match(/[a-z0-9]{2,}/g) ?? []);
const genreTokens = (a:Anime) => [...a.genres,...a.themes,...a.demographics].flatMap(x=>token(x).map(t=>'g:'+t));
const studioTokens = (a:Anime) => a.studios.flatMap(x=>token(x).map(t=>'v:'+t));
const visualTokens = (a:Anime) => getAnimeVisualStyles(a).flatMap(x=>token(x).map(t=>'a:'+t));
const STOP = new Set('the and for are with this that into from their they have has was were anime story about young who his her she him you its all but not can out one two where when them those these after before while there many more than what will each own'.split(' '));
const synTokens = (a:Anime) => {const w=token(a.synopsis).filter(x=>!STOP.has(x)).slice(0,180);return [...w.map(t=>'s:'+t),...w.slice(1).map((t,i)=>'b:'+w[i]+'_'+t)]};
const norm=(v:Sparse):Sparse=>{let n=0;for(const x of v.values())n+=x*x;n=Math.sqrt(n);if(n>0)for(const [k,x] of v)v.set(k,x/n);return v};
export const dot=(a:Sparse,b:Sparse)=>{let x=0;const [small,big]=a.size<=b.size?[a,b]:[b,a];for(const [k,v] of small)x+=v*(big.get(k)??0);return x};
const add=(to:Sparse,from:Sparse,m:number)=>{for(const [k,v] of from)to.set(k,(to.get(k)??0)+m*v)};
const root=(s:string)=>s.toLowerCase().replace(/\b(season|part|movie|ova)\s*\d+\b/g,'').replace(/\s*[:\-]\s*(ii|iii|iv|2nd|3rd|final).*$/,'').trim();

/** Explicit, auditable model weights. A mode changes genre / theme / visual influence,
 * while rating and hard franchise/release/length rules stay independent. */
export const MODE_WEIGHTS = {
 eyecandy: {genre:0.15, theme:0.15, visual:0.70},
 content:  {genre:0.48, theme:0.42, visual:0.10},
 lapsed:   {genre:0.38, theme:0.37, visual:0.25}
} as const;
function buildTaste(profile:Profile, history:Feedback[], byId:Map<number,Anime>){
 const genre=new Map<string,number>(),theme=new Map<string,number>(),visual=new Map<string,number>();
 const plus=(map:Map<string,number>,tag:string,amount:number)=>map.set(tag,(map.get(tag)??0)+amount);
 for(const g of profile.genres)plus(genre,g,1.8);
 for(const g of (profile.pastGenres??[]))plus(genre,g,profile.mode==='lapsed'?1.05:0.65);
 for(const g of (profile.exploreGenres??[]))plus(genre,g,1.4);
 for(const t of profile.themes)plus(theme,t,1.9);
 for(const v of profile.artstyles)plus(visual,v,1.4);
 // Only the latest decision per anime matters for current taste. A Pass is a
 // negative signal, not a permanent deletion from the radar/feature model.
 const latest=new Map<number,Feedback>();for(const f of history)latest.set(f.animeId,f);
 for(const f of latest.values()){
  const anime=byId.get(f.animeId);if(!anime)continue;
  const old=profile.mode==='lapsed'&&f.createdAt&&Date.now()-new Date(f.createdAt).getTime()>14*86400000;
  const amount=(f.action==='like'?1:f.action==='pass'?-0.75:0.20)*(old?0.4:1);
  for(const g of new Set(anime.genres))plus(genre,g,amount);
  for(const t of new Set(anime.themes))plus(theme,t,amount);
  for(const v of new Set(getAnimeVisualStyles(anime)))plus(visual,v,amount);
 }
 return {genre,theme,visual};
}
function axisAffinity(weights:Map<string,number>,tags:string[]):number|null{
 if(!weights.size||!tags.length)return null;
 const relevant=[...new Set(tags)].map(tag=>weights.get(tag)??0);
 // Missing tags are neutral; negative contributions lower the preference.
 const denom=Math.sqrt(Math.max(1,weights.size)*Math.max(1,relevant.length));
 return Math.max(-1,Math.min(1,relevant.reduce((a,b)=>a+b,0)/denom));
}
export function computeTasteSignals(a:Anime,profile:Profile,history:Feedback[],byId:Map<number,Anime>){
 const model=buildTaste(profile,history,byId);
 const w=MODE_WEIGHTS[profile.mode]??MODE_WEIGHTS.content;
 const channels=[
  {name:'genre',weight:w.genre,value:axisAffinity(model.genre,a.genres)},
  {name:'theme',weight:w.theme,value:axisAffinity(model.theme,a.themes)},
  {name:'visual',weight:w.visual,value:axisAffinity(model.visual,getAnimeVisualStyles(a))}
 ];
 // Channel without usable evidence does not pretend to be a positive match.
 const effective=channels.filter(x=>x.value!==null);
 const total=effective.reduce((s,x)=>s+x.weight,0);
 const score=effective.reduce((s,x)=>s+x.weight*(x.value??0),0);
 return {score,channels,modeWeights:w};
}

export class AniMatchEngine {
 readonly anime:Anime[]; readonly byId:Map<number,Anime>; private vectors=new Map<number,Sparse>(); private content=new Map<number,Sparse>();private visual=new Map<number,Sparse>();
 constructor(anime:Anime[]){this.anime=anime;this.byId=new Map(anime.map(a=>[a.id,a])); const docs=anime.map(a=>({a,g:genreTokens(a),s:synTokens(a),v:studioTokens(a),art:visualTokens(a)}));
 const df=new Map<string,number>();for(const d of docs)for(const t of new Set([...d.g,...d.s,...d.v,...d.art]))df.set(t,(df.get(t)??0)+1);
 const N=anime.length; const idf=(t:string)=>Math.log((N+1)/((df.get(t)??0)+1))+1;
 for(const d of docs){const make=(items:string[],weight:number)=>{const count=new Map<string,number>();for(const t of items)count.set(t,(count.get(t)??0)+1);const m:Sparse=new Map;for(const [t,c] of count)m.set(t,(1+Math.log(c))*idf(t));norm(m);for(const [t,v] of m)m.set(t,v*weight);return m};
 const c:Sparse=new Map; add(c,make(d.g,2.5),1);add(c,make(d.s,0.8),1);add(c,make(d.v,0.35),1);this.content.set(d.a.id,c);const art=make(d.art,1);this.visual.set(d.a.id,art);const full=new Map(c);add(full,art,0.65);this.vectors.set(d.a.id,full);
 } }
 vector(id:number){return this.vectors.get(id)??new Map<string,number>()}
 userVector(profile:Pick<Profile,'genres'|'themes'|'demographics'|'artstyles'|'pastGenres'|'exploreGenres'|'lengthPreference'>,history:Feedback[], visualWeight=0.3,lapsed=false):Sparse{
 const v:Sparse=new Map;const choices=this.anime.filter(a=>a.genres.some(g=>profile.genres.includes(g)||(profile.pastGenres??[]).includes(g)||(profile.exploreGenres??[]).includes(g))||a.themes.some(t=>profile.themes.includes(t))||a.demographics.some(t=>profile.demographics.includes(t))||getAnimeVisualStyles(a).some(t=>profile.artstyles.includes(t))).slice(0,160);
 if(choices.length)for(const a of choices)add(v,this.vector(a.id),1/choices.length);
 // Onboarding terms are also added directly so rare preferences remain represented.
 const terms=[...profile.genres,...profile.themes,...(profile.pastGenres??[]),...(profile.exploreGenres??[]),...profile.demographics].flatMap(s=>token(s).map(t=>'g:'+t));for(const t of terms)v.set(t,(v.get(t)??0)+0.4);
 for(const x of profile.artstyles)for(const t of token(x))v.set('a:'+t,(v.get('a:'+t)??0)+0.4*visualWeight);
 norm(v);for(const h of history){const a=this.byId.get(h.animeId);if(!a)continue;const stale=lapsed&&h.createdAt&&Date.now()-new Date(h.createdAt).getTime()>14*86400000;const weight=(h.action==='like'?0.3:h.action==='watched'?0.21:-0.15)*(stale?.38:1);for(const [t,x] of v)v.set(t,x*0.95);add(v,this.vector(a.id),weight);norm(v)}return v;
 }
 recommend(profile:Pick<Profile,'genres'|'themes'|'demographics'|'artstyles'|'pastGenres'|'exploreGenres'|'lengthPreference'>,history:Feedback[],opt:RecommendOptions={}):Recommendation[]{
 const {size=8,lambda=0.75,visualWeight=0.3,exclude=[],minScore=8,diversity=true,guard=true,lapsed=false,pivot=false}=opt;
 const latest=new Map<number,Feedback>();for(const h of history)latest.set(h.animeId,h);
 const seen=new Set([...latest.keys(),...exclude]);const watched=new Set(history.filter(x=>x.action==='watched').map(x=>x.animeId));const v=this.userVector(profile,history,visualWeight,lapsed);
 const today=new Date().toISOString().slice(0,10);
 const available=(a:Anime)=>!seen.has(a.id)&&!a.genres.some(g=>['Hentai','Erotica','Ecchi'].includes(g)) && !/not yet aired|not aired|upcoming/i.test(a.status??'') && !!a.airedFrom && a.airedFrom.slice(0,10)<=today && !/unknown/i.test(a.type??'') && (
  (profile.lengthPreference??'all')==='all' ||
  ((profile.lengthPreference??'all')==='one' && a.episodes===1 && /movie/i.test(a.type)) ||
  ((profile.lengthPreference??'all')==='short' && a.episodes!==null && a.episodes>=2 && a.episodes<=12) ||
  ((profile.lengthPreference??'all')==='medium' && a.episodes!==null && a.episodes>=13 && a.episodes<=26) ||
  ((profile.lengthPreference??'all')==='long' && a.episodes!==null && a.episodes>=27 && a.episodes<=52) ||
  ((profile.lengthPreference??'all')==='epic' && a.episodes!==null && a.episodes>=53));
 // Reject N/A ratings under every mode. 8.0+ is the preferred quality tier.
 // Only expand to 7.0+ if the strict length, release and history filters leave a small pool.
 const base=this.anime.filter(a=>available(a)&&a.score!==null&&Number.isFinite(a.score)&&a.score>=7);
 const preferred=base.filter(a=>a.score!==null&&a.score>=Math.max(8,minScore));
 const candidates=preferred.length>=Math.max(size*3,18)?preferred:base;
 const recent=new Set(history.slice(-4).map(h=>root(this.byId.get(h.animeId)?.title??'')));
 const taste=buildTaste(profile as Profile,history,this.byId);const modeWeights=MODE_WEIGHTS[(profile as Profile).mode]??MODE_WEIGHTS.content;
 const scored=candidates.map(a=>{const contentScore=dot(v,this.content.get(a.id)!);const visualScore=dot(v,this.visual.get(a.id)!);const similarity=dot(v,this.vector(a.id));const quality=Math.max(0,Math.min(1,((a.score??0)-6)/3.5)); const genreMatch=axisAffinity(taste.genre,a.genres),themeMatch=axisAffinity(taste.theme,a.themes),visualMatch=axisAffinity(taste.visual,getAnimeVisualStyles(a));
 const channels=[{weight:modeWeights.genre,value:genreMatch},{weight:modeWeights.theme,value:themeMatch},{weight:modeWeights.visual,value:visualMatch}].filter(x=>x.value!==null);
 const weightTotal=channels.reduce((sum,x)=>sum+x.weight,0);
 const blend=channels.reduce((sum,x)=>sum+x.weight*(x.value??0),0);
 const hiddenGem=pivot&&a.score!==null&&a.score>=8&&a.popularity!==null&&a.popularity>=2500 ? .20 : 0;
 const exploration=pivot ? .14*(1-Math.max(0,Math.min(1,contentScore))) : 0;
 const relevance=(pivot?.55:.72)*blend+.05*similarity+.23*quality+hiddenGem+exploration-((a.score??0)<8?.35:0)-(recent.has(root(a.title))?.13:0);return {a,relevance,contentScore,visualScore}}).sort((x,y)=>y.relevance-x.relevance||x.a.id-y.a.id);
 const pool:typeof scored=[];const visited=new Set<number>();for(const s of scored){const a=s.a;if(guard){if(a.prereqId&&a.prereqId!==a.id&&!watched.has(a.prereqId))continue;if(a.entryId&&a.entryId!==a.id&&!watched.has(a.entryId))continue;}if(!visited.has(a.id)){pool.push(s);visited.add(a.id)}if(pool.length>=Math.max(75,size*12))break}
 const selected:typeof scored=[];const selectedRoots=new Set<string>();const studios=new Map<string,number>();while(selected.length<size&&pool.length){let best=-1,bestM=-Infinity;for(let i=0;i<pool.length;i++){const x=pool[i];const franchise=root(x.a.title);if(selectedRoots.has(franchise))continue;const studio=x.a.studios[0]||'unknown';if((studios.get(studio)||0)>=2)continue;const penalty=diversity?Math.max(0,...selected.map(y=>dot(this.vector(x.a.id),this.vector(y.a.id)))):0;const mmr=lambda*x.relevance-(1-lambda)*penalty;if(mmr>bestM){best=i;bestM=mmr}}
 if(best<0)break;const [picked]=pool.splice(best,1);selected.push(picked);selectedRoots.add(root(picked.a.title));const studio=picked.a.studios[0]||'unknown';studios.set(studio,(studios.get(studio)||0)+1)}
 return selected.map(x=>({anime:x.a,relevance:Number(x.relevance.toFixed(4)),contentScore:Number(x.contentScore.toFixed(4)),visualScore:Number(x.visualScore.toFixed(4)),explanation:this.explain(x.a,profile as Profile,history)}));}
 explain(a:Anime,p:Profile,history:Feedback[]=[]):string{
 const t=buildTaste(p,history,this.byId);
 const ranked=[
  ...a.genres.filter(x=>(t.genre.get(x)??0)>0).map(x=>({name:x,kind:'Genre',v:t.genre.get(x)??0})),
  ...a.themes.filter(x=>(t.theme.get(x)??0)>0).map(x=>({name:x,kind:'Theme',v:t.theme.get(x)??0})),
  ...getAnimeVisualStyles(a).filter(x=>(t.visual.get(x)??0)>0).map(x=>({name:x,kind:'Visual',v:t.visual.get(x)??0}))
 ].sort((x,y)=>y.v-x.v).slice(0,3);
 const match=ranked.map(x=>`${x.kind} ${x.name}`).join(' · ');
 return match?`Phù hợp với gu ${p.mode}: ${match}. Rating ${a.score?.toFixed(2)??'N/A'}/10.`:`Chưa có đủ tín hiệu tag phù hợp; lựa chọn dựa nhiều hơn vào chất lượng (★ ${a.score?.toFixed(2)??'N/A'}) và khám phá.`;
 }
}
