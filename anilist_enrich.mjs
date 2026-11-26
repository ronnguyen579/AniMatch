#!/usr/bin/env node
/**
 * AniMatch incremental metadata enrichment via AniList GraphQL.
 * Requires Node.js 20+; no extra npm dependencies.
 * Does not modify the original CSV or the AniMatch database.
 *
 * node anilist_enrich.mjs fetch /path/to/data/anime.csv
 * node anilist_enrich.mjs merge /path/to/data/anime.csv
 */
import fs from 'node:fs';
import path from 'node:path';

const command=process.argv[2]||'fetch';
const input=process.argv[3]||'data/anime.csv';
const dest=process.argv[4]||'anilist_enriched.jsonl';
const output=process.argv[5]||'anime_multisource_export.csv';
if(!fs.existsSync(input)){console.error('CSV not found:', input);process.exit(1)}
const sleep=(ms)=>new Promise(r=>setTimeout(r,ms));
function parseCsv(text){
  if(text.charCodeAt(0)===0xfeff)text=text.slice(1);
  const data=[];let row=[],value='',quoted=false;
  for(let i=0;i<text.length;i++){const c=text[i];
    if(c==='"'){if(quoted&&text[i+1]==='"'){value+='"';i++}else quoted=!quoted}
    else if(c===','&&!quoted){row.push(value);value=''}
    else if((c==='\r'||c==='\n')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(value);value='';if(row.some(x=>x))data.push(row);row=[]}
    else value+=c
  }
  if(value||row.length){row.push(value);data.push(row)}
  const header=data.shift()||[];
  return {header,records:data.map(row=>Object.fromEntries(header.map((x,i)=>[x,row[i]||''])))};
}
function toCsv(cols,records){
  const esc=v=>{const s=String(v??'');return /[,"\r\n]/.test(s)?`"${s.replaceAll('"','""')}"`:s};
  return [cols.join(','),...records.map(r=>cols.map(c=>esc(r[c])).join(','))].join('\r\n')+'\r\n';
}
const {header,records}=parseCsv(fs.readFileSync(input,'utf8'));
const prior=new Map();
if(fs.existsSync(dest)){
  for(const line of fs.readFileSync(dest,'utf8').split(/\r?\n/)){
    if(!line.trim())continue;
    try{const j=JSON.parse(line);if(Number.isInteger(j.mal_id)&&j.fetched_at)prior.set(j.mal_id,j)}catch{}
  }
}
const query=`query ($mal: Int) {
  Media(idMal: $mal, type: ANIME) {
    id idMal isAdult format status episodes duration seasonYear
    startDate { year month day }
    endDate { year month day }
    genres
    tags { id name rank isMediaSpoiler isGeneralSpoiler }
    coverImage { extraLarge large }
    bannerImage
    siteUrl
    studios(isMain: true) { nodes { name } }
    relations { edges {
      relationType
      node { id idMal format title { romaji english } }
    }}
  }
}`;
async function request(id){
 let last;
 for(let attempt=0;attempt<5;attempt++){
   const controller=new AbortController();
   const t=setTimeout(()=>controller.abort(),15000);
   try{
     const r=await fetch('https://graphql.anilist.co',{
       method:'POST',signal:controller.signal,headers:{'Content-Type':'application/json','Accept':'application/json'},
       body:JSON.stringify({query,variables:{mal:id}})
     });
     if(r.status===429 || r.status>=500){
       const ra=Number(r.headers.get('retry-after')||0);
       await sleep(Math.max(ra*1000, 2000*(2**attempt)));continue;
     }
     if(!r.ok)throw Error('HTTP '+r.status);
     const json=await r.json();
     if(json.errors?.length)throw Error(JSON.stringify(json.errors).slice(0,400));
     return json.data?.Media??null;
   }catch(e){
     last=e;
     if(attempt<4)await sleep(1200*(2**attempt));
   }finally{clearTimeout(t)}
 }
 throw last||new Error('Maximum retries');
}
if(command==='fetch'){
 const ids=records.map(r=>Number(r.mal_id)).filter(Number.isInteger);
 let done=0;
 for(const id of ids){
   if(prior.has(id))continue;
   try{
     const data=await request(id);
     const item={mal_id:id,fetched_at:new Date().toISOString(),found:!!data,source:'AniList GraphQL',data};
     fs.appendFileSync(dest,JSON.stringify(item)+'\n');
     prior.set(id,item);
   }catch(err){
     console.error(`ID ${id}: ${err?.message||err}; stopping so it can resume safely.`);
     process.exitCode=1;break;
   }
   done++;
   if(done%25===0)console.log(`Fetched ${done}; total cached ${prior.size}/${ids.length}`);
   // approximately 30 calls/min (conservative, adjust to current platform limits)
   await sleep(2050);
 }
 console.log(`Finished batch: ${done} newly fetched. Cache: ${prior.size}/${ids.length}`);
}else if(command==='merge'){
 const newFields=['anilist_id','anilist_status','anilist_format','anilist_episode_count','anilist_genres_json','anilist_tags_json','anilist_studios_json','anilist_relations_json','anilist_is_adult','anilist_poster_url','anilist_banner_url','anilist_source_url','anilist_fetched_at'];
 const converted=records.map(r=>{
   const source=prior.get(Number(r.mal_id));
   const d=source?.data;
   return {...r,anilist_id:d?.id??'',anilist_status:d?.status??'',
     anilist_format:d?.format??'',anilist_episode_count:d?.episodes??'',
     anilist_genres_json:d?JSON.stringify(d.genres||[]):'',
     anilist_tags_json:d?JSON.stringify(d.tags||[]):'',
     anilist_studios_json:d?JSON.stringify(d.studios?.nodes||[]):'',
     anilist_relations_json:d?JSON.stringify(d.relations?.edges||[]):'',
     anilist_is_adult:d?String(!!d.isAdult):'',
     anilist_poster_url:d?.coverImage?.extraLarge||d?.coverImage?.large||'',
     anilist_banner_url:d?.bannerImage||'',
     anilist_source_url:d?.siteUrl||'',
     anilist_fetched_at:source?.fetched_at||''
   }
 });
 fs.writeFileSync(output,toCsv([...header,...newFields],converted),'utf8');
 console.log(`Wrote ${output}: ${converted.length} rows, ${prior.size} cached lookups`);
}else{
 console.error('Use: fetch or merge');process.exitCode=1;
}
