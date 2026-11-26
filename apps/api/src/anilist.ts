/** Franchise prerequisite checks based on AniList's PREQUEL edges, matched by MAL ID.
 * Caches authoritative responses in PostgreSQL. On an outage, do NOT pretend the
 * relations were verified: callers should retain the dataset-based guard.
 */
import type pg from 'pg';
export type Relation = { idMal:number|null; title:string; relationType:string };
export type GraphResult = { relations:Map<number,Relation[]>; verified:Set<number>; warning:string|null };
const TTL_DAYS=7;
const requestTimeout=6500;
const validRelation=(v:unknown):v is Relation=>!!v&&typeof v==='object'&&typeof (v as Relation).relationType==='string';
export function missingPrequels(relations:Relation[],watched:Set<number>):Relation[]{
 return relations.filter(r=>r.relationType==='PREQUEL' && (!r.idMal || !watched.has(r.idMal)));
}
async function anilistBatch(ids:number[]):Promise<Map<number,Relation[]>>{
 if(!ids.length)return new Map();
 const fields=ids.map((id,i)=>`m${i}: Media(idMal: ${id}, type: ANIME) { idMal relations { edges { relationType node { idMal title { romaji english } } } } }`).join('\n');
 const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),requestTimeout);
 try{
  const r=await fetch('https://graphql.anilist.co',{method:'POST',signal:controller.signal,headers:{'Content-Type':'application/json',Accept:'application/json'},body:JSON.stringify({query:`query AniMatchFranchise { ${fields} }`})});
  if(!r.ok)throw new Error(`AniList HTTP ${r.status}`);
  const data=await r.json() as any;
  if(data.errors?.length)throw new Error('AniList GraphQL errors');
  const out=new Map<number,Relation[]>();
  ids.forEach((id,i)=>{
   const item=data.data?.[`m${i}`];
   if(!item || item.idMal!==id)return; // No verified mapping for this MAL ID.
   const edges=item.relations?.edges;
   const rels:Array<Relation>=Array.isArray(edges)?edges.filter((x:any)=>x?.node).map((x:any)=>({
    idMal:Number.isInteger(x.node.idMal)&&x.node.idMal>0?x.node.idMal:null,
    title:String(x.node.title?.english||x.node.title?.romaji||'Unknown prequel'),
    relationType:String(x.relationType||'UNKNOWN')
   })):[];
   out.set(id,rels);
  });
  return out;
 }finally{clearTimeout(timeout)}
}
/** Keep a best-effort cache: failures are reported, never represented as no relations. */
export async function fetchFranchiseRelations(db:pg.Pool,ids:number[]):Promise<GraphResult>{
 const unique=[...new Set(ids.filter(id=>Number.isInteger(id)&&id>0))];
 const relations=new Map<number,Relation[]>(),verified=new Set<number>();
 if(!unique.length)return {relations,verified,warning:null};
 const cached=await db.query('SELECT anime_id,relations,fetched_at FROM anilist_relation_cache WHERE anime_id=ANY($1::int[])',[unique]);
 for(const row of cached.rows){
  if(Date.now()-new Date(row.fetched_at).getTime()<TTL_DAYS*86400000 && Array.isArray(row.relations)){
   relations.set(row.anime_id,row.relations.filter(validRelation));verified.add(row.anime_id);
  }
 }
 const pending=unique.filter(id=>!verified.has(id));let warning:string|null=null;
 for(let i=0;i<pending.length;i+=12){
  const part=pending.slice(i,i+12);
  try{
   const fresh=await anilistBatch(part);
   for(const [id,rels] of fresh){
    await db.query(`INSERT INTO anilist_relation_cache (anime_id,relations,fetched_at) VALUES ($1,$2::jsonb,now()) ON CONFLICT (anime_id) DO UPDATE SET relations=excluded.relations,fetched_at=excluded.fetched_at`,[id,JSON.stringify(rels)]);
    relations.set(id,rels);verified.add(id);
   }
   if(fresh.size<part.length)warning='Một số anime không có mapping AniList theo MAL ID.';
  }catch(e){warning=`AniList tạm thời không khả dụng (${e instanceof Error?e.message:'network error'}). Sử dụng franchise guard từ dataset.`;break}
 }
 return {relations,verified,warning};
}
