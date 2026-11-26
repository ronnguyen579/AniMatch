import type {Anime} from '@animatch/shared';
import type {Relation} from './anilist.js';
const key=(s:string)=>s.toLowerCase().replace(/\s+/g,' ').trim();
/** Dataset may isolate every Bleach season into [its own MAL ID]. Detect exact
 * title prefixes that are also catalog entries; never use substring contains. */
export function titlePrerequisites(a:Anime,catalog:Anime[]):number[]{
 const titles=new Map<string,Anime[]>();
 for(const x of catalog){const k=key(x.title);const arr=titles.get(k)??[];arr.push(x);titles.set(k,arr)}
 const k=key(a.title),roots=new Set<number>();
 for(const sep of [': ',' - ']){
  let index=k.indexOf(sep);
  while(index>0){const prefix=k.slice(0,index);
   for(const candidate of titles.get(prefix)??[]){
    if(candidate.id!==a.id && (candidate.year===null||a.year===null||candidate.year<=a.year))roots.add(candidate.id);
   }
   index=k.indexOf(sep,index+sep.length);
  }
 }
 return [...roots];
}
export function sequelHint(a:Anime):boolean{
 return /(?:\b(?:2nd|3rd|4th|5th|6th)\s+season\b|\bseason\s*[2-9]\b|\bpart\s*[2-9]\b|\b(?:s2|s3|s4)\b)/i.test(a.title) || (!!a.prereqId&&a.prereqId!==a.id) || (!!a.entryId&&a.entryId!==a.id);
}
export function datasetMissing(a:Anime,catalog:Anime[],watched:Set<number>){
 const ids=new Set<number>([a.entryId,a.prereqId,...titlePrerequisites(a,catalog)].filter((x):x is number=>!!x&&x!==a.id));
 return [...ids].filter(id=>!watched.has(id)).map(id=>({idMal:id,title:catalog.find(x=>x.id===id)?.title??`MAL #${id}`,relationType:'DATASET_PREREQUISITE'} as Relation));
}
