import { searchYouTubeFirst } from './youtube.js';
/** Lazy enrichment from exact MyAnimeList ID through Jikan. Never extrapolate HD. */
type Media={imageUrl:string;trailerYoutubeId:string|null;trailerTitle:string|null;trailerSource:string|null;source:string;imageQuality:string};
const cache=new Map<number,{data:Media;expires:number}>();
let lastRequest=0;let queue=Promise.resolve();
const mediaId=(v:unknown):string|null=>typeof v==='string'&&/^[a-zA-Z0-9_-]{11}$/.test(v)?v:null;
async function fetchJikan(id:number,original:string):Promise<Media>{
 const baseline:Media={imageUrl:original,trailerYoutubeId:null,trailerTitle:null,trailerSource:null,source:'dataset',imageQuality:'original dataset; resolution not verified'};
 try{
  const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),7000);
  let response:Response;
  try{response=await fetch(`https://api.jikan.moe/v4/anime/${id}`,{signal:controller.signal,headers:{Accept:'application/json'}})}finally{clearTimeout(timeout)}
  if(!response.ok)return baseline;
  const payload=await response.json() as any;const data=payload?.data;
  const image=data?.images?.webp?.large_image_url||data?.images?.jpg?.large_image_url||original;
  const validImage=typeof image==='string'&&/^https:\/\//.test(image)?image:original;
  return {imageUrl:validImage,trailerYoutubeId:mediaId(data?.trailer?.youtube_id),trailerTitle:null,trailerSource:mediaId(data?.trailer?.youtube_id)?'Jikan':null,source:'Jikan / MyAnimeList',imageQuality:'large source variant; Full HD not guaranteed'};
 }catch{return baseline}
}
export async function getAnimeMedia(id:number,original:string,title:string,year:number|null):Promise<Media>{
 const cached=cache.get(id);if(cached&&cached.expires>Date.now())return cached.data;
 // Serial queue limits outgoing third-party requests; avoid bulk calls on browse.
 const task=queue.then(async()=>{const existing=cache.get(id);if(existing&&existing.expires>Date.now())return existing.data;
   const elapsed=Date.now()-lastRequest;if(elapsed<1100)await new Promise(r=>setTimeout(r,1100-elapsed));lastRequest=Date.now();
   const data=await fetchJikan(id,original);
   // When configured, always use the FIRST result of the YouTube search API.
   // If search is unavailable, fall back to the MAL/Jikan trailer ID.
   if(process.env.YOUTUBE_API_KEY?.trim()){
     try{const result=await searchYouTubeFirst(title,year);if(result){data.trailerYoutubeId=result.id;data.trailerTitle=result.title;data.trailerSource='YouTube search #1';}}
     catch(e){console.warn('[AniMatch media] YouTube search unavailable; keeping Jikan fallback:',e instanceof Error?e.message:e)}
   }
   cache.set(id,{data,expires:Date.now()+(data.source==='dataset'&&!data.trailerYoutubeId?10*60*1000:24*60*60*1000)});return data;
 });queue=task.then(()=>undefined,()=>undefined);return task;
}
