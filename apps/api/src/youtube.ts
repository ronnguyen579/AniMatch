/** First embeddable YouTube search result for the anime title.
 * Uses official YouTube Data API v3; no scraping, no fabricated video IDs.
 * Google quota: search.list costs 100 units. Cache 14 days server-side.
 */
export type VideoResult={id:string;title:string;source:'youtube-search'|'jikan';url:string};
const cache=new Map<string,{value:VideoResult|null;at:number}>();
const videoId=(v:unknown):v is string=>typeof v==='string'&&/^[A-Za-z0-9_-]{11}$/.test(v);
export async function searchYouTubeFirst(title:string,year:number|null):Promise<VideoResult|null>{
 const key=process.env.YOUTUBE_API_KEY?.trim();if(!key)return null;
 const query=`${title}${year?` ${year}`:''} anime official trailer`;
 const hit=cache.get(query);if(hit&&Date.now()-hit.at<14*86400000)return hit.value;
 const params=new URLSearchParams({part:'snippet',q:query,type:'video',videoEmbeddable:'true',safeSearch:'strict',order:'relevance',maxResults:'1',key});
 const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),6500);
 try{
  const response=await fetch('https://www.googleapis.com/youtube/v3/search?'+params.toString(),{signal:controller.signal});
  if(!response.ok)throw new Error('YouTube search HTTP '+response.status);
  const body=await response.json() as any;const first=body.items?.[0];
  const id=first?.id?.videoId;
  const value=videoId(id)?{id,title:String(first.snippet?.title||'Anime trailer'),source:'youtube-search' as const,url:`https://www.youtube.com/watch?v=${id}`} : null;
  cache.set(query,{value,at:Date.now()});return value;
 }finally{clearTimeout(timer)}
}
