import assert from 'node:assert/strict';
import test from 'node:test';
import { missingPrequels, fetchFranchiseRelations } from '../apps/api/src/anilist.ts';
import { searchYouTubeFirst } from '../apps/api/src/youtube.ts';

test('AniList PREQUEL: only missing prerequisite locks sequel',()=>{
 const relations=[{idMal:11,title:'Season One',relationType:'PREQUEL'},{idMal:12,title:'OVA',relationType:'SIDE_STORY'}];
 assert.equal(missingPrequels(relations,new Set([])).length,1);
 assert.equal(missingPrequels(relations,new Set([11])).length,0);
});

test('AniList GraphQL lookup uses MAL IDs and caches verified PREQUEL edges',async()=>{
 const originalFetch=globalThis.fetch;
 const persisted:Array<string>=[];
 const mockDb={query:async(q:string)=>{
  if(q.includes('SELECT anime_id'))return {rows:[]};
  if(q.includes('INSERT INTO anilist_relation_cache'))persisted.push(q);
  return {rows:[]};
 }};
 globalThis.fetch=async(_url,init)=>{
  const q=JSON.parse(String(init?.body)).query as string;
  assert.match(q,/Media\(idMal: 1001, type: ANIME\)/);
  return {ok:true,json:async()=>({data:{m0:{idMal:1001,relations:{edges:[{relationType:'PREQUEL',node:{idMal:1000,title:{english:'Previous Season'}}}]}}}})} as Response;
 };
 try{
  const graph=await fetchFranchiseRelations(mockDb as never,[1001]);
  assert.ok(graph.verified.has(1001));
  assert.equal(graph.relations.get(1001)?.[0]?.idMal,1000);
  assert.equal(persisted.length,1);
 }finally{globalThis.fetch=originalFetch}
});

test('YouTube: first embeddable relevance search result is returned',async()=>{
 const oldKey=process.env.YOUTUBE_API_KEY,originalFetch=globalThis.fetch;
 process.env.YOUTUBE_API_KEY='test-key';
 globalThis.fetch=async(url)=>{
  const u=new URL(String(url));
  assert.equal(u.searchParams.get('type'),'video');
  assert.equal(u.searchParams.get('maxResults'),'1');
  assert.equal(u.searchParams.get('videoEmbeddable'),'true');
  return {ok:true,json:async()=>({items:[{id:{videoId:'abcdefghijk'},snippet:{title:'A valid trailer'}}]})} as Response;
 };
 try{assert.equal((await searchYouTubeFirst('Testing Franchise',2026))?.id,'abcdefghijk')}
 finally{globalThis.fetch=originalFetch;if(oldKey===undefined)delete process.env.YOUTUBE_API_KEY;else process.env.YOUTUBE_API_KEY=oldKey}
});
