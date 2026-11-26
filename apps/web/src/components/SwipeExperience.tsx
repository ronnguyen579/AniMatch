import React,{useEffect,useMemo,useRef,useState} from 'react';
import {BookmarkPlus,ExternalLink,RotateCcw,Play,Heart,ChevronLeft,ChevronRight} from 'lucide-react';
import type {Anime,Action,Recommendation} from '@animatch/shared';

export type VibeAxis={name:string;count:number;passes:number;score:number};
export type VibeData={genres:VibeAxis[];themes:VibeAxis[];visual:VibeAxis[];visualCoverage?:number;lastLikeGenres:string[];lastAction?:string|null;likes:number;passes:number;total:number};
const cleanTitle=(s:string)=>s.replace(/_/g,' ').replace(/\s+/g,' ').trim();
export function VibeRadar({vibe}:{vibe:VibeData}){
 const [tab,setTab]=useState<'genres'|'themes'|'visual'>('genres');const axes=(tab==='genres'?vibe.genres:tab==='themes'?vibe.themes:vibe.visual).filter(a=>a.score>0.05).slice(0,10);
 const n=axes.length,c=165,r=106;const max=Math.max(1,...axes.map(a=>Math.abs(a.score)));
 const xy=(i:number,rad:number)=>{const a=-Math.PI/2+2*Math.PI*i/Math.max(3,n);return [c+Math.cos(a)*rad,c+Math.sin(a)*rad]};
 const points=axes.map((a,i)=>xy(i,12+Math.max(0,a.score)/max*94));
 const ring=(scale:number)=>Array.from({length:Math.max(3,n)},(_,i)=>xy(i,r*scale).join(',')).join(' ');
 return <section className="vibe-card"><div className="vibe-header"><div><span className="eyebrow">LIVE TASTE DNA</span><h3>Vibe Radar <span className="vibe-live">● LIVE</span></h3></div><span className="vibe-count">{n} {tab==='genres'?'GENRES':tab==='themes'?'THEMES':'VISUAL STYLES'}</span></div>
 <div className="radar-tabs"><button className={tab==='genres'?'active':''} onClick={()=>setTab('genres')}>Genre Radar</button><button className={tab==='themes'?'active':''} onClick={()=>setTab('themes')}>Theme Radar</button><button className={tab==='visual'?'active':''} onClick={()=>setTab('visual')}>Visual Radar</button></div>
 {n===0?<div className="vibe-empty"><div className="vibe-ghost">✧</div><p>{tab==='visual'?`Chưa đủ metadata artstyle đáng tin cậy để vẽ Radar (${vibe.visualCoverage??0} anime có style tag). Không tự suy diễn style từ poster.`:tab==='themes'?'Theme Radar được học từ Like và Pass theo mô típ.':'Mỗi lượt Like hoặc Pass sẽ tác động đến những thể loại liên quan.'}</p></div>:<><svg viewBox="0 0 330 330" className="vibe-svg" role="img" aria-label={`${tab} radar: ${axes.map(a=>a.name+': '+a.score.toFixed(1)).join(', ')}`}>
 {[.25,.5,.75,1].map(t=><polygon key={t} points={ring(t)} fill="none" stroke="#514b60" strokeWidth="1" opacity={.65}/>)}
 {axes.map((a,i)=>{const [x,y]=xy(i,r);return <line key={a.name} x1={c} y1={c} x2={x} y2={y} stroke="#5d5269"/>})}
 {n>=3?<polygon className="vibe-shape" points={points.map(x=>x.join(',')).join(' ')} stroke="#bca5ff" strokeWidth="2" fill="rgba(161,128,255,.26)"/>:n===2?<line stroke="#bca5ff" strokeWidth="3" x1={points[0][0]} y1={points[0][1]} x2={points[1][0]} y2={points[1][1]}/>:null}
 {axes.map((a,i)=>{const [x,y]=points[i],[lx,ly]=xy(i,r+24);return <g key={a.name}><circle className="vibe-dot" cx={x} cy={y} r="5" fill={a.score<0?'#ff8d9b':'#b8ffdb'}/><text x={lx} y={ly} fill={a.score<0?'#ffa3af':'#e0d9f9'} textAnchor="middle" dominantBaseline="middle" fontSize="9" fontWeight="600">{cleanTitle(a.name).slice(0,18)}</text></g>})}
 </svg><div className="vibe-foot"><span>♥ {vibe.likes} LIKES · ✕ {vibe.passes} PASSES</span><span>{vibe.lastAction==='pass'?'PASS UPDATED TASTE':vibe.lastAction==='like'?'LIKE UPDATED TASTE':'TASTE EVOLVING'}</span></div><p className="radar-help">Xanh: sở thích tăng · Đỏ: sở thích giảm. Pass giảm điểm trục liên quan, không xoá tag khỏi lịch sử.</p></>}
 </section>;
}
type Media={imageUrl:string;trailerYoutubeId:string|null;trailerTitle?:string|null;trailerSource?:string|null;source:string; imageQuality:string};
export function Trailer({animeId}:{animeId:number}){
 const [media,setMedia]=useState<Media|null>(null);const [loading,setLoading]=useState(true);
 useEffect(()=>{let live=true;setMedia(null);setLoading(true);fetch(`/api/anime/${animeId}/media`).then(r=>r.ok?r.json():null).then(d=>{if(live)setMedia(d)}).catch(()=>{}).finally(()=>{if(live)setLoading(false)});return()=>{live=false}},[animeId]);
 const id=media?.trailerYoutubeId;
 return <div className="trailer-section"><div className="trailer-title"><Play size={17}/><b>YOUTUBE TRAILER</b></div>{loading?<p className="muted">Đang tìm trailer…</p>:id?<><div className="trailer-frame"><iframe key={id} loading="lazy" src={`https://www.youtube-nocookie.com/embed/${encodeURIComponent(id)}?rel=0`} title={media?.trailerTitle||'Anime trailer'} allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" referrerPolicy="strict-origin-when-cross-origin" allowFullScreen/></div><p className="muted" style={{fontSize:11}}>Nguồn: {media?.trailerSource||'Jikan'} · Video phát ngay trên AniMatch (nhấn Play trong khung).</p></>:<div className="trailer-unavailable">Không tìm thấy trailer nhúng. Hãy cấu hình YOUTUBE_API_KEY ở backend để tìm video đầu tiên trên YouTube. <a href={`https://www.youtube.com/results?search_query=${encodeURIComponent('anime '+animeId+' official trailer')}`} target="_blank" rel="noopener noreferrer">Tìm trên YouTube <ExternalLink size={13}/></a></div>}</div>;

}


/** Full-viewport, vertically scrollable detail sheet. Swiping downward opens it;
 * a Swipe action is never committed while this screen is active. */
export function AnimeDetailScreen({anime,explanation,onClose,onBookmark,franchise}:{anime:Anime;explanation?:string;onClose:()=>void;onBookmark?:()=>void;franchise?:React.ReactNode}){
 useEffect(()=>{const before=document.body.style.overflow;document.body.style.overflow='hidden';const handle=(e:KeyboardEvent)=>{if(e.key==='Escape')onClose()};window.addEventListener('keydown',handle);return()=>{document.body.style.overflow=before;window.removeEventListener('keydown',handle)}},[onClose]);
 return <div className="anime-detail-backdrop" role="presentation" onClick={onClose}>
  <section className="anime-detail-screen" role="dialog" aria-modal="true" aria-label={`Chi tiết anime ${anime.title}`} onClick={e=>e.stopPropagation()}>
   <header className="anime-detail-topbar"><span>ANIMATCH · ANIME DETAIL</span><button type="button" onClick={onClose} aria-label="Đóng chi tiết">✕ Đóng</button></header>
   <div className="anime-detail-hero"><div className="anime-detail-poster">{anime.imageUrl?<img src={anime.imageUrl} alt={`Poster ${anime.title}`}/>:<div className="no-poster">✦</div>}</div>
    <div className="anime-detail-head"><span className="eyebrow">FULL ANIME PROFILE</span><h1>{anime.title}</h1>{anime.titleEnglish&&anime.titleEnglish!==anime.title&&<p>{anime.titleEnglish}</p>}
     <div className="anime-detail-facts"><span>★ {anime.score?.toFixed(2)??'N/A'}</span><span>{anime.type||'Anime'}</span><span>{anime.episodes??'?'} tập</span><span>{anime.year??'—'}</span></div>
     <div className="genres">{anime.genres.map(g=><span className="tag" key={g}>{g}</span>)}</div>
     {onBookmark&&<button className="primary" type="button" onClick={onBookmark}><BookmarkPlus size={17}/> Add to Watchlist</button>}
    </div>
   </div>
   <div className="anime-detail-sections"><section><div className="eyebrow">WHY THIS MATCH?</div><h2>Lý do được đề xuất</h2><p>{explanation||'Thông tin giải thích cá nhân hóa chưa có cho anime này.'}</p></section>
   <section><div className="eyebrow">STORY</div><h2>Full Synopsis</h2><p className="full-synopsis">{anime.synopsis?.trim()||'Chưa có synopsis cho anime này.'}</p></section>
   <section><div className="eyebrow">PRODUCTION</div><h2>Studio & Release Details</h2><dl className="anime-detail-metadata"><div><dt>Studio</dt><dd>{anime.studios.length?anime.studios.join(', '):'Chưa có thông tin'}</dd></div><div><dt>Type</dt><dd>{anime.type||'—'}</dd></div><div><dt>Episodes</dt><dd>{anime.episodes??'Chưa xác định'}</dd></div><div><dt>Status</dt><dd>{anime.status||'—'}</dd></div><div><dt>Aired</dt><dd>{anime.airedFrom||'—'}</dd></div><div><dt>Genres / Themes</dt><dd>{[...anime.genres,...anime.themes].join(', ')||'—'}</dd></div></dl></section>
   <section><div className="eyebrow">WATCH</div><h2>Official / Available Trailer</h2><Trailer animeId={anime.id}/></section>
   {franchise&&<section>{franchise}</section>}
   </div>
  </section>
 </div>;
}

export function SwipeExperience({deck,vibe,pivot,busy,onAction,onUndo,onBookmark,historyCount,renderFranchise}:{deck:Recommendation[];vibe:VibeData;pivot:{active:boolean;reason:string|null};busy:boolean;onAction:(id:number,action:Action)=>Promise<void>;onUndo:()=>Promise<void>;onBookmark:(id:number)=>Promise<void>;historyCount:number;renderFranchise?:(id:number)=>React.ReactNode}){
 const card=deck[0]?.anime;const [betterImage,setBetterImage]=useState('');const [detailsOpen,setDetailsOpen]=useState(false);
 useEffect(()=>{if(!card){setBetterImage('');return};let active=true;setBetterImage('');fetch(`/api/anime/${card.id}/media`).then(r=>r.ok?r.json():null).then(d=>{if(active&&d?.imageUrl)setBetterImage(d.imageUrl)}).catch(()=>{});return()=>{active=false}},[card?.id]);
 const [drag,setDrag]=useState({x:0,y:0});const [anim,setAnim]=useState<'left'|'right'|'up'|null>(null);
 const pointer=useRef<{id:number;startX:number;startY:number}|null>(null);
 const locked=useRef(false);
 useEffect(()=>{setDrag({x:0,y:0});setAnim(null);setDetailsOpen(false);locked.current=false},[card?.id]);
 async function commit(direction:'left'|'right'|'up'){
  if(!card||busy||locked.current)return;locked.current=true;setAnim(direction);
  await new Promise(resolve=>setTimeout(resolve,230));
  try{await onAction(card.id,direction==='left'?'pass':direction==='right'?'like':'watched')}finally{locked.current=false;setAnim(null);setDrag({x:0,y:0})}
 }
 useEffect(()=>{const handler=(e:KeyboardEvent)=>{if(e.target instanceof HTMLInputElement||e.target instanceof HTMLTextAreaElement||e.repeat)return;if(e.key==='ArrowLeft'){e.preventDefault();void commit('left')}if(e.key==='ArrowRight'){e.preventDefault();void commit('right')}if(e.key==='ArrowUp'){e.preventDefault();void commit('up')}if(e.key==='ArrowDown'&&card){e.preventDefault();setDetailsOpen(true)}};window.addEventListener('keydown',handler);return()=>window.removeEventListener('keydown',handler)},[card?.id,busy]);
 const rotation=drag.x/19;
 const onPointerDown=(e:React.PointerEvent<HTMLDivElement>)=>{if(busy||locked.current||detailsOpen||!card||e.target instanceof HTMLElement && e.target.closest('a,button'))return;pointer.current={id:e.pointerId,startX:e.clientX,startY:e.clientY};e.currentTarget.setPointerCapture(e.pointerId)};
 const onPointerMove=(e:React.PointerEvent<HTMLDivElement>)=>{if(pointer.current?.id!==e.pointerId||busy)return;setDrag({x:e.clientX-pointer.current.startX,y:e.clientY-pointer.current.startY})};
 const onPointerUp=(e:React.PointerEvent<HTMLDivElement>)=>{if(pointer.current?.id!==e.pointerId)return;const delta={x:e.clientX-pointer.current.startX,y:e.clientY-pointer.current.startY};pointer.current=null;if(Math.abs(delta.x)>95&&Math.abs(delta.x)>Math.abs(delta.y)*1.1)void commit(delta.x>0?'right':'left');else if(delta.y< -110&&Math.abs(delta.y)>Math.abs(delta.x))void commit('up');else if(delta.y>100&&Math.abs(delta.y)>Math.abs(delta.x)){setDrag({x:0,y:0});setDetailsOpen(true)}else setDrag({x:0,y:0})};
 return <div className="swipe-experience">
  <div className="swipe-stage">
   <div className="swipe-topline"><div><span className="eyebrow">ONE SWIPE CLOSER</span><h2>Discover your next obsession.</h2><p>Quẹt <strong>phải để Like</strong> · <strong>trái để Pass</strong> · <strong>lên để Watched</strong> · <strong>xuống để xem chi tiết</strong></p></div><button className="quiet" onClick={()=>void onUndo()} disabled={busy||historyCount===0}><RotateCcw size={15}/> Undo</button></div>
   {pivot.active&&<div className="pivot-banner">✦ AUTO-PIVOT ACTIVE · {pivot.reason} · MMR sẽ cân bằng sự mới lạ và phù hợp.</div>}
   {card?<><div className="card-stack"><div className="stack-card stack-back"/>
    <div className={'gesture-card '+(anim?`flying-${anim}`:'')} role="group" aria-label={`Anime ${card.title}, kéo sang phải để Like, trái để Pass, lên để đã xem`}
      style={{transform:anim?undefined:`translate(${drag.x}px,${drag.y}px) rotate(${rotation}deg)`,touchAction:'none'}}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={()=>{pointer.current=null;setDrag({x:0,y:0})}}>
      {(betterImage||card.imageUrl)?<img draggable={false} src={betterImage||card.imageUrl} alt={`Poster ${card.title}`} loading="eager"/>:<div className="no-poster">✦</div>}
      <div className="gesture-gradient"/>
      <span className="gesture-stamp stamp-like" style={{opacity:Math.max(0,Math.min(1,drag.x/75))}}>LIKE ♥</span><span className="gesture-stamp stamp-pass" style={{opacity:Math.max(0,Math.min(1,-drag.x/75))}}>PASS ✕</span><span className="gesture-stamp stamp-watched" style={{opacity:Math.max(0,Math.min(1,-drag.y/90))}}>WATCHED ✓</span>
      <div className="gesture-caption"><span className="meta">{card.type} · {card.year||'N/A'} · ★ {card.score??'N/A'}</span><h2>{card.title}</h2><div className="genres">{card.genres.slice(0,4).map(g=><span className="tag" key={g}>{g}</span>)}</div></div>
    </div></div>
    <div className="swipe-guide" aria-label="Hướng dẫn điều khiển"><span><ChevronLeft size={17}/> PASS</span><span>↑ WATCHED</span><span>↓ DETAILS</span><span>LIKE <ChevronRight size={17}/></span></div>
    <div className="card-subinfo"><span>MATCH SCORE: {deck[0].relevance.toFixed(3)}</span><button type="button" className="quiet open-details" onClick={()=>setDetailsOpen(true)}>↓ Xem chi tiết</button><button onClick={()=>void onBookmark(card.id)} className="quiet" disabled={busy}><BookmarkPlus size={15}/> Watchlist</button></div>
    </>:<div className="vibe-empty">Không còn anime phù hợp. Hãy cập nhật sở thích hoặc thử lại.</div>}
  </div>
  <div className="swipe-insights"><VibeRadar vibe={vibe}/></div>{card&&detailsOpen&&<AnimeDetailScreen anime={card} explanation={deck[0].explanation} onClose={()=>setDetailsOpen(false)} onBookmark={()=>void onBookmark(card.id)} franchise={renderFranchise?.(card.id)}/>}
 </div>
}
