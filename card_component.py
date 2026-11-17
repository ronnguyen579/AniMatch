import re
import urllib.parse
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data_loader import get_main_studio


# Lấy link nhúng YouTube Trailer từ cột trailer_url trong CSV (không chặn luồng mạng)
@st.cache_data(ttl=86400, show_spinner=False)
def get_youtube_trailer_embed(title: str, csv_url: str = "") -> str:
    url = str(csv_url).strip()
    if url and url.lower() != "nan":
        if "youtube.com/embed/" in url:
            return url
        match_v = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        if match_v:
            return f"https://www.youtube.com/embed/{match_v.group(1)}?rel=0"
    return ""


# Render thẻ anime bằng HTML/CSS/JS hỗ trợ kéo chuột/cảm ứng (Trái=Pass, Phải=Like, Lên=Trailer)
def render_tinder_draggable_card(anime_row: pd.Series, sim_score: float, explanation_text: str, pos_tags: list[str]):
    title = str(anime_row["title"])
    img_url = str(anime_row["image_url"]).strip()
    if not img_url or img_url == "nan":
        img_url = "https://placehold.co/600x850/1e1b4b/f43f5e?text=AniMatch+Poster"

    studio = get_main_studio(anime_row["studios"])
    score = str(anime_row.get("score", "N/A"))
    year = str(anime_row.get("year", "N/A"))
    episodes = str(anime_row.get("episodes", "N/A"))
    anime_type = str(anime_row.get("type", "TV"))
    full_synopsis = str(anime_row["synopsis"]).replace('"', '&quot;').replace("'", "&#39;")
    short_synopsis = full_synopsis if len(full_synopsis) <= 230 else full_synopsis[:227] + "..."

    genres_list = [g.strip() for g in re.split(r"[|,]", str(anime_row["genres"])) if g.strip()][:5]
    genres_badges_html = "".join([f'<span class="genre-pill">{g}</span>' for g in genres_list])

    pos_tags_html = ""
    if pos_tags:
        pos_tags_html = f'<div class="key-match">✨ <b>Matched Features:</b> {", ".join(pos_tags[:4])}</div>'

    embed_url = get_youtube_trailer_embed(title, anime_row.get("trailer_url", ""))
    yt_search_query = urllib.parse.quote(f"{title} official anime trailer")
    yt_search_url = f"https://www.youtube.com/results?search_query={yt_search_query}"

    if embed_url:
        trailer_html = f"""
        <div class="trailer-container">
            <iframe src="{embed_url}" title="Anime Trailer" frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen></iframe>
        </div>
        """
    else:
        trailer_html = f"""
        <div class="no-trailer-box">
            <div>🎬 Direct trailer embed unavailable for this title.</div>
            <a href="{yt_search_url}" target="_blank" class="yt-btn">▶️ Watch Trailer on YouTube</a>
        </div>
        """

    f_role = str(anime_row.get("franchise_role", "ENTRY")).upper()
    parent_title = html.escape(str(anime_row.get("parent_title", "")).strip())
    cur_title = html.escape(str(anime_row.get("title", "")).strip())

    relation_badge_html = ""
    if f_role in ("SEQUEL", "PREQUEL", "SPIN_OFF") and parent_title and parent_title.lower() != cur_title.lower():
        badge_styles = {
            "SEQUEL": ("⏩ SEQUEL OF", "rgba(245, 158, 11, 0.18)", "#fbbf24", "rgba(245, 158, 11, 0.45)"),
            "PREQUEL": ("⏪ PREQUEL TO", "rgba(56, 189, 248, 0.18)", "#38bdf8", "rgba(56, 189, 248, 0.45)"),
            "SPIN_OFF": ("🔀 SPIN-OFF OF", "rgba(192, 132, 252, 0.18)", "#c084fc", "rgba(192, 132, 252, 0.45)"),
        }
        label_prefix, bg_col, txt_col, border_col = badge_styles[f_role]
        relation_badge_html = (
            f'<div style="display:inline-block; margin: 4px 0 6px 0; padding: 3px 9px; '
            f'border-radius: 6px; font-size: 11.5px; font-weight: 700; '
            f'background: {bg_col}; color: {txt_col}; border: 1px solid {border_col};">'
            f'{label_prefix}: {parent_title}</div>'
        )

    html_code = f"""<!-- card_id:{anime_row['anime_id']} -->
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{
            box-sizing: border-box;
            user-select: none;
            -webkit-user-select: none;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        body {{
            margin: 0;
            padding: 4px 0;
            background: transparent;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .hint-bar {{
            width: 100%;
            max-width: 460px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            padding: 6px 10px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.2px;
        }}
        .hint-left {{ color: #fb7185; }}
        .hint-center {{ color: #94a3b8; text-align: center; }}
        .hint-up {{ color: #38bdf8; }}
        .hint-down {{ color: #c084fc; }}
        .hint-right {{ color: #34d399; }}
        .arena {{
            position: relative;
            width: 100%;
            max-width: 420px;
            height: 605px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .tinder-card {{
            position: relative;
            width: 398px;
            height: 590px;
            background: linear-gradient(165deg, #1e293b 0%, #0f172a 100%);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.55);
            overflow: hidden;
            cursor: grab;
            touch-action: none;
            will-change: transform;
            display: flex;
            flex-direction: column;
            <div class="hint-bar">
                <span class="hint-left">⬅️ [←] PASS</span>
                <span class="hint-up">⬆️ [↑] Trailer • ⬇️ [↓] Watched</span>
                <span class="hint-right">LIKE [→] ➡️</span>
            </div>;
        }}
        .tinder-card:active {{ cursor: grabbing; }}
        .poster-wrap {{
            position: relative;
            width: 100%;
            height: 295px;
            overflow: hidden;
            background: #090d16;
        }}
        .poster-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            pointer-events: none;
        }}
        .poster-gradient {{
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 110px;
            background: linear-gradient(to top, #0f172a 5%, transparent 100%);
        }}
        .sim-badge {{
            position: absolute;
            top: 14px; right: 14px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(6px);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.4);
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }}
        .meta-badge {{
            position: absolute;
            top: 14px; left: 14px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(6px);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.4);
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }}
        .stamp {{
            position: absolute;
            top: 38px;
            padding: 8px 18px;
            border-radius: 10px;
            font-size: 26px;
            font-weight: 900;
            letter-spacing: 2px;
            opacity: 0;
            z-index: 20;
            pointer-events: none;
        }}
        .stamp-like {{
            left: 22px;
            color: #10b981;
            border: 4px solid #10b981;
            transform: rotate(-14deg);
            background: rgba(16, 185, 129, 0.15);
        }}
        .stamp-pass {{
            right: 22px;
            color: #f43f5e;
            border: 4px solid #f43f5e;
            transform: rotate(14deg);
            background: rgba(244, 63, 94, 0.15);
        }}
        .stamp-info {{
            left: 50%;
            top: 120px;
            transform: translateX(-50%);
            color: #38bdf8;
            border: 4px solid #38bdf8;
            background: rgba(15, 23, 42, 0.85);
            font-size: 20px;
            white-space: nowrap;
        }}
        .stamp-watched {{
            top: 40px; left: 50%; transform: translateX(-50%);
            color: #c084fc; border-color: #c084fc;
            background: rgba(192, 132, 252, 0.16);
        }}
        .card-body {{
            padding: 12px 18px 14px 18px;
            display: flex;
            flex-direction: column;
            gap: 7px;
            flex: 1;
            color: #f8fafc;
        }}
        .title-row {{
            font-size: 19px;
            font-weight: 800;
            line-height: 1.2;
            color: #ffffff;
        }}
        .sub-meta {{
            font-size: 12px;
            color: #94a3b8;
            font-weight: 600;
        }}
        .genres-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}
        .genre-pill {{
            background: rgba(244, 63, 94, 0.18);
            color: #fda4af;
            border: 1px solid rgba(244, 63, 94, 0.35);
            padding: 2px 9px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
        }}
        .synopsis {{
            font-size: 12px;
            line-height: 1.4;
            color: #cbd5e1;
            max-height: 66px;
            overflow: hidden;
        }}
        .explain-box {{
            margin-top: auto;
            background: rgba(56, 189, 248, 0.1);
            border-left: 3px solid #38bdf8;
            padding: 7px 10px;
            border-radius: 8px;
            font-size: 11px;
            color: #e0f2fe;
            line-height: 1.35;
        }}
        .key-match {{
            margin-top: 3px;
            color: #7dd3fc;
            font-size: 10.5px;
        }}
        .swipe-up-cue {{
            text-align: center;
            font-size: 11px;
            color: #38bdf8;
            font-weight: 700;
            padding-top: 2px;
            cursor: pointer;
        }}
        .detail-drawer {{
            position: absolute;
            left: 0; right: 0; bottom: 0;
            height: 100%;
            background: #0f172a;
            z-index: 50;
            transform: translateY(100%);
            transition: transform 0.32s cubic-bezier(0.2, 0.85, 0.25, 1);
            display: flex;
            flex-direction: column;
            padding: 16px 18px;
            color: #f8fafc;
            cursor: default;
        }}
        .detail-drawer.open {{
            transform: translateY(0%);
        }}
        .drawer-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.12);
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}
        .drawer-title {{
            font-size: 16px;
            font-weight: 800;
            color: #38bdf8;
        }}
        .close-btn {{
            background: rgba(244, 63, 94, 0.2);
            color: #fda4af;
            border: 1px solid rgba(244, 63, 94, 0.4);
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
        }}
        .close-btn:hover {{
            background: rgba(244, 63, 94, 0.35);
        }}
        .drawer-content {{
            overflow-y: auto;
            flex: 1;
            padding-right: 4px;
            user-select: text;
            -webkit-user-select: text;
        }}
        .drawer-section-label {{
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #94a3b8;
            margin: 10px 0 6px 0;
        }}
        .trailer-container {{
            position: relative;
            width: 100%;
            padding-bottom: 56.25%;
            border-radius: 12px;
            overflow: hidden;
            background: #000;
            border: 1px solid rgba(255,255,255,0.15);
        }}
        .trailer-container iframe {{
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
        }}
        .no-trailer-box {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px dashed rgba(56, 189, 248, 0.4);
            border-radius: 12px;
            padding: 14px;
            text-align: center;
            font-size: 12px;
            color: #cbd5e1;
        }}
        .yt-btn {{
            display: inline-block;
            margin-top: 8px;
            background: #ef4444;
            color: #ffffff;
            text-decoration: none;
            padding: 6px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 12px;
        }}
        .full-synopsis-text {{
            font-size: 13px;
            line-height: 1.55;
            color: #e2e8f0;
            white-space: pre-line;
        }}
    </style>
    </head>
    <body>
       <div class="hint-bar">
            <span class="hint-left">⬅️ [←] Pass</span>
            <span class="hint-center">
                <span class="hint-up">⬆️ [↑] Trailer</span> • 
                <span class="hint-down">⬇️ [↓] Watched</span>
            </span>
            <span class="hint-right">Like [→] ➡️</span>
        </div>
        <div class="arena">
            <div id="tinderCard" class="tinder-card">
                <div id="stampLike" class="stamp stamp-like">LIKE 💚</div>
                <div id="stampPass" class="stamp stamp-pass">PASS ❌</div>
                <div id="stampInfo" class="stamp stamp-info">INFO & TRAILER ⬆️</div>
                <div class="stamp stamp-watched" id="stampWatched">ALREADY WATCHED</div>

                <div class="poster-wrap">
                    <img src="{img_url}" alt="Poster" onerror="this.src='https://placehold.co/600x850/1e1b4b/f43f5e?text=Anime+Poster'"/>
                    <div class="poster-gradient"></div>
                    <div class="meta-badge">⭐ {score} | {anime_type} ({year})</div>
                    <div class="sim-badge">Match: {sim_score:.3f}</div>
                </div>

                <div class="card-body">
                    <div class="title-row">{title}</div>
                    {relation_badge_html}
                    <div class="sub-meta">🎬 Studio: {studio} &nbsp;•&nbsp; 📺 {episodes} eps</div>
                    <div class="genres-row">{genres_badges_html}</div>
                    <div class="synopsis">{short_synopsis}</div>
                    <div class="explain-box">
                        💡 <b>Explainable AI:</b> {explanation_text}
                        {pos_tags_html}
                    </div>
                    <div id="btnOpenDrawer" class="swipe-up-cue">⬆️ Drag card UP to watch Trailer & read Full Synopsis</div>
                </div>

                <div id="detailDrawer" class="detail-drawer">
                    <div class="drawer-header">
                        <div class="drawer-title">🎬 {title}</div>
                        <button id="btnCloseDrawer" class="close-btn">⬇️ Close</button>
                    </div>
                    <div class="drawer-content">
                        <div class="drawer-section-label">🎥 Official Trailer</div>
                        {trailer_html}
                        <div class="drawer-section-label">📖 Full Synopsis</div>
                        <div class="full-synopsis-text">{full_synopsis}</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const CURRENT_CARD_ID = "{int(anime_row['anime_id'])}";
            if (window.parent) {{
                // Mở khóa khi server đã render sang thẻ mới
                if (window.parent._animatchLastRenderedId !== CURRENT_CARD_ID) {{
                    window.parent._animatchLastRenderedId = CURRENT_CARD_ID;
                    window.parent._animatchBusy = false;
                }}
            }}
            function hideBridgeButtons() {{
                try {{
                    const buttons = window.parent.document.querySelectorAll('button');
                    buttons.forEach(btn => {{
                        if (btn.innerText && btn.innerText.includes('BRIDGE_SWIPE_')) {{
                            const box = btn.closest('div[data-testid="stButton"]') || btn.parentElement;
                            if (box) {{
                                box.style.position = 'fixed';
                                box.style.left = '-9999px';
                                box.style.opacity = '0';
                                box.style.height = '0';
                                box.style.overflow = 'hidden';
                            }}
                        }}
                    }});
                }} catch (e) {{}}
            }}
            hideBridgeButtons();
            setTimeout(hideBridgeButtons, 150);

            const card = document.getElementById('tinderCard');
            const stampLike = document.getElementById('stampLike');
            const stampPass = document.getElementById('stampPass');
            const stampInfo = document.getElementById('stampInfo');
            const stampWatched = document.getElementById('stampWatched');
            const detailDrawer = document.getElementById('detailDrawer');
            const btnCloseDrawer = document.getElementById('btnCloseDrawer');
            const btnOpenDrawer = document.getElementById('btnOpenDrawer');

            let isDragging = false;
            let isDrawerOpen = false;
            let isAnimatingKeyboard = false;
            let startX = 0, startY = 0, currentX = 0, currentY = 0;
            const SWIPE_X_THRESHOLD = 100;
            const SWIPE_UP_THRESHOLD = -80;
            const SWIPE_DOWN_THRESHOLD = 85;

            function openDrawer() {{
                isDrawerOpen = true;
                detailDrawer.classList.add('open');
                card.style.transform = 'translate(0px, 0px) rotate(0deg)';
                card.style.borderColor = 'rgba(56, 189, 248, 0.75)';
                stampLike.style.opacity = 0;
                stampPass.style.opacity = 0;
                stampInfo.style.opacity = 0;
                stampWatched.style.opacity = 0;
            }}

            function closeDrawer() {{
                isDrawerOpen = false;
                detailDrawer.classList.remove('open');
                card.style.borderColor = 'rgba(255, 255, 255, 0.14)';
            }}

            btnCloseDrawer.addEventListener('pointerdown', (e) => e.stopPropagation());
            btnCloseDrawer.addEventListener('click', (e) => {{
                e.stopPropagation();
                closeDrawer();
            }});

            btnOpenDrawer.addEventListener('pointerdown', (e) => e.stopPropagation());
            btnOpenDrawer.addEventListener('click', (e) => {{
                e.stopPropagation();
                openDrawer();
            }});

            detailDrawer.addEventListener('pointerdown', (e) => e.stopPropagation());

            function triggerStreamlitSwipe(direction) {{
                if (window.parent && window.parent._animatchBusy) return;
                let targetMarker = 'BRIDGE_SWIPE_PASS';
                if (direction === 'right') targetMarker = 'BRIDGE_SWIPE_LIKE';
                else if (direction === 'down') targetMarker = 'BRIDGE_SWIPE_WATCHED';

                try {{
                    const parentButtons = window.parent.document.querySelectorAll('button');
                    for (const btn of parentButtons) {{
                        if (btn.innerText && btn.innerText.includes(targetMarker)) {{
                            btn.click();
                            return;
                        }}
                    }}
                }} catch (err) {{}}
            }}

            card.addEventListener('pointerdown', (e) => {{
                if (isDrawerOpen || isAnimatingKeyboard) return;
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
                currentX = 0;
                currentY = 0;
                card.style.transition = 'none';
                card.setPointerCapture(e.pointerId);
            }});

            card.addEventListener('pointermove', (e) => {{
                if (!isDragging || isDrawerOpen) return;
                currentX = e.clientX - startX;
                currentY = e.clientY - startY;

                if (currentY < -20 && Math.abs(currentY) > Math.abs(currentX)) {{
                    const liftY = Math.max(currentY * 0.45, -70);
                    card.style.transform = `translate(0px, ${{liftY}}px) scale(1.02)`;
                    stampInfo.style.opacity = Math.min((Math.abs(currentY) - 20) / 60, 1);
                    stampLike.style.opacity = 0;
                    stampPass.style.opacity = 0;
                    stampWatched.style.opacity = 0;
                    card.style.borderColor = 'rgba(56, 189, 248, 0.8)';
                }} else if (currentY > 25 && Math.abs(currentY) > Math.abs(currentX)) {{
                    const dropY = Math.min(currentY * 0.65, 160);
                    card.style.transform = `translate(0px, ${{dropY}}px) scale(0.97)`;
                    stampWatched.style.opacity = Math.min((currentY - 25) / 60, 1);
                    stampLike.style.opacity = 0;
                    stampPass.style.opacity = 0;
                    stampInfo.style.opacity = 0;
                    card.style.borderColor = 'rgba(192, 132, 252, 0.85)';
                }} else {{
                    const dampY = currentY * 0.25;
                    const rotate = currentX * 0.06;
                    card.style.transform = `translate(${{currentX}}px, ${{dampY}}px) rotate(${{rotate}}deg)`;
                    stampInfo.style.opacity = 0;
                    stampWatched.style.opacity = 0;

                    if (currentX > 20) {{
                        stampLike.style.opacity = Math.min((currentX - 20) / 80, 1);
                        stampPass.style.opacity = 0;
                        card.style.borderColor = 'rgba(16, 185, 129, 0.75)';
                    }} else if (currentX < -20) {{
                        stampPass.style.opacity = Math.min((Math.abs(currentX) - 20) / 80, 1);
                        stampLike.style.opacity = 0;
                        card.style.borderColor = 'rgba(244, 63, 94, 0.75)';
                    }} else {{
                        stampLike.style.opacity = 0;
                        stampPass.style.opacity = 0;
                        card.style.borderColor = 'rgba(255, 255, 255, 0.14)';
                    }}
                }}
            }});

            function endDrag() {{
                if (!isDragging || isDrawerOpen) return;
                isDragging = false;
                card.style.transition = 'transform 0.18s cubic-bezier(0.2, 0.8, 0.2, 1), opacity 0.18s ease';

                if (currentY <= SWIPE_UP_THRESHOLD && Math.abs(currentY) > Math.abs(currentX)) {{
                    openDrawer();
                }} else if (currentY >= SWIPE_DOWN_THRESHOLD && Math.abs(currentY) > Math.abs(currentX)) {{
                    card.style.transform = 'translate(0px, 520px) scale(0.92)';
                    card.style.opacity = '0';
                    setTimeout(() => triggerStreamlitSwipe('down'), 110);
                }} else if (currentX >= SWIPE_X_THRESHOLD) {{
                    card.style.transform = `translate(650px, ${{currentY * 0.25}}px) rotate(28deg)`;
                    card.style.opacity = '0';
                    setTimeout(() => triggerStreamlitSwipe('right'), 110);
                }} else if (currentX <= -SWIPE_X_THRESHOLD) {{
                    card.style.transform = `translate(-650px, ${{currentY * 0.25}}px) rotate(-28deg)`;
                    card.style.opacity = '0';
                    setTimeout(() => triggerStreamlitSwipe('left'), 110);
                }} else {{
                    card.style.transform = 'translate(0px, 0px) rotate(0deg)';
                    card.style.borderColor = 'rgba(255, 255, 255, 0.14)';
                    stampLike.style.opacity = 0;
                    stampPass.style.opacity = 0;
                    stampInfo.style.opacity = 0;
                    stampWatched.style.opacity = 0;
                }}
                currentX = 0;
                currentY = 0;
            }}

            card.addEventListener('pointerup', endDrag);
            card.addEventListener('pointercancel', endDrag);

            function handleKeyAction(e) {{
                const activeTag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
                if (activeTag === 'input' || activeTag === 'textarea' || activeTag === 'select') return;
                if (isAnimatingKeyboard || isDragging) return;

                if (isDrawerOpen) {{
                    if (e.key === 'Escape' || e.key === 'ArrowDown') {{
                        e.preventDefault();
                        closeDrawer();
                    }}
                    return;
                }}

                if (e.key === 'ArrowRight') {{
                    e.preventDefault();
                    isAnimatingKeyboard = true;
                    card.style.transition = 'transform 0.18s cubic-bezier(0.2, 0.8, 0.2, 1), opacity 0.18s ease';
                    stampLike.style.opacity = 1;
                    card.style.borderColor = 'rgba(16, 185, 129, 0.75)';
                    card.style.transform = 'translate(650px, 20px) rotate(28deg)';
                    card.style.opacity = '0';
                    setTimeout(() => triggerStreamlitSwipe('right'), 110);
                }} else if (e.key === 'ArrowLeft') {{
                    e.preventDefault();
                    isAnimatingKeyboard = true;
                    card.style.transition = 'transform 0.18s cubic-bezier(0.2, 0.8, 0.2, 1), opacity 0.18s ease';
                    stampPass.style.opacity = 1;
                    card.style.borderColor = 'rgba(244, 63, 94, 0.75)';
                    card.style.transform = 'translate(-650px, 20px) rotate(-28deg)';
                    card.style.opacity = '0';
                    setTimeout(() => triggerStreamlitSwipe('left'), 110);
                }} else if (e.key === 'ArrowDown') {{
                    e.preventDefault();
                    isAnimatingKeyboard = true;
                    card.style.transition = 'transform 0.18s cubic-bezier(0.2, 0.8, 0.2, 1), opacity 0.18s ease';
                    stampWatched.style.opacity = 1;
                    card.style.borderColor = 'rgba(192, 132, 252, 0.85)';
                    card.style.transform = 'translate(0px, 520px) scale(0.92)';
                    card.style.opacity = '0';
                    setTimeout(() => triggerStreamlitSwipe('down'), 110);
                }} else if (e.key === 'ArrowUp') {{
                    e.preventDefault();
                    openDrawer();
                }}
            }}

            window.addEventListener('keydown', handleKeyAction);
            try {{
                if (window.parent && window.parent !== window) {{
                    if (window.parent._animatchKeyHandler) {{
                        window.parent.removeEventListener('keydown', window.parent._animatchKeyHandler);
                    }}
                    window.parent._animatchKeyHandler = handleKeyAction;
                    window.parent.addEventListener('keydown', handleKeyAction);
                }}
            }} catch (err) {{}}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=640, scrolling=False)
