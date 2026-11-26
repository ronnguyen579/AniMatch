export type Action = 'like'|'pass'|'watched';
export interface Anime { id:number; title:string; titleEnglish:string; type:string; year:number|null; score:number|null; genres:string[]; themes:string[]; demographics:string[]; studios:string[]; synopsis:string; imageUrl:string; artstyles:string[]; visualTags:string[]; entryId:number|null; prereqId:number|null; popularity:number|null; episodes:number|null; status:string; airedFrom:string|null }
export type DiscoveryMode = 'eyecandy'|'content'|'lapsed';
export interface Profile {id:string; displayName:string; mode:DiscoveryMode; genres:string[];themes:string[];demographics:string[];artstyles:string[]; lengthPreference?:string; pastGenres?:string[]; exploreGenres?:string[]}
export interface Interaction {id:number; animeId:number;action:Action;createdAt:string}
export interface Recommendation { anime:Anime; relevance:number; contentScore:number; visualScore:number; explanation:string }

/** Curated visual *style* vocabulary: exclude objects, clothing, weapons and watermarks.
 * Both training and radar must use this exact predicate. */
export function isVisualStyleTag(raw:string):boolean {
 const t=raw.toLowerCase().trim().replace(/[\s-]+/g,'_');
 if(!t || /(copyright|artist_name|weapon|gun|shirt|pants|shorts|breasts|boots|holding|dress|hair|eyes|focus|looking_at|character|text|logo|watermark|signature|rating|tagme|unknown)/.test(t))return false;
 return /(?:^|_)(?:retro_artstyle|artstyle|blue_theme|red_theme|green_theme|yellow_theme|pink_theme|orange_theme|purple_theme|black_theme|limited_palette|watercolor|watercolour|ink_wash|ink_painting|oil_painting|pastel|monochrome|greyscale|grayscale|black_and_white|pixel_art|pixelated|cel_shading|soft_shading|flat_color|flat_colors|lineart|line_art|sketch|painterly|traditional_media|chibi|silhouette|high_contrast|film_grain|color_palette|colour_palette|gradient|cinematic_lighting|dramatic_lighting|soft_lighting|rim_lighting|neon_lighting|vintage_style|anime_screencap|3d|cgi|limited_palette)(?:_|$)/.test(t)||/^(?:19[7-9]0s|20[0-2]0s)\s*\(style\)$/.test(raw.trim().toLowerCase());
}
export function getAnimeVisualStyles(a:Pick<Anime,'artstyles'|'visualTags'>):string[] {
 return [...new Set([...a.artstyles,...a.visualTags].filter(isVisualStyleTag))];
}
