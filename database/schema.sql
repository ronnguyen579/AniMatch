CREATE TABLE IF NOT EXISTS anime (
 id INTEGER PRIMARY KEY, title TEXT NOT NULL, title_english TEXT NOT NULL DEFAULT '',
 type TEXT NOT NULL DEFAULT 'TV', year INTEGER, episodes INTEGER, status TEXT NOT NULL DEFAULT '', aired_from DATE, score DOUBLE PRECISION, popularity INTEGER,
 genres TEXT[] NOT NULL DEFAULT '{}', themes TEXT[] NOT NULL DEFAULT '{}', demographics TEXT[] NOT NULL DEFAULT '{}',
 studios TEXT[] NOT NULL DEFAULT '{}', synopsis TEXT NOT NULL DEFAULT '', image_url TEXT NOT NULL DEFAULT '',
 artstyle_tags TEXT[] NOT NULL DEFAULT '{}', visual_tags TEXT[] NOT NULL DEFAULT '{}',
 franchise_ids INTEGER[] NOT NULL DEFAULT '{}', entry_id INTEGER, prereq_id INTEGER,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS anime_score_idx ON anime(score DESC);
CREATE INDEX IF NOT EXISTS anime_genre_idx ON anime USING GIN(genres);
CREATE INDEX IF NOT EXISTS anime_title_search_idx ON anime USING GIN(to_tsvector('simple', title));
CREATE TABLE IF NOT EXISTS profiles (
 id UUID PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE,
 display_name TEXT NOT NULL DEFAULT 'Anime Explorer', mode TEXT NOT NULL DEFAULT 'content',
 genres TEXT[] NOT NULL DEFAULT '{}', themes TEXT[] NOT NULL DEFAULT '{}', demographics TEXT[] NOT NULL DEFAULT '{}', artstyles TEXT[] NOT NULL DEFAULT '{}', length_preference TEXT NOT NULL DEFAULT 'all', past_genres TEXT[] NOT NULL DEFAULT '{}', explore_genres TEXT[] NOT NULL DEFAULT '{}',
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS interactions (
 id BIGSERIAL PRIMARY KEY, profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
 anime_id INTEGER NOT NULL REFERENCES anime(id), action TEXT NOT NULL CHECK (action IN ('like','pass','watched')), source TEXT NOT NULL DEFAULT 'swipe',
 created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS interactions_profile_idx ON interactions(profile_id,id DESC);
CREATE TABLE IF NOT EXISTS recommendation_events (
 id BIGSERIAL PRIMARY KEY, profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
 anime_ids INTEGER[] NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS watchlist (
 profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
 anime_id INTEGER NOT NULL REFERENCES anime(id), created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY(profile_id,anime_id)
);

CREATE TABLE IF NOT EXISTS reentry_queue (
 profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
 anime_id INTEGER NOT NULL REFERENCES anime(id),
 eligible_after BIGINT NOT NULL DEFAULT 0,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY(profile_id,anime_id)
);
ALTER TABLE anime ADD COLUMN IF NOT EXISTS episodes INTEGER;
ALTER TABLE anime ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT '';
ALTER TABLE anime ADD COLUMN IF NOT EXISTS aired_from DATE;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS length_preference TEXT NOT NULL DEFAULT 'all';
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS past_genres TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS explore_genres TEXT[] NOT NULL DEFAULT '{}';
-- Safe incremental migration for profiles created with AniMatch 2.0 beta.
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'content';

-- Updates for installations created by AniMatch 2.0/2.1.
ALTER TABLE anime ADD COLUMN IF NOT EXISTS demographics TEXT[] NOT NULL DEFAULT '{}';
CREATE TABLE IF NOT EXISTS anilist_relation_cache (
  anime_id INTEGER PRIMARY KEY REFERENCES anime(id) ON DELETE CASCADE,
  relations JSONB NOT NULL DEFAULT '[]'::jsonb,
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS demographics TEXT[] NOT NULL DEFAULT '{}';

CREATE TABLE IF NOT EXISTS external_watched (
 profile_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
 mal_id INTEGER NOT NULL CHECK (mal_id>0),
 title TEXT NOT NULL DEFAULT '',
 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
 PRIMARY KEY(profile_id,mal_id)
);
CREATE TABLE IF NOT EXISTS anime_media_cache (
 anime_id INTEGER PRIMARY KEY REFERENCES anime(id) ON DELETE CASCADE,
 payload JSONB NOT NULL,
 fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE interactions ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'swipe';
