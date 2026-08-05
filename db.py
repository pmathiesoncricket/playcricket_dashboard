import streamlit as st
import pandas as pd
import uuid
import time
from sqlalchemy.exc import OperationalError, TimeoutError as SATimeoutError

# ---------- Setup ----------

conn = st.connection("postgresql", type="sql")

# Retry/backoff specifically for connection-pool exhaustion. This happens
# when many @st.cache_data functions miss at once (e.g. right after a
# deploy or a "Clear caches") and all race for a DB connection from a
# small SQLAlchemy pool. A short wait usually frees up a connection as
# other queries finish, so we retry a few times with increasing delay
# before giving up and raising.
MAX_QUERY_ATTEMPTS = 4
RETRY_BASE_DELAY_SECONDS = 1.5


def _query_with_retry(sql, params=None, ttl=0):
    last_exc = None
    for attempt in range(1, MAX_QUERY_ATTEMPTS + 1):
        try:
            return conn.query(sql, params=params, ttl=ttl)
        except (OperationalError, SATimeoutError) as exc:
            last_exc = exc
            if attempt < MAX_QUERY_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY_SECONDS * attempt)
            else:
                raise
    raise last_exc


# ---------- Internal helpers ----------

def _stringify_uuids(df: pd.DataFrame) -> pd.DataFrame:
    """
    psycopg2/SQLAlchemy return PostgreSQL `uuid` columns as native
    uuid.UUID objects by default. Supabase's PostgREST layer always
    serialized them as plain strings instead, which the rest of this app's
    filtering logic (e.g. `batter_id == str(selected_id)`) depends on.
    This normalizes any uuid.UUID values back to plain strings immediately
    after fetching, so every downstream comparison keeps working exactly
    as it did against Supabase.
    """
    for col in df.columns:
        non_null = df[col].dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], uuid.UUID):
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, uuid.UUID) else x)
    return df


def fetch_all_rows(table_name: str, select_str: str, eq_filters: dict | None = None,
                    order_col: str | None = None, page_size: int = 1000,
                    id_col: str | None = None) -> pd.DataFrame:
    """
    Fetch ALL rows from a Postgres table, paging manually via raw SQL.

    Two pagination strategies:
    - Offset pagination (default): LIMIT/OFFSET. Fine for small/medium tables.
    - Keyset pagination (pass id_col, e.g. a primary key): "WHERE id_col >
      last_seen_id ORDER BY id_col LIMIT page_size" instead of OFFSET.
      Required for large tables (e.g. deliveries) -- OFFSET has to scan and
      discard every prior row on each page, which gets slower as the offset
      grows. Keyset pagination stays fast regardless of table depth.
    """
    pages = []

    if id_col:
        last_id = None
        while True:
            where_clauses = []
            params = {}
            if eq_filters:
                for i, (col, val) in enumerate(eq_filters.items()):
                    where_clauses.append(f"{col} = :eq_{i}")
                    params[f"eq_{i}"] = val
            if last_id is not None:
                where_clauses.append(f"{id_col} > :last_id")
                params["last_id"] = last_id
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            params["page_size"] = page_size
            sql = f"SELECT {select_str} FROM {table_name} {where_sql} ORDER BY {id_col} LIMIT :page_size"

            df_page = _query_with_retry(sql, params=params, ttl=0)
            if df_page.empty:
                break
            pages.append(df_page)
            if len(df_page) < page_size:
                break
            last_id = df_page.iloc[-1][id_col]
    else:
        start = 0
        while True:
            where_clauses = []
            params = {}
            if eq_filters:
                for i, (col, val) in enumerate(eq_filters.items()):
                    where_clauses.append(f"{col} = :eq_{i}")
                    params[f"eq_{i}"] = val
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            order_sql = f"ORDER BY {order_col}" if order_col else ""
            params["limit"] = page_size
            params["offset"] = start
            sql = f"SELECT {select_str} FROM {table_name} {where_sql} {order_sql} LIMIT :limit OFFSET :offset"

            df_page = _query_with_retry(sql, params=params, ttl=0)
            if df_page.empty:
                break
            pages.append(df_page)
            if len(df_page) < page_size:
                break
            start += page_size

    if not pages:
        return pd.DataFrame()
    return _stringify_uuids(pd.concat(pages, ignore_index=True))


# ---------- Cached data getters ----------

@st.cache_data(ttl=300)
def get_matches():
    return fetch_all_rows(
        "matches",
        "match_id, grade, match_type, round, venue, ground, day_1_start, day_2_start, "
        "home_team_id, home_team, away_team_id, away_team, result_text, "
        "organisation_id, competition_id, competition_name, "
        "day1_stream_url, day1_stream_start, day2_stream_url, day2_stream_start",
        order_col="match_id",
    )


@st.cache_data(ttl=300)
def get_innings():
    """
    Full `innings` table -- one row per innings with the OFFICIAL runs,
    wickets, overs and extras breakdown (byes/leg byes/wides/no-balls/
    penalties). This is the only place total innings extras exist; the
    per-player batting/bowling figures in player_innings don't carry them.
    Powers the Match Summary tab's score-summary lines and the
    "team total vs sum of individual batting figures" extras note.
    """
    return fetch_all_rows(
        "innings",
        "innings_id, match_id, innings_number, innings_order, innings_name, "
        "batting_team_id, batting_team, bowling_team_id, bowling_team, "
        "close_type, declared, runs, wickets, overs, extras, byes, leg_byes, "
        "wides, no_balls, penalties",
        order_col="match_id",
    )


@st.cache_data(ttl=300)
def get_batting_innings():
    """
    player_innings for role='batting', joined to matches for grade/match_type/date/opponent,
    and joined to player_style for bowling type (pace_spin) and bowling style (bowl_style).
    """
    pi_df = fetch_all_rows("player_innings", "*", eq_filters={"role": "batting"})

    m_df = get_matches()
    if m_df.empty or pi_df.empty:
        return pd.DataFrame()

    ps_df = fetch_all_rows("player_style", "player_id, pace_spin, bowl_style")

    pi_df = pi_df.merge(ps_df, on="player_id", how="left")

    df = pi_df.merge(m_df, on="match_id", how="left")

    df["opponent_team"] = None
    mask_home = df["team"] == df["home_team"]
    df.loc[mask_home, "opponent_team"] = df.loc[mask_home, "away_team"]
    df.loc[~mask_home, "opponent_team"] = df.loc[~mask_home, "home_team"]

    return df


@st.cache_data(ttl=300)
def get_bowling_innings():
    """
    player_innings for role='bowling', joined to matches for grade/match_type/date/opponent,
    and joined to player_style for the BOWLER'S OWN batting hand / pace_spin / bowl_style.

    This mirrors get_batting_innings() exactly, including the same
    self-referential player_style join (on this row's own player_id) that
    powers the Batting tab's "Bowling type"/"Bowling style" sidebar filters.
    Here it powers "Bowling type", "Bowling style" AND the new "Batter hand
    (bowler's own)" filter -- all three describe the bowler themselves, not
    whoever they were bowling to on a given ball. The hand of the batter
    actually FACED (a ball-by-ball concept) is handled separately in
    tab_bowling.py by joining player_style a second time onto deliveries via
    batter_id -- i.e. the same player_style table joined twice, under two
    different aliases, for two different purposes.
    """
    pi_df = fetch_all_rows("player_innings", "*", eq_filters={"role": "bowling"})

    m_df = get_matches()
    if m_df.empty or pi_df.empty:
        return pd.DataFrame()

    ps_df = fetch_all_rows("player_style", "player_id, batter_hand, pace_spin, bowl_style")

    pi_df = pi_df.merge(ps_df, on="player_id", how="left")

    df = pi_df.merge(m_df, on="match_id", how="left")

    df["opponent_team"] = None
    mask_home = df["team"] == df["home_team"]
    df.loc[mask_home, "opponent_team"] = df.loc[mask_home, "away_team"]
    df.loc[~mask_home, "opponent_team"] = df.loc[~mask_home, "home_team"]

    return df


@st.cache_data(ttl=300)
def get_deliveries_for_batter(batter_id: str):
    pages = []
    start = 0
    page_size = 1000

    while True:
        sql = """
            SELECT * FROM deliveries
            WHERE batter_id = :batter_id
            ORDER BY innings_id, over, ball_number
            LIMIT :limit OFFSET :offset
        """
        df_page = _query_with_retry(
            sql,
            params={"batter_id": batter_id, "limit": page_size, "offset": start},
            ttl=0,
        )
        if df_page.empty:
            break
        pages.append(df_page)
        if len(df_page) < page_size:
            break
        start += page_size

    if not pages:
        return pd.DataFrame()
    return _stringify_uuids(pd.concat(pages, ignore_index=True))


@st.cache_data(ttl=300)
def get_deliveries_for_bowler(bowler_id: str):
    """Mirrors get_deliveries_for_batter, filtered on bowler_id instead --
    used by the Bowling tab's per-bowler detail sections (batter-hand
    breakdown, match-by-match ball-by-ball listing, position/phase splits,
    boundary rate, highlights)."""
    pages = []
    start = 0
    page_size = 1000

    while True:
        sql = """
            SELECT * FROM deliveries
            WHERE bowler_id = :bowler_id
            ORDER BY innings_id, over, ball_number
            LIMIT :limit OFFSET :offset
        """
        df_page = _query_with_retry(
            sql,
            params={"bowler_id": bowler_id, "limit": page_size, "offset": start},
            ttl=0,
        )
        if df_page.empty:
            break
        pages.append(df_page)
        if len(df_page) < page_size:
            break
        start += page_size

    if not pages:
        return pd.DataFrame()
    return _stringify_uuids(pd.concat(pages, ignore_index=True))


@st.cache_data(ttl=300)
def get_deliveries_for_batters(batter_ids: tuple[str, ...]):
    """Bulk fetch of deliveries for a set of batters (used by the season
    report, which needs many batters' deliveries in one shot rather than
    one connection per batter)."""
    if not batter_ids:
        return pd.DataFrame()

    pages = []
    start = 0
    page_size = 5000
    params = {}
    placeholders = []
    for i, bid in enumerate(batter_ids):
        key = f"bid_{i}"
        placeholders.append(f":{key}")
        params[key] = bid
    in_sql = ", ".join(placeholders)

    while True:
        params["limit"] = page_size
        params["offset"] = start
        sql = f"""
            SELECT * FROM deliveries
            WHERE batter_id IN ({in_sql})
            ORDER BY match_id, innings_id, over, ball_number
            LIMIT :limit OFFSET :offset
        """
        df_page = _query_with_retry(sql, params=params, ttl=0)
        if df_page.empty:
            break
        pages.append(df_page)
        if len(df_page) < page_size:
            break
        start += page_size

    if not pages:
        return pd.DataFrame()
    return _stringify_uuids(pd.concat(pages, ignore_index=True))


@st.cache_data(ttl=300)
def get_deliveries_for_match(match_id: str):
    """
    All deliveries for a single match -- a cheap point-lookup (the
    deliveries_match_innings_idx index covers exactly this), typically a
    few hundred rows for a whole match. Powers the entire Match Summary
    tab (batting singles/dot-ball counts, bowling ball-by-ball detail,
    5-dot-over calculation) from ONE fetch instead of one query per player.
    """
    pages = []
    start = 0
    page_size = 1000

    while True:
        sql = """
            SELECT * FROM deliveries
            WHERE match_id = :match_id
            ORDER BY innings_id, over, ball_number
            LIMIT :limit OFFSET :offset
        """
        df_page = _query_with_retry(
            sql,
            params={"match_id": match_id, "limit": page_size, "offset": start},
            ttl=0,
        )
        if df_page.empty:
            break
        pages.append(df_page)
        if len(df_page) < page_size:
            break
        start += page_size

    if not pages:
        return pd.DataFrame()
    return _stringify_uuids(pd.concat(pages, ignore_index=True))


@st.cache_data(ttl=300)
def get_highlights():
    """Highlight clips (fours, sixes, dismissals, etc.) with video URLs."""
    return fetch_all_rows("highlights", "*")


@st.cache_data(ttl=300)
def get_bowler_summary():
    """
    One row per bowler with total balls bowled and the set of grades they
    appear in -- computed entirely in Postgres via GROUP BY, so only a
    few hundred rows cross the network instead of every individual delivery.
    """
    sql = """
        SELECT
            d.bowler_id,
            MAX(d.bowler) AS bowler_name,
            COUNT(*) AS balls,
            array_agg(DISTINCT m.grade) AS grades
        FROM deliveries d
        LEFT JOIN matches m ON m.match_id = d.match_id
        WHERE d.bowler_id IS NOT NULL
        GROUP BY d.bowler_id
    """
    df = _query_with_retry(sql, ttl=0)
    return _stringify_uuids(df)


@st.cache_data(ttl=300)
def get_batter_summary():
    """
    One row per batter with total balls faced and the set of grades they
    appear in -- mirrors get_bowler_summary() exactly (same GROUP BY
    approach), just keyed on batter_id/batter instead of bowler_id/bowler.
    Powers the Batter Style tab.
    """
    sql = """
        SELECT
            d.batter_id,
            MAX(d.batter) AS batter_name,
            COUNT(*) AS balls,
            array_agg(DISTINCT m.grade) AS grades
        FROM deliveries d
        LEFT JOIN matches m ON m.match_id = d.match_id
        WHERE d.batter_id IS NOT NULL
        GROUP BY d.batter_id
    """
    df = _query_with_retry(sql, ttl=0)
    return _stringify_uuids(df)


@st.cache_data(ttl=300)
def get_bowling_conceded_summary():
    """
    One row per (bowler_id, match_id, innings_id) with legal balls bowled,
    runs conceded (charged to the bowler), and fours/sixes conceded --
    aggregated server-side via GROUP BY so the full deliveries table never
    has to cross the network just to power fours/sixes-conceded figures
    (which don't exist as columns on player_innings for bowling rows) or
    population-level boundary-rate comparisons on the Bowling tab.

    Grouping by match_id + innings_id (not just bowler_id) lets the caller
    filter this down to whatever subset of matches the sidebar filters
    currently select, by inner-joining on the same (bowler_id, match_id,
    innings_id) keys as the filtered player_innings rows -- rather than
    always reflecting a bowler's whole career regardless of filters.

    Fours/sixes are identified by batter_runs = 4 / 6 (i.e. runs actually
    scored off the bat on that delivery) -- NOT by parsing the free-text
    `description` column, which doesn't reliably contain the words
    "FOUR"/"SIX".

    runs conceded uses deliveries.bowler_runs where populated (the ball's
    true bowler-attributed runs, including wide/no-ball penalty runs but
    excluding byes/leg-byes), falling back to batter_runs + wides + no_balls
    for any rows where bowler_runs hasn't been backfilled.
    """
    sql = """
        SELECT
            bowler_id,
            match_id,
            innings_id,
            COUNT(*) FILTER (WHERE wides = 0 AND no_balls = 0) AS legal_balls,
            SUM(COALESCE(bowler_runs, batter_runs + wides + no_balls)) AS runs_conceded,
            COUNT(*) FILTER (WHERE batter_runs = 4) AS fours,
            COUNT(*) FILTER (WHERE batter_runs = 6) AS sixes
        FROM deliveries
        WHERE bowler_id IS NOT NULL
        GROUP BY bowler_id, match_id, innings_id
    """
    df = _query_with_retry(sql, ttl=0)
    return _stringify_uuids(df)


@st.cache_data(ttl=300)
def get_wicket_deliveries():
    """
    One row per recorded dismissal in `deliveries` (bowler_id, batter_id,
    dismissal_type, plus match/innings ids) -- NOT the full deliveries
    table. This is the only place a per-wicket bowler attribution actually
    exists: player_innings (role='bowling') only stores an aggregate
    wickets_taken COUNT, with no per-dismissal detail or bowler_id-to-type
    mapping. Filtering to `WHERE dismissal_type IS NOT NULL` keeps this to
    roughly one row per wicket (a tiny fraction of all deliveries), so it's
    an efficient stand-in for a "dismissals table" wherever a bowler-side
    dismissal-type breakdown is needed (e.g. the population comparison on
    the Bowling tab), without pulling every ball ever bowled.
    """
    sql = """
        SELECT match_id, innings_id, bowler_id, batter_id, dismissed_player_id, dismissal_type
        FROM deliveries
        WHERE dismissal_type IS NOT NULL
    """
    df = _query_with_retry(sql, ttl=0)
    return _stringify_uuids(df)


@st.cache_data(ttl=300)
def get_highlights_for_bowlers(bowler_ids: tuple[str, ...], max_per_bowler: int = 10):
    """
    Fetch up to `max_per_bowler` highlights per bowler for the provided set
    of bowlers, newest first within each bowler.
    """
    if not bowler_ids:
        return pd.DataFrame()

    params = {"max_per_bowler": max_per_bowler}
    placeholders = []

    for i, bid in enumerate(bowler_ids):
        key = f"bid_{i}"
        placeholders.append(f":{key}")
        params[key] = bid

    in_sql = ", ".join(placeholders)

    sql = f"""
        WITH ranked AS (
            SELECT
                h.*,
                ROW_NUMBER() OVER (
                    PARTITION BY h.bowler_id
                    ORDER BY h.match_id DESC, h.innings_number, h.over, h.ball_number
                ) AS rn
            FROM highlights h
            WHERE h.bowler_id IN ({in_sql})
        )
        SELECT *
        FROM ranked
        WHERE rn <= :max_per_bowler
        ORDER BY bowler_id, rn
    """

    df = _query_with_retry(sql, params=params, ttl=0)
    return _stringify_uuids(df)


@st.cache_data(ttl=300)
def get_highlights_for_batters(batter_ids: tuple[str, ...], max_per_batter: int = 10):
    """
    Fetch up to `max_per_batter` highlights per batter for the provided set
    of batters, newest first within each batter -- mirrors
    get_highlights_for_bowlers() exactly, just partitioned by batter_id.
    Powers the Batter Style tab.
    """
    if not batter_ids:
        return pd.DataFrame()

    params = {"max_per_batter": max_per_batter}
    placeholders = []

    for i, bid in enumerate(batter_ids):
        key = f"bid_{i}"
        placeholders.append(f":{key}")
        params[key] = bid

    in_sql = ", ".join(placeholders)

    sql = f"""
        WITH ranked AS (
            SELECT
                h.*,
                ROW_NUMBER() OVER (
                    PARTITION BY h.batter_id
                    ORDER BY h.match_id DESC, h.innings_number, h.over, h.ball_number
                ) AS rn
            FROM highlights h
            WHERE h.batter_id IN ({in_sql})
        )
        SELECT *
        FROM ranked
        WHERE rn <= :max_per_batter
        ORDER BY batter_id, rn
    """

    df = _query_with_retry(sql, params=params, ttl=0)
    return _stringify_uuids(df)


@st.cache_data(ttl=300)
def get_player_style():
    """Full player_style table (batter_hand, pace_spin, bowl_hand, bowl_style)."""
    return fetch_all_rows("player_style", "*")
