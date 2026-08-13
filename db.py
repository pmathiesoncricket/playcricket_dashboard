import streamlit as st
import pandas as pd
import uuid
import time
from sqlalchemy.exc import OperationalError, TimeoutError as SATimeoutError

# ---------- Setup ----------

conn = st.connection("postgresql", type="sql")

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
    for col in df.columns:
        non_null = df[col].dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], uuid.UUID):
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, uuid.UUID) else x)
    return df


def fetch_all_rows(table_name: str, select_str: str, eq_filters: dict | None = None,
                    order_col: str | None = None, page_size: int = 1000,
                    id_col: str | None = None) -> pd.DataFrame:
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
    Full, unscoped batting player_innings history joined to matches/style.
    Kept for any caller that genuinely needs the whole dataset, but this
    always pulls EVERY batting innings ever recorded regardless of any
    Grade/Season/Team filter -- for a club with a few years of history,
    that's a slow, unbounded query. The Batting tab should use
    get_batting_innings_for_matches() below instead, which filters at the
    SQL level to just the matches actually in scope.
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
def get_batting_innings_for_matches(match_ids: tuple[str, ...]):
    """
    Same shape/joins/columns as get_batting_innings() (including
    opponent_team), but filtered to a specific set of match_ids at the SQL
    level via WHERE match_id IN (...), instead of fetching every batting
    innings ever recorded and filtering in pandas afterward. This is what
    the Batting tab now uses once Grade/Match type/Season have narrowed
    down the match set, so the query cost scales with how much you've
    actually filtered rather than always being "the whole club's history."
    """
    if not match_ids:
        return pd.DataFrame()

    params = {}
    placeholders = []
    for i, mid in enumerate(match_ids):
        key = f"mid_{i}"
        placeholders.append(f":{key}")
        params[key] = mid
    in_sql = ", ".join(placeholders)

    pages = []
    start = 0
    page_size = 2000
    while True:
        params["limit"] = page_size
        params["offset"] = start
        sql = f"""
            SELECT * FROM player_innings
            WHERE role = 'batting' AND match_id IN ({in_sql})
            ORDER BY match_id
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
    pi_df = _stringify_uuids(pd.concat(pages, ignore_index=True))

    m_df = get_matches()
    if m_df.empty:
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
def get_deliveries_for_matches(match_ids: tuple[str, ...]):
    """
    Bulk fetch of every raw delivery row for a set of matches. Still used
    where genuine ball-by-ball detail is needed (Match Summary aggregate
    view). For the Batting tab's per-batter summary table, prefer
    get_batting_deliveries_summary() below -- it does the same job with a
    fraction of the rows/columns crossing the network, since the
    aggregation happens in Postgres instead of pandas.
    """
    if not match_ids:
        return pd.DataFrame()

    pages = []
    start = 0
    page_size = 5000
    params = {}
    placeholders = []
    for i, mid in enumerate(match_ids):
        key = f"mid_{i}"
        placeholders.append(f":{key}")
        params[key] = mid
    in_sql = ", ".join(placeholders)

    while True:
        params["limit"] = page_size
        params["offset"] = start
        sql = f"""
            SELECT * FROM deliveries
            WHERE match_id IN ({in_sql})
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
def get_batting_deliveries_summary(match_ids: tuple[str, ...]):
    """
    Server-side pre-aggregation of deliveries for the Batting tab's
    per-batter summary table. Grouped by (match_id, innings_id,
    batting_team_id, batter_id, bowler_pace_spin, bowler_bowl_style) so
    Postgres does the GROUP BY (using the existing indexes on
    deliveries.batter_id / match_id / bowler_id) and only the compact
    aggregated rows -- one per unique combination, not one per ball --
    cross the network.
    """
    if not match_ids:
        return pd.DataFrame()

    params = {}
    placeholders = []
    for i, mid in enumerate(match_ids):
        key = f"mid_{i}"
        placeholders.append(f":{key}")
        params[key] = mid
    in_sql = ", ".join(placeholders)

    sql = f"""
        SELECT
            d.match_id,
            d.innings_id,
            d.batting_team_id,
            d.batter_id,
            MAX(d.batter) AS batter_name,
            ps.pace_spin AS bowler_pace_spin,
            ps.bowl_style AS bowler_bowl_style,
            SUM(d.batter_runs) AS runs,
            COUNT(*) FILTER (WHERE d.wides = 0) AS legal_balls,
            COUNT(*) FILTER (WHERE d.batter_runs = 4) AS fours,
            COUNT(*) FILTER (WHERE d.batter_runs = 6) AS sixes,
            COUNT(*) FILTER (
                WHERE d.dismissal_type IS NOT NULL AND d.dismissed_player_id = d.batter_id
            ) AS dismissals
        FROM deliveries d
        LEFT JOIN player_style ps ON d.bowler_id = ps.player_id
        WHERE d.match_id IN ({in_sql}) AND d.batter_id IS NOT NULL
        GROUP BY
            d.match_id, d.innings_id, d.batting_team_id, d.batter_id,
            ps.pace_spin, ps.bowl_style
    """
    df = _query_with_retry(sql, params=params, ttl=0)
    return _stringify_uuids(df)


@st.cache_data(ttl=300)
def get_highlights():
    """Highlight clips (fours, sixes, dismissals, etc.) with video URLs."""
    return fetch_all_rows("highlights", "*")


@st.cache_data(ttl=300)
def get_bowler_summary():
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
    Up to `max_per_bowler` highlights per bowler, spread across as many
    DIFFERENT MATCHES as possible.
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
        WITH per_match_ranked AS (
            SELECT
                h.highlight_id, h.match_id, h.innings_id, h.innings_number, h.innings_order,
                h.ball_id, h.over, h.ball_number, h.batter_id, h.batter, h.bowler_id, h.bowler,
                h.highlight_type, h.metrics, h.description, h.highlight_url,
                h.created_at, h.updated_at,
                ROW_NUMBER() OVER (
                    PARTITION BY h.bowler_id, h.match_id
                    ORDER BY h.innings_number, h.over, h.ball_number
                ) AS rank_within_match,
                DENSE_RANK() OVER (
                    PARTITION BY h.bowler_id
                    ORDER BY m.day_1_start DESC NULLS LAST, h.match_id DESC
                ) AS match_recency_rank
            FROM highlights h
            LEFT JOIN matches m ON m.match_id = h.match_id
            WHERE h.bowler_id IN ({in_sql})
        ),
        final_ranked AS (
            SELECT
                highlight_id, match_id, innings_id, innings_number, innings_order,
                ball_id, over, ball_number, batter_id, batter, bowler_id, bowler,
                highlight_type, metrics, description, highlight_url, created_at, updated_at,
                ROW_NUMBER() OVER (
                    PARTITION BY bowler_id
                    ORDER BY rank_within_match ASC, match_recency_rank ASC
                ) AS overall_rank
            FROM per_match_ranked
        )
        SELECT
            highlight_id, match_id, innings_id, innings_number, innings_order,
            ball_id, over, ball_number, batter_id, batter, bowler_id, bowler,
            highlight_type, metrics, description, highlight_url, created_at, updated_at
        FROM final_ranked
        WHERE overall_rank <= :max_per_bowler
        ORDER BY bowler_id, overall_rank
    """

    df = _query_with_retry(sql, params=params, ttl=0)
    return _stringify_uuids(df)


@st.cache_data(ttl=300)
def get_highlights_for_batters(batter_ids: tuple[str, ...], max_per_batter: int = 10):
    """
    Up to `max_per_batter` highlights per batter, spread across as many
    DIFFERENT MATCHES as possible.
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
        WITH per_match_ranked AS (
            SELECT
                h.highlight_id, h.match_id, h.innings_id, h.innings_number, h.innings_order,
                h.ball_id, h.over, h.ball_number, h.batter_id, h.batter, h.bowler_id, h.bowler,
                h.highlight_type, h.metrics, h.description, h.highlight_url,
                h.created_at, h.updated_at,
                ROW_NUMBER() OVER (
                    PARTITION BY h.batter_id, h.match_id
                    ORDER BY h.innings_number, h.over, h.ball_number
                ) AS rank_within_match,
                DENSE_RANK() OVER (
                    PARTITION BY h.batter_id
                    ORDER BY m.day_1_start DESC NULLS LAST, h.match_id DESC
                ) AS match_recency_rank
            FROM highlights h
            LEFT JOIN matches m ON m.match_id = h.match_id
            WHERE h.batter_id IN ({in_sql})
        ),
        final_ranked AS (
            SELECT
                highlight_id, match_id, innings_id, innings_number, innings_order,
                ball_id, over, ball_number, batter_id, batter, bowler_id, bowler,
                highlight_type, metrics, description, highlight_url, created_at, updated_at,
                ROW_NUMBER() OVER (
                    PARTITION BY batter_id
                    ORDER BY rank_within_match ASC, match_recency_rank ASC
                ) AS overall_rank
            FROM per_match_ranked
        )
        SELECT
            highlight_id, match_id, innings_id, innings_number, innings_order,
            ball_id, over, ball_number, batter_id, batter, bowler_id, bowler,
            highlight_type, metrics, description, highlight_url, created_at, updated_at
        FROM final_ranked
        WHERE overall_rank <= :max_per_batter
        ORDER BY batter_id, overall_rank
    """

    df = _query_with_retry(sql, params=params, ttl=0)
    return _stringify_uuids(df)


@st.cache_data(ttl=300)
def get_player_style():
    """Full player_style table (batter_hand, pace_spin, bowl_hand, bowl_style)."""
    return fetch_all_rows("player_style", "*")
