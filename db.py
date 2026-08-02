import streamlit as st
import pandas as pd
import uuid


# ---------- Setup ----------

conn = st.connection("postgresql", type="sql")


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
      Required for large tables (e.g. deliveries) — OFFSET has to scan and
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

            df_page = conn.query(sql, params=params, ttl=0)
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

            df_page = conn.query(sql, params=params, ttl=0)
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
        "match_id, grade, matchtype, day1start, hometeamid, hometeam, awayteamid, awayteam, organisation_id, competition_id, competition_name",
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
        df_page = conn.query(
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
def get_highlights():
    """Highlight clips (fours, sixes, dismissals, etc.) with video URLs."""
    return fetch_all_rows("highlights", "*")


@st.cache_data(ttl=300)
def get_bowler_summary():
    """
    One row per bowler with total balls bowled and the set of grades they
    appear in — computed entirely in Postgres via GROUP BY, so only a
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
    df = conn.query(sql, ttl=0)
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

    df = conn.query(sql, params=params, ttl=0)
    return _stringify_uuids(df)

@st.cache_data(ttl=300)
def get_player_style():
    """Full player_style table (batter_hand, pace_spin, bowl_hand, bowl_style)."""
    return fetch_all_rows("player_style", "*")
