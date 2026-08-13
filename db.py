@st.cache_data(ttl=300)
def get_wicketkeepers_for_matches(match_ids: tuple[str, ...]):
    """One row per player_innings record with wicketkeeper = true, for a
    given match set -- powers Team Preview's Lineup module keeper tag."""
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
        SELECT DISTINCT match_id, team_id, player_id
        FROM player_innings
        WHERE wicketkeeper = true AND match_id IN ({in_sql})
    """
    df = _query_with_retry(sql, params=params, ttl=0)
    return _stringify_uuids(df)


@st.cache_data(ttl=300)
def get_match_results():
    """
    Parses matches.result_text into a structured winner_team_id +
    outcome_type per match. result_text consistently follows the pattern
    "<winning team name> won by <margin>" (also seen: "won by forfeit",
    "won" with no margin, "Won on disqualification"), plus special cases
    "Match drawn", "Match tied", "Result pending", and "<team> trails by
    <N>" (an unfinished/abandoned two-day match sitting mid-way, treated
    as no_result since it isn't a concluded outcome).

    outcome_type is one of:
      'decisive'  -- winner_team_id is populated
      'draw', 'tie', 'no_result' -- winner_team_id is None
      'unparsed'  -- the winning team name in result_text didn't match
                     either home_team or away_team (name mismatch/typo);
                     winner_team_id is None. Kept as its own bucket rather
                     than silently lumped into no_result, so it can be
                     surfaced/audited if it ever shows up in volume.
    """
    m = get_matches()[["match_id", "home_team_id", "home_team", "away_team_id", "away_team", "result_text"]].copy()
    if m.empty:
        return pd.DataFrame(columns=["match_id", "outcome_type", "winner_team_id"])

    def resolve(row):
        text = (row["result_text"] or "").strip()
        low = text.lower()
        if not text:
            return pd.Series(["no_result", None])
        if low.startswith("match drawn"):
            return pd.Series(["draw", None])
        if low.startswith("match tied"):
            return pd.Series(["tie", None])
        if "result pending" in low:
            return pd.Series(["no_result", None])
        if "trails by" in low:
            return pd.Series(["no_result", None])

        idx = low.find(" won")
        if idx == -1:
            return pd.Series(["unparsed", None])
        prefix = text[:idx].strip().lower()
        home = (row["home_team"] or "").strip().lower()
        away = (row["away_team"] or "").strip().lower()

        if prefix == home:
            return pd.Series(["decisive", row["home_team_id"]])
        if prefix == away:
            return pd.Series(["decisive", row["away_team_id"]])
        if home and (prefix in home or home in prefix):
            return pd.Series(["decisive", row["home_team_id"]])
        if away and (prefix in away or away in prefix):
            return pd.Series(["decisive", row["away_team_id"]])
        return pd.Series(["unparsed", None])

    m[["outcome_type", "winner_team_id"]] = m.apply(resolve, axis=1)
    return _stringify_uuids(m[["match_id", "outcome_type", "winner_team_id"]])


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
            sql, params={"batter_id": batter_id, "limit": page_size, "offset": start}, ttl=0,
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
            sql, params={"bowler_id": bowler_id, "limit": page_size, "offset": start}, ttl=0,
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
            sql, params={"match_id": match_id, "limit": page_size, "offset": start}, ttl=0,
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
    """Bulk fetch of every raw delivery row for a set of matches."""
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
    """Server-side pre-aggregation of deliveries for the Batting tab's
    per-batter summary table (one row per match/innings/batting-team/
    batter/bowler-style combo)."""
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
    return fetch_all_rows("player_style", "*")
