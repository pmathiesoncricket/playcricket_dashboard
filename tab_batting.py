import streamlit as st
import pandas as pd
import plotly.express as px
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from db import (
    get_batting_innings, get_batting_deliveries_summary, get_deliveries_for_batter,
    get_highlights, get_matches, get_player_style,
)
from helpers import add_season_column, cascading_multiselect, segment_label, SEGMENT_ORDER, MAROON, MAROON_SHADES

MAX_STREAM_GAP_HOURS = 18
NBSP = "\u00A0"
UNKNOWN_BOWL_LABEL = "Unknown / unrecorded"

# Safety cap: get_batting_deliveries_summary() runs a GROUP BY over every
# delivery in every match still in scope. Left unfiltered (e.g. no Grade or
# Season selected), that can mean the WHOLE deliveries table, which can be
# slow enough to hold a database connection open long enough to exhaust the
# shared connection pool and time out unrelated queries for every user of
# the app -- this happened in practice. Narrowing Grade/Season/Match type
# below this many matches keeps that query fast and bounded.
MAX_SCOPED_MATCHES_FOR_SUMMARY = 400


def build_timestamped_youtube_url(base_url, seconds):
    """Build a YouTube URL timestamped to `seconds` into the stream, preserving
    existing query params (e.g. si=) and supporting both /watch and /live paths."""
    if not base_url or pd.isna(seconds):
        return None
    try:
        seconds = max(int(round(seconds)), 0)
    except (TypeError, ValueError):
        return None

    parsed = urlparse(base_url)
    query = parse_qs(parsed.query)
    query["t"] = [f"{seconds}s"]
    new_query = urlencode(query, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", new_query, ""))


def resolve_ball_video_url(ball_time, day1_url, day1_start, day2_url, day2_start, offset_adjustment=-5):
    """
    Picks day1 or day2 stream based on how close the ball's timestamp is to
    each stream's start time (all values are UTC). A ball is considered to
    belong to a stream if it falls within MAX_STREAM_GAP_HOURS of that
    stream's start. Day 1 is checked first; day 2 is only used if day 1
    doesn't match. If neither stream is within range, falls back to day 1
    (if a day 1 stream exists at all). `offset_adjustment` is an extra
    seconds nudge (positive or negative) to compensate for stream-specific
    lag between the actual ball time and the video's timestamp. Returns
    None if no stream is usable.
    """
    if pd.isna(ball_time):
        return None
    ball_time_ts = pd.to_datetime(ball_time, utc=True)

    max_gap = pd.Timedelta(hours=MAX_STREAM_GAP_HOURS)

    def within_window(stream_start):
        if pd.isna(stream_start):
            return False
        start_ts = pd.to_datetime(stream_start, utc=True)
        return abs(ball_time_ts - start_ts) <= max_gap

    day1_start_ts = pd.to_datetime(day1_start, utc=True) if pd.notna(day1_start) else None
    day2_start_ts = pd.to_datetime(day2_start, utc=True) if pd.notna(day2_start) else None

    if day1_url and within_window(day1_start_ts):
        stream_url, stream_start_ts = day1_url, day1_start_ts
    elif day2_url and within_window(day2_start_ts):
        stream_url, stream_start_ts = day2_url, day2_start_ts
    elif day1_url and day1_start_ts is not None:
        stream_url, stream_start_ts = day1_url, day1_start_ts
    elif day2_url and day2_start_ts is not None:
        stream_url, stream_start_ts = day2_url, day2_start_ts
    else:
        return None

    offset_seconds = (ball_time_ts - stream_start_ts).total_seconds() + offset_adjustment
    return build_timestamped_youtube_url(stream_url, offset_seconds)


def _pad(text, width):
    """Left-align/truncate `text` to a fixed character width using
    non-breaking spaces for padding, so it stays visually aligned inside a
    monospace-rendered markdown code span (regular spaces collapse in HTML)."""
    text = "" if text is None else str(text)
    if len(text) > width:
        return text[: max(width - 1, 0)] + "\u2026"
    return text + NBSP * (width - len(text))


def _apply_bowler_style_filter(df, column, selected_values):
    """
    Filters rows by the ACTUAL bowler's own pace_spin/bowl_style (already
    joined onto whichever dataframe is passed in via bowler_id) -- i.e.
    who the batter faced, not the batter's own bowling ability.
    UNKNOWN_BOWL_LABEL is a selectable bucket covering any row where the
    bowler has no recorded style, so incomplete style data doesn't
    silently drop rows when "everything" is selected (the default).
    """
    if not selected_values:
        return df
    mask = df[column].isin(selected_values)
    if UNKNOWN_BOWL_LABEL in selected_values:
        mask = mask | df[column].isna()
    return df[mask]


def _summary_from_aggregated(d):
    """
    Per-batter Batting summary figures, built from the PRE-AGGREGATED
    get_batting_deliveries_summary() result (one row per match/innings/
    batting-team/batter/bowler-style combo) rather than raw ball-by-ball
    data -- Postgres already did the SUM/COUNT server-side, this just
    rolls those rows up to one row per batter. Runs/fours/sixes/dismissals
    all originate from batter_runs / dismissed_player_id in that query
    (never team_runs).
    """
    columns = ["player_id", "player_name", "innings", "total_runs", "total_balls", "dismissals", "fours", "sixes"]
    if d.empty:
        return pd.DataFrame(columns=columns)

    innings_counts = (
        d[["batter_id", "match_id", "innings_id"]].drop_duplicates()
        .groupby("batter_id").size().reset_index(name="innings")
    )
    totals = d.groupby("batter_id").agg(
        player_name=("batter_name", "first"),
        total_runs=("runs", "sum"),
        total_balls=("legal_balls", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        dismissals=("dismissals", "sum"),
    ).reset_index()

    g = totals.merge(innings_counts, on="batter_id", how="left")
    g["innings"] = g["innings"].fillna(0).astype(int)
    return g.rename(columns={"batter_id": "player_id"})[columns]


def batting_tab():
    st.header("Batting")

    matches_df = get_matches()
    if matches_df.empty:
        st.info("No match data available.")
        return
    matches_df = add_season_column(matches_df, "day_1_start")

    st.sidebar.markdown("### Batting filters")
    st.sidebar.caption(
        "Grade, Match type, Season, Team, and Opponent stay interactive as you narrow them. "
        "Bowling type/style and the batting summary itself only load once you click "
        "**Apply Filters** below, so a query doesn't fire on every small tweak."
    )

    # ---------------------------------------------------------------
    # Grade / Match type / Season -- applied at the match_id level, using
    # only the lightweight `matches` table, so this stays instantly
    # interactive (and correctly cascading -- e.g. Season options only
    # show seasons that grade was actually played in) without touching
    # anything expensive.
    # ---------------------------------------------------------------
    stage_matches = matches_df.copy()

    grade_options = sorted(stage_matches["grade"].dropna().unique().tolist())
    selected_grade = cascading_multiselect(
        st.sidebar, "Grade", grade_options, "filter_grade", enable_quick_add=True
    )
    if selected_grade:
        stage_matches = stage_matches[stage_matches["grade"].isin(selected_grade)]

    match_type_options = sorted(stage_matches["match_type"].dropna().unique().tolist())
    selected_match_type = cascading_multiselect(
        st.sidebar, "Match type", match_type_options, "filter_match_type",
        default_options=match_type_options,
    )
    if selected_match_type:
        stage_matches = stage_matches[stage_matches["match_type"].isin(selected_match_type)]

    season_options = (
        stage_matches["season"].dropna().drop_duplicates().sort_values(ascending=False).tolist()
    )
    selected_season = cascading_multiselect(st.sidebar, "Season (July\u2013June)", season_options, "filter_season")
    if selected_season:
        stage_matches = stage_matches[stage_matches["season"].isin(selected_season)]

    if stage_matches.empty:
        st.sidebar.warning("No matches match Grade/Match type/Season.")
        return

    valid_match_ids = set(stage_matches["match_id"])

    # ---------------------------------------------------------------
    # Team / Opponent -- also stay live: player_innings is already cached
    # by @st.cache_data, so re-slicing it as Team/Opponent change costs
    # nothing extra beyond ordinary pandas filtering.
    # ---------------------------------------------------------------
    batting_df = get_batting_innings()
    if batting_df.empty:
        st.info("No batting data available.")
        return
    batting_df = add_season_column(batting_df, "day_1_start")

    stage_df = batting_df[batting_df["match_id"].isin(valid_match_ids)].copy()
    if stage_df.empty:
        st.sidebar.warning("No batting records for the current Grade/Match type/Season.")
        return

    team_options = sorted(stage_df["team"].dropna().unique().tolist())
    selected_team = cascading_multiselect(
        st.sidebar, "Team (batting team)", team_options, "filter_team", enable_quick_add=True
    )
    if selected_team:
        stage_df = stage_df[stage_df["team"].isin(selected_team)]

    if stage_df.empty:
        st.sidebar.warning("No batting records match the current filters.")
        return

    opponent_options = sorted(stage_df["opponent_team"].dropna().unique().tolist())
    selected_opponent = cascading_multiselect(
        st.sidebar, "Opponent (bowling team)", opponent_options, "filter_opponent"
    )
    if selected_opponent:
        stage_df = stage_df[stage_df["opponent_team"].isin(selected_opponent)]

    if stage_df.empty:
        st.sidebar.warning("No batting records match the current filters.")
        return

    scoped_match_ids = tuple(str(m) for m in stage_df["match_id"].unique())

    # ---------------------------------------------------------------
    # Apply Filters gate -- everything from here on (the ball-by-ball
    # aggregate query, Bowling type/style, and the whole summary/detail
    # page) only runs once the CURRENT Grade/Match type/Season/Team/
    # Opponent combination has been explicitly applied. Changing any of
    # those five afterward reverts the page back to "click Apply Filters"
    # until you do, so it can never silently show stale results.
    # ---------------------------------------------------------------
    current_filter_key = (
        tuple(sorted(selected_grade)), tuple(sorted(selected_match_type)), tuple(sorted(selected_season)),
        tuple(sorted(selected_team)), tuple(sorted(selected_opponent)),
    )

    apply_clicked = st.sidebar.button("Apply Filters", type="primary", key="batting_apply_filters")
    if apply_clicked:
        st.session_state["batting_applied_key"] = current_filter_key

    if st.session_state.get("batting_applied_key") != current_filter_key:
        st.info(
            "Set your Grade / Match type / Season / Team / Opponent filters in the sidebar, "
            "then click **Apply Filters** to load the batting summary."
        )
        return

    if len(scoped_match_ids) > MAX_SCOPED_MATCHES_FOR_SUMMARY:
        st.warning(
            f"{len(scoped_match_ids)} matches match these filters, which is over the "
            f"{MAX_SCOPED_MATCHES_FOR_SUMMARY}-match limit for a single batting summary run. "
            f"That's the point at which the underlying ball-by-ball query risks tying up the "
            f"shared database connection for everyone using the app, so it's blocked outright "
            f"rather than just being slow. Narrow Grade, Season, or Team above and click "
            f"Apply Filters again."
        )
        return

    # ---------------------------------------------------------------
    # Bowling type / Bowling style -- applied to deliveries.bowler_id's
    # OWN pace_spin/bowl_style (who the batter actually faced), not the
    # batter's own bowling ability. Uses the pre-aggregated, server-side
    # summary (get_batting_deliveries_summary) rather than pulling raw
    # ball-by-ball rows for every match in scope -- Postgres does the
    # SUM/COUNT and only sends back one row per (match, innings, batting
    # team, batter, bowler-style) combination, not one row per ball. Once
    # fetched (gated above), toggling these two filters is instant --
    # they're just filtering this already-in-memory result.
    # ---------------------------------------------------------------
    agg_scope = get_batting_deliveries_summary(scoped_match_ids)

    if not agg_scope.empty:
        valid_team_keys = stage_df[["match_id", "team_id"]].drop_duplicates()
        agg_scope = agg_scope.merge(
            valid_team_keys,
            left_on=["match_id", "batting_team_id"],
            right_on=["match_id", "team_id"],
            how="inner",
        )
    else:
        agg_scope["bowler_pace_spin"] = pd.Series(dtype="object")
        agg_scope["bowler_bowl_style"] = pd.Series(dtype="object")

    if agg_scope.empty:
        st.warning("No ball-by-ball data available for the current filters.")
        return

    bowling_type_options = (
        sorted(agg_scope["bowler_pace_spin"].dropna().unique().tolist()) + [UNKNOWN_BOWL_LABEL]
    )
    selected_bowling_type = cascading_multiselect(
        st.sidebar, "Bowling type (pace/spin)", bowling_type_options, "filter_bowling_type",
        default_options=bowling_type_options,
    )
    agg_after_type = _apply_bowler_style_filter(agg_scope, "bowler_pace_spin", selected_bowling_type)

    bowl_style_options = (
        sorted(agg_after_type["bowler_bowl_style"].dropna().unique().tolist()) + [UNKNOWN_BOWL_LABEL]
    )
    selected_bowl_style = cascading_multiselect(
        st.sidebar, "Bowling style", bowl_style_options, "filter_bowl_style",
        default_options=bowl_style_options,
    )
    agg_filtered = _apply_bowler_style_filter(agg_after_type, "bowler_bowl_style", selected_bowl_style)

    if agg_filtered.empty:
        st.warning("No ball-by-ball data matches the current filters.")
        return

    grouped_all = _summary_from_aggregated(agg_filtered)
    batter_options = sorted(grouped_all["player_name"].dropna().tolist())

    if "pending_batter_filter" in st.session_state:
        st.session_state["filter_batter"] = st.session_state.pop("pending_batter_filter")

    if "filter_batter" in st.session_state and st.session_state["filter_batter"] not in (
        ["All batters"] + batter_options
    ):
        st.session_state["filter_batter"] = "All batters"

    # Population base for the innings-grain sections further down
    # (Distribution of innings runs, Dismissal distribution vs population,
    # Boundary rate vs population, Metrics by batting position). These
    # respect Grade/Match type/Season/Team/Opponent but NOT the ball-level
    # Bowling type/style filter -- an innings doesn't cleanly subset by
    # which bowler type was faced within it, so they're left at the
    # whole-innings player_innings grain rather than rebuilt from deliveries.
    population_df = stage_df

    st.subheader("Batting summary (filtered)")
    selected_batter_name = st.selectbox(
        "Batter (applies to whole page) \u2014 or click a row below",
        options=["All batters"] + batter_options,
        index=0,
        key="filter_batter",
    )

    selected_batter_id = None
    filtered_agg = agg_filtered
    filtered = stage_df
    if selected_batter_name != "All batters":
        batter_row = grouped_all[grouped_all["player_name"] == selected_batter_name].iloc[0]
        selected_batter_id = batter_row["player_id"]
        filtered_agg = agg_filtered[agg_filtered["batter_id"] == str(selected_batter_id)]
        filtered = stage_df[stage_df["player_id"] == selected_batter_id]

    if filtered_agg.empty:
        st.warning("No batting records match the current filters.")
        return

    grouped = _summary_from_aggregated(filtered_agg)

    grouped["average"] = grouped.apply(
        lambda row: row["total_runs"] / row["dismissals"] if row["dismissals"] > 0 else None,
        axis=1,
    )
    grouped["strike_rate"] = grouped.apply(
        lambda row: 100 * row["total_runs"] / row["total_balls"] if row["total_balls"] > 0 else None,
        axis=1,
    )
    grouped["BPD"] = grouped.apply(
        lambda row: row["total_balls"] / row["dismissals"] if row["dismissals"] > 0 else None,
        axis=1,
    )

    display_df = grouped.copy()
    display_df["average"] = display_df["average"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "\u2013"
    )
    display_df["strike_rate"] = display_df["strike_rate"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "\u2013"
    )
    display_df["BPD"] = display_df["BPD"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "\u2013"
    )

    summary_table = display_df[
        ["player_name", "innings", "total_runs", "average", "strike_rate", "BPD", "fours", "sixes"]
    ].sort_values("total_runs", ascending=False).reset_index(drop=True)

    # Table height is capped so a single-batter selection doesn't leave a
    # block of empty blank rows, but tall enough that most multi-grade
    # filters show everyone without scrolling. The caption states the true
    # count either way, so it's never ambiguous whether you're seeing
    # everyone who matches the filters.
    MAX_ROWS_VISIBLE = 25
    rows_to_show = min(len(summary_table), MAX_ROWS_VISIBLE)
    TABLE_HEIGHT = (rows_to_show + 1) * 35 + 3

    st.caption(
        f"{len(summary_table)} batter(s) match the current filters."
        + (" Scroll within the table below to see all of them." if len(summary_table) > MAX_ROWS_VISIBLE else "")
    )

    summary_event = st.dataframe(
        summary_table,
        width='stretch',
        height=TABLE_HEIGHT,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="batting_summary_table",
    )

    if summary_event.selection.rows:
        clicked_idx = summary_event.selection.rows[0]
        if clicked_idx < len(summary_table):
            clicked_name = summary_table.iloc[clicked_idx]["player_name"]
            if st.session_state.get("filter_batter") != clicked_name:
                st.session_state["pending_batter_filter"] = clicked_name
                st.rerun()
        else:
            # Stale selection from a since-shrunk table — ignore it
            pass

    if selected_batter_id is not None:
        selected_row = grouped[grouped["player_id"] == str(selected_batter_id)].iloc[0]

        st.subheader(f"Batter detail \u2014 {selected_row['player_name']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total runs", int(selected_row["total_runs"]))
        col2.metric(
            "Average",
            f"{selected_row['average']:.2f}" if pd.notna(selected_row["average"]) else "\u2013",
        )
        col3.metric(
            "Strike rate",
            f"{selected_row['strike_rate']:.0f}" if pd.notna(selected_row["strike_rate"]) else "\u2013",
        )

        # From here on, sections drill into ONE batter's ball-by-ball data
        # (video links, over/ball numbering, free-text descriptions) --
        # that's cheap to fetch directly for a single player, so we no
        # longer need the heavier match-wide aggregate used for the
        # summary table above.
        full_batter_deliveries = get_deliveries_for_batter(str(selected_batter_id))
        if not full_batter_deliveries.empty:
            full_batter_deliveries = full_batter_deliveries.copy()
            full_batter_deliveries["wides"] = full_batter_deliveries["wides"].fillna(0)
            full_batter_deliveries["is_legal_for_batter"] = full_batter_deliveries["wides"] == 0
            full_batter_deliveries["is_dismissal"] = (
                full_batter_deliveries["dismissal_type"].notna()
                & (full_batter_deliveries["dismissed_player_id"] == full_batter_deliveries["batter_id"])
            )
            full_batter_deliveries["is_four"] = full_batter_deliveries["batter_runs"] == 4
            full_batter_deliveries["is_six"] = full_batter_deliveries["batter_runs"] == 6

            bowler_style_lookup = get_player_style()[["player_id", "pace_spin", "bowl_style"]].rename(
                columns={"player_id": "bowler_id", "pace_spin": "bowler_pace_spin", "bowl_style": "bowler_bowl_style"}
            )
            full_batter_deliveries = full_batter_deliveries.merge(bowler_style_lookup, on="bowler_id", how="left")

            # Restrict to the current Grade/Match type/Season/Team/Opponent
            # scope (get_deliveries_for_batter returns this batter's whole
            # history, unfiltered).
            valid_team_keys = stage_df[["match_id", "team_id"]].drop_duplicates()
            full_batter_deliveries = full_batter_deliveries.merge(
                valid_team_keys,
                left_on=["match_id", "batting_team_id"],
                right_on=["match_id", "team_id"],
                how="inner",
            )

        st.subheader("Ball\u2011segment breakdown (deliveries)")

        if full_batter_deliveries.empty:
            st.info("No deliveries found for this batter.")
        else:
            seg_df = full_batter_deliveries.copy()
            seg_df["legal_ball"] = seg_df["is_legal_for_batter"]
            seg_df["ball_index"] = (
                seg_df.groupby("innings_id")["legal_ball"]
                .cumsum()
                .where(seg_df["legal_ball"], None)
            )

            seg_df = seg_df[seg_df["ball_index"].notna()]
            seg_df["ball_index"] = seg_df["ball_index"].astype(int)
            seg_df["segment"] = pd.Categorical(
                seg_df["ball_index"].map(segment_label),
                categories=SEGMENT_ORDER,
                ordered=True,
            )

            # NOW apply the sidebar's Bowling type/style filter -- ball
            # position numbering above reflects the batter's true innings
            # sequence; this just decides which of those (correctly
            # numbered) balls count toward each segment's totals.
            seg_df = _apply_bowler_style_filter(seg_df, "bowler_pace_spin", selected_bowling_type)
            seg_df = _apply_bowler_style_filter(seg_df, "bowler_bowl_style", selected_bowl_style)

            seg = seg_df.groupby("segment", observed=True).agg(
                balls=("ball_index", "count"),
                runs=("batter_runs", "sum"),
                dismissals=("is_dismissal", "sum"),
                fours=("is_four", "sum"),
                sixes=("is_six", "sum"),
            ).reset_index()

            seg["strike_rate"] = seg.apply(
                lambda row: 100 * row["runs"] / row["balls"] if row["balls"] > 0 else None,
                axis=1,
            )
            seg["BPD"] = seg.apply(
                lambda row: row["balls"] / row["dismissals"] if row["dismissals"] > 0 else None,
                axis=1,
            )

            seg["segment"] = pd.Categorical(seg["segment"], categories=SEGMENT_ORDER, ordered=True)
            seg = seg.sort_values("segment").reset_index(drop=True)

            seg_display = seg.copy()
            seg_display["strike_rate"] = seg_display["strike_rate"].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else "\u2013"
            )
            seg_display["BPD"] = seg_display["BPD"].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else "\u2013"
            )

            st.dataframe(seg_display, width='stretch')

            fig_seg = px.bar(
                seg,
                x="segment",
                y="strike_rate",
                category_orders={"segment": SEGMENT_ORDER},
                title=f"Strike rate by ball segment for {selected_row['player_name']}",
                color_discrete_sequence=[MAROON],
            )
            st.plotly_chart(fig_seg, width='stretch')

            st.subheader("Batting vs bowling style (deliveries)")

            # Deliberately the FULL (not sidebar-style-filtered) set, since
            # this table's whole purpose is comparing performance across
            # every style faced -- pre-filtering it by the sidebar's own
            # Bowling type/style selection would collapse it to one row.
            legal = full_batter_deliveries[full_batter_deliveries["is_legal_for_batter"]].copy()
            legal["bowl_style"] = legal["bowler_bowl_style"].fillna("Unknown")
            by_style = legal.groupby("bowl_style").agg(
                runs=("batter_runs", "sum"),
                balls=("bowler_id", "count"),
                dismissals=("is_dismissal", "sum"),
                fours=("is_four", "sum"),
                sixes=("is_six", "sum"),
            ).reset_index()

            by_style["average"] = by_style.apply(
                lambda r: r["runs"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1
            )
            by_style["strike_rate"] = by_style.apply(
                lambda r: 100 * r["runs"] / r["balls"] if r["balls"] > 0 else None, axis=1
            )
            by_style["BPD"] = by_style.apply(
                lambda r: r["balls"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1
            )
            by_style["boundaries"] = by_style["fours"] + by_style["sixes"]
            by_style["boundary_rate"] = by_style.apply(
                lambda r: r["balls"] / r["boundaries"] if r["boundaries"] > 0 else None, axis=1
            )

            by_style_display = by_style.copy()
            by_style_display["average"] = by_style_display["average"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "\u2013"
            )
            by_style_display["strike_rate"] = by_style_display["strike_rate"].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else "\u2013"
            )
            by_style_display["BPD"] = by_style_display["BPD"].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else "\u2013"
            )
            by_style_display["boundary_rate"] = by_style_display["boundary_rate"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "\u2013"
            )

            st.dataframe(
                by_style_display[
                    ["bowl_style", "runs", "balls", "average", "strike_rate", "BPD", "boundary_rate"]
                ].rename(columns={
                    "bowl_style": "Bowling style", "balls": "BF",
                    "strike_rate": "SR", "boundary_rate": "Balls/Boundary",
                }).sort_values("runs", ascending=False),
                width='stretch',
                hide_index=True,
            )

        st.subheader("Match-by-match batting")

        # Which (match, innings) show up is restricted to those where this
        # batter faced at least one delivery matching the current Bowling
        # type/style filter; the RUNS/BALLS/SR shown in each row's label
        # are the batter's TRUE full-innings figures from player_innings,
        # since bowler-style filtering doesn't cleanly subset "which
        # innings" -- only "which balls within it".
        qualifying_innings = (
            filtered_agg[["match_id", "innings_id"]].drop_duplicates()
            if selected_batter_id is not None else pd.DataFrame(columns=["match_id", "innings_id"])
        )

        match_df = filtered.merge(qualifying_innings, on=["match_id", "innings_id"], how="inner").copy()
        match_df["boundaries"] = match_df["fours"].fillna(0) + match_df["sixes"].fillna(0)
        match_df["match_name"] = match_df["match_type"].astype(str) + " v " + match_df["opponent_team"].astype(str)
        match_df["SR"] = match_df.apply(
            lambda r: 100 * r["runs"] / r["balls_faced"] if r["balls_faced"] and r["balls_faced"] > 0 else None,
            axis=1,
        )

        match_rows = match_df[
            ["match_id", "innings_id", "day_1_start", "match_name", "bat_position",
             "runs", "balls_faced", "SR", "boundaries"]
        ].copy()
        match_rows = match_rows.sort_values("day_1_start", ascending=False).reset_index(drop=True)

        stream_cols = ["match_id", "day1_stream_url", "day1_stream_start", "day2_stream_url", "day2_stream_start"]
        available_stream_cols = [c for c in stream_cols if c in matches_df.columns]
        matches_stream_df = (
            matches_df[available_stream_cols].copy()
            if available_stream_cols else pd.DataFrame(columns=stream_cols)
        )

        st.caption(f"{len(match_rows)} innings \u2014 expand a row to see ball-by-ball detail and video links")

        st.markdown(
            """
            <style>
            .streamlit-expanderHeader p code, div[data-testid="stExpander"] summary code {
                font-size: 1.05rem !important;
                letter-spacing: 0.03em;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for _, m_row in match_rows.iterrows():
            date_str = (
                pd.to_datetime(m_row["day_1_start"]).strftime("%d %b %Y")
                if pd.notna(m_row["day_1_start"]) else "Unknown date"
            )
            sr_str = f"{m_row['SR']:.0f}" if pd.notna(m_row["SR"]) else "\u2013"
            runs_str = f"{int(m_row['runs'])}" if pd.notna(m_row["runs"]) else "\u2013"
            bf_str = f"{int(m_row['balls_faced'])}" if pd.notna(m_row["balls_faced"]) else "\u2013"
            pos_str = f"Pos {m_row['bat_position']}" if pd.notna(m_row["bat_position"]) else "Pos \u2013"
            score_str = f"{runs_str} ({bf_str})"
            sr_display = f"SR {sr_str}"

            label = (
                f"`{_pad(date_str, 16)}{_pad(m_row['match_name'], 36)}{_pad(pos_str, 10)}"
                f"{_pad(score_str, 16)}{_pad(sr_display, 10)}`"
            )

            with st.expander(label):
                if full_batter_deliveries.empty:
                    st.info("No ball-by-ball data available for this innings.")
                    continue

                innings_balls = full_batter_deliveries[
                    (full_batter_deliveries["match_id"] == m_row["match_id"])
                    & (full_batter_deliveries["innings_id"] == m_row["innings_id"])
                ].copy()

                # Apply the sidebar Bowling type/style filter to the
                # ball-by-ball listing shown here.
                innings_balls = _apply_bowler_style_filter(innings_balls, "bowler_pace_spin", selected_bowling_type)
                innings_balls = _apply_bowler_style_filter(innings_balls, "bowler_bowl_style", selected_bowl_style)

                if innings_balls.empty:
                    st.info("No ball-by-ball data available for this innings (with current bowling filters).")
                    continue

                innings_balls = innings_balls.sort_values(["over", "ball_number"]).reset_index(drop=True)

                match_stream_row = matches_stream_df[matches_stream_df["match_id"] == m_row["match_id"]]
                day1_url = day1_start = day2_url = day2_start = None
                if not match_stream_row.empty:
                    srow = match_stream_row.iloc[0]
                    day1_url = srow.get("day1_stream_url")
                    day1_start = srow.get("day1_stream_start")
                    day2_url = srow.get("day2_stream_url")
                    day2_start = srow.get("day2_stream_start")

                offset_key = f"offset_{m_row['match_id']}_{m_row['innings_id']}"
                offset_seconds_adjustment = st.number_input(
                    "Video timestamp offset (seconds) for this innings' stream",
                    value=-5,
                    step=1,
                    help=(
                        "Fine-tune how far into the stream each ball's video link jumps. "
                        "Negative values start the clip slightly earlier. Different streams "
                        "can have different lag, so this is set per innings."
                    ),
                    key=offset_key,
                )

                innings_balls["video_url"] = innings_balls["ball_time"].apply(
                    lambda bt: resolve_ball_video_url(
                        bt, day1_url, day1_start, day2_url, day2_start, offset_seconds_adjustment
                    )
                )
                innings_balls["over_ball"] = (
                    innings_balls["over"].astype("Int64").astype(str) + "."
                    + innings_balls["ball_number"].astype("Int64").astype(str)
                )

                if day1_url is None and day2_url is None:
                    st.caption("No stream found for this match \u2014 video links unavailable.")

                ball_display = innings_balls[
                    ["over_ball", "bowler", "bowler_pace_spin", "bowler_bowl_style",
                     "batter_runs", "description", "video_url"]
                ].rename(columns={
                    "over_ball": "Over.Ball", "bowler": "Bowler",
                    "bowler_pace_spin": "Type", "bowler_bowl_style": "Style",
                    "batter_runs": "Runs", "description": "Description", "video_url": "Video",
                })

                st.dataframe(
                    ball_display,
                    hide_index=True,
                    width='stretch',
                    column_config={
                        "Video": st.column_config.LinkColumn("Video", display_text="\u25b6 Watch"),
                    },
                )

        st.subheader("Dismissal type distribution \u2014 player vs population")

        pop_innings = population_df

        # Retired / Retired Hurt / Retired Not Out excluded alongside the
        # not-a-real-dismissal types, since none of them reflect the
        # bowler/fielding side doing anything -- same reasoning as
        # excluding Did Not Bat / Not Out.
        excluded_types = {
            "Did Not Bat", "did not bat", "DNB",
            "Not Out", "not out",
            "Retired", "retired",
            "Retired Hurt", "retired hurt",
            "Retired Not Out", "retired not out",
        }
        pop_dismiss_df = pop_innings[
            pop_innings["dismissal_type"].notna()
            & ~pop_innings["dismissal_type"].isin(excluded_types)
        ]

        player_dismiss_df = pd.DataFrame()
        if selected_batter_id is not None:
            player_dismiss_df = filtered[
                filtered["dismissal_type"].notna()
                & ~filtered["dismissal_type"].isin(excluded_types)
            ]

        if pop_dismiss_df.empty or player_dismiss_df.empty or selected_batter_id is None:
            st.info("Not enough dismissal data to compare player vs population.")
        else:
            all_types = sorted(
                pd.concat(
                    [pop_dismiss_df["dismissal_type"], player_dismiss_df["dismissal_type"]],
                    ignore_index=True,
                )
                .dropna()
                .unique()
                .tolist()
            )

            pop_counts = pop_dismiss_df["dismissal_type"].value_counts().reindex(all_types, fill_value=0)
            player_counts = player_dismiss_df["dismissal_type"].value_counts().reindex(all_types, fill_value=0)

            pop_total = pop_counts.sum()
            player_total = player_counts.sum()

            pop_pct = (pop_counts / pop_total * 100) if pop_total > 0 else pop_counts
            player_pct = (player_counts / player_total * 100) if player_total > 0 else player_counts

            # Player columns/series come first -- both in the table and the
            # chart -- per the "player stats first, then population" ask.
            comp_df = pd.DataFrame({
                "dismissal_type": all_types,
                "Player %": player_pct.values,
                "Player count": player_counts.values,
                "Population %": pop_pct.values,
                "Population count": pop_counts.values,
            })

            comp_melt = comp_df.melt(
                id_vars=["dismissal_type"],
                value_vars=["Player %", "Population %"],
                var_name="group",
                value_name="percentage",
            )

            fig_comp = px.bar(
                comp_melt,
                x="dismissal_type",
                y="percentage",
                color="group",
                barmode="group",
                category_orders={"group": ["Player %", "Population %"]},
                title="Dismissal type % \u2014 player vs population",
                color_discrete_sequence=MAROON_SHADES[:2],
            )
            st.plotly_chart(fig_comp, width='stretch')

            st.dataframe(comp_df, width='stretch')

        st.subheader("Distribution of innings scores (runs)")
        fig_runs = px.histogram(
            filtered,
            x="runs",
            nbins=20,
            title="Distribution of individual innings runs",
            color_discrete_sequence=[MAROON],
            text_auto=True,
        )
        fig_runs.update_layout(bargap=0.15)
        st.plotly_chart(fig_runs, width='stretch')

        st.subheader("Boundary rate vs population")

        pop_boundary_df = population_df

        pop_boundary = pop_boundary_df.groupby("player_id").agg(
            player_name=("player_name", "first"),
            runs=("runs", "sum"),
            balls=("balls_faced", "sum"),
            fours=("fours", "sum"),
            sixes=("sixes", "sum"),
        ).reset_index()

        pop_boundary["fours_per_100_balls"] = pop_boundary.apply(
            lambda r: 100 * r["fours"] / r["balls"] if r["balls"] > 0 else None,
            axis=1,
        )
        pop_boundary["sixes_per_100_balls"] = pop_boundary.apply(
            lambda r: 100 * r["sixes"] / r["balls"] if r["balls"] > 0 else None,
            axis=1,
        )

        pcol1, pcol2 = st.columns(2)
        pcol1.metric(
            "Population avg fours/100 balls",
            f"{pop_boundary['fours_per_100_balls'].mean():.2f}"
            if pop_boundary["fours_per_100_balls"].notna().any()
            else "\u2013",
        )
        pcol2.metric(
            "Population avg sixes/100 balls",
            f"{pop_boundary['sixes_per_100_balls'].mean():.2f}"
            if pop_boundary["sixes_per_100_balls"].notna().any()
            else "\u2013",
        )

        if selected_batter_id is not None:
            batter_boundary_row = pop_boundary[pop_boundary["player_id"] == selected_batter_id].iloc[0]

            bcol1, bcol2 = st.columns(2)
            bcol1.metric(
                "Fours per 100 balls (batter)",
                f"{batter_boundary_row['fours_per_100_balls']:.2f}"
                if pd.notna(batter_boundary_row["fours_per_100_balls"])
                else "\u2013",
            )
            bcol2.metric(
                "Sixes per 100 balls (batter)",
                f"{batter_boundary_row['sixes_per_100_balls']:.2f}"
                if pd.notna(batter_boundary_row["sixes_per_100_balls"])
                else "\u2013",
            )

        st.subheader("Metrics by batting position")

        pos = filtered.groupby("bat_position").agg(
            innings=("runs", "count"),
            runs=("runs", "sum"),
            balls=("balls_faced", "sum"),
            dismissals=("dismissal_type", lambda x: x.notna().sum()),
            fours=("fours", "sum"),
            sixes=("sixes", "sum"),
        ).reset_index()

        pos["average"] = pos.apply(
            lambda r: r["runs"] / r["dismissals"] if r["dismissals"] > 0 else None,
            axis=1,
        )
        pos["strike_rate"] = pos.apply(
            lambda r: 100 * r["runs"] / r["balls"] if r["balls"] > 0 else None,
            axis=1,
        )
        pos["fours_per_100_balls"] = pos.apply(
            lambda r: 100 * r["fours"] / r["balls"] if r["balls"] > 0 else None,
            axis=1,
        )
        pos["sixes_per_100_balls"] = pos.apply(
            lambda r: 100 * r["sixes"] / r["balls"] if r["balls"] > 0 else None,
            axis=1,
        )

        pos_display = pos.copy()
        pos_display["average"] = pos_display["average"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "\u2013"
        )
        pos_display["strike_rate"] = pos_display["strike_rate"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "\u2013"
        )
        pos_display["fours_per_100_balls"] = pos_display["fours_per_100_balls"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "\u2013"
        )
        pos_display["sixes_per_100_balls"] = pos_display["sixes_per_100_balls"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "\u2013"
        )

        st.dataframe(
            pos_display[
                ["bat_position", "innings", "runs", "average", "strike_rate",
                 "fours_per_100_balls", "sixes_per_100_balls", "dismissals", "fours", "sixes"]
            ].sort_values("bat_position"),
            width='stretch',
        )

        st.subheader("Highlights viewer")

        highlights_df = get_highlights()
        if highlights_df.empty:
            st.info("No highlights available.")
        else:
            highlights_df = highlights_df.merge(
                matches_df[["match_id", "grade", "match_type", "day_1_start"]],
                on="match_id",
                how="left",
            )
            highlights_df = add_season_column(highlights_df, "day_1_start")

            h_filtered = highlights_df.copy()
            if selected_grade:
                h_filtered = h_filtered[h_filtered["grade"].isin(selected_grade)]
            if selected_match_type:
                h_filtered = h_filtered[h_filtered["match_type"].isin(selected_match_type)]
            if selected_season:
                h_filtered = h_filtered[h_filtered["season"].isin(selected_season)]

            if selected_batter_id is not None:
                h_filtered = h_filtered[h_filtered["batter_id"] == str(selected_batter_id)]

            st.markdown("Highlight type filter")
            highlight_type_options = ["All", "Fours", "Sixes", "Dismissals"]
            selected_h_type = st.selectbox("Highlight category", highlight_type_options, index=0)

            def highlight_type_filter(row):
                desc = (row.get("description") or "").upper()
                h_type = (row.get("highlight_type") or "").upper()

                if selected_h_type == "Fours":
                    return "FOUR" in desc or "FOUR" in h_type
                if selected_h_type == "Sixes":
                    return "SIX" in desc or "SIX" in h_type
                if selected_h_type == "Dismissals":
                    return ("OUT" in desc) or ("WICKET" in desc) or ("OUT" in h_type) or ("WICKET" in h_type)
                return True

            h_filtered = h_filtered[h_filtered.apply(highlight_type_filter, axis=1)]

            if h_filtered.empty:
                st.info("No highlights match the current filters.")
            else:
                h_sorted = h_filtered.sort_values(
                    ["day_1_start", "innings_number", "over", "ball_number"],
                    ascending=[False, True, True, True],
                ).reset_index(drop=True)

                default_id = h_sorted.iloc[0]["highlight_id"]
                if (
                    "selected_highlight_id" not in st.session_state
                    or st.session_state["selected_highlight_id"] not in h_sorted["highlight_id"].values
                ):
                    # A brand-new/reset selection (page load, or the
                    # previous selection no longer exists under the current
                    # filters) should NOT autoplay -- only an explicit
                    # "Play" click should.
                    st.session_state["selected_highlight_id"] = default_id
                    st.session_state["highlight_autoplay"] = False

                list_col, video_col = st.columns([3, 2])

                with list_col:
                    st.caption(f"{len(h_sorted)} highlights \u2014 tap \u25b6 to play")
                    with st.container(height=480):
                        for _, row in h_sorted.iterrows():
                            hl_id = row["highlight_id"]
                            row_text_col, row_btn_col = st.columns([5, 1])
                            with row_text_col:
                                date_str = (
                                    row["day_1_start"].strftime("%d %b %Y")
                                    if pd.notna(row.get("day_1_start"))
                                    else ""
                                )
                                st.markdown(
                                    f"**{row.get('batter', '')}** vs {row.get('bowler', '')} "
                                    f"\u2014 {row.get('highlight_type', '')}  \n"
                                    f"{row.get('description', '')}  \n"
                                    f"<span style='color:gray;font-size:0.8em'>{date_str}</span>",
                                    unsafe_allow_html=True,
                                )
                            with row_btn_col:
                                if st.button("\u25b6", key=f"play_{hl_id}"):
                                    st.session_state["selected_highlight_id"] = hl_id
                                    # Explicit play request -- autoplay on
                                    # the rerun this triggers, whether it's
                                    # the same clip replayed or a new one.
                                    st.session_state["highlight_autoplay"] = True
                            st.divider()

                with video_col:
                    selected_highlight = h_sorted[
                        h_sorted["highlight_id"] == st.session_state["selected_highlight_id"]
                    ].iloc[0]
                    st.markdown(f"**{selected_highlight.get('description', '')}**")
                    url = selected_highlight.get("highlight_url")
                    if url:
                        # .pop() consumes the flag so autoplay only fires on
                        # the ONE rerun immediately following a Play click --
                        # any later, unrelated rerun (e.g. changing another
                        # filter) defaults back to paused.
                        autoplay = st.session_state.pop("highlight_autoplay", False)
                        st.video(url, autoplay=autoplay)
                    else:
                        st.info("No video URL available for this highlight.")
