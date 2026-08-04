import streamlit as st
import pandas as pd
import plotly.express as px

from db import (
    get_bowling_innings, get_batting_innings, get_deliveries_for_bowler,
    get_bowling_conceded_summary, get_wicket_deliveries, get_highlights,
    get_matches, get_player_style,
)
from helpers import add_season_column, cascading_multiselect, MAROON, MAROON_SHADES
# Reuse the exact YouTube-timestamping / label-padding machinery already
# ironed out on the Batting tab, rather than re-deriving it.
from tab_batting import build_timestamped_youtube_url, resolve_ball_video_url, _pad, NBSP

DASH = "\u2013"

# Dismissal types NOT credited as a wicket to the bowler (case-insensitive
# substring match, so it also catches "Retired Not Out", "Retired Hurt",
# "Obstructing the Field", etc).
WICKET_EXCLUDE_PATTERNS = ["run out", "retired", "obstruct"]

# Dismissal types excluded from the dismissal-type DISTRIBUTION chart (a
# different, narrower exclusion list than the wicket-credit rule above --
# this one also drops Hit Wicket, which IS still credited as a bowler
# wicket, just not shown in this particular type breakdown).
DISMISSAL_DIST_EXCLUDE_PATTERNS = ["hit wicket", "obstruct", "retired", "run out"]

# Over ranges are inclusive, matching deliveries.over as-is (40-over cricket)
PHASES = [
    ("Powerplay (0-9)", 0, 9),
    ("Middle (10-29)", 10, 29),
    ("L10 (30-40)", 30, 40),
]
PHASE_ORDER = [p[0] for p in PHASES]

POSITION_BUCKETS = ["Top order (1-3)", "Middle order (4-7)", "Tail (8+)"]


def _fmt(x, dp=2):
    return f"{x:.{dp}f}" if pd.notna(x) else DASH


def _matches_any(text, patterns):
    if pd.isna(text):
        return False
    t = str(text).lower()
    return any(p in t for p in patterns)


def is_bowler_wicket(dismissal_type):
    if pd.isna(dismissal_type):
        return False
    return not _matches_any(dismissal_type, WICKET_EXCLUDE_PATTERNS)


def is_countable_dismissal(dismissal_type):
    if pd.isna(dismissal_type):
        return False
    return not _matches_any(dismissal_type, DISMISSAL_DIST_EXCLUDE_PATTERNS)


def overs_to_balls(overs):
    """Converts cricket-notation overs (e.g. 4.3 = 4 overs, 3 balls) to a
    total ball count. Assumes the numeric(6,1) `overs` column stores the
    ball-in-over digit directly (0-5) in its single decimal place."""
    if pd.isna(overs):
        return 0
    overs = float(overs)
    whole = int(overs)
    frac_balls = int(round((overs - whole) * 10))
    return whole * 6 + frac_balls


def _phase_of(over):
    for label, lo, hi in PHASES:
        if lo <= over <= hi:
            return label
    return None


def _position_bucket(pos):
    if pd.isna(pos):
        return "Unknown"
    pos = int(pos)
    if pos <= 3:
        return "Top order (1-3)"
    if pos <= 7:
        return "Middle order (4-7)"
    return "Tail (8+)"


def _legal_deliveries(deliveries_df):
    """Legal balls only: exclude wides AND no-balls (matches the same
    convention used for bowling economy elsewhere in this app)."""
    d = deliveries_df.copy()
    d["wides"] = d["wides"].fillna(0)
    d["no_balls"] = d["no_balls"].fillna(0)
    return d


def _prep_bowling_deliveries(deliveries_df):
    """Adds the derived per-ball columns every bowling breakdown in this
    tab depends on: is_legal, runs_charged, is_wicket, is_four, is_six."""
    d = _legal_deliveries(deliveries_df)
    d["is_legal"] = (d["wides"] == 0) & (d["no_balls"] == 0)
    if "bowler_runs" in d.columns:
        fallback = d["batter_runs"].fillna(0) + d["wides"].fillna(0) + d["no_balls"].fillna(0)
        d["runs_charged"] = d["bowler_runs"].fillna(fallback)
    else:
        d["runs_charged"] = d["batter_runs"].fillna(0) + d["wides"].fillna(0) + d["no_balls"].fillna(0)
    d["is_wicket"] = d["dismissal_type"].apply(is_bowler_wicket)
    d["is_four"] = d["description"].str.contains("FOUR", case=False, na=False)
    d["is_six"] = d["description"].str.contains("SIX", case=False, na=False)
    return d


def _bowling_metrics(df, group_cols):
    """Core bowling metrics (wickets/average/economy/BPD/fours/sixes) from
    prepped ball-by-ball data, grouped by `group_cols`."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    g = df.groupby(group_cols, dropna=False).agg(
        balls=("is_legal", "sum"),
        runs_conceded=("runs_charged", "sum"),
        wickets=("is_wicket", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
    ).reset_index()
    g["average"] = g.apply(lambda r: r["runs_conceded"] / r["wickets"] if r["wickets"] > 0 else None, axis=1)
    g["economy"] = g.apply(lambda r: r["runs_conceded"] / (r["balls"] / 6) if r["balls"] > 0 else None, axis=1)
    g["BPD"] = g.apply(lambda r: r["balls"] / r["wickets"] if r["wickets"] > 0 else None, axis=1)
    return g


def _render_metrics_table(df, category_col, category_label, sort_col="wickets"):
    """Renders a bowling-metrics table. Sorting is done BEFORE the display
    rename (sorting after rename was the source of a KeyError, since
    "wickets"/the category column no longer existed under their original
    names post-rename). When sort_col is the category column itself
    (e.g. an ordered phase/position categorical), we sort ascending to
    preserve its natural order; otherwise (e.g. "wickets") we sort
    descending so the best performers show first.
    """
    disp = df.copy()
    ascending = sort_col == category_col
    disp = disp.sort_values(sort_col, ascending=ascending)

    disp["average"] = disp["average"].apply(lambda x: _fmt(x, 2))
    disp["economy"] = disp["economy"].apply(lambda x: _fmt(x, 2))
    disp["BPD"] = disp["BPD"].apply(lambda x: _fmt(x, 0) if pd.notna(x) else DASH)

    st.dataframe(
        disp[[category_col, "wickets", "runs_conceded", "average", "economy", "BPD", "fours", "sixes"]]
        .rename(columns={
            category_col: category_label, "wickets": "Wickets", "runs_conceded": "Runs",
            "average": "Average", "economy": "Economy", "BPD": "BPD",
            "fours": "Fours", "sixes": "Sixes",
        }),
        width="stretch", hide_index=True,
    )


def bowling_tab():
    st.header("Bowling")

    bowling_df = get_bowling_innings()
    if bowling_df.empty:
        st.info("No bowling data available.")
        return

    bowling_df = add_season_column(bowling_df, "day_1_start")
    matches_df = get_matches()

    st.sidebar.markdown("### Bowling filters")
    st.sidebar.caption("Filters are interdependent \u2014 each one narrows the options below it.")

    stage_df = bowling_df.copy()

    grade_options = sorted(stage_df["grade"].dropna().unique().tolist())
    selected_grade = cascading_multiselect(
        st.sidebar, "Grade", grade_options, "bowl_filter_grade"
    )
    if selected_grade:
        stage_df = stage_df[stage_df["grade"].isin(selected_grade)]

    match_type_options = sorted(stage_df["match_type"].dropna().unique().tolist())
    selected_match_type = cascading_multiselect(
        st.sidebar, "Match type", match_type_options, "bowl_filter_match_type",
        default_options=match_type_options,
    )
    if selected_match_type:
        stage_df = stage_df[stage_df["match_type"].isin(selected_match_type)]

    season_options = (
        stage_df["season"]
        .dropna()
        .drop_duplicates()
        .sort_values(ascending=False)
        .tolist()
    )
    selected_season = cascading_multiselect(
        st.sidebar, "Season (July\u2013June)", season_options, "bowl_filter_season"
    )
    if selected_season:
        stage_df = stage_df[stage_df["season"].isin(selected_season)]

    pace_spin_options = sorted(stage_df["pace_spin"].dropna().unique().tolist())
    selected_pace_spin = cascading_multiselect(
        st.sidebar, "Bowling type (pace/spin)", pace_spin_options, "bowl_filter_pace_spin",
        default_options=pace_spin_options,
    )
    if selected_pace_spin:
        stage_df = stage_df[stage_df["pace_spin"].isin(selected_pace_spin)]

    bowl_style_options = sorted(stage_df["bowl_style"].dropna().unique().tolist())
    selected_bowl_style = cascading_multiselect(
        st.sidebar, "Bowling style", bowl_style_options, "bowl_filter_bowl_style",
        default_options=bowl_style_options,
    )
    if selected_bowl_style:
        stage_df = stage_df[stage_df["bowl_style"].isin(selected_bowl_style)]

    # Batter hand -- mirrors the pace_spin/bowl_style filters above exactly:
    # joined onto this row's own player_id, so it filters the pool of
    # BOWLERS by their own batting hand. The hand of whoever they actually
    # bowled AT (a ball-by-ball concept) is covered separately by the
    # "Bowling vs batter hand" and "Summary by batter hand" sections below,
    # which join player_style a second time -- onto deliveries.batter_id.
    batter_hand_options = sorted(stage_df["batter_hand"].dropna().unique().tolist())
    selected_batter_hand = cascading_multiselect(
        st.sidebar, "Batter hand (bowler's own)", batter_hand_options, "bowl_filter_batter_hand",
        default_options=batter_hand_options,
    )
    if selected_batter_hand:
        stage_df = stage_df[stage_df["batter_hand"].isin(selected_batter_hand)]

    opponent_options = sorted(stage_df["opponent_team"].dropna().unique().tolist())
    selected_opponent = cascading_multiselect(
        st.sidebar, "Opponent (batting team)", opponent_options, "bowl_filter_opponent"
    )
    if selected_opponent:
        stage_df = stage_df[stage_df["opponent_team"].isin(selected_opponent)]

    if stage_df.empty:
        st.warning("No bowling records match the current filters.")
        return

    grouped_all = stage_df.groupby("player_id").agg(
        player_name=("player_name", "first"),
    ).reset_index()
    bowler_options = sorted(grouped_all["player_name"].dropna().tolist())

    if "pending_bowler_filter" in st.session_state:
        st.session_state["filter_bowler"] = st.session_state.pop("pending_bowler_filter")

    if "filter_bowler" in st.session_state and st.session_state["filter_bowler"] not in (
        ["All bowlers"] + bowler_options
    ):
        st.session_state["filter_bowler"] = "All bowlers"

    population_df = stage_df

    st.subheader("Bowling summary (filtered)")
    selected_bowler_name = st.selectbox(
        "Bowler (applies to whole page) \u2014 or click a row below",
        options=["All bowlers"] + bowler_options,
        index=0,
        key="filter_bowler",
    )

    filtered = stage_df.copy()
    selected_bowler_id = None
    if selected_bowler_name != "All bowlers":
        bowler_row = grouped_all[grouped_all["player_name"] == selected_bowler_name].iloc[0]
        selected_bowler_id = bowler_row["player_id"]
        filtered = filtered[filtered["player_id"] == selected_bowler_id]

    if filtered.empty:
        st.warning("No bowling records match the current filters.")
        return

    # ---------------- Summary table (whole-innings figures from player_innings) ----------------
    filtered = filtered.copy()
    filtered["balls_bowled"] = filtered["overs"].apply(overs_to_balls)

    grouped = filtered.groupby("player_id").agg(
        player_name=("player_name", "first"),
        innings=("wickets_taken", "count"),
        wickets=("wickets_taken", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        balls_bowled=("balls_bowled", "sum"),
    ).reset_index()

    # Fours/sixes conceded don't exist on player_innings for bowling rows --
    # pull them from the pre-aggregated (server-side GROUP BY) deliveries
    # summary, scoped down to exactly the match/innings rows currently in
    # `filtered` so it still respects every sidebar filter.
    conceded_df = get_bowling_conceded_summary()
    if not conceded_df.empty:
        scope_keys = filtered[["match_id", "innings_id", "player_id"]].rename(
            columns={"player_id": "bowler_id"}
        )
        conceded_scope = conceded_df.merge(
            scope_keys, on=["bowler_id", "match_id", "innings_id"], how="inner"
        )
        fours_sixes = conceded_scope.groupby("bowler_id").agg(
            fours=("fours", "sum"), sixes=("sixes", "sum"),
        ).reset_index().rename(columns={"bowler_id": "player_id"})
        grouped = grouped.merge(fours_sixes, on="player_id", how="left")
    else:
        grouped["fours"] = None
        grouped["sixes"] = None
    grouped["fours"] = grouped["fours"].fillna(0).astype(int)
    grouped["sixes"] = grouped["sixes"].fillna(0).astype(int)

    grouped["average"] = grouped.apply(
        lambda r: r["runs_conceded"] / r["wickets"] if r["wickets"] > 0 else None, axis=1,
    )
    grouped["economy"] = grouped.apply(
        lambda r: r["runs_conceded"] / (r["balls_bowled"] / 6) if r["balls_bowled"] > 0 else None, axis=1,
    )
    grouped["BPD"] = grouped.apply(
        lambda r: r["balls_bowled"] / r["wickets"] if r["wickets"] > 0 else None, axis=1,
    )

    display_df = grouped.copy()
    display_df["average"] = display_df["average"].apply(lambda x: _fmt(x, 2))
    display_df["economy"] = display_df["economy"].apply(lambda x: _fmt(x, 2))
    display_df["BPD"] = display_df["BPD"].apply(lambda x: _fmt(x, 0) if pd.notna(x) else DASH)

    summary_table = display_df[
        ["player_name", "innings", "wickets", "average", "economy", "BPD", "fours", "sixes"]
    ].sort_values("wickets", ascending=False).reset_index(drop=True)
    summary_table["wickets"] = summary_table["wickets"].fillna(0).astype(int)

    MAX_ROWS_VISIBLE = 10
    rows_to_show = min(len(summary_table), MAX_ROWS_VISIBLE)
    TABLE_HEIGHT = (rows_to_show + 1) * 35 + 3

    summary_event = st.dataframe(
        summary_table,
        width="stretch",
        height=TABLE_HEIGHT,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="bowling_summary_table",
    )

    if summary_event.selection.rows:
        clicked_idx = summary_event.selection.rows[0]
        if clicked_idx < len(summary_table):
            clicked_name = summary_table.iloc[clicked_idx]["player_name"]
            if st.session_state.get("filter_bowler") != clicked_name:
                st.session_state["pending_bowler_filter"] = clicked_name
                st.rerun()
        # else: stale selection from a since-shrunk table -- ignore it

    if selected_bowler_id is None:
        st.info("Select a bowler above (or click a row in the summary table) to see full bowling detail.")
        return

    # =========================================================
    # Everything below only activates once a single bowler is selected.
    # =========================================================
    selected_row = grouped[grouped["player_id"] == selected_bowler_id].iloc[0]

    st.subheader(f"Bowler detail \u2014 {selected_row['player_name']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total wickets", int(selected_row["wickets"]) if pd.notna(selected_row["wickets"]) else 0)
    col2.metric(
        "Average",
        f"{selected_row['average']:.2f}" if pd.notna(selected_row["average"]) else "\u2013",
    )
    col3.metric(
        "Economy",
        f"{selected_row['economy']:.2f}" if pd.notna(selected_row["economy"]) else "\u2013",
    )

    # No ball-segment breakdown for bowling (no equivalent requested).

    deliveries_df = get_deliveries_for_bowler(str(selected_bowler_id))
    if not deliveries_df.empty:
        deliveries_df = _prep_bowling_deliveries(deliveries_df)
        # Batter-hand-of-the-batter-FACED, joined onto deliveries via
        # batter_id -- this is player_style joined a second time, under a
        # different key, from the same table used for the bowler's own
        # attributes up in get_bowling_innings().
        batter_style_lookup = get_player_style()[["player_id", "batter_hand"]].rename(
            columns={"player_id": "batter_id", "batter_hand": "faced_batter_hand"}
        )
        deliveries_df = deliveries_df.merge(batter_style_lookup, on="batter_id", how="left")
        deliveries_df["faced_batter_hand"] = deliveries_df["faced_batter_hand"].fillna("Unknown")

    # ---------------- Bowling vs batter hand ----------------
    st.subheader("Bowling vs batter hand (deliveries)")
    if deliveries_df.empty:
        st.info("No deliveries found for this bowler.")
    else:
        by_hand = _bowling_metrics(deliveries_df, "faced_batter_hand")
        _render_metrics_table(by_hand, "faced_batter_hand", "Batter hand")

    # ---------------- Match-by-match bowling ----------------
    st.subheader("Match-by-match bowling")

    match_df = filtered.copy()
    match_df["match_name"] = match_df["match_type"].astype(str) + " v " + match_df["opponent_team"].astype(str)
    match_df["wickets_taken"] = match_df["wickets_taken"].fillna(0)
    match_df["runs_conceded"] = match_df["runs_conceded"].fillna(0)
    match_df["figures"] = (
        match_df["wickets_taken"].astype(int).astype(str) + "/" + match_df["runs_conceded"].astype(int).astype(str)
    )
    match_df["economy"] = match_df.apply(
        lambda r: r["runs_conceded"] / (r["balls_bowled"] / 6) if r["balls_bowled"] > 0 else None, axis=1,
    )

    match_rows = match_df[
        ["match_id", "innings_id", "day_1_start", "match_name", "overs", "figures", "economy"]
    ].copy()
    match_rows = match_rows.sort_values("day_1_start", ascending=False).reset_index(drop=True)

    stream_cols = ["match_id", "day1_stream_url", "day1_stream_start", "day2_stream_url", "day2_stream_start"]
    available_stream_cols = [c for c in stream_cols if c in matches_df.columns]
    matches_stream_df = (
        matches_df[available_stream_cols].copy()
        if available_stream_cols else pd.DataFrame(columns=stream_cols)
    )

    st.caption(f"{len(match_rows)} innings \u2014 expand a row to see ball-by-ball detail and video links")

    for _, m_row in match_rows.iterrows():
        date_str = (
            pd.to_datetime(m_row["day_1_start"]).strftime("%d %b %Y")
            if pd.notna(m_row["day_1_start"]) else "Unknown date"
        )
        overs_str = f"{m_row['overs']:.1f}" if pd.notna(m_row["overs"]) else "\u2013"
        econ_str = f"Econ {m_row['economy']:.2f}" if pd.notna(m_row["economy"]) else "Econ \u2013"

        label = (
            f"`{_pad(date_str, 16)}{_pad(m_row['match_name'], 36)}{_pad('Ov ' + overs_str, 10)}"
            f"{_pad(m_row['figures'], 10)}{_pad(econ_str, 12)}`"
        )

        with st.expander(label):
            if deliveries_df.empty:
                st.info("No ball-by-ball data available for this innings.")
                continue

            innings_balls = deliveries_df[
                (deliveries_df["match_id"] == m_row["match_id"])
                & (deliveries_df["innings_id"] == m_row["innings_id"])
            ].copy()

            if innings_balls.empty:
                st.info("No ball-by-ball data available for this innings.")
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

            offset_key = f"bowl_offset_{m_row['match_id']}_{m_row['innings_id']}"
            offset_seconds_adjustment = st.number_input(
                "Video timestamp offset (seconds) for this innings' stream",
                value=-5,
                step=1,
                help=(
                    "Fine-tune how far into the stream each ball's video link jumps. "
                    "Negative values start the clip slightly earlier."
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
                ["over_ball", "batter", "faced_batter_hand", "runs_charged", "description", "video_url"]
            ].rename(columns={
                "over_ball": "Over.Ball", "batter": "Batter",
                "faced_batter_hand": "Batter hand", "runs_charged": "Runs",
                "description": "Description", "video_url": "Video",
            })

            st.dataframe(
                ball_display,
                hide_index=True,
                width="stretch",
                column_config={
                    "Video": st.column_config.LinkColumn("Video", display_text="\u25b6 Watch"),
                },
            )

    # ---------------- Dismissal type distribution vs population ----------------
    st.subheader("Dismissal type distribution \u2014 player vs population")

    wicket_df = get_wicket_deliveries()
    if wicket_df.empty:
        st.info("No dismissal data available.")
    else:
        # Population = every dismissal off a bowler whose innings are in the
        # currently-filtered population_df's match/innings scope.
        pop_scope_keys = population_df[["match_id", "innings_id"]].drop_duplicates()
        pop_wickets = wicket_df.merge(pop_scope_keys, on=["match_id", "innings_id"], how="inner")
        pop_wickets = pop_wickets[pop_wickets["dismissal_type"].apply(is_countable_dismissal)]

        player_wickets = pop_wickets[pop_wickets["bowler_id"] == str(selected_bowler_id)]

        if pop_wickets.empty or player_wickets.empty:
            st.info("Not enough dismissal data to compare player vs population.")
        else:
            all_types = sorted(pop_wickets["dismissal_type"].dropna().unique().tolist())

            pop_counts = pop_wickets["dismissal_type"].value_counts().reindex(all_types, fill_value=0)
            player_counts = player_wickets["dismissal_type"].value_counts().reindex(all_types, fill_value=0)

            pop_total = pop_counts.sum()
            player_total = player_counts.sum()

            pop_pct = (pop_counts / pop_total * 100) if pop_total > 0 else pop_counts
            player_pct = (player_counts / player_total * 100) if player_total > 0 else player_counts

            comp_df = pd.DataFrame({
                "dismissal_type": all_types,
                "Population %": pop_pct.values,
                "Population count": pop_counts.values,
                "Player %": player_pct.values,
                "Player count": player_counts.values,
            })

            comp_melt = comp_df.melt(
                id_vars=["dismissal_type"],
                value_vars=["Population %", "Player %"],
                var_name="group",
                value_name="percentage",
            )
            fig_comp = px.bar(
                comp_melt,
                x="dismissal_type",
                y="percentage",
                color="group",
                barmode="group",
                title="Dismissal type % \u2014 player vs population",
                color_discrete_sequence=MAROON_SHADES[:2],
            )
            st.plotly_chart(fig_comp, width="stretch")
            st.dataframe(comp_df, width="stretch")

    # No "Distribution of innings scores" section -- no batting equivalent for bowling.

    # ---------------- Boundary rate vs population ----------------
    st.subheader("Boundary rate vs population (conceded)")

    conceded_df = get_bowling_conceded_summary()
    if conceded_df.empty:
        st.info("No delivery data available for boundary-rate comparison.")
    else:
        pop_scope_keys = population_df[["match_id", "innings_id", "player_id"]].rename(
            columns={"player_id": "bowler_id"}
        )
        pop_conceded = conceded_df.merge(pop_scope_keys, on=["bowler_id", "match_id", "innings_id"], how="inner")

        pop_by_bowler = pop_conceded.groupby("bowler_id").agg(
            legal_balls=("legal_balls", "sum"), fours=("fours", "sum"), sixes=("sixes", "sum"),
        ).reset_index()
        pop_by_bowler["fours_per_100_balls"] = pop_by_bowler.apply(
            lambda r: 100 * r["fours"] / r["legal_balls"] if r["legal_balls"] > 0 else None, axis=1,
        )
        pop_by_bowler["sixes_per_100_balls"] = pop_by_bowler.apply(
            lambda r: 100 * r["sixes"] / r["legal_balls"] if r["legal_balls"] > 0 else None, axis=1,
        )

        pcol1, pcol2 = st.columns(2)
        pcol1.metric(
            "Population avg fours conceded/100 balls",
            f"{pop_by_bowler['fours_per_100_balls'].mean():.2f}"
            if pop_by_bowler["fours_per_100_balls"].notna().any() else "\u2013",
        )
        pcol2.metric(
            "Population avg sixes conceded/100 balls",
            f"{pop_by_bowler['sixes_per_100_balls'].mean():.2f}"
            if pop_by_bowler["sixes_per_100_balls"].notna().any() else "\u2013",
        )

        bowler_boundary_rows = pop_by_bowler[pop_by_bowler["bowler_id"] == str(selected_bowler_id)]
        if not bowler_boundary_rows.empty:
            bowler_boundary_row = bowler_boundary_rows.iloc[0]
            bcol1, bcol2 = st.columns(2)
            bcol1.metric(
                "Fours conceded per 100 balls (bowler)",
                f"{bowler_boundary_row['fours_per_100_balls']:.2f}"
                if pd.notna(bowler_boundary_row["fours_per_100_balls"]) else "\u2013",
            )
            bcol2.metric(
                "Sixes conceded per 100 balls (bowler)",
                f"{bowler_boundary_row['sixes_per_100_balls']:.2f}"
                if pd.notna(bowler_boundary_row["sixes_per_100_balls"]) else "\u2013",
            )

    # ---------------- Metrics by batting position (Top/Middle/Tail) ----------------
    st.subheader("Metrics by batting position (bowling)")

    if deliveries_df.empty:
        st.info("No delivery data available for this bowler.")
    else:
        batting_pi = get_batting_innings()
        if batting_pi.empty:
            st.info("Batting position lookup unavailable right now.")
        else:
            pos_lookup = batting_pi[["match_id", "innings_id", "player_id", "bat_position"]].rename(
                columns={"player_id": "batter_id"}
            )
            pos_deliveries = deliveries_df.merge(
                pos_lookup, on=["match_id", "innings_id", "batter_id"], how="left"
            )
            pos_deliveries["position_bucket"] = pos_deliveries["bat_position"].apply(_position_bucket)
            pos_deliveries["position_bucket"] = pd.Categorical(
                pos_deliveries["position_bucket"], categories=POSITION_BUCKETS + ["Unknown"], ordered=True,
            )
            by_position = _bowling_metrics(pos_deliveries, "position_bucket")
            _render_metrics_table(by_position, "position_bucket", "Batting position", sort_col="position_bucket")

    # ---------------- Summary by Game Type ----------------
    st.subheader("Bowling metrics by game type")

    if deliveries_df.empty:
        st.info("No delivery data available for this bowler.")
    else:
        gt_deliveries = deliveries_df.merge(
            matches_df[["match_id", "match_type"]], on="match_id", how="left"
        )
        by_game_type = _bowling_metrics(gt_deliveries, "match_type")
        _render_metrics_table(by_game_type, "match_type", "Game type")

        # ---------------- One Day phase breakdown ----------------
        od_deliveries = gt_deliveries[gt_deliveries["match_type"] == "One Day"].copy()
        if od_deliveries.empty:
            st.caption("No One Day deliveries for this bowler \u2014 phase report skipped.")
        else:
            st.markdown("**One Day \u2014 phase breakdown**")
            od_deliveries["phase"] = od_deliveries["over"].apply(_phase_of)
            od_deliveries = od_deliveries.dropna(subset=["phase"])
            od_deliveries["phase"] = pd.Categorical(od_deliveries["phase"], categories=PHASE_ORDER, ordered=True)
            by_phase = _bowling_metrics(od_deliveries, "phase")
            _render_metrics_table(by_phase, "phase", "Phase", sort_col="phase")

    # ---------------- Summary by Batter Hand (faced) ----------------
    st.subheader("Bowling metrics by batter hand (faced)")

    if deliveries_df.empty:
        st.info("No delivery data available for this bowler.")
    else:
        by_hand_summary = _bowling_metrics(deliveries_df, "faced_batter_hand")
        _render_metrics_table(by_hand_summary, "faced_batter_hand", "Batter hand")

    # ---------------- Highlights viewer ----------------
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

        h_filtered = h_filtered[h_filtered["bowler_id"] == str(selected_bowler_id)]

        st.markdown("Highlight type filter")
        highlight_type_options = ["All", "Fours", "Sixes", "Dismissals"]
        selected_h_type = st.selectbox("Highlight category", highlight_type_options, index=0, key="bowl_highlight_type")

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
                "selected_bowl_highlight_id" not in st.session_state
                or st.session_state["selected_bowl_highlight_id"] not in h_sorted["highlight_id"].values
            ):
                st.session_state["selected_bowl_highlight_id"] = default_id

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
                                f"\u2014 {row.get('highlight_type', '')} \n"
                                f"{row.get('description', '')} \n"
                                f"{date_str}",
                                unsafe_allow_html=True,
                            )
                        with row_btn_col:
                            if st.button("\u25b6", key=f"bowl_play_{hl_id}"):
                                st.session_state["selected_bowl_highlight_id"] = hl_id
                        st.divider()

            with video_col:
                selected_highlight = h_sorted[
                    h_sorted["highlight_id"] == st.session_state["selected_bowl_highlight_id"]
                ].iloc[0]
                st.markdown(f"**{selected_highlight.get('description', '')}**")
                url = selected_highlight.get("highlight_url")
                if url:
                    st.video(url, autoplay=True)
                else:
                    st.info("No video URL available for this highlight.")
