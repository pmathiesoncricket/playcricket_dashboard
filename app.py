import os
import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from dotenv import load_dotenv  # make sure python-dotenv is installed


# ---------- Setup ----------

load_dotenv()  # load .env from project root

st.set_page_config(page_title="PlayCricket Dashboard", layout="wide")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------- Helper functions ----------

def fetch_all_rows(table_name: str, select_str: str, eq_filters: dict | None = None,
                    order_col: str | None = None, page_size: int = 1000) -> pd.DataFrame:
    """
    Fetch ALL rows from a Supabase table, paging past PostgREST's default
    1000-row-per-request limit. Without this, .execute() silently truncates
    large tables and any "distinct values" derived from the result (e.g.
    grade/season/opponent filter options) will only reflect whatever slice
    of rows happened to come back.
    """
    all_rows = []
    start = 0

    while True:
        query = supabase.table(table_name).select(select_str)
        if eq_filters:
            for col, val in eq_filters.items():
                query = query.eq(col, val)
        if order_col:
            query = query.order(order_col)
        query = query.range(start, start + page_size - 1)

        resp = query.execute()
        rows = resp.data or []
        all_rows.extend(rows)

        if len(rows) < page_size:
            break
        start += page_size

    return pd.DataFrame(all_rows)


@st.cache_data(ttl=300)
def get_matches():
    return fetch_all_rows(
        "matches",
        "match_id, grade, match_type, day_1_start, "
        "home_team_id, home_team, away_team_id, away_team",
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

    # Join player_style for bowling type / style
    ps_df = fetch_all_rows("player_style", "player_id, pace_spin, bowl_style")

    pi_df = pi_df.merge(ps_df, on="player_id", how="left")

    # Merge matches to get grade, match_type, date, and opponent info
    df = pi_df.merge(m_df, on="match_id", how="left")

    # Derive opponent (bowling team) from batting team vs home/away
    df["opponent_team"] = None
    mask_home = df["team"] == df["home_team"]
    df.loc[mask_home, "opponent_team"] = df.loc[mask_home, "away_team"]
    df.loc[~mask_home, "opponent_team"] = df.loc[~mask_home, "home_team"]

    return df


@st.cache_data(ttl=300)
def get_deliveries_for_batter(batter_id: str):
    all_rows = []
    start = 0
    page_size = 1000

    while True:
        resp = (
            supabase.table("deliveries")
            .select("*")
            .eq("batter_id", batter_id)
            .order("innings_id", desc=False)
            .order("over", desc=False)
            .order("ball_number", desc=False)
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size

    return pd.DataFrame(all_rows)


@st.cache_data(ttl=300)
def get_highlights():
    """
    Highlight clips (fours, sixes, dismissals, etc.) with video URLs.
    Paginated fetch to avoid PostgREST's default 1000-row cap.
    """
    return fetch_all_rows("highlights", "*")


@st.cache_data(ttl=300)
def get_bowling_deliveries():
    """
    Deliveries at bowler grain — just enough columns to count legal balls
    bowled per bowler and join to matches for the grade filter.
    """
    return fetch_all_rows(
        "deliveries",
        "bowler_id, bowler, match_id, wides",
    )


@st.cache_data(ttl=300)
def get_player_style():
    """Full player_style table (batter_hand, pace_spin, bowl_hand, bowl_style)."""
    return fetch_all_rows("player_style", "*")


def add_season_column(df: pd.DataFrame, date_col: str = "day_1_start") -> pd.DataFrame:
    """
    Add season column with July–June seasons, formatted as 'YYYY/YYYY'.
    """
    if df.empty or date_col not in df.columns:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    year = df[date_col].dt.year
    month = df[date_col].dt.month
    season_start_year = year.where(month >= 7, year - 1)
    season_end_year = season_start_year + 1
    df["season"] = season_start_year.astype(str) + "/" + season_end_year.astype(str)
    return df


SEGMENT_ORDER = ["1–10", "11–20", "21–30", "31–50", "51–75", "76+"]


def sanitize_multiselect_state(key: str, valid_options: list) -> None:
    """
    Cascading filters mean an options list can shrink from one rerun to the
    next. If a widget's stored selection contains values no longer present
    in its options, Streamlit raises an error on render — so strip any
    now-invalid values from session_state before the widget is created.
    """
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in valid_options]


def segment_label(ball_index: int) -> str:
    if ball_index <= 10:
        return "1–10"
    elif ball_index <= 20:
        return "11–20"
    elif ball_index <= 30:
        return "21–30"
    elif ball_index <= 50:
        return "31–50"
    elif ball_index <= 75:
        return "51–75"
    else:
        return "76+"
        
        
        
# ---------- Batting tab ----------

def batting_tab():
    st.header("Batting")

    batting_df = get_batting_innings()
    if batting_df.empty:
        st.info("No batting data available.")
        return

    batting_df = add_season_column(batting_df, "day_1_start")

    # Sidebar filters
    st.sidebar.markdown("### Batting filters")
    st.sidebar.caption("Filters are interdependent — each one narrows the options below it.")

    stage_df = batting_df.copy()

    # Grade: all present, sorted alphabetically. Default = no filter (nothing selected).
    grade_options = sorted(stage_df["grade"].dropna().unique().tolist())
    sanitize_multiselect_state("filter_grade", grade_options)
    selected_grade = st.sidebar.multiselect(
        "Grade", grade_options, default=[], key="filter_grade"
    )
    if selected_grade:
        stage_df = stage_df[stage_df["grade"].isin(selected_grade)]

    # Match type: all present given grade selection, sorted alphabetically
    match_type_options = sorted(stage_df["match_type"].dropna().unique().tolist())
    sanitize_multiselect_state("filter_match_type", match_type_options)
    selected_match_type = st.sidebar.multiselect(
        "Match type", match_type_options,
        default=match_type_options,
        key="filter_match_type",
    )
    if selected_match_type:
        stage_df = stage_df[stage_df["match_type"].isin(selected_match_type)]

    # Season: all present given grade/match type selection, sorted descending. Default = no filter.
    season_options = (
        stage_df["season"]
        .dropna()
        .drop_duplicates()
        .sort_values(ascending=False)
        .tolist()
    )
    sanitize_multiselect_state("filter_season", season_options)
    selected_season = st.sidebar.multiselect(
        "Season (July–June)", options=season_options, default=[], key="filter_season"
    )
    if selected_season:
        stage_df = stage_df[stage_df["season"].isin(selected_season)]

    # Bowling type (pace_spin): all present given filters so far
    bowling_type_options = sorted(stage_df["pace_spin"].dropna().unique().tolist())
    sanitize_multiselect_state("filter_bowling_type", bowling_type_options)
    selected_bowling_type = st.sidebar.multiselect(
        "Bowling type (pace/spin)", bowling_type_options,
        default=bowling_type_options,
        key="filter_bowling_type",
    )
    if selected_bowling_type:
        stage_df = stage_df[stage_df["pace_spin"].isin(selected_bowling_type)]

    # Bowling style (bowl_style): separate filter, given filters so far
    bowl_style_options = sorted(stage_df["bowl_style"].dropna().unique().tolist())
    sanitize_multiselect_state("filter_bowl_style", bowl_style_options)
    selected_bowl_style = st.sidebar.multiselect(
        "Bowling style", bowl_style_options,
        default=bowl_style_options,
        key="filter_bowl_style",
    )
    if selected_bowl_style:
        stage_df = stage_df[stage_df["bowl_style"].isin(selected_bowl_style)]

    # Opponent team: all present given filters so far, sorted alphabetically. Default = no filter.
    opponent_options = sorted(stage_df["opponent_team"].dropna().unique().tolist())
    sanitize_multiselect_state("filter_opponent", opponent_options)
    selected_opponent = st.sidebar.multiselect(
        "Opponent (bowling team)", opponent_options, default=[], key="filter_opponent"
    )
    if selected_opponent:
        stage_df = stage_df[stage_df["opponent_team"].isin(selected_opponent)]

    # Batter filter (page-wide) — only batters with innings under the filters above
    grouped_all = stage_df.groupby("player_id").agg(
        player_name=("player_name", "first"),
    ).reset_index()

    batter_options = sorted(grouped_all["player_name"].dropna().tolist())
    if "filter_batter" in st.session_state and st.session_state["filter_batter"] not in (
        ["All batters"] + batter_options
    ):
        st.session_state["filter_batter"] = "All batters"
    selected_batter_name = st.sidebar.selectbox(
        "Batter (applies to whole page)",
        options=["All batters"] + batter_options,
        index=0,
        key="filter_batter",
    )

    # `stage_df` already reflects every filter above (grade, match type, season,
    # bowling type/style, opponent) — this is also the "population" for
    # player-vs-population comparisons further down, before the batter filter narrows it.
    population_df = stage_df

    # Apply batter filter on top of the population to get the final working set
    filtered = stage_df.copy()
    selected_batter_id = None
    if selected_batter_name != "All batters":
        batter_row = grouped_all[grouped_all["player_name"] == selected_batter_name].iloc[0]
        selected_batter_id = batter_row["player_id"]
        filtered = filtered[filtered["player_id"] == selected_batter_id]

    if filtered.empty:
        st.warning("No batting records match the current filters.")
        return

    # Aggregate per batter (within filters)
    grouped = filtered.groupby("player_id").agg(
        player_name=("player_name", "first"),
        innings=("runs", "count"),
        total_runs=("runs", "sum"),
        total_balls=("balls_faced", "sum"),
        dismissals=("dismissal_type", lambda x: x.notna().sum()),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
    ).reset_index()

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

    # Format: average 2dp, SR & BPD 0dp
    display_df = grouped.copy()
    display_df["average"] = display_df["average"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "–"
    )
    display_df["strike_rate"] = display_df["strike_rate"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "–"
    )
    display_df["BPD"] = display_df["BPD"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "–"
    )

    st.subheader("Batting summary (filtered)")
    st.dataframe(
        display_df[
            ["player_name", "innings", "total_runs", "average", "strike_rate", "BPD", "fours", "sixes"]
        ].sort_values("player_name"),
        use_container_width=True,
    )

    # Batter detail metrics (if specific batter selected)
    if selected_batter_id is not None:
        selected_row = grouped[grouped["player_id"] == selected_batter_id].iloc[0]

        st.subheader(f"Batter detail — {selected_row['player_name']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total runs", int(selected_row["total_runs"]))
        col2.metric(
            "Average",
            f"{selected_row['average']:.2f}" if pd.notna(selected_row["average"]) else "–",
        )
        col3.metric(
            "Strike rate",
            f"{selected_row['strike_rate']:.0f}" if pd.notna(selected_row["strike_rate"]) else "–",
        )

        # ---------- 10-ball segments from deliveries ----------
        st.subheader("Ball‑segment breakdown (deliveries)")

        deliveries_df = get_deliveries_for_batter(str(selected_batter_id))
        if deliveries_df.empty:
            st.info("No deliveries found for this batter.")
        else:
            deliveries_df = deliveries_df.copy()
            deliveries_df["wides"] = deliveries_df["wides"].fillna(0)
            deliveries_df["legal_ball"] = deliveries_df["wides"] == 0

            deliveries_df["ball_index"] = (
                deliveries_df.groupby("innings_id")["legal_ball"]
                .cumsum()
                .where(deliveries_df["legal_ball"], None)
            )

            deliveries_df = deliveries_df[deliveries_df["ball_index"].notna()]
            deliveries_df["ball_index"] = deliveries_df["ball_index"].astype(int)
            deliveries_df["segment"] = pd.Categorical(
                deliveries_df["ball_index"].map(segment_label),
                categories=SEGMENT_ORDER,
                ordered=True,
            )

            seg = deliveries_df.groupby("segment", observed=True).agg(
                balls=("ball_index", "count"),
                runs=("batter_runs", "sum"),
                dismissals=("dismissal_type", lambda x: x.notna().sum()),
                fours=("description", lambda x: x.str.contains("FOUR", case=False, na=False).sum()),
                sixes=("description", lambda x: x.str.contains("SIX", case=False, na=False).sum()),
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
                lambda x: f"{x:.0f}" if pd.notna(x) else "–"
            )
            seg_display["BPD"] = seg_display["BPD"].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else "–"
            )

            st.dataframe(seg_display, use_container_width=True)

            fig_seg = px.bar(
                seg,
                x="segment",
                y="strike_rate",
                category_orders={"segment": SEGMENT_ORDER},
                title=f"Strike rate by ball segment for {selected_row['player_name']}",
            )
            st.plotly_chart(fig_seg, use_container_width=True)
            
            
            
    # ---------- Dismissal type distribution — player vs population ----------
        # Dismissal type distribution — player vs population
        st.subheader("Dismissal type distribution — player vs population")

        # population_df already reflects grade/match type/season/bowling type &
        # style/opponent filters (computed once, above, before the batter filter)
        pop_innings = population_df

        # Exclude did not bat and not out
        excluded_types = {"Did Not Bat", "did not bat", "DNB", "Not Out", "not out"}
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
            # All dismissal types present in either set
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
                title="Dismissal type % — player vs population",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.dataframe(comp_df, use_container_width=True)

        # Distribution of innings scores
        st.subheader("Distribution of innings scores (runs)")
        fig_runs = px.histogram(
            filtered,
            x="runs",
            nbins=20,
            title="Distribution of individual innings runs",
        )
        st.plotly_chart(fig_runs, use_container_width=True)

        # Boundary rate vs population (same filters except batter)
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
            else "–",
        )
        pcol2.metric(
            "Population avg sixes/100 balls",
            f"{pop_boundary['sixes_per_100_balls'].mean():.2f}"
            if pop_boundary["sixes_per_100_balls"].notna().any()
            else "–",
        )

        if selected_batter_id is not None:
            batter_boundary_row = pop_boundary[pop_boundary["player_id"] == selected_batter_id].iloc[0]

            bcol1, bcol2 = st.columns(2)
            bcol1.metric(
                "Fours per 100 balls (batter)",
                f"{batter_boundary_row['fours_per_100_balls']:.2f}"
                if pd.notna(batter_boundary_row["fours_per_100_balls"])
                else "–",
            )
            bcol2.metric(
                "Sixes per 100 balls (batter)",
                f"{batter_boundary_row['sixes_per_100_balls']:.2f}"
                if pd.notna(batter_boundary_row["sixes_per_100_balls"])
                else "–",
            )

        # Metrics by batting position
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
        # Correct: per 100 balls, not above 100
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
            lambda x: f"{x:.2f}" if pd.notna(x) else "–"
        )
        pos_display["strike_rate"] = pos_display["strike_rate"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "–"
        )
        pos_display["fours_per_100_balls"] = pos_display["fours_per_100_balls"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "–"
        )
        pos_display["sixes_per_100_balls"] = pos_display["sixes_per_100_balls"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "–"
        )

        st.dataframe(
            pos_display[
                ["bat_position", "innings", "runs", "average", "strike_rate",
                 "fours_per_100_balls", "sixes_per_100_balls", "dismissals", "fours", "sixes"]
            ].sort_values("bat_position"),
            use_container_width=True,
        )

        # Highlights viewer
        st.subheader("Highlights viewer")

        highlights_df = get_highlights()
        if highlights_df.empty:
            st.info("No highlights available.")
        else:
            # Join to matches for season, grade, match_type
            matches_df = get_matches()
            highlights_df = highlights_df.merge(
                matches_df[["match_id", "grade", "match_type", "day_1_start"]],
                on="match_id",
                how="left",
            )
            highlights_df = add_season_column(highlights_df, "day_1_start")

            # Apply same filters (grade, match_type, season, opponent is implicit via match)
            h_filtered = highlights_df.copy()
            if selected_grade:
                h_filtered = h_filtered[h_filtered["grade"].isin(selected_grade)]
            if selected_match_type:
                h_filtered = h_filtered[h_filtered["match_type"].isin(selected_match_type)]
            if selected_season:
                h_filtered = h_filtered[h_filtered["season"].isin(selected_season)]

            # Filter by batter if selected
            if selected_batter_id is not None:
                h_filtered = h_filtered[h_filtered["batter_id"] == str(selected_batter_id)]

            # Extra filter for highlight type: fours, sixes, dismissals
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
                # Sort: most recent match first, then chronological order within the match
                h_sorted = h_filtered.sort_values(
                    ["day_1_start", "innings_number", "over", "ball_number"],
                    ascending=[False, True, True, True],
                ).reset_index(drop=True)

                # Keep the current selection valid as filters change; default to
                # the most recent highlight.
                default_id = h_sorted.iloc[0]["highlight_id"]
                if (
                    "selected_highlight_id" not in st.session_state
                    or st.session_state["selected_highlight_id"] not in h_sorted["highlight_id"].values
                ):
                    st.session_state["selected_highlight_id"] = default_id

                list_col, video_col = st.columns([3, 2])

                with list_col:
                    st.caption(f"{len(h_sorted)} highlights — tap ▶ to play")
                    # Fixed-height scrollable container holds the FULL list (not just
                    # a top-10 slice), so it works the same on desktop and mobile.
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
                                    f"— {row.get('highlight_type', '')}  \n"
                                    f"{row.get('description', '')}  \n"
                                    f"<span style='color:gray;font-size:0.8em'>{date_str}</span>",
                                    unsafe_allow_html=True,
                                )
                            with row_btn_col:
                                if st.button("▶", key=f"play_{hl_id}"):
                                    st.session_state["selected_highlight_id"] = hl_id
                            st.divider()

                with video_col:
                    selected_highlight = h_sorted[
                        h_sorted["highlight_id"] == st.session_state["selected_highlight_id"]
                    ].iloc[0]
                    st.markdown(f"**{selected_highlight.get('description', '')}**")
                    url = selected_highlight.get("highlight_url")
                    if url:
                        st.video(url, autoplay=True)
                    else:
                        st.info("No video URL available for this highlight.")


# ---------- Bowler Style tab ----------

PACE_SPIN_OPTIONS = ["Pace", "Spin"]
BOWL_HAND_OPTIONS = ["Right", "Left"]
BOWL_STYLE_OPTIONS = ["Right Pace", "Left Pace", "LAOS", "Off Spin", "Leg Spin"]


def bowler_style_tab():
    st.header("Bowler Style")
    st.caption(
        "Identify bowlers with missing style data, and set pace/spin, "
        "bowling hand, and bowling style directly against each bowler."
    )

    deliveries_df = get_bowling_deliveries()
    if deliveries_df.empty:
        st.info("No delivery data available.")
        return

    matches_df = get_matches()
    deliveries_df = deliveries_df.merge(
        matches_df[["match_id", "grade"]], on="match_id", how="left"
    )

    style_df = get_player_style()

    # ---- Filters ----
    filt_col1, filt_col2 = st.columns([2, 1])
    with filt_col1:
        grade_options = sorted(deliveries_df["grade"].dropna().unique().tolist())
        selected_grade = st.multiselect(
            "Grade", grade_options, default=[], key="bowler_filter_grade"
        )
    with filt_col2:
        style_status = st.selectbox(
            "Bowl style",
            ["All", "Populated", "Unpopulated"],
            key="bowler_filter_style_status",
        )

    d_filtered = deliveries_df.copy()
    if selected_grade:
        d_filtered = d_filtered[d_filtered["grade"].isin(selected_grade)]

    # Legal deliveries only (same convention as the batting tab's ball segments)
    d_filtered["wides"] = d_filtered["wides"].fillna(0)
    d_filtered = d_filtered[d_filtered["wides"] == 0]
    d_filtered = d_filtered[d_filtered["bowler_id"].notna()]

    if d_filtered.empty:
        st.info("No bowling deliveries match the current filters.")
        return

    bowler_summary = d_filtered.groupby("bowler_id").agg(
        bowler_name=("bowler", "first"),
        balls=("bowler_id", "count"),
    ).reset_index()

    bowler_summary = bowler_summary.merge(
        style_df, left_on="bowler_id", right_on="player_id", how="left"
    )
    bowler_summary["bowl_style_populated"] = (
        bowler_summary["bowl_style"].notna() & (bowler_summary["bowl_style"] != "")
    )

    if style_status == "Populated":
        bowler_summary = bowler_summary[bowler_summary["bowl_style_populated"]]
    elif style_status == "Unpopulated":
        bowler_summary = bowler_summary[~bowler_summary["bowl_style_populated"]]

    bowler_summary = bowler_summary.sort_values("balls", ascending=False).reset_index(drop=True)

    if bowler_summary.empty:
        st.info("No bowlers match the current filters.")
        return

    st.caption(f"{len(bowler_summary)} bowlers")

    # Option lists: the fixed standard values, plus anything already in the
    # data (in case older/other values are present) so nothing gets hidden.
    pace_spin_opts = sorted(set(PACE_SPIN_OPTIONS) | set(style_df["pace_spin"].dropna().unique().tolist()))
    bowl_hand_opts = sorted(set(BOWL_HAND_OPTIONS) | set(style_df["bowl_hand"].dropna().unique().tolist()))
    bowl_style_opts = sorted(set(BOWL_STYLE_OPTIONS) | set(style_df["bowl_style"].dropna().unique().tolist()))

    if (
        "selected_bowler_id" not in st.session_state
        or st.session_state["selected_bowler_id"] not in bowler_summary["bowler_id"].values
    ):
        st.session_state["selected_bowler_id"] = bowler_summary.iloc[0]["bowler_id"]

    def _dropdown_index(options, current_value):
        full_options = ["–"] + options
        if pd.notna(current_value) and current_value in options:
            return full_options.index(current_value)
        return 0

    with st.container(height=480):
        for _, row in bowler_summary.iterrows():
            bid = row["bowler_id"]
            c_name, c_ps, c_hand, c_style, c_save, c_select = st.columns([2, 1, 1, 1, 1, 1])

            with c_name:
                st.markdown(f"**{row['bowler_name']}**  \n{int(row['balls'])} balls")
            with c_ps:
                ps_val = st.selectbox(
                    "Pace/Spin", ["–"] + pace_spin_opts,
                    index=_dropdown_index(pace_spin_opts, row.get("pace_spin")),
                    key=f"ps_{bid}", label_visibility="collapsed",
                )
            with c_hand:
                hand_val = st.selectbox(
                    "Hand", ["–"] + bowl_hand_opts,
                    index=_dropdown_index(bowl_hand_opts, row.get("bowl_hand")),
                    key=f"hand_{bid}", label_visibility="collapsed",
                )
            with c_style:
                style_val = st.selectbox(
                    "Style", ["–"] + bowl_style_opts,
                    index=_dropdown_index(bowl_style_opts, row.get("bowl_style")),
                    key=f"style_{bid}", label_visibility="collapsed",
                )
            with c_save:
                if st.button("💾 Save", key=f"save_{bid}"):
                    payload = {
                        "player_id": str(bid),
                        "pace_spin": None if ps_val == "–" else ps_val,
                        "bowl_hand": None if hand_val == "–" else hand_val,
                        "bowl_style": None if style_val == "–" else style_val,
                    }
                    try:
                        supabase.table("player_style").upsert(
                            payload, on_conflict="player_id"
                        ).execute()
                        st.success(f"Saved {row['bowler_name']}")
                        get_player_style.clear()
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            with c_select:
                if st.button("Highlights", key=f"sel_{bid}"):
                    st.session_state["selected_bowler_id"] = bid

            st.divider()

    # ---- Highlights panel for the selected bowler ----
    selected_row = bowler_summary[
        bowler_summary["bowler_id"] == st.session_state["selected_bowler_id"]
    ].iloc[0]
    st.subheader(f"Highlights — {selected_row['bowler_name']}")

    highlights_df = get_highlights()
    bowler_highlights = highlights_df[
        highlights_df["bowler_id"] == str(selected_row["bowler_id"])
    ].copy()

    if bowler_highlights.empty:
        st.info("No highlights available for this bowler.")
        return

    bowler_highlights = bowler_highlights.merge(
        matches_df[["match_id", "day_1_start"]], on="match_id", how="left"
    )
    bh_sorted = bowler_highlights.sort_values(
        ["day_1_start", "innings_number", "over", "ball_number"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    if (
        "selected_bowler_highlight_id" not in st.session_state
        or st.session_state["selected_bowler_highlight_id"] not in bh_sorted["highlight_id"].values
    ):
        st.session_state["selected_bowler_highlight_id"] = bh_sorted.iloc[0]["highlight_id"]

    # Equal columns so the video takes up roughly half the screen width
    list_col, video_col = st.columns(2)

    with list_col:
        st.caption(f"{len(bh_sorted)} highlights — tap ▶ to play")
        with st.container(height=400):
            for _, hrow in bh_sorted.iterrows():
                hid = hrow["highlight_id"]
                txt_col, btn_col = st.columns([5, 1])
                with txt_col:
                    date_str = (
                        hrow["day_1_start"].strftime("%d %b %Y")
                        if pd.notna(hrow.get("day_1_start"))
                        else ""
                    )
                    st.markdown(
                        f"**{hrow.get('batter', '')}** — {hrow.get('highlight_type', '')}  \n"
                        f"{hrow.get('description', '')}  \n"
                        f"<span style='color:gray;font-size:0.8em'>{date_str}</span>",
                        unsafe_allow_html=True,
                    )
                with btn_col:
                    if st.button("▶", key=f"bowlerhl_play_{hid}"):
                        st.session_state["selected_bowler_highlight_id"] = hid
                st.divider()

    with video_col:
        sel_h = bh_sorted[
            bh_sorted["highlight_id"] == st.session_state["selected_bowler_highlight_id"]
        ].iloc[0]
        st.markdown(f"**{sel_h.get('description', '')}**")
        url = sel_h.get("highlight_url")
        if url:
            st.video(url, autoplay=True)
        else:
            st.info("No video URL available for this highlight.")


# ---------- Bowling tab (placeholder) ----------

def bowling_tab():
    st.header("Bowling")
    st.info("Bowling metrics to be implemented next.")


# ---------- Team tab (placeholder) ----------

def team_tab():
    st.header("Team")
    st.info("Team-level metrics to be implemented next.")


# ---------- Main app ----------

st.title("PlayCricket Dashboard")

tabs = st.tabs(["Batting", "Bowling", "Bowler Style", "Team"])

with tabs[0]:
    batting_tab()

with tabs[1]:
    bowling_tab()

with tabs[2]:
    bowler_style_tab()

with tabs[3]:
    team_tab()
