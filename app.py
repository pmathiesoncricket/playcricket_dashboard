import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
from sqlalchemy import text


# ---------- Setup ----------

st.set_page_config(page_title="PlayCricket Dashboard", layout="wide")

conn = st.connection("postgresql", type="sql")

# Colour palette centred on maroon, used for every bar/histogram chart.
MAROON = "#73173F"
MAROON_SHADES = ["#73173F", "#C97292", "#9E3A5D", "#4A0F29", "#D9A0B5"]


# ---------- Helper functions ----------

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
    return fetch_all_rows("highlights", "*")


@st.cache_data(ttl=300)
def get_bowling_deliveries():
    return fetch_all_rows(
        "deliveries",
        "ball_id, bowler_id, bowler, match_id",
        eq_filters={"wides": 0},
        id_col="ball_id",
    )


@st.cache_data(ttl=300)
def get_player_style():
    return fetch_all_rows("player_style", "*")


def add_season_column(df: pd.DataFrame, date_col: str = "day_1_start") -> pd.DataFrame:
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


SEGMENT_ORDER = ["1\u201310", "11\u201320", "21\u201330", "31\u201350", "51\u201375", "76+"]


def sanitize_multiselect_state(key: str, valid_options: list) -> None:
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in valid_options]


def cascading_multiselect(container, label: str, options: list, key: str,
                           default_options: list | None = None):
    sanitize_multiselect_state(key, options)
    kwargs = {}
    if key not in st.session_state:
        kwargs["default"] = default_options if default_options is not None else []
    return container.multiselect(label, options, key=key, **kwargs)


def segment_label(ball_index: int) -> str:
    if ball_index <= 10:
        return "1\u201310"
    elif ball_index <= 20:
        return "11\u201320"
    elif ball_index <= 30:
        return "21\u201330"
    elif ball_index <= 50:
        return "31\u201350"
    elif ball_index <= 75:
        return "51\u201375"
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

    st.sidebar.markdown("### Batting filters")
    st.sidebar.caption("Filters are interdependent — each one narrows the options below it.")

    stage_df = batting_df.copy()

    grade_options = sorted(stage_df["grade"].dropna().unique().tolist())
    selected_grade = cascading_multiselect(
        st.sidebar, "Grade", grade_options, "filter_grade"
    )
    if selected_grade:
        stage_df = stage_df[stage_df["grade"].isin(selected_grade)]

    match_type_options = sorted(stage_df["match_type"].dropna().unique().tolist())
    selected_match_type = cascading_multiselect(
        st.sidebar, "Match type", match_type_options, "filter_match_type",
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
        st.sidebar, "Season (July\u2013June)", season_options, "filter_season"
    )
    if selected_season:
        stage_df = stage_df[stage_df["season"].isin(selected_season)]

    bowling_type_options = sorted(stage_df["pace_spin"].dropna().unique().tolist())
    selected_bowling_type = cascading_multiselect(
        st.sidebar, "Bowling type (pace/spin)", bowling_type_options, "filter_bowling_type",
        default_options=bowling_type_options,
    )
    if selected_bowling_type:
        stage_df = stage_df[stage_df["pace_spin"].isin(selected_bowling_type)]

    bowl_style_options = sorted(stage_df["bowl_style"].dropna().unique().tolist())
    selected_bowl_style = cascading_multiselect(
        st.sidebar, "Bowling style", bowl_style_options, "filter_bowl_style",
        default_options=bowl_style_options,
    )
    if selected_bowl_style:
        stage_df = stage_df[stage_df["bowl_style"].isin(selected_bowl_style)]

    opponent_options = sorted(stage_df["opponent_team"].dropna().unique().tolist())
    selected_opponent = cascading_multiselect(
        st.sidebar, "Opponent (bowling team)", opponent_options, "filter_opponent"
    )
    if selected_opponent:
        stage_df = stage_df[stage_df["opponent_team"].isin(selected_opponent)]

    # Batter filter moved here (main panel, above the summary table) instead
    # of the sidebar — the sidebar stack was tall enough that this dropdown's
    # menu rendered partially off-screen with nowhere to open. This also
    # co-locates it with the new "click a row to filter" behaviour below.
    grouped_all = stage_df.groupby("player_id").agg(
        player_name=("player_name", "first"),
    ).reset_index()
    batter_options = sorted(grouped_all["player_name"].dropna().tolist())
    if "filter_batter" in st.session_state and st.session_state["filter_batter"] not in (
        ["All batters"] + batter_options
    ):
        st.session_state["filter_batter"] = "All batters"

    # Snapshot before the batter filter narrows things further — used for
    # player-vs-population comparisons later in this function.
    population_df = stage_df

    st.subheader("Batting summary (filtered)")
    selected_batter_name = st.selectbox(
        "Batter (applies to whole page) \u2014 or click a row below",
        options=["All batters"] + batter_options,
        index=0,
        key="filter_batter",
    )

    filtered = stage_df.copy()
    selected_batter_id = None
    if selected_batter_name != "All batters":
        batter_row = grouped_all[grouped_all["player_name"] == selected_batter_name].iloc[0]
        selected_batter_id = batter_row["player_id"]
        filtered = filtered[filtered["player_id"] == selected_batter_id]

    if filtered.empty:
        st.warning("No batting records match the current filters.")
        return

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

    ROWS_VISIBLE = 10
    TABLE_HEIGHT = (ROWS_VISIBLE + 1) * 35 + 3

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
        clicked_name = summary_table.iloc[clicked_idx]["player_name"]
        if st.session_state.get("filter_batter") != clicked_name:
            st.session_state["filter_batter"] = clicked_name
            st.rerun()

    if selected_batter_id is not None:
        selected_row = grouped[grouped["player_id"] == selected_batter_id].iloc[0]

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

        st.subheader("Ball\u2011segment breakdown (deliveries)")

        deliveries_df = get_deliveries_for_batter(str(selected_batter_id))
        if deliveries_df.empty:
            st.info("No deliveries found for this batter.")
        else:
            deliveries_df = deliveries_df.copy()
            deliveries_df["wides"] = deliveries_df["wides"].fillna(0)
            deliveries_df["legal_ball"] = deliveries_df["wides"] == 0

            seg_df = deliveries_df.copy()
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

            seg = seg_df.groupby("segment", observed=True).agg(
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

            # ---------- Batting vs bowling style (deliveries) ----------
            st.subheader("Batting vs bowling style (deliveries)")

            style_lookup = get_player_style()[["player_id", "bowl_style"]].rename(
                columns={"player_id": "bowler_id"}
            )
            style_deliveries = deliveries_df.merge(style_lookup, on="bowler_id", how="left")
            style_deliveries["bowl_style"] = style_deliveries["bowl_style"].fillna("Unknown")

            legal = style_deliveries[style_deliveries["wides"] == 0].copy()
            by_style = legal.groupby("bowl_style").agg(
                runs=("batter_runs", "sum"),
                balls=("bowler_id", "count"),
                dismissals=("dismissal_type", lambda x: x.notna().sum()),
                fours=("description", lambda x: x.str.contains("FOUR", case=False, na=False).sum()),
                sixes=("description", lambda x: x.str.contains("SIX", case=False, na=False).sum()),
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

        # ---------- Match-by-match batting ----------
        st.subheader("Match-by-match batting")

        match_df = filtered.copy()
        match_df["boundaries"] = match_df["fours"].fillna(0) + match_df["sixes"].fillna(0)
        match_df["match_name"] = match_df["match_type"].astype(str) + " v " + match_df["opponent_team"].astype(str)
        match_df["SR"] = match_df.apply(
            lambda r: 100 * r["runs"] / r["balls_faced"] if r["balls_faced"] and r["balls_faced"] > 0 else None,
            axis=1,
        )

        match_display = match_df[
            ["day_1_start", "match_name", "bat_position", "runs", "balls_faced", "SR", "boundaries"]
        ].copy()
        match_display = match_display.sort_values("day_1_start", ascending=False)
        match_display["day_1_start"] = pd.to_datetime(match_display["day_1_start"]).dt.strftime("%d %b %Y")
        match_display["SR"] = match_display["SR"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "\u2013"
        )
        match_display = match_display.rename(columns={
            "day_1_start": "Date", "match_name": "Match", "bat_position": "Pos",
            "balls_faced": "BF", "boundaries": "Boundaries",
        })

        st.dataframe(match_display, width='stretch', hide_index=True)

        st.subheader("Dismissal type distribution \u2014 player vs population")

        pop_innings = population_df

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
            matches_df = get_matches()
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
                    st.session_state["selected_highlight_id"] = default_id

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

ROW_DIVIDER = "<hr style='margin:2px 0;border:none;border-top:1px solid #333;'>"


def bowler_style_tab():
    st.header("Bowler Style")
    st.caption(
        "Identify bowlers with missing style data, and set pace/spin, "
        "bowling hand, and bowling style directly against each bowler. "
        "Make changes across as many bowlers as you like, then click "
        "'Save all changes' once to write everything in a single batch."
    )

    deliveries_df = get_bowling_deliveries()
    if deliveries_df.empty:
        st.info("No delivery data available.")
        return

    matches_df = get_matches()
    matches_df["day_1_start"] = pd.to_datetime(matches_df["day_1_start"])

    deliveries_df = deliveries_df.merge(
        matches_df[["match_id", "grade"]], on="match_id", how="left"
    )

    style_df = get_player_style()

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

    pace_spin_opts = sorted(set(PACE_SPIN_OPTIONS) | set(style_df["pace_spin"].dropna().unique().tolist()))
    bowl_hand_opts = sorted(set(BOWL_HAND_OPTIONS) | set(style_df["bowl_hand"].dropna().unique().tolist()))
    bowl_style_opts = sorted(set(BOWL_STYLE_OPTIONS) | set(style_df["bowl_style"].dropna().unique().tolist()))

    if (
        "selected_bowler_id" not in st.session_state
        or st.session_state["selected_bowler_id"] not in bowler_summary["bowler_id"].values
    ):
        st.session_state["selected_bowler_id"] = bowler_summary.iloc[0]["bowler_id"]

    def _dropdown_index(options, current_value):
        full_options = ["\u2013"] + options
        if pd.notna(current_value) and current_value in options:
            return full_options.index(current_value)
        return 0

    PANEL_HEIGHT = 560
    list_col, highlight_col = st.columns([2, 3])

    with list_col:
        save_clicked = st.button("\U0001F4BE Save all changes", key="save_all_styles", type="primary")

        with st.container(height=PANEL_HEIGHT):
            for _, row in bowler_summary.iterrows():
                bid = row["bowler_id"]
                st.markdown(f"**{row['bowler_name']}** \u2014 {int(row['balls'])} balls")

                c_ps, c_hand, c_style, c_select = st.columns([1, 1, 1, 1])
                with c_ps:
                    st.selectbox(
                        "Pace/Spin", ["\u2013"] + pace_spin_opts,
                        index=_dropdown_index(pace_spin_opts, row.get("pace_spin")),
                        key=f"ps_{bid}", label_visibility="collapsed",
                    )
                with c_hand:
                    st.selectbox(
                        "Hand", ["\u2013"] + bowl_hand_opts,
                        index=_dropdown_index(bowl_hand_opts, row.get("bowl_hand")),
                        key=f"hand_{bid}", label_visibility="collapsed",
                    )
                with c_style:
                    st.selectbox(
                        "Style", ["\u2013"] + bowl_style_opts,
                        index=_dropdown_index(bowl_style_opts, row.get("bowl_style")),
                        key=f"style_{bid}", label_visibility="collapsed",
                    )
                with c_select:
                    if st.button("\u25b6", key=f"sel_{bid}", help="Show highlights"):
                        st.session_state["selected_bowler_id"] = bid
                st.markdown(ROW_DIVIDER, unsafe_allow_html=True)

        if save_clicked:
            changed = []
            for _, row in bowler_summary.iterrows():
                bid = row["bowler_id"]
                ps_val = st.session_state.get(f"ps_{bid}", "\u2013")
                hand_val = st.session_state.get(f"hand_{bid}", "\u2013")
                style_val = st.session_state.get(f"style_{bid}", "\u2013")

                new_pace_spin = None if ps_val == "\u2013" else ps_val
                new_bowl_hand = None if hand_val == "\u2013" else hand_val
                new_bowl_style = None if style_val == "\u2013" else style_val

                orig_pace_spin = row.get("pace_spin") if pd.notna(row.get("pace_spin")) else None
                orig_bowl_hand = row.get("bowl_hand") if pd.notna(row.get("bowl_hand")) else None
                orig_bowl_style = row.get("bowl_style") if pd.notna(row.get("bowl_style")) else None

                if (new_pace_spin, new_bowl_hand, new_bowl_style) != (orig_pace_spin, orig_bowl_hand, orig_bowl_style):
                    changed.append({
                        "player_id": str(bid),
                        "pace_spin": new_pace_spin,
                        "bowl_hand": new_bowl_hand,
                        "bowl_style": new_bowl_style,
                    })

            if not changed:
                st.info("No changes to save.")
            else:
                try:
                    with conn.session as s:
                        s.execute(
                            text(
                                """
                                INSERT INTO player_style (player_id, pace_spin, bowl_hand, bowl_style)
                                VALUES (:player_id, :pace_spin, :bowl_hand, :bowl_style)
                                ON CONFLICT (player_id) DO UPDATE SET
                                    pace_spin = EXCLUDED.pace_spin,
                                    bowl_hand = EXCLUDED.bowl_hand,
                                    bowl_style = EXCLUDED.bowl_style
                                """
                            ),
                            changed,
                        )
                        s.commit()
                    st.success(f"Saved {len(changed)} change(s).")
                    get_player_style.clear()
                except Exception as e:
                    st.error(f"Bulk save failed: {e}")

    with highlight_col:
        selected_row = bowler_summary[
            bowler_summary["bowler_id"] == st.session_state["selected_bowler_id"]
        ].iloc[0]
        st.subheader(f"Highlights \u2014 {selected_row['bowler_name']}")

        highlights_df = get_highlights()
        bowler_highlights = highlights_df[
            highlights_df["bowler_id"] == str(selected_row["bowler_id"])
        ].copy()

        if bowler_highlights.empty:
            st.info("No highlights available for this bowler.")
            return

        bowler_highlights = bowler_highlights.merge(
            matches_df[["match_id", "day_1_start", "home_team"]], on="match_id", how="left"
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

        st.caption(f"{len(bh_sorted)} highlights \u2014 tap \u25b6 to play")

        list_height = 200
        with st.container(height=list_height):
            for _, hrow in bh_sorted.iterrows():
                hid = hrow["highlight_id"]
                txt_col, btn_col = st.columns([5, 1])
                with txt_col:
                    date_str = (
                        hrow["day_1_start"].strftime("%d %b %Y")
                        if pd.notna(hrow.get("day_1_start"))
                        else ""
                    )
                    home_team = hrow.get("home_team") or ""
                    st.markdown(
                        f"**{hrow.get('batter', '')}** \u2014 {hrow.get('highlight_type', '')}  \n"
                        f"{hrow.get('description', '')}  \n"
                        f"<span style='color:gray;font-size:0.8em'>{date_str} \u00b7 {home_team}</span>",
                        unsafe_allow_html=True,
                    )
                with btn_col:
                    if st.button("\u25b6", key=f"bowlerhl_play_{hid}"):
                        st.session_state["selected_bowler_highlight_id"] = hid
                st.markdown(ROW_DIVIDER, unsafe_allow_html=True)

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
