import streamlit as st
import pandas as pd
import plotly.express as px

from db import (
    get_batting_innings, get_deliveries_for_batters, get_highlights,
    get_matches, get_player_style,
)
from helpers import add_season_column, MAROON, MAROON_SHADES

MIN_BALLS = 20
DASH = "\u2013"

# Over ranges are inclusive, matching deliveries.over as-is (40-over cricket)
PHASES = [
    ("Powerplay (0-9)", 0, 9),
    ("Middle (10-29)", 10, 29),
    ("L10 (30-40)", 30, 40),
]
DISMISSAL_TYPES = ["Bowled", "LBW", "Caught", "Run Out", "Stumped"]
OUTCOME_ORDER = ["0", "1", "2/3", "4", "6"]
RUN_SHOT_ORDER = ["1", "2", "3", "4", "6"]


def _phase_of(over):
    for label, lo, hi in PHASES:
        if lo <= over <= hi:
            return label
    return None


def _fmt(x, dp=2):
    return f"{x:.{dp}f}" if pd.notna(x) else DASH


def _summary_table(df):
    """df = player_innings rows. Returns core batting metrics per player."""
    g = df.groupby("player_id").agg(
        player_name=("player_name", "first"),
        team=("team", "first"),
        innings=("runs", "count"),
        runs=("runs", "sum"),
        balls=("balls_faced", "sum"),
        dismissals=("dismissal_type", lambda x: x.notna().sum()),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
    ).reset_index()
    g["average"] = g.apply(lambda r: r["runs"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
    g["SR"] = g.apply(lambda r: 100 * r["runs"] / r["balls"] if r["balls"] > 0 else None, axis=1)
    g["BPD"] = g.apply(lambda r: r["balls"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
    g["boundaries"] = g["fours"] + g["sixes"]
    g["BPB"] = g.apply(lambda r: r["balls"] / r["boundaries"] if r["boundaries"] > 0 else None, axis=1)
    return g


def _legal_deliveries(deliveries_df):
    """Legal balls only: exclude wides AND no-balls, per season-report definition."""
    d = deliveries_df.copy()
    d["wides"] = d["wides"].fillna(0)
    d["no_balls"] = d["no_balls"].fillna(0)
    return d[(d["wides"] == 0) & (d["no_balls"] == 0)]


def _outcome_bucket(runs):
    if runs == 0:
        return "0"
    if runs == 1:
        return "1"
    if runs in (2, 3):
        return "2/3"
    if runs == 4:
        return "4"
    if runs == 6:
        return "6"
    return None  # ignore 5s/odd extras-only cases


def _pct_row_chart(pct_df, count_df, category_col, value_cols, title, order_labels, y_order):
    """Builds a 100%-stacked horizontal bar chart matching the reference formatting
    (row percents inside segments, category on y-axis, legend to the right)."""
    melt = pct_df.melt(id_vars=[category_col], value_vars=value_cols,
                        var_name="outcome", value_name="pct")
    counts = count_df.melt(id_vars=[category_col], value_vars=value_cols,
                            var_name="outcome", value_name="count")
    melt = melt.merge(counts, on=[category_col, "outcome"])
    melt["outcome"] = pd.Categorical(melt["outcome"], categories=order_labels, ordered=True)

    fig = px.bar(
        melt, x="pct", y=category_col, color="outcome", orientation="h",
        category_orders={category_col: y_order, "outcome": order_labels},
        text=melt["pct"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) and x >= 3 else ""),
        custom_data=["count"],
        color_discrete_sequence=MAROON_SHADES,
        title=title,
    )
    fig.update_traces(
        textposition="inside",
        hovertemplate="%{y}<br>%{data.name}: %{x:.1f}% (n=%{customdata[0]})<extra></extra>",
    )
    fig.update_layout(barmode="stack", xaxis_title="Percent", yaxis_title=None,
                       legend_title_text=category_col.replace("_", " ").title())
    fig.update_xaxes(ticksuffix="%")
    return fig


def batting_season_tab():
    st.header("Batting Season Report")

    batting_df = get_batting_innings()
    if batting_df.empty:
        st.info("No batting data available.")
        return

    batting_df = add_season_column(batting_df, "day_1_start")
    matches_df = get_matches()

    # ---------------- Top filters ----------------
    fcol1, fcol2 = st.columns(2)
    season_options = (
        batting_df["season"].dropna().drop_duplicates().sort_values(ascending=False).tolist()
    )
    with fcol1:
        selected_season = st.multiselect("Season", season_options, key="season_report_season")
    stage_df = batting_df.copy()
    if selected_season:
        stage_df = stage_df[stage_df["season"].isin(selected_season)]

    team_options = sorted(stage_df["team"].dropna().unique().tolist())
    with fcol2:
        selected_team = st.multiselect("Team (batting team)", team_options, key="season_report_team")
    if selected_team:
        stage_df = stage_df[stage_df["team"].isin(selected_team)]

    if stage_df.empty:
        st.warning("No batting records match the selected season/team.")
        return

    # ---------------- Batter summary table ----------------
    st.subheader("Batter summary")
    summary = _summary_table(stage_df)

    legal_lookup_ids = tuple(summary["player_id"].dropna().unique().tolist())
    deliveries_df = get_deliveries_for_batters(legal_lookup_ids)
    if not deliveries_df.empty:
        deliveries_df = deliveries_df.merge(
            matches_df[["match_id", "grade", "match_type", "day_1_start"]], on="match_id", how="left"
        )
        deliveries_df = add_season_column(deliveries_df, "day_1_start")
        if selected_season:
            deliveries_df = deliveries_df[deliveries_df["season"].isin(selected_season)]
        if selected_team:
            deliveries_df = deliveries_df[
                deliveries_df["batter_id"].isin(stage_df["player_id"])
            ]

    legal_df = _legal_deliveries(deliveries_df) if not deliveries_df.empty else pd.DataFrame()
    if not legal_df.empty:
        shot_stats = legal_df.groupby("batter_id").agg(
            legal_balls=("batter_runs", "count"),
            scoring_shots=("batter_runs", lambda x: (x > 0).sum()),
        ).reset_index().rename(columns={"batter_id": "player_id"})
        shot_stats["scoring_shot_pct"] = 100 * shot_stats["scoring_shots"] / shot_stats["legal_balls"]
        summary = summary.merge(shot_stats[["player_id", "scoring_shot_pct"]], on="player_id", how="left")
    else:
        summary["scoring_shot_pct"] = None

    display = summary.copy()
    for col, dp in [("average", 2), ("SR", 0), ("BPD", 0), ("BPB", 1), ("scoring_shot_pct", 1)]:
        display[col] = display[col].apply(lambda x, dp=dp: _fmt(x, dp))

    summary_table = display[
        ["player_name", "team", "runs", "average", "balls", "SR", "BPD", "BPB", "scoring_shot_pct"]
    ].rename(columns={
        "player_name": "Batter", "team": "Team", "runs": "Runs", "average": "Average",
        "balls": "BF", "SR": "SR", "BPD": "BPD", "BPB": "BPB", "scoring_shot_pct": "Scoring shot %",
    }).sort_values("Runs", ascending=False).reset_index(drop=True)

    st.dataframe(summary_table, width="stretch", hide_index=True)

    # Batters eligible for the % breakdown charts (20+ legal balls faced)
    qualified_ids = summary[summary["balls"] >= MIN_BALLS].sort_values("runs", ascending=False)
    qualified_order = (
        qualified_ids["team"].astype(str) + " | " + qualified_ids["player_name"].astype(str)
    ).tolist()

    if legal_df.empty:
        st.info("No delivery-level data available for the outcome/runs/dismissal charts.")
    else:
        chart_df = legal_df.merge(
            summary[["player_id", "player_name", "team", "runs"]],
            left_on="batter_id", right_on="player_id", how="left",
        )
        chart_df = chart_df[chart_df["batter_id"].isin(qualified_ids["player_id"])]
        chart_df["row_label"] = chart_df["team"].astype(str) + " | " + chart_df["player_name"].astype(str)

        # ---- Chart 1: balls faced by outcome ----
        st.subheader("Balls faced by outcome (legal balls only)")
        chart_df["outcome"] = chart_df["batter_runs"].apply(_outcome_bucket)
        outc = chart_df.dropna(subset=["outcome"]).groupby(["row_label", "outcome"]).size().reset_index(name="n")
        outc_totals = outc.groupby("row_label")["n"].sum().rename("total")
        outc = outc.merge(outc_totals, on="row_label")
        outc["pct"] = 100 * outc["n"] / outc["total"]
        pct_wide = outc.pivot(index="row_label", columns="outcome", values="pct").reindex(columns=OUTCOME_ORDER).fillna(0).reset_index()
        cnt_wide = outc.pivot(index="row_label", columns="outcome", values="n").reindex(columns=OUTCOME_ORDER).fillna(0).reset_index()
        fig1 = _pct_row_chart(pct_wide, cnt_wide, "row_label", OUTCOME_ORDER,
                               "Balls faced by outcome", OUTCOME_ORDER, list(reversed(qualified_order)))
        st.plotly_chart(fig1, width="stretch")

        # ---- Chart 2: runs scored breakdown ----
        st.subheader("Runs scored by shot type")
        scoring = chart_df[chart_df["batter_runs"].isin([1, 2, 3, 4, 6])].copy()
        scoring["shot"] = scoring["batter_runs"].astype(int).astype(str)
        scoring["runs_from_shot"] = scoring["batter_runs"]
        runs_agg = scoring.groupby(["row_label", "shot"])["runs_from_shot"].sum().reset_index()
        runs_totals = runs_agg.groupby("row_label")["runs_from_shot"].sum().rename("total")
        runs_agg = runs_agg.merge(runs_totals, on="row_label")
        runs_agg["pct"] = 100 * runs_agg["runs_from_shot"] / runs_agg["total"]
        pct_wide2 = runs_agg.pivot(index="row_label", columns="shot", values="pct").reindex(columns=RUN_SHOT_ORDER).fillna(0).reset_index()
        cnt_wide2 = runs_agg.pivot(index="row_label", columns="shot", values="runs_from_shot").reindex(columns=RUN_SHOT_ORDER).fillna(0).reset_index()
        fig2 = _pct_row_chart(pct_wide2, cnt_wide2, "row_label", RUN_SHOT_ORDER,
                               "Runs scored by shot type", RUN_SHOT_ORDER, list(reversed(qualified_order)))
        st.plotly_chart(fig2, width="stretch")

        # ---- Chart 3: dismissal breakdown ----
        st.subheader("Dismissal type breakdown")
        dismiss_df = deliveries_df[
            deliveries_df["dismissal_type"].isin(DISMISSAL_TYPES)
            & deliveries_df["batter_id"].isin(qualified_ids["player_id"])
        ].merge(summary[["player_id", "player_name", "team"]], left_on="batter_id", right_on="player_id", how="left")
        dismiss_df["row_label"] = dismiss_df["team"].astype(str) + " | " + dismiss_df["player_name"].astype(str)
        dis_agg = dismiss_df.groupby(["row_label", "dismissal_type"]).size().reset_index(name="n")
        dis_totals = dis_agg.groupby("row_label")["n"].sum().rename("total")
        dis_agg = dis_agg.merge(dis_totals, on="row_label")
        dis_agg["pct"] = 100 * dis_agg["n"] / dis_agg["total"]
        pct_wide3 = dis_agg.pivot(index="row_label", columns="dismissal_type", values="pct").reindex(columns=DISMISSAL_TYPES).fillna(0).reset_index()
        cnt_wide3 = dis_agg.pivot(index="row_label", columns="dismissal_type", values="n").reindex(columns=DISMISSAL_TYPES).fillna(0).reset_index()
        fig3 = _pct_row_chart(pct_wide3, cnt_wide3, "row_label", DISMISSAL_TYPES,
                               "Dismissal type breakdown", DISMISSAL_TYPES, list(reversed(qualified_order)))
        st.plotly_chart(fig3, width="stretch")

    # ---------------- One Day phase report ----------------
    od_df = stage_df[stage_df["match_type"] == "One Day"]
    if not od_df.empty and not deliveries_df.empty:
        st.subheader("One Day \u2014 phase metrics")
        od_deliveries = _legal_deliveries(deliveries_df[deliveries_df["match_type"] == "One Day"])
        od_deliveries["phase"] = od_deliveries["over"].apply(_phase_of)
        od_deliveries = od_deliveries.dropna(subset=["phase"])

        def _phase_metrics(d, group_cols):
            g = d.groupby(group_cols).agg(
                runs=("batter_runs", "sum"),
                balls=("batter_runs", "count"),
                dismissals=("dismissal_type", lambda x: x.notna().sum()),
                fours=("description", lambda x: x.str.contains("FOUR", case=False, na=False).sum()),
                sixes=("description", lambda x: x.str.contains("SIX", case=False, na=False).sum()),
                scoring=("batter_runs", lambda x: (x > 0).sum()),
            ).reset_index()
            g["average"] = g.apply(lambda r: r["runs"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
            g["SR"] = g.apply(lambda r: 100 * r["runs"] / r["balls"] if r["balls"] > 0 else None, axis=1)
            g["BPD"] = g.apply(lambda r: r["balls"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
            g["boundaries"] = g["fours"] + g["sixes"]
            g["BPB"] = g.apply(lambda r: r["balls"] / r["boundaries"] if r["boundaries"] > 0 else None, axis=1)
            g["scoring_shot_pct"] = g.apply(lambda r: 100 * r["scoring"] / r["balls"] if r["balls"] > 0 else None, axis=1)
            return g

        phase_order = [p[0] for p in PHASES]

        st.markdown("**Team level**")
        team_phase = _phase_metrics(od_deliveries, ["phase"])
        team_phase["phase"] = pd.Categorical(team_phase["phase"], categories=phase_order, ordered=True)
        team_phase = team_phase.sort_values("phase")
        tp_disp = team_phase.copy()
        for col, dp in [("average", 2), ("SR", 0), ("BPD", 0), ("BPB", 1), ("scoring_shot_pct", 1)]:
            tp_disp[col] = tp_disp[col].apply(lambda x, dp=dp: _fmt(x, dp))
        st.dataframe(
            tp_disp[["phase", "runs", "average", "balls", "SR", "BPD", "BPB", "scoring_shot_pct"]]
            .rename(columns={"phase": "Phase", "runs": "Runs", "average": "Average", "balls": "BF",
                              "scoring_shot_pct": "Scoring shot %"}),
            width="stretch", hide_index=True,
        )

        st.markdown("**Batter level**")
        bat_phase = od_deliveries.merge(
            summary[["player_id", "player_name"]], left_on="batter_id", right_on="player_id", how="left"
        )
        bp = _phase_metrics(bat_phase, ["player_name", "phase"])
        bp["phase"] = pd.Categorical(bp["phase"], categories=phase_order, ordered=True)
        bp = bp.sort_values(["player_name", "phase"])
        bp_disp = bp.copy()
        for col, dp in [("average", 2), ("SR", 0), ("BPD", 0), ("BPB", 1), ("scoring_shot_pct", 1)]:
            bp_disp[col] = bp_disp[col].apply(lambda x, dp=dp: _fmt(x, dp))
        st.dataframe(
            bp_disp[["player_name", "phase", "runs", "average", "balls", "SR", "BPD", "BPB", "scoring_shot_pct"]]
            .rename(columns={"player_name": "Batter", "phase": "Phase", "runs": "Runs",
                              "average": "Average", "balls": "BF", "scoring_shot_pct": "Scoring shot %"}),
            width="stretch", hide_index=True,
        )
    elif od_df.empty:
        st.caption("No One Day matches in the current season/team selection \u2014 phase report skipped.")

    # ---------------- Individual batter section ----------------
    st.subheader("Individual batter detail")
    batter_names = sorted(stage_df["player_name"].dropna().unique().tolist())
    selected_batter = st.selectbox("Batter (within selected season/team)", ["\u2014 select \u2014"] + batter_names,
                                    key="season_report_batter")

    if selected_batter != "\u2014 select \u2014":
        b_row = summary[summary["player_name"] == selected_batter].iloc[0]
        b_id = b_row["player_id"]
        b_innings = stage_df[stage_df["player_id"] == b_id]
        b_deliveries = deliveries_df[deliveries_df["batter_id"] == b_id] if not deliveries_df.empty else pd.DataFrame()

        st.markdown(f"**Metrics by game type \u2014 {selected_batter}**")
        by_type = _summary_table(b_innings.assign(player_id=b_innings["match_type"]))
        by_type = by_type.rename(columns={"player_id": "match_type"})
        bt_disp = by_type.copy()
        for col, dp in [("average", 2), ("SR", 0), ("BPD", 0), ("BPB", 1)]:
            bt_disp[col] = bt_disp[col].apply(lambda x, dp=dp: _fmt(x, dp))
        st.dataframe(
            bt_disp[["match_type", "runs", "average", "balls", "SR", "BPD", "BPB"]]
            .rename(columns={"match_type": "Match type", "runs": "Runs", "average": "Average", "balls": "BF"}),
            width="stretch", hide_index=True,
        )

        st.markdown(f"**Metrics by bowling type \u2014 {selected_batter}**")
        if b_deliveries.empty:
            st.info("No ball-by-ball data for this batter.")
        else:
            style_lookup = get_player_style()[["player_id", "pace_spin"]].rename(columns={"player_id": "bowler_id"})
            bd = _legal_deliveries(b_deliveries).merge(style_lookup, on="bowler_id", how="left")
            bd["pace_spin"] = bd["pace_spin"].fillna("Unknown")
            by_bowl = bd.groupby("pace_spin").agg(
                runs=("batter_runs", "sum"),
                balls=("batter_runs", "count"),
                dismissals=("dismissal_type", lambda x: x.notna().sum()),
                fours=("description", lambda x: x.str.contains("FOUR", case=False, na=False).sum()),
                sixes=("description", lambda x: x.str.contains("SIX", case=False, na=False).sum()),
            ).reset_index()
            by_bowl["average"] = by_bowl.apply(lambda r: r["runs"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
            by_bowl["SR"] = by_bowl.apply(lambda r: 100 * r["runs"] / r["balls"] if r["balls"] > 0 else None, axis=1)
            by_bowl["BPD"] = by_bowl.apply(lambda r: r["balls"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
            by_bowl["boundaries"] = by_bowl["fours"] + by_bowl["sixes"]
            by_bowl["BPB"] = by_bowl.apply(lambda r: r["balls"] / r["boundaries"] if r["boundaries"] > 0 else None, axis=1)
            bb_disp = by_bowl.copy()
            for col, dp in [("average", 2), ("SR", 0), ("BPD", 0), ("BPB", 1)]:
                bb_disp[col] = bb_disp[col].apply(lambda x, dp=dp: _fmt(x, dp))
            st.dataframe(
                bb_disp[["pace_spin", "runs", "average", "balls", "SR", "BPD", "BPB"]]
                .rename(columns={"pace_spin": "Bowling type", "runs": "Runs", "average": "Average", "balls": "BF"})
                .sort_values("Runs", ascending=False),
                width="stretch", hide_index=True,
            )

        # ---- Highlights viewer (mirrors tab_batting.py pattern) ----
        st.markdown(f"**Highlights \u2014 {selected_batter}**")
        highlights_df = get_highlights()
        if highlights_df.empty:
            st.info("No highlights available.")
        else:
            highlights_df = highlights_df.merge(
                matches_df[["match_id", "grade", "match_type", "day_1_start"]], on="match_id", how="left"
            )
            highlights_df = add_season_column(highlights_df, "day_1_start")
            style_lookup = get_player_style()[["player_id", "pace_spin"]].rename(columns={"player_id": "bowler_id"})
            highlights_df = highlights_df.merge(style_lookup, on="bowler_id", how="left")
            highlights_df["pace_spin"] = highlights_df["pace_spin"].fillna("Unknown")

            h = highlights_df[highlights_df["batter_id"] == str(b_id)]
            if selected_season:
                h = h[h["season"].isin(selected_season)]

            hcol1, hcol2 = st.columns(2)
            with hcol1:
                htype_options = ["All"] + sorted(h["highlight_type"].dropna().unique().tolist())
                sel_htype = st.selectbox("Highlight type", htype_options, key="season_report_htype")
            with hcol2:
                bowl_options = ["All"] + sorted(h["pace_spin"].dropna().unique().tolist())
                sel_bowl = st.selectbox("Bowler type", bowl_options, key="season_report_bowltype")

            if sel_htype != "All":
                h = h[h["highlight_type"] == sel_htype]
            if sel_bowl != "All":
                h = h[h["pace_spin"] == sel_bowl]

            if h.empty:
                st.info("No highlights match the current filters.")
            else:
                h_sorted = h.sort_values(["day_1_start", "innings_number", "over", "ball_number"],
                                          ascending=[False, True, True, True]).reset_index(drop=True)
                for _, row in h_sorted.iterrows():
                    st.markdown(f"{row.get('batter')} vs {row.get('bowler')} \u2014 {row.get('highlight_type')}: {row.get('description')}")
                    url = row.get("highlight_url")
                    if url:
                        st.video(url, autoplay=False)
