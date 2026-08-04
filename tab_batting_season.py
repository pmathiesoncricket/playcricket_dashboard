import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy.exc import OperationalError, TimeoutError as SATimeoutError

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

# Palette: maroon family (base 115,23,63 = #73173F) for "lower value" outcomes,
# plus the two requested blues for the standout/boundary outcomes -- gives a
# clear low-to-high visual progression across each 5-category stacked bar.
MAROON_DARK = "#450E26"    # darker shade of base maroon
MAROON_BASE = "#73173F"    # (115, 23, 63) as requested
MAROON_LIGHT = "#A46882"   # lighter shade of base maroon
BLUE_1 = "#0739BE"
BLUE_2 = "#0093E0"

RICH_PALETTE = [MAROON_DARK, MAROON_BASE, MAROON_LIGHT, BLUE_1, BLUE_2]

ROW_HEIGHT_PX = 32  # vertical space per row -- scales chart height with row count
MIN_CHART_HEIGHT = 500

TEAM_ROW_LABEL = "TEAM (all batters)"
UNKNOWN_LABEL = "Unknown"


def _safe_fetch(fetch_fn, *args, label="data", **kwargs):
    """Wraps a db.py fetch call so a transient DB/connection-pool issue
    (e.g. sqlalchemy TimeoutError) shows a friendly warning instead of
    crashing the whole app. db.py itself now also retries pool timeouts
    with backoff before raising, so this is a second line of defence."""
    try:
        return fetch_fn(*args, **kwargs)
    except (OperationalError, SATimeoutError) as exc:
        st.warning(
            f"Could not load {label} right now (database connection pool busy). "
            f"Try refreshing in a few seconds. ({type(exc).__name__})"
        )
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001 -- last-resort guard so one section's
        # failure doesn't take down the rest of the page.
        st.warning(f"Could not load {label}: {exc}")
        return pd.DataFrame()


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
    """Legal balls only: exclude wides AND no-balls."""
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
    return None  # ignore rarities (5s, etc.)


def _pct_row_chart(pct_df, count_df, category_col, value_cols, title, order_labels, y_order):
    """100%-stacked horizontal bar chart: percent inside segments, absolute
    count in tooltip, richer palette, height scales with number of rows."""
    melt = pct_df.melt(id_vars=[category_col], value_vars=value_cols,
                        var_name="outcome", value_name="pct")
    counts = count_df.melt(id_vars=[category_col], value_vars=value_cols,
                            var_name="outcome", value_name="count")
    melt = melt.merge(counts, on=[category_col, "outcome"])
    melt["outcome"] = pd.Categorical(melt["outcome"], categories=order_labels, ordered=True)

    chart_height = max(MIN_CHART_HEIGHT, ROW_HEIGHT_PX * len(y_order) + 160)

    fig = px.bar(
        melt, x="pct", y=category_col, color="outcome", orientation="h",
        category_orders={category_col: y_order, "outcome": order_labels},
        text=melt["pct"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) and x >= 3 else ""),
        custom_data=["count"],
        color_discrete_sequence=RICH_PALETTE,
        title=title,
        height=chart_height,
    )
    fig.update_traces(
        textposition="inside",
        hovertemplate="%{y}<br>%{data.name}: %{x:.1f}%% (n=%{customdata[0]})<extra></extra>",
    )
    fig.update_layout(
        barmode="stack", xaxis_title="Percent", yaxis_title=None,
        legend_title_text=category_col.replace("_", " ").title(),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(automargin=True)
    return fig


def _one_level_breakdown(bd, group_col):
    """Aggregate legal deliveries by a single grouping column (pace_spin OR
    bowl_style), always including an 'Unknown' row/bucket for any bowler
    that couldn't be matched to a player_style record (or had a null value
    for that specific column)."""
    g = bd.groupby(group_col, dropna=False).agg(
        runs=("batter_runs", "sum"),
        balls=("batter_runs", "count"),
        dismissals=("dismissal_type", lambda x: x.notna().sum()),
        fours=("description", lambda x: x.str.contains("FOUR", case=False, na=False).sum()),
        sixes=("description", lambda x: x.str.contains("SIX", case=False, na=False).sum()),
    ).reset_index().rename(columns={group_col: "category"})
    g["average"] = g.apply(lambda r: r["runs"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
    g["SR"] = g.apply(lambda r: 100 * r["runs"] / r["balls"] if r["balls"] > 0 else None, axis=1)
    g["BPD"] = g.apply(lambda r: r["balls"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
    g["boundaries"] = g["fours"] + g["sixes"]
    g["BPB"] = g.apply(lambda r: r["balls"] / r["boundaries"] if r["boundaries"] > 0 else None, axis=1)
    return g.sort_values("runs", ascending=False).reset_index(drop=True)


def _bowling_type_breakdowns(legal_deliveries, style_lookup):
    """Returns two SEPARATE dataframes:
      1. Breakdown by pace_spin (Pace / Spin / Unknown)
      2. Breakdown by bowl_style (each specific style / Unknown)
    Both are computed from the SAME merged delivery set, but reported as two
    distinct tables so the bowl_style detail isn't buried underneath the
    pace_spin rows. Any bowler that doesn't match a player_style row at all,
    or has a null value in either column specifically, falls into an
    'Unknown' bucket for that column so no deliveries are silently dropped."""
    bd = legal_deliveries.merge(style_lookup, on="bowler_id", how="left")
    bd["pace_spin"] = bd["pace_spin"].fillna(UNKNOWN_LABEL)
    bd["pace_spin"] = bd["pace_spin"].replace("", UNKNOWN_LABEL)
    bd["bowl_style"] = bd["bowl_style"].fillna(UNKNOWN_LABEL)
    bd["bowl_style"] = bd["bowl_style"].replace("", UNKNOWN_LABEL)

    pace_spin_df = _one_level_breakdown(bd, "pace_spin")
    bowl_style_df = _one_level_breakdown(bd, "bowl_style")
    return pace_spin_df, bowl_style_df


def _render_breakdown_table(df, category_label):
    disp = df.copy()
    for col, dp in [("average", 2), ("SR", 0), ("BPD", 0), ("BPB", 1)]:
        disp[col] = disp[col].apply(lambda x, dp=dp: _fmt(x, dp))
    st.dataframe(
        disp[["category", "runs", "average", "balls", "SR", "BPD", "BPB"]]
        .rename(columns={
            "category": category_label, "runs": "Runs", "average": "Average", "balls": "BF",
        }),
        width="stretch", hide_index=True,
    )


def batting_season_tab():
    st.header("Batting Season Report")

    # ---------------- Top filters (lightweight only) ----------------
    # Season/team options are derived from get_matches() only, NOT from
    # get_batting_innings()/get_deliveries_for_batters()/get_highlights().
    # Those heavier queries are deferred until BOTH a season and a team are
    # selected below -- there's no meaningful report without both, so there
    # is no reason to hit the DB for them (and hold a connection open) on
    # every page load / rerun before the user has picked anything.
    matches_df = _safe_fetch(get_matches, label="matches")
    if matches_df.empty:
        st.info("No match data available.")
        return

    matches_df = add_season_column(matches_df, "day_1_start")
    season_options = (
        matches_df["season"].dropna().drop_duplicates().sort_values(ascending=False).tolist()
    )

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        selected_season = st.multiselect("Season", season_options, key="season_report_season")

    season_matches = matches_df[matches_df["season"].isin(selected_season)] if selected_season else matches_df
    team_options = sorted(
        pd.concat([season_matches["home_team"], season_matches["away_team"]]).dropna().unique().tolist()
    )
    with fcol2:
        selected_team = st.multiselect("Team (batting team)", team_options, key="season_report_team")

    if not selected_season or not selected_team:
        st.info("Select at least one season AND at least one team to load the batting season report.")
        return

    # ---------------- Heavier fetches (only once season + team chosen) ----------------
    batting_df = _safe_fetch(get_batting_innings, label="batting innings")
    if batting_df.empty:
        st.info("No batting data available.")
        return

    batting_df = add_season_column(batting_df, "day_1_start")

    # Fetch player_style ONCE for the whole tab and reuse everywhere below
    # (bowling-type breakdown + highlights).
    style_lookup_all = _safe_fetch(get_player_style, label="player style")

    stage_df = batting_df[batting_df["season"].isin(selected_season)]
    stage_df = stage_df[stage_df["team"].isin(selected_team)]

    if stage_df.empty:
        st.warning("No batting records match the selected season/team.")
        return

    # ---------------- Batter summary table ----------------
    st.subheader("Batter summary")
    summary = _summary_table(stage_df)

    all_batter_ids = tuple(summary["player_id"].dropna().unique().tolist())
    deliveries_df = _safe_fetch(get_deliveries_for_batters, all_batter_ids, label="deliveries")
    if not deliveries_df.empty:
        deliveries_df = deliveries_df.merge(
            matches_df[["match_id", "grade", "match_type", "day_1_start", "season"]], on="match_id", how="left"
        )
        deliveries_df = deliveries_df[deliveries_df["season"].isin(selected_season)]
        deliveries_df = deliveries_df[deliveries_df["batter_id"].isin(stage_df["player_id"])]

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

    # =========================================================
    # Percentage-breakdown charts (outcome / runs / dismissals)
    # =========================================================
    st.markdown("### Ball outcome & dismissal breakdowns")
    st.caption(
        "Filters below apply per-innings: selecting positions 1-6 only includes "
        "innings where the batter actually batted in one of those positions."
    )

    fchcol1, fchcol2 = st.columns(2)

    pos_options = sorted(stage_df["bat_position"].dropna().unique().astype(int).tolist())
    with fchcol1:
        selected_positions = st.multiselect(
            "Batting position (per innings)", pos_options,
            default=pos_options, key="season_report_batpos_filter",
        )
    with fchcol2:
        min_bf = st.number_input(
            "Minimum balls faced (across qualifying innings)", min_value=0, value=MIN_BALLS, step=5,
            key="season_report_minbf_filter",
        )

    if selected_positions:
        pos_filtered_innings = stage_df[stage_df["bat_position"].isin(selected_positions)]
    else:
        pos_filtered_innings = stage_df.iloc[0:0]

    innings_keys = pos_filtered_innings[["match_id", "innings_id", "player_id"]].drop_duplicates()

    pos_summary = _summary_table(pos_filtered_innings) if not pos_filtered_innings.empty else pd.DataFrame(
        columns=["player_id", "player_name", "team", "balls", "runs"]
    )

    qualified_ids = pos_summary[pos_summary["balls"] >= min_bf].sort_values("balls", ascending=False)
    qualified_order = (
        qualified_ids["team"].astype(str) + " | " + qualified_ids["player_name"].astype(str)
    ).tolist()

    if legal_df.empty or innings_keys.empty:
        st.info("No delivery-level data available for the current filters.")
    else:
        pos_legal_df = legal_df.merge(
            innings_keys.rename(columns={"player_id": "batter_id"}),
            on=["match_id", "innings_id", "batter_id"], how="inner",
        )
        pos_legal_df = pos_legal_df.merge(
            summary[["player_id", "player_name", "team"]],
            left_on="batter_id", right_on="player_id", how="left",
        )

        qualified_chart_df = pos_legal_df[pos_legal_df["batter_id"].isin(qualified_ids["player_id"])].copy()
        qualified_chart_df["row_label"] = (
            qualified_chart_df["team"].astype(str) + " | " + qualified_chart_df["player_name"].astype(str)
        )

        team_chart_df = legal_df.copy()
        team_chart_df["row_label"] = TEAM_ROW_LABEL

        y_order_with_team = [TEAM_ROW_LABEL] + list(reversed(qualified_order))

        if qualified_order:
            # ---- Chart 1: balls faced by outcome ----
            st.subheader("Balls faced by outcome (legal balls only)")
            for d in (qualified_chart_df, team_chart_df):
                d["outcome"] = d["batter_runs"].apply(_outcome_bucket)

            combined1 = pd.concat([team_chart_df, qualified_chart_df], ignore_index=True)
            outc = combined1.dropna(subset=["outcome"]).groupby(["row_label", "outcome"]).size().reset_index(name="n")
            outc_totals = outc.groupby("row_label")["n"].sum().rename("total")
            outc = outc.merge(outc_totals, on="row_label")
            outc["pct"] = 100 * outc["n"] / outc["total"]
            pct_wide = outc.pivot(index="row_label", columns="outcome", values="pct").reindex(columns=OUTCOME_ORDER).fillna(0).reset_index()
            cnt_wide = outc.pivot(index="row_label", columns="outcome", values="n").reindex(columns=OUTCOME_ORDER).fillna(0).reset_index()
            fig1 = _pct_row_chart(pct_wide, cnt_wide, "row_label", OUTCOME_ORDER,
                                   "Balls faced by outcome", OUTCOME_ORDER, y_order_with_team)
            st.plotly_chart(fig1, width="stretch")

            # ---- Chart 2: runs scored breakdown ----
            st.subheader("Runs scored by shot type")
            combined2 = pd.concat([team_chart_df, qualified_chart_df], ignore_index=True)
            scoring = combined2[combined2["batter_runs"].isin([1, 2, 3, 4, 6])].copy()
            scoring["shot"] = scoring["batter_runs"].astype(int).astype(str)
            runs_agg = scoring.groupby(["row_label", "shot"])["batter_runs"].sum().reset_index(name="runs_from_shot")
            runs_totals = runs_agg.groupby("row_label")["runs_from_shot"].sum().rename("total")
            runs_agg = runs_agg.merge(runs_totals, on="row_label")
            runs_agg["pct"] = 100 * runs_agg["runs_from_shot"] / runs_agg["total"]
            pct_wide2 = runs_agg.pivot(index="row_label", columns="shot", values="pct").reindex(columns=RUN_SHOT_ORDER).fillna(0).reset_index()
            cnt_wide2 = runs_agg.pivot(index="row_label", columns="shot", values="runs_from_shot").reindex(columns=RUN_SHOT_ORDER).fillna(0).reset_index()
            fig2 = _pct_row_chart(pct_wide2, cnt_wide2, "row_label", RUN_SHOT_ORDER,
                                   "Runs scored by shot type", RUN_SHOT_ORDER, y_order_with_team)
            st.plotly_chart(fig2, width="stretch")

            # ---- Chart 3: dismissal breakdown ----
            st.subheader("Dismissal type breakdown")
            dismiss_all = deliveries_df[deliveries_df["dismissal_type"].isin(DISMISSAL_TYPES)].copy()
            dismiss_all_pos = dismiss_all.merge(
                innings_keys.rename(columns={"player_id": "batter_id"}),
                on=["match_id", "innings_id", "batter_id"], how="inner",
            )
            dismiss_all_pos = dismiss_all_pos.merge(
                summary[["player_id", "player_name", "team"]], left_on="batter_id", right_on="player_id", how="left"
            )
            dismiss_qual = dismiss_all_pos[dismiss_all_pos["batter_id"].isin(qualified_ids["player_id"])].copy()
            dismiss_qual["row_label"] = dismiss_qual["team"].astype(str) + " | " + dismiss_qual["player_name"].astype(str)

            dismiss_team = dismiss_all.copy()
            dismiss_team["row_label"] = TEAM_ROW_LABEL

            combined3 = pd.concat([dismiss_team, dismiss_qual], ignore_index=True)
            dis_agg = combined3.groupby(["row_label", "dismissal_type"]).size().reset_index(name="n")
            dis_totals = dis_agg.groupby("row_label")["n"].sum().rename("total")
            dis_agg = dis_agg.merge(dis_totals, on="row_label")
            dis_agg["pct"] = 100 * dis_agg["n"] / dis_agg["total"]
            pct_wide3 = dis_agg.pivot(index="row_label", columns="dismissal_type", values="pct").reindex(columns=DISMISSAL_TYPES).fillna(0).reset_index()
            cnt_wide3 = dis_agg.pivot(index="row_label", columns="dismissal_type", values="n").reindex(columns=DISMISSAL_TYPES).fillna(0).reset_index()
            fig3 = _pct_row_chart(pct_wide3, cnt_wide3, "row_label", DISMISSAL_TYPES,
                                   "Dismissal type breakdown", DISMISSAL_TYPES, y_order_with_team)
            st.plotly_chart(fig3, width="stretch")
        else:
            st.info("No batters meet the current position / minimum balls faced filters.")

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
        elif style_lookup_all.empty:
            st.info("Bowling style lookup unavailable right now.")
        else:
            style_lookup = style_lookup_all[["player_id", "pace_spin", "bowl_style"]].rename(
                columns={"player_id": "bowler_id"}
            )
            legal_b_deliveries = _legal_deliveries(b_deliveries)
            pace_spin_df, bowl_style_df = _bowling_type_breakdowns(legal_b_deliveries, style_lookup)

            st.markdown("*By pace / spin*")
            _render_breakdown_table(pace_spin_df, "Pace / Spin")

            st.markdown("*By bowl style*")
            _render_breakdown_table(bowl_style_df, "Bowl style")

            st.caption(
                "Both tables are computed from the same deliveries faced, just grouped "
                "differently -- 'Unknown' captures any bowler with no matching (or "
                "blank) value for that specific column in player_style."
            )

        # ---- Highlights viewer (mirrors tab_batting.py pattern) ----
        st.markdown(f"**Highlights \u2014 {selected_batter}**")
        highlights_df = _safe_fetch(get_highlights, label="highlights")
        if highlights_df.empty:
            st.info("No highlights available (or they could not be loaded \u2014 see warning above, if any).")
        else:
            highlights_df = highlights_df.merge(
                matches_df[["match_id", "grade", "match_type", "day_1_start", "season"]], on="match_id", how="left"
            )
            # Reuse the same style_lookup_all fetched once at the top of the
            # tab, instead of calling get_player_style() again here.
            if not style_lookup_all.empty:
                style_lookup_h = style_lookup_all[["player_id", "pace_spin"]].rename(columns={"player_id": "bowler_id"})
                highlights_df = highlights_df.merge(style_lookup_h, on="bowler_id", how="left")
                highlights_df["pace_spin"] = highlights_df["pace_spin"].fillna(UNKNOWN_LABEL)
            else:
                highlights_df["pace_spin"] = UNKNOWN_LABEL

            h = highlights_df[highlights_df["batter_id"] == str(b_id)]
            if selected_season and "season" in h.columns:
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
