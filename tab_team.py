import re
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from db import (
    get_matches, get_innings, get_match_results, get_batting_innings_for_matches,
    get_bowling_innings_for_matches, get_deliveries_for_matches,
    get_wicketkeepers_for_matches, get_player_style,
)
from helpers import add_season_column, cascading_multiselect, MAROON, MAROON_SHADES

DASH = "\u2013"
MAX_MATCHES = 400

PHASES = {
    "One Day": [("1-10", 0, 9), ("11-30", 10, 29), ("31-40", 30, 39)],
    "Two Day": [("1-20", 0, 19), ("21-50", 20, 49), ("51+", 50, 10_000)],
    "T20": [("1-6", 0, 5), ("7-15", 6, 14), ("16-20", 15, 19)],
}
OUTCOME_ORDER = ["Win", "Draw", "Tie", "Loss"]
OUTCOME_COLORS = {"Win": MAROON, "Draw": "#9E3A5D", "Tie": "#C97292", "Loss": "#4A0F29"}
PACE_SPIN_COLORS = {"Pace": "#0739BE", "Spin": "#73173F", "Unknown": "#6b6b6b"}
SHADE_A = "rgba(115,23,63,0.10)"
SHADE_B = "rgba(7,57,190,0.10)"


def _fmt(x, dp=2):
    return f"{x:.{dp}f}" if pd.notna(x) else DASH


def _pct(x):
    return f"{x:.0f}%" if pd.notna(x) else DASH


def _phase_label(match_type, over):
    for label, lo, hi in PHASES.get(match_type, []):
        if lo <= over <= hi:
            return label
    return None


def _phase_order(match_type):
    return [p[0] for p in PHASES.get(match_type, [])]


def _build_initials(names):
    base = {}
    for n in names:
        parts = n.split()
        if len(parts) >= 2:
            ini = (parts[0][0] + parts[-1][0]).upper()
        else:
            ini = n[:2].upper()
        base[n] = ini

    groups = defaultdict(list)
    for n, ini in base.items():
        groups[ini].append(n)

    result = {}
    for ini, ns in groups.items():
        if len(ns) == 1:
            result[ns[0]] = ini
        else:
            for i, n in enumerate(sorted(ns), 1):
                result[n] = f"{ini}{i}"
    return result


def _is_bowler_wicket(dismissal_type):
    if pd.isna(dismissal_type):
        return False
    t = str(dismissal_type).lower()
    return not any(p in t for p in ["run out", "retired", "obstruct"])


def _score_label(runs, wickets, overs):
    runs_i = int(runs) if pd.notna(runs) else 0
    overs_str = f"{overs:.1f}" if pd.notna(overs) else DASH
    if pd.notna(wickets) and int(wickets) >= 10:
        return f"{runs_i} all out ({overs_str} ov)"
    if pd.notna(wickets):
        return f"{int(wickets)}-{runs_i} ({overs_str} ov)"
    return f"{runs_i} ({overs_str} ov)"


def _html_table(df, align=None, center_all=False, bold_before=None, font_size="0.9rem", stretch=False, col_shades=None):
    """Renders a dataframe as a plain HTML table with per-column text
    alignment, optional bold top-border row separators, and optional
    subtle per-column background shading (col_shades: {colname: css_color})
    -- used to visually group "breakdown" columns (e.g. Pace vs Spin,
    v LHB vs v RHB) apart from the headline totals."""
    align = align or {}
    bold_before = bold_before or set()
    col_shades = col_shades or {}
    width_style = "width:100%;" if stretch else ""
    header = "".join(
        f'<th style="text-align:{align.get(c, "left")};padding:4px 8px;border-bottom:1px solid #444;'
        f'font-size:{font_size};background:{col_shades.get(c, "transparent")};">{c}</th>'
        for c in df.columns
    )
    rows_html = []
    for i, (_, row) in enumerate(df.iterrows()):
        border = "border-top:2px solid #888;" if i in bold_before else ""
        cells = "".join(
            f'<td style="text-align:{"center" if center_all else align.get(c, "left")};padding:4px 8px;'
            f'font-size:{font_size};background:{col_shades.get(c, "transparent")};{border}">{row[c]}</td>'
            for c in df.columns
        )
        rows_html.append(f"<tr>{cells}</tr>")
    table_html = (
        f'<table style="border-collapse:collapse;{width_style}">'
        f"<tr>{header}</tr>{''.join(rows_html)}</table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def team_tab():
    st.header("Team Preview")

    matches_df = get_matches()
    if matches_df.empty:
        st.info("No match data available.")
        return
    matches_df = add_season_column(matches_df, "day_1_start")

    st.sidebar.markdown("### Team Preview filters")

    season_options = sorted(matches_df["season"].dropna().unique().tolist(), reverse=True)
    selected_seasons = cascading_multiselect(st.sidebar, "Season (July\u2013June)", season_options, "team_filter_season")

    team_name_options = sorted(
        pd.concat([matches_df["home_team"], matches_df["away_team"]]).dropna().unique().tolist()
    )
    selected_teams = cascading_multiselect(
        st.sidebar, "Team", team_name_options, "team_filter_team", enable_quick_add=True
    )

    match_type_options = sorted(matches_df["match_type"].dropna().unique().tolist())
    selected_match_types = cascading_multiselect(
        st.sidebar, "Match type", match_type_options, "team_filter_match_type",
        default_options=match_type_options,
    )

    if not selected_seasons or not selected_teams:
        st.info("Select at least one Season and Team in the sidebar, then click Apply Filters.")
        return

    current_key = (
        tuple(sorted(selected_seasons)), tuple(sorted(selected_teams)), tuple(sorted(selected_match_types)),
    )
    apply_clicked = st.sidebar.button("Apply Filters", type="primary", key="team_apply_filters")
    if apply_clicked:
        st.session_state["team_applied_key"] = current_key

    if st.session_state.get("team_applied_key") != current_key:
        st.info("Set your Season / Team / Match type filters, then click **Apply Filters**.")
        return

    team_set = set(selected_teams)
    team_matches = matches_df[
        matches_df["season"].isin(selected_seasons)
        & matches_df["match_type"].isin(selected_match_types)
        & (matches_df["home_team"].isin(team_set) | matches_df["away_team"].isin(team_set))
    ].copy()

    if team_matches.empty:
        st.warning("No matches found for the current filters.")
        return

    team_matches["our_team_id"] = np.where(
        team_matches["home_team"].isin(team_set), team_matches["home_team_id"], team_matches["away_team_id"]
    )
    team_matches["opponent_team_id"] = np.where(
        team_matches["home_team"].isin(team_set), team_matches["away_team_id"], team_matches["home_team_id"]
    )
    team_matches["opponent_team"] = np.where(
        team_matches["home_team"].isin(team_set), team_matches["away_team"], team_matches["home_team"]
    )
    team_matches["our_batted_first"] = team_matches["batted_first_team_id"] == team_matches["our_team_id"]
    team_matches["our_won_toss"] = team_matches["toss_winner_team_id"] == team_matches["our_team_id"]

    results = get_match_results()
    team_matches = team_matches.merge(results, on="match_id", how="left")

    def _outcome(row):
        if row["outcome_type"] == "decisive":
            return "Win" if row["winner_team_id"] == row["our_team_id"] else "Loss"
        if row["outcome_type"] == "draw":
            return "Draw"
        if row["outcome_type"] == "tie":
            return "Tie"
        return None

    team_matches["outcome"] = team_matches.apply(_outcome, axis=1)
    team_matches["grade_season"] = team_matches["season"] + " (" + team_matches["grade"].fillna("Unknown").astype(str) + ")"

    unparsed_n = (team_matches["outcome_type"] == "unparsed").sum()
    if unparsed_n:
        st.caption(f"Note: {unparsed_n} match(es) had a result_text that couldn't be matched to either team name and are excluded from win/loss figures.")

    grade_season_pairs = team_matches[["grade", "season"]].drop_duplicates()
    pop_matches = matches_df.merge(grade_season_pairs, on=["grade", "season"], how="inner")
    pop_matches = pop_matches[pop_matches["match_type"].isin(selected_match_types)]

    if len(pop_matches) > MAX_MATCHES or len(team_matches) > MAX_MATCHES:
        st.warning(
            f"This selection resolves to {len(pop_matches)} matches across the relevant grade/season "
            f"combinations, over the {MAX_MATCHES}-match limit for a single Team Preview run. "
            f"Please remove a season and click Apply Filters again."
        )
        return

    team_match_ids = tuple(str(m) for m in team_matches["match_id"].unique())
    pop_match_ids = tuple(str(m) for m in pop_matches["match_id"].unique())

    batting_pi = get_batting_innings_for_matches(pop_match_ids)
    bowling_pi = get_bowling_innings_for_matches(pop_match_ids)
    pop_deliveries = get_deliveries_for_matches(pop_match_ids)
    style_df = get_player_style()

    if pop_deliveries.empty:
        st.warning("No ball-by-ball data available for the current filters.")
        return

    pop_deliveries = pop_deliveries.merge(
        matches_df[["match_id", "match_type"]], on="match_id", how="left"
    )
    bowler_style = style_df[["player_id", "pace_spin"]].rename(
        columns={"player_id": "bowler_id", "pace_spin": "bowler_pace_spin"}
    )
    pop_deliveries = pop_deliveries.merge(bowler_style, on="bowler_id", how="left")
    pop_deliveries["bowler_pace_spin"] = pop_deliveries["bowler_pace_spin"].fillna("Unknown")
    batter_hand_lookup = style_df[["player_id", "batter_hand"]].rename(
        columns={"player_id": "batter_id", "batter_hand": "faced_batter_hand"}
    )
    pop_deliveries = pop_deliveries.merge(batter_hand_lookup, on="batter_id", how="left")

    pop_deliveries["wides"] = pop_deliveries["wides"].fillna(0)
    pop_deliveries["no_balls"] = pop_deliveries["no_balls"].fillna(0)
    pop_deliveries["is_legal"] = (pop_deliveries["wides"] == 0) & (pop_deliveries["no_balls"] == 0)
    pop_deliveries["is_legal_for_batter"] = pop_deliveries["wides"] == 0
    pop_deliveries["is_dismissal_batter"] = (
        pop_deliveries["dismissal_type"].notna()
        & (pop_deliveries["dismissed_player_id"] == pop_deliveries["batter_id"])
    )
    pop_deliveries["is_bowler_wicket"] = pop_deliveries["dismissal_type"].apply(_is_bowler_wicket)
    pop_deliveries["is_scoring"] = pop_deliveries["is_legal_for_batter"] & (pop_deliveries["batter_runs"].fillna(0) > 0)
    if "bowler_runs" in pop_deliveries.columns:
        fallback = pop_deliveries["batter_runs"].fillna(0) + pop_deliveries["wides"] + pop_deliveries["no_balls"]
        pop_deliveries["runs_charged"] = pop_deliveries["bowler_runs"].fillna(fallback)
    else:
        pop_deliveries["runs_charged"] = pop_deliveries["batter_runs"].fillna(0) + pop_deliveries["wides"] + pop_deliveries["no_balls"]

    team_deliveries = pop_deliveries[pop_deliveries["match_id"].isin(team_match_ids)].merge(
        team_matches[["match_id", "our_team_id", "opponent_team_id", "outcome"]], on="match_id", how="left"
    )
    our_batting_deliveries = team_deliveries[team_deliveries["batting_team_id"] == team_deliveries["our_team_id"]].copy()
    our_bowling_deliveries = team_deliveries[team_deliveries["batting_team_id"] == team_deliveries["opponent_team_id"]].copy()

    _render_win_rate(team_matches)
    _render_recent_form(team_matches)
    _render_toss(team_matches)
    _render_batting_avg_scores(team_matches, batting_pi)
    _render_lineup(team_matches, batting_pi, bowling_pi, our_bowling_deliveries, style_df)
    _render_batting_phase(our_batting_deliveries, pop_deliveries, selected_match_types)
    _render_batting_bowltype_phase(our_batting_deliveries, pop_deliveries, selected_match_types)
    _render_batting_individuals(team_matches, batting_pi, our_batting_deliveries, style_df, selected_match_types)
    _render_bowling_deployment(team_matches, our_bowling_deliveries, style_df)
    _render_bowling_phase(our_bowling_deliveries, pop_deliveries, selected_match_types)
    _render_bowling_individuals(team_matches, bowling_pi, our_bowling_deliveries, style_df, selected_match_types)


# =========================================================================
# Win Rate
# =========================================================================

def _render_win_rate(team_matches):
    st.subheader("Win Rate")

    valid = team_matches[team_matches["outcome"].notna()].copy()
    if valid.empty:
        st.info("No decided matches to compute a win rate from.")
        return

    def _agg(df, group_cols):
        g = df.groupby(group_cols + ["outcome"]).size().reset_index(name="count")
        totals = g.groupby(group_cols)["count"].sum().rename("total")
        g = g.merge(totals, on=group_cols)
        g["pct"] = 100 * g["count"] / g["total"]
        g["label"] = g["pct"].round(0).astype(int).astype(str) + "%"

        toss = df.groupby(group_cols + ["outcome"]).apply(
            lambda x: pd.Series({
                "toss_won_n": int(x["our_won_toss"].sum()),
                "bat_pct": 100 * (x["our_won_toss"] & x["our_batted_first"]).sum() / max(x["our_won_toss"].sum(), 1),
                "bowl_pct": 100 * (x["our_won_toss"] & ~x["our_batted_first"]).sum() / max(x["our_won_toss"].sum(), 1),
            })
        ).reset_index()
        return g.merge(toss, on=group_cols + ["outcome"], how="left")

    overall = _agg(valid, ["grade_season"])
    fig = px.bar(
        overall, x="grade_season", y="pct", color="outcome",
        category_orders={"outcome": OUTCOME_ORDER, "grade_season": sorted(overall["grade_season"].unique())},
        barmode="stack", color_discrete_map=OUTCOME_COLORS,
        custom_data=["count", "toss_won_n", "bat_pct", "bowl_pct"],
        text="label",
        title="Win / Draw / Loss % by season",
    )
    fig.update_traces(
        textposition="inside",
        hovertemplate=(
            "%{x} \u2014 %{fullData.name}: %{y:.0f}% (n=%{customdata[0]})<br>"
            "When won toss (n=%{customdata[1]}): Bat %{customdata[2]:.0f}% / Bowl %{customdata[3]:.0f}%"
            "<extra></extra>"
        )
    )
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**By match type**")
    by_mt = _agg(valid, ["grade_season", "match_type"])
    by_mt["group"] = by_mt["grade_season"] + " | " + by_mt["match_type"]
    fig2 = px.bar(
        by_mt, x="group", y="pct", color="outcome",
        category_orders={"outcome": OUTCOME_ORDER},
        barmode="stack", color_discrete_map=OUTCOME_COLORS,
        custom_data=["count", "toss_won_n", "bat_pct", "bowl_pct"],
        text="label",
    )
    fig2.update_traces(
        textposition="inside",
        hovertemplate=(
            "%{x} \u2014 %{fullData.name}: %{y:.0f}% (n=%{customdata[0]})<br>"
            "When won toss (n=%{customdata[1]}): Bat %{customdata[2]:.0f}% / Bowl %{customdata[3]:.0f}%"
            "<extra></extra>"
        )
    )
    fig2.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig2, width="stretch")


# =========================================================================
# Recent Form
# =========================================================================

def _render_recent_form(team_matches):
    st.subheader("Recent Form (last 6)")

    recent = team_matches.sort_values("day_1_start", ascending=False).head(6)
    if recent.empty:
        st.info("No recent matches available.")
        return

    innings_df = get_innings()
    innings_df = innings_df[innings_df["match_id"].isin(recent["match_id"])].sort_values("innings_order")

    rows = []
    for _, r in recent.iterrows():
        our_innings = innings_df[(innings_df["match_id"] == r["match_id"]) & (innings_df["batting_team_id"] == r["our_team_id"])]
        opp_innings = innings_df[(innings_df["match_id"] == r["match_id"]) & (innings_df["batting_team_id"] == r["opponent_team_id"])]
        our_last = our_innings.tail(1)
        opp_last = opp_innings.tail(1)
        our_score = _score_label(
            our_last["runs"].iloc[0] if not our_last.empty else None,
            our_last["wickets"].iloc[0] if not our_last.empty else None,
            our_last["overs"].iloc[0] if not our_last.empty else None,
        ) if not our_last.empty else DASH
        opp_score = _score_label(
            opp_last["runs"].iloc[0] if not opp_last.empty else None,
            opp_last["wickets"].iloc[0] if not opp_last.empty else None,
            opp_last["overs"].iloc[0] if not opp_last.empty else None,
        ) if not opp_last.empty else DASH

        date_str = pd.to_datetime(r["day_1_start"]).strftime("%d %b %Y") if pd.notna(r["day_1_start"]) else "Unknown date"
        rows.append({
            "Date": date_str, "Opponent": r["opponent_team"], "Venue": r.get("venue") or r.get("ground") or DASH,
            "Match type": r["match_type"], "Bat": "1st" if r["our_batted_first"] else "2nd",
            "Our score": our_score, "Opponent score": opp_score, "Result": r["outcome"] or "n/a",
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# =========================================================================
# Toss
# =========================================================================

def _render_toss(team_matches):
    st.subheader("Toss")

    won_toss = team_matches[team_matches["our_won_toss"]].copy()
    if won_toss.empty:
        st.info("No toss-win records available.")
        return

    def _bat_bowl_pct(df):
        n = len(df)
        bat_n = df["our_batted_first"].sum()
        return pd.Series({
            "n": n, "Bat %": 100 * bat_n / n if n else None, "Bowl %": 100 * (n - bat_n) / n if n else None,
        })

    overall = _bat_bowl_pct(won_toss)
    col1, col2, col3 = st.columns(3)
    col1.metric("Toss wins", int(overall["n"]))
    col2.metric("Chose to bat", f"{overall['Bat %']:.0f}%" if pd.notna(overall["Bat %"]) else DASH)
    col3.metric("Chose to bowl", f"{overall['Bowl %']:.0f}%" if pd.notna(overall["Bowl %"]) else DASH)

    by_mt = won_toss.groupby("match_type").apply(_bat_bowl_pct).reset_index()
    melt = by_mt.melt(id_vars=["match_type", "n"], value_vars=["Bat %", "Bowl %"], var_name="choice", value_name="pct")
    melt["label"] = melt["pct"].round(0).astype(int).astype(str) + "%"
    fig = px.bar(
        melt, x="match_type", y="pct", color="choice", barmode="stack",
        color_discrete_sequence=MAROON_SHADES[:2],
        title="When winning the toss: bat vs bowl % by match type",
        custom_data=["n"], text="label",
    )
    fig.update_traces(textposition="inside", hovertemplate="%{x} \u2014 %{fullData.name}: %{y:.0f}% (toss wins n=%{customdata[0]})<extra></extra>")
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, width="stretch")


# =========================================================================
# Batting Average Scores
# =========================================================================

def _render_batting_avg_scores(team_matches, batting_pi):
    st.subheader("Batting \u2014 Average Scores (batting first)")

    if batting_pi.empty:
        st.info("No batting data available.")
        return

    bf = team_matches[team_matches["our_batted_first"]].copy()
    if bf.empty:
        st.info("No matches where this team batted first.")
        return

    inn_totals = batting_pi[batting_pi["match_id"].isin(bf["match_id"])].groupby(["match_id", "innings_id", "team_id"]).agg(
        runs=("runs", "sum")
    ).reset_index()
    inn_totals = inn_totals.merge(bf[["match_id", "our_team_id", "match_type", "outcome"]], on="match_id", how="inner")
    inn_totals = inn_totals[inn_totals["team_id"] == inn_totals["our_team_id"]]

    rows = []
    for mt in ["One Day", "Two Day"]:
        sub = inn_totals[inn_totals["match_type"] == mt]
        if sub.empty:
            continue
        win_sub = sub[sub["outcome"] == "Win"]
        sub_nonzero = sub[sub["runs"] > 0]
        win_nonzero = win_sub[win_sub["runs"] > 0]
        rows.append({
            "Match type": mt,
            "Innings": len(sub),
            "Avg score": _fmt(sub["runs"].mean(), 1),
            "Lowest": int(sub_nonzero["runs"].min()) if not sub_nonzero.empty else DASH,
            "Highest": int(sub["runs"].max()) if not sub.empty else DASH,
            "Innings (wins)": len(win_sub),
            "Avg score (wins)": _fmt(win_sub["runs"].mean(), 1) if not win_sub.empty else DASH,
            "Lowest (wins)": int(win_nonzero["runs"].min()) if not win_nonzero.empty else DASH,
            "Highest (wins)": int(win_sub["runs"].max()) if not win_sub.empty else DASH,
        })

    if not rows:
        st.info("No One Day / Two Day innings where this team batted first.")
        return

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    all_totals = batting_pi[batting_pi["match_id"].isin(team_matches["match_id"])].groupby(
        ["match_id", "innings_id", "team_id"]
    ).agg(runs=("runs", "sum")).reset_index()
    all_totals = all_totals.merge(team_matches[["match_id", "our_team_id", "match_type"]], on="match_id", how="inner")
    all_totals = all_totals[all_totals["team_id"] == all_totals["our_team_id"]]

    if not all_totals.empty:
        all_totals["bucket"] = (all_totals["runs"] // 50 * 50).astype(int)
        bucket_labels = sorted(all_totals["bucket"].unique())
        all_totals["bucket_label"] = all_totals["bucket"].apply(lambda b: f"{b}-{b+49}")
        order = [f"{b}-{b+49}" for b in bucket_labels]

        counts = all_totals.groupby(["bucket_label", "match_type"]).size().reset_index(name="count")
        fig_hist = px.bar(
            counts, x="bucket_label", y="count", color="match_type", barmode="group",
            category_orders={"bucket_label": order},
            color_discrete_sequence=MAROON_SHADES,
            title="Distribution of innings scores (all match types, 50-run buckets)",
        )
        fig_hist.update_layout(xaxis_title="Score range", yaxis_title="Innings")
        st.plotly_chart(fig_hist, width="stretch")


# =========================================================================
# Lineup
# =========================================================================

def _render_lineup(team_matches, batting_pi, bowling_pi, our_bowling_deliveries, style_df):
    st.subheader("Lineup")

    hand_lookup = style_df.set_index("player_id")["batter_hand"].to_dict()
    bowl_style_lookup = style_df.set_index("player_id")["bowl_style"].to_dict()

    for mt in sorted(team_matches["match_type"].unique()):
        mt_matches = team_matches[team_matches["match_type"] == mt].sort_values("day_1_start", ascending=False)
        if mt_matches.empty:
            continue

        labels = []
        label_to_row = {}
        for _, r in mt_matches.iterrows():
            date_str = pd.to_datetime(r["day_1_start"]).strftime("%d %b %Y") if pd.notna(r["day_1_start"]) else "Unknown date"
            label = f"{date_str} \u2014 vs {r['opponent_team']} ({r['outcome'] or 'Result n/a'})"
            labels.append(label)
            label_to_row[label] = r

        sel_key = f"lineup_match_{mt}"
        selected_label = st.selectbox(f"{mt} \u2014 select match", labels, index=0, key=sel_key)
        row = label_to_row[selected_label]
        match_id, our_team_id = row["match_id"], row["our_team_id"]

        bat_rows = batting_pi[(batting_pi["match_id"] == match_id) & (batting_pi["team_id"] == our_team_id)].copy()
        bowl_rows = bowling_pi[(bowling_pi["match_id"] == match_id) & (bowling_pi["team_id"] == our_team_id)].copy()
        keepers = get_wicketkeepers_for_matches((str(match_id),))
        keeper_ids = set(keepers[keepers["team_id"] == our_team_id]["player_id"]) if not keepers.empty else set()

        if bat_rows.empty:
            st.info(f"No batting lineup recorded for {selected_label}.")
            continue

        bat_rows = bat_rows.sort_values("bat_position")

        mt_match_ids = set(mt_matches["match_id"])
        bat_appearances = batting_pi[
            batting_pi["match_id"].isin(mt_match_ids) & (batting_pi["team_id"] == our_team_id)
        ][["player_id", "match_id"]]
        bowl_appearances = bowling_pi[
            bowling_pi["match_id"].isin(mt_match_ids) & (bowling_pi["team_id"] == our_team_id)
        ][["player_id", "match_id"]]
        appearances = pd.concat([bat_appearances, bowl_appearances]).drop_duplicates()
        matches_played_map = appearances.groupby("player_id")["match_id"].nunique().to_dict()

        match_bowl_deliveries = our_bowling_deliveries[our_bowling_deliveries["match_id"] == match_id]
        bowl_order_map = {}
        if not match_bowl_deliveries.empty:
            first_over = match_bowl_deliveries.groupby("bowler_id")["over"].min().sort_values()
            bowl_order_map = {pid: i + 1 for i, pid in enumerate(first_over.index)}

        bowl_by_player = bowl_rows.set_index("player_id")

        display_rows = []
        for _, br in bat_rows.iterrows():
            pid = br["player_id"]
            hand = hand_lookup.get(pid)
            hand_str = hand if pd.notna(hand) and hand else "-"
            sr_val = br.get("strike_rate")
            sr_str = f"{sr_val:>3.0f}" if pd.notna(sr_val) else "  -"
            score_str = f"{int(br['runs'])} @ {sr_str}"
            keeper_tag = " (WK)" if pid in keeper_ids else ""

            overs_str = "-"
            bowl_str = "-"
            style_str = "-"
            bowl_ord_str = "-"
            if pid in bowl_by_player.index:
                bwr = bowl_by_player.loc[pid]
                has_overs = pd.notna(bwr["overs"]) and float(bwr["overs"]) > 0
                if has_overs:
                    overs_str = f"{bwr['overs']:.1f}"
                    w = int(bwr["wickets_taken"]) if pd.notna(bwr["wickets_taken"]) else 0
                    r = int(bwr["runs_conceded"]) if pd.notna(bwr["runs_conceded"]) else 0
                    bowl_str = f"{w}/{r}"
                style = bowl_style_lookup.get(pid)
                style_str = style if pd.notna(style) and style else "-"
                bowl_ord_str = str(bowl_order_map.get(pid, "-"))

            # Column order: batting/match-level fields grouped together
            # first (Pos, Player, Hand, Runs @ SR), then bowling fields
            # (Bowl Ord, Type, Overs, Figures), with Matches last.
            display_rows.append({
                "Pos": br["bat_position"], "Player": br["player_name"] + keeper_tag, "Hand": hand_str,
                "Runs @ SR": score_str,
                "Bowl Ord": bowl_ord_str, "Type": style_str, "Overs": overs_str, "Figures": bowl_str,
                "Matches": matches_played_map.get(pid, 1),
            })

        disp = pd.DataFrame(display_rows)
        _html_table(
            disp,
            align={"Runs @ SR": "right", "Pos": "center", "Matches": "center", "Bowl Ord": "center", "Overs": "center", "Figures": "center"},
            stretch=True,
        )


# =========================================================================
# Batting Scoring by Phase
# =========================================================================

def _phase_metrics(d, group_cols, match_type):
    if d.empty:
        return pd.DataFrame()
    d = d.copy()
    d["phase"] = d["over"].apply(lambda o: _phase_label(match_type, o))
    d = d.dropna(subset=["phase"])
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(group_cols + ["phase"], observed=True).agg(
        runs=("batter_runs", "sum"),
        balls=("is_legal_for_batter", "sum"),
        dismissals=("is_dismissal_batter", "sum"),
        scoring=("is_scoring", "sum"),
    ).reset_index()
    g["SR"] = g.apply(lambda r: 100 * r["runs"] / r["balls"] if r["balls"] > 0 else None, axis=1)
    g["BPD"] = g.apply(lambda r: r["balls"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
    g["SSpct"] = g.apply(lambda r: 100 * r["scoring"] / r["balls"] if r["balls"] > 0 else None, axis=1)
    g["phase"] = pd.Categorical(g["phase"], categories=_phase_order(match_type), ordered=True)
    return g.sort_values(group_cols + ["phase"])


def _render_batting_phase(our_batting_deliveries, pop_deliveries, selected_match_types):
    st.subheader("Batting \u2014 Scoring by Phase")

    for mt in [m for m in ["One Day", "Two Day", "T20"] if m in selected_match_types]:
        team_mt = our_batting_deliveries[our_batting_deliveries["match_type"] == mt]
        pop_mt = pop_deliveries[pop_deliveries["match_type"] == mt]
        if team_mt.empty:
            continue

        st.markdown(f"**{mt}**")

        team_overall = _phase_metrics(team_mt, [], mt)
        pop_overall = _phase_metrics(pop_mt, [], mt)
        if team_overall.empty:
            continue
        merged = team_overall.merge(pop_overall, on="phase", suffixes=("", "_all"), how="left")

        disp = merged[["phase", "SR", "BPD", "SSpct", "SR_all", "BPD_all", "SSpct_all"]].copy()
        for c in ["SR", "BPD", "SR_all", "BPD_all"]:
            disp[c] = disp[c].apply(lambda x: _fmt(x, 0))
        for c in ["SSpct", "SSpct_all"]:
            disp[c] = disp[c].apply(_pct)
        st.dataframe(
            disp.rename(columns={
                "phase": "Phase", "SR": "Team SR", "BPD": "Team BPD", "SSpct": "Team SS%",
                "SR_all": "All Teams SR", "BPD_all": "All Teams BPD", "SSpct_all": "All Teams SS%",
            }),
            width="stretch", hide_index=True,
        )

        for outcome_label, outcome_val in [("Batting In Wins", "Win"), ("Batting In Losses", "Loss")]:
            sub = team_mt[team_mt["outcome"] == outcome_val]
            m = _phase_metrics(sub, [], mt)
            if m.empty:
                continue
            m_disp = m[["phase", "SR", "BPD", "SSpct"]].copy()
            m_disp["SR"] = m_disp["SR"].apply(lambda x: _fmt(x, 0))
            m_disp["BPD"] = m_disp["BPD"].apply(lambda x: _fmt(x, 0))
            m_disp["SSpct"] = m_disp["SSpct"].apply(_pct)
            st.caption(outcome_label)
            st.dataframe(
                m_disp.rename(columns={"phase": "Phase", "SSpct": "SS%"}),
                width="stretch", hide_index=True,
            )


# =========================================================================
# Batting Scoring by Bowling Type
# =========================================================================

def _render_batting_bowltype_phase(our_batting_deliveries, pop_deliveries, selected_match_types):
    st.subheader("Batting \u2014 Scoring by Bowling Type")

    for mt in [m for m in ["One Day", "Two Day", "T20"] if m in selected_match_types]:
        team_mt = our_batting_deliveries[our_batting_deliveries["match_type"] == mt]
        pop_mt = pop_deliveries[pop_deliveries["match_type"] == mt]
        if team_mt.empty:
            continue
        st.markdown(f"**{mt}**")
        m = _phase_metrics(team_mt, ["bowler_pace_spin"], mt)
        pop_m = _phase_metrics(pop_mt, ["bowler_pace_spin"], mt)
        if m.empty:
            st.info("No data.")
            continue
        merged = m.merge(pop_m, on=["bowler_pace_spin", "phase"], suffixes=("", "_all"), how="left")
        disp = merged[["bowler_pace_spin", "phase", "SR", "BPD", "SSpct", "SR_all", "BPD_all", "SSpct_all"]].copy()
        for c in ["SR", "BPD", "SR_all", "BPD_all"]:
            disp[c] = disp[c].apply(lambda x: _fmt(x, 0))
        for c in ["SSpct", "SSpct_all"]:
            disp[c] = disp[c].apply(_pct)
        st.dataframe(
            disp.rename(columns={
                "bowler_pace_spin": "Bowling type", "phase": "Phase",
                "SR": "Team SR", "BPD": "Team BPD", "SSpct": "Team SS%",
                "SR_all": "All Teams SR", "BPD_all": "All Teams BPD", "SSpct_all": "All Teams SS%",
            }),
            width="stretch", hide_index=True,
        )


# =========================================================================
# Batting Individuals
# =========================================================================

def _render_batting_individuals(team_matches, batting_pi, our_batting_deliveries, style_df, selected_match_types):
    st.subheader("Batting \u2014 Individuals")

    hand_lookup = style_df.set_index("player_id")["batter_hand"].to_dict()

    def _summary(df):
        g = df.groupby("player_id").agg(
            player_name=("player_name", "first"),
            runs=("runs", "sum"),
            balls=("balls_faced", "sum"),
            dismissals=("dismissal_type", lambda x: x.notna().sum()),
        ).reset_index()
        g["Average"] = g.apply(lambda r: r["runs"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
        g["SR"] = g.apply(lambda r: 100 * r["runs"] / r["balls"] if r["balls"] > 0 else None, axis=1)
        g["BPD"] = g.apply(lambda r: r["balls"] / r["dismissals"] if r["dismissals"] > 0 else None, axis=1)
        return g

    def _delivery_metrics(df):
        balls = df["is_legal_for_batter"].sum()
        dismissals = df["is_dismissal_batter"].sum()
        scoring = df["is_scoring"].sum()
        return pd.Series({
            "SR": 100 * df["batter_runs"].sum() / balls if balls > 0 else None,
            "BPD": balls / dismissals if dismissals > 0 else None,
            "SSpct": 100 * scoring / balls if balls > 0 else None,
        })

    pace_cols = ["SR (Pace)", "BPD (Pace)", "SS% (Pace)"]
    spin_cols = ["SR (Spin)", "BPD (Spin)", "SS% (Spin)"]

    for mt in [m for m in ["One Day", "Two Day", "T20"] if m in selected_match_types]:
        mt_matches = team_matches[team_matches["match_type"] == mt]
        if mt_matches.empty:
            continue
        mt_bat = batting_pi[
            batting_pi["match_id"].isin(mt_matches["match_id"])
            & (batting_pi["team_id"].isin(mt_matches["our_team_id"]))
        ]
        mt_deliveries = our_batting_deliveries[our_batting_deliveries["match_type"] == mt]

        overall = _summary(mt_bat).sort_values("runs", ascending=False).head(10)
        if overall.empty:
            continue

        rows = []
        for _, r in overall.iterrows():
            pid = r["player_id"]
            hand = hand_lookup.get(pid)
            player_balls = mt_deliveries[mt_deliveries["batter_id"] == pid]

            total_ss = _delivery_metrics(player_balls)["SSpct"] if not player_balls.empty else None
            pace_m = _delivery_metrics(player_balls[player_balls["bowler_pace_spin"] == "Pace"]) if not player_balls.empty else pd.Series()
            spin_m = _delivery_metrics(player_balls[player_balls["bowler_pace_spin"] == "Spin"]) if not player_balls.empty else pd.Series()

            rows.append({
                "Player": r["player_name"], "Hand": hand if pd.notna(hand) and hand else "-",
                "Runs": str(int(r["runs"])), "Average": _fmt(r["Average"]), "SR": _fmt(r["SR"], 0), "BPD": _fmt(r["BPD"], 0),
                "SS%": _pct(total_ss),
                "SR (Pace)": _fmt(pace_m.get("SR"), 0), "BPD (Pace)": _fmt(pace_m.get("BPD"), 0), "SS% (Pace)": _pct(pace_m.get("SSpct")),
                "SR (Spin)": _fmt(spin_m.get("SR"), 0), "BPD (Spin)": _fmt(spin_m.get("BPD"), 0), "SS% (Spin)": _pct(spin_m.get("SSpct")),
            })

        st.markdown(f"**{mt} \u2014 top run scorers**")
        num_cols = ["Runs", "Average", "SR", "BPD", "SS%"] + pace_cols + spin_cols
        col_shades = {c: SHADE_A for c in pace_cols}
        col_shades.update({c: SHADE_B for c in spin_cols})
        _html_table(pd.DataFrame(rows), align={c: "right" for c in num_cols}, stretch=True, col_shades=col_shades)


# =========================================================================
# Bowling Deployment
# =========================================================================

def _render_bowling_deployment(team_matches, our_bowling_deliveries, style_df):
    st.subheader("Bowling \u2014 Deployment")

    style_lookup = style_df.set_index("player_id")[["pace_spin", "bowl_style"]]

    for mt in ["One Day", "T20"]:
        mt_matches = team_matches[team_matches["match_type"] == mt].sort_values("day_1_start", ascending=False)
        if mt_matches.empty:
            continue

        labels, label_to_row = [], {}
        for _, r in mt_matches.iterrows():
            date_str = pd.to_datetime(r["day_1_start"]).strftime("%d %b %Y") if pd.notna(r["day_1_start"]) else "Unknown date"
            label = f"{date_str} \u2014 vs {r['opponent_team']}"
            labels.append(label)
            label_to_row[label] = r

        selected_label = st.selectbox(f"{mt} deployment \u2014 select match", labels, index=0, key=f"deploy_{mt}")
        row = label_to_row[selected_label]
        match_id = row["match_id"]

        match_deliveries = our_bowling_deliveries[our_bowling_deliveries["match_id"] == match_id]
        if match_deliveries.empty:
            st.info("No ball-by-ball data for this match.")
            continue

        max_over = int(match_deliveries["over"].max()) + 1
        bowler_names = match_deliveries["bowler"].dropna().unique().tolist()
        initials_map = _build_initials(bowler_names)

        per_over = match_deliveries.groupby("over").agg(
            bowler=("bowler", "first"), bowler_id=("bowler_id", "first"),
            runs=("runs_charged", "sum"), wickets=("is_bowler_wicket", "sum"),
        ).reset_index()

        cells = {}
        for _, r in per_over.iterrows():
            over_num = int(r["over"]) + 1
            style = style_lookup.loc[r["bowler_id"], "pace_spin"] if r["bowler_id"] in style_lookup.index else None
            style = style if pd.notna(style) and style else "Unknown"
            bowl_style_full = style_lookup.loc[r["bowler_id"], "bowl_style"] if r["bowler_id"] in style_lookup.index else None
            bowl_style_full = bowl_style_full if pd.notna(bowl_style_full) and bowl_style_full else "Unknown"
            initials = initials_map.get(r["bowler"], "??")
            color = PACE_SPIN_COLORS.get(style, PACE_SPIN_COLORS["Unknown"])
            tooltip = f"{r['bowler']} ({bowl_style_full}) \u2014 {int(r['wickets'])}/{int(r['runs'])} this over"
            cells[over_num] = (initials, color, tooltip, r["bowler_id"])

        cell_font = "1rem"
        cell_pad = "8px"
        col_width = 100 / ((max_over + 1) // 2)
        EDGE_BORDER = "3px solid #eee"
        SPELL_BORDER = "3px solid #eee"

        def _row_cells(over_range):
            over_list = list(over_range)
            html_parts = []
            for idx, o in enumerate(over_list):
                is_table_start = idx == 0
                is_table_end = idx == len(over_list) - 1
                cur_bowler = cells[o][3] if o in cells else None
                # Look ahead within THIS row's sequence (same end -- odds or
                # evens), since a spell is consecutive overs from one end.
                next_o = over_list[idx + 1] if idx + 1 < len(over_list) else None
                next_bowler = cells[next_o][3] if (next_o is not None and next_o in cells) else None
                spell_ends_here = (cur_bowler is not None) and (next_bowler != cur_bowler)

                left = EDGE_BORDER if is_table_start else "none"
                right = EDGE_BORDER if is_table_end else (SPELL_BORDER if spell_ends_here else "none")
                border_style = f"border-left:{left};border-right:{right};border-top:{EDGE_BORDER};border-bottom:{EDGE_BORDER};"

                if o not in cells:
                    html_parts.append(f'<td style="padding:{cell_pad};width:{col_width:.2f}%;{border_style}"></td>')
                else:
                    ini, color, tip, _ = cells[o]
                    html_parts.append(
                        f'<td style="background:{color};color:white;text-align:center;padding:{cell_pad};'
                        f'font-size:{cell_font};width:{col_width:.2f}%;{border_style}" title="Over {o}: {tip}">{ini}</td>'
                    )
            return "".join(html_parts)

        odd_range = range(1, max_over + 1, 2)
        even_range = range(2, max_over + 1, 2)
        odd_cells = _row_cells(odd_range)
        even_cells = _row_cells(even_range)
        header_odd = "".join(f'<th style="font-size:0.8rem;padding:4px;text-align:center">{o}</th>' for o in odd_range)
        header_even = "".join(f'<th style="font-size:0.8rem;padding:4px;text-align:center">{o}</th>' for o in even_range)

        st.markdown(
            f"""
            <table style="border-collapse:collapse;font-family:monospace;width:100%;table-layout:fixed;">
                <tr>{header_odd}</tr>
                <tr>{odd_cells}</tr>
                <tr style="height:10px"></tr>
                <tr>{header_even}</tr>
                <tr>{even_cells}</tr>
            </table>
            """,
            unsafe_allow_html=True,
        )
        legend = " &nbsp; ".join(
            f'<span style="background:{c};color:white;padding:2px 6px;font-size:0.8rem">{k}</span>'
            for k, c in PACE_SPIN_COLORS.items()
        )
        st.markdown(legend, unsafe_allow_html=True)

    two_day_matches = team_matches[team_matches["match_type"] == "Two Day"].sort_values("day_1_start", ascending=False)
    if not two_day_matches.empty:
        labels, label_to_row = [], {}
        for _, r in two_day_matches.iterrows():
            date_str = pd.to_datetime(r["day_1_start"]).strftime("%d %b %Y") if pd.notna(r["day_1_start"]) else "Unknown date"
            label = f"{date_str} \u2014 vs {r['opponent_team']}"
            labels.append(label)
            label_to_row[label] = r
        selected_label = st.selectbox("Two Day deployment \u2014 select match", labels, index=0, key="deploy_2day")
        row = label_to_row[selected_label]
        match_deliveries = our_bowling_deliveries[our_bowling_deliveries["match_id"] == row["match_id"]].copy()
        if not match_deliveries.empty:
            match_deliveries["phase"] = match_deliveries["over"].apply(lambda o: _phase_label("Two Day", o))
            match_deliveries = match_deliveries.dropna(subset=["phase"])
            g = match_deliveries.groupby(["phase", "bowler", "bowler_id"]).agg(
                overs=("over", "nunique"), runs=("runs_charged", "sum"), wickets=("is_bowler_wicket", "sum"),
            ).reset_index()
            g["economy"] = g.apply(lambda r: r["runs"] / r["overs"] if r["overs"] > 0 else None, axis=1)
            g["phase"] = pd.Categorical(g["phase"], categories=_phase_order("Two Day"), ordered=True)
            g = g.sort_values(["phase", "overs"], ascending=[True, False]).reset_index(drop=True)

            bowl_style_lookup = style_df.set_index("player_id")["bowl_style"].to_dict()
            g["Type"] = g["bowler_id"].map(bowl_style_lookup).fillna("Unknown")
            g["economy"] = g["economy"].apply(lambda x: _fmt(x, 2))

            disp = g[["phase", "bowler", "Type", "overs", "runs", "wickets", "economy"]].rename(columns={
                "phase": "Phase", "bowler": "Bowler", "overs": "Overs", "runs": "Runs",
                "wickets": "Wickets", "economy": "Econ",
            })
            bold_before = set()
            last_phase = None
            for i, p in enumerate(disp["Phase"]):
                if last_phase is not None and p != last_phase:
                    bold_before.add(i)
                last_phase = p

            _html_table(
                disp,
                align={"Overs": "center", "Runs": "center", "Wickets": "center", "Econ": "center"},
                bold_before=bold_before,
                stretch=True,
            )


# =========================================================================
# Bowling By Phase
# =========================================================================

def _bowling_phase_metrics(d, match_type):
    if d.empty:
        return pd.DataFrame()
    d = d.copy()
    d["phase"] = d["over"].apply(lambda o: _phase_label(match_type, o))
    d = d.dropna(subset=["phase"])
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("phase", observed=True).agg(
        runs=("runs_charged", "sum"), balls=("is_legal", "sum"), wickets=("is_bowler_wicket", "sum"),
    ).reset_index()
    g["economy"] = g.apply(lambda r: r["runs"] / (r["balls"] / 6) if r["balls"] > 0 else None, axis=1)
    g["BPD"] = g.apply(lambda r: r["balls"] / r["wickets"] if r["wickets"] > 0 else None, axis=1)

    style_pct = d.groupby("phase", observed=True)["bowler_pace_spin"].value_counts(normalize=True).mul(100).unstack(fill_value=0)
    g = g.merge(style_pct, on="phase", how="left")
    g["phase"] = pd.Categorical(g["phase"], categories=_phase_order(match_type), ordered=True)
    return g.sort_values("phase")


def _render_bowling_phase(our_bowling_deliveries, pop_deliveries, selected_match_types):
    st.subheader("Bowling \u2014 By Phase")

    for mt in [m for m in ["One Day", "Two Day", "T20"] if m in selected_match_types]:
        team_mt = our_bowling_deliveries[our_bowling_deliveries["match_type"] == mt]
        pop_mt = pop_deliveries[pop_deliveries["match_type"] == mt]
        if team_mt.empty:
            continue
        st.markdown(f"**{mt}**")
        team_m = _bowling_phase_metrics(team_mt, mt)
        pop_m = _bowling_phase_metrics(pop_mt, mt)
        if team_m.empty:
            continue
        merged = team_m.merge(pop_m[["phase", "economy", "BPD"]], on="phase", suffixes=("", "_all"), how="left")

        style_cols = [c for c in ["Pace", "Spin", "Unknown"] if c in merged.columns]
        disp_cols = ["phase", "economy", "BPD", "economy_all", "BPD_all"] + style_cols
        disp = merged[disp_cols].copy()
        for c in ["economy", "BPD", "economy_all", "BPD_all"]:
            disp[c] = disp[c].apply(lambda x: _fmt(x, 2))
        for c in style_cols:
            disp[c] = disp[c].apply(_pct)
        st.dataframe(
            disp.rename(columns={
                "phase": "Phase", "economy": "Team Econ", "BPD": "Team BPD",
                "economy_all": "All Teams Econ", "BPD_all": "All Teams BPD",
            }),
            width="stretch", hide_index=True,
        )


# =========================================================================
# Bowling Individuals
# =========================================================================

def _render_bowling_individuals(team_matches, bowling_pi, our_bowling_deliveries, style_df, selected_match_types):
    st.subheader("Bowling \u2014 Individuals")

    style_lookup = style_df.set_index("player_id")["bowl_style"].to_dict()
    lhb_cols = ["Econ (v LHB)", "BPD (v LHB)"]
    rhb_cols = ["Econ (v RHB)", "BPD (v RHB)"]

    for mt in [m for m in ["One Day", "Two Day", "T20"] if m in selected_match_types]:
        mt_matches = team_matches[team_matches["match_type"] == mt]
        if mt_matches.empty:
            continue
        mt_bowl = bowling_pi[
            bowling_pi["match_id"].isin(mt_matches["match_id"])
            & (bowling_pi["team_id"].isin(mt_matches["our_team_id"]))
        ]
        top10 = mt_bowl.groupby("player_id").agg(
            player_name=("player_name", "first"),
            wickets=("wickets_taken", "sum"),
            runs_conceded=("runs_conceded", "sum"),
            balls=("overs", lambda s: sum(int(o) * 6 + int(round((float(o) - int(o)) * 10)) for o in s if pd.notna(o))),
        ).reset_index().sort_values("wickets", ascending=False).head(10)
        if top10.empty:
            continue
        top10["economy"] = top10.apply(lambda r: r["runs_conceded"] / (r["balls"] / 6) if r["balls"] > 0 else None, axis=1)
        top10["BPD"] = top10.apply(lambda r: r["balls"] / r["wickets"] if r["wickets"] > 0 else None, axis=1)

        mt_deliveries = our_bowling_deliveries[our_bowling_deliveries["match_type"] == mt]

        rows = []
        for _, r in top10.iterrows():
            pid = r["player_id"]
            style = style_lookup.get(pid)
            player_balls = mt_deliveries[mt_deliveries["bowler_id"] == pid]

            def _hand_metrics(hand):
                sub = player_balls[player_balls["faced_batter_hand"] == hand]
                if sub.empty:
                    return None, None
                balls_h = sub["is_legal"].sum()
                wkts_h = sub["is_bowler_wicket"].sum()
                runs_h = sub["runs_charged"].sum()
                econ_h = runs_h / (balls_h / 6) if balls_h > 0 else None
                bpd_h = balls_h / wkts_h if wkts_h > 0 else None
                return econ_h, bpd_h

            econ_l, bpd_l = _hand_metrics("Left")
            econ_r, bpd_r = _hand_metrics("Right")

            rows.append({
                "Player": r["player_name"], "Style": style if pd.notna(style) and style else "-",
                "Wickets": str(int(r["wickets"])), "Economy": _fmt(r["economy"]), "BPD": _fmt(r["BPD"], 0),
                "Econ (v LHB)": _fmt(econ_l), "BPD (v LHB)": _fmt(bpd_l, 0),
                "Econ (v RHB)": _fmt(econ_r), "BPD (v RHB)": _fmt(bpd_r, 0),
            })

        st.markdown(f"**{mt} \u2014 top wicket takers**")
        num_cols = ["Wickets", "Economy", "BPD"] + lhb_cols + rhb_cols
        col_shades = {c: SHADE_A for c in lhb_cols}
        col_shades.update({c: SHADE_B for c in rhb_cols})
        _html_table(pd.DataFrame(rows), align={c: "right" for c in num_cols}, stretch=True, col_shades=col_shades)
