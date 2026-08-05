import re
import streamlit as st
import pandas as pd

from db import get_matches, get_innings, get_batting_innings, get_bowling_innings, get_deliveries_for_match
from helpers import add_season_column, cascading_multiselect
# Reuse the overs<->balls conversion and the bowling-metric prep helpers
# already ironed out on the Bowling tab, rather than re-deriving them.
from tab_bowling import overs_to_balls, _prep_bowling_deliveries

DASH = "\u2013"


def _fmt(x, dp=2):
    return f"{x:.{dp}f}" if pd.notna(x) else DASH


def _safe_int(x):
    return int(x) if pd.notna(x) else 0


def balls_to_overs_str(balls):
    if balls is None or pd.isna(balls) or balls == 0:
        return "0.0"
    balls = int(balls)
    return f"{balls // 6}.{balls % 6}"


def _short_round(round_text):
    if pd.isna(round_text) or not str(round_text).strip():
        return ""
    m = re.search(r"\d+", str(round_text))
    return f"R{m.group()}" if m else str(round_text)


def _match_label(row):
    round_short = _short_round(row.get("round"))
    home = row.get("home_team") or "?"
    away = row.get("away_team") or "?"
    mtype = row.get("match_type") or ""
    prefix = f"{round_short} - " if round_short else ""
    return f"{prefix}{home} v {away} - {mtype}"


def _boundary_detail(fours, sixes):
    fours, sixes = _safe_int(fours), _safe_int(sixes)
    parts = []
    if fours:
        parts.append(f"{fours}x4")
    if sixes:
        parts.append(f"{sixes}x6")
    return ", ".join(parts)


def _innings_score_label(row):
    runs = _safe_int(row.get("runs"))
    wickets = row.get("wickets")
    if pd.notna(wickets) and int(wickets) >= 10:
        return f"{runs} all out"
    if row.get("declared"):
        return f"{_safe_int(wickets)}-{runs} declared"
    if pd.notna(wickets):
        return f"{int(wickets)}-{runs}"
    return f"{runs}"


def _innings_summary_line(row):
    overs_val = row.get("overs")
    overs_str = f"{overs_val:.1f}" if pd.notna(overs_val) else DASH
    rr_str = DASH
    if pd.notna(overs_val) and overs_val > 0:
        balls = overs_to_balls(overs_val)
        if balls > 0:
            rr_str = f"{row['runs'] / (balls / 6):.2f}"
    return f"**{row.get('batting_team', 'Unknown')}** {_innings_score_label(row)} ({overs_str} overs, RR {rr_str})"


def match_summary_tab():
    st.header("Match Summary")

    matches_df = get_matches()
    if matches_df.empty:
        st.info("No match data available.")
        return
    matches_df = add_season_column(matches_df, "day_1_start")

    st.sidebar.markdown("### Match Summary filters")
    st.sidebar.caption("Filters are interdependent \u2014 each one narrows the options below it.")

    stage_matches = matches_df.copy()

    grade_options = sorted(stage_matches["grade"].dropna().unique().tolist())
    selected_grade = cascading_multiselect(st.sidebar, "Grade", grade_options, "ms_filter_grade")
    if selected_grade:
        stage_matches = stage_matches[stage_matches["grade"].isin(selected_grade)]

    season_options = (
        stage_matches["season"].dropna().drop_duplicates().sort_values(ascending=False).tolist()
    )
    selected_season = cascading_multiselect(st.sidebar, "Season (July\u2013June)", season_options, "ms_filter_season")
    if selected_season:
        stage_matches = stage_matches[stage_matches["season"].isin(selected_season)]

    match_type_options = sorted(stage_matches["match_type"].dropna().unique().tolist())
    selected_match_type = cascading_multiselect(
        st.sidebar, "Match type", match_type_options, "ms_filter_match_type",
        default_options=match_type_options,
    )
    if selected_match_type:
        stage_matches = stage_matches[stage_matches["match_type"].isin(selected_match_type)]

    if stage_matches.empty:
        st.warning("No matches match the current filters.")
        return

    # ---- Team (single-select; drives BOTH the batting and bowling sides
    # of the report, since the whole page is "one team's match", mirroring
    # the screenshot). ----
    team_options = sorted(
        pd.concat([stage_matches["home_team"], stage_matches["away_team"]]).dropna().unique().tolist()
    )
    if "ms_filter_team" in st.session_state and st.session_state["ms_filter_team"] not in (
        ["\u2014 select \u2014"] + team_options
    ):
        st.session_state["ms_filter_team"] = "\u2014 select \u2014"
    selected_team = st.sidebar.selectbox(
        "Team", ["\u2014 select \u2014"] + team_options, key="ms_filter_team"
    )
    if selected_team == "\u2014 select \u2014":
        st.info("Select a team in the sidebar to continue.")
        return

    team_matches = stage_matches[
        (stage_matches["home_team"] == selected_team) | (stage_matches["away_team"] == selected_team)
    ].copy()

    # ---- Opposition (single-select; also narrows both sides of the report
    # by narrowing which matches are eligible). ----
    opponent_options = sorted(
        pd.concat([
            team_matches.loc[team_matches["home_team"] == selected_team, "away_team"],
            team_matches.loc[team_matches["away_team"] == selected_team, "home_team"],
        ]).dropna().unique().tolist()
    )
    if "ms_filter_opponent" in st.session_state and st.session_state["ms_filter_opponent"] not in (
        ["All opponents"] + opponent_options
    ):
        st.session_state["ms_filter_opponent"] = "All opponents"
    selected_opponent = st.sidebar.selectbox(
        "Opposition", ["All opponents"] + opponent_options, key="ms_filter_opponent"
    )
    if selected_opponent != "All opponents":
        team_matches = team_matches[
            ((team_matches["home_team"] == selected_team) & (team_matches["away_team"] == selected_opponent))
            | ((team_matches["away_team"] == selected_team) & (team_matches["home_team"] == selected_opponent))
        ]

    if team_matches.empty:
        st.warning("No matches found for this team/opposition combination.")
        return

    team_matches = team_matches.copy()
    team_matches["match_label"] = team_matches.apply(_match_label, axis=1)
    team_matches = team_matches.sort_values("day_1_start", ascending=False)

    match_options = team_matches["match_label"].tolist()
    if "ms_filter_match" in st.session_state and st.session_state["ms_filter_match"] not in match_options:
        st.session_state.pop("ms_filter_match", None)

    selected_match_label = st.sidebar.selectbox("Match", match_options, key="ms_filter_match")
    match_row = team_matches[team_matches["match_label"] == selected_match_label].iloc[0]
    match_id = match_row["match_id"]
    opponent_name = match_row["away_team"] if match_row["home_team"] == selected_team else match_row["home_team"]

    _render_match_report(match_id, selected_team, opponent_name, match_row)


def _render_match_report(match_id, selected_team, opponent_name, match_row):
    innings_df = get_innings()
    innings_df = innings_df[innings_df["match_id"] == match_id].sort_values("innings_order")

    batting_all = get_batting_innings()
    bowling_all = get_bowling_innings()
    deliveries_df = get_deliveries_for_match(str(match_id))

    match_batting = batting_all[
        (batting_all["match_id"] == match_id) & (batting_all["team"] == selected_team)
    ].copy()
    match_bowling = bowling_all[
        (bowling_all["match_id"] == match_id) & (bowling_all["team"] == selected_team)
    ].copy()

    round_short = _short_round(match_row.get("round"))
    venue = match_row.get("venue") or match_row.get("ground") or "Unknown venue"
    grade = match_row.get("grade") or ""
    vs_prefix = f"{round_short} vs" if round_short else "vs"

    st.subheader(f"{selected_team} \u2014 {vs_prefix} {opponent_name} ({venue}, {grade})")
    st.caption(f"{match_row.get('match_type', '')} \u2014 {match_row.get('result_text') or 'Result unavailable'}")

    if innings_df.empty:
        st.info("No innings-level data recorded for this match.")
    else:
        st.markdown("**Score summary**")
        for _, inn in innings_df.iterrows():
            st.markdown(f"- {_innings_summary_line(inn)}")

    if deliveries_df.empty:
        st.warning("No ball-by-ball data available for this match \u2014 some detail below will be limited or skipped.")

    st.divider()

    our_batting_innings = innings_df[innings_df["batting_team"] == selected_team]
    if our_batting_innings.empty:
        st.info(f"No batting innings recorded for {selected_team} in this match.")
    for _, inn_row in our_batting_innings.iterrows():
        _render_batting_innings(inn_row, match_batting, deliveries_df, selected_team, vs_prefix, opponent_name, venue, grade)
        st.divider()

    our_bowling_innings = innings_df[innings_df["bowling_team"] == selected_team]
    if our_bowling_innings.empty:
        st.info(f"No bowling innings recorded for {selected_team} in this match.")
    for _, inn_row in our_bowling_innings.iterrows():
        _render_bowling_innings(inn_row, match_bowling, deliveries_df, selected_team)
        st.divider()

    st.markdown("### Notes")
    st.caption(
        "- Team totals in the batting table are the sum of individual batting figures; the official "
        "innings total (score summary above) may be higher \u2014 the difference is extras (byes, leg "
        "byes, wides, no-balls, penalties) not attributed to a specific batter.\n"
        "- Bowling figures (O, M, R, W, Econ, Wd, NB) are sourced directly from the official scorecard "
        "(player_innings).\n"
        "- Extrapolated ball-by-ball detail (Total Balls, Legal Deliveries, Dot Balls, Dot%, Boundaries "
        "Conceded, Boundary%, Scoring Shot%, 5-Dot Overs, Balls Scored From) is reconstructed from "
        "ball-by-ball deliveries.\n"
        "- Dot Balls = legal deliveries with zero runs charged to the bowler (leg byes/byes are credited "
        "to team extras, not the bowler's figures, but still count as dot balls here).\n"
        "- 5-Dot Overs = overs (legal deliveries only) in which 5 or 6 balls were dots.\n"
        "- Balls Scored From = Legal Deliveries minus Dot Balls."
    )


def _render_batting_innings(inn_row, match_batting, deliveries_df, selected_team, vs_prefix, opponent_name, venue, grade):
    innings_id = inn_row["innings_id"]
    label = _innings_score_label(inn_row)
    st.subheader(f"{selected_team} Batting \u2014 {vs_prefix} {opponent_name} ({venue}, {grade}) \u2014 Innings: {label}")

    bat_rows = match_batting[match_batting["innings_id"] == innings_id].copy()
    if bat_rows.empty:
        st.info("No individual batting records for this innings.")
        return

    bat_rows = bat_rows.sort_values("bat_position")
    bat_rows["fours"] = bat_rows["fours"].fillna(0)
    bat_rows["sixes"] = bat_rows["sixes"].fillna(0)

    if not deliveries_df.empty:
        inn_deliveries = deliveries_df[deliveries_df["innings_id"] == innings_id].copy()
        inn_deliveries["wides"] = inn_deliveries["wides"].fillna(0)
        # A batter faces (and can score off) a no-ball, just not a wide --
        # matches the "legal_ball = wides == 0" convention already used
        # for batting stats elsewhere in this app.
        legal_for_batter = inn_deliveries[inn_deliveries["wides"] == 0]
        ball_stats = legal_for_batter.groupby("batter_id").agg(
            dot_balls=("batter_runs", lambda x: (x == 0).sum()),
            singles=("batter_runs", lambda x: (x == 1).sum()),
        ).reset_index()
    else:
        ball_stats = pd.DataFrame(columns=["batter_id", "dot_balls", "singles"])

    bat_rows = bat_rows.merge(ball_stats, left_on="player_id", right_on="batter_id", how="left")
    bat_rows["dot_balls"] = bat_rows["dot_balls"].fillna(0).astype(int)
    bat_rows["singles"] = bat_rows["singles"].fillna(0).astype(int)
    bat_rows["boundaries"] = bat_rows["fours"] + bat_rows["sixes"]
    bat_rows["boundary_detail"] = bat_rows.apply(lambda r: _boundary_detail(r["fours"], r["sixes"]), axis=1)

    bat_rows["SR"] = bat_rows.apply(lambda r: 100 * r["runs"] / r["balls_faced"] if r["balls_faced"] else None, axis=1)
    bat_rows["S_pct"] = bat_rows.apply(lambda r: 100 * r["singles"] / r["balls_faced"] if r["balls_faced"] else None, axis=1)
    bat_rows["D_pct"] = bat_rows.apply(lambda r: 100 * r["dot_balls"] / r["balls_faced"] if r["balls_faced"] else None, axis=1)
    bat_rows["B_pct"] = bat_rows.apply(lambda r: 100 * r["boundaries"] / r["balls_faced"] if r["balls_faced"] else None, axis=1)

    display_cols = bat_rows[[
        "player_name", "runs", "balls_faced", "SR", "singles", "S_pct",
        "dot_balls", "D_pct", "boundaries", "B_pct", "boundary_detail",
    ]].copy()

    total_runs = bat_rows["runs"].sum()
    total_balls = bat_rows["balls_faced"].sum()
    total_singles = bat_rows["singles"].sum()
    total_dots = bat_rows["dot_balls"].sum()
    total_boundaries = bat_rows["boundaries"].sum()
    total_row = pd.DataFrame([{
        "player_name": "Team Total",
        "runs": total_runs,
        "balls_faced": total_balls,
        "SR": 100 * total_runs / total_balls if total_balls else None,
        "singles": total_singles,
        "S_pct": 100 * total_singles / total_balls if total_balls else None,
        "dot_balls": total_dots,
        "D_pct": 100 * total_dots / total_balls if total_balls else None,
        "boundaries": total_boundaries,
        "B_pct": 100 * total_boundaries / total_balls if total_balls else None,
        "boundary_detail": "",
    }])
    display_cols = pd.concat([display_cols, total_row], ignore_index=True)

    for col, dp in [("SR", 0), ("S_pct", 1), ("D_pct", 1), ("B_pct", 1)]:
        display_cols[col] = display_cols[col].apply(lambda x, dp=dp: _fmt(x, dp))

    st.dataframe(
        display_cols.rename(columns={
            "player_name": "Batsman", "runs": "Runs", "balls_faced": "Balls Faced", "SR": "SR",
            "singles": "Singles", "S_pct": "S%", "dot_balls": "Dot Balls", "D_pct": "D%",
            "boundaries": "Boundaries", "B_pct": "B%", "boundary_detail": "Boundary Detail",
        }),
        width="stretch", hide_index=True,
    )

    if pd.notna(inn_row.get("extras")):
        wkts = inn_row.get("wickets")
        wkts_str = str(int(wkts)) if pd.notna(wkts) else DASH
        st.caption(
            f"Official innings total: {_safe_int(inn_row.get('runs'))} ({wkts_str} wkts). "
            f"Extras: {_safe_int(inn_row.get('extras'))} "
            f"(byes {_safe_int(inn_row.get('byes'))}, leg byes {_safe_int(inn_row.get('leg_byes'))}, "
            f"wides {_safe_int(inn_row.get('wides'))}, no-balls {_safe_int(inn_row.get('no_balls'))}, "
            f"penalties {_safe_int(inn_row.get('penalties'))})."
        )


def _render_bowling_innings(inn_row, match_bowling, deliveries_df, selected_team):
    innings_id = inn_row["innings_id"]
    label = _innings_score_label(inn_row)
    st.subheader(f"{selected_team} Bowling \u2014 vs {inn_row.get('batting_team', 'Unknown')} \u2014 Innings: {label}")

    bowl_rows = match_bowling[match_bowling["innings_id"] == innings_id].copy()
    if bowl_rows.empty:
        st.info("No individual bowling records for this innings.")
        return

    inn_deliveries = (
        deliveries_df[deliveries_df["innings_id"] == innings_id].copy()
        if not deliveries_df.empty else pd.DataFrame()
    )

    # Order both tables by first over bowled, so they read in the same
    # chronological bowling order as the official scorecard.
    if not inn_deliveries.empty:
        first_over = (
            inn_deliveries.groupby("bowler_id")["over"].min()
            .reset_index().rename(columns={"over": "first_over"})
        )
        bowl_rows = bowl_rows.merge(first_over, left_on="player_id", right_on="bowler_id", how="left")
    else:
        bowl_rows["first_over"] = None
    bowl_rows = bowl_rows.sort_values("first_over", na_position="last")

    # ---- Official figures (from player_innings -- the source scorecard) ----
    official = bowl_rows[[
        "player_name", "overs", "maidens", "runs_conceded", "wickets_taken",
        "economy", "wides_bowled", "no_balls_bowled",
    ]].copy()

    total_balls_off = bowl_rows["overs"].apply(overs_to_balls).sum()
    total_runs_off = bowl_rows["runs_conceded"].sum()
    total_row_off = pd.DataFrame([{
        "player_name": "Team Total",
        "overs": balls_to_overs_str(total_balls_off),
        "maidens": bowl_rows["maidens"].sum(),
        "runs_conceded": total_runs_off,
        "wickets_taken": bowl_rows["wickets_taken"].sum(),
        "economy": (total_runs_off / (total_balls_off / 6)) if total_balls_off > 0 else None,
        "wides_bowled": bowl_rows["wides_bowled"].sum(),
        "no_balls_bowled": bowl_rows["no_balls_bowled"].sum(),
    }])
    official = pd.concat([official, total_row_off], ignore_index=True)
    official["economy"] = official["economy"].apply(lambda x: _fmt(x, 2))

    st.markdown("**Bowling figures (official scorecard)**")
    st.dataframe(
        official.rename(columns={
            "player_name": "Bowler", "overs": "O", "maidens": "M", "runs_conceded": "R",
            "wickets_taken": "W", "economy": "Econ", "wides_bowled": "Wd", "no_balls_bowled": "NB",
        }),
        width="stretch", hide_index=True,
    )

    # ---- Extrapolated ball-by-ball detail ----
    if inn_deliveries.empty:
        st.info("No ball-by-ball data available for this innings \u2014 extrapolated detail skipped.")
        return

    prepped = _prep_bowling_deliveries(inn_deliveries)
    prepped["is_dot"] = prepped["is_legal"] & (prepped["runs_charged"] == 0)
    prepped["is_boundary"] = prepped["is_four"] | prepped["is_six"]

    bbb = prepped.groupby("bowler_id").agg(
        bowler_name=("bowler", "first"),
        total_balls=("ball_id", "count"),
        legal_deliveries=("is_legal", "sum"),
        dot_balls=("is_dot", "sum"),
        boundaries_conceded=("is_boundary", "sum"),
    ).reset_index()
    bbb["balls_scored_from"] = bbb["legal_deliveries"] - bbb["dot_balls"]
    bbb["dot_pct"] = bbb.apply(
        lambda r: 100 * r["dot_balls"] / r["legal_deliveries"] if r["legal_deliveries"] > 0 else None, axis=1,
    )
    bbb["boundary_pct"] = bbb.apply(
        lambda r: 100 * r["boundaries_conceded"] / r["legal_deliveries"] if r["legal_deliveries"] > 0 else None, axis=1,
    )
    bbb["scoring_shot_pct"] = bbb.apply(
        lambda r: 100 * r["balls_scored_from"] / r["legal_deliveries"] if r["legal_deliveries"] > 0 else None, axis=1,
    )

    # 5-Dot Overs: legal deliveries only, grouped by bowler + over number --
    # an over where 5 or 6 of the legal balls were dots.
    legal_only = prepped[prepped["is_legal"]]
    per_over = legal_only.groupby(["bowler_id", "over"]).agg(
        dots_in_over=("is_dot", "sum"),
    ).reset_index()
    five_dot = (
        per_over[per_over["dots_in_over"] >= 5]
        .groupby("bowler_id").size().reset_index(name="five_dot_overs")
    )
    bbb = bbb.merge(five_dot, on="bowler_id", how="left")
    bbb["five_dot_overs"] = bbb["five_dot_overs"].fillna(0).astype(int)

    order_lookup = bowl_rows[["player_id", "first_over"]].rename(columns={"player_id": "bowler_id"})
    bbb = bbb.merge(order_lookup, on="bowler_id", how="left").sort_values("first_over", na_position="last")

    bbb_cols = [
        "bowler_name", "total_balls", "legal_deliveries", "dot_balls", "dot_pct",
        "boundaries_conceded", "boundary_pct", "scoring_shot_pct", "five_dot_overs", "balls_scored_from",
    ]
    total_legal = bbb["legal_deliveries"].sum()
    total_dots = bbb["dot_balls"].sum()
    total_boundaries = bbb["boundaries_conceded"].sum()
    total_scored_from = bbb["balls_scored_from"].sum()
    total_row_bbb = pd.DataFrame([{
        "bowler_name": "Team Total",
        "total_balls": bbb["total_balls"].sum(),
        "legal_deliveries": total_legal,
        "dot_balls": total_dots,
        "dot_pct": 100 * total_dots / total_legal if total_legal > 0 else None,
        "boundaries_conceded": total_boundaries,
        "boundary_pct": 100 * total_boundaries / total_legal if total_legal > 0 else None,
        "scoring_shot_pct": 100 * total_scored_from / total_legal if total_legal > 0 else None,
        "five_dot_overs": bbb["five_dot_overs"].sum(),
        "balls_scored_from": total_scored_from,
    }])

    bbb_display = pd.concat([bbb[bbb_cols], total_row_bbb[bbb_cols]], ignore_index=True)
    for col in ["dot_pct", "boundary_pct", "scoring_shot_pct"]:
        bbb_display[col] = bbb_display[col].apply(lambda x: _fmt(x, 1))

    st.markdown("**Extrapolated ball-by-ball detail**")
    st.dataframe(
        bbb_display.rename(columns={
            "bowler_name": "Bowler", "total_balls": "Total Balls", "legal_deliveries": "Legal Deliveries",
            "dot_balls": "Dot Balls", "dot_pct": "Dot%", "boundaries_conceded": "Boundaries Conceded",
            "boundary_pct": "Boundary%", "scoring_shot_pct": "Scoring Shot%",
            "five_dot_overs": "5-Dot Overs", "balls_scored_from": "Balls Scored From",
        }),
        width="stretch", hide_index=True,
    )
