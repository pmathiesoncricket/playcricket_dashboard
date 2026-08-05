import re
import json
import time
from datetime import datetime

import requests
import streamlit as st
from sqlalchemy import text

from db import (
    conn, get_matches, get_innings, get_batting_innings, get_bowling_innings,
    get_bowler_summary, get_batter_summary, get_highlights,
    get_wicket_deliveries, get_bowling_conceded_summary,
)

# ============================================================================
# Ported from the local Tkinter "Play Cricket: CSV + AWS RDS" tool.
# Deliberately DROPPED vs the original: the Tkinter GUI, all CSV read/write
# (save_csv / load_csv_rows / existing_match_ids_csv / delivery_count_csv),
# local file browsing, and the separate "upload player_style.csv" button
# (that's a local-file feature -- player_style is now managed through the
# Bowler Style / Batter Style tabs instead).
#
# Also different from the original by necessity: there's no background
# worker thread + queue here. Streamlit has no equivalent -- a script run
# executes top-to-bottom and streams its output to the browser as it goes,
# so a plain for-loop with a live st.empty()/st.progress() placeholder
# does the same job the Tkinter queue-polling loop did, just simpler.
# The trade-off: this tab is "busy" for the whole run (no true background
# processing, no clean mid-run cancel) -- see the safety cap below.
# ============================================================================

BASE = "https://grassrootsapiproxy.cricket.com.au/scores"
FIXTURES_BASE = "https://grassrootsapiproxy.cricket.com.au/fixturesladders"
HEADERS = {
    "Accept": "application/json",
    "Referer": "https://play.cricket.com.au/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
}
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
TABLES = ("matches", "innings", "player_innings", "deliveries", "highlights", "officials")
MAX_DEBUG_CHARS = 12000
CONFLICT_KEYS = {
    "matches": "match_id",
    "innings": "innings_id",
    "player_innings": "match_id,innings_id,player_id,role",
    "deliveries": "match_id,ball_id",
    "highlights": "highlight_id",
    "officials": "match_id,official_name,role",
}
FETCH_RETRIES = 4
FETCH_BACKOFF = 2.0
INTRA_MATCH_DELAY = 0.5
MODE_SKIP = "skip"
MODE_REPLACE = "replace"
MODE_CHECK_DELIVERIES = "check_deliveries"
MODE_LABELS = {
    MODE_SKIP: "Skip if exists (no re-fetch, fastest)",
    MODE_REPLACE: "Replace regardless (always re-fetch and overwrite)",
    MODE_CHECK_DELIVERIES: "Check for new deliveries (re-process only if ball count changed)",
}

# Safety cap: this tab runs synchronously in your browser tab with no
# background processing and no clean mid-run cancel (see chat explanation).
# Re-running the same IDs later is always safe (everything upserts on
# stable keys), so split anything bigger than this into multiple runs.
MAX_MATCHES_PER_RUN = 150

# Shared across sessions on purpose -- these just cache stable reference
# data (seasons/competitions don't change per-user), so sharing avoids
# redundant API calls. LAST_DEBUG is intentionally NOT shared (session_state).
SEASONS_CACHE = {}
GRADE_LOOKUP_CACHE = {}


# ---------------------------------------------------------------------------
# Debug helpers (session-scoped, so concurrent users don't see each other's
# last-error payloads)
# ---------------------------------------------------------------------------

def pretty_json(value):
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except Exception:
        return str(value)


def trim_debug_text(text_value, limit=MAX_DEBUG_CHARS):
    text_value = text_value or ""
    if len(text_value) <= limit:
        return text_value
    return text_value[:limit] + f"\n... [truncated, total {len(text_value)} chars]"


def set_last_debug(url, response):
    st.session_state["_loader_last_debug"] = {"url": url, "response": response}


def format_last_debug():
    debug = st.session_state.get("_loader_last_debug") or {}
    if not debug.get("url"):
        return ""
    payload = pretty_json(debug.get("response"))
    return f"Last debug URL:\n{debug['url']}\n\nLast debug response:\n{trim_debug_text(payload)}"


# ---------------------------------------------------------------------------
# API fetch helpers (unchanged from the original, minus CSV/Tkinter ties)
# ---------------------------------------------------------------------------

def extract_id(value):
    value = (
        value.strip()
        .replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
        .replace("\u200b", "").replace("\ufeff", "")
    )
    match = UUID_RE.search(value)
    if not match:
        raise ValueError(f"No valid UUID found: {value!r}")
    return match.group(0).lower()


def fetch(url, allow_empty=False, capture_debug=False):
    last_err = None
    last_text = ""
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            body = r.text.strip()
            last_text = body
            if not body:
                if allow_empty:
                    if capture_debug:
                        set_last_debug(url, {})
                    return {}
                raise ValueError("Empty response body")
            data = r.json()
            if capture_debug:
                set_last_debug(url, data)
            return data
        except Exception as e:
            last_err = e
            if attempt < FETCH_RETRIES:
                time.sleep(FETCH_BACKOFF * attempt)
    if allow_empty:
        if capture_debug:
            set_last_debug(url, {"raw_text": last_text})
        return {}
    raise RuntimeError(f"Failed after {FETCH_RETRIES} attempts: {url} ({last_err})")


def match_api(match_id, suffix="", query="", allow_empty=False):
    return fetch(f"{BASE}/matches/{match_id}{suffix}{query}", allow_empty=allow_empty)


def grade_match_ids(grade_id, completed_only):
    data = fetch(f"{BASE}/grades/{grade_id}/matches?jsconfig=eccn%3Atrue")
    return [
        x["id"] for x in data.get("matches", [])
        if x.get("id") and (not completed_only or x.get("status") == "COMPLETED")
    ]


def get_competition_seasons(organisation_id, season_id=None, capture_debug=False):
    url = f"{FIXTURES_BASE}/organisations/{organisation_id}/competition-seasons?responseModifier=includeGrades&jsconfig=eccn%3Atrue"
    if season_id:
        url += f"&seasonId={season_id}"
    data = fetch(url, allow_empty=True, capture_debug=capture_debug)
    return data.get("competitionSeasons", []) or []


def get_seasons(organisation_id, capture_debug=False):
    if organisation_id in SEASONS_CACHE:
        return SEASONS_CACHE[organisation_id]
    url = f"{FIXTURES_BASE}/organisations/{organisation_id}/seasons?jsconfig=eccn%3Atrue"
    data = fetch(url, allow_empty=True, capture_debug=capture_debug)
    seasons = data.get("seasons", []) or []
    SEASONS_CACHE[organisation_id] = seasons
    return seasons


def select_latest_season_id(competition_seasons):
    season_rows = []
    for row in competition_seasons:
        season = row.get("season", {}) or {}
        sid = season.get("id")
        end_date = season.get("endDate") or ""
        start_date = season.get("startDate") or ""
        name = season.get("name") or ""
        if sid:
            season_rows.append((end_date, start_date, name, sid))
    if not season_rows:
        return ""
    season_rows.sort(reverse=True)
    return season_rows[0][3]


def parse_iso_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def pick_best_season_for_match(score, seasons):
    schedule = score.get("matchSchedule", []) or []
    match_dt = None
    for row in schedule:
        match_dt = parse_iso_dt(row.get("startDateTime"))
        if match_dt:
            break
    if not match_dt:
        return None

    candidates = []
    for s in seasons:
        sid = s.get("id")
        if not sid:
            continue
        start_dt = parse_iso_dt(s.get("startDate"))
        if not start_dt:
            continue
        if start_dt <= match_dt:
            candidates.append((start_dt, s.get("name") or "", s))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]

    rows = []
    for s in seasons:
        sid = s.get("id")
        if sid:
            rows.append((s.get("startDate") or "", s.get("name") or "", s))
    if not rows:
        return None
    rows.sort(reverse=True)
    return rows[0][2]


def build_grade_competition_lookup(organisation_id, season_id):
    cache_key = (organisation_id, season_id)
    if cache_key in GRADE_LOOKUP_CACHE:
        return GRADE_LOOKUP_CACHE[cache_key]
    lookup = {}
    competition_seasons = get_competition_seasons(organisation_id, season_id, capture_debug=True)
    for comp_season in competition_seasons:
        competition = comp_season.get("competition", {}) or {}
        season = comp_season.get("season", {}) or {}
        for grade in comp_season.get("grades", []) or []:
            gid = grade.get("id")
            if gid:
                lookup[str(gid)] = {
                    "organisation_id": organisation_id,
                    "competition_id": competition.get("id", ""),
                    "competition_name": competition.get("name", ""),
                    "season_id": season.get("id", "") or season_id,
                    "season_name": season.get("name", ""),
                }
    GRADE_LOOKUP_CACHE[cache_key] = lookup
    return lookup


def extract_organisation_id_from_score(score):
    """
    Auto-discover the organisation_id directly from the match scorecard payload --
    no manual club/API URL required. Prefers the home team's owning organisation,
    falls back to the away team, then to the grade's organisation if present.
    """
    teams = score.get("teams", []) or []
    sm = score.get("matchSummary", {}) or {}
    rt = sm.get("teams", []) or []
    home_id = next((x.get("id") for x in rt if x.get("isHome")), None)

    home_team = next((t for t in teams if t.get("id") == home_id), None) if home_id else None
    if home_team:
        org = (home_team.get("owningOrganisation") or {}).get("id")
        if org:
            return str(org).lower()

    for t in teams:
        org = (t.get("owningOrganisation") or {}).get("id")
        if org:
            return str(org).lower()

    grade_org = (score.get("grade", {}) or {}).get("organisation", {}) or {}
    if grade_org.get("id"):
        return str(grade_org["id"]).lower()

    return ""


def resolve_match_context(score):
    """
    Fully automatic context resolution -- organisation_id is discovered from the
    match payload itself (teams[].owningOrganisation.id), then seasons and
    competition-seasons are queried to fill season/competition metadata.
    """
    organisation_id = extract_organisation_id_from_score(score)
    if not organisation_id:
        return {
            "organisation_id": "", "season_id": "", "season_name": "",
            "competition_id": "", "competition_name": "",
        }

    grade_id = str((score.get("grade") or {}).get("id", "") or "")

    seasons = get_seasons(organisation_id, capture_debug=True)
    season = pick_best_season_for_match(score, seasons)

    if not season:
        competition_seasons = get_competition_seasons(organisation_id, capture_debug=True)
        sid = select_latest_season_id(competition_seasons)
        if not sid:
            return {
                "organisation_id": organisation_id, "season_id": "", "season_name": "",
                "competition_id": "", "competition_name": "",
            }
        lookup = build_grade_competition_lookup(organisation_id, sid)
        meta = lookup.get(grade_id, {})
        return {
            "organisation_id": organisation_id,
            "season_id": meta.get("season_id", sid),
            "season_name": meta.get("season_name", ""),
            "competition_id": meta.get("competition_id", ""),
            "competition_name": meta.get("competition_name", ""),
        }

    season_id = str(season.get("id") or "").lower()
    season_name = season.get("name", "")
    lookup = build_grade_competition_lookup(organisation_id, season_id)
    meta = lookup.get(grade_id, {})
    return {
        "organisation_id": organisation_id,
        "season_id": meta.get("season_id", season_id),
        "season_name": meta.get("season_name", season_name),
        "competition_id": meta.get("competition_id", ""),
        "competition_name": meta.get("competition_name", ""),
    }


def match_display_name(score):
    sm = score.get("matchSummary", {})
    rt = sm.get("teams", [])
    home = next((x for x in rt if x.get("isHome")), {})
    away = next((x for x in rt if not x.get("isHome")), {})
    home_name = home.get("displayName") or ""
    away_name = away.get("displayName") or ""
    grade = score.get("grade", {}).get("name", "")
    label = f"{home_name} vs {away_name}" if home_name and away_name else score.get("id", "Unknown match")
    return f"{label} ({grade})" if grade else label


def api_delivery_count(balls_json):
    return sum(len(z.get("balls", [])) for z in balls_json.get("innings", []))


# ---------------------------------------------------------------------------
# Row-shaping (verbatim from the original -- pure dict transforms, no I/O)
# ---------------------------------------------------------------------------

def parse(score, balls, highlights, officials, source_url, organisation_id="", competition_id="",
          competition_name="", season_id="", season_name=""):
    mid = score["id"]
    teams = {t.get("id"): t.get("displayName", "") for t in score.get("teams", [])}
    players = {
        p.get("participantId"): p.get("name") or p.get("shortName", "")
        for t in score.get("teams", []) for p in t.get("players", [])
    }
    sm = score.get("matchSummary", {})
    rt = sm.get("teams", [])
    home = next((x for x in rt if x.get("isHome")), {})
    away = next((x for x in rt if not x.get("isHome")), {})

    toss_winner_team_id = next((x.get("id") for x in rt if x.get("wonToss")), "")
    batted_first_team_id = next((x.get("id") for x in rt if x.get("battedFirst")), "")

    v = score.get("venue", {})
    gsurf = v.get("playingSurface", {})
    sch = score.get("matchSchedule", []) or []
    grade = score.get("grade", {}) or {}
    match_streams = score.get("matchStreams", []) or []
    streams_by_order = {x.get("streamOrderNumber"): x for x in match_streams if x.get("streamOrderNumber") in (1, 2)}
    day1_stream = streams_by_order.get(1, {}) or {}
    day2_stream = streams_by_order.get(2, {}) or {}

    matches = [{
        "match_id": mid,
        "source_url": source_url,
        "status": score.get("status", ""),
        "match_type": score.get("matchType", ""),
        "grade": grade.get("name", ""),
        "grade_id": grade.get("id", ""),
        "organisation_id": organisation_id,
        "competition_id": competition_id,
        "competition_name": competition_name,
        "season_id": season_id,
        "season_name": season_name,
        "round": score.get("round", {}).get("name", ""),
        "home_team_id": home.get("id", ""),
        "home_team": home.get("displayName", ""),
        "away_team_id": away.get("id", ""),
        "away_team": away.get("displayName", ""),
        "result_text": sm.get("resultText", ""),
        "toss_winner_team_id": toss_winner_team_id,
        "batted_first_team_id": batted_first_team_id,
        "venue": v.get("name", ""),
        "address": v.get("line1", ""),
        "suburb": v.get("suburb", ""),
        "postcode": v.get("postCode", ""),
        "state": v.get("stateName", ""),
        "ground": gsurf.get("name", ""),
        "latitude": gsurf.get("latitude", ""),
        "longitude": gsurf.get("longitude", ""),
        "day_1_start": sch[0].get("startDateTime", "") if sch else "",
        "day_2_start": sch[1].get("startDateTime", "") if len(sch) > 1 else "",
        "day1_stream_url": day1_stream.get("streamUrl", ""),
        "day1_stream_start": day1_stream.get("recordingStartDateTime", ""),
        "day2_stream_url": day2_stream.get("streamUrl", ""),
        "day2_stream_start": day2_stream.get("recordingStartDateTime", ""),
    }]

    innings, perf = [], []
    for z in score.get("innings", []):
        iid = z.get("id", "")
        bat = z.get("battingTeamId", "")
        bowl = next((x for x in teams if x != bat), "")
        base = {"match_id": mid, "innings_id": iid, "innings_number": z.get("inningsNumber", ""), "innings_order": z.get("inningsOrder", "")}
        innings.append({
            **base,
            "innings_name": z.get("name", ""),
            "batting_team_id": bat,
            "batting_team": teams.get(bat, ""),
            "bowling_team_id": bowl,
            "bowling_team": teams.get(bowl, ""),
            "close_type": z.get("inningsCloseType", ""),
            "declared": z.get("isDeclared", False),
            "runs": z.get("runsScored", 0),
            "wickets": z.get("numberOfWicketsFallen", 0),
            "overs": z.get("oversBowled", 0),
            "extras": z.get("totalExtras", 0),
            "byes": z.get("byesRuns", 0),
            "leg_byes": z.get("legByesRuns", 0),
            "wides": z.get("wideBalls", 0),
            "no_balls": z.get("noBalls", 0),
            "penalties": z.get("penalties", 0),
        })
        for role, key in [("batting", "batting"), ("bowling", "bowling"), ("fielding", "fielding")]:
            for x in z.get(key, []):
                pid = x.get("participantId", "")
                perf.append({
                    **base,
                    "team_id": bat if role == "batting" else bowl,
                    "team": teams.get(bat if role == "batting" else bowl, ""),
                    "player_id": pid,
                    "player_name": players.get(pid, x.get("playerShortName", "")),
                    "role": role,
                    "bat_position": x.get("batOrder", "") if role == "batting" else "",
                    "runs": x.get("runsScored", "") if role == "batting" else "",
                    "balls_faced": x.get("ballsFaced", "") if role == "batting" else "",
                    "fours": x.get("foursScored", "") if role == "batting" else "",
                    "sixes": x.get("sixesScored", "") if role == "batting" else "",
                    "strike_rate": x.get("strikeRate", "") if role == "batting" else "",
                    "dismissal_type": x.get("dismissalType", "") if role == "batting" else "",
                    "dismissal_text": x.get("dismissalText", "") if role == "batting" else "",
                    "overs": x.get("oversBowled", "") if role == "bowling" else "",
                    "maidens": x.get("maidensBowled", "") if role == "bowling" else "",
                    "runs_conceded": x.get("runsConceded", "") if role == "bowling" else "",
                    "wickets_taken": x.get("wicketsTaken", "") if role == "bowling" else "",
                    "wides_bowled": x.get("wideBalls", "") if role == "bowling" else "",
                    "no_balls_bowled": x.get("noBalls", "") if role == "bowling" else "",
                    "economy": x.get("economy", "") if role == "bowling" else "",
                    "catches": x.get("totalCatches", "") if role == "fielding" else "",
                    "wicketkeeper_catches": x.get("wicketKeeperCatches", "") if role == "fielding" else "",
                    "run_outs": x.get("runOuts", "") if role == "fielding" else "",
                    "stumpings": x.get("stumpings", "") if role == "fielding" else "",
                })

    deliveries = []
    for z in balls.get("innings", []):
        for x in z.get("balls", []):
            batter_runs = x.get("runsBat", 0) or 0
            wides = x.get("wides", 0) or 0
            no_balls = x.get("noBalls", 0) or 0
            byes = x.get("byes", 0) or 0
            leg_byes = x.get("legByes", 0) or 0
            deliveries.append({
                "match_id": mid,
                "innings_id": z.get("id", ""),
                "innings_number": z.get("inningsNumber", ""),
                "innings_order": z.get("inningsOrder", ""),
                "batting_team_id": z.get("battingTeamId", ""),
                "ball_id": x.get("id", ""),
                "over": x.get("overNumber", 0),
                "ball_number": x.get("ballNumber", 0),
                "ball_display_number": x.get("ballDisplayNumber", 0),
                "ball_time": x.get("ballTime", ""),
                "bowler_start_time": x.get("bowlerStartTime", ""),
                "batter_id": x.get("strikerParticipantId", ""),
                "batter": players.get(x.get("strikerParticipantId"), ""),
                "non_striker_id": x.get("nonStrikerParticipantId", ""),
                "non_striker": players.get(x.get("nonStrikerParticipantId"), ""),
                "bowler_id": x.get("bowlerParticipantId", ""),
                "bowler": players.get(x.get("bowlerParticipantId"), ""),
                "batter_runs": batter_runs,
                "wides": wides,
                "no_balls": no_balls,
                "byes": byes,
                "leg_byes": leg_byes,
                "penalty_runs": x.get("penaltyRuns", 0),
                "team_runs": batter_runs + wides + no_balls + byes + leg_byes,
                "bowler_runs": batter_runs + wides + no_balls,
                "legal_balls": 1 if (wides + no_balls) == 0 else 0,
                "team_wickets": x.get("progressWickets", 0),
                "team_score": x.get("progressScore", ""),
                "dismissed_player_id": x.get("dismissedParticipantId", ""),
                "dismissal_type": x.get("dismissalType", ""),
                "fielder_id": x.get("fielderParticipantId", ""),
                "fielder": players.get(x.get("fielderParticipantId"), ""),
                "description": x.get("description", ""),
                "cum_striker_runs": x.get("strikerRunsScored", ""),
                "cum_striker_balls": x.get("strikerBallsFaced", ""),
                "cum_nonstriker_runs": x.get("nonStrikerRunsScored", ""),
                "cum_nonstriker_balls": x.get("nonStrikerBallsFaced", ""),
            })

    high = [{
        "match_id": mid,
        "innings_id": z.get("id", ""),
        "innings_number": z.get("inningsNumber", ""),
        "innings_order": z.get("inningsOrder", ""),
        "highlight_id": x.get("id", ""),
        "ball_id": x.get("ballId", ""),
        "over": x.get("overNumber", 0),
        "ball_number": x.get("ballNumber", 0),
        "batter_id": x.get("strikerParticipantId", ""),
        "batter": players.get(x.get("strikerParticipantId"), ""),
        "bowler_id": x.get("bowlerParticipantId", ""),
        "bowler": players.get(x.get("bowlerParticipantId"), ""),
        "highlight_type": x.get("highlightType", ""),
        "metrics": x.get("metrics", ""),
        "description": x.get("description", ""),
        "highlight_url": x.get("highlightURL", ""),
    } for z in highlights.get("innings", []) for x in z.get("highlight", [])]

    off = [{
        "match_id": mid,
        "official_name": x.get("name", ""),
        "official_short_name": x.get("shortName", ""),
        "role": x.get("role", ""),
    } for x in officials.get("officials", [])]

    return matches, innings, perf, deliveries, high, off


# ---------------------------------------------------------------------------
# DB helpers -- rewritten from psycopg2/execute_values onto the app's
# existing SQLAlchemy `conn` (st.connection("postgresql", type="sql")),
# same executemany-with-dicts pattern already used by tab_bowler_style.py /
# tab_batter_style.py for player_style upserts.
# ---------------------------------------------------------------------------

def db_value(v):
    if v == "":
        return None
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def dedupe_by_conflict(rows, conflict_cols):
    if not conflict_cols:
        return rows
    keys = conflict_cols.split(",")
    deduped = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        deduped[key] = row
    return list(deduped.values())


def upload_rows(table, rows):
    if not rows:
        return 0
    conflict_cols = CONFLICT_KEYS.get(table)
    rows = dedupe_by_conflict(rows, conflict_cols)
    columns = list(rows[0].keys())
    col_list = ", ".join(columns)
    param_list = ", ".join(f":{c}" for c in columns)

    if conflict_cols:
        conflict_col_set = set(conflict_cols.split(","))
        update_cols = [c for c in columns if c not in conflict_col_set]
        if update_cols:
            update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
            sql = f"INSERT INTO {table} ({col_list}) VALUES ({param_list}) ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}"
        else:
            sql = f"INSERT INTO {table} ({col_list}) VALUES ({param_list}) ON CONFLICT ({conflict_cols}) DO NOTHING"
    else:
        sql = f"INSERT INTO {table} ({col_list}) VALUES ({param_list}) ON CONFLICT DO NOTHING"

    cleaned = [{c: db_value(row.get(c, "")) for c in columns} for row in rows]

    with conn.session as s:
        for i in range(0, len(cleaned), 500):
            s.execute(text(sql), cleaned[i:i + 500])
        s.commit()
    return len(rows)


def existing_match_ids_db():
    try:
        with conn.session as s:
            result = s.execute(text("SELECT match_id FROM matches"))
            return {str(row[0]) for row in result}
    except Exception:
        return None


def delivery_count_db(match_id):
    try:
        with conn.session as s:
            result = s.execute(text("SELECT COUNT(*) FROM deliveries WHERE match_id = :mid"), {"mid": match_id})
            return result.scalar()
    except Exception:
        return None


def _clear_caches():
    for fn in (
        get_matches, get_innings, get_batting_innings, get_bowling_innings,
        get_bowler_summary, get_batter_summary, get_highlights,
        get_wicket_deliveries, get_bowling_conceded_summary,
    ):
        try:
            fn.clear()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Password gate -- reuses the already-configured DB password from secrets
# (st.secrets["connections"]["postgresql"]) rather than a separate secret.
# NOTE: verify this actually matches your secrets.toml layout -- Streamlit
# SQL connections can be configured either as discrete fields (password=...)
# or as a single connection `url`; this handles both, but you should
# double-check it resolves correctly for your setup before relying on it.
# ---------------------------------------------------------------------------

def _get_configured_db_password():
    try:
        pg_secrets = st.secrets["connections"]["postgresql"]
    except Exception:
        return None
    if "password" in pg_secrets:
        return pg_secrets["password"]
    url = pg_secrets.get("url", "")
    m = re.search(r"://[^:]+:([^@]+)@", url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------

def data_loader_tab():
    st.header("Data Loader")
    st.caption(
        "Fetch match data from Play Cricket and load it into the database. "
        "This writes directly to the production database -- double-check "
        "match/grade IDs before running. Safe to re-run: everything upserts "
        "on stable keys, so already-loaded matches are overwritten in place, "
        "never duplicated."
    )

    configured_password = _get_configured_db_password()
    if configured_password is None:
        st.error(
            "Could not read the database password from st.secrets to use as the "
            "unlock gate -- check the secrets.toml layout matches what "
            "_get_configured_db_password() expects in tab_data_loader.py."
        )
        return

    db_password = st.text_input("Database password (unlocks the loader)", type="password", key="loader_db_password")
    if not db_password:
        st.info("Enter the database password above to unlock the loader.")
        return
    if db_password != configured_password:
        st.error("Incorrect password.")
        return

    st.success("Unlocked.")

    match_text = st.text_area("Matches \u2014 IDs or URLs, one per line", height=150, key="loader_matches_text")
    grade_text = st.text_area("Grades \u2014 IDs or URLs, one per line", height=100, key="loader_grades_text")

    col1, col2, col3 = st.columns(3)
    with col1:
        completed_only = st.checkbox("Completed matches only (grades)", value=True, key="loader_completed_only")
    with col2:
        mode = st.selectbox(
            "Existing-match handling",
            options=[MODE_SKIP, MODE_REPLACE, MODE_CHECK_DELIVERIES],
            format_func=lambda m: MODE_LABELS[m],
            index=0, key="loader_mode",
        )
    with col3:
        delay_seconds = st.number_input(
            "Delay between matches (s)", min_value=0.0, value=1.0, step=0.5, key="loader_delay"
        )

    run_clicked = st.button("Run", type="primary", key="loader_run_button")

    if not run_clicked:
        return

    try:
        ids = [extract_id(x) for x in match_text.splitlines() if x.strip()]
        for x in grade_text.splitlines():
            if x.strip():
                ids.extend(grade_match_ids(extract_id(x), completed_only))
        ids = list(dict.fromkeys(ids))
    except Exception as e:
        st.error(f"Setup error: {e}")
        st.code(format_last_debug())
        return

    if not ids:
        st.warning("No match IDs supplied or resolved from grades.")
        return

    if len(ids) > MAX_MATCHES_PER_RUN:
        st.error(
            f"{len(ids)} matches queued, which is above the {MAX_MATCHES_PER_RUN}-match "
            f"safety cap for a single run (this tab runs synchronously in your browser "
            f"tab, with no background processing). Split this into smaller batches -- "
            f"it's always safe to re-run the same IDs later since everything upserts."
        )
        return

    _run_load_job(ids, mode, delay_seconds)


def _run_load_job(ids, mode, delay_seconds):
    existing_ids = set()
    if mode == MODE_SKIP:
        result = existing_match_ids_db()
        if result is not None:
            existing_ids = result
        else:
            st.warning("Could not query existing match IDs from the database; all matches will be processed.")

    st.write(f"**Mode:** {MODE_LABELS.get(mode, mode)}")

    total = len(ids)
    progress = st.progress(0.0)
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        log_placeholder.code("\n".join(log_lines[-300:]))

    for n, mid in enumerate(ids, 1):
        status_placeholder.write(f"Processing {n}/{total}...")

        if mode == MODE_SKIP and mid in existing_ids:
            log(f"[{n}/{total}] {mid} -> SKIPPED (already exists, mode=Skip if exists)")
            progress.progress(n / total)
            time.sleep(delay_seconds)
            continue

        try:
            score = match_api(mid, query="?responseModifier=includeScorecard&jsconfig=eccn%3Atrue")
            label = match_display_name(score)
            log(f"[{n}/{total}] {label}")
            time.sleep(INTRA_MATCH_DELAY)

            balls = match_api(mid, "/balls", "?jsconfig=eccn%3Atrue")
            time.sleep(INTRA_MATCH_DELAY)

            if mode == MODE_CHECK_DELIVERIES:
                api_count = api_delivery_count(balls)
                existing_count = delivery_count_db(mid)
                if existing_count is not None and existing_count == api_count:
                    log(f"  -> SKIPPED (deliveries unchanged: {existing_count})")
                    progress.progress(n / total)
                    time.sleep(delay_seconds)
                    continue
                else:
                    log(f"  -> Delivery count changed ({existing_count} -> {api_count}); reprocessing")

            highlights = match_api(mid, "/highlights", "?jsconfig=eccn%3Atrue", allow_empty=True)
            time.sleep(INTRA_MATCH_DELAY)
            officials = match_api(mid, "/officials", "?jsconfig=eccn%3Atrue", allow_empty=True)

            context = resolve_match_context(score)
            if context.get("organisation_id") and not context.get("competition_id"):
                grade_id = str((score.get("grade") or {}).get("id", "") or "")
                log(f"  -> No competition mapping found for grade_id {grade_id}; saving blank competition fields")
            elif not context.get("organisation_id"):
                log("  -> Could not auto-detect organisation_id; organisation/season/competition fields will be blank")

            tables = parse(
                score, balls, highlights, officials,
                f"https://play.cricket.com.au/match/{mid}/",
                organisation_id=context.get("organisation_id", ""),
                competition_id=context.get("competition_id", ""),
                competition_name=context.get("competition_name", ""),
                season_id=context.get("season_id", ""),
                season_name=context.get("season_name", ""),
            )

            status_parts = []
            for name, rows in zip(TABLES, tables):
                if rows:
                    try:
                        uploaded = upload_rows(name, rows)
                        status_parts.append(f"{name}={uploaded} uploaded")
                    except Exception as ue:
                        status_parts.append(f"{name}=FAILED({ue})")
                else:
                    status_parts.append(f"{name}=0 (missing)")
            log("  -> " + ", ".join(status_parts))
            log("  Complete")
        except Exception as e:
            log(f"[{n}/{total}] {mid} -> ERROR: {e}")
            debug = format_last_debug()
            if debug:
                log(debug)

        progress.progress(n / total)
        time.sleep(delay_seconds)

    status_placeholder.write(f"Finished all {total} matches.")
    _clear_caches()
    st.success("Load complete \u2014 cached report data has been refreshed across the app.")
