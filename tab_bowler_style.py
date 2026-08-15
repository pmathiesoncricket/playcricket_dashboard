import math
import streamlit as st
import pandas as pd
from sqlalchemy import text

from db import (
    conn, get_matches, get_bowler_summary, get_bowler_teams, get_player_style,
    get_highlights_for_bowlers, get_ball_times,
)
from helpers import cascading_multiselect
from tab_batting import resolve_ball_video_url

PACE_SPIN_OPTIONS = ["Pace", "Spin"]
BOWL_HAND_OPTIONS = ["Right", "Left"]
BOWL_STYLE_OPTIONS = ["Right Pace", "Left Pace", "LAOS", "Off Spin", "Leg Spin"]

# If a row's bowl_style is being set/changed and pace_spin/bowl_hand are
# LEFT BLANK, inherit the conventional values below rather than leaving
# them empty. Never overwrites a value the user explicitly chose. Note
# the assumption baked in here: with no left-arm-wrist-spin ("chinaman")
# option in BOWL_STYLE_OPTIONS, Off Spin and Leg Spin both conventionally
# imply a RIGHT-arm bowler within this specific option set.
BOWL_STYLE_INFERENCE = {
    "Right Pace": ("Pace", "Right"),
    "Left Pace": ("Pace", "Left"),
    "LAOS": ("Spin", "Left"),
    "Off Spin": ("Spin", "Right"),
    "Leg Spin": ("Spin", "Right"),
}

# Explicit placeholder for "no value" -- always sorted/inserted at index 0,
# so a genuinely blank/NULL field renders as blank in the dropdown instead
# of silently landing on whichever real option is alphabetically first.
# Without this, saving a page you hadn't touched could actually WRITE a
# real value over every NULL row, since a selectbox always has SOME option
# selected and there was no way to tell "still blank" apart from
# "explicitly chose the first option".
BLANK_OPTION = "\u2014"
ROW_DIVIDER = "<hr style='margin:2px 0;border:none;border-top:1px solid #333'>"
MAX_HIGHLIGHTS_STEP = 10
BOWLERS_PER_PAGE = 20


def bowler_style_tab():
    st.header("Bowler Style")
    st.caption(
        "Review bowlers with missing style data, in pages of 20. "
        "Each page preloads up to 10 highlights per visible bowler (use "
        "\"Show more highlights\" below the list to load additional ones)."
    )

    bowler_summary = get_bowler_summary()
    if bowler_summary.empty:
        st.info("No delivery data available.")
        return

    matches_df = get_matches()
    matches_df["day_1_start"] = pd.to_datetime(matches_df["day_1_start"])

    style_df = get_player_style()
    teams_df = get_bowler_teams()
    bowler_summary = bowler_summary.merge(teams_df, on="bowler_id", how="left")

    all_grades = sorted({g for grades in bowler_summary["grades"] for g in (grades or []) if g})
    all_teams = sorted({t for teams in bowler_summary["teams"] for t in (teams or []) if t})
    all_bowlers = sorted(bowler_summary["bowler_name"].dropna().unique().tolist())

    filt_col1, filt_col2 = st.columns([2, 1])
    with filt_col1:
        selected_grade = cascading_multiselect(
            filt_col1, "Grade", all_grades, "bowler_filter_grade", enable_quick_add=True
        )
    with filt_col2:
        style_status = st.selectbox(
            "Bowl style", ["All", "Populated", "Unpopulated"], index=2, key="bowler_filter_style_status"
        )

    filt_col3, filt_col4 = st.columns(2)
    with filt_col3:
        selected_team = cascading_multiselect(
            filt_col3, "Team", all_teams, "bowler_filter_team", enable_quick_add=True
        )
    with filt_col4:
        selected_bowler_names = cascading_multiselect(
            filt_col4, "Player", all_bowlers, "bowler_filter_player"
        )

    if selected_grade:
        bowler_summary = bowler_summary[
            bowler_summary["grades"].apply(lambda g: any(x in selected_grade for x in (g or [])))
        ]
    if selected_team:
        bowler_summary = bowler_summary[
            bowler_summary["teams"].apply(lambda t: any(x in selected_team for x in (t or [])))
        ]
    if selected_bowler_names:
        bowler_summary = bowler_summary[bowler_summary["bowler_name"].isin(selected_bowler_names)]

    if bowler_summary.empty:
        st.info("No bowling deliveries match the current filters.")
        return

    bowler_summary = bowler_summary.merge(style_df, left_on="bowler_id", right_on="player_id", how="left")
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

    total_bowlers = len(bowler_summary)
    total_pages = max(1, math.ceil(total_bowlers / BOWLERS_PER_PAGE))

    if "bowler_page_num" not in st.session_state:
        st.session_state["bowler_page_num"] = 0
    if st.session_state["bowler_page_num"] >= total_pages:
        st.session_state["bowler_page_num"] = total_pages - 1

    current_page = st.session_state["bowler_page_num"]
    page_start = current_page * BOWLERS_PER_PAGE
    page_end = page_start + BOWLERS_PER_PAGE
    page_df = bowler_summary.iloc[page_start:page_end].copy().reset_index(drop=True)

    if page_df.empty:
        st.info("No bowlers on this page.")
        return

    st.caption(f"{total_bowlers} bowlers \u2014 page {current_page + 1} of {total_pages} (showing {len(page_df)})")

    pace_spin_opts = [BLANK_OPTION] + sorted(
        set(PACE_SPIN_OPTIONS) | set(style_df["pace_spin"].dropna().unique().tolist())
    )
    bowl_hand_opts = [BLANK_OPTION] + sorted(
        set(BOWL_HAND_OPTIONS) | set(style_df["bowl_hand"].dropna().unique().tolist())
    )
    bowl_style_opts = [BLANK_OPTION] + sorted(
        set(BOWL_STYLE_OPTIONS) | set(style_df["bowl_style"].dropna().unique().tolist())
    )

    page_bowler_ids = page_df["bowler_id"].tolist()
    page_bowler_ids_str = tuple(str(x) for x in page_bowler_ids)

    if "selected_bowler_id" not in st.session_state or st.session_state["selected_bowler_id"] not in page_bowler_ids:
        st.session_state["selected_bowler_id"] = page_df.iloc[0]["bowler_id"]

    def dropdown_index(options, current_value):
        if pd.notna(current_value) and current_value in options:
            return options.index(current_value)
        return 0

    def save_changed_rows(df):
        changed = []
        for _, row in df.iterrows():
            bid = row["bowler_id"]
            ps_val = st.session_state.get(f"ps_{bid}")
            hand_val = st.session_state.get(f"hand_{bid}")
            style_val = st.session_state.get(f"style_{bid}")
            new_pace_spin = None if (not ps_val or ps_val == BLANK_OPTION) else ps_val
            new_bowl_hand = None if (not hand_val or hand_val == BLANK_OPTION) else hand_val
            new_bowl_style = None if (not style_val or style_val == BLANK_OPTION) else style_val

            # Inherit pace_spin/bowl_hand from bowl_style ONLY where the
            # user left that specific field blank -- never overrides an
            # explicit choice.
            if new_bowl_style in BOWL_STYLE_INFERENCE:
                inferred_pace_spin, inferred_hand = BOWL_STYLE_INFERENCE[new_bowl_style]
                if new_pace_spin is None:
                    new_pace_spin = inferred_pace_spin
                if new_bowl_hand is None:
                    new_bowl_hand = inferred_hand

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
            return 0
        # NOTE: batter_hand is deliberately NOT part of this INSERT/UPDATE --
        # Postgres's ON CONFLICT DO UPDATE only touches columns listed in
        # SET, so a bowler who also bats and already has a batter_hand set
        # (via the Batter Style tab) keeps it untouched here automatically.
        with conn.session as s:
            s.execute(
                text("""
                    INSERT INTO player_style (player_id, pace_spin, bowl_hand, bowl_style)
                    VALUES (:player_id, :pace_spin, :bowl_hand, :bowl_style)
                    ON CONFLICT (player_id) DO UPDATE SET
                        pace_spin = EXCLUDED.pace_spin,
                        bowl_hand = EXCLUDED.bowl_hand,
                        bowl_style = EXCLUDED.bowl_style
                """),
                changed,
            )
            s.commit()
        get_player_style.clear()
        return len(changed)

    # Highlight limit for the currently-selected bowler grows by
    # MAX_HIGHLIGHTS_STEP each time "Show more highlights" is clicked, and
    # resets back to the base amount whenever a different bowler is
    # selected (tracked separately per bowler_id so flipping between two
    # bowlers doesn't lose either one's expanded count).
    if "bowler_highlight_limits" not in st.session_state:
        st.session_state["bowler_highlight_limits"] = {}
    current_limit = st.session_state["bowler_highlight_limits"].get(
        str(st.session_state["selected_bowler_id"]), MAX_HIGHLIGHTS_STEP
    )

    page_cache_key = (current_page, tuple(selected_grade), tuple(selected_team), tuple(selected_bowler_names), style_status, page_bowler_ids_str, current_limit)
    if st.session_state.get("bowler_highlights_page_key") != page_cache_key:
        page_highlights = get_highlights_for_bowlers(page_bowler_ids_str, max_per_bowler=current_limit)
        if not page_highlights.empty:
            page_highlights = page_highlights.merge(
                matches_df[["match_id", "day_1_start", "home_team", "day1_stream_url", "day1_stream_start",
                            "day2_stream_url", "day2_stream_start"]],
                on="match_id", how="left",
            )
            page_highlights["day_1_start"] = pd.to_datetime(page_highlights["day_1_start"])
            ball_ids = tuple(page_highlights["ball_id"].dropna().astype(str).unique())
            ball_times = get_ball_times(ball_ids)
            if not ball_times.empty:
                page_highlights = page_highlights.merge(ball_times, on="ball_id", how="left")
            else:
                page_highlights["ball_time"] = None
        st.session_state["bowler_highlights_page_df"] = page_highlights
        st.session_state["bowler_highlights_page_key"] = page_cache_key
        st.session_state.pop("selected_bowler_highlight_id", None)

    page_highlights_df = st.session_state.get("bowler_highlights_page_df", pd.DataFrame())

    PANEL_HEIGHT = 560
    list_col, highlight_col = st.columns([2, 3])

    with list_col:
        with st.form("bowler_styles_form", clear_on_submit=False):
            nav1, nav2, nav3 = st.columns([1, 1, 2])
            with nav1:
                prev_page_clicked = st.form_submit_button(
                    "< Prev page", disabled=current_page == 0, use_container_width=True,
                )
            with nav2:
                next_page_clicked = st.form_submit_button(
                    "Next page >", disabled=current_page == total_pages - 1, use_container_width=True,
                )
            with nav3:
                save_all_clicked = st.form_submit_button(
                    "Save page changes", type="primary", use_container_width=True,
                )

            show_hl_clicks = {}
            with st.container(height=PANEL_HEIGHT):
                for _, row in page_df.iterrows():
                    bid = row["bowler_id"]
                    is_selected = bid == st.session_state["selected_bowler_id"]
                    selected_marker = " (current)" if is_selected else ""
                    st.markdown(f"**{row['bowler_name']}** ({row['balls']} balls){selected_marker}")

                    cps, chand, cstyle, cselect = st.columns([1, 1, 1, 1])
                    with cps:
                        st.selectbox(
                            "Pace/Spin", pace_spin_opts,
                            index=dropdown_index(pace_spin_opts, row.get("pace_spin")),
                            key=f"ps_{bid}", label_visibility="collapsed",
                        )
                    with chand:
                        st.selectbox(
                            "Hand", bowl_hand_opts,
                            index=dropdown_index(bowl_hand_opts, row.get("bowl_hand")),
                            key=f"hand_{bid}", label_visibility="collapsed",
                        )
                    with cstyle:
                        st.selectbox(
                            "Style", bowl_style_opts,
                            index=dropdown_index(bowl_style_opts, row.get("bowl_style")),
                            key=f"style_{bid}", label_visibility="collapsed",
                        )
                    with cselect:
                        show_hl_clicks[bid] = st.form_submit_button(
                            "Show highlights", key=f"show_hl_{bid}",
                            help="Show highlights", use_container_width=True,
                        )
                    st.markdown(ROW_DIVIDER, unsafe_allow_html=True)

        if prev_page_clicked:
            try:
                saved_count = save_changed_rows(page_df)
                if saved_count:
                    st.success(f"Saved {saved_count} changes.")
            except Exception as e:
                st.error(f"Save failed: {e}")
            else:
                st.session_state["bowler_page_num"] = max(0, current_page - 1)
                st.session_state["selected_bowler_id"] = None
                st.rerun()

        if next_page_clicked:
            try:
                saved_count = save_changed_rows(page_df)
                if saved_count:
                    st.success(f"Saved {saved_count} changes.")
            except Exception as e:
                st.error(f"Save failed: {e}")
            else:
                st.session_state["bowler_page_num"] = min(total_pages - 1, current_page + 1)
                st.session_state["selected_bowler_id"] = None
                st.rerun()

        if save_all_clicked:
            try:
                saved_count = save_changed_rows(page_df)
                if saved_count:
                    st.success(f"Saved {saved_count} changes.")
                else:
                    st.info("No changes to save.")
                st.rerun()
            except Exception as e:
                st.error(f"Bulk save failed: {e}")

        clicked_bowler_id = next((bid for bid, clicked in show_hl_clicks.items() if clicked), None)
        if clicked_bowler_id is not None:
            st.session_state["selected_bowler_id"] = clicked_bowler_id
            st.session_state.pop("selected_bowler_highlight_id", None)
            st.rerun()

    with highlight_col:
        selected_row = page_df[page_df["bowler_id"] == st.session_state["selected_bowler_id"]]
        if selected_row.empty:
            st.info("Select a bowler to view highlights.")
            return
        selected_row = selected_row.iloc[0]
        selected_bowler_id_str = str(selected_row["bowler_id"])

        if page_highlights_df.empty:
            st.info("No highlights available for bowlers on this page.")
            return

        bh_sorted = page_highlights_df[page_highlights_df["bowler_id"] == selected_bowler_id_str].copy()
        if bh_sorted.empty:
            st.info("No highlights available for this bowler.")
            return

        bh_sorted = bh_sorted.sort_values(
            ["day_1_start", "innings_number", "over", "ball_number"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)

        if (
            "selected_bowler_highlight_id" not in st.session_state
            or st.session_state["selected_bowler_highlight_id"] not in bh_sorted["highlight_id"].values
        ):
            st.session_state["selected_bowler_highlight_id"] = bh_sorted.iloc[0]["highlight_id"]

        st.caption(f"{len(bh_sorted)} highlights shown \u2014 tap to play")

        list_height = 240
        with st.container(height=list_height):
            for _, h_row in bh_sorted.iterrows():
                hid = h_row["highlight_id"]
                txt_col, play_col, yt_col = st.columns([4, 1, 1])
                with txt_col:
                    date_str = (
                        h_row["day_1_start"].strftime("%d %b %Y") if pd.notna(h_row.get("day_1_start")) else ""
                    )
                    home_team = h_row.get("home_team") or ""
                    st.markdown(
                        f"**{h_row.get('batter')}** vs {h_row.get('bowler')} \u2014 {h_row.get('highlight_type')}  \n"
                        f"{h_row.get('description')}  \n"
                        f"<span style='color:gray;font-size:0.8em'>{date_str} {home_team}</span>",
                        unsafe_allow_html=True,
                    )
                with play_col:
                    if st.button("Play", key=f"bowler_hl_play_{hid}"):
                        st.session_state["selected_bowler_highlight_id"] = hid
                with yt_col:
                    yt_url = resolve_ball_video_url(
                        h_row.get("ball_time"), h_row.get("day1_stream_url"), h_row.get("day1_stream_start"),
                        h_row.get("day2_stream_url"), h_row.get("day2_stream_start"),
                    )
                    if yt_url:
                        st.link_button("YouTube", yt_url)
                    else:
                        st.button("YouTube", key=f"bowler_hl_yt_disabled_{hid}", disabled=True)
                st.markdown(ROW_DIVIDER, unsafe_allow_html=True)

        show_more_clicked = st.button("Show more highlights", key="bowler_show_more_highlights")
        if show_more_clicked:
            limits = st.session_state["bowler_highlight_limits"]
            limits[selected_bowler_id_str] = limits.get(selected_bowler_id_str, MAX_HIGHLIGHTS_STEP) + MAX_HIGHLIGHTS_STEP
            st.rerun()

        sel_h = bh_sorted[bh_sorted["highlight_id"] == st.session_state["selected_bowler_highlight_id"]].iloc[0]
        url = sel_h.get("highlight_url")
        if url:
            st.video(url, autoplay=True)
        else:
            st.info("No video URL available for this highlight.")
