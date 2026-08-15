import math
import streamlit as st
import pandas as pd
from sqlalchemy import text

from db import (
    conn, get_matches, get_batter_summary, get_batter_teams, get_player_style,
    get_highlights_for_batters, get_ball_times,
)
from helpers import cascading_multiselect
from tab_batting import resolve_ball_video_url

# This tab is a deliberate mirror of tab_bowler_style.py's layout and
# save/pagination mechanics -- same page size, same "form + scrollable
# container" pattern, same prev/next/save-page button trio, same
# highlights panel on the right. The only real differences: it edits
# batter_hand (a single dropdown) instead of pace_spin/bowl_hand/bowl_style
# (three dropdowns), it's keyed on batter_id/batter throughout instead of
# bowler_id/bowler, and highlights are filtered to the tagged BATTER
# instead of the tagged bowler.

BATTER_HAND_OPTIONS = ["Right", "Left"]
BLANK_OPTION = "\u2014"
ROW_DIVIDER = "<hr style='margin:2px 0;border:none;border-top:1px solid #333'>"
MAX_HIGHLIGHTS_STEP = 10
BATTERS_PER_PAGE = 20


def _as_list(x):
    """
    Normalizes a Postgres array_agg() result to a plain Python list for
    safe iteration. After a LEFT JOIN, any batter with no matching row
    (e.g. get_batter_teams() found no player_innings row with role=
    'batting' and a populated team) gets NaN in that column instead of an
    empty list -- and `nan or []` evaluates to nan itself (bool(nan) is
    True), so `for t in (nan or [])` throws "float object is not
    iterable". This catches that case (and any other non-list value)
    before iteration ever happens. This hasn't been observed failing on
    this specific tab yet, but the same LEFT JOIN pattern caused exactly
    this crash on the Bowler Style tab, so it's applied here pre-emptively.
    """
    return x if isinstance(x, list) else []


def batter_style_tab():
    st.header("Batter Style")
    st.caption(
        "Review batters with missing batting-hand data, in pages of 20. "
        "Each page preloads up to 10 highlights per visible batter (use "
        "\"Show more highlights\" below the list to load additional ones)."
    )

    batter_summary = get_batter_summary()
    if batter_summary.empty:
        st.info("No delivery data available.")
        return

    matches_df = get_matches()
    matches_df["day_1_start"] = pd.to_datetime(matches_df["day_1_start"])

    style_df = get_player_style()
    teams_df = get_batter_teams()
    batter_summary = batter_summary.merge(teams_df, on="batter_id", how="left")

    all_grades = sorted({g for grades in batter_summary["grades"] for g in _as_list(grades) if g})
    all_teams = sorted({t for teams in batter_summary["teams"] for t in _as_list(teams) if t})
    all_batters = sorted(batter_summary["batter_name"].dropna().unique().tolist())

    filt_col1, filt_col2 = st.columns([2, 1])
    with filt_col1:
        selected_grade = cascading_multiselect(
            filt_col1, "Grade", all_grades, "batter_filter_grade", enable_quick_add=True
        )
    with filt_col2:
        hand_status = st.selectbox(
            "Batter hand", ["All", "Populated", "Unpopulated"], index=2, key="batter_filter_hand_status"
        )

    filt_col3, filt_col4 = st.columns(2)
    with filt_col3:
        selected_team = cascading_multiselect(
            filt_col3, "Team", all_teams, "batter_filter_team", enable_quick_add=True
        )
    with filt_col4:
        selected_batter_names = cascading_multiselect(
            filt_col4, "Player", all_batters, "batter_filter_player"
        )

    if selected_grade:
        batter_summary = batter_summary[
            batter_summary["grades"].apply(lambda g: any(x in selected_grade for x in _as_list(g)))
        ]
    if selected_team:
        batter_summary = batter_summary[
            batter_summary["teams"].apply(lambda t: any(x in selected_team for x in _as_list(t)))
        ]
    if selected_batter_names:
        batter_summary = batter_summary[batter_summary["batter_name"].isin(selected_batter_names)]

    if batter_summary.empty:
        st.info("No batting deliveries match the current filters.")
        return

    batter_summary = batter_summary.merge(style_df, left_on="batter_id", right_on="player_id", how="left")
    batter_summary["batter_hand_populated"] = (
        batter_summary["batter_hand"].notna() & (batter_summary["batter_hand"] != "")
    )

    if hand_status == "Populated":
        batter_summary = batter_summary[batter_summary["batter_hand_populated"]]
    elif hand_status == "Unpopulated":
        batter_summary = batter_summary[~batter_summary["batter_hand_populated"]]

    batter_summary = batter_summary.sort_values("balls", ascending=False).reset_index(drop=True)

    if batter_summary.empty:
        st.info("No batters match the current filters.")
        return

    total_batters = len(batter_summary)
    total_pages = max(1, math.ceil(total_batters / BATTERS_PER_PAGE))

    if "batter_page_num" not in st.session_state:
        st.session_state["batter_page_num"] = 0
    if st.session_state["batter_page_num"] >= total_pages:
        st.session_state["batter_page_num"] = total_pages - 1

    current_page = st.session_state["batter_page_num"]
    page_start = current_page * BATTERS_PER_PAGE
    page_end = page_start + BATTERS_PER_PAGE
    page_df = batter_summary.iloc[page_start:page_end].copy().reset_index(drop=True)

    if page_df.empty:
        st.info("No batters on this page.")
        return

    st.caption(f"{total_batters} batters \u2014 page {current_page + 1} of {total_pages} (showing {len(page_df)})")

    batter_hand_opts = [BLANK_OPTION] + sorted(
        set(BATTER_HAND_OPTIONS) | set(style_df["batter_hand"].dropna().unique().tolist())
    )

    page_batter_ids = page_df["batter_id"].tolist()
    page_batter_ids_str = tuple(str(x) for x in page_batter_ids)

    if "selected_batter_id" not in st.session_state or st.session_state["selected_batter_id"] not in page_batter_ids:
        st.session_state["selected_batter_id"] = page_df.iloc[0]["batter_id"]

    def dropdown_index(options, current_value):
        if pd.notna(current_value) and current_value in options:
            return options.index(current_value)
        return 0

    def save_changed_rows(df):
        changed = []
        for _, row in df.iterrows():
            bid = row["batter_id"]
            hand_val = st.session_state.get(f"bhand_{bid}")
            new_batter_hand = None if (not hand_val or hand_val == BLANK_OPTION) else hand_val
            orig_batter_hand = row.get("batter_hand") if pd.notna(row.get("batter_hand")) else None
            if new_batter_hand != orig_batter_hand:
                changed.append({
                    "player_id": str(bid),
                    "batter_hand": new_batter_hand,
                    "pace_spin": row.get("pace_spin") if pd.notna(row.get("pace_spin")) else None,
                    "bowl_hand": row.get("bowl_hand") if pd.notna(row.get("bowl_hand")) else None,
                    "bowl_style": row.get("bowl_style") if pd.notna(row.get("bowl_style")) else None,
                })
        if not changed:
            return 0
        with conn.session as s:
            s.execute(
                text("""
                    INSERT INTO player_style (player_id, batter_hand, pace_spin, bowl_hand, bowl_style)
                    VALUES (:player_id, :batter_hand, :pace_spin, :bowl_hand, :bowl_style)
                    ON CONFLICT (player_id) DO UPDATE SET
                        batter_hand = EXCLUDED.batter_hand,
                        pace_spin = EXCLUDED.pace_spin,
                        bowl_hand = EXCLUDED.bowl_hand,
                        bowl_style = EXCLUDED.bowl_style
                """),
                changed,
            )
            s.commit()
        get_player_style.clear()
        return len(changed)

    if "batter_highlight_limits" not in st.session_state:
        st.session_state["batter_highlight_limits"] = {}
    current_limit = st.session_state["batter_highlight_limits"].get(
        str(st.session_state["selected_batter_id"]), MAX_HIGHLIGHTS_STEP
    )

    page_cache_key = (current_page, tuple(selected_grade), tuple(selected_team), tuple(selected_batter_names), hand_status, page_batter_ids_str, current_limit)
    if st.session_state.get("batter_highlights_page_key") != page_cache_key:
        page_highlights = get_highlights_for_batters(page_batter_ids_str, max_per_batter=current_limit)
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
        st.session_state["batter_highlights_page_df"] = page_highlights
        st.session_state["batter_highlights_page_key"] = page_cache_key
        st.session_state.pop("selected_batter_highlight_id", None)

    page_highlights_df = st.session_state.get("batter_highlights_page_df", pd.DataFrame())

    PANEL_HEIGHT = 560
    list_col, highlight_col = st.columns([2, 3])

    with list_col:
        with st.form("batter_styles_form", clear_on_submit=False):
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
                    bid = row["batter_id"]
                    is_selected = bid == st.session_state["selected_batter_id"]
                    selected_marker = " (current)" if is_selected else ""
                    st.markdown(f"**{row['batter_name']}** ({row['balls']} balls){selected_marker}")

                    chand, cselect = st.columns([1, 1])
                    with chand:
                        st.selectbox(
                            "Hand", batter_hand_opts,
                            index=dropdown_index(batter_hand_opts, row.get("batter_hand")),
                            key=f"bhand_{bid}", label_visibility="collapsed",
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
                st.session_state["batter_page_num"] = max(0, current_page - 1)
                st.session_state["selected_batter_id"] = None
                st.rerun()

        if next_page_clicked:
            try:
                saved_count = save_changed_rows(page_df)
                if saved_count:
                    st.success(f"Saved {saved_count} changes.")
            except Exception as e:
                st.error(f"Save failed: {e}")
            else:
                st.session_state["batter_page_num"] = min(total_pages - 1, current_page + 1)
                st.session_state["selected_batter_id"] = None
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

        clicked_batter_id = next((bid for bid, clicked in show_hl_clicks.items() if clicked), None)
        if clicked_batter_id is not None:
            st.session_state["selected_batter_id"] = clicked_batter_id
            st.session_state.pop("selected_batter_highlight_id", None)
            st.rerun()

    with highlight_col:
        selected_row = page_df[page_df["batter_id"] == st.session_state["selected_batter_id"]]
        if selected_row.empty:
            st.info("Select a batter to view highlights.")
            return
        selected_row = selected_row.iloc[0]
        selected_batter_id_str = str(selected_row["batter_id"])

        if page_highlights_df.empty:
            st.info("No highlights available for batters on this page.")
            return

        bh_sorted = page_highlights_df[page_highlights_df["batter_id"] == selected_batter_id_str].copy()
        if bh_sorted.empty:
            st.info("No highlights available for this batter.")
            return

        bh_sorted = bh_sorted.sort_values(
            ["day_1_start", "innings_number", "over", "ball_number"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)

        if (
            "selected_batter_highlight_id" not in st.session_state
            or st.session_state["selected_batter_highlight_id"] not in bh_sorted["highlight_id"].values
        ):
            st.session_state["selected_batter_highlight_id"] = bh_sorted.iloc[0]["highlight_id"]

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
                    if st.button("Play", key=f"batter_hl_play_{hid}"):
                        st.session_state["selected_batter_highlight_id"] = hid
                with yt_col:
                    yt_url = resolve_ball_video_url(
                        h_row.get("ball_time"), h_row.get("day1_stream_url"), h_row.get("day1_stream_start"),
                        h_row.get("day2_stream_url"), h_row.get("day2_stream_start"),
                    )
                    if yt_url:
                        st.link_button("YouTube", yt_url)
                    else:
                        st.button("YouTube", key=f"batter_hl_yt_disabled_{hid}", disabled=True)
                st.markdown(ROW_DIVIDER, unsafe_allow_html=True)

        show_more_clicked = st.button("Show more highlights", key="batter_show_more_highlights")
        if show_more_clicked:
            limits = st.session_state["batter_highlight_limits"]
            limits[selected_batter_id_str] = limits.get(selected_batter_id_str, MAX_HIGHLIGHTS_STEP) + MAX_HIGHLIGHTS_STEP
            st.rerun()

        sel_h = bh_sorted[bh_sorted["highlight_id"] == st.session_state["selected_batter_highlight_id"]].iloc[0]
        url = sel_h.get("highlight_url")
        if url:
            st.video(url, autoplay=True)
        else:
            st.info("No video URL available for this highlight.")
