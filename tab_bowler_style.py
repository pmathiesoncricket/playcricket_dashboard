import math

import streamlit as st
import pandas as pd
from sqlalchemy import text

from db import (
    conn,
    get_matches,
    get_bowler_summary,
    get_player_style,
    get_highlights_for_bowlers,
)

PACE_SPIN_OPTIONS = ["Pace", "Spin"]
BOWL_HAND_OPTIONS = ["Right", "Left"]
BOWL_STYLE_OPTIONS = ["Right Pace", "Left Pace", "LAOS", "Off Spin", "Leg Spin"]

ROW_DIVIDER = "<hr style='margin:2px 0;border:none;border-top:1px solid #333;'>"
MAX_HIGHLIGHTS = 10
BOWLERS_PER_PAGE = 20


def bowler_style_tab():
    st.header("Bowler Style")
    st.caption(
        "Review bowlers with missing style data in pages of 20. "
        "Each page preloads up to 10 highlights per visible bowler."
    )

    bowler_summary = get_bowler_summary()
    if bowler_summary.empty:
        st.info("No delivery data available.")
        return

    matches_df = get_matches()
    matches_df["day_1_start"] = pd.to_datetime(matches_df["day_1_start"])

    style_df = get_player_style()

    all_grades = sorted(
        {g for grades in bowler_summary["grades"] for g in (grades or []) if g}
    )

    filt_col1, filt_col2 = st.columns([2, 1])
    with filt_col1:
        selected_grade = st.multiselect(
            "Grade",
            all_grades,
            default=[],
            key="bowler_filter_grade",
        )
    with filt_col2:
        style_status = st.selectbox(
            "Bowl style",
            ["All", "Populated", "Unpopulated"],
            index=2,
            key="bowler_filter_style_status",
        )

    if selected_grade:
        bowler_summary = bowler_summary[
            bowler_summary["grades"].apply(
                lambda g: any(x in selected_grade for x in (g or []))
            )
        ]

    if bowler_summary.empty:
        st.info("No bowling deliveries match the current filters.")
        return

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

    st.caption(
        f"{total_bowlers} bowlers • page {current_page + 1} of {total_pages} • "
        f"showing {len(page_df)}"
    )

    pace_spin_opts = sorted(
        set(PACE_SPIN_OPTIONS) | set(style_df["pace_spin"].dropna().unique().tolist())
    )
    bowl_hand_opts = sorted(
        set(BOWL_HAND_OPTIONS) | set(style_df["bowl_hand"].dropna().unique().tolist())
    )
    bowl_style_opts = sorted(
        set(BOWL_STYLE_OPTIONS) | set(style_df["bowl_style"].dropna().unique().tolist())
    )

    page_bowler_ids = page_df["bowler_id"].tolist()
    page_bowler_ids_str = tuple(str(x) for x in page_bowler_ids)

    if (
        "selected_bowler_id" not in st.session_state
        or st.session_state["selected_bowler_id"] not in page_bowler_ids
    ):
        st.session_state["selected_bowler_id"] = page_df.iloc[0]["bowler_id"]

    def _dropdown_index(options, current_value):
        full_options = ["–"] + options
        if pd.notna(current_value) and current_value in options:
            return full_options.index(current_value)
        return 0

    def _save_changed_rows(df):
        changed = []
        for _, row in df.iterrows():
            bid = row["bowler_id"]
            ps_val = st.session_state.get(f"ps_{bid}", "–")
            hand_val = st.session_state.get(f"hand_{bid}", "–")
            style_val = st.session_state.get(f"style_{bid}", "–")

            new_pace_spin = None if ps_val == "–" else ps_val
            new_bowl_hand = None if hand_val == "–" else hand_val
            new_bowl_style = None if style_val == "–" else style_val

            orig_pace_spin = row.get("pace_spin") if pd.notna(row.get("pace_spin")) else None
            orig_bowl_hand = row.get("bowl_hand") if pd.notna(row.get("bowl_hand")) else None
            orig_bowl_style = row.get("bowl_style") if pd.notna(row.get("bowl_style")) else None

            if (
                new_pace_spin,
                new_bowl_hand,
                new_bowl_style,
            ) != (
                orig_pace_spin,
                orig_bowl_hand,
                orig_bowl_style,
            ):
                changed.append(
                    {
                        "player_id": str(bid),
                        "pace_spin": new_pace_spin,
                        "bowl_hand": new_bowl_hand,
                        "bowl_style": new_bowl_style,
                    }
                )

        if not changed:
            return 0

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

        get_player_style.clear()
        return len(changed)

    page_cache_key = (
        current_page,
        tuple(selected_grade),
        style_status,
        tuple(page_bowler_ids_str),
    )

    if st.session_state.get("bowler_highlights_page_key") != page_cache_key:
        page_highlights = get_highlights_for_bowlers(
            page_bowler_ids_str,
            max_per_bowler=MAX_HIGHLIGHTS,
        )

        if not page_highlights.empty:
            page_highlights = page_highlights.merge(
                matches_df[["match_id", "day_1_start", "home_team"]],
                on="match_id",
                how="left",
            )
            page_highlights["day_1_start"] = pd.to_datetime(page_highlights["day_1_start"])

        st.session_state["bowler_highlights_page_df"] = page_highlights
        st.session_state["bowler_highlights_page_key"] = page_cache_key
        st.session_state.pop("selected_bowler_highlight_id", None)

    page_highlights_df = st.session_state.get("bowler_highlights_page_df", pd.DataFrame())

    PANEL_HEIGHT = 560
    list_col, highlight_col = st.columns([2, 3])

    with list_col:
        nav1, nav2, nav3 = st.columns([1, 1, 2])

        with nav1:
            prev_page_clicked = st.button(
                "◀ Prev page",
                disabled=(current_page == 0),
                use_container_width=True,
                key="bowler_prev_page_btn",
            )
        with nav2:
            next_page_clicked = st.button(
                "Next page ▶",
                disabled=(current_page >= total_pages - 1),
                use_container_width=True,
                key="bowler_next_page_btn",
            )
        with nav3:
            save_all_clicked = st.button(
                "💾 Save page changes",
                type="primary",
                use_container_width=True,
                key="save_all_styles",
            )

        if prev_page_clicked:
            try:
                saved_count = _save_changed_rows(page_df)
                if saved_count:
                    st.success(f"Saved {saved_count} change(s).")
            except Exception as e:
                st.error(f"Save failed: {e}")
            else:
                st.session_state["bowler_page_num"] = max(0, current_page - 1)
                st.session_state["selected_bowler_id"] = None
                st.rerun()

        if next_page_clicked:
            try:
                saved_count = _save_changed_rows(page_df)
                if saved_count:
                    st.success(f"Saved {saved_count} change(s).")
            except Exception as e:
                st.error(f"Save failed: {e}")
            else:
                st.session_state["bowler_page_num"] = min(total_pages - 1, current_page + 1)
                st.session_state["selected_bowler_id"] = None
                st.rerun()

        if save_all_clicked:
            try:
                saved_count = _save_changed_rows(page_df)
                if saved_count:
                    st.success(f"Saved {saved_count} change(s).")
                else:
                    st.info("No changes to save.")
                st.rerun()
            except Exception as e:
                st.error(f"Bulk save failed: {e}")

        with st.form("bowler_styles_form", clear_on_submit=False):
            with st.container(height=PANEL_HEIGHT):
                for _, row in page_df.iterrows():
                    bid = row["bowler_id"]
                    is_selected = bid == st.session_state["selected_bowler_id"]
                    selected_marker = " ⟵ current" if is_selected else ""
                    st.markdown(
                        f"**{row['bowler_name']}** — {int(row['balls'])} balls{selected_marker}"
                    )

                    c_ps, c_hand, c_style, c_select = st.columns([1, 1, 1, 1])
                    with c_ps:
                        st.selectbox(
                            "Pace/Spin",
                            ["–"] + pace_spin_opts,
                            index=_dropdown_index(pace_spin_opts, row.get("pace_spin")),
                            key=f"ps_{bid}",
                            label_visibility="collapsed",
                        )
                    with c_hand:
                        st.selectbox(
                            "Hand",
                            ["–"] + bowl_hand_opts,
                            index=_dropdown_index(bowl_hand_opts, row.get("bowl_hand")),
                            key=f"hand_{bid}",
                            label_visibility="collapsed",
                        )
                    with c_style:
                        st.selectbox(
                            "Style",
                            ["–"] + bowl_style_opts,
                            index=_dropdown_index(bowl_style_opts, row.get("bowl_style")),
                            key=f"style_{bid}",
                            label_visibility="collapsed",
                        )
                    with c_select:
                        if st.form_submit_button(
                            "▶",
                            key=f"show_hl_{bid}",
                            help="Show highlights",
                            use_container_width=True,
                        ):
                            st.session_state["selected_bowler_id"] = bid
                            st.session_state.pop("selected_bowler_highlight_id", None)
                            st.rerun()

                    st.markdown(ROW_DIVIDER, unsafe_allow_html=True)

    with highlight_col:
        selected_row = page_df[
            page_df["bowler_id"] == st.session_state["selected_bowler_id"]
        ]

        if selected_row.empty:
            st.info("Select a bowler to view highlights.")
            return

        selected_row = selected_row.iloc[0]
        selected_bowler_id_str = str(selected_row["bowler_id"])

        st.subheader(f"Highlights — {selected_row['bowler_name']}")

        if page_highlights_df.empty:
            st.info("No highlights available for bowlers on this page.")
            return

        bh_sorted = page_highlights_df[
            page_highlights_df["bowler_id"] == selected_bowler_id_str
        ].copy()

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

        st.caption(f"{len(bh_sorted)} highlights shown — tap ▶ to play")

        list_height = 240
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
                        f"**{hrow.get('batter', '')}** — {hrow.get('highlight_type', '')}  \n"
                        f"{hrow.get('description', '')}  \n"
                        f"<span style='color:gray;font-size:0.8em'>{date_str} · {home_team}</span>",
                        unsafe_allow_html=True,
                    )
                with btn_col:
                    if st.button("▶", key=f"bowlerhl_play_{hid}"):
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
