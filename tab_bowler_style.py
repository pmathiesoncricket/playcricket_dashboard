import streamlit as st
import pandas as pd
from sqlalchemy import text

from db import conn, get_matches, get_bowler_summary, get_player_style, get_highlights

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

        bowler_summary = get_bowler_summary()
    if bowler_summary.empty:
        st.info("No delivery data available.")
        return

    matches_df = get_matches()
    matches_df["day_1_start"] = pd.to_datetime(matches_df["day_1_start"])

    style_df = get_player_style()

        all_grades = sorted({g for grades in bowler_summary["grades"] for g in (grades or []) if g})

    filt_col1, filt_col2 = st.columns([2, 1])
    with filt_col1:
        selected_grade = st.multiselect(
            "Grade", all_grades, default=[], key="bowler_filter_grade"
        )
    with filt_col2:
        style_status = st.selectbox(
            "Bowl style",
            ["All", "Populated", "Unpopulated"],
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
