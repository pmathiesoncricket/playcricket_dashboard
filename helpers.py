import streamlit as st
import pandas as pd

# Colour palette centred on maroon, used for every bar/histogram chart.
MAROON = "#73173F"
MAROON_SHADES = ["#73173F", "#C97292", "#9E3A5D", "#4A0F29", "#D9A0B5"]

SEGMENT_ORDER = ["1\u201310", "11\u201320", "21\u201330", "31\u201350", "51\u201375", "76+"]


def add_season_column(df: pd.DataFrame, date_col: str = "day_1_start") -> pd.DataFrame:
    """
    Add season column with July-June seasons, formatted as 'YYYY/YYYY'.
    """
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


def sanitize_multiselect_state(key: str, valid_options: list) -> None:
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in valid_options]


def cascading_multiselect(container, label: str, options: list, key: str,
                           default_options: list | None = None, enable_quick_add: bool = False):
    """
    Standard multiselect used throughout the app, with cascading behaviour
    (stale selections no longer present in `options` get dropped), a full
    list of whatever's currently selected shown as plain text underneath,
    and an optional "search + tick" helper for quickly selecting several
    options that share a keyword.

    The plain-text "currently selected" list exists because Streamlit's
    own selected-value pills truncate long names with no way to attach a
    native hover tooltip via the public API -- this is the practical
    workaround for actually being able to read a long grade/team name.

    enable_quick_add=True adds a search box under the multiselect; typing
    into it shows a live checklist of every currently-available option
    containing that keyword, and ticking any of them adds/removes it from
    the selection immediately (no extra button click needed). This exists
    because Streamlit's own multiselect dropdown resets its search box
    back to the full option list after every click, so there's no way to
    select several options sharing a keyword (e.g. every grade with
    "Woolnough" in the name) without retyping the search after each pick.
    The search box here is a separate widget from the checkboxes, so
    ticking one doesn't clear or affect it -- it stays exactly as typed
    while you work through the rest of the matches.
    """
    sanitize_multiselect_state(key, options)

    current_selection = st.session_state.get(
        key, list(default_options) if default_options is not None else []
    )

    if enable_quick_add and options:
        search_term = container.text_input(
            f"Search {label}",
            key=f"{key}_quick_search",
            placeholder=f"Type a keyword shared by several {label} options",
        )
        term = search_term.strip().lower()
        if term:
            matches = [opt for opt in options if term in opt.lower()]
            if matches:
                new_selection = list(current_selection)
                changed = False
                for opt in matches:
                    cb_key = f"{key}_cb_{opt}"
                    # Sync the checkbox to the TRUE current selection before
                    # it's instantiated this run, so it can never drift out
                    # of sync with the main multiselect (e.g. if an option
                    # was instead removed via its pill's "x").
                    st.session_state[cb_key] = opt in new_selection
                    checked = container.checkbox(opt, key=cb_key)
                    if checked and opt not in new_selection:
                        new_selection.append(opt)
                        changed = True
                    elif not checked and opt in new_selection:
                        new_selection.remove(opt)
                        changed = True
                if changed:
                    st.session_state[key] = new_selection
                    current_selection = new_selection
            else:
                container.caption(f"No {label} options match \u201c{search_term}\u201d.")

    kwargs = {}
    if key not in st.session_state:
        kwargs["default"] = default_options if default_options is not None else []
    container.multiselect(label, options, key=key, **kwargs)

    selected = st.session_state.get(key, [])
    if selected:
        container.caption(f"**{label}:** " + ", ".join(selected))

    return selected


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
