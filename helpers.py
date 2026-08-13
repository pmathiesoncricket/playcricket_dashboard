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
    (stale selections no longer present in `options` get dropped) and an
    optional "quick add" helper.

    enable_quick_add=True renders a small text input + button UNDER the
    multiselect: type a keyword and click "Add matches" to select every
    currently-available option containing that keyword in one go. This
    exists because Streamlit's own multiselect search box resets back to
    the full option list after every click -- there's no parameter to keep
    it open, so selecting several options that share a substring (e.g.
    every grade with "Woolnough" in the name) normally means re-typing the
    search term after each individual click. The mutation happens BEFORE
    the multiselect widget is instantiated in this same run, since
    Streamlit disallows writing to st.session_state[key] for a widget
    that's already been created earlier in the same script execution.
    """
    sanitize_multiselect_state(key, options)

    if enable_quick_add and options:
        quick_add_key = f"{key}_quick_add_text"
        search_text = container.text_input(
            f"Quick-add to {label}",
            key=quick_add_key,
            placeholder=f"Type a keyword shared by several {label} options, then click Add matches",
            help=(
                "Streamlit's dropdown search resets after each click, so this lets you add every "
                "matching option in one go instead of re-searching for each one."
            ),
        )
        if container.button(f"Add matches to {label}", key=f"{key}_quick_add_btn"):
            term = search_text.strip().lower()
            if term:
                matches = [opt for opt in options if term in opt.lower()]
                if matches:
                    current = st.session_state.get(
                        key, list(default_options) if default_options is not None else []
                    )
                    merged = list(dict.fromkeys(current + matches))
                    if merged != current:
                        st.session_state[key] = merged

    kwargs = {}
    if key not in st.session_state:
        kwargs["default"] = default_options if default_options is not None else []
    container.multiselect(label, options, key=key, **kwargs)
    return st.session_state.get(key, [])


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
