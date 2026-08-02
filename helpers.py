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
                           default_options: list | None = None):
    sanitize_multiselect_state(key, options)
    kwargs = {}
    if key not in st.session_state:
        kwargs["default"] = default_options if default_options is not None else []
    return container.multiselect(label, options, key=key, **kwargs)


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
