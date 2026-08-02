import streamlit as st

from tab_batting import batting_tab
from tab_bowling import bowling_tab
from tab_bowler_style import bowler_style_tab
from tab_team import team_tab


st.set_page_config(page_title="PlayCricket Dashboard", layout="wide")

st.title("PlayCricket Dashboard")

tabs = st.tabs(["Batting", "Bowling", "Bowler Style", "Team"])

with tabs[0]:
    batting_tab()

with tabs[1]:
    bowling_tab()

with tabs[2]:
    bowler_style_tab()

with tabs[3]:
    team_tab()
