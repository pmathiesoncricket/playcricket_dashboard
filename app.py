import streamlit as st

from tab_batting import batting_tab
from tab_batting_season import batting_season_tab
from tab_bowling import bowling_tab
from tab_bowler_style import bowler_style_tab
from tab_team import team_tab

st.set_page_config(page_title="PlayCricket Dashboard", layout="wide")
st.title("PlayCricket Dashboard")

tabs = st.tabs(["Batting", "Batting Season Report", "Bowling", "Bowler Style", "Team"])

with tabs[0]:
    batting_tab()
with tabs[1]:
    batting_season_tab()
with tabs[2]:
    bowling_tab()
with tabs[3]:
    bowler_style_tab()
with tabs[4]:
    team_tab()
