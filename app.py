import streamlit as st

from tab_batting import batting_tab
from tab_batting_season import batting_season_tab
from tab_bowling import bowling_tab
from tab_bowler_style import bowler_style_tab
from tab_match_summary import match_summary_tab
from tab_team import team_tab

st.set_page_config(page_title="PlayCricket Dashboard", layout="wide")
st.title("PlayCricket Dashboard")

# NOTE: st.tabs() runs the code inside EVERY `with tabs[i]:` block on every
# rerun, regardless of which tab is actually visible in the UI -- this is a
# Streamlit platform limitation, not something configurable. That means all
# six tabs' database queries fire on every interaction, not just the one
# you're looking at. If that becomes a problem again, the fix is either a
# "Load report" button gating each tab's body, or going back to a
# single-active-page selector (e.g. st.sidebar.radio) instead of tabs.
tabs = st.tabs(
    ["Batting", "Batting Season Report", "Bowling", "Bowler Style", "Match Summary", "Team"]
)

with tabs[0]:
    batting_tab()
with tabs[1]:
    batting_season_tab()
with tabs[2]:
    bowling_tab()
with tabs[3]:
    bowler_style_tab()
with tabs[4]:
    match_summary_tab()
with tabs[5]:
    team_tab()
