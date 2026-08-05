
import streamlit as st

from tab_batting import batting_tab
from tab_batting_season import batting_season_tab
from tab_bowling import bowling_tab
from tab_bowler_style import bowler_style_tab
from tab_batter_style import batter_style_tab
from tab_match_summary import match_summary_tab
from tab_team import team_tab

st.set_page_config(page_title="PlayCricket Dashboard", layout="wide")
st.title("PlayCricket Dashboard")

# A sidebar dropdown selector (instead of st.tabs) means only the currently
# selected page's function actually executes on each rerun. st.tabs()
# unconditionally runs the code inside EVERY tab on every rerun regardless
# of which one is visible -- this is what was driving the load times up
# with 7 tabs' worth of DB queries firing at once. With a selectbox, the
# other six pages' queries simply never run until you pick them.

PAGES = {
    "Batting": batting_tab,
    "Batting Season Report": batting_season_tab,
    "Bowling": bowling_tab,
    "Bowler Style": bowler_style_tab,
    "Batter Style": batter_style_tab,
    "Match Summary": match_summary_tab,
    "Team": team_tab,
}

st.sidebar.markdown("### Navigation")
selected_page = st.sidebar.selectbox("Report", list(PAGES.keys()), key="nav_page")

page_fn = PAGES[selected_page]
page_fn()
