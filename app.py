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
# rerun, regardless of which tab is actually visible -- so with the old
# tabs-based layout, batting_tab() (and every other tab's DB queries) fired
# on every single page load/interaction, even if you were looking at a
# different tab. Using a sidebar radio "page selector" instead means only
# the currently-selected page's function ever actually executes, and
# nothing loads at all until a page other than Home is chosen.
PAGES = {
    "Home": None,
    "Batting": batting_tab,
    "Batting Season Report": batting_season_tab,
    "Bowling": bowling_tab,
    "Bowler Style": bowler_style_tab,
    "Match Summary": match_summary_tab,
    "Team": team_tab,
}

st.sidebar.markdown("### Navigation")
selected_page = st.sidebar.radio("Go to", list(PAGES.keys()), key="nav_page")

if selected_page == "Home":
    st.header("Welcome")
    st.markdown(
        "Pick a report from the **Navigation** list in the sidebar to get started.\n\n"
        "Nothing is loaded from the database on this landing page \u2014 each report "
        "only queries data once you actually select it, to keep the app responsive."
    )
else:
    page_fn = PAGES[selected_page]
    page_fn()
