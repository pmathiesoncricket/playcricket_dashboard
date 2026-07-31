import os
import streamlit as st
import pandas as pd
from supabase import create_client, Client

st.set_page_config(page_title="PlayCricket Dashboard", layout="wide")

# --- Supabase client ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Helper: query matches ---
@st.cache_data(ttl=300)
def get_matches():
    resp = supabase.table("matches").select("*").order("day_1_start", desc=True).limit(100).execute()
    return pd.DataFrame(resp.data)

# --- App layout ---
st.title("PlayCricket Dashboard")

tabs = st.tabs(["Batting", "Bowling", "Team"])

# Batting tab
with tabs[0]:
    st.header("Batting")
    matches = get_matches()
    if not matches.empty:
        st.subheader("Recent matches")
        st.dataframe(
            matches[["day_1_start", "home_team", "away_team", "grade", "result_text"]].head(10)
        )
    else:
        st.info("No matches found yet.")

# Bowling tab
with tabs[1]:
    st.header("Bowling")
    st.write("Bowling stats coming here.")

# Team tab
with tabs[2]:
    st.header("Team")
    st.write("Team-level analytics coming here.")
