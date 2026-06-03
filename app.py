import streamlit as st
from scout import graph

st.set_page_config(
    page_title="Football Scout AI",
    page_icon="⚽",
    layout="centered",
)

st.markdown("""
<style>
    .stApp {
        background-color: #0d2b1a;
        color: #ffffff;
    }

    section[data-testid="stMain"] > div {
        background-color: #0d2b1a;
    }

    .scout-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 2px;
        padding: 1rem 0 0.2rem 0;
    }

    .scout-subtitle {
        text-align: center;
        font-size: 0.95rem;
        color: #81c784;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    .stTextInput label {
        color: #81c784 !important;
        font-size: 0.9rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 600;
    }

    .stTextInput > div > div > input {
        background-color: #1a4a2e !important;
        color: #ffffff !important;
        border: 2px solid #2d7a4a !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #5a9a6a !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4caf50 !important;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.25) !important;
    }

    .stButton > button {
        background-color: #2d7a4a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        background-color: #3d9960 !important;
    }

    .player-card {
        background-color: #1a4a2e;
        border: 1px solid #2d7a4a;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0 1rem 0;
    }

    .player-card-name {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #2d7a4a;
        letter-spacing: 1px;
    }

    .player-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid #1f5c35;
    }

    .player-row:last-of-type {
        border-bottom: none;
    }

    .row-label {
        color: #81c784;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .row-value {
        color: #e8f5e9;
        font-size: 0.95rem;
        font-weight: 500;
    }

    .position-badge {
        display: inline-block;
        color: #ffffff;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 1rem;
    }

    .badge-forward    { background-color: #c0392b; }
    .badge-midfielder { background-color: #2d7a4a; }
    .badge-defender   { background-color: #1a5276; }
    .badge-goalkeeper { background-color: #6c3483; }
    .badge-manager    { background-color: #7d6608; }

    .report-container {
        background-color: #132e1c;
        border-left: 4px solid #4caf50;
        border-radius: 0 8px 8px 0;
        padding: 1.5rem 1.8rem;
        margin-top: 0.5rem;
    }

    .report-header {
        font-size: 0.85rem;
        font-weight: 700;
        color: #4caf50;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 1rem;
    }

    .report-body {
        color: #d4edda;
        font-size: 0.97rem;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    .divider {
        border: none;
        border-top: 1px solid #2d7a4a;
        margin: 1.2rem 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="scout-title">⚽ Football Scout AI</div>', unsafe_allow_html=True)
st.markdown('<div class="scout-subtitle">AI-Powered Player Scouting Reports</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

player_name = st.text_input("Player or Manager Name", placeholder="e.g. Erling Haaland, Virgil van Dijk, Pep Guardiola")

if st.button("Scout Player"):
    if not player_name.strip():
        st.warning("Please enter a player name.")
    else:
        with st.spinner("Fetching player data and generating report..."):
            result = graph.invoke(
                {
                    "player_name": player_name.strip(),
                    "player_data": {},
                    "position_category": "",
                    "scouting_report": "",
                }
            )

        player_data = result.get("player_data", {})
        position_category = result.get("position_category", "")
        scouting_report = result.get("scouting_report", "")

        if not player_data:
            st.error(f"No player found for **{player_name}**. Check the spelling and try again.")
        else:
            st.markdown(f"""
            <div class="player-card">
                <div class="player-card-name">{player_data.get('strTeam', '') and player_name.title()}</div>
                <div class="player-row">
                    <span class="row-label">Club</span>
                    <span class="row-value">{player_data.get('strTeam', 'Unknown')}</span>
                </div>
                <div class="player-row">
                    <span class="row-label">Position</span>
                    <span class="row-value">{player_data.get('strPosition', 'Unknown')}</span>
                </div>
                <div class="player-row">
                    <span class="row-label">Nationality</span>
                    <span class="row-value">{player_data.get('strNationality', 'Unknown')}</span>
                </div>
                <div class="player-row">
                    <span class="row-label">Date of Birth</span>
                    <span class="row-value">{player_data.get('dateBorn', 'Unknown')}</span>
                </div>
                <div>
                    <span class="position-badge badge-{position_category}">{position_category}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="report-container"><div class="report-header">Scouting Report</div></div>', unsafe_allow_html=True)
            st.write(scouting_report)
