# main.py

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"
import streamlit as st
from app_pages.setup import setup_page
from app_pages.mapping import mapping_page
from app_pages.dashboard_owner import business_owner_dashboard
from app_pages.ai_assistant import ai_assistant_page
from app_pages.evaluation import evaluation_page



# Set page configuration for a wide layout and custom theme
st.set_page_config(
    page_title="Retail Database Management App",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    /* Header styling */
    h1 {
        color: #1e3a8a;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Subheader styling */
    h2 {
        color: #3b82f6;
    }
    /* Custom button styling */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1e3a8a;
        color: white;
    }
    /* Card-like containers for features */
    .feature-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        text-align: center;
    }
    /* Welcome message */
    .welcome-text {
        font-size: 18px;
        color: #4b5563;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 50px;
        font-size: 14px;
    }
    /* Selectbox styling */
    .stSelectbox {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 5px;
    }
    .stSelectbox > div > div {
        border: 1px solid #3b82f6;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

def home_page():
    """Render the home page with a beautiful design"""
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1>🏬 Retail Database Management App</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">Welcome to Prismatik, your all-in-one solution to seamlessly manage, explore, and analyze your retail data. Simplify your database tasks and unlock insights effortlessly!</p>', unsafe_allow_html=True)
    
    # Features section
    st.markdown('<h2>✨ Key Features</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>🔌 Database Connectivity</h3>
                <p>Connect seamlessly to SQLite, PostgreSQL, or MySQL databases.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>📊 Table Exploration</h3>
                <p>Discover and inspect tables with real-time previews and performance tests.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3>🚀 Data Insights</h3>
                <p>Analyze your retail data with intuitive tools and visualizations.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown('<div style="text-align: center; margin-top: 30px;">', unsafe_allow_html=True)
    if st.button("Get Started with Database Setup 🚀"):
        st.session_state.page = "setup"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="Streamlit Demo"</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)



def main():
    # Initialize session state defaults
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'db_config' not in st.session_state:
        st.session_state.db_config = {
            'db_type': None,
            'host': None,
            'port': None,
            'database': None,
            'user': None,
            'password': None,
            'db_path': None
        }
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = None
    if 'column_mappings' not in st.session_state:
        st.session_state.column_mappings = {}
    if 'csv_paths' not in st.session_state:
        st.session_state.csv_paths = {}
    if 'dataframes' not in st.session_state:
        st.session_state.dataframes = {}  # Initialize for in-memory storage
    if 'refresh_data' not in st.session_state:
        st.session_state.refresh_data = False
        
    # Navigation with selectbox
    st.sidebar.title("🧭 Navigation")
    page_options = ["home", "setup", "mapping", "dashboard", "ai_assistant", "evaluation"]
    page_display = ["Home", "Database Setup", "Table & Column Mapping", "Business Owner Dashboard", "AI Business Analyst","Evaluation"]
    page = st.sidebar.selectbox(
        "Select Page",
        options=page_options,
        format_func=lambda x: page_display[page_options.index(x)],
        index=page_options.index(st.session_state.page),
        key="page_selector"
    )

    # Update page state only when selection changes
    if page != st.session_state.page:
        st.session_state.page = page

    # Render the selected page
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "setup":
        setup_page()
    elif st.session_state.page == "mapping":
        mapping_page()
    elif st.session_state.page == "dashboard":
        business_owner_dashboard()
    elif st.session_state.page == "ai_assistant":
        ai_assistant_page()
    elif st.session_state.page == "evaluation":
        evaluation_page()


if __name__ == "__main__":
    main()