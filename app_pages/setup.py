# pages/setup.py
import streamlit as st
import os
import pandas as pd
from utils.config import DB_TYPES
from utils.db_utils import create_connection, get_table_info
import logging
import time

logger = logging.getLogger(__name__)

def setup_page():
    st.header("🔌 Database Connection")
    
    # Progress indicator simplifié
    progress_col1, progress_col2 = st.columns(2)
    with progress_col1:
        db_connected = bool(st.session_state.get('source_engine'))
        st.metric("Database", "Connected ✅" if db_connected else "Not Connected ❌")
    with progress_col2:
        tables_found = len(st.session_state.get('source_tables', []))
        st.metric("Tables Found", f"{tables_found}")

    # --- Database Connection ---
    st.subheader("📂 Connect to Your Database")
    
    # Database type selection
    db_type = st.selectbox(
        "Select Database Type",
        options=list(DB_TYPES.keys()),
        help="Choose your database type"
    )

    # Connection interface based on type
    if db_type == "SQLite":
        handle_sqlite_connection()
    else:
        handle_external_db_connection(db_type)

    # Table Discovery
    if st.session_state.get('source_engine'):
        discover_tables()

    # Sidebar status
    show_sidebar_status()

def handle_sqlite_connection():
    """Handle SQLite database connection via file upload"""
    st.markdown("**Upload your SQLite database file:**")
    
    # Show current connection status if already connected
    if st.session_state.get('source_engine') and st.session_state.get('uploaded_filename'):
        st.success(f"✅ Currently connected to: **{st.session_state.uploaded_filename}**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Upload Different File", type="secondary"):
                clear_connection()
                st.rerun()
        with col2:
            if st.button("❌ Disconnect", type="secondary"):
                clear_connection()
                st.rerun()
                
        # Test Query Performance Section
        if st.session_state.get('source_tables'):
            with st.expander("🚀 Test Query Performance"):
                source_tables = st.session_state.get('source_tables', [])
                
                if 'selected_test_table' not in st.session_state:
                    st.session_state.selected_test_table = source_tables[0] if source_tables else None
                
                test_table = st.selectbox(
                    "Select table to test:", 
                    source_tables,
                    index=source_tables.index(st.session_state.selected_test_table) if st.session_state.selected_test_table in source_tables else 0,
                    key="test_table_selector"
                )
                
                if test_table != st.session_state.selected_test_table:
                    st.session_state.selected_test_table = test_table
                
                if st.button("Run Performance Tests"):
                    run_performance_tests(test_table)
        return
    
    # File upload interface
    uploaded_file = st.file_uploader(
        "Choose SQLite file",
        type=["sqlite", "db", "sqlite3"],
        help="Upload your SQLite database file"
    )
    
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)

def process_uploaded_file(uploaded_file):
    """Process uploaded SQLite file"""
    file_bytes = uploaded_file.read()
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    
    with st.spinner("Connecting to database..."):
        try:
            source_conn, source_engine = create_connection(DB_TYPES["SQLite"], {'db_path': tmp_path})
            if source_conn and source_engine:
                st.session_state.source_engine = source_engine
                st.session_state.source_conn = source_conn
                st.session_state.db_type = "SQLite"
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.connection_established = True
                st.success(f"✅ Connected to **{uploaded_file.name}** successfully!")
                
                # Immediately discover tables
                try:
                    source_tables, source_table_columns = get_table_info(source_engine)
                    st.session_state.source_tables = source_tables
                    st.session_state.source_table_columns = source_table_columns
                    st.info(f"📊 Found {len(source_tables)} tables. Scroll down to explore...")
                except Exception as e:
                    st.warning(f"Connected but couldn't analyze tables: {str(e)}")
                
                st.rerun()
            else:
                st.error("❌ Failed to connect to the database. Please check the file.")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

def handle_external_db_connection(db_type):
    """Handle PostgreSQL/MySQL database connections"""
    
    if (st.session_state.get('source_engine') and 
        st.session_state.get('db_type') == db_type and 
        st.session_state.get('db_params')):
        
        db_name = st.session_state.db_params.get('database', 'Unknown')
        st.success(f"✅ Currently connected to: **{db_name}** ({db_type})")
        
        if st.button("❌ Disconnect", type="secondary"):
            clear_connection()
            st.rerun()
        return
    
    st.markdown("**Enter your database connection details:**")
    
    with st.form("db_connection_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            host = st.text_input("Host", value="localhost")
            database = st.text_input("Database Name", placeholder="retail_db")
            username = st.text_input("Username")
        
        with col2:
            port = st.text_input("Port", value="5432" if db_type == "PostgreSQL" else "3306")
            password = st.text_input("Password", type="password")
        
        connect_btn = st.form_submit_button("🔗 Connect to Database", type="primary")
        
        if connect_btn:
            if not all([host, port, database, username]):
                st.error("Please fill in all required fields.")
                return
            
            connect_external_db(db_type, host, port, database, username, password)

def connect_external_db(db_type, host, port, database, username, password):
    """Connect to external database"""
    db_params = {
        'host': host,
        'port': port,
        'database': database,
        'user': username,
        'password': password
    }
    
    with st.spinner("Testing connection..."):
        try:
            source_conn, source_engine = create_connection(DB_TYPES[db_type], db_params)
            if source_conn and source_engine:
                st.session_state.source_engine = source_engine
                st.session_state.source_conn = source_conn
                st.session_state.db_type = db_type
                st.session_state.db_params = db_params
                st.session_state.connection_established = True
                st.success(f"✅ Connected to {db_type} database **{database}**!")
                
                # Immediately discover tables
                try:
                    source_tables, source_table_columns = get_table_info(source_engine)
                    st.session_state.source_tables = source_tables
                    st.session_state.source_table_columns = source_table_columns
                    st.info(f"📊 Found {len(source_tables)} tables. Scroll down to explore...")
                except Exception as e:
                    st.warning(f"Connected but couldn't analyze tables: {str(e)}")
                
                st.rerun()
            else:
                st.error("❌ Connection failed. Please check your credentials.")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")

def discover_tables():
    """Discover and display available tables"""
    st.subheader("📊 Database Tables")
    
    source_tables = st.session_state.get('source_tables', [])
    source_table_columns = st.session_state.get('source_table_columns', {})
    
    if source_tables:
        st.success(f"✅ Found {len(source_tables)} tables!")
        
        # Display tables in a grid
        cols = st.columns(min(3, len(source_tables)))
        for i, table in enumerate(source_tables):
            with cols[i % 3]:
                column_count = len(source_table_columns.get(table, []))
                st.info(f"**{table}**\n{column_count} columns")
        
        # Table inspector
        with st.expander("🔍 Inspect Table Structure"):
            selected_table = st.selectbox("Select table:", source_tables)
            if selected_table:
                columns = source_table_columns.get(selected_table, [])
                st.write("**Columns:**")
                for col in columns:
                    st.write(f"• {col}")
                
                # Data preview
                if st.button(f"Preview {selected_table} Data"):
                    preview_table_data(selected_table)
    else:
        st.warning("⚠️ No tables found in the database.")

def preview_table_data(table_name):
    """Preview data from selected table"""
    try:
        query = f"SELECT * FROM {table_name} LIMIT 5"
        start_time = time.time()
        
        df = pd.read_sql(query, st.session_state.source_engine)
        query_time = time.time() - start_time
        
        st.write(f"Query executed in {query_time:.2f} seconds")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Preview error: {str(e)}")

def run_performance_tests(test_table):
    """Run performance tests on selected table"""
    if test_table:
        st.write(f"Testing performance on: **{test_table}**")
        
        tests = [
            ("Row count", f"SELECT COUNT(*) as count FROM `{test_table}`"),
            ("Small sample", f"SELECT * FROM `{test_table}` LIMIT 5"),
            ("Large sample", f"SELECT * FROM `{test_table}` LIMIT 10000"),
        ]
        
        for test_name, query in tests:
            try:
                start = time.time()
                df = pd.read_sql(query, st.session_state.source_engine)
                elapsed = time.time() - start
                
                row_count = len(df) if 'LIMIT' in query else df.iloc[0, 0]
                st.metric(test_name, f"{elapsed:.3f}s ({row_count:,} rows)")
                
            except Exception as e:
                st.error(f"Error in {test_name}: {str(e)}")

def clear_connection():
    """Clear database connection and related state"""
    connection_keys = [
        'source_engine', 'source_conn', 'db_type', 'uploaded_filename',
        'source_tables', 'source_table_columns', 'connection_established', 
        'db_params', 'selected_test_table'
    ]
    for key in connection_keys:
        if key in st.session_state:
            del st.session_state[key]

def show_sidebar_status():
    """Show current status in sidebar"""
    st.sidebar.markdown("### 📊 Connection Status")
    
    # Database connection
    if st.session_state.get('source_engine'):
        if st.session_state.get('db_type') == 'SQLite':
            filename = st.session_state.get('uploaded_filename', 'SQLite DB')
            st.sidebar.success(f"✅ Database: {filename}")
        else:
            db_name = st.session_state.get('db_params', {}).get('database', 'Connected')
            st.sidebar.success(f"✅ Database: {db_name}")
    else:
        st.sidebar.error("❌ No database connected")
    
    # Tables discovered
    table_count = len(st.session_state.get('source_tables', []))
    if table_count > 0:
        st.sidebar.success(f"✅ {table_count} tables found")
    else:
        st.sidebar.error("❌ No tables discovered")
    
    # Next steps
    st.sidebar.markdown("---")
    if not st.session_state.get('source_engine'):
        st.sidebar.info("👆 Connect to your database above")
    elif not table_count:
        st.sidebar.info("🔍 Check your database connection")
    else:
        st.sidebar.success("🚀 Ready to explore data!")
        st.sidebar.info("Navigate to **Table Mapping** to configure your schema")