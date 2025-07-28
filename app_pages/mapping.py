# pages/mapping.py
import streamlit as st
import pandas as pd

import logging
from utils.db_utils import escape_table_name
from utils.etl_utils import transform_to_dataframe
from utils.config import DEFAULT_TABLE_MAPPINGS, DEFAULT_COLUMN_MAPPINGS

logger = logging.getLogger(__name__)

def mapping_page():
    st.markdown('<h1 style="color: #1e3a8a;">🗺️ Table & Column Mapping</h1>', unsafe_allow_html=True)
    
    # Check prerequisites
    if not st.session_state.get('source_engine'):
        st.error("⚠️ Please connect to a database first on the Setup page.")
        st.markdown('<p style="color: #4b5563;">Go to <a href="/Database_Setup" target="_self">Database Setup</a> to connect.</p>', unsafe_allow_html=True)
        return

    # Progress indicators
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    with progress_col1:
        mapped_tables = len([t for t, s in st.session_state.get('table_mapping', {}).items() if s != 'None'])
        total_tables = len(DEFAULT_TABLE_MAPPINGS)
        st.metric("Tables Mapped", f"{mapped_tables}/{total_tables}")
    with progress_col2:
        mapped_columns = len(st.session_state.get('column_mappings', {}))
        st.metric("Column Mappings", f"{mapped_columns}/{total_tables}")
    with progress_col3:
        exports_done = len(st.session_state.get('dataframes', {}))
        st.metric("Data Exports", f"{exports_done}")

    st.markdown("---")

    # Main mapping interface
    handle_table_mapping()
    handle_column_mapping()

    # Bulk export section
    show_bulk_export_section()

    # Export summary
    if st.session_state.get('dataframes'):
        show_export_summary()

def handle_table_mapping():
    """Handle table mapping interface"""
    st.markdown('<h2 style="color: #3b82f6;">📋 Table Mapping</h2>', unsafe_allow_html=True)
    
    source_tables = st.session_state.get('source_tables', [])
    if not source_tables:
        st.warning("⚠️ No tables found in the database. Check your connection.")
        return

    # Initialize table_mapping in session state
    if 'table_mapping' not in st.session_state:
        st.session_state.table_mapping = {target: 'None' for target in DEFAULT_TABLE_MAPPINGS.keys()}

    with st.form("table_mapping_form"):
        st.markdown("**Map your database tables to the expected schema:**")
        mapping_changed = False
        cols = st.columns(3)

        for i, target_table in enumerate(DEFAULT_TABLE_MAPPINGS.keys()):
            with cols[i % 3]:
                options = ['None'] + source_tables
                current_selection = st.session_state.table_mapping.get(target_table, 'None')
                current_index = options.index(current_selection) if current_selection in options else 0
                
                selected_table = st.selectbox(
                    f"**{target_table}**",
                    options=options,
                    index=current_index,
                    key=f"table_map_{target_table}",
                    help=f"Map expected table '{target_table}' to a source table"
                )

                if selected_table != st.session_state.table_mapping[target_table]:
                    st.session_state.table_mapping[target_table] = selected_table
                    mapping_changed = True

        if st.form_submit_button("💾 Save Table Mappings", type="primary"):
            if mapping_changed:
                st.success("✅ Table mappings updated!")
            else:
                st.info("ℹ️ No changes to table mappings.")

def handle_column_mapping():
    """Handle column mapping interface"""
    st.markdown('<h2 style="color: #3b82f6;">🔗 Column Mapping</h2>', unsafe_allow_html=True)
    
    table_mapping = st.session_state.get('table_mapping', {})
    source_table_columns = st.session_state.get('source_table_columns', {})
    
    # Initialize column_mappings in session state
    if 'column_mappings' not in st.session_state:
        st.session_state.column_mappings = {}

    mapped_tables = [(target, source) for target, source in table_mapping.items() if source != 'None']
    
    if not mapped_tables:
        st.info("ℹ️ Please map at least one table before configuring columns.")
        return

    for target_table, source_table in mapped_tables:
        with st.expander(f"🔧 Map Columns for {target_table} ← {source_table}", expanded=False):
            expected_columns = list(DEFAULT_COLUMN_MAPPINGS.get(target_table, {}).keys())
            source_columns = source_table_columns.get(source_table, [])
            
            # Display column info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Expected columns ({len(expected_columns)}):**")
                st.caption(", ".join(expected_columns[:4]) + ("..." if len(expected_columns) > 4 else ""))
            with col2:
                st.write(f"**Available columns ({len(source_columns)}):**")
                st.caption(", ".join(source_columns[:4]) + ("..." if len(source_columns) > 4 else ""))

            # Column mapping
            column_mapping = configure_column_mapping(target_table, expected_columns, source_columns)
            
            # Show export status
            if column_mapping:
                show_table_export_status(target_table, source_table)
                
                # Data preview
                st.subheader("👀 Data Preview")
                if st.checkbox(f"Show {source_table} data preview", key=f"preview_check_{source_table}"):
                    show_data_preview(source_table)

def configure_column_mapping(target_table, expected_columns, source_columns):
    """Configure column mappings for a table"""
    column_mappings = st.session_state.get('column_mappings', {})
    if target_table not in column_mappings:
        column_mappings[target_table] = {}
    
    default_mappings = DEFAULT_COLUMN_MAPPINGS.get(target_table, {})
    
    cols = st.columns(3)
    mapping_changed = False
    
    # Itérer directement sur la session_state pour ne pas perdre les mises à jour
    for i, target_col in enumerate(expected_columns):
        with cols[i % 3]:
            options = ['None'] + source_columns
            # Lire la valeur courante dans la session_state ou valeur par défaut
            current_selection = column_mappings[target_table].get(target_col, default_mappings.get(target_col, 'None'))
            current_index = options.index(current_selection) if current_selection in options else 0
            
            selected_col = st.selectbox(
                f"**{target_col}**",
                options=options,
                index=current_index,
                key=f"col_map_{target_table}_{target_col}",
                help=f"Map {target_col} to a source column"
            )
            
            # Mise à jour immédiate dans st.session_state
            if selected_col != 'None':
                if column_mappings[target_table].get(target_col) != selected_col:
                    column_mappings[target_table][target_col] = selected_col
                    mapping_changed = True
            else:
                # Supprimer la clé si None sélectionné
                if target_col in column_mappings[target_table]:
                    del column_mappings[target_table][target_col]
                    mapping_changed = True
    
    # Mise à jour de st.session_state en une fois à la fin
    if mapping_changed:
        st.session_state.column_mappings = column_mappings
        st.success(f"✅ {target_table} column mappings updated!")
    
    # Affichage état mapping
    mapped_count = len(column_mappings.get(target_table, {}))
    total_count = len(expected_columns)
    if mapped_count == total_count and mapped_count > 0:
        st.success(f"✅ All {total_count} columns mapped!")
    elif mapped_count > 0:
        unmapped = [col for col in expected_columns if col not in column_mappings.get(target_table, {})]
        st.info(f"📊 {mapped_count}/{total_count} columns mapped")
        st.caption(f"Unmapped: {', '.join(unmapped)}")
    else:
        st.warning("⚠️ No columns mapped yet")
    
    return column_mappings.get(target_table, {})


def show_table_export_status(target_table,source_table):
    """Show export status for a table without individual export buttons"""
    dataframes = st.session_state.get('dataframes', {})
    column_mappings = st.session_state.get('column_mappings', {})
    
    st.subheader("📤 Export Status")
    if target_table in dataframes:
        st.success(f"✅ **{target_table}** exported successfully")
        df_preview = dataframes[target_table].head(3)
        st.write("**Preview (first 3 rows):**")
        st.dataframe(df_preview, use_container_width=True)
    else:
        column_mapping = column_mappings.get(target_table, {})
        if column_mapping:
            st.info(f"📋 **{target_table}** ready for export ({len(column_mapping)} columns mapped)")
        else:
            st.warning("⚠️ Configure column mappings to enable export")

def show_bulk_export_section():
    """Show bulk export section for all mapped tables"""
    table_mapping = st.session_state.get('table_mapping', {})
    column_mappings = st.session_state.get('column_mappings', {})
    dataframes = st.session_state.get('dataframes', {})
    
    tables_with_mappings = [
        (target, source) for target, source in table_mapping.items()
        if source != 'None' and target in column_mappings and column_mappings[target]
    ]
    
    if not tables_with_mappings:
        st.info("ℹ️ Map tables and columns to enable bulk export.")
        return
    
    st.markdown('<h2 style="color: #3b82f6;">🚀 Bulk Export</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"**{len(tables_with_mappings)} tables** ready for export")
        with st.expander("📋 Export Preview", expanded=False):
            for target_table, source_table in tables_with_mappings:
                cols_mapped = len(column_mappings.get(target_table, {}))
                st.write(f"• **{target_table}** ← {source_table} ({cols_mapped} columns)")
    
    with col2:
        already_exported = [t for t, _ in tables_with_mappings if t in dataframes]
        not_exported = [t for t, _ in tables_with_mappings if t not in dataframes]
        if already_exported:
            st.success(f"✅ {len(already_exported)} already exported")
        if not_exported:
            st.warning(f"⏳ {len(not_exported)} pending export")
    
    if not_exported:
        if st.button("🚀 **Export All Mapped Tables**", type="primary", use_container_width=True):
            perform_bulk_export(tables_with_mappings)
    
    if already_exported:
        if st.button("🔄 **Re-export All Tables**", type="secondary", use_container_width=True):
            perform_bulk_export(tables_with_mappings, force_reexport=True)

def perform_bulk_export(tables_with_mappings, force_reexport=False):
    """Perform bulk export of all mapped tables to in-memory DataFrames"""
    column_mappings = st.session_state.get('column_mappings', {})
    if 'dataframes' not in st.session_state:
        st.session_state.dataframes = {}
    
    total_tables = len(tables_with_mappings)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_exports = 0
    failed_exports = []
    
    for i, (target_table, source_table) in enumerate(tables_with_mappings):
        if not force_reexport and target_table in st.session_state.dataframes:
            successful_exports += 1
            progress_bar.progress((i + 1) / total_tables)
            status_text.text(f"Skipping {target_table} (already exported)...")
            continue
        
        status_text.text(f"Exporting {target_table}... ({i + 1}/{total_tables})")
        try:
            column_mapping = column_mappings.get(target_table, {})
            df, success = transform_to_dataframe(
                engine=st.session_state.source_engine,
                table_name=source_table,
                columns=list(column_mapping.values()),
                target_columns=list(column_mapping.keys())
                
            )


            
            if success and not df.empty:
                # Rename columns to target columns to reflect the mapping
                df.columns = list(column_mapping.keys())
                st.session_state.dataframes[target_table] = df
                successful_exports += 1
                status_text.text(f"✅ {target_table} exported successfully with {len(df)} rows!")
                logger.info(f"Exported {target_table} with {len(df)} rows")
            else:
                failed_exports.append(target_table)
                status_text.text(f"❌ Failed to export {target_table}: Empty table or error")
                logger.warning(f"Failed to export {target_table}: Empty table or error")
        except Exception as e:
            failed_exports.append(target_table)
            status_text.text(f"❌ Error exporting {target_table}: {str(e)}")
            logger.error(f"Bulk export error for {target_table}: {e}")
        
        progress_bar.progress((i + 1) / total_tables)
    
    progress_bar.progress(1.0)
    if successful_exports == total_tables:
        status_text.empty()
        st.success(f"🎉 **All {total_tables} tables exported successfully!**")
    elif successful_exports > 0:
        status_text.empty()
        st.warning(f"⚠️ **{successful_exports}/{total_tables} tables exported**")
        if failed_exports:
            st.error(f"Failed exports: {', '.join(failed_exports)}")
    else:
        status_text.empty()
        st.error("❌ **No tables exported successfully**")
        if failed_exports:
            st.error(f"Failed exports: {', '.join(failed_exports)}")
    
    st.rerun()

def show_data_preview(source_table):
    """Show data preview for a source table"""
    try:
        query = f"SELECT * FROM {escape_table_name(source_table, st.session_state.get('db_type'), None)} LIMIT 5"
        df_preview = pd.read_sql(query, st.session_state.source_engine)
        if not df_preview.empty:
            st.dataframe(df_preview, use_container_width=True)
        else:
            st.info("Table is empty")
    except Exception as e:
        st.error(f"Preview error: {e}")

def show_export_summary():
    """Show summary of all exported DataFrames"""
    st.markdown('<h2 style="color: #3b82f6;">📄 Export Summary</h2>', unsafe_allow_html=True)
    
    dataframes = st.session_state.get('dataframes', {})
    column_mappings = st.session_state.get('column_mappings', {})
    
    if not dataframes:
        st.info("No exports completed yet.")
        return
    
    total_exported = len(dataframes)
    total_mapped = len([t for t in column_mappings.values() if t])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tables Exported", total_exported)
    with col2:
        st.metric("Tables Mapped", total_mapped)
    with col3:
        completion = f"{(total_exported/total_mapped*100):.0f}%" if total_mapped > 0 else "0%"
        st.metric("Completion", completion)
    
    st.subheader("Exported DataFrames")
    for table_name in dataframes:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{table_name}**")
            mapped_cols = len(column_mappings.get(table_name, {}))
            st.caption(f"{mapped_cols} columns mapped")
        with col2:
            df = dataframes[table_name]
            st.write(f"{len(df):,} rows")
            st.caption(f"Columns: {', '.join(df.columns[:3])}{'...' if len(df.columns) > 3 else ''}")
        with col3:
            csv = df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="⬇️",
                data=csv,
                file_name=f"{table_name}.csv",
                mime="text/csv",
                key=f"final_download_{table_name}"
            )
    
    if total_exported == total_mapped and total_exported > 0:
        st.success("🎉 All tables exported successfully! Ready for analysis.")