import pandas as pd
from sqlalchemy import inspect, text
import streamlit as st
import logging

logger = logging.getLogger(__name__)

def transform_to_dataframe(engine, table_name, columns, target_columns, chunk_size=10000):
    """
    Transform and return data from source table as a DataFrame with optional transformations.
    
    Args:
        engine: SQLAlchemy engine for database connection
        table_name: Name of source table
        columns: List of source column names to extract
        target_columns: List of target column names for output
        chunk_size: Number of rows to process at once
    
    Returns:
        pandas.DataFrame: Transformed DataFrame, or empty DataFrame if unsuccessful
        bool: True if successful, False otherwise
    """
    try:
        # Validate inputs
        if not columns:
            st.warning(f"No columns mapped for table '{table_name}'. Skipping.")
            logger.warning(f"No columns mapped for table '{table_name}'.")
            return pd.DataFrame(), False

        if len(columns) != len(target_columns):
            st.error(f"Mismatch between source columns ({len(columns)}) and target columns ({len(target_columns)})")
            logger.error(f"Column count mismatch for table '{table_name}'")
            return pd.DataFrame(), False

        # Normalize column names to UTF-8
        columns = [col.encode('utf-8').decode('utf-8') for col in columns]
        target_columns = [col.encode('utf-8').decode('utf-8') for col in target_columns]

        # Inspect table structure
        inspector = inspect(engine)
        actual_columns = [col['name'].encode('utf-8').decode('utf-8') for col in inspector.get_columns(table_name)]
        logger.info(f"Actual columns in {table_name}: {actual_columns}")
        
        # Validate columns exist in source table
        valid_columns = []
        valid_target_columns = []
        
        for src_col, tgt_col in zip(columns, target_columns):
            if src_col in actual_columns:
                valid_columns.append(src_col)
                valid_target_columns.append(tgt_col)
            else:
                st.warning(f"Column '{src_col}' not found in source table '{table_name}'. Skipping.")
                logger.warning(f"Column '{src_col}' not found in source table '{table_name}'.")

        if not valid_columns:
            st.warning(f"No valid columns to process for table '{table_name}'. Skipping.")
            logger.warning(f"No valid columns to process for table '{table_name}'.")
            return pd.DataFrame(), False

        # Build query with proper column escaping
        escaped_columns = []
        for col in valid_columns:
            if engine.dialect.name == 'sqlite':
                escaped_columns.append(f'`{col}`')
            elif engine.dialect.name == 'mysql':
                escaped_columns.append(f'`{col}`')
            elif engine.dialect.name == 'postgresql':
                escaped_columns.append(f'"{col}"')
            else:
                escaped_columns.append(f'[{col}]')
        
        # Escape table name
        if engine.dialect.name == 'sqlite':
            escaped_table = f'`{table_name}`'
        elif engine.dialect.name == 'mysql':
            escaped_table = f'`{table_name}`'
        elif engine.dialect.name == 'postgresql':
            escaped_table = f'"{table_name}"'
        else:
            escaped_table = f'[{table_name}]'
        
        query = f"SELECT {', '.join(escaped_columns)} FROM {escaped_table}"
        logger.info(f"Executing query: {query}")
        
        # Process data in chunks
        try:
            chunks = pd.read_sql(query, engine, chunksize=chunk_size)
        except Exception as e:
            chunks = pd.read_sql(text(query), engine, chunksize=chunk_size)
        
        dfs = []
        total_rows = 0
        
        for df in chunks:
            if df.empty:
                continue
            # Rename columns to target names
            df.columns = valid_target_columns
            dfs.append(df)
            total_rows += len(df)

        if not dfs:
            st.warning(f"Table '{table_name}' is empty. Returning empty DataFrame.")
            logger.warning(f"Table '{table_name}' is empty.")
            return pd.DataFrame(columns=valid_target_columns), True

        # Concatenate chunks
        final_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Processed {total_rows} rows from table '{table_name}' with columns: {list(final_df.columns)}")
        
        return final_df, True
        
    except Exception as e:
        logger.error(f"Error processing table '{table_name}': {e}")
        st.error(f"Error processing table '{table_name}': {e}")
        return pd.DataFrame(), False


def get_table_row_count(engine, table_name):
    """
    Get the number of rows in a table.
    
    Args:
        engine: SQLAlchemy engine
        table_name: Name of table
        
    Returns:
        int: Number of rows, or 0 if error
    """
    try:
        # Escape table name based on database type
        if engine.dialect.name == 'sqlite':
            escaped_table = f'`{table_name}`'
        elif engine.dialect.name == 'mysql':
            escaped_table = f'`{table_name}`'
        elif engine.dialect.name == 'postgresql':
            escaped_table = f'"{table_name}"'
        else:
            escaped_table = f'[{table_name}]'
            
        query = f"SELECT COUNT(*) as row_count FROM {escaped_table}"
        
        try:
            result = pd.read_sql(query, engine)
        except:
            result = pd.read_sql(text(query), engine)
            
        return result['row_count'].iloc[0]
    except Exception as e:
        logger.error(f"Error getting row count for {table_name}: {e}")
        return 0

def validate_export_file(file_path, expected_columns):
    """
    Validate that an exported CSV file has the expected structure.
    
    Args:
        file_path: Path to CSV file
        expected_columns: List of expected column names
        
    Returns:
        dict: Validation results with 'valid', 'message', 'row_count', 'column_count'
    """
    try:
        # Read just the header and first few rows
        df = pd.read_csv(file_path, nrows=5)
        
        actual_columns = list(df.columns)
        row_count = len(pd.read_csv(file_path))  # Get full row count
        
        # Check columns match
        if set(actual_columns) == set(expected_columns):
            return {
                'valid': True,
                'message': f'File valid: {row_count} rows, {len(actual_columns)} columns',
                'row_count': row_count,
                'column_count': len(actual_columns)
            }
        else:
            missing = set(expected_columns) - set(actual_columns)
            extra = set(actual_columns) - set(expected_columns)
            message = f"Column mismatch. Missing: {missing}, Extra: {extra}"
            return {
                'valid': False,
                'message': message,
                'row_count': row_count,
                'column_count': len(actual_columns)
            }
            
    except Exception as e:
        return {
            'valid': False,
            'message': f'Error reading file: {e}',
            'row_count': 0,
            'column_count': 0
        }