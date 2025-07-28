# utils/db_utils.py
import sqlite3
from sqlalchemy import create_engine, inspect
import mysql.connector
import psycopg2
from urllib.parse import quote_plus
import logging
import streamlit as st

logger = logging.getLogger(__name__)

def create_connection(db_type, db_params):
    try:
        if db_type == "sqlite":
            engine = create_engine(f"sqlite:///{db_params['db_path']}")
            conn = sqlite3.connect(db_params['db_path'])
            # Validate connection
            conn.execute("SELECT 1")
        elif db_type == "postgresql":
            conn_str = (
                f"postgresql://{db_params['user']}:{quote_plus(db_params['password'])}@"
                f"{db_params['host']}:{db_params['port']}/{db_params['database']}"
            )
            engine = create_engine(conn_str)
            conn = psycopg2.connect(
                dbname=db_params['database'],
                user=db_params['user'],
                password=db_params['password'],
                host=db_params['host'],
                port=db_params['port']
            )
            # Validate connection
            conn.cursor().execute("SELECT 1")
        elif db_type == "mysql":
            conn_str = (
                f"mysql+mysqlconnector://{db_params['user']}:{quote_plus(db_params['password'])}@"
                f"{db_params['host']}:{db_params['port']}/{db_params['database']}"
            )
            engine = create_engine(conn_str)
            conn = mysql.connector.connect(
                user=db_params['user'],
                password=db_params['password'],
                host=db_params['host'],
                port=db_params['port'],
                database=db_params['database']
            )
            # Validate connection
            conn.cursor().execute("SELECT 1")
        else:
            raise ValueError("Unsupported database type")
        
        # Ensure connections are closed properly
        if db_type != "sqlite":  # SQLite doesn't need explicit connection pooling
            engine = create_engine(conn_str, pool_pre_ping=True, pool_timeout=30)
        
        return conn, engine
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        st.error(f"Failed to connect to database: {e}")
        return None, None
    finally:
        # Close raw connection if created
        if 'conn' in locals() and conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")

def get_table_info(engine, schema=None):
    try:
        inspector = inspect(engine)
        # Use specified schema for PostgreSQL, default to 'public' or None for others
        if engine.dialect.name == 'postgresql' and schema:
            tables = inspector.get_table_names(schema=schema)
        else:
            tables = inspector.get_table_names()
        
        table_columns = {}
        for table in tables:
            try:
                columns = [col['name'] for col in inspector.get_columns(table, schema=schema)]
                table_columns[table] = columns
            except Exception as e:
                logger.warning(f"Could not retrieve columns for table {table}: {e}")
                table_columns[table] = []
        
        if not tables:
            logger.warning("No tables found in the database")
            st.warning("No tables found in the database. Check schema or database permissions.")
        
        return tables, table_columns
    except Exception as e:
        logger.error(f"Error getting table info: {e}")
        st.error(f"Error getting table info: {e}")
        return [], {}

def escape_table_name(table_name, db_type, schema=None):
    if schema and db_type == "postgresql":
        table_name = f"{schema}.{table_name}"
    if db_type == "mysql":
        return f"`{table_name}`"
    return f'"{table_name}"'