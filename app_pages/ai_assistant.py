# app_pages/ai_assistant.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import re
import os
import logging
import requests
from io import StringIO
from dotenv import load_dotenv
import warnings
import contextlib
import datetime

# Load environment variables from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, '.env'))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


matplotlib.use('Agg')
plt.style.use('default')
sns.set_palette("husl")

# Suppress matplotlib warning
warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')

# Consistent color scheme
COLORS = {
    'primary': '#1E3A8A',
    'secondary': '#3B82F6',
    'success': '#10B981',
    'warning': '#F59E0B',
    'error': '#EF4444',
    'background': '#F8F9FA',
    'text': '#1F2937',
    'accent': '#8B5CF6',
}


def load_dataframe(table_name):
    """Load DataFrame from session state with error handling"""
    try:
        df = st.session_state.get('dataframes', {}).get(table_name, pd.DataFrame())
        if df.empty:
            logger.warning(f"No DataFrame found for {table_name}")
            return pd.DataFrame()
        logger.info(f"Loaded {table_name} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading {table_name}: {e}")
        return pd.DataFrame()

def get_column_mapping(table_name, column_name):
    """Retrieve mapped column name, fallback to original if not found"""
    mappings = st.session_state.get('column_mappings', {}).get(table_name, {})
    return mappings.get(column_name, column_name)



def prepare_enriched_data():
    """Prepare a single enriched DataFrame from stored DataFrames"""
    try:
        # Load DataFrames
        data = {
            'Transactions': load_dataframe('Transactions'),
            'Client': load_dataframe('Client'),
            'Produit': load_dataframe('Produit'),
            'Magasin': load_dataframe('Magasin'),
            'Localisation': load_dataframe('Localisation'),
        }

        # Check if Transactions has data
        if data['Transactions'].empty:
            logger.error("No transaction data available")
            return pd.DataFrame(), "❌ No transaction data available"

        enriched_df = data['Transactions'].copy()
        enriched_info = ["Transactions"]

        # Define column lists
        client_cols = ['nom', 'genre', 'âge', 'ville', 'Tier_fidelité', 'premier_achat']
        produit_cols = ['nom_produit', 'catégorie', 'sous_catégorie', 'prix_vente']
        magasin_cols = ['nom_magasin', 'type', 'id_localisation']
        localisation_cols = ['ville', 'gouvernorat', 'pays']

        # Filter only available columns
        client_cols = [col for col in client_cols if get_column_mapping('Client', col) in data['Client'].columns]
        produit_cols = [col for col in produit_cols if get_column_mapping('Produit', col) in data['Produit'].columns]
        magasin_cols = [col for col in magasin_cols if get_column_mapping('Magasin', col) in data['Magasin'].columns]
        localisation_cols = [col for col in localisation_cols if get_column_mapping('Localisation', col) in data['Localisation'].columns]

        # --- Merge with Client ---
        if not data['Client'].empty:
            left_key = get_column_mapping('Transactions', 'id_client')
            right_key = get_column_mapping('Client', 'id_client')
            client_selected = [right_key] + [get_column_mapping('Client', col) for col in client_cols]

            enriched_df = enriched_df.merge(
                data['Client'][client_selected],
                how='left',
                left_on=left_key,
                right_on=right_key
            ).rename(columns={
                get_column_mapping('Client', col): f'client_{col}' for col in client_cols
            })

            enriched_info.append(f"Client ({len(client_cols)} columns)")

        # --- Merge with Produit ---
        if not data['Produit'].empty:
            left_key = get_column_mapping('Transactions', 'id_produit')
            right_key = get_column_mapping('Produit', 'id_produit')
            produit_selected = [right_key] + [get_column_mapping('Produit', col) for col in produit_cols]

            enriched_df = enriched_df.merge(
                data['Produit'][produit_selected],
                how='left',
                left_on=left_key,
                right_on=right_key
            ).rename(columns={
                get_column_mapping('Produit', col): f'produit_{col}' for col in produit_cols
            })

            enriched_info.append(f"Produit ({len(produit_cols)} columns)")

        # --- Merge with Magasin ---
        if not data['Magasin'].empty:
            left_key = get_column_mapping('Transactions', 'id_magasin')
            right_key = get_column_mapping('Magasin', 'id_magasin')
            magasin_selected = [right_key] + [get_column_mapping('Magasin', col) for col in magasin_cols]

            enriched_df = enriched_df.merge(
                data['Magasin'][magasin_selected],
                how='left',
                left_on=left_key,
                right_on=right_key
            ).rename(columns={
                get_column_mapping('Magasin', col): f'magasin_{col}' for col in magasin_cols
            })

            enriched_info.append(f"Magasin ({len(magasin_cols)} columns)")

        # --- Merge with Localisation ---
        if not data['Localisation'].empty and get_column_mapping('Magasin', 'id_localisation') in enriched_df.columns:
            left_key = get_column_mapping('Magasin', 'id_localisation')
            right_key = get_column_mapping('Localisation', 'id_localisation')
            localisation_selected = [right_key] + [get_column_mapping('Localisation', col) for col in localisation_cols]

            enriched_df = enriched_df.merge(
                data['Localisation'][localisation_selected],
                how='left',
                left_on=left_key,
                right_on=right_key
            ).rename(columns={
                get_column_mapping('Localisation', col): f'localisation_{col}' for col in localisation_cols
            })

            enriched_info.append(f"Localisation ({len(localisation_cols)} columns)")

        # Final summary
        summary = f"📊 Enriched Data: {' + '.join(enriched_info)}"
        logger.info(f"Enriched data created: {len(enriched_df)} rows, {len(enriched_df.columns)} columns")

        return enriched_df, summary

    except Exception as e:
        logger.error(f"Error preparing enriched data: {e}")
        
        return pd.DataFrame(), f"❌ Error: {str(e)}"


def extract_python_code(response_text):
    """Extract Python code from AI response"""
    code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        code = "\n".join(code_blocks)
        code = re.sub(r'plt\.show\(\)\s*', '', code)
        return code
    return None

def format_conversation_history(messages):
    """Format conversation history for AI context """
    formatted_history = []
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    for msg in recent_messages:
        if msg['role'] == 'user':
            formatted_history.append(f"User: {msg['content']}")
        else:
            content = msg['content']
            if len(content) > 200:
                content = content[:200] + "..."
            formatted_history.append(f"Assistant: {content}")
    return "\n".join(formatted_history)

def create_business_summary(enriched_df):
    """Create a concise business summary of the data"""
   
    summary_parts = []
        
    # Basic metrics
    total_rows = len(enriched_df)
    summary_parts.append(f"Dataset contains {total_rows:,} transaction records")
        
      # Sales metrics
    if 'montant_total' in enriched_df.columns:
        total_sales = enriched_df['montant_total'].sum()
        avg_transaction = enriched_df['montant_total'].mean()
        summary_parts.append(f"Total sales: {total_sales:,.2f}, Average transaction: {avg_transaction:.2f}")
        
        # Customer metrics
    if 'id_client' in enriched_df.columns:
        unique_customers = enriched_df['id_client'].nunique()
        summary_parts.append(f"Unique customers: {unique_customers:,}")
        
        # Product metrics
    if 'produit_nom_produit' in enriched_df.columns:
        unique_products = enriched_df['produit_nom_produit'].nunique()
        summary_parts.append(f"Unique products: {unique_products:,}")
        
        # Store metrics
    if 'magasin_nom_magasin' in enriched_df.columns:
        unique_stores = enriched_df['magasin_nom_magasin'].nunique()
        summary_parts.append(f"Stores: {unique_stores:,}")
        
    return " | ".join(summary_parts)


def process_ai_request(user_question, enriched_df, messages, api_endpoint, api_key, api_version):
    """Process AI request with enriched data, returning direct answers from df."""
    if enriched_df.empty:
        return {'role': 'assistant', 'content': "No data available for analysis."}

    try:
        # Business summary and conversation history
        business_summary = create_business_summary(enriched_df)
        conversation_history = format_conversation_history(messages)
        key_columns = list(enriched_df.columns)
        sample_data = {
            'magasins_count': enriched_df['magasin_nom_magasin'].nunique() if 'magasin_nom_magasin' in enriched_df.columns else 'N/A',
            'villes_count': enriched_df['localisation_ville'].nunique() if 'localisation_ville' in enriched_df.columns else 'N/A',
            'produits_count': enriched_df['produit_nom_produit'].nunique() if 'produit_nom_produit' in enriched_df.columns else 'N/A'
        }

                # Prompt
        prompt = f"""
        USER QUESTION: {user_question}
        BUSINESS DATA SUMMARY: {business_summary}
        STATISTICS: Rows: {len(enriched_df)}, Stores: {sample_data['magasins_count']}, Cities: {sample_data['villes_count']}, Products: {sample_data['produits_count']}
        AVAILABLE COLUMNS: {', '.join(key_columns)}
        RECENT CONVERSATION: {conversation_history}

        INSTRUCTIONS:
        1. Answer EVERY question by generating executable Python code using `df` (Pandas DataFrame, {len(enriched_df)} rows).
        2. Use exact column names from AVAILABLE COLUMNS: {', '.join(key_columns)}.
        3. Return response in the format:
        ```python
        # [Brief comment describing the code]
        [Your code here]
        print("[Result description]: [Result]")
        Always format printed outputs to exclude technical Pandas metadata (e.g., index names, "dtype", etc.).

        For client-related queries:Use 'client_nom' for names if available; otherwise, use 'id_client'.
        Use 'client_âge' for age if available.

        For listing queries (e.g., "name 3 clients"):Use df['client_nom'].drop_duplicates().sample(3).tolist() or similar to select random unique items.

        For superlatives (e.g., "top", "best", "most", "oldest"):Use df['column'].idxmax() or value_counts() to compute exact values.
        For follow-up questions (e.g., "who is the oldest among them"), extract the previously listed items from RECENT CONVERSATION (e.g., using regex to find "Three clients: ...") and filter df to those items.

        For visualizations ("plot", "répartition", "graphique", "visualisation", "histogramme"):Generate plot with matplotlib/seaborn, use plt.figure(figsize=(12, 8)), plt.tight_layout().
        Do NOT include plt.show().
        Ensure columns used exist in AVAILABLE COLUMNS.

        If required columns are missing, generate code to print: "Information not available due to missing columns: [list columns]."
        Do NOT invent data, use placeholders, or return non-code responses.
        Use concise, professional language in the print statement.
        PROHIBITED: Non-code responses, invented data, placeholders, emojis, plt.show(), mentioning code/process outside the code block.
        """



        # System prompt
        system_prompt = f"""
            Data analysis assistant for `df` ({len(enriched_df)} rows).
            OBJECTIVE: Generate executable Python code for all queries using only `df` data, concisely and professionally.
            PROCESS:
            1. Check `df` for required columns using AVAILABLE COLUMNS: {', '.join(key_columns)}.
            2. Generate Python code to extract real values from `df`.
            3. Return code in the format:
            ```python
            # [Brief comment describing the code]
            [Code]
            print("[Result description]: [Result]")
            Always format printed outputs to exclude technical Pandas metadata (e.g., index names, "dtype", etc.).

            For visualizations ("plot", "répartition", etc.), generate plot with matplotlib/seaborn, use plt.figure(figsize=(12, 8)), plt.tight_layout().
            If data is missing, generate code to print: "Information not available due to missing columns: [list columns]."
            For client-related queries, prefer 'client_nom' for names, 'client_âge' for age, or fall back to 'id_client'.
            For follow-up queries (e.g., "among them"), filter df to items listed in RECENT CONVERSATION.
            PROHIBITED: Non-code responses, invented data, placeholders, emojis, plt.show(), mentioning code/process outside code block.
            ALLOWED: Python code using real df data, concise print statements.
            COLUMNS: {', '.join(key_columns)}
            """



        # API request
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        api_url = f"{api_endpoint.rstrip('/')}/openai/deployments/gpt-4-turbo/chat/completions"
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0
        }

        response = requests.post(api_url, headers=headers, json=payload, params={"api-version": api_version}, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        if not response_data.get('choices'):
            raise ValueError("Invalid API response.")

        response_content = response_data['choices'][0]['message']['content'].strip()

        """# Validate response
        invalid_phrases = ["analyzing", "executing", "I will", "let me", "code", "client_nom", "produit_A", "placeholder"]
        if any(phrase in response_content.lower() for phrase in invalid_phrases):
            response_content = "Information not available." 
        """

        # Handle visualizations
        code_to_execute = extract_python_code(response_content)
        if code_to_execute:
            plt.clf()
            plt.switch_backend('Agg')
            exec_globals = {
                'plt': plt, 'sns': sns, 'df': enriched_df.copy(), 'pd': pd,
                'np': __import__('numpy'), 'matplotlib': matplotlib, 'datetime': datetime
            }
            exec_utils = """
def safe_get(df, column, default=None):
    return df[column] if column in df.columns else default
"""
            full_code = exec_utils + "\n" + code_to_execute

            with StringIO() as buf, contextlib.redirect_stdout(buf):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    local_vars = {}
                    exec(full_code, exec_globals, local_vars)
                    interpretation = buf.getvalue().strip() or "Visualization generated."

            fig = plt.gcf()
            clean_content = response_content.replace(f"```python\n{code_to_execute}\n```", "").strip()
            """
            if any(phrase in clean_content.lower() for phrase in invalid_phrases):
                clean_content = "Information not available."
            """

            response = {'role': 'assistant', 'content': clean_content, 'interpretation': interpretation}
            if len(fig.axes) > 0:
                fig.tight_layout()
                response['figure'] = fig
            return response

        # Validate required columns
        required_columns = []
        if "magasin" in user_question.lower():
            required_columns.append('magasin_nom_magasin')
        elif "produit" in user_question.lower():
            required_columns.append('produit_nom_produit')
        elif "client" in user_question.lower():
            required_columns.append('client_nom')
        if required_columns and not all(col in enriched_df.columns for col in required_columns):
            return {'role': 'assistant', 'content': f"Information ({', '.join(required_columns)}) not available."}

        return {'role': 'assistant', 'content': response_content}

    except Exception as e:
        logger.error(f"Error in process_ai_request: {str(e)}")
        return {'role': 'assistant', 'content': "Error processing request. Please try again."}

def textual_analysis_agent(query, df, api_endpoint, api_key, api_version, context=None):
    """Textual Analysis Agent: Generates Python code for textual responses"""
    try:
        prompt = f"""
        USER QUERY: {query}
        AVAILABLE COLUMNS: {', '.join(df.columns)}
        INSTRUCTIONS:
        - Generate Python code to answer the query using `df` (Pandas DataFrame, {len(df)} rows).
        - Focus on textual outputs (e.g., summaries, statistics, specific queries).
        - Use exact column names from AVAILABLE COLUMNS.
        - Return code in format:
          ```python
          # [Brief comment]
          [Code]
          print("[Result description]: [Result]")
          ```
        - If columns are missing, print: "Information not available due to missing columns: [list]."
        """
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        api_url = f"{api_endpoint.rstrip('/')}/openai/deployments/gpt-4-turbo/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0
        }
        response = requests.post(api_url, headers=headers, json=payload, params={"api-version": api_version}, timeout=30)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content'].strip()
        code = extract_python_code(content)
        if not code:
            return {'content': "No valid code generated.", 'interpretation': None, 'figure': None}
        
        exec_globals = {'plt': plt, 'sns': sns, 'df': df.copy(), 'pd': pd, 'np': __import__('numpy')}
        with StringIO() as buf, contextlib.redirect_stdout(buf):
            exec(code, exec_globals)
            interpretation = buf.getvalue().strip() or "Textual analysis completed."
        
        return {'content': content.replace(f"```python\n{code}\n```", "").strip(), 'interpretation': interpretation, 'figure': None}
    except Exception as e:
        logger.error(f"Textual Analysis Agent error: {str(e)}")
        return {'content': f"Error: {str(e)}", 'interpretation': None, 'figure': None}

def visual_analysis_agent(query, df, api_endpoint, api_key, api_version, context=None):
    """Visual Analysis Agent: Generates Python code for visualizations"""
    try:
        prompt = f"""
        USER QUERY: {query}
        AVAILABLE COLUMNS: {', '.join(df.columns)}
        INSTRUCTIONS:
        - Generate Python code to create a visualization (e.g., bar, line, histogram) using `df` ({len(df)} rows).
        - Use matplotlib/seaborn, set plt.figure(figsize=(12, 8)), plt.tight_layout().
        - Do NOT include plt.show().
        - Return code in format:
          ```python
          # [Brief comment]
          [Code]
          print("[Result description]: [Result]")
          ```
        - If columns are missing, print: "Information not available due to missing columns: [list]."
        """
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        api_url = f"{api_endpoint.rstrip('/')}/openai/deployments/gpt-4-turbo/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0
        }
        response = requests.post(api_url, headers=headers, json=payload, params={"api-version": api_version}, timeout=30)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content'].strip()
        code = extract_python_code(content)
        if not code:
            return {'content': "No valid visualization code generated.", 'interpretation': None, 'figure': None}
        
        plt.clf()
        exec_globals = {'plt': plt, 'sns': sns, 'df': df.copy(), 'pd': pd, 'np': __import__('numpy')}
        with StringIO() as buf, contextlib.redirect_stdout(buf):
            exec(code, exec_globals)
            interpretation = buf.getvalue().strip() or "Visualization generated."
        
        fig = plt.gcf()
        response = {'content': content.replace(f"```python\n{code}\n```", "").strip(), 'interpretation': interpretation}
        if len(fig.axes) > 0:
            fig.tight_layout()
            response['figure'] = fig
        return response
    except Exception as e:
        logger.error(f"Visual Analysis Agent error: {str(e)}")
        return {'content': f"Error: {str(e)}", 'interpretation': None, 'figure': None}

def agentic_orchestrator(query, df, messages, api_endpoint, api_key, api_version):
    """Orchestrator: Coordinates agentic processing for V2"""
    try:
        # Simple task classification (expand with Planner Agent for complex queries)
        query_lower = query.lower()
        is_visual = any(keyword in query_lower for keyword in ['plot', 'graph', 'visualize', 'histogram', 'chart'])
        
        # Shared context (e.g., conversation history)
        context = format_conversation_history(messages)
        
        if is_visual:
            response = visual_analysis_agent(query, df, api_endpoint, api_key, api_version, context)
        else:
            response = textual_analysis_agent(query, df, api_endpoint, api_key, api_version, context)
        
        return {'role': 'assistant', **response}
    except Exception as e:
        logger.error(f"Orchestrator error: {str(e)}")
        return {'role': 'assistant', 'content': f"Error processing request: {str(e)}", 'interpretation': None, 'figure': None}

def ai_assistant_page():
    """AI Assistant page with V1/V2 toggle"""
    # Header with dropdown
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<h1 style="color: {COLORS["primary"]};">🤖 AI Business Analyst</h1>', unsafe_allow_html=True)
    with col2:
        mode = st.selectbox(
            "Select Assistant Mode",
            ["V1 (Classic)", "V2 (BETA - Agentic)"],
            key="assistant_mode",
            help="Choose between classic (V1) and agentic (V2) modes"
        )
    
    st.markdown(f'<p style="color: {COLORS["text"]};">Ask about your business data for insights and visualizations!</p>', unsafe_allow_html=True)

    # Existing CSS 
    st.markdown("""
        <style>
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            position: relative;
            animation: fadeIn 0.5s ease-in;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 80px;
            margin-right: 20px;
            border: 1px solid #bbdefb;
        }
        .assistant-message {
            background-color: #f8f9fa;
            margin-right: 80px;
            margin-left: 20px;
            border: 1px solid #e9ecef;
        }
        .message-header {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 8px;
            font-weight: bold;
        }
        .user-header {
            color: #1976d2;
        }
        .assistant-header {
            color: #2e7d32;
        }
        .message-content {
            line-height: 1.6;
        }
        .interpretation-box {
            background-color: #f0f7ff;
            border-left: 4px solid #2196f3;
            padding: 12px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .data-status {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)

    # Prepare enriched data 
    with st.spinner("🔄 Preparing enriched data for AI analysis..."):
        if 'enriched_data' not in st.session_state or st.session_state.get('refresh_data', False):
            enriched_df, enriched_info = prepare_enriched_data()
            if enriched_df.empty:
                st.error("❌ No data available for analysis. Please check your CSV files.")
                return
            st.session_state.enriched_data = enriched_df
            st.session_state.enriched_info = enriched_info
            st.session_state.refresh_data = False

        enriched_df = st.session_state.enriched_data
        enriched_info = st.session_state.enriched_info

    # Display data status
    st.markdown(f"""
        <div class="data-status">
            <strong>📊 Data Status:</strong> {enriched_info}<br>
            <strong>📈 Ready for Analysis:</strong> {len(enriched_df):,} rows × {len(enriched_df.columns)} columns
        </div>
    """, unsafe_allow_html=True)

    # Sidebar 
    with st.sidebar:
        st.markdown(f'<h3 style="color: {COLORS["secondary"]};">🛠️ Assistant Settings</h3>', unsafe_allow_html=True)
        api_endpoint = os.getenv('EYQ_INCUBATOR_ENDPOINT')
        api_key = os.getenv('EYQ_INCUBATOR_KEY')
        api_version = os.getenv('API_VERSION', '2023-05-15')
        if api_endpoint and api_key:
            st.session_state.eyq_endpoint = api_endpoint
            st.session_state.eyq_api_key = api_key
            st.session_state.eyq_api_version = api_version
            st.success("✅ EYQ API configured!")
        else:
            st.error("❌ EYQ API configuration missing!")
            return
        st.markdown(f'<h3 style="color: {COLORS["secondary"]};">🚀 Quick Actions</h3>', unsafe_allow_html=True)
        if st.button("🔄 New Conversation"):
            st.session_state.ai_messages = []
            st.rerun()
        if st.button("📊 Refresh Data"):
            st.session_state.pop('enriched_data', None)
            st.session_state.pop('enriched_info', None)
            st.session_state.refresh_data = True
            st.rerun()
        if st.button("📥 Download Enriched Data"):
            if 'enriched_data' in st.session_state:
                enriched_df = st.session_state.enriched_data
                csv = enriched_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='enriched_business_data.csv',
                    mime='text/csv',
                    key='download_enriched_data'
                )
            else:
                st.error("❌ No enriched data available to download. Please refresh the data first.")
        st.markdown(f'<h3 style="color: {COLORS["secondary"]};">📋 Enriched Data Info</h3>', unsafe_allow_html=True)
        with st.expander("🔍 Data Details", expanded=False):
            st.write(f"**Total Rows:** {len(enriched_df):,}")
            st.write(f"**Total Columns:** {len(enriched_df.columns)}")
            transaction_cols = [col for col in enriched_df.columns if not col.startswith(('client_', 'produit_', 'magasin_', 'localisation_', 'employe_'))]
            client_cols = [col for col in enriched_df.columns if col.startswith('client_')]
            product_cols = [col for col in enriched_df.columns if col.startswith('produit_')]
            store_cols = [col for col in enriched_df.columns if col.startswith('magasin_')]
            location_cols = [col for col in enriched_df.columns if col.startswith('localisation_')]
            employee_cols = [col for col in enriched_df.columns if col.startswith('employe_')]
            st.write("**Column Categories:**")
            st.write(f"• Transaction: {len(transaction_cols)}")
            st.write(f"• Client: {len(client_cols)}")
            st.write(f"• Product: {len(product_cols)}")
            st.write(f"• Store: {len(store_cols)}")
            st.write(f"• Location: {len(location_cols)}")
            st.write(f"• Employee: {len(employee_cols)}")

    # Main chat interface 
    st.markdown("---")
    if 'ai_messages' not in st.session_state:
        st.session_state.ai_messages = []
    if not st.session_state.ai_messages:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header assistant-header">🤖 AI Business Analyst</div>
            <div class="message-content">
                👋 Hello! I'm your AI business analyst working with enriched transaction data. I can help you:
                <br>• 📊 Create comprehensive business visualizations
                <br>• 📈 Analyze customer behavior and segmentation
                <br>• 🛍️ Evaluate product performance and trends
                <br>• 🏪 Assess store performance and location analysis
                <br>• 💰 Generate financial insights and KPIs
                <br>• 🔍 Discover patterns and business opportunities
            </div>
        </div>
        """, unsafe_allow_html=True)
    with st.expander("📊 Preview: Enriched Business Data", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            display_cols = []
            important_cols = enriched_df.columns.tolist()
            for col in important_cols:
                if col in enriched_df.columns:
                    display_cols.append(col)
            for col in enriched_df.columns:
                if col not in display_cols and len(display_cols) < 12:
                    display_cols.append(col)
            st.dataframe(enriched_df[display_cols].head(10), use_container_width=True)
        with col2:
            st.write("**Quick Stats:**")
            if 'montant_total' in enriched_df.columns:
                total_sales = enriched_df['montant_total'].sum()
                st.metric("Total Sales", f"{total_sales:,.2f} TND")
            if 'id_transaction' in enriched_df.columns:
                total_transactions = len(enriched_df['id_transaction'].unique())
                st.metric("Total Transactions", f"{total_transactions:,}")
            if 'id_client' in enriched_df.columns:
                unique_customers = len(enriched_df['id_client'].unique())
                st.metric("Unique Customers", f"{unique_customers:,}")
            if 'produit_nom_produit' in enriched_df.columns:
                unique_products = len(enriched_df['produit_nom_produit'].unique())
                st.metric("Unique Products", f"{unique_products:,}")
        st.write("**Available columns:**")
        st.write(", ".join(enriched_df.columns))

    # Display chat history 
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.ai_messages:
        role = message['role']
        content = message.get('content', '')
        figure = message.get('figure', None)
        interpretation = message.get('interpretation', '')
        message_class = 'user-message' if role == 'user' else 'assistant-message'
        header_class = 'user-header' if role == 'user' else 'assistant-header'
        header_text = 'You' if role == 'user' else '🤖 AI Business Analyst'
        st.markdown(f"""
            <div class="chat-message {message_class}">
                <div class="message-header {header_class}">{header_text}</div>
                <div class="message-content">{content}</div>
        """, unsafe_allow_html=True)
        if figure:
            st.pyplot(figure)
        if interpretation:
            st.markdown(f"""
                <div class="interpretation-box">
                    <strong>🔍 Analysis:</strong><br>
                    {interpretation.replace('\n', '<br>')}
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input and processing
    user_input = st.chat_input(
        "💬 Ask your question about the data (e.g., 'Show top selling products by category' or 'Analyze sales trends over time')",
        key="chat_input"
    )
    if user_input and user_input.strip():
        st.session_state.ai_messages.append({'role': 'user', 'content': user_input})
        with st.spinner('🧠 Analyzing your request...'):
            if mode == "V1 (Classic)":
                response_message = process_ai_request(
                    user_input,
                    enriched_df,
                    st.session_state.ai_messages,
                    st.session_state.eyq_endpoint,
                    st.session_state.eyq_api_key,
                    st.session_state.eyq_api_version
                )
            else:  # V2 (BETA - Agentic)
                response_message = agentic_orchestrator(
                    user_input,
                    enriched_df,
                    st.session_state.ai_messages,
                    st.session_state.eyq_endpoint,
                    st.session_state.eyq_api_key,
                    st.session_state.eyq_api_version
                )
            st.session_state.ai_messages.append(response_message)
            st.rerun()

if __name__ == "__main__":
    ai_assistant_page()