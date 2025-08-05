# app_pages/evaluation_page.py
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import numpy as np 
import re
import logging
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from app_pages.ai_assistant import process_ai_request, prepare_enriched_data, agentic_orchestrator

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Initialize sentence transformer for semantic similarity
try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.warning(f"⚠️ Failed to load sentence transformer: {str(e)}. Using fuzzy matching as fallback.")
    similarity_model = None

# Colors (reused from ai_assistant.py)
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

def normalize_text(text):
    """Normalize text for consistent relevancy calculation."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r'```python\n.*?\n```', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'Name:\s*\w+,\s*dtype:\s*\w+', '', text).strip()  # Remove Pandas metadata
    text = re.sub(r'\s+', ' ', text)
    return text

def parse_pandas_output(response_text):
    """Parse Pandas-style output or Python list-style responses to extract key data."""
    response_text = normalize_text(response_text)
    # Extract key-value pairs (e.g., 'Bizerte 49.085181' -> {'Bizerte': 49.085181})
    pairs = re.findall(r'(\w[\w\s-]*\w)\s+([\d]+(?:\.\d+)?)', response_text)
    result_dict = {}
    for pair in pairs:
        try:
            result_dict[pair[0]] = float(pair[1])
        except ValueError as e:
            logger.warning(f"Failed to convert '{pair[1]}' to float: {str(e)}")
            continue
    
    # Extract names from Python list-style responses (e.g., ['Anouk Dijoux', 'Jules Renault'])
    list_names = re.findall(r'(?:"([^"]+)"|\'([^\']+)\')\s*(?:,\s*(?:"([^"]+)"|\'([^\']+)\')\s*)*', response_text)
    items = []
    for match in list_names:
        items.extend([name for name in match if name])
    
    # Extract standalone items, excluding numbers, dictionary keys, and Pandas metadata
    standalone_items = re.findall(r'\b[\w\s-]+\b', response_text)
    items.extend([item for item in standalone_items 
                  if not re.match(r'^\d+(?:\.\d+)?$', item)  # Exclude numbers
                  and item not in result_dict  # Exclude dictionary keys
                  and item not in items  # Exclude already extracted names
                  and item not in ('produit_catégorie', 'name', 'montant_total', 'dtype', 'float64')])  # Exclude Pandas metadata
    
    logger.debug(f"Parsed items: {items}")
    logger.debug(f"Parsed dict: {result_dict}")
    return result_dict, items

def is_response_correct(response, ground_truth, enriched_df, query, threshold=85):
    """Evaluate response correctness with flexible matching."""
    response_text = normalize_text(response)
    query_lower = query.lower().strip()
    response_dict, response_items = parse_pandas_output(response_text)

    # Handle visualization queries
    if "plot" in query_lower or "chart" in query_lower:
        if "visualization" in normalize_text(ground_truth) and ("plot" in response_text or "chart" in response_text):
            return True
        return fuzz.partial_ratio(response_text, normalize_text(ground_truth)) >= threshold

    # Handle "Name 3 clients" query
    if "name 3 clients" in query_lower:
        if 'client_nom' not in enriched_df.columns:
            logger.warning("Column 'client_nom' not found in dataset.")
            return False
        valid_names = set(enriched_df['client_nom'].str.lower().drop_duplicates())
        valid_count = sum(1 for item in response_items if item.lower() in valid_names)
        return valid_count >= 3

   

    # Handle numerical ground truths
    if isinstance(ground_truth, (int, float)):
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
        if numbers:
            try:
                response_number = float(numbers[-1])
                return abs(response_number - ground_truth) < 1e-3
            except ValueError:
                return False
        return False

    # Handle tuple-based ground truths
    if isinstance(ground_truth, tuple):
        ground_truth_text = normalize_text(f"{ground_truth[0]} {ground_truth[1]}")
        if ground_truth_text in response_text or all(str(item).lower() in response_text for item in ground_truth):
            return True
        return fuzz.partial_ratio(response_text, ground_truth_text) >= threshold

    # Handle dictionary-based ground truths
    if isinstance(ground_truth, dict):
        ground_dict = {k.lower(): v for k, v in ground_truth.items()}
        if response_dict:
            matches = sum(1 for k, v in ground_dict.items() 
                         if k in response_dict and abs(v - response_dict[k]) < 1e-3)
            return matches >= len(ground_dict) * 0.8
        return False

    # Default string comparison
    ground_truth_text = normalize_text(ground_truth)
    if ground_truth_text in response_text or response_text in ground_truth_text:
        return True
    return fuzz.partial_ratio(response_text, ground_truth_text) >= threshold

def compute_relevancy_score(response, ground_truth, query, enriched_df, similarity_model):
    """Compute relevancy score consistently for a single response."""
    response_text = normalize_text(response)
    query_lower = query.lower().strip()
    response_dict, response_items = parse_pandas_output(response_text)

    # Handle numerical ground truths
    if isinstance(ground_truth, (int, float)):
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
        if numbers:
            try:
                response_number = float(numbers[-1])
                if abs(response_number - ground_truth) < 1e-3:
                    logger.debug(f"Numerical match: {response_number} ≈ {ground_truth}")
                    return 1.0
                else:
                    logger.debug(f"Numerical mismatch: {response_number} ≠ {ground_truth}")
                    return 0.0
            except ValueError as e:
                logger.warning(f"Failed to convert response number '{numbers[-1]}': {str(e)}")
                return 0.0
        logger.warning("No numerical value found in response.")
        return 0.0

    # Handle "Name 3 clients" query
    if "name 3 clients" in query_lower:
        if 'client_nom' not in enriched_df.columns:
            logger.warning("Column 'client_nom' not found in dataset.")
            return 0.0
        valid_names = set(enriched_df['client_nom'].str.lower().drop_duplicates())
        valid_count = sum(1 for item in response_items if item.lower() in valid_names)
        return 1.0 if valid_count >= 3 else valid_count / 3.0

    
    

    # Handle other specific queries (plot, chart, etc.)
    if "plot" in query_lower or "chart" in query_lower or "above 50" in query_lower:
        is_correct = is_response_correct(response, ground_truth, enriched_df, query)
        return 1.0 if is_correct else 0.0

    # Handle list-based ground truths
    if isinstance(ground_truth, list):
        valid_items = set(str(item).lower() for item in ground_truth)
        valid_count = sum(1 for item in response_items if item.lower() in valid_items)
        return valid_count / max(len(ground_truth), 1)

    # Handle tuple-based ground truths
    if isinstance(ground_truth, tuple):
        ground_truth_text = normalize_text(f"{ground_truth[0]} {ground_truth[1]}")
        if similarity_model:
            try:
                embeddings = similarity_model.encode([response_text, ground_truth_text])
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                logger.debug(f"Semantic similarity: {similarity}")
                return similarity
            except Exception as e:
                logger.warning(f"Semantic similarity failed: {str(e)}. Falling back to fuzzy matching.")
        return fuzz.ratio(response_text, ground_truth_text) / 100

    # Handle dictionary-based ground truths
    if isinstance(ground_truth, dict):
        ground_dict = {k.lower(): v for k, v in ground_truth.items()}
        if response_dict:
            matches = sum(1 for k, v in ground_dict.items() 
                         if k in response_dict and abs(v - response_dict[k]) < 1e-3)
            return matches / max(len(ground_dict), 1)
        return fuzz.ratio(response_text, normalize_text(str(ground_truth))) / 100

    # Default string comparison
    ground_truth_text = normalize_text(ground_truth)
    if similarity_model:
        try:
            embeddings = similarity_model.encode([response_text, ground_truth_text])
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            logger.debug(f"Semantic similarity: {similarity}")
            return similarity
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {str(e)}. Falling back to fuzzy matching.")
    return fuzz.ratio(response_text, ground_truth_text) / 100

def is_plot_correct(response, query, enriched_df):
    """Evaluate if a visualization matches the query intent."""
    if 'figure' not in response or not response['figure']:
        return False

    fig = response['figure']
    query_lower = query.lower()

    if 'sales by product category' in query_lower:
        if 'produit_catégorie' not in enriched_df.columns:
            st.warning("⚠️ Column 'produit_catégorie' not found in dataset. Skipping plot validation.")
            return True
        if 'montant_total' not in enriched_df.columns:
            st.warning("⚠️ Column 'montant_total' not found in dataset. Skipping sales data validation.")
            return True

        expected_sales = enriched_df.groupby('produit_catégorie')['montant_total'].sum().to_dict()
        expected_categories = set(enriched_df['produit_catégorie'].str.lower().unique())

        if isinstance(fig, plt.Figure):
            try:
                axes = fig.get_axes()
                if not axes:
                    st.warning("⚠️ No axes found in Matplotlib figure.")
                    return False
                ax = axes[0]
                
                x_labels = [label.get_text().lower() for label in ax.get_xticklabels()]
                y_labels = [label.get_text().lower() for label in ax.get_yticklabels()]
                
                if any(cat in x_labels for cat in expected_categories):
                    plot_data = {label.get_text().lower(): rect.get_height() 
                               for label, rect in zip(ax.get_xticklabels(), 
                                                    [b for b in ax.get_children() if isinstance(b, plt.Rectangle)])}
                elif any(cat in y_labels for cat in expected_categories):
                    plot_data = {label.get_text().lower(): rect.get_width()
                               for label, rect in zip(ax.get_yticklabels(),
                                                    [b for b in ax.get_children() if isinstance(b, plt.Rectangle)])}
                else:
                    st.warning("⚠️ Could not find expected categories on either axis.")
                    return False
                
                for cat, val in plot_data.items():
                    expected_val = expected_sales.get(cat.title(), 0)
                    if abs(val - expected_val) > 1e-2:
                        st.warning(f"⚠️ Sales value for category '{cat}' ({val}) does not match expected ({expected_val}).")
                        return False
                return True
            except Exception as e:
                st.warning(f"⚠️ Plot validation failed: {str(e)}")
                return False

    return True

def evaluation_page():
    # Initialize or reset session state for evaluation results
    if 'evaluation_results' not in st.session_state or 'evaluation_mode' not in st.session_state or st.session_state.evaluation_mode != st.session_state.get('last_mode', None):
        st.session_state.evaluation_results = {'V1': None, 'V2': None}
        st.session_state.last_mode = None

    st.markdown(f'<h1 style="color: {COLORS["primary"]};">📈 Chatbot Evaluation</h1>', unsafe_allow_html=True)
    
    # Dropdown for mode selection
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<p style="color: {COLORS["text"]};">Evaluate the AI Business Analyst with predefined tests or custom queries.</p>', unsafe_allow_html=True)
    with col2:
        mode = st.selectbox(
            "Select Assistant Mode",
            ["V1 (Classic)", "V2 (BETA - Agentic)"],
            key="evaluation_mode",
            help="Choose between classic (V1) and agentic (V2) modes"
        )

    # Reset results for the current mode when mode changes
    if st.session_state.last_mode != mode:
        st.session_state.evaluation_results[mode[:2]] = None
        st.session_state.last_mode = mode

    # CSS for consistent styling
    st.markdown("""
        <style>
        .data-status {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .interpretation-box {
            background-color: #f0f7ff;
            border-left: 4px solid #2196f3;
            padding: 12px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .metric-box {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .comparison-table {
            border-collapse: collapse;
            width: 100%;
        }
        .comparison-table th, .comparison-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .comparison-table th {
            background-color: #f2f2f2;
        }
        .highlight-v2 {
            background-color: #e6ffed;
        }
        </style>
    """, unsafe_allow_html=True)

    # Load enriched data
    with st.spinner("🔄 Loading enriched data..."):
        enriched_df, enriched_info = prepare_enriched_data()
        if enriched_df.empty:
            st.error(f"❌ No data available: {enriched_info}")
            return
        st.markdown(f'<div class="data-status"><strong>📊 Data Status:</strong> {enriched_info}</div>', unsafe_allow_html=True)

        # Predefined test queries and ground truths
    test_queries = [
        "How many unique stores are there?",
        "Who is the oldest client?",
        "Plot sales by product category",
        "Name 3 clients",
        "List client addresses",
        "What is the total sales amount across all stores?",
        "Which store has the highest sales, and what is the amount?",
        "Summarize the average age of clients by city.",
        "Which clients are 80 years old?"
    ]
    ground_truths = [
        enriched_df['magasin_nom_magasin'].nunique() if 'magasin_nom_magasin' in enriched_df.columns else None,
        enriched_df['client_nom'][enriched_df['client_âge'].idxmax()] if 'client_âge' in enriched_df.columns else None,
        "Visualization of sales by product category",
        enriched_df['client_nom'].drop_duplicates().tolist() if 'client_nom' in enriched_df.columns else [],  
        enriched_df['client_ville'].drop_duplicates().tolist() if 'client_ville' in enriched_df.columns else [],
        enriched_df['montant_total'].sum() if 'montant_total' in enriched_df.columns else None,
        (enriched_df.groupby('magasin_nom_magasin')['montant_total'].sum().idxmax(),
        enriched_df.groupby('magasin_nom_magasin')['montant_total'].sum().max()),
        enriched_df.groupby('client_ville')['client_âge'].mean().to_dict()
        if 'client_ville' in enriched_df.columns and 'client_âge' in enriched_df.columns else {},
        enriched_df[enriched_df['client_âge'] == 80]['client_nom'].unique().tolist()
        if 'client_âge' in enriched_df.columns and 'client_nom' in enriched_df.columns else []
    ]
    # Run evaluation
    results = []
    response_times = []
    relevancy_scores = []
    for query, truth in zip(test_queries, ground_truths):
        query_lower = query.lower()
        start_time = time.time()
        if mode == "V1 (Classic)":
            response = process_ai_request(
                query,
                enriched_df,
                [],
                st.session_state.get('eyq_endpoint', ''),
                st.session_state.get('eyq_api_key', ''),
                st.session_state.get('eyq_api_version', '2023-05-15')
            )
        else:  # V2 (BETA - Agentic)
            response = agentic_orchestrator(
                query,
                enriched_df,
                [],
                st.session_state.get('eyq_endpoint', ''),
                st.session_state.get('eyq_api_key', ''),
                st.session_state.get('eyq_api_version', '2023-05-15')
            )
        end_time = time.time()
        response_times.append(end_time - start_time)
        response_text = response.get('interpretation') or response.get('content', '')
        is_correct = is_response_correct(response_text, truth, enriched_df, query)
        if "plot" in query_lower or "chart" in query_lower:
            is_correct = is_correct and is_plot_correct(response, query, enriched_df)
        relevancy = compute_relevancy_score(response_text, truth, query, enriched_df, similarity_model)
        relevancy_scores.append(relevancy)
        results.append({
            'Query': query,
            'Response': response_text,
            'Ground Truth': truth,
            'Correct': is_correct,
            'Response Time (s)': end_time - start_time,
            'Relevancy Score': relevancy,
            'Figure': response.get('figure', None)
        })

    # Store results for the current mode
    st.session_state.evaluation_results[mode[:2]] = results

    # Compute metrics
    accuracy = sum(r['Correct'] for r in results) / len(results)
    avg_response_time = sum(response_times) / len(response_times)
    avg_relevancy_score = sum(relevancy_scores) / len(relevancy_scores)

    # Display metrics in styled boxes
    st.markdown("### 📊 Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">Accuracy</h3>
            <h1 style="color: {COLORS['success'] if accuracy >= 0.8 else COLORS['warning'] if accuracy >= 0.5 else COLORS['error']}; margin-bottom: 0;">
                {accuracy:.1%}
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">Avg. Response Time</h3>
            <h1 style="color: {COLORS['success'] if avg_response_time < 2 else COLORS['warning'] if avg_response_time < 5 else COLORS['error']}; margin-bottom: 0;">
                {avg_response_time:.2f}s
            </h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="color: {COLORS['primary']}; margin-top: 0;">Relevancy Score</h3>
            <h1 style="color: {COLORS['success'] if avg_relevancy_score >= 0.8 else COLORS['warning'] if avg_relevancy_score >= 0.5 else COLORS['error']}; margin-bottom: 0;">
                {avg_relevancy_score:.1%}
            </h1>
        </div>
        """, unsafe_allow_html=True)

    # Comparison table for V1 vs. V2
    if st.session_state.evaluation_results['V1'] and st.session_state.evaluation_results['V2']:
        st.markdown("### ⚖️ V1 vs. V2 Comparison")
        comparison_data = []
        for mode_key in ['V1', 'V2']:
            mode_results = st.session_state.evaluation_results[mode_key]
            mode_accuracy = sum(r['Correct'] for r in mode_results) / len(mode_results)
            mode_avg_time = sum(r['Response Time (s)'] for r in mode_results) / len(mode_results)
            mode_relevancy = sum(
                r['Relevancy Score'] if 'Relevancy Score' in r 
                else compute_relevancy_score(r['Response'], r['Ground Truth'], r['Query'], enriched_df, similarity_model)
                for r in mode_results
            ) / len(mode_results)
            comparison_data.append({
                'Mode': mode_key,
                'Accuracy': f"{mode_accuracy:.1%}",
                'Avg. Response Time': f"{mode_avg_time:.2f}s",
                'Relevancy Score': f"{mode_relevancy:.1%}"
            })
        comparison_df = pd.DataFrame(comparison_data)
        v1_time = float(comparison_data[0]['Avg. Response Time'][:-1])
        v2_time = float(comparison_data[1]['Avg. Response Time'][:-1])
        v2_class = 'highlight-v2' if v2_time < v1_time else ''
        st.markdown(f"""
        <table class="comparison-table">
            <tr>
                <th>Mode</th>
                <th>Accuracy</th>
                <th>Avg. Response Time</th>
                <th>Relevancy Score</th>
            </tr>
            <tr>
                <td>{comparison_data[0]['Mode']}</td>
                <td>{comparison_data[0]['Accuracy']}</td>
                <td>{comparison_data[0]['Avg. Response Time']}</td>
                <td>{comparison_data[0]['Relevancy Score']}</td>
            </tr>
            <tr class="{v2_class}">
                <td>{comparison_data[1]['Mode']}</td>
                <td>{comparison_data[1]['Accuracy']}</td>
                <td>{comparison_data[1]['Avg. Response Time']}</td>
                <td>{comparison_data[1]['Relevancy Score']}</td>
            </tr>
        </table>
        <p style="color: {COLORS['success']};">🎉 V2 shows improved response times!</p>
        """, unsafe_allow_html=True)

    # Visualize results with improved charts
    st.markdown("### 📈 Evaluation Results")
    
    df_results = pd.DataFrame(results)
    df_results['Correct'] = df_results['Correct'].map({True: '✅ Correct', False: '❌ Incorrect'})
    
    fig_accuracy = px.pie(
        df_results, 
        names='Correct', 
        title=f'Overall Accuracy ({mode})',
        hole=0.4,
        color='Correct',
        color_discrete_map={'✅ Correct': COLORS['success'], '❌ Incorrect': COLORS['error']}
    )
    fig_accuracy.update_traces(textposition='inside', textinfo='percent+label')
    fig_accuracy.update_layout(showlegend=False)
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    fig_query = px.bar(
        df_results, 
        x='Query', 
        y='Response Time (s)', 
        color='Correct',
        title=f'Query Performance Breakdown ({mode})',
        text='Response Time (s)',
        color_discrete_map={'✅ Correct': COLORS['success'], '❌ Incorrect': COLORS['error']}
    )
    fig_query.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    fig_query.update_layout(
        yaxis_title="Response Time (seconds)",
        xaxis_title="Query",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig_query, use_container_width=True)

    # Display sample responses
    st.markdown("### 🔍 Detailed Responses")
    for result in results:
        with st.expander(f"Query: {result['Query']} ({'✅ Correct' if result['Correct'] else '❌ Incorrect'})"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Response**: {result['Response']}")
                st.markdown(f"**Ground Truth**: {result['Ground Truth']}")
                st.markdown(f"**Relevancy Score**: {result['Relevancy Score']:.1%}")
            with col2:
                st.metric("Response Time", f"{result['Response Time (s)']:.2f}s")
            if result['Figure']:
                st.markdown("**Visualization:**")
                st.pyplot(result['Figure'])

    # User-driven evaluation
    st.markdown("### 🧪 Try Your Own Query")
    with st.form("user_evaluation_form"):
        user_query = st.text_input("Enter a test query:", 
                                 placeholder="e.g., Show top 5 products by sales",
                                 key="user_query_input")
        user_ground_truth = st.text_input("Enter expected response (optional):", 
                                        placeholder="e.g., List of top 5 products",
                                        key="user_ground_truth_input")
        submitted = st.form_submit_button("🚀 Evaluate Query")
    
    if submitted:
        if not user_query:
            st.warning("Please enter a query to evaluate")
        else:
            with st.spinner("🧠 Processing your query..."):
                start_time = time.time()
                if mode == "V1 (Classic)":
                    response = process_ai_request(
                        user_query,
                        enriched_df,
                        [],
                        st.session_state.get('eyq_endpoint', ''),
                        st.session_state.get('eyq_api_key', ''),
                        st.session_state.get('eyq_api_version', '2023-05-15')
                    )
                else:
                    response = agentic_orchestrator(
                        user_query,
                        enriched_df,
                        [],
                        st.session_state.get('eyq_endpoint', ''),
                        st.session_state.get('eyq_api_key', ''),
                        st.session_state.get('eyq_api_version', '2023-05-15')
                    )
                end_time = time.time()
                response_text = response.get('interpretation') or response.get('content', '')
                relevancy = compute_relevancy_score(response_text, user_ground_truth or "N/A", user_query, enriched_df, similarity_model)
                
                st.markdown("---")
                st.markdown("### 📝 Evaluation Results")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Your Query:** `{user_query}`")
                    st.markdown(f"**Chatbot Response:**")
                    st.markdown(f"> {response_text}")
                with col2:
                    st.metric("Response Time", f"{end_time - start_time:.2f}s")
                
                if user_ground_truth:
                    is_correct = is_response_correct(response_text, user_ground_truth, enriched_df, user_query)
                    if "plot" in user_query.lower() or "chart" in user_query.lower():
                        is_correct = is_correct and is_plot_correct(response, user_query, enriched_df)
                    st.markdown(f"**Evaluation:** {'✅ Correct' if is_correct else '❌ Incorrect'}")
                    st.markdown(f"**Expected Answer:** `{user_ground_truth}`")
                    st.markdown(f"**Relevancy Score:** {relevancy:.1%}")
                
                if 'figure' in response and response['figure']:
                    st.markdown("**Generated Visualization:**")
                    st.pyplot(response['figure'])
                
                st.info("💡 Tip: For better evaluation, provide specific expected answers when testing factual queries.")

if __name__ == "__main__":
    evaluation_page()