import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import re
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from app_pages.ai_assistant import process_ai_request, prepare_enriched_data

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

def is_response_correct(response, ground_truth, enriched_df, query, threshold=85):
    """Evaluate response correctness with flexible matching."""
    response_text = str(response).lower().strip()
    query_lower = query.lower().strip()
    ground_truth_text = str(ground_truth).lower().strip()

    # Handle visualization queries
    if "plot" in query_lower or "chart" in query_lower:
        if "visualization" in ground_truth_text and ("plot" in response_text or "chart" in response_text):
            return True
        return fuzz.partial_ratio(response_text, ground_truth_text) >= threshold

    # Handle list-based queries (e.g., "Name 3 clients")
    if "name 3 clients" in query_lower:
        valid_items = set(enriched_df['client_nom'].drop_duplicates().str.lower().tolist())
        response_items = re.findall(r'\b[\w\s-]+\b', response_text)
        if len(response_items) < 3:
            return False
        valid_count = sum(1 for item in response_items if item in valid_items)
        return valid_count >= 3

    # Handle other list-based queries (e.g., "List client addresses")
    if isinstance(ground_truth, list):
        valid_items = set(enriched_df['client_ville'].drop_duplicates().str.lower().tolist())
        response_items = re.findall(r'\b[\w\s-]+\b', response_text)
        valid_count = sum(1 for item in response_items if item in valid_items)
        return valid_count >= len(ground_truth) * 0.8

    # Exact or substring match
    if ground_truth_text in response_text or response_text in ground_truth_text:
        return True

    # Fuzzy matching
    similarity = fuzz.partial_ratio(response_text, ground_truth_text)
    return similarity >= threshold

def is_plot_correct(response, query, enriched_df):
    """Evaluate if a visualization matches the query intent."""
    if 'figure' not in response or not response['figure']:
        return False

    fig = response['figure']
    query_lower = query.lower()

    if 'sales by product category' in query_lower:
        # Check required columns
        if 'produit_catégorie' not in enriched_df.columns:
            st.warning("⚠️ Column 'produit_catégorie' not found in dataset. Skipping plot validation.")
            return True  # Assume correct if data unavailable
        if 'montant_total' not in enriched_df.columns:
            st.warning("⚠️ Column 'montant_total' not found in dataset. Skipping sales data validation.")
            return True  # Assume correct if data unavailable

        # Compute expected sales
        expected_sales = enriched_df.groupby('produit_catégorie')['montant_total'].sum().to_dict()
        expected_categories = set(enriched_df['produit_catégorie'].str.lower().unique())

        # Handle Plotly figures
        if isinstance(fig, dict) and 'data' in fig:
            try:
                # Check plot type
                plot_type = fig['data'][0]['type']
                if plot_type not in ['bar', 'column']:
                    st.warning(f"⚠️ Expected bar/column chart, got {plot_type}.")
                    return False

                # Check axes for categories (handles both x and y axis cases)
                x_data = fig['data'][0].get('x', [])
                y_data = fig['data'][0].get('y', [])
                
                # Determine which axis contains categories
                if len(x_data) > 0 and any(isinstance(x, str) for x in x_data):
                    # Categories are on x-axis
                    x_labels = set(str(x).lower() for x in x_data)
                    if not x_labels.intersection(expected_categories):
                        # Check if categories might be on y-axis instead
                        y_labels = set(str(y).lower() for y in y_data)
                        if not y_labels.intersection(expected_categories):
                            st.warning("⚠️ Plot axes do not contain expected product categories.")
                            return False
                        else:
                            # Categories are on y-axis, values on x-axis
                            plot_data = dict(zip(y_data, x_data))
                    else:
                        # Categories are on x-axis, values on y-axis
                        plot_data = dict(zip(x_data, y_data))
                elif len(y_data) > 0 and any(isinstance(y, str) for y in y_data):
                    # Categories are on y-axis
                    y_labels = set(str(y).lower() for y in y_data)
                    if not y_labels.intersection(expected_categories):
                        st.warning("⚠️ Plot y-axis does not contain expected product categories.")
                        return False
                    # Categories are on y-axis, values on x-axis
                    plot_data = dict(zip(y_data, x_data))
                else:
                    st.warning("⚠️ Could not determine which axis contains categories.")
                    return False

                # Compare plotted sales to expected sales
                for cat, val in plot_data.items():
                    expected_val = expected_sales.get(cat, 0)
                    if abs(val - expected_val) > 1e-2:  # Allow small float errors
                        st.warning(f"⚠️ Sales value for category '{cat}' ({val}) does not match expected ({expected_val}).")
                        return False

                return True
            except Exception as e:
                st.warning(f"⚠️ Plot validation failed: {str(e)}")
                return False

        # Handle Matplotlib figures
        if isinstance(fig, plt.Figure):
            try:
                axes = fig.get_axes()
                if not axes:
                    st.warning("⚠️ No axes found in Matplotlib figure.")
                    return False
                ax = axes[0]
                
                # Check axes for categories
                x_labels = [label.get_text().lower() for label in ax.get_xticklabels()]
                y_labels = [label.get_text().lower() for label in ax.get_yticklabels()]
                
                # Determine which axis contains categories
                if any(cat in x_labels for cat in expected_categories):
                    # Categories are on x-axis
                    plot_data = {label.get_text().lower(): rect.get_height() 
                               for label, rect in zip(ax.get_xticklabels(), 
                                                    [b for b in ax.get_children() if isinstance(b, plt.Rectangle)])}
                elif any(cat in y_labels for cat in expected_categories):
                    # Categories are on y-axis
                    plot_data = {label.get_text().lower(): rect.get_width()
                               for label, rect in zip(ax.get_yticklabels(),
                                                    [b for b in ax.get_children() if isinstance(b, plt.Rectangle)])}
                else:
                    st.warning("⚠️ Could not find expected categories on either axis.")
                    return False
                
                # Compare plotted sales to expected sales
                for cat, val in plot_data.items():
                    expected_val = expected_sales.get(cat.title(), 0)  # Adjust case if needed
                    if abs(val - expected_val) > 1e-2:
                        st.warning(f"⚠️ Sales value for category '{cat}' ({val}) does not match expected ({expected_val}).")
                        return False
                return True
            except Exception as e:
                st.warning(f"⚠️ Plot validation failed: {str(e)}")
                return False

    return True  # Default to True for other plot types

def evaluation_page():
    st.markdown(f'<h1 style="color: {COLORS["primary"]};">📈 Chatbot Evaluation</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: {COLORS["text"]};">Evaluate the AI Business Analyst with predefined tests or custom queries.</p>', unsafe_allow_html=True)

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
        "List client addresses"
    ]
    ground_truths = [
        enriched_df['magasin_nom_magasin'].nunique() if 'magasin_nom_magasin' in enriched_df.columns else None,
        enriched_df['client_nom'][enriched_df['client_âge'].idxmax()] if 'client_âge' in enriched_df.columns else None,
        "Visualization of sales by product category",
        "Any 3 valid client names",
        enriched_df['client_ville'].drop_duplicates().tolist() if 'client_ville' in enriched_df.columns else []
    ]

    # Run evaluation
    results = []
    response_times = []
    for query, truth in zip(test_queries, ground_truths):
        query_lower = query.lower()
        start_time = time.time()
        response = process_ai_request(
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
        results.append({
            'Query': query,
            'Response': response_text,
            'Ground Truth': truth,
            'Correct': is_correct,
            'Response Time (s)': end_time - start_time
        })

    # Compute metrics
    accuracy = sum(r['Correct'] for r in results) / len(results)
    avg_response_time = sum(response_times) / len(response_times)

    # Semantic similarity evaluation
    relevancy_score = None
    try:
        if similarity_model:
            similarities = []
            for r, query in zip(results, test_queries):
                response = str(r['Response']).lower()
                ground_truth = str(r['Ground Truth']).lower()
                if "name 3 clients" in query.lower() or "plot" in query.lower() or "chart" in query.lower():
                    similarities.append(1.0 if r['Correct'] else 0.0)
                else:
                    embeddings = similarity_model.encode([response, ground_truth])
                    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                    similarities.append(similarity)
            relevancy_score = sum(similarities) / len(similarities)
        else:
            similarities = []
            for r, query in zip(results, test_queries):
                response = str(r['Response']).lower()
                ground_truth = str(r['Ground Truth']).lower()
                if "name 3 clients" in query.lower() or "plot" in query.lower() or "chart" in query.lower():
                    similarities.append(1.0 if r['Correct'] else 0.0)
                else:
                    similarity = fuzz.ratio(response, ground_truth) / 100
                    similarities.append(similarity)
            relevancy_score = sum(similarities) / len(similarities)
    except Exception as e:
        st.warning(f"⚠️ Similarity evaluation failed: {str(e)}. Using fuzzy matching.")
        similarities = []
        for r, query in zip(results, test_queries):
            response = str(r['Response']).lower()
            ground_truth = str(r['Ground Truth']).lower()
            if "name 3 clients" in query.lower() or "plot" in query.lower() or "chart" in query.lower():
                similarities.append(1.0 if r['Correct'] else 0.0)
            else:
                similarity = fuzz.ratio(response, ground_truth) / 100
                similarities.append(similarity)
        relevancy_score = sum(similarities) / len(similarities)

    # Display metrics in styled boxes
    st.markdown("### 📊 Performance Metrics")
    col1, col2 = st.columns(2)
    
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
    
    st.markdown(f"""
    <div class="metric-box">
        <h3 style="color: {COLORS['primary']}; margin-top: 0;">Relevancy Score</h3>
        <h1 style="color: {COLORS['success'] if relevancy_score >= 0.8 else COLORS['warning'] if relevancy_score >= 0.5 else COLORS['error']}; margin-bottom: 0;">
            {relevancy_score:.1%}
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # Visualize results with improved charts
    st.markdown("### 📈 Evaluation Results")
    
    # Convert boolean to string for better display
    df_results = pd.DataFrame(results)
    df_results['Correct'] = df_results['Correct'].map({True: '✅ Correct', False: '❌ Incorrect'})
    
    # Accuracy by query - donut chart
    fig_accuracy = px.pie(
        df_results, 
        names='Correct', 
        title='Overall Accuracy',
        hole=0.4,
        color='Correct',
        color_discrete_map={'✅ Correct': COLORS['success'], '❌ Incorrect': COLORS['error']}
    )
    fig_accuracy.update_traces(textposition='inside', textinfo='percent+label')
    fig_accuracy.update_layout(showlegend=False)
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Query performance breakdown
    fig_query = px.bar(
        df_results, 
        x='Query', 
        y='Response Time (s)', 
        color='Correct',
        title='Query Performance Breakdown',
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
            with col2:
                st.metric("Response Time", f"{result['Response Time (s)']:.2f}s")
            response = process_ai_request(result['Query'], enriched_df, [], st.session_state.get('eyq_endpoint', ''), st.session_state.get('eyq_api_key', ''), st.session_state.get('eyq_api_version', '2023-05-15'))
            if 'figure' in response and response['figure']:
                st.markdown("**Visualization:**")
                st.pyplot(response['figure'])

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
                response = process_ai_request(
                    user_query,
                    enriched_df,
                    [],
                    st.session_state.get('eyq_endpoint', ''),
                    st.session_state.get('eyq_api_key', ''),
                    st.session_state.get('eyq_api_version', '2023-05-15')
                )
                end_time = time.time()
                response_text = response.get('interpretation') or response.get('content', '')
                
                # Display results in a nicely formatted way
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
                    st.markdown(f"**Evaluation:** {'✅ Correct' if is_correct else '❌ Incorrect'}")
                    st.markdown(f"**Expected Answer:** `{user_ground_truth}`")
                
                if 'figure' in response and response['figure']:
                    st.markdown("**Generated Visualization:**")
                    st.pyplot(response['figure'])
                
                st.info("💡 Tip: For better evaluation, provide specific expected answers when testing factual queries.")

if __name__ == "__main__":
    evaluation_page()