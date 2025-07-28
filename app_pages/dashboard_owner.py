# pages/dashboard_owner.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import  timedelta
import numpy as np

import logging
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Consistent color scheme for the dashboard
COLORS = {
    'primary': '#1E3A8A',    # Dark blue
    'secondary': '#3B82F6',  # Lighter blue
    'success': '#10B981',    # Green
    'warning': '#F59E0B',    # Orange
    'error': '#EF4444',      # Red
    'background': '#F8F9FA', # Light gray
    'text': '#1F2937',       # Dark gray
    'accent': '#8B5CF6',     # Purple for trends
}


def load_data(table_name):
    """Load data from mapped DataFrames in session state with error handling"""
    try:
        df = st.session_state.get('dataframes', {}).get(table_name, pd.DataFrame())
        if df.empty:
            logger.warning(f"No DataFrame found for {table_name}")
            return pd.DataFrame()
        logger.info(f"Loaded {table_name} with {len(df)} rows, columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading {table_name}: {e}")
        return pd.DataFrame()

def get_column_mapping(table_name, column_name):
    """Retrieve mapped column name, fallback to original if not found"""
    mappings = st.session_state.get('column_mappings', {}).get(table_name, {})
    df = st.session_state.get('dataframes', {}).get(table_name, pd.DataFrame())
    available_columns = list(df.columns)
    logger.info(f"Table: {table_name}, Column: {column_name}, Mappings: {mappings}, Available columns: {available_columns}")
    mapped_column = mappings.get(column_name, column_name)
    if mapped_column not in available_columns:
        logger.warning(f"Mapped column '{mapped_column}' not found in {table_name}. Falling back to '{column_name}'.")
    return mapped_column

def prepare_store_data(df_trans, df_magasin):
    """Enhance transaction data with store names"""
    if df_magasin.empty:
        return df_trans.assign(store_name='Unknown', store_display=df_trans[get_column_mapping('Transactions', 'id_magasin')].astype(str))

    try:
        store_col = get_column_mapping('Transactions', 'id_magasin')
        magasin_id_col = get_column_mapping('Magasin', 'id_magasin')
        name_cols = [col for col in df_magasin.columns if 'nom' in col.lower() or 'name' in col.lower()]
        
        if name_cols:
            store_name_map = dict(zip(df_magasin[magasin_id_col], df_magasin[name_cols[0]]))
            return df_trans.assign(
                store_name=df_trans[store_col].map(store_name_map).fillna('Unknown Store'),
                store_display=df_trans[store_col].map(store_name_map).fillna(f"Store {df_trans[store_col].astype(str)}")
            )
        return df_trans.assign(
            store_name=f"Store {df_trans[store_col].astype(str)}",
            store_display=f"Store {df_trans[store_col].astype(str)}"
        )
    except Exception as e:
        logger.error(f"Error preparing store data: {e}")
        return df_trans

def add_date_range_selector(df_trans):
    """Add date range selector and filter transactions"""
    try:
        date_col = get_column_mapping('Transactions', 'date_heure')
        df_clean = df_trans.copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[date_col])
        
        if df_clean.empty:
            st.sidebar.error("No valid date data found in transactions.")
            return pd.DataFrame()
        
        min_date, max_date = df_clean[date_col].min().date(), df_clean[date_col].max().date()
        
        st.sidebar.markdown(f"**📅 Date Range** (Available: {min_date} to {max_date})")
        quick_filter = st.sidebar.radio(
            "Quick Filters",
            ["Last 30 Days", "Last 90 Days", "Last 6 Months", "All Time"],
            key="quick_filter",
            label_visibility="collapsed"
        )
        
        if quick_filter == "Last 30 Days":
            default_start = max_date - timedelta(days=30)
        elif quick_filter == "Last 90 Days":
            default_start = max_date - timedelta(days=90)
        elif quick_filter == "Last 6 Months":
            default_start = max_date - timedelta(days=180)
        else:
            default_start = min_date
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start", default_start, min_date, max_date, key="start_date")
        with col2:
            end_date = st.date_input("End", max_date, min_date, max_date, key="end_date")
        
        if start_date > end_date:
            st.sidebar.error("Start date cannot be after end date!")
            return pd.DataFrame()
        
        mask = (df_clean[date_col].dt.date >= start_date) & (df_clean[date_col].dt.date <= end_date)
        df_filtered = df_clean[mask]
        
        total_trans = len(df_trans)
        filtered_trans = len(df_filtered)
        st.sidebar.metric("Filtered Transactions", f"{filtered_trans:,}", f"{filtered_trans/total_trans*100:.1f}%")
        
        return df_filtered
    except Exception as e:
        logger.error(f"Error in date range selector: {e}")
        st.error(f"Date filter error: {e}")
        return df_trans

def calculate_period_growth(df_trans, days=30):
    """Calculate revenue growth rate between periods"""
    try:
        date_col = get_column_mapping('Transactions', 'date_heure')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        df_clean = df_trans.copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[date_col, amount_col])
        
        if len(df_clean) < 2:
            return 0
        
        recent_date = df_clean[date_col].max()
        current_period = df_clean[df_clean[date_col] >= recent_date - timedelta(days=days)]
        previous_period = df_clean[
            (df_clean[date_col] >= recent_date - timedelta(days=days*2)) & 
            (df_clean[date_col] < recent_date - timedelta(days=days))
        ]
        
        current_sum = current_period[amount_col].sum()
        previous_sum = previous_period[amount_col].sum()
        return ((current_sum - previous_sum) / previous_sum) * 100 if previous_sum > 0 else 0
    except:
        return 0

def business_owner_dashboard():
    """Business owner dashboard with strategic insights"""
    st.markdown(f'<h1 style="color: {COLORS["primary"]};">🏢 Business Owner Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: {COLORS["text"]};">Strategic insights for multi-store performance</p>', unsafe_allow_html=True)
    
    # Load data
    data = {
        'Transactions': load_data('Transactions'),
        'Magasin': load_data('Magasin'),
        'Client': load_data('Client'),
        'Produit': load_data('Produit'),
        'Stock': load_data('Stock'),
        'Employé': load_data('Employé'),
        'Localisation': load_data('Localisation')
    }
    
    if data['Transactions'].empty:
        st.error("⚠️ No transaction data available. Complete the ETL pipeline on the Mapping page.")
        return
    
    # Prepare store-enhanced data
    df_trans_with_stores = prepare_store_data(data['Transactions'], data['Magasin'])
    
    # Apply date filter
    df_trans_filtered = add_date_range_selector(df_trans_with_stores)
    if df_trans_filtered.empty:
        st.warning("No data for selected date range. Adjust the filter.")
        return
    
    # Executive KPIs
    show_executive_kpis(df_trans_filtered, data['Client'], data['Magasin'], data['Employé'])
    
    # Tabs for strategic sections
    tabs = st.tabs(["🎯 Overview", "🏪 Stores", "💰 Financials", "🚀 Growth"])
    
    with tabs[0]:
        business_overview_section(df_trans_filtered, data['Produit'], data['Client'])
    with tabs[1]:
        store_performance_section(df_trans_filtered, data['Magasin'], data['Localisation'])
    with tabs[2]:
        financial_analysis_section(df_trans_filtered, data['Produit'])
    with tabs[3]:
        growth_strategy_section(df_trans_filtered, data['Client'], data['Magasin'])
    
   
def show_executive_kpis(df_trans, df_client, df_magasin, df_employee):
    """Display executive KPIs"""
    st.markdown(f'<h2 style="color: {COLORS["secondary"]};">📊 Executive Summary</h2>', unsafe_allow_html=True)
    
    amount_col = get_column_mapping('Transactions', 'montant_total')
    total_revenue = df_trans[amount_col].sum() if amount_col in df_trans.columns else 0
    total_stores = len(df_magasin)
    total_customers = len(df_client)
    total_employees = len(df_employee)
    total_transactions = len(df_trans)
    revenue_growth = calculate_period_growth(df_trans, 30)
    
    cols = st.columns(5)
    with cols[0]:
        st.metric("Total Revenue", f"{total_revenue:,.0f} TND", f"{revenue_growth:+.1f}%")
    with cols[1]:
        st.metric("Active Stores", f"{total_stores}")
    with cols[2]:
        st.metric("Total Customers", f"{total_customers:,}")
    with cols[3]:
        st.metric("Staff Members", f"{total_employees}")
    with cols[4]:
        avg_transaction = total_revenue / total_transactions if total_transactions > 0 else 0
        st.metric("Avg Transaction", f"{avg_transaction:.0f} TND")

def business_overview_section(df_trans, df_prod, df_client):
    """Business overview analytics"""
    col1, col2 = st.columns(2)
    with col1:
        create_revenue_trend_chart(df_trans)
        create_category_revenue_chart(df_trans, df_prod)
    with col2:
        create_customer_acquisition_chart(df_trans)
        create_weekday_business_chart(df_trans)
    create_seasonal_business_analysis(df_trans)

def store_performance_section(df_trans, df_magasin, df_location):
    """Store performance analysis"""
    st.markdown(f'<h2 style="color: {COLORS["secondary"]};">🏪 Store Performance</h2>', unsafe_allow_html=True)
    create_store_comparison_chart(df_trans)
    col1, col2 = st.columns(2)
    with col1:
        create_geographic_performance_chart(df_trans, df_magasin, df_location)
    with col2:
        create_store_efficiency_ranking(df_trans, df_magasin)

def financial_analysis_section(df_trans, df_prod):
    """Financial performance analysis"""
    st.markdown(f'<h2 style="color: {COLORS["secondary"]};">💰 Financial Analysis</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        create_revenue_profit_chart(df_trans, df_prod)
    with col2:
        create_product_profitability_chart(df_trans, df_prod)
    create_cost_analysis_chart(df_trans, df_prod)

def growth_strategy_section(df_trans, df_client, df_magasin):
    """Growth strategy insights"""
    st.markdown(f'<h2 style="color: {COLORS["secondary"]};">🚀 Growth Strategy</h2>', unsafe_allow_html=True)
    create_business_performance_dashboard(df_trans)
    col1, col2 = st.columns(2)
    with col1:
        create_expansion_opportunities_chart(df_trans, df_magasin)
    with col2:
        create_customer_ltv_analysis(df_trans, df_client)

def create_revenue_trend_chart(df_trans):
    """Revenue trend chart"""
    try:
        date_col = get_column_mapping('Transactions', 'date_heure')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        df_clean = df_trans.copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[date_col, amount_col])
        
        if df_clean.empty:
            st.warning("No data for revenue trend.")
            return
        
        date_range = (df_clean[date_col].max() - df_clean[date_col].min()).days
        if date_range <= 90:
            df_clean['period'] = df_clean[date_col].dt.date
            period_label = 'Date'
        elif date_range <= 730:
            df_clean['period'] = df_clean[date_col].dt.to_period('W').astype(str)
            period_label = 'Week'
        else:
            df_clean['period'] = df_clean[date_col].dt.to_period('M').astype(str)
            period_label = 'Month'
        
        period_revenue = df_clean.groupby('period')[amount_col].sum().reset_index()
        
        fig = px.line(period_revenue, x='period', y=amount_col, 
                      title="📈 Revenue Trend", 
                      labels={'period': period_label, amount_col: 'Revenue (TND)'})
        fig.update_traces(line=dict(color=COLORS['primary'], width=3))
        fig.update_layout(
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if len(period_revenue) >= 2:
            growth = ((period_revenue.iloc[-1][amount_col] - period_revenue.iloc[-2][amount_col]) / 
                      period_revenue.iloc[-2][amount_col] * 100 if period_revenue.iloc[-2][amount_col] > 0 else 0)
            trend = "📈 Growing" if growth > 0 else "📉 Declining"
            st.info(f"**Trend:** {trend} ({growth:+.1f}% vs previous period)")
    except Exception as e:
        st.error(f"Revenue trend error: {e}")

def create_category_revenue_chart(df_trans, df_prod):
    """Product category revenue chart"""
    if df_prod.empty:
        st.info("No product data for category analysis.")
        return
    
    try:
        trans_prod_col = get_column_mapping('Transactions', 'id_produit')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        prod_id_col = get_column_mapping('Produit', 'id_produit')
        category_col = get_column_mapping('Produit', 'catégorie')
        
        df_merged = df_trans.merge(df_prod[[prod_id_col, category_col]], 
                                  left_on=trans_prod_col, right_on=prod_id_col, how='left')
        category_revenue = df_merged.groupby(category_col)[amount_col].sum().sort_values(ascending=False).head(10)
        
        if category_revenue.empty:
            st.warning("No category data available.")
            return
        
        fig = px.bar(x=category_revenue.values, y=category_revenue.index, orientation='h',
                     title="💼 Revenue by Category",
                     labels={'x': 'Revenue (TND)', 'y': 'Category'})
        fig.update_traces(marker_color=COLORS['secondary'])
        fig.update_layout(
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        top_category = category_revenue.index[0]
        top_share = (category_revenue.iloc[0] / category_revenue.sum()) * 100
        st.success(f"**Leading Category:** {top_category} ({top_share:.1f}% of revenue)")
    except Exception as e:
        st.error(f"Category revenue error: {e}")

def create_customer_acquisition_chart(df_trans):
    """Customer acquisition trend chart"""
    try:
        date_col = get_column_mapping('Transactions', 'date_heure')
        client_col = get_column_mapping('Transactions', 'id_client')
        df_clean = df_trans.copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[date_col, client_col])
        
        if df_clean.empty:
            st.warning("No data for customer acquisition.")
            return
        
        date_range = (df_clean[date_col].max() - df_clean[date_col].min()).days
        period_col = 'date' if date_range <= 90 else 'month'
        df_clean['period'] = df_clean[date_col].dt.date if period_col == 'date' else df_clean[date_col].dt.to_period('M').astype(str)
        
        first_purchase = df_clean.groupby(client_col)[date_col].min().reset_index()
        first_purchase['period'] = first_purchase[date_col].dt.date if period_col == 'date' else first_purchase[date_col].dt.to_period('M').astype(str)
        new_customers = first_purchase.groupby('period').size().reset_index(name='new_customers')
        
        fig = px.bar(new_customers, x='period', y='new_customers',
                     title="👥 New Customer Acquisition",
                     labels={'period': period_col.capitalize(), 'new_customers': 'New Customers'})
        fig.update_traces(marker_color=COLORS['success'])
        fig.update_layout(
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        avg_period = new_customers['new_customers'].sum() / len(new_customers) if len(new_customers) > 0 else 0
        st.info(f"**Average Acquisition:** {avg_period:.0f} new customers per {period_col}")
    except Exception as e:
        st.error(f"Customer acquisition error: {e}")

def create_weekday_business_chart(df_trans):
    """Weekday business performance chart"""
    try:
        date_col = get_column_mapping('Transactions', 'date_heure')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        df_clean = df_trans.copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[date_col, amount_col])
        
        if df_clean.empty:
            st.warning("No data for weekday analysis.")
            return
        
        df_clean['weekday'] = df_clean[date_col].dt.day_name()
        weekday_performance = df_clean.groupby('weekday')[amount_col].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_performance = weekday_performance.reindex([day for day in day_order if day in weekday_performance.index])
        
        fig = px.bar(x=weekday_performance.index, y=weekday_performance.values,
                     title="📅 Average Daily Performance",
                     labels={'x': 'Day', 'y': 'Avg Revenue (TND)'})
        fig.update_traces(marker_color=COLORS['warning'])
        fig.update_layout(
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if not weekday_performance.empty:
            best_day = weekday_performance.idxmax()
            st.success(f"**Best Day:** {best_day} ({weekday_performance.max():,.0f} TND avg)")
    except Exception as e:
        st.error(f"Weekday chart error: {e}")

def create_seasonal_business_analysis(df_trans):
    """Seasonal business analysis"""
    st.markdown(f'<h3 style="color: {COLORS["secondary"]};">🌍 Seasonal Analysis</h3>', unsafe_allow_html=True)
    try:
        date_col = get_column_mapping('Transactions', 'date_heure')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        df_clean = df_trans.copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[date_col, amount_col])
        
        if df_clean.empty:
            st.warning("No data for seasonal analysis.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            df_clean['quarter'] = df_clean[date_col].dt.quarter
            quarterly_revenue = df_clean.groupby('quarter')[amount_col].sum()
            if len(quarterly_revenue) > 1:
                fig = px.bar(x=[f'Q{q}' for q in quarterly_revenue.index], y=quarterly_revenue.values,
                             title="Quarterly Revenue", labels={'x': 'Quarter', 'y': 'Revenue (TND)'})
                fig.update_traces(marker_color=COLORS['primary'])
                fig.update_layout(
                    height=350, 
                    plot_bgcolor='white', 
                    paper_bgcolor=COLORS['background'],
                    font_color=COLORS['text'],
                    title_font=dict(size=14, color=COLORS['secondary'])
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for quarterly analysis.")
        
        with col2:
            df_clean['month'] = df_clean[date_col].dt.month
            monthly_avg = df_clean.groupby('month')[amount_col].mean()
            if not monthly_avg.empty:
                fig = px.line(x=monthly_avg.index, y=monthly_avg.values,
                              title="Monthly Seasonality", labels={'x': 'Month', 'y': 'Avg Revenue (TND)'})
                fig.update_traces(line=dict(color=COLORS['accent'], width=3))
                fig.update_layout(
                    height=350, 
                    plot_bgcolor='white', 
                    paper_bgcolor=COLORS['background'],
                    font_color=COLORS['text'],
                    title_font=dict(size=14, color=COLORS['secondary'])
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data for monthly analysis.")
        
        if not quarterly_revenue.empty:
            peak_quarter = f"Q{quarterly_revenue.idxmax()}"
            st.info(f"**Peak Period:** {peak_quarter} ({quarterly_revenue.max():,.0f} TND)")
    except Exception as e:
        st.error(f"Seasonal analysis error: {e}")

def create_store_comparison_chart(df_trans):
    """Store revenue comparison chart"""
    try:
        store_col = get_column_mapping('Transactions', 'id_magasin')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        store_revenue = df_trans.groupby('store_display' if 'store_display' in df_trans else store_col)[amount_col].sum().sort_values(ascending=False).head(10)
        
        if store_revenue.empty:
            st.warning("No store data available.")
            return
        
        fig = px.bar(x=store_revenue.index, y=store_revenue.values,
                     title="🏪 Store Revenue Comparison",
                     labels={'x': 'Store', 'y': 'Revenue (TND)'})
        fig.update_traces(marker_color=COLORS['secondary'])
        fig.update_layout(
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary']),
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if len(store_revenue) > 1:
            top_store, top_revenue = store_revenue.index[0], store_revenue.iloc[0]
            bottom_store, bottom_revenue = store_revenue.index[-1], store_revenue.iloc[-1]
            performance_gap = ((top_revenue - bottom_revenue) / bottom_revenue * 100) if bottom_revenue > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"**Top Store:** {top_store} ({top_revenue:,.0f} TND)")
            with col2:
                st.warning(f"**Needs Attention:** {bottom_store} ({bottom_revenue:,.0f} TND)")
            with col3:
                st.info(f"**Performance Gap:** {performance_gap:.0f}%")
    except Exception as e:
        st.error(f"Store comparison error: {e}")

def create_geographic_performance_chart(df_trans, df_magasin, df_location):
    """Geographic performance chart"""
    if df_location.empty or df_magasin.empty:
        st.info("🗺️ Geographic data not available.")
        return
    
    try:
        store_col = get_column_mapping('Transactions', 'id_magasin')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        magasin_id_col = get_column_mapping('Magasin', 'id_magasin')
        location_id_col = get_column_mapping('Magasin', 'id_localisation')
        
        store_revenue = df_trans.groupby(['store_display', store_col])[amount_col].agg(['sum', 'count', 'mean']).reset_index()
        store_revenue.columns = ['store_display', store_col, 'total_revenue', 'transaction_count', 'avg_transaction']
        
        store_locations = df_magasin.merge(df_location, left_on=location_id_col, right_on=get_column_mapping('Localisation', 'id_localisation'))
        geo_performance = store_revenue.merge(store_locations, left_on=store_col, right_on=magasin_id_col)
        
        if geo_performance.empty:
            st.warning("No geographic performance data.")
            return
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=geo_performance['total_revenue'],
            y=geo_performance['avg_transaction'],
            mode='markers',
            marker=dict(
                size=np.clip(geo_performance['transaction_count']/10, 8, 40),
                color=geo_performance['total_revenue'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Revenue (TND)", thickness=10),
                line=dict(width=1, color='white')
            ),
            text=[f"{row.store_display}<br>Revenue: {row.total_revenue:,.0f} TND<br>Trans: {row.transaction_count:,}<br>Avg: {row.avg_transaction:.0f} TND" 
                  for _, row in geo_performance.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        fig.update_layout(
            title="🗺️ Geographic Performance",
            xaxis=dict(title="Total Revenue (TND)", gridcolor='lightgray', showgrid=True),
            yaxis=dict(title="Avg Transaction (TND)", gridcolor='lightgray', showgrid=True),
            plot_bgcolor='white',
            paper_bgcolor=COLORS['background'],
            height=450,
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            top = geo_performance.loc[geo_performance['total_revenue'].idxmax()]
            st.metric("Top Performer", top['store_display'], f"{top['total_revenue']:,.0f} TND")
        with col2:
            efficient = geo_performance.loc[geo_performance['avg_transaction'].idxmax()]
            st.metric("Most Efficient", efficient['store_display'], f"{efficient['avg_transaction']:.0f} TND/trans")
        with col3:
            consistency = (1 - geo_performance['total_revenue'].std()/geo_performance['total_revenue'].mean()) * 100 if geo_performance['total_revenue'].mean() > 0 else 0
            st.metric("Consistency", f"{consistency:.0f}%", f"{len(geo_performance)} stores")
    except Exception as e:
        st.error(f"Geographic performance error: {e}")

def create_store_efficiency_ranking(df_trans, df_magasin):
    """Store efficiency ranking chart"""
    try:
        store_col = get_column_mapping('Transactions', 'id_magasin')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        date_col = get_column_mapping('Transactions', 'date_heure')
        
        df_clean = df_trans.copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        group_col = 'store_display' if 'store_display' in df_clean else store_col
        
        store_metrics = df_clean.groupby(group_col).agg({
            amount_col: ['sum', 'count', 'mean'],
            date_col: lambda x: (x.max() - x.min()).days + 1
        }).round(2)
        store_metrics.columns = ['total_revenue', 'total_transactions', 'avg_transaction', 'active_days']
        store_metrics['revenue_per_day'] = store_metrics['total_revenue'] / store_metrics['active_days']
        store_metrics['transactions_per_day'] = store_metrics['total_transactions'] / store_metrics['active_days']
        store_metrics['efficiency_score'] = (
            (store_metrics['revenue_per_day'] / store_metrics['revenue_per_day'].max()) * 0.4 +
            (store_metrics['avg_transaction'] / store_metrics['avg_transaction'].max()) * 0.3 +
            (store_metrics['transactions_per_day'] / store_metrics['transactions_per_day'].max()) * 0.3
        ) * 100
        store_metrics = store_metrics.sort_values('efficiency_score', ascending=True).reset_index()
        
        fig = go.Figure()
        colors = px.colors.sequential.Blues[::-1][:len(store_metrics)]
        fig.add_trace(go.Bar(
            y=store_metrics[group_col],
            x=store_metrics['efficiency_score'],
            orientation='h',
            marker=dict(color=colors, line=dict(width=0.5, color='white')),
            text=[f"{score:.1f}" for score in store_metrics['efficiency_score']],
            textposition='inside',
            textfont=dict(color='white', size=12)
        ))
        fig.update_layout(
            title="⚡ Store Efficiency Ranking",
            xaxis=dict(title="Efficiency Score (0-100)", gridcolor='lightgray', showgrid=True),
            yaxis=dict(showgrid=False, tickfont=dict(size=11)),
            plot_bgcolor='white',
            paper_bgcolor=COLORS['background'],
            height=max(300, len(store_metrics) * 40),
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("**Top Performers**")
            for _, store in store_metrics.tail(3).iterrows():
                st.write(f"• {store[group_col]}: {store['efficiency_score']:.1f} pts")
        with col2:
            st.warning("**Improvement Needed**")
            for _, store in store_metrics.head(3).iterrows():
                st.write(f"• {store[group_col]}: {store['efficiency_score']:.1f} pts")
    except Exception as e:
        st.error(f"Efficiency ranking error: {e}")

def create_revenue_profit_chart(df_trans, df_prod):
    """Revenue vs profit analysis chart"""
    if df_prod.empty:
        st.info("Product data needed for profit analysis.")
        return
    
    try:
        trans_prod_col = get_column_mapping('Transactions', 'id_produit')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        quantity_col = get_column_mapping('Transactions', 'quantité')
        prod_id_col = get_column_mapping('Produit', 'id_produit')
        prix_achat_col = get_column_mapping('Produit', 'prix_achat')
        prix_vente_col = get_column_mapping('Produit', 'prix_vente')
        
        df_merged = df_trans.merge(df_prod[[prod_id_col, prix_achat_col, prix_vente_col]], 
                                  left_on=trans_prod_col, right_on=prod_id_col, how='left')
        df_merged['real_cost'] = df_merged[quantity_col] * df_merged[prix_achat_col]
        df_merged['real_profit'] = df_merged[amount_col] - df_merged['real_cost']
        
        total_revenue = df_merged[amount_col].sum()
        total_cost = df_merged['real_cost'].sum()
        total_profit = df_merged['real_profit'].sum()
        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        fig = go.Figure(data=[go.Pie(
            labels=['Profit', 'Costs'],
            values=[total_profit, total_cost],
            hole=0.4,
            marker_colors=[COLORS['success'], COLORS['error']],
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>%{value:,.0f} TND'
        )])
        fig.update_layout(
            title="💰 Revenue Breakdown",
            plot_bgcolor='white',
            paper_bgcolor=COLORS['background'],
            height=400,
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Revenue", f"{total_revenue:,.0f} TND")
        with col2:
            st.metric("Total Profit", f"{total_profit:,.0f} TND")
        with col3:
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
        
        st.markdown("**Profitability Assessment**")
        if profit_margin >= 40:
            st.success("🎯 Exceptional profitability")
        elif profit_margin >= 25:
            st.success("✅ Strong profitability")
        elif profit_margin >= 15:
            st.info("📊 Good profitability")
        else:
            st.warning("⚠️ Low profitability")
    except Exception as e:
        st.error(f"Revenue/profit error: {e}")

def create_product_profitability_chart(df_trans, df_prod):
    """Product profitability ranking chart"""
    if df_prod.empty:
        st.info("Product data needed for profitability analysis.")
        return
    
    try:
        trans_prod_col = get_column_mapping('Transactions', 'id_produit')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        prod_id_col = get_column_mapping('Produit', 'id_produit')
        prod_name_col = get_column_mapping('Produit', 'nom_produit')
        
        product_revenue = df_trans.groupby(trans_prod_col)[amount_col].agg(['sum', 'count', 'mean']).round(2)
        product_revenue.columns = ['total_revenue', 'quantity_sold', 'avg_price']
        product_revenue = product_revenue.reset_index().merge(
            df_prod[[prod_id_col, prod_name_col]], left_on=trans_prod_col, right_on=prod_id_col, how='left'
        )
        product_revenue['product_label'] = product_revenue[prod_name_col].fillna(f'Product {product_revenue[trans_prod_col]}')
        product_revenue = product_revenue.sort_values('total_revenue', ascending=False).head(10)
        
        fig = go.Figure()
        colors = px.colors.sequential.Blues[::-1][:len(product_revenue)]
        fig.add_trace(go.Bar(
            y=product_revenue['product_label'],
            x=product_revenue['total_revenue'],
            orientation='h',
            marker=dict(color=colors, line=dict(width=0.5, color='white')),
            text=[f"{rev:,.0f}" for rev in product_revenue['total_revenue']],
            textposition='inside',
            textfont=dict(color='white', size=11)
        ))
        fig.update_layout(
            title="📈 Top Product Revenue",
            xaxis=dict(title="Revenue (TND)", gridcolor='lightgray', showgrid=True),
            yaxis=dict(showgrid=False, tickfont=dict(size=10)),
            plot_bgcolor='white',
            paper_bgcolor=COLORS['background'],
            height=max(400, len(product_revenue) * 30),
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            top = product_revenue.iloc[0]
            st.success(f"**Top Earner:** {top['product_label'][:20]}... ({top['total_revenue']:,.0f} TND)")
        with col2:
            high_volume = product_revenue.loc[product_revenue['quantity_sold'].idxmax()]
            st.info(f"**Volume Leader:** {high_volume['product_label'][:20]}... ({high_volume['quantity_sold']:,.0f} units)")
        with col3:
            premium = product_revenue.loc[product_revenue['avg_price'].idxmax()]
            st.warning(f"**Premium Product:** {premium['product_label'][:20]}... ({premium['avg_price']:.0f} TND)")
    except Exception as e:
        st.error(f"Product profitability error: {e}")

def create_cost_analysis_chart(df_trans, df_prod):
    """Cost structure analysis by category"""
    if df_prod.empty:
        st.info("Product data needed for cost analysis.")
        return
    
    try:
        trans_prod_col = get_column_mapping('Transactions', 'id_produit')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        quantity_col = get_column_mapping('Transactions', 'quantité')
        prod_id_col = get_column_mapping('Produit', 'id_produit')
        category_col = get_column_mapping('Produit', 'catégorie')
        prix_achat_col = get_column_mapping('Produit', 'prix_achat')
        
        df_merged = df_trans.merge(df_prod[[prod_id_col, category_col, prix_achat_col]], 
                                  left_on=trans_prod_col, right_on=prod_id_col, how='left')
        df_merged['real_cost'] = df_merged[quantity_col] * df_merged[prix_achat_col]
        df_merged['real_profit'] = df_merged[amount_col] - df_merged['real_cost']
        
        category_analysis = df_merged.groupby(category_col).agg({
            amount_col: 'sum',
            'real_cost': 'sum',
            'real_profit': 'sum',
            quantity_col: 'sum'
        }).round(2)
        category_analysis.columns = ['revenue', 'total_cost', 'total_profit', 'units_sold']
        category_analysis['avg_margin'] = (category_analysis['total_profit'] / category_analysis['revenue'] * 100).round(1)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=category_analysis.index,
            y=category_analysis['revenue'],
            name='Revenue',
            marker_color=COLORS['primary'],
            opacity=0.6,
            text=[f"{val:,.0f}" for val in category_analysis['revenue']],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=category_analysis.index,
            y=category_analysis['total_cost'],
            name='Costs',
            marker_color=COLORS['error'],
            text=[f"{val:,.0f}" for val in category_analysis['total_cost']],
            textposition='inside'
        ))
        fig.add_trace(go.Scatter(
            x=category_analysis.index,
            y=category_analysis['avg_margin'],
            mode='lines+markers+text',
            name='Profit Margin %',
            line=dict(color=COLORS['success'], width=3),
            marker=dict(size=8),
            text=[f"{val:.1f}%" for val in category_analysis['avg_margin']],
            textposition='top center',
            yaxis='y2'
        ))
        fig.update_layout(
            title="💸 Cost Structure by Category",
            xaxis=dict(title="Category", tickangle=45),
            yaxis=dict(title="Amount (TND)", gridcolor='lightgray', showgrid=True),
            yaxis2=dict(title="Profit Margin (%)", overlaying='y', side='right', showgrid=False),
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor=COLORS['background'],
            height=500,
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary']),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            cost_ratio = (category_analysis['total_cost'].sum() / category_analysis['revenue'].sum() * 100) if category_analysis['revenue'].sum() > 0 else 0
            st.metric("Cost Ratio", f"{cost_ratio:.1f}%")
        with col2:
            st.metric("Profit Margin", f"{category_analysis['avg_margin'].mean():.1f}%")
        with col3:
            top = category_analysis.loc[category_analysis['total_profit'].idxmax()]
            st.metric("Top Category", top.name)
    except Exception as e:
        st.error(f"Cost analysis error: {e}")

def create_expansion_opportunities_chart(df_trans, df_magasin):
    """Market expansion opportunities chart"""
    try:
        store_col = get_column_mapping('Transactions', 'id_magasin')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        date_col = get_column_mapping('Transactions', 'date_heure')
        
        df_clean = df_trans.copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        store_metrics = df_clean.groupby('store_display' if 'store_display' in df_clean else store_col).agg({
            amount_col: ['sum', 'count', 'mean'],
            date_col: ['min', 'max']
        })
        store_metrics.columns = ['revenue', 'transactions', 'avg_ticket', 'first_transaction', 'last_transaction']
        store_metrics['days_active'] = (store_metrics['last_transaction'] - store_metrics['first_transaction']).dt.days + 1
        store_metrics['revenue_per_day'] = store_metrics['revenue'] / store_metrics['days_active']
        store_metrics['maturity_score'] = store_metrics['days_active'] / store_metrics['days_active'].max() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=store_metrics['maturity_score'],
            y=store_metrics['revenue_per_day'],
            mode='markers',
            marker=dict(
                size=np.clip(store_metrics['transactions']/20, 15, 40),
                color=store_metrics['avg_ticket'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Avg Ticket (TND)", thickness=15),
                line=dict(width=1, color='white')
            ),
            text=[f"{idx}<br>Rev/Day: {row.revenue_per_day:,.0f} TND<br>Trans: {row.transactions:,}" 
                  for idx, row in store_metrics.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        fig.add_hline(y=store_metrics['revenue_per_day'].median(), line_dash="dot", line_color="gray")
        fig.add_vline(x=store_metrics['maturity_score'].median(), line_dash="dot", line_color="gray")
        fig.update_layout(
            title="🎯 Expansion Opportunities",
            xaxis=dict(title="Store Maturity (%)", gridcolor='lightgray', showgrid=True),
            yaxis=dict(title="Revenue/Day (TND)", gridcolor='lightgray', showgrid=True),
            plot_bgcolor='white',
            paper_bgcolor=COLORS['background'],
            height=500,
            font_color=COLORS['text'],
            title_font=dict(size=16, color=COLORS['secondary']),
            annotations=[
                dict(x=25, y=store_metrics['revenue_per_day'].quantile(0.75), text="New Stars", showarrow=False, font=dict(color=COLORS['success'])),
                dict(x=75, y=store_metrics['revenue_per_day'].quantile(0.75), text="Champions", showarrow=False, font=dict(color=COLORS['primary'])),
                dict(x=25, y=store_metrics['revenue_per_day'].quantile(0.25), text="Growth Potential", showarrow=False, font=dict(color=COLORS['warning'])),
                dict(x=75, y=store_metrics['revenue_per_day'].quantile(0.25), text="Optimize", showarrow=False, font=dict(color=COLORS['error']))
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        median_maturity, median_performance = store_metrics['maturity_score'].median(), store_metrics['revenue_per_day'].median()
        new_stars = len(store_metrics[(store_metrics['maturity_score'] < median_maturity) & (store_metrics['revenue_per_day'] > median_performance)])
        growth_potential = len(store_metrics[(store_metrics['maturity_score'] < median_maturity) & (store_metrics['revenue_per_day'] < median_performance)])
        champions = len(store_metrics[(store_metrics['maturity_score'] > median_maturity) & (store_metrics['revenue_per_day'] > median_performance)])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"**Rising Stars:** {new_stars} stores")
        with col2:
            st.info(f"**Growth Potential:** {growth_potential} stores")
        with col3:
            st.warning(f"**Champions:** {champions} stores")
    except Exception as e:
        st.error(f"Expansion opportunities error: {e}")

def create_customer_ltv_analysis(df_trans, df_client):
    """Customer lifetime value analysis"""
    try:
        client_col = get_column_mapping('Transactions', 'id_client')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        date_col = get_column_mapping('Transactions', 'date_heure')
        
        df_clean = df_trans.copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[date_col])
        
        customer_metrics = df_clean.groupby(client_col).agg({
            amount_col: ['sum', 'count', 'mean'],
            date_col: ['min', 'max']
        })
        customer_metrics.columns = ['total_spent', 'visit_frequency', 'avg_spend', 'first_visit', 'last_visit']
        customer_metrics['customer_lifespan'] = (customer_metrics['last_visit'] - customer_metrics['first_visit']).dt.days + 1
        customer_metrics['ltv_estimate'] = customer_metrics['total_spent']
        customer_metrics['ltv_quartile'] = pd.qcut(customer_metrics['ltv_estimate'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(
                x=customer_metrics['ltv_estimate'],
                nbinsx=min(30, max(10, len(customer_metrics) // 10)),
                marker=dict(color=COLORS['primary'], line=dict(color='white', width=1)),
                name='LTV Distribution'
            ))
            mean_ltv, median_ltv = customer_metrics['ltv_estimate'].mean(), customer_metrics['ltv_estimate'].median()
            fig1.add_vline(x=mean_ltv, line_dash="solid", line_color=COLORS['error'], annotation_text=f"Mean: {mean_ltv:,.0f} TND")
            fig1.add_vline(x=median_ltv, line_dash="dot", line_color=COLORS['warning'], annotation_text=f"Median: {median_ltv:,.0f} TND")
            fig1.update_layout(
                title="💎 LTV Distribution",
                xaxis=dict(title="Lifetime Value (TND)", gridcolor='lightgray', showgrid=True),
                yaxis=dict(title="Customers", gridcolor='lightgray', showgrid=True),
                plot_bgcolor='white',
                paper_bgcolor=COLORS['background'],
                height=450,
                font_color=COLORS['text'],
                title_font=dict(size=16, color=COLORS['secondary'])
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("**Distribution Summary**")
            st.metric("Mean LTV", f"{mean_ltv:,.0f} TND")
            st.metric("Median LTV", f"{median_ltv:,.0f} TND")
            skewness = "Right-skewed" if mean_ltv > median_ltv else "Left-skewed" if mean_ltv < median_ltv else "Symmetric"
            st.metric("Distribution", skewness)
        
        with col2:
            fig2 = go.Figure()
            color_map = {'Low': COLORS['error'], 'Medium': COLORS['warning'], 'High': COLORS['success'], 'Premium': COLORS['primary']}
            for segment in color_map:
                segment_data = customer_metrics[customer_metrics['ltv_quartile'] == segment]
                if not segment_data.empty:
                    fig2.add_trace(go.Scatter(
                        x=segment_data['visit_frequency'],
                        y=segment_data['avg_spend'],
                        mode='markers',
                        marker=dict(size=np.clip(segment_data['total_spent']/50, 8, 25), color=color_map[segment], line=dict(width=1, color='white')),
                        name=segment,
                        text=[f"Customer {idx}<br>Spent: {spent:,.0f} TND<br>Visits: {visits}" 
                              for idx, spent, visits in zip(segment_data.index, segment_data['total_spent'], segment_data['visit_frequency'])],
                        hovertemplate='%{text}<extra></extra>'
                    ))
            median_freq, median_spend = customer_metrics['visit_frequency'].median(), customer_metrics['avg_spend'].median()
            fig2.add_vline(x=median_freq, line_dash="dot", line_color="gray")
            fig2.add_hline(y=median_spend, line_dash="dot", line_color="gray")
            fig2.update_layout(
                title="🎯 Customer Behavior",
                xaxis=dict(title="Visit Frequency", gridcolor='lightgray', showgrid=True),
                yaxis=dict(title="Avg Spend (TND)", gridcolor='lightgray', showgrid=True),
                plot_bgcolor='white',
                paper_bgcolor=COLORS['background'],
                height=450,
                font_color=COLORS['text'],
                title_font=dict(size=16, color=COLORS['secondary']),
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            high_freq_high_spend = len(customer_metrics[(customer_metrics['visit_frequency'] > median_freq) & (customer_metrics['avg_spend'] > median_spend)])
            total_customers = len(customer_metrics)
            golden_pct = (high_freq_high_spend / total_customers * 100) if total_customers > 0 else 0
            premium_count = len(customer_metrics[customer_metrics['ltv_quartile'] == 'Premium'])
            premium_pct = (premium_count / total_customers * 100) if total_customers > 0 else 0
            st.success(f"**Golden Customers:** {high_freq_high_spend} ({golden_pct:.1f}%)")
            st.info(f"**Premium Segment:** {premium_count} ({premium_pct:.1f}%)")
        
        st.markdown("**Strategic Insights**")
        premium_revenue = customer_metrics[customer_metrics['ltv_quartile'] == 'Premium']['total_spent'].sum()
        premium_contribution = (premium_revenue / customer_metrics['total_spent'].sum() * 100) if customer_metrics['total_spent'].sum() > 0 else 0
        avg_retention = customer_metrics['customer_lifespan'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Premium Revenue Share", f"{premium_contribution:.1f}%")
        with col2:
            st.metric("Avg Lifespan", f"{avg_retention:.0f} days")
        with col3:
            st.metric("Premium Customers", f"{premium_count} ({premium_pct:.1f}%)")
        
        recommendations = []
        if premium_contribution < 60:
            recommendations.append("🎯 Focus on premium customer development")
        if golden_pct < 10:
            recommendations.append("⚡ Boost engagement for high-value customers")
        if avg_retention < 90:
            recommendations.append("🔄 Improve retention with loyalty programs")
        for rec in recommendations:
            st.info(rec)
    except Exception as e:
        st.error(f"LTV analysis error: {e}")

def create_business_performance_dashboard(df_trans):
    """Business performance dashboard"""
    try:
        date_col = get_column_mapping('Transactions', 'date_heure')
        amount_col = get_column_mapping('Transactions', 'montant_total')
        df_clean = df_trans.copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[date_col])
        
        df_clean['month'] = df_clean[date_col].dt.to_period('M')
        monthly_data = df_clean.groupby('month').agg({
            amount_col: ['sum', 'count', 'mean', 'std'],
            date_col: 'nunique'
        })
        monthly_data.columns = ['revenue', 'transactions', 'avg_ticket', 'revenue_std', 'active_days']
        monthly_data['revenue_growth'] = monthly_data['revenue'].pct_change() * 100
        monthly_data['revenue_volatility'] = (monthly_data['revenue_std'] / monthly_data['revenue']) * 100
        monthly_data = monthly_data.reset_index()
        monthly_data['month_str'] = monthly_data['month'].astype(str)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trend', 'Avg Ticket', 'Volatility', 'Acceleration'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        x_numeric = np.arange(len(monthly_data))
        revenue_trend = np.polyfit(x_numeric, monthly_data['revenue'], 1)
        fig.add_trace(
            go.Scatter(x=monthly_data['month_str'], y=monthly_data['revenue'], mode='lines+markers', 
                       name='Revenue', line=dict(color=COLORS['primary'], width=3), fill='tonexty', fillcolor='rgba(30,58,138,0.1)'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=monthly_data['month_str'], y=np.poly1d(revenue_trend)(x_numeric), mode='lines', 
                       name='Trend', line=dict(color=COLORS['accent'], width=2, dash='dash'), showlegend=False),
            row=1, col=1
        )
        
        ticket_trend = np.polyfit(x_numeric, monthly_data['avg_ticket'], 1)
        fig.add_trace(
            go.Scatter(x=monthly_data['month_str'], y=monthly_data['avg_ticket'], mode='lines+markers', 
                       name='Avg Ticket', line=dict(color=COLORS['success'], width=3), fill='tonexty', fillcolor='rgba(16,185,129,0.1)'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=monthly_data['month_str'], y=np.poly1d(ticket_trend)(x_numeric), mode='lines', 
                       name='Ticket Trend', line=dict(color=COLORS['accent'], width=2, dash='dash'), showlegend=False),
            row=1, col=2
        )
        
        volatility_colors = [COLORS['error'] if x > 20 else COLORS['warning'] if x > 10 else COLORS['success'] 
                            for x in monthly_data['revenue_volatility']]
        fig.add_trace(
            go.Bar(x=monthly_data['month_str'], y=monthly_data['revenue_volatility'], name='Volatility %', 
                   marker_color=volatility_colors, text=[f'{x:.1f}%' for x in monthly_data['revenue_volatility']], textposition='outside'),
            row=2, col=1
        )
        
        if len(monthly_data) > 2:
            acceleration = np.gradient(np.gradient(monthly_data['revenue']))
            acceleration_colors = [COLORS['error'] if x < 0 else COLORS['success'] for x in acceleration]
            fig.add_trace(
                go.Bar(x=monthly_data['month_str'], y=acceleration, name='Acceleration', marker_color=acceleration_colors, 
                       text=[f'{x:+.0f}' for x in acceleration], textposition='outside'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="📊 Business Performance",
            plot_bgcolor='white',
            paper_bgcolor=COLORS['background'],
            height=700,
            font_color=COLORS['text'],
            title_font=dict(size=22, color=COLORS['secondary']),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        fig.update_xaxes(tickangle=45, gridcolor='lightgray', showgrid=True)
        fig.update_yaxes(gridcolor='lightgray', showgrid=True)
        fig.update_yaxes(title_text="Revenue (TND)", row=1, col=1)
        fig.update_yaxes(title_text="Avg Ticket (TND)", row=1, col=2)
        fig.update_yaxes(title_text="Volatility %", row=2, col=1)
        fig.update_yaxes(title_text="Acceleration", row=2, col=2)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Key Performance Indicators**")
        col1, col2, col3 = st.columns(3)
        latest_month = monthly_data.iloc[-1] if not monthly_data.empty else None
        if latest_month is not None:
            with col1:
                st.metric("Monthly Revenue", f"{latest_month['revenue']:,.0f} TND", 
                          f"{latest_month['revenue_growth']:+.1f}%" if not pd.isna(latest_month['revenue_growth']) else "N/A")
            with col2:
                st.metric("Monthly Transactions", f"{latest_month['transactions']:,.0f}")
            with col3:
                st.metric("Avg Volatility", f"{monthly_data['revenue_volatility'].mean():.1f}%")
    except Exception as e:
        st.error(f"Business performance error: {e}")



if __name__ == "__main__":
    business_owner_dashboard()