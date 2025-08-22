import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_generator import DataGenerator
from credit_scoring import CreditScoringEngine
from explainability import ExplainabilityEngine
from dashboard import DashboardComponents

# Page configuration
st.set_page_config(
    page_title="Credit Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DataGenerator()
    st.session_state.scoring_engine = CreditScoringEngine()
    st.session_state.explainability_engine = ExplainabilityEngine()
    st.session_state.dashboard = DashboardComponents()

def main():
    st.title("🏦 Explainable Credit Intelligence Platform")
    st.markdown("### Real-time credit scoring with ML-driven insights and transparent explanations")
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Page:",
            ["Dashboard", "Company Analysis", "Model Insights", "Data Sources", "Alerts", 
             "Portfolio Management", "Stress Testing", "Risk Assessment", "Benchmarking", 
             "Market Insights", "Reports & Export", "Settings"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        # Generate sample data for quick stats
        companies_data = st.session_state.data_generator.generate_companies_data(10)
        avg_score = companies_data['credit_score'].mean()
        score_change = np.random.uniform(-5, 5)
        
        st.metric("Average Credit Score", f"{avg_score:.1f}", f"{score_change:+.1f}")
        st.metric("Companies Monitored", "10", "+2")
        st.metric("Data Sources Active", "4", "0")
    
    # Main content based on selected page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Company Analysis":
        render_company_analysis()
    elif page == "Model Insights":
        render_model_insights()
    elif page == "Data Sources":
        render_data_sources()
    elif page == "Alerts":
        render_alerts()
    elif page == "Portfolio Management":
        render_portfolio_management()
    elif page == "Stress Testing":
        render_stress_testing()
    elif page == "Risk Assessment":
        render_risk_assessment()
    elif page == "Benchmarking":
        render_benchmarking()
    elif page == "Market Insights":
        render_market_insights()
    elif page == "Reports & Export":
        render_reports_export()
    elif page == "Settings":
        render_settings()

def render_dashboard():
    st.header("📊 Credit Intelligence Dashboard")
    
    # Generate dashboard data
    companies_data = st.session_state.data_generator.generate_companies_data(10)
    market_data = st.session_state.data_generator.generate_market_data()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = companies_data['credit_score'].mean()
        st.metric("Average Credit Score", f"{avg_score:.1f}", "+1.2")
    
    with col2:
        high_risk = len(companies_data[companies_data['credit_score'] < 600])
        st.metric("High Risk Companies", high_risk, "-1")
    
    with col3:
        score_volatility = companies_data['credit_score'].std()
        st.metric("Score Volatility", f"{score_volatility:.1f}", "+0.5")
    
    with col4:
        last_update = datetime.now().strftime("%H:%M")
        st.metric("Last Update", last_update, "Live")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Credit Score Distribution")
        fig = px.histogram(
            companies_data, 
            x='credit_score', 
            nbins=15,
            title="Distribution of Credit Scores",
            labels={'credit_score': 'Credit Score', 'count': 'Number of Companies'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Distribution by Sector")
        sector_risk = companies_data.groupby('sector')['credit_score'].mean().reset_index()
        fig = px.bar(
            sector_risk,
            x='sector',
            y='credit_score',
            title="Average Credit Score by Sector",
            labels={'credit_score': 'Average Credit Score', 'sector': 'Sector'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Companies table
    st.subheader("📋 Companies Overview")
    
    # Add risk category
    companies_data['risk_category'] = companies_data['credit_score'].apply(
        lambda x: 'Low Risk' if x >= 700 else 'Medium Risk' if x >= 600 else 'High Risk'
    )
    
    # Color code the risk categories
    def color_risk(val):
        if val == 'Low Risk':
            return 'background-color: #d4edda'
        elif val == 'Medium Risk':
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    styled_df = companies_data[['company_name', 'sector', 'credit_score', 'risk_category']].style.applymap(
        color_risk, subset=['risk_category']
    )
    
    st.dataframe(styled_df, use_container_width=True)

def render_company_analysis():
    st.header("🏢 Company Analysis")
    
    # Company selector
    companies_data = st.session_state.data_generator.generate_companies_data(10)
    selected_company = st.selectbox(
        "Select Company for Analysis:",
        companies_data['company_name'].tolist()
    )
    
    if selected_company:
        company_data = companies_data[companies_data['company_name'] == selected_company].iloc[0]
        
        # Company overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Credit Score", f"{company_data['credit_score']:.1f}")
        with col2:
            st.metric("Sector", company_data['sector'])
        with col3:
            risk_category = 'Low Risk' if company_data['credit_score'] >= 700 else 'Medium Risk' if company_data['credit_score'] >= 600 else 'High Risk'
            st.metric("Risk Category", risk_category)
        
        # Generate detailed financial data for the selected company
        financial_data = st.session_state.data_generator.generate_financial_metrics(selected_company)
        
        # Train model and get predictions
        X, feature_names = st.session_state.scoring_engine.prepare_features(financial_data)
        # Create y values that match X dimensions
        y_values = [company_data['credit_score']] * len(X)
        model = st.session_state.scoring_engine.train_model(X, y_values)
        prediction = st.session_state.scoring_engine.predict(model, X)
        
        # Get SHAP explanations
        shap_values, feature_importance = st.session_state.explainability_engine.get_shap_explanation(
            model, X, feature_names
        )
        
        # Feature importance chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df.tail(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 SHAP Feature Contributions")
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Value': shap_values[0] if len(shap_values.shape) > 1 else shap_values
            }).sort_values('SHAP_Value', key=abs, ascending=False).head(10)
            
            fig = px.bar(
                shap_df,
                x='SHAP_Value',
                y='Feature',
                orientation='h',
                title="Feature Contributions to Credit Score",
                color='SHAP_Value',
                color_continuous_scale=['red', 'white', 'green']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Historical trend
        st.subheader("📊 Credit Score Trend")
        historical_data = st.session_state.data_generator.generate_historical_scores(selected_company)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['credit_score'],
            mode='lines+markers',
            name='Credit Score',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title="Credit Score Over Time",
            xaxis_title="Date",
            yaxis_title="Credit Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # News sentiment analysis
        st.subheader("📰 Recent News Impact")
        news_data = st.session_state.data_generator.generate_news_sentiment(selected_company)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            for _, news in news_data.iterrows():
                sentiment_color = "🟢" if news['sentiment'] > 0.1 else "🔴" if news['sentiment'] < -0.1 else "🟡"
                st.write(f"{sentiment_color} **{news['headline']}**")
                st.write(f"*Sentiment Score: {news['sentiment']:.2f} | {news['date']}*")
                st.write("---")
        
        with col2:
            avg_sentiment = news_data['sentiment'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            
            sentiment_impact = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            st.metric("Overall Impact", sentiment_impact)

def render_model_insights():
    st.header("🤖 Model Insights & Performance")
    
    # Generate sample data for model evaluation
    companies_data = st.session_state.data_generator.generate_companies_data(50)
    
    # Prepare features and train model
    all_financial_data = []
    all_scores = []
    
    for _, company in companies_data.iterrows():
        financial_data = st.session_state.data_generator.generate_financial_metrics(company['company_name'])
        all_financial_data.append(financial_data)
        # Create scores for each row in the financial data
        for _ in range(len(financial_data)):
            all_scores.append(company['credit_score'])
    
    # Combine all financial data
    combined_data = pd.concat(all_financial_data, ignore_index=True)
    X, feature_names = st.session_state.scoring_engine.prepare_features(combined_data)
    
    # Train and evaluate model
    model = st.session_state.scoring_engine.train_model(X, all_scores)
    predictions = st.session_state.scoring_engine.predict(model, X)
    
    # Model performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mae = np.mean(np.abs(np.array(all_scores) - predictions))
        st.metric("Mean Absolute Error", f"{mae:.2f}")
    
    with col2:
        rmse = np.sqrt(np.mean((np.array(all_scores) - predictions) ** 2))
        st.metric("RMSE", f"{rmse:.2f}")
    
    with col3:
        r2 = 1 - (np.sum((np.array(all_scores) - predictions) ** 2) / 
                  np.sum((np.array(all_scores) - np.mean(all_scores)) ** 2))
        st.metric("R² Score", f"{r2:.3f}")
    
    with col4:
        accuracy_within_10 = np.mean(np.abs(np.array(all_scores) - predictions) <= 10) * 100
        st.metric("Accuracy (±10)", f"{accuracy_within_10:.1f}%")
    
    # Prediction vs Actual scatter plot
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction vs Actual")
        fig = px.scatter(
            x=all_scores,
            y=predictions,
            title="Model Predictions vs Actual Scores",
            labels={'x': 'Actual Credit Score', 'y': 'Predicted Credit Score'}
        )
        
        # Add perfect prediction line
        min_score, max_score = min(all_scores), max(all_scores)
        fig.add_trace(go.Scatter(
            x=[min_score, max_score],
            y=[min_score, max_score],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Feature Importance Global")
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True).tail(15)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Global Feature Importance"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.subheader("🔄 Model Comparison")
    
    # Compare different models
    model_results = st.session_state.scoring_engine.compare_models(X, all_scores)
    
    results_df = pd.DataFrame(model_results).T
    results_df = results_df.round(3)
    
    st.dataframe(results_df, use_container_width=True)
    
    # Model performance over time
    st.subheader("⏱️ Model Performance Over Time")
    
    performance_history = st.session_state.data_generator.generate_model_performance_history()
    
    fig = go.Figure()
    
    for metric in ['MAE', 'RMSE', 'R2']:
        fig.add_trace(go.Scatter(
            x=performance_history['date'],
            y=performance_history[metric],
            mode='lines+markers',
            name=metric
        ))
    
    fig.update_layout(
        title="Model Performance Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Metric Value",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def render_data_sources():
    st.header("📡 Data Sources & Pipeline Status")
    
    # Data source status
    data_sources = [
        {"name": "Financial Ratios", "status": "Active", "last_update": "2 min ago", "records": 1250},
        {"name": "Market Data", "status": "Active", "last_update": "1 min ago", "records": 5420},
        {"name": "Economic Indicators", "status": "Active", "last_update": "5 min ago", "records": 230},
        {"name": "News Sentiment", "status": "Active", "last_update": "3 min ago", "records": 890},
        {"name": "SEC Filings", "status": "Warning", "last_update": "15 min ago", "records": 45},
        {"name": "Credit Agency Ratings", "status": "Inactive", "last_update": "1 hour ago", "records": 120}
    ]
    
    # Create status indicators
    status_colors = {"Active": "🟢", "Warning": "🟡", "Inactive": "🔴"}
    
    for source in data_sources:
        col1, col2, col3, col4, col5 = st.columns([3, 1, 2, 2, 1])
        
        with col1:
            st.write(f"**{source['name']}**")
        with col2:
            st.write(f"{status_colors[source['status']]} {source['status']}")
        with col3:
            st.write(source['last_update'])
        with col4:
            st.write(f"{source['records']:,} records")
        with col5:
            if st.button("Refresh", key=f"refresh_{source['name']}"):
                st.success(f"Refreshed {source['name']}")
    
    st.markdown("---")
    
    # Data quality metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Quality Metrics")
        
        quality_data = st.session_state.data_generator.generate_data_quality_metrics()
        
        fig = px.bar(
            quality_data,
            x='source',
            y='quality_score',
            title="Data Quality Score by Source",
            color='quality_score',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Data Pipeline Throughput")
        
        throughput_data = st.session_state.data_generator.generate_throughput_data()
        
        fig = px.line(
            throughput_data,
            x='timestamp',
            y='records_per_minute',
            title="Data Ingestion Rate (Records/Minute)",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent data samples
    st.subheader("🔍 Recent Data Samples")
    
    tab1, tab2, tab3 = st.tabs(["Financial Data", "Market Data", "News Sentiment"])
    
    with tab1:
        financial_sample = st.session_state.data_generator.generate_financial_metrics("Sample Corp")
        st.dataframe(financial_sample.head(), use_container_width=True)
    
    with tab2:
        market_sample = st.session_state.data_generator.generate_market_data()
        st.dataframe(market_sample.head(), use_container_width=True)
    
    with tab3:
        news_sample = st.session_state.data_generator.generate_news_sentiment("Sample Corp")
        st.dataframe(news_sample, use_container_width=True)

def render_alerts():
    st.header("🚨 Alerts & Monitoring")
    
    # Alert configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Alert Settings")
        
        score_threshold = st.slider("Score Change Threshold", 1, 50, 10)
        enable_email = st.checkbox("Email Notifications", value=True)
        enable_sms = st.checkbox("SMS Notifications", value=False)
        
        st.selectbox("Alert Frequency", ["Immediate", "Hourly", "Daily"])
        
        if st.button("Save Settings"):
            st.success("Alert settings saved!")
    
    with col2:
        st.subheader("🔔 Recent Alerts")
        
        alerts_data = st.session_state.data_generator.generate_alerts()
        
        for _, alert in alerts_data.iterrows():
            severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            
            with st.expander(f"{severity_color[alert['severity']]} {alert['title']} - {alert['timestamp']}"):
                st.write(f"**Company:** {alert['company']}")
                st.write(f"**Message:** {alert['message']}")
                st.write(f"**Severity:** {alert['severity']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Acknowledge", key=f"ack_{alert['id']}"):
                        st.success("Alert acknowledged")
                with col2:
                    if st.button("Dismiss", key=f"dismiss_{alert['id']}"):
                        st.success("Alert dismissed")
                with col3:
                    if st.button("View Details", key=f"details_{alert['id']}"):
                        st.info("Redirecting to company analysis...")
    
    # Alert statistics
    st.subheader("📈 Alert Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alerts = len(alerts_data)
        st.metric("Total Alerts (24h)", total_alerts)
    
    with col2:
        high_severity = len(alerts_data[alerts_data['severity'] == 'High'])
        st.metric("High Severity", high_severity)
    
    with col3:
        avg_response_time = np.random.randint(5, 30)
        st.metric("Avg Response Time", f"{avg_response_time} min")
    
    with col4:
        resolved_rate = np.random.randint(85, 95)
        st.metric("Resolution Rate", f"{resolved_rate}%")
    
    # Alert trends
    col1, col2 = st.columns(2)
    
    with col1:
        alert_trends = st.session_state.data_generator.generate_alert_trends()
        
        fig = px.line(
            alert_trends,
            x='date',
            y='alert_count',
            color='severity',
            title="Alert Trends by Severity"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        alert_by_company = alerts_data.groupby('company').size().reset_index(name='count')
        
        fig = px.pie(
            alert_by_company,
            values='count',
            names='company',
            title="Alerts by Company"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_portfolio_management():
    st.header("💼 Portfolio Management")
    
    # Portfolio creation and management
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Create Portfolio")
        portfolio_name = st.text_input("Portfolio Name")
        companies_data = st.session_state.data_generator.generate_companies_data(20)
        selected_companies = st.multiselect(
            "Select Companies",
            companies_data['company_name'].tolist(),
            default=companies_data['company_name'].tolist()[:5]
        )
        
        if st.button("Create Portfolio") and portfolio_name:
            st.success(f"Portfolio '{portfolio_name}' created with {len(selected_companies)} companies")
    
    with col2:
        st.subheader("📈 Portfolio Analytics")
        
        if selected_companies:
            portfolio_data = companies_data[companies_data['company_name'].isin(selected_companies)]
            
            # Portfolio metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_score = portfolio_data['credit_score'].mean()
                st.metric("Average Score", f"{avg_score:.1f}")
            
            with col2:
                portfolio_risk = portfolio_data['credit_score'].std()
                st.metric("Portfolio Risk", f"{portfolio_risk:.1f}")
            
            with col3:
                high_risk_count = len(portfolio_data[portfolio_data['credit_score'] < 600])
                st.metric("High Risk Companies", high_risk_count)
            
            with col4:
                diversification = len(portfolio_data['sector'].unique())
                st.metric("Sector Diversity", diversification)
            
            # Portfolio composition
            fig = px.pie(
                portfolio_data,
                names='sector',
                title="Portfolio Composition by Sector"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio comparison
    st.subheader("📊 Risk-Return Analysis")
    
    risk_return_data = []
    for _, company in companies_data.iterrows():
        risk_return_data.append({
            'Company': company['company_name'],
            'Risk': np.random.uniform(5, 25),
            'Expected_Return': company['credit_score'] / 10 + np.random.uniform(-2, 2),
            'Credit_Score': company['credit_score']
        })
    
    risk_return_df = pd.DataFrame(risk_return_data)
    
    fig = px.scatter(
        risk_return_df,
        x='Risk',
        y='Expected_Return',
        size='Credit_Score',
        color='Credit_Score',
        hover_name='Company',
        title="Risk-Return Profile",
        labels={'Risk': 'Risk Level', 'Expected_Return': 'Expected Return (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)

def render_stress_testing():
    st.header("⚡ Stress Testing & Scenario Analysis")
    
    # Stress test scenarios
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛️ Scenario Configuration")
        
        scenario_type = st.selectbox(
            "Stress Test Scenario",
            ["Economic Recession", "Interest Rate Shock", "Market Crash", "Sector Crisis", "Custom"]
        )
        
        if scenario_type == "Custom":
            gdp_shock = st.slider("GDP Growth Change (%)", -10.0, 5.0, -3.0, 0.1)
            interest_shock = st.slider("Interest Rate Change (%)", -5.0, 10.0, 2.0, 0.1)
            unemployment_shock = st.slider("Unemployment Change (%)", -2.0, 8.0, 3.0, 0.1)
        else:
            # Predefined scenarios
            scenarios = {
                "Economic Recession": {"gdp": -4.0, "interest": 1.5, "unemployment": 4.0},
                "Interest Rate Shock": {"gdp": -1.0, "interest": 5.0, "unemployment": 1.0},
                "Market Crash": {"gdp": -2.0, "interest": 0.5, "unemployment": 2.0},
                "Sector Crisis": {"gdp": -1.5, "interest": 1.0, "unemployment": 2.5}
            }
            
            if scenario_type in scenarios:
                gdp_shock = scenarios[scenario_type]["gdp"]
                interest_shock = scenarios[scenario_type]["interest"]
                unemployment_shock = scenarios[scenario_type]["unemployment"]
            else:
                gdp_shock, interest_shock, unemployment_shock = 0, 0, 0
        
        st.write(f"**GDP Impact:** {gdp_shock:+.1f}%")
        st.write(f"**Interest Rate Impact:** {interest_shock:+.1f}%")
        st.write(f"**Unemployment Impact:** {unemployment_shock:+.1f}%")
        
        run_stress_test = st.button("Run Stress Test", type="primary")
    
    with col2:
        st.subheader("📊 Stress Test Results")
        
        if run_stress_test or 'stress_test_run' not in st.session_state:
            st.session_state.stress_test_run = True
            
            # Generate companies data
            companies_data = st.session_state.data_generator.generate_companies_data(15)
            
            # Simulate stress impact on credit scores
            stress_results = []
            for _, company in companies_data.iterrows():
                base_score = company['credit_score']
                
                # Calculate stress impact based on sector sensitivity
                sector_sensitivity = {
                    'Financial Services': 1.5, 'Real Estate': 1.3, 'Energy': 1.2,
                    'Manufacturing': 1.1, 'Technology': 0.8, 'Healthcare': 0.7
                }
                
                sensitivity = sector_sensitivity.get(company['sector'], 1.0)
                
                # Apply stress scenario
                stress_impact = (gdp_shock * 2 + interest_shock * 1.5 + unemployment_shock * 1.0) * sensitivity
                stressed_score = max(300, base_score + stress_impact)
                
                stress_results.append({
                    'Company': company['company_name'],
                    'Sector': company['sector'],
                    'Base_Score': base_score,
                    'Stressed_Score': stressed_score,
                    'Impact': stressed_score - base_score,
                    'Rating_Change': 'Downgraded' if stressed_score < base_score - 10 else 'Stable'
                })
            
            stress_df = pd.DataFrame(stress_results)
            
            # Show impact summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_impact = stress_df['Impact'].mean()
                st.metric("Avg Score Impact", f"{avg_impact:+.1f}")
            
            with col2:
                downgrades = len(stress_df[stress_df['Impact'] < -10])
                st.metric("Downgrades", downgrades)
            
            with col3:
                worst_impact = stress_df['Impact'].min()
                st.metric("Worst Impact", f"{worst_impact:+.1f}")
            
            with col4:
                at_risk = len(stress_df[stress_df['Stressed_Score'] < 600])
                st.metric("High Risk After", at_risk)
            
            # Stress test visualization
            fig = px.scatter(
                stress_df,
                x='Base_Score',
                y='Stressed_Score',
                color='Sector',
                size=abs(stress_df['Impact']),
                hover_name='Company',
                title="Credit Score Impact from Stress Test"
            )
            
            # Add diagonal line for no-impact reference
            fig.add_trace(go.Scatter(
                x=[300, 850],
                y=[300, 850],
                mode='lines',
                name='No Impact Line',
                line=dict(dash='dash', color='gray')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            st.dataframe(stress_df, use_container_width=True)

def render_risk_assessment():
    st.header("⚠️ Advanced Risk Assessment")
    
    # Risk metrics dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Value at Risk (VaR)")
        confidence_level = st.slider("Confidence Level", 90, 99, 95)
        time_horizon = st.selectbox("Time Horizon", ["1 Day", "1 Week", "1 Month", "1 Quarter"])
        
        # Simulate VaR calculation
        portfolio_value = 10000000  # $10M portfolio
        var_percentage = np.random.uniform(2, 8)
        var_amount = portfolio_value * (var_percentage / 100)
        
        st.metric("VaR (95%)", f"${var_amount:,.0f}", f"{var_percentage:.1f}% of portfolio")
        st.metric("Expected Shortfall", f"${var_amount * 1.3:,.0f}", f"{var_percentage * 1.3:.1f}% of portfolio")
    
    with col2:
        st.subheader("🎯 Risk Concentration")
        
        companies_data = st.session_state.data_generator.generate_companies_data(10)
        
        # Risk concentration by sector
        sector_exposure = companies_data.groupby('sector').size().reset_index(name='count')
        sector_exposure['percentage'] = (sector_exposure['count'] / sector_exposure['count'].sum() * 100).round(1)
        
        fig = px.pie(
            sector_exposure,
            names='sector',
            values='percentage',
            title="Risk Concentration by Sector"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("📈 Risk Trends")
        
        # Generate risk trend data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        risk_trend_data = pd.DataFrame({
            'Date': dates,
            'Portfolio_Risk': np.random.uniform(15, 25, 30) + np.sin(np.arange(30) * 0.2) * 3,
            'Market_Risk': np.random.uniform(12, 20, 30) + np.cos(np.arange(30) * 0.15) * 2
        })
        
        fig = px.line(
            risk_trend_data,
            x='Date',
            y=['Portfolio_Risk', 'Market_Risk'],
            title="Risk Metrics Trend (30 Days)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk heat map
    st.subheader("🔥 Risk Heat Map")
    
    # Create correlation matrix
    risk_factors = ['Interest Rate', 'GDP Growth', 'Inflation', 'Unemployment', 'Market Vol', 'Credit Spread']
    correlation_matrix = np.random.uniform(-0.8, 0.8, (6, 6))
    np.fill_diagonal(correlation_matrix, 1.0)
    
    fig = px.imshow(
        correlation_matrix,
        x=risk_factors,
        y=risk_factors,
        title="Risk Factor Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_benchmarking():
    st.header("📊 Industry Benchmarking")
    
    # Industry selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🏭 Select Industry")
        
        industry = st.selectbox(
            "Industry Sector",
            ['Technology', 'Financial Services', 'Healthcare', 'Energy', 'Manufacturing', 'Retail']
        )
        
        benchmark_type = st.selectbox(
            "Benchmark Type",
            ["Peer Comparison", "Historical Trend", "Credit Rating Distribution"]
        )
        
        companies_data = st.session_state.data_generator.generate_companies_data(20)
        industry_companies = companies_data[companies_data['sector'] == industry]
        
        st.write(f"**Companies in {industry}:** {len(industry_companies)}")
        st.write(f"**Average Credit Score:** {industry_companies['credit_score'].mean():.1f}")
    
    with col2:
        st.subheader(f"📈 {industry} Benchmarks")
        
        if benchmark_type == "Peer Comparison":
            # Peer comparison chart
            fig = px.box(
                companies_data,
                x='sector',
                y='credit_score',
                title="Credit Score Distribution by Sector"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Industry rankings
            industry_stats = companies_data.groupby('sector')['credit_score'].agg(['mean', 'std']).reset_index()
            industry_stats.columns = ['Sector', 'Avg_Score', 'Score_Volatility']
            industry_stats = industry_stats.sort_values('Avg_Score', ascending=False)
            
            st.subheader("🏆 Industry Rankings")
            st.dataframe(industry_stats, use_container_width=True)
            
        elif benchmark_type == "Historical Trend":
            # Historical performance
            dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
            trend_data = pd.DataFrame({
                'Month': dates,
                f'{industry}': np.random.uniform(650, 720, 12),
                'Market Average': np.random.uniform(640, 700, 12)
            })
            
            fig = px.line(
                trend_data,
                x='Month',
                y=[f'{industry}', 'Market Average'],
                title=f"{industry} vs Market Average (12 Months)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Credit Rating Distribution
            # Rating distribution
            ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C']
            rating_counts = np.random.randint(1, 15, len(ratings))
            
            rating_df = pd.DataFrame({
                'Rating': ratings,
                'Count': rating_counts,
                'Percentage': (rating_counts / rating_counts.sum() * 100).round(1)
            })
            
            fig = px.bar(
                rating_df,
                x='Rating',
                y='Count',
                title=f"Credit Rating Distribution - {industry}"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_market_insights():
    st.header("🌍 Market Insights & Economic Indicators")
    
    # Economic indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gdp_growth = np.random.uniform(2.0, 4.0)
        st.metric("GDP Growth", f"{gdp_growth:.1f}%", f"+{np.random.uniform(0.1, 0.5):.1f}%")
    
    with col2:
        inflation = np.random.uniform(2.0, 6.0)
        st.metric("Inflation Rate", f"{inflation:.1f}%", f"+{np.random.uniform(-0.2, 0.3):.1f}%")
    
    with col3:
        unemployment = np.random.uniform(3.5, 7.0)
        st.metric("Unemployment", f"{unemployment:.1f}%", f"{np.random.uniform(-0.3, 0.2):+.1f}%")
    
    with col4:
        fed_rate = np.random.uniform(4.0, 6.0)
        st.metric("Fed Rate", f"{fed_rate:.2f}%", f"+{np.random.uniform(0, 0.25):.2f}%")
    
    # Market trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Credit Market Trends")
        
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        market_data = pd.DataFrame({
            'Date': dates,
            'Credit_Spreads': np.random.uniform(150, 300, 90) + np.sin(np.arange(90) * 0.1) * 20,
            'Default_Rate': np.random.uniform(1, 4, 90) + np.cos(np.arange(90) * 0.08) * 0.5
        })
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Credit Spreads (bps)", "Default Rate (%)"),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=market_data['Date'], y=market_data['Credit_Spreads'], name="Credit Spreads"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=market_data['Date'], y=market_data['Default_Rate'], name="Default Rate"),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Sector Performance")
        
        sectors = ['Technology', 'Financial Services', 'Healthcare', 'Energy', 'Manufacturing', 'Retail']
        performance_data = pd.DataFrame({
            'Sector': sectors,
            'YTD_Performance': np.random.uniform(-15, 25, len(sectors)),
            'Credit_Quality_Change': np.random.uniform(-10, 15, len(sectors))
        })
        
        fig = px.scatter(
            performance_data,
            x='YTD_Performance',
            y='Credit_Quality_Change',
            text='Sector',
            title="Sector Performance vs Credit Quality",
            labels={'YTD_Performance': 'YTD Performance (%)', 'Credit_Quality_Change': 'Credit Quality Change'}
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        
        # Economic calendar
        st.subheader("📅 Economic Calendar")
        economic_events = pd.DataFrame({
            'Date': pd.date_range(start='2024-08-21', periods=5, freq='D'),
            'Event': ['Fed Meeting Minutes', 'GDP Report', 'Employment Data', 'CPI Release', 'Retail Sales'],
            'Impact': ['High', 'High', 'Medium', 'High', 'Low'],
            'Forecast': ['Neutral', 'Growth', 'Improvement', 'Inflation concerns', 'Stable']
        })
        st.dataframe(economic_events, use_container_width=True)

def render_reports_export():
    st.header("📄 Reports & Export")
    
    # Report generation
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Generate Reports")
        
        report_type = st.selectbox(
            "Report Type",
            ["Portfolio Summary", "Risk Assessment", "Credit Analysis", "Benchmark Report", "Stress Test Results"]
        )
        
        date_range = st.date_input(
            "Report Period",
            value=(datetime.now().date() - timedelta(days=30), datetime.now().date())
        )
        
        include_charts = st.checkbox("Include Charts", value=True)
        include_details = st.checkbox("Include Detailed Analysis", value=True)
        
        export_format = st.selectbox("Export Format", ["PDF", "Excel", "CSV", "JSON"])
        
        if st.button("Generate Report", type="primary"):
            st.success(f"✅ {report_type} report generated successfully!")
            st.info(f"📥 Report available for download in {export_format} format")
    
    with col2:
        st.subheader("📋 Report Preview")
        
        # Sample report content
        st.write(f"**Report Type:** {report_type}")
        try:
            # Handle different date_range formats safely
            if hasattr(date_range, '__len__') and len(date_range) >= 2:
                date_list = list(date_range)
                start_date = date_list[0] if date_list else datetime.now().date()
                end_date = date_list[1] if len(date_list) > 1 else start_date
                st.write(f"**Period:** {start_date} to {end_date}")
            elif hasattr(date_range, '__len__') and len(date_range) == 1:
                date_list = list(date_range)
                st.write(f"**Period:** {date_list[0] if date_list else datetime.now().date()}")
            else:
                st.write(f"**Period:** {date_range}")
        except (IndexError, TypeError, ValueError):
            st.write(f"**Period:** {datetime.now().date()}")
        st.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown("---")
        
        if report_type == "Portfolio Summary":
            st.write("### Portfolio Overview")
            companies_data = st.session_state.data_generator.generate_companies_data(10)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Companies", len(companies_data))
            with col2:
                st.metric("Avg Credit Score", f"{companies_data['credit_score'].mean():.1f}")
            with col3:
                st.metric("High Risk Count", len(companies_data[companies_data['credit_score'] < 600]))
                
            st.dataframe(companies_data[['company_name', 'sector', 'credit_score']].head(), use_container_width=True)
        
        # Download buttons (simulated)
        st.subheader("📥 Download Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Download PDF"):
                st.success("PDF download started")
        
        with col2:
            if st.button("📈 Download Excel"):
                st.success("Excel download started")
        
        with col3:
            if st.button("📋 Download CSV"):
                st.success("CSV download started")
        
        with col4:
            if st.button("💾 Download JSON"):
                st.success("JSON download started")
    
    # Scheduled reports
    st.subheader("⏰ Scheduled Reports")
    
    scheduled_reports = pd.DataFrame({
        'Report Name': ['Daily Risk Summary', 'Weekly Portfolio Review', 'Monthly Benchmark'],
        'Type': ['Risk Assessment', 'Portfolio Summary', 'Benchmark Report'],
        'Schedule': ['Daily 9:00 AM', 'Monday 8:00 AM', '1st of Month'],
        'Last Generated': ['2024-08-20 09:00', '2024-08-19 08:00', '2024-08-01 09:30'],
        'Status': ['Active', 'Active', 'Active']
    })
    
    st.dataframe(scheduled_reports, use_container_width=True)

def render_settings():
    st.header("⚙️ Settings & Configuration")
    
    # User preferences
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 User Preferences")
        
        display_name = st.text_input("Display Name", value="Credit Analyst")
        email_notifications = st.checkbox("Email Notifications", value=True)
        sms_alerts = st.checkbox("SMS Alerts for High Risk", value=False)
        
        refresh_interval = st.selectbox(
            "Data Refresh Interval",
            ["Real-time", "5 minutes", "15 minutes", "30 minutes", "1 hour"]
        )
        
        default_portfolio = st.selectbox(
            "Default Portfolio View",
            ["All Companies", "High Risk Only", "My Portfolio", "Custom"]
        )
        
        if st.button("Save User Preferences"):
            st.success("✅ User preferences saved!")
    
    with col2:
        st.subheader("🎛️ Model Configuration")
        
        model_type = st.selectbox(
            "Default ML Model",
            ["Random Forest", "Gradient Boosting", "Linear Regression", "Decision Tree"]
        )
        
        risk_threshold = st.slider("High Risk Threshold", 300, 700, 600, 10)
        confidence_level = st.slider("Confidence Level for VaR", 90, 99, 95, 1)
        
        update_frequency = st.selectbox(
            "Model Retraining Frequency",
            ["Daily", "Weekly", "Monthly", "Quarterly"]
        )
        
        feature_selection = st.multiselect(
            "Key Features to Monitor",
            ["Credit Score", "Debt Ratio", "ROE", "Current Ratio", "Market Cap"],
            default=["Credit Score", "Debt Ratio", "ROE"]
        )
        
        if st.button("Save Model Configuration"):
            st.success("✅ Model configuration saved!")
    
    # Data source settings
    st.subheader("🔗 Data Source Settings")
    
    data_sources = pd.DataFrame({
        'Data Source': ['Bloomberg API', 'SEC EDGAR', 'Market Data Feed', 'Credit Agencies', 'News API'],
        'Status': ['Connected', 'Connected', 'Connected', 'Error', 'Connected'],
        'Last Sync': ['2 min ago', '5 min ago', '1 min ago', '2 hours ago', '10 min ago'],
        'Records Today': ['1,250', '45', '5,420', '0', '890']
    })
    
    st.dataframe(data_sources, use_container_width=True)
    
    # API configuration
    with st.expander("🔧 API Configuration"):
        st.text_input("Bloomberg API Key", type="password", placeholder="Enter Bloomberg API key...")
        st.text_input("News API Key", type="password", placeholder="Enter News API key...")
        st.text_input("Credit Agency API Key", type="password", placeholder="Enter Credit Agency API key...")
        
        if st.button("Test API Connections"):
            st.success("✅ All API connections tested successfully!")
    
    # System information
    st.subheader("💻 System Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("App Version", "v2.1.3")
    
    with col2:
        st.metric("Database Size", "2.4 GB")
    
    with col3:
        st.metric("Active Users", "23")
    
    with col4:
        st.metric("Uptime", "99.9%")

if __name__ == "__main__":
    main()
