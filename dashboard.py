import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class DashboardComponents:
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def create_metric_card(self, title, value, delta=None, help_text=None):
        """Create a styled metric card"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                help=help_text
            )
    
    def create_score_gauge(self, score, title="Credit Score"):
        """Create a gauge chart for credit score"""
        
        # Determine color based on score
        if score >= 700:
            color = 'green'
        elif score >= 600:
            color = 'yellow'
        else:
            color = 'red'
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 650},
            gauge = {
                'axis': {'range': [None, 850]},
                'bar': {'color': color},
                'steps': [
                    {'range': [300, 550], 'color': "lightgray"},
                    {'range': [550, 650], 'color': "gray"},
                    {'range': [650, 750], 'color': "lightblue"},
                    {'range': [750, 850], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 600
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_waterfall_chart(self, waterfall_data):
        """Create waterfall chart for feature contributions"""
        
        x_values = [item['feature'] for item in waterfall_data]
        y_values = [item['contribution'] for item in waterfall_data]
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Base score
        fig.add_trace(go.Waterfall(
            name="Credit Score Components",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(waterfall_data) - 2) + ["total"],
            x=x_values,
            textposition="outside",
            text=[f"{y:.1f}" for y in y_values],
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#d62728"}},
            increasing={"marker": {"color": "#2ca02c"}},
            totals={"marker": {"color": "#1f77b4"}}
        ))
        
        fig.update_layout(
            title="Credit Score Breakdown",
            showlegend=False,
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_names, importance_values, title="Feature Importance"):
        """Create horizontal bar chart for feature importance"""
        
        # Create DataFrame and sort by importance
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True)
        
        # Take top 15 features
        df = df.tail(15)
        
        fig = px.bar(
            df,
            x='importance',
            y='feature',
            orientation='h',
            title=title,
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_score_trend_chart(self, trend_data, title="Credit Score Trend"):
        """Create line chart for score trends"""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_data['date'],
            y=trend_data['credit_score'],
            mode='lines+markers',
            name='Credit Score',
            line=dict(width=3, color=self.color_palette['primary']),
            marker=dict(size=6)
        ))
        
        # Add horizontal lines for rating thresholds
        fig.add_hline(y=700, line_dash="dash", line_color="green", 
                     annotation_text="Investment Grade", annotation_position="top right")
        fig.add_hline(y=600, line_dash="dash", line_color="orange", 
                     annotation_text="Sub-Investment Grade", annotation_position="bottom right")
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Credit Score",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_risk_distribution_chart(self, companies_data):
        """Create risk distribution visualization"""
        
        # Categorize risk levels
        companies_data['risk_category'] = companies_data['credit_score'].apply(
            lambda x: 'Low Risk (700+)' if x >= 700 else 
                     'Medium Risk (600-699)' if x >= 600 else 
                     'High Risk (<600)'
        )
        
        # Count by risk category
        risk_counts = companies_data['risk_category'].value_counts()
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Distribution",
            color_discrete_map={
                'Low Risk (700+)': '#2ca02c',
                'Medium Risk (600-699)': '#ff7f0e',
                'High Risk (<600)': '#d62728'
            }
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_sector_analysis_chart(self, companies_data):
        """Create sector-wise credit analysis"""
        
        sector_stats = companies_data.groupby('sector').agg({
            'credit_score': ['mean', 'std', 'count']
        }).round(2)
        
        sector_stats.columns = ['Average_Score', 'Volatility', 'Count']
        sector_stats = sector_stats.reset_index()
        
        # Create bubble chart
        fig = px.scatter(
            sector_stats,
            x='Average_Score',
            y='Volatility',
            size='Count',
            color='Average_Score',
            hover_name='sector',
            title="Sector Analysis: Average Score vs Volatility",
            labels={
                'Average_Score': 'Average Credit Score',
                'Volatility': 'Score Volatility (Std Dev)',
                'Count': 'Number of Companies'
            },
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_correlation_heatmap(self, financial_data, feature_subset=None):
        """Create correlation heatmap for financial metrics"""
        
        if feature_subset:
            corr_data = financial_data[feature_subset].corr()
        else:
            # Select subset of key financial metrics
            key_metrics = [
                'current_ratio', 'debt_to_equity', 'roe', 'roa', 
                'net_margin', 'asset_turnover', 'revenue_growth',
                'ebitda_margin', 'interest_coverage', 'pe_ratio'
            ]
            available_metrics = [col for col in key_metrics if col in financial_data.columns]
            corr_data = financial_data[available_metrics].corr()
        
        fig = px.imshow(
            corr_data,
            title="Financial Metrics Correlation",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_peer_comparison_chart(self, company_scores, company_name):
        """Create peer comparison visualization"""
        
        # Highlight selected company
        colors = ['red' if name == company_name else 'blue' for name in company_scores.index]
        
        fig = go.Figure(data=[
            go.Bar(
                x=company_scores.index,
                y=company_scores.values,
                marker_color=colors,
                text=company_scores.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Peer Comparison - {company_name}",
            xaxis_title="Companies",
            yaxis_title="Credit Score",
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_alert_timeline(self, alerts_data):
        """Create timeline visualization for alerts"""
        
        # Convert timestamp to datetime
        alerts_data['datetime'] = pd.to_datetime(alerts_data['timestamp'])
        
        # Color mapping for severity
        color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        
        fig = px.scatter(
            alerts_data,
            x='datetime',
            y='company',
            color='severity',
            color_discrete_map=color_map,
            title="Alert Timeline",
            hover_data=['title', 'message']
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Company"
        )
        
        return fig
    
    def create_model_performance_chart(self, performance_data):
        """Create model performance tracking chart"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Absolute Error', 'RMSE', 'R² Score', 'Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # MAE
        fig.add_trace(
            go.Scatter(x=performance_data['date'], y=performance_data['MAE'], 
                      name='MAE', line=dict(color='red')),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Scatter(x=performance_data['date'], y=performance_data['RMSE'], 
                      name='RMSE', line=dict(color='orange')),
            row=1, col=2
        )
        
        # R2
        fig.add_trace(
            go.Scatter(x=performance_data['date'], y=performance_data['R2'], 
                      name='R²', line=dict(color='green')),
            row=2, col=1
        )
        
        # Accuracy (mock)
        accuracy = np.random.uniform(0.8, 0.95, len(performance_data))
        fig.add_trace(
            go.Scatter(x=performance_data['date'], y=accuracy, 
                      name='Accuracy', line=dict(color='blue')),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Model Performance Metrics Over Time",
            showlegend=False
        )
        
        return fig
    
    def create_data_quality_dashboard(self, quality_data):
        """Create data quality monitoring dashboard"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quality Score by Source', 'Completeness', 'Accuracy', 'Timeliness'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Quality Score
        fig.add_trace(
            go.Bar(x=quality_data['source'], y=quality_data['quality_score'], 
                   name='Quality Score', marker_color='blue'),
            row=1, col=1
        )
        
        # Completeness
        fig.add_trace(
            go.Bar(x=quality_data['source'], y=quality_data['completeness'], 
                   name='Completeness', marker_color='green'),
            row=1, col=2
        )
        
        # Accuracy
        fig.add_trace(
            go.Bar(x=quality_data['source'], y=quality_data['accuracy'], 
                   name='Accuracy', marker_color='orange'),
            row=2, col=1
        )
        
        # Timeliness
        fig.add_trace(
            go.Bar(x=quality_data['source'], y=quality_data['timeliness'], 
                   name='Timeliness', marker_color='red'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Data Quality Metrics",
            showlegend=False
        )
        
        # Update y-axis to show percentages
        fig.update_yaxes(tickformat='.0%')
        
        return fig
    
    def display_company_card(self, company_data):
        """Display company information card"""
        
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader(company_data['company_name'])
                st.write(f"**Sector:** {company_data['sector']}")
                
            with col2:
                score = company_data['credit_score']
                st.metric("Credit Score", f"{score:.0f}")
                
            with col3:
                risk_level = 'Low Risk' if score >= 700 else 'Medium Risk' if score >= 600 else 'High Risk'
                color = 'green' if score >= 700 else 'orange' if score >= 600 else 'red'
                st.markdown(f"**Risk Level:** :{color}[{risk_level}]")
    
    def create_summary_statistics(self, data):
        """Create summary statistics table"""
        
        summary_stats = data.describe()
        
        # Format the statistics nicely
        formatted_stats = summary_stats.round(2)
        
        return formatted_stats
    
    def create_interactive_filter_panel(self):
        """Create interactive filter panel for dashboard"""
        
        with st.sidebar.expander("📊 Dashboard Filters", expanded=True):
            
            # Date range filter
            from datetime import date
            date_range = st.date_input(
                "Date Range",
                value=(date(2024, 1, 1), date.today()),
                help="Select date range for analysis"
            )
            
            # Score range filter
            score_range = st.slider(
                "Credit Score Range",
                min_value=300,
                max_value=850,
                value=(400, 800),
                step=10,
                help="Filter companies by credit score range"
            )
            
            # Sector filter
            sectors = ['All', 'Technology', 'Financial Services', 'Healthcare', 
                      'Energy', 'Manufacturing', 'Retail', 'Real Estate', 'Telecommunications']
            selected_sectors = st.multiselect(
                "Sectors",
                sectors,
                default=['All'],
                help="Select sectors to include in analysis"
            )
            
            # Risk level filter
            risk_levels = st.multiselect(
                "Risk Levels",
                ['Low Risk', 'Medium Risk', 'High Risk'],
                default=['Low Risk', 'Medium Risk', 'High Risk'],
                help="Filter by risk categories"
            )
            
            return {
                'date_range': date_range,
                'score_range': score_range,
                'sectors': selected_sectors,
                'risk_levels': risk_levels
            }
