import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

class DataGenerator:
    def __init__(self):
        self.fake = Faker()
        np.random.seed(42)  # For reproducible results
        random.seed(42)
        
        # Company data
        self.sectors = ['Technology', 'Financial Services', 'Healthcare', 'Energy', 
                       'Manufacturing', 'Retail', 'Real Estate', 'Telecommunications']
        
        self.company_names = [
            'TechGlobal Corp', 'Financial Solutions Inc', 'HealthCare Dynamics', 
            'Energy Innovations LLC', 'Manufacturing Excellence', 'Retail Giants Co',
            'PropTech Ventures', 'Telecom Networks Ltd', 'DataSoft Solutions',
            'Investment Partners Group'
        ]
    
    def generate_companies_data(self, n_companies=10):
        """Generate sample company data with credit scores"""
        companies = []
        
        for i in range(min(n_companies, len(self.company_names))):
            company = {
                'company_id': i + 1,
                'company_name': self.company_names[i],
                'sector': random.choice(self.sectors),
                'credit_score': np.random.normal(650, 80),  # Mean 650, std 80
                'market_cap': np.random.lognormal(15, 1.5),  # Log-normal distribution
                'employees': np.random.randint(100, 50000),
                'founded_year': np.random.randint(1950, 2020),
                'last_updated': datetime.now()
            }
            
            # Ensure credit score is in reasonable range
            company['credit_score'] = max(300, min(850, company['credit_score']))
            companies.append(company)
        
        return pd.DataFrame(companies)
    
    def generate_financial_metrics(self, company_name):
        """Generate comprehensive financial metrics for a company"""
        n_periods = 12  # 12 quarters of data
        
        data = []
        base_date = datetime.now() - timedelta(days=90 * n_periods)
        
        for i in range(n_periods):
            # Base financial ratios with some trend and noise
            trend_factor = 1 + (i * 0.02)  # Slight upward trend
            noise = np.random.normal(1, 0.1)
            
            metrics = {
                'company_name': company_name,
                'date': base_date + timedelta(days=90 * i),
                'quarter': f"Q{(i % 4) + 1}",
                'year': (base_date + timedelta(days=90 * i)).year,
                
                # Liquidity Ratios
                'current_ratio': max(0.5, np.random.normal(2.1, 0.3) * trend_factor * noise),
                'quick_ratio': max(0.3, np.random.normal(1.5, 0.25) * trend_factor * noise),
                'cash_ratio': max(0.1, np.random.normal(0.8, 0.15) * trend_factor * noise),
                
                # Leverage Ratios
                'debt_to_equity': max(0.1, np.random.normal(0.4, 0.15) * noise),
                'debt_to_assets': max(0.1, np.random.normal(0.3, 0.1) * noise),
                'interest_coverage': max(1, np.random.normal(8, 2) * trend_factor * noise),
                
                # Profitability Ratios
                'roe': np.random.normal(0.12, 0.05) * trend_factor * noise,
                'roa': np.random.normal(0.08, 0.03) * trend_factor * noise,
                'net_margin': np.random.normal(0.1, 0.05) * trend_factor * noise,
                'gross_margin': max(0.1, np.random.normal(0.35, 0.1) * noise),
                
                # Efficiency Ratios
                'asset_turnover': max(0.5, np.random.normal(1.2, 0.3) * noise),
                'inventory_turnover': max(2, np.random.normal(6, 1.5) * noise),
                'receivables_turnover': max(4, np.random.normal(8, 2) * noise),
                
                # Market Ratios
                'pe_ratio': max(5, np.random.normal(18, 5) * noise),
                'pb_ratio': max(0.5, np.random.normal(2.5, 0.8) * noise),
                'market_to_book': max(0.8, np.random.normal(1.8, 0.5) * noise),
                
                # Cash Flow Ratios
                'operating_cf_ratio': np.random.normal(0.15, 0.05) * trend_factor * noise,
                'free_cf_yield': np.random.normal(0.08, 0.03) * trend_factor * noise,
                
                # Growth Ratios
                'revenue_growth': np.random.normal(0.05, 0.1) * trend_factor,
                'earnings_growth': np.random.normal(0.08, 0.15) * trend_factor,
                
                # Industry Specific Metrics
                'ebitda_margin': max(0.05, np.random.normal(0.2, 0.08) * trend_factor * noise),
                'working_capital_ratio': np.random.normal(0.15, 0.05) * noise,
                
                # Macroeconomic Factors
                'gdp_growth': np.random.normal(0.025, 0.01),
                'inflation_rate': np.random.normal(0.03, 0.01),
                'unemployment_rate': np.random.normal(0.05, 0.02),
                'interest_rate': np.random.normal(0.04, 0.015),
                
                # Market Conditions
                'market_volatility': max(0.1, np.random.normal(0.2, 0.05)),
                'sector_performance': np.random.normal(0.08, 0.12),
                'credit_spread': max(0.01, np.random.normal(0.03, 0.01))
            }
            
            data.append(metrics)
        
        return pd.DataFrame(data)
    
    def generate_market_data(self):
        """Generate market and economic indicator data"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        market_data = []
        
        for date in dates:
            data_point = {
                'date': date,
                'sp500_index': 4200 + np.random.normal(0, 50),
                'vix_index': max(10, np.random.normal(20, 5)),
                'treasury_10y': max(1, np.random.normal(4.5, 0.3)),
                'treasury_2y': max(0.5, np.random.normal(4.2, 0.4)),
                'credit_spreads_ig': max(0.5, np.random.normal(1.2, 0.2)),
                'credit_spreads_hy': max(2, np.random.normal(4.5, 0.8)),
                'dollar_index': 100 + np.random.normal(0, 2),
                'gold_price': 2000 + np.random.normal(0, 30),
                'oil_price': 80 + np.random.normal(0, 5),
                'gdp_nowcast': np.random.normal(2.5, 0.5),
                'inflation_expectation': max(1, np.random.normal(3.2, 0.3))
            }
            market_data.append(data_point)
        
        return pd.DataFrame(market_data)
    
    def generate_news_sentiment(self, company_name):
        """Generate news headlines and sentiment scores"""
        headlines_templates = [
            f"{company_name} reports strong quarterly earnings",
            f"{company_name} announces major acquisition deal",
            f"{company_name} faces regulatory challenges",
            f"{company_name} launches innovative product line",
            f"Analysts upgrade {company_name} stock rating",
            f"{company_name} CEO announces strategic restructuring",
            f"{company_name} expands operations to new markets",
            f"Investment firm increases stake in {company_name}",
            f"{company_name} faces supply chain disruptions",
            f"{company_name} reports better than expected revenue"
        ]
        
        news_data = []
        base_date = datetime.now() - timedelta(days=7)
        
        for i in range(10):
            headline = random.choice(headlines_templates)
            
            # Generate sentiment based on headline keywords
            positive_keywords = ['strong', 'announces', 'innovative', 'upgrade', 'expands', 'better', 'expected']
            negative_keywords = ['challenges', 'faces', 'disruptions', 'restructuring']
            
            sentiment_score = 0
            for word in positive_keywords:
                if word in headline.lower():
                    sentiment_score += np.random.uniform(0.1, 0.3)
            
            for word in negative_keywords:
                if word in headline.lower():
                    sentiment_score -= np.random.uniform(0.1, 0.3)
            
            # Add some random noise
            sentiment_score += np.random.normal(0, 0.1)
            sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
            
            news_item = {
                'headline': headline,
                'sentiment': sentiment_score,
                'date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                'source': random.choice(['Reuters', 'Bloomberg', 'Financial Times', 'WSJ', 'MarketWatch'])
            }
            news_data.append(news_item)
        
        return pd.DataFrame(news_data)
    
    def generate_historical_scores(self, company_name):
        """Generate historical credit score data"""
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        
        # Generate base score with trend and noise
        base_score = 650
        trend = np.random.uniform(-0.1, 0.1)  # Daily trend
        
        scores = []
        current_score = base_score
        
        for i, date in enumerate(dates):
            # Add trend and random walk
            current_score += trend + np.random.normal(0, 2)
            current_score = max(300, min(850, current_score))  # Clamp to valid range
            
            # Add some volatility events
            if np.random.random() < 0.05:  # 5% chance of significant event
                current_score += np.random.uniform(-20, 20)
                current_score = max(300, min(850, current_score))
            
            scores.append({
                'date': date,
                'credit_score': current_score,
                'company_name': company_name
            })
        
        return pd.DataFrame(scores)
    
    def generate_data_quality_metrics(self):
        """Generate data quality metrics for different sources"""
        sources = ['Financial Ratios', 'Market Data', 'News Sentiment', 'SEC Filings', 'Economic Indicators']
        
        quality_data = []
        for source in sources:
            quality_score = np.random.uniform(0.7, 1.0)  # Quality score between 70-100%
            quality_data.append({
                'source': source,
                'quality_score': quality_score,
                'completeness': np.random.uniform(0.85, 1.0),
                'accuracy': np.random.uniform(0.9, 1.0),
                'timeliness': np.random.uniform(0.8, 1.0)
            })
        
        return pd.DataFrame(quality_data)
    
    def generate_throughput_data(self):
        """Generate data pipeline throughput metrics"""
        timestamps = pd.date_range(end=datetime.now(), periods=24, freq='H')
        
        throughput_data = []
        for timestamp in timestamps:
            # Simulate varying load throughout the day
            hour = timestamp.hour
            base_load = 100 + 50 * np.sin(2 * np.pi * hour / 24)  # Sinusoidal pattern
            actual_load = max(10, base_load + np.random.normal(0, 20))
            
            throughput_data.append({
                'timestamp': timestamp,
                'records_per_minute': actual_load
            })
        
        return pd.DataFrame(throughput_data)
    
    def generate_alerts(self):
        """Generate sample alert data"""
        companies = self.company_names[:5]  # Use first 5 companies
        
        alert_types = [
            'Credit score dropped significantly',
            'Unusual market activity detected',
            'Negative news sentiment spike',
            'Financial ratio deterioration',
            'Peer comparison outlier'
        ]
        
        alerts = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(8):
            alert = {
                'id': i + 1,
                'company': random.choice(companies),
                'title': random.choice(alert_types),
                'message': f"Alert triggered based on automated monitoring. Immediate review recommended.",
                'severity': random.choice(['High', 'Medium', 'Low']),
                'timestamp': (base_time + timedelta(hours=i * 3)).strftime('%Y-%m-%d %H:%M'),
                'status': random.choice(['Active', 'Acknowledged', 'Resolved'])
            }
            alerts.append(alert)
        
        return pd.DataFrame(alerts)
    
    def generate_alert_trends(self):
        """Generate alert trend data"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        severities = ['High', 'Medium', 'Low']
        
        trend_data = []
        for date in dates:
            for severity in severities:
                count = np.random.poisson(3 if severity == 'Low' else 2 if severity == 'Medium' else 1)
                trend_data.append({
                    'date': date,
                    'severity': severity,
                    'alert_count': count
                })
        
        return pd.DataFrame(trend_data)
    
    def generate_model_performance_history(self):
        """Generate historical model performance data"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        performance_data = []
        base_mae = 15
        base_rmse = 20
        base_r2 = 0.85
        
        for i, date in enumerate(dates):
            # Add some variation over time
            mae = base_mae + np.random.normal(0, 2) + 0.1 * np.sin(2 * np.pi * i / 30)
            rmse = base_rmse + np.random.normal(0, 3) + 0.2 * np.sin(2 * np.pi * i / 30)
            r2 = base_r2 + np.random.normal(0, 0.02) + 0.01 * np.sin(2 * np.pi * i / 30)
            
            performance_data.append({
                'date': date,
                'MAE': max(5, mae),
                'RMSE': max(8, rmse),
                'R2': max(0.6, min(0.95, r2))
            })
        
        return pd.DataFrame(performance_data)
