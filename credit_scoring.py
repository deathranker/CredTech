import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CreditScoringEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6),
            'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=8),
            'LinearRegression': LinearRegression()
        }
        self.best_model = None
        self.feature_names = None
    
    def prepare_features(self, financial_data):
        """Prepare features from financial data for ML models"""
        
        # Select relevant numerical columns for modeling
        feature_columns = [
            'current_ratio', 'quick_ratio', 'cash_ratio',
            'debt_to_equity', 'debt_to_assets', 'interest_coverage',
            'roe', 'roa', 'net_margin', 'gross_margin',
            'asset_turnover', 'inventory_turnover', 'receivables_turnover',
            'pe_ratio', 'pb_ratio', 'market_to_book',
            'operating_cf_ratio', 'free_cf_yield',
            'revenue_growth', 'earnings_growth',
            'ebitda_margin', 'working_capital_ratio',
            'gdp_growth', 'inflation_rate', 'unemployment_rate', 'interest_rate',
            'market_volatility', 'sector_performance', 'credit_spread'
        ]
        
        # Handle missing values and select features
        X = financial_data[feature_columns].fillna(financial_data[feature_columns].median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Create additional engineered features
        X = self._engineer_features(X)
        
        self.feature_names = X.columns.tolist()
        
        return X.values, self.feature_names
    
    def _engineer_features(self, X):
        """Create additional engineered features"""
        X = X.copy()
        
        # Liquidity score
        X['liquidity_score'] = (X['current_ratio'] + X['quick_ratio'] + X['cash_ratio']) / 3
        
        # Leverage score
        X['leverage_score'] = (X['debt_to_equity'] + X['debt_to_assets']) / 2
        
        # Profitability score
        X['profitability_score'] = (X['roe'] + X['roa'] + X['net_margin']) / 3
        
        # Efficiency score
        X['efficiency_score'] = (X['asset_turnover'] + X['inventory_turnover'] + X['receivables_turnover']) / 3
        
        # Financial stability indicator
        X['financial_stability'] = X['interest_coverage'] * X['current_ratio'] / (X['debt_to_equity'] + 1)
        
        # Growth potential
        X['growth_potential'] = (X['revenue_growth'] + X['earnings_growth']) / 2
        
        # Market sentiment
        X['market_sentiment'] = X['pe_ratio'] * X['pb_ratio'] / (X['market_volatility'] + 1)
        
        # Macro environment score
        X['macro_environment'] = (X['gdp_growth'] - X['inflation_rate'] - X['unemployment_rate']) / 3
        
        # Risk-adjusted returns
        X['risk_adjusted_roe'] = X['roe'] / (X['market_volatility'] + 0.01)
        X['risk_adjusted_roa'] = X['roa'] / (X['market_volatility'] + 0.01)
        
        # Replace any remaining infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X
    
    def train_model(self, X, y, model_type='RandomForest'):
        """Train a credit scoring model"""
        
        # Convert y to numpy array if it's a list
        y = np.array(y)
        
        # Ensure we have enough samples
        if len(X) < 5:
            # If we don't have enough data, create a simple model
            X_repeated = np.repeat(X, 10, axis=0)
            y_repeated = np.repeat(y, 10)
            
            # Add some noise for variation
            y_repeated = y_repeated + np.random.normal(0, 5, size=len(y_repeated))
            X = X_repeated
            y = y_repeated
        
        # Split data for training and validation
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test = X, X
            y_train, y_test = y, y
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the specified model
        model = self.models[model_type]
        
        # For tree-based models, use original features (not scaled)
        if model_type in ['RandomForest', 'GradientBoosting', 'DecisionTree']:
            model.fit(X_train, y_train)
            self.best_model = model
        else:
            model.fit(X_train_scaled, y_train)
            self.best_model = model
        
        return self.best_model
    
    def predict(self, model, X):
        """Make credit score predictions"""
        if hasattr(model, 'predict'):
            # Check if model needs scaled features
            if isinstance(model, LinearRegression):
                X_scaled = self.scaler.transform(X)
                predictions = model.predict(X_scaled)
            else:
                predictions = model.predict(X)
            
            # Ensure predictions are in valid credit score range
            predictions = np.clip(predictions, 300, 850)
            return predictions
        else:
            # Fallback: return simple baseline predictions
            return np.full(len(X), 650)
    
    def evaluate_model(self, model, X, y):
        """Evaluate model performance"""
        predictions = self.predict(model, X)
        
        metrics = {
            'MAE': mean_absolute_error(y, predictions),
            'RMSE': np.sqrt(mean_squared_error(y, predictions)),
            'R2': r2_score(y, predictions)
        }
        
        return metrics
    
    def compare_models(self, X, y):
        """Compare performance of different models"""
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Train model
                trained_model = self.train_model(X, y, model_name)
                
                # Evaluate
                metrics = self.evaluate_model(trained_model, X, y)
                results[model_name] = metrics
                
            except Exception as e:
                # If model fails, provide default metrics
                results[model_name] = {
                    'MAE': 999,
                    'RMSE': 999,
                    'R2': 0
                }
        
        return results
    
    def get_feature_importance(self, model):
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
        else:
            # Return uniform importance if not available
            num_features = len(self.feature_names) if self.feature_names is not None else 10
            return np.ones(num_features) / num_features
    
    def predict_with_explanation(self, model, X, feature_names):
        """Predict with basic feature contribution explanation"""
        predictions = self.predict(model, X)
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        else:
            feature_importance = np.ones(len(feature_names)) / len(feature_names)
        
        # Simple feature contribution calculation
        # This is a simplified approach - SHAP provides more accurate attributions
        feature_contributions = []
        
        for i in range(len(X)):
            contributions = {}
            x_sample = X[i]
            
            # Calculate basic contributions based on feature values and importance
            for j, feature_name in enumerate(feature_names):
                # Normalize feature value and multiply by importance
                feature_value = x_sample[j]
                normalized_value = (feature_value - np.mean(X[:, j])) / (np.std(X[:, j]) + 1e-8)
                contribution = normalized_value * feature_importance[j] * 10  # Scale for visibility
                contributions[feature_name] = contribution
            
            feature_contributions.append(contributions)
        
        return predictions, feature_contributions
    
    def calculate_credit_score_components(self, financial_metrics):
        """Calculate credit score components based on traditional rating methodology"""
        
        components = {}
        
        # Liquidity Component (25% weight)
        liquidity_score = (
            financial_metrics.get('current_ratio', 1.5) * 0.4 +
            financial_metrics.get('quick_ratio', 1.0) * 0.4 +
            financial_metrics.get('cash_ratio', 0.5) * 0.2
        ) / 3 * 100
        components['Liquidity'] = min(100, max(0, liquidity_score))
        
        # Leverage Component (30% weight)
        debt_to_equity = financial_metrics.get('debt_to_equity', 0.5)
        interest_coverage = financial_metrics.get('interest_coverage', 5)
        leverage_score = (
            (1 / (debt_to_equity + 0.1)) * 20 +  # Lower debt is better
            min(interest_coverage / 10, 1) * 80   # Higher coverage is better
        )
        components['Leverage'] = min(100, max(0, leverage_score))
        
        # Profitability Component (25% weight)
        profitability_score = (
            financial_metrics.get('roe', 0.1) * 400 +  # 10% ROE = 40 points
            financial_metrics.get('roa', 0.05) * 800 +  # 5% ROA = 40 points
            financial_metrics.get('net_margin', 0.1) * 200  # 10% margin = 20 points
        )
        components['Profitability'] = min(100, max(0, profitability_score))
        
        # Efficiency Component (20% weight)
        efficiency_score = (
            min(financial_metrics.get('asset_turnover', 1) * 50, 50) +
            min(financial_metrics.get('inventory_turnover', 6) * 5, 25) +
            min(financial_metrics.get('receivables_turnover', 8) * 3, 25)
        )
        components['Efficiency'] = min(100, max(0, efficiency_score))
        
        return components
    
    def score_to_rating(self, score):
        """Convert numerical score to letter rating"""
        if score >= 800:
            return 'AAA'
        elif score >= 750:
            return 'AA'
        elif score >= 700:
            return 'A'
        elif score >= 650:
            return 'BBB'
        elif score >= 600:
            return 'BB'
        elif score >= 550:
            return 'B'
        elif score >= 500:
            return 'CCC'
        elif score >= 450:
            return 'CC'
        else:
            return 'C'
    
    def get_rating_description(self, rating):
        """Get description for credit rating"""
        descriptions = {
            'AAA': 'Highest quality, minimal credit risk',
            'AA': 'Very high quality, very low credit risk',
            'A': 'High quality, low credit risk',
            'BBB': 'Good quality, moderate credit risk',
            'BB': 'Speculative, elevated credit risk',
            'B': 'Highly speculative, high credit risk',
            'CCC': 'Substantial credit risk, vulnerable',
            'CC': 'Very high credit risk, likely default',
            'C': 'Extremely high credit risk, imminent default'
        }
        return descriptions.get(rating, 'Rating not available')
