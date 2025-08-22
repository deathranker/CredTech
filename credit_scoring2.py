import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def credit_score_model(data: pd.DataFrame):
    """
    Simple scoring engine:
    - Uses price volatility and recent returns as features.
    - Outputs a score between 300 and 850.
    """

    if data is None or data.empty:
        return 500, "Fallback"

    # Feature Engineering
    data["Returns"] = data["Close"].pct_change()
    avg_return = data["Returns"].mean()
    volatility = data["Returns"].std()

    # Mock model (Logistic regression style logic)
    features = np.array([[avg_return, volatility]])
    # Normalize into a credit score range (300-850)
    score = int(700 + avg_return * 1000 - volatility * 200)

    # Clamp score to realistic range
    score = max(300, min(score, 850))

    return score, "Simple Volatility-Return Model"
