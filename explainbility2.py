import pandas as pd

def explain_score(data: pd.DataFrame):
    """
    Generate a plain-language explanation of the score
    based on market volatility and returns.
    """

    if data is None or data.empty:
        return "No data available for explanation."

    data["Returns"] = data["Close"].pct_change()
    avg_return = round(data["Returns"].mean() * 100, 2)
    volatility = round(data["Returns"].std() * 100, 2)

    explanation = f"""
    ### Why This Score?
    - *Average Daily Return:* {avg_return}%  
    - *Volatility:* {volatility}%  

    *Interpretation:*
    - Higher returns increase the creditworthiness score.  
    - Higher volatility reduces the creditworthiness score (risk factor).  
    - Recent market behavior suggests that credit risk is being influenced by these two factors.  

    Note: Real deployment can include additional signals like balance sheet filings, 
    macroeconomic indicators, and financial news sentiment.
    """

    return explanation
