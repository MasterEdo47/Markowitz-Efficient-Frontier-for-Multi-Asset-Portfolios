# === 1. IMPORT LIBRARIES ===
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import plotly.graph_objs as go

# === 2. SETTINGS: TIME RANGE & TICKERS ===
years = 15
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365 * years)
tickers = ['SPY', 'BND', 'GLD', 'BTC']

# === 3. DOWNLOAD ADJUSTED CLOSE PRICES ===
adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    adj_close_df[ticker] = data['Close']  #Already adjusted if auto_adjust=True

# === 4. CALCULATE DAILY LOG RETURNS ===
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

# === 5. METRIC FUNCTIONS ===

# Annualized expected portfolio return
def portfolio_return(weights):
    return np.dot(log_returns.mean(), weights) * 252

# Annualized portfolio volatility (standard deviation)
def portfolio_stdev(weights):
    cov_matrix = log_returns.cov()
    portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
    return np.sqrt(portfolio_var) * np.sqrt(252)

# Generate random weights that sum to 1
def generate_weights(n_assets):
    weights = np.random.random(n_assets)
    return weights / np.sum(weights)

# === 6. SIMULATE RANDOM PORTFOLIOS ===
num_portfolios = 10000
returns = []
volatilities = []
weights_list = []

for _ in range(num_portfolios):
    weights = generate_weights(len(tickers))
    weights_list.append(weights)
    returns.append(portfolio_return(weights))
    volatilities.append(portfolio_stdev(weights))

returns = np.array(returns)
volatilities = np.array(volatilities)
weights_list = np.array(weights_list)

# === 7. BUILD HOVER TEXT TO DISPLAY WEIGHTS ===
hover_text = []
for i in range(len(weights_list)):
    weight_info = [f"{tickers[j]}: {round(weights_list[i][j]*100, 2)}%" for j in range(len(tickers))]
    hover_text.append("<br>".join(weight_info))

# === 8. PLOT INTERACTIVE EFFICIENT FRONTIER ===
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=volatilities,
    y=returns,
    mode='markers',
    marker=dict(
        color=returns / volatilities,  # Sharpe Ratio (risk-free = 0)
        colorscale='Viridis',
        size=6,
        opacity=0.8,
        showscale=True,
        colorbar=dict(title='Ratio')
    ),
    text=hover_text,
    hovertemplate=
        'Volatility: %{x:.2%}<br>' +
        'Return: %{y:.2%}<br><br>' +
        '%{text}<extra></extra>',
    name='Portfolios'
))

fig.update_layout(
    title='Efficient Frontier (Annualized Return & Volatility)',
    xaxis_title='Volatility (Annualized Std Dev)',
    yaxis_title='Expected Return (Annualized)',
    template='plotly_white',
    width=900,
    height=600
)

max_sharpe_idx = (returns / volatilities).argmax()
fig.add_trace(go.Scatter(
    x=[volatilities[max_sharpe_idx]],
    y=[returns[max_sharpe_idx]],
    mode='markers+text',
    marker=dict(color='red', size=10, symbol='x'),
    text=["Max Sharpe"],
    textposition='top left',  # ← Moves the label away from the marker
    textfont=dict(size=12, color='red'),
    name='Max Sharpe'
))
