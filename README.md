# Stat Arb with Pairs Trading 

This project implements a generalized pairs trading strategy in Python, expanding upon previous versions that were limited to trading only two stocks simultaneously.  This enhanced algorithm can identify and trade multiple cointegrated pairs concurrently, significantly increasing potential trading opportunities.

## Key Improvements & Features:

* **Multiple Pair Trading:**  The algorithm now identifies and manages trades across multiple cointegrated pairs simultaneously, rather than being restricted to a single pair.  This allows for greater diversification and the potential to capture more market inefficiencies.
* **Dynamic Pair Selection:** Cointegrated pairs are dynamically selected based on statistical tests (Augmented Dickey-Fuller test) and sorted by p-value, ensuring that only the most statistically significant pairs are traded.  The top 24 pairs (by p-value) are selected for trading.
* **Bollinger Bands & RSI:** The trading logic uses Bollinger Bands to identify entry and exit points based on spread deviations from the moving average.  The Relative Strength Index (RSI) is incorporated to confirm overbought/oversold conditions, enhancing the trading signals.
* **Risk Management:**  The strategy includes basic risk management parameters like a defined trading window, standard deviation multiplier for Bollinger Bands, and an RSI threshold to filter out weak signals.  A transaction cost rate is also included for realistic backtesting.
* **Performance Analytics:**  The backtesting module computes key performance metrics including total PNL, Sharpe Ratio, and Maximum Drawdown, providing a comprehensive evaluation of the strategy's performance.
* **Improved Data Handling:**  The code now handles potential data errors more robustly, including checks for missing data, NaN values, and insufficient data points for specific tickers.  It also downloads data for all S&P 500 Information Technology sector stocks and filters out those with insufficient data.

## How it works:

1. **Data Download & Preprocessing:** Downloads historical price data for S&P 500 Information Technology stocks from Yahoo Finance.  Filters and prepares the data for analysis.

2. **Cointegration Test:** Identifies cointegrated pairs using the Augmented Dickey-Fuller test on the residuals of a linear regression between the log returns of each pair of stocks.

3. **Spread Calculation & Indicator Generation:**  Calculates the spread between each cointegrated pair and generates Bollinger Bands and RSI values.

4. **Trading Logic:**  Generates buy/sell signals based on the spread crossing the Bollinger Bands and RSI thresholds.

5. **Backtesting & Performance Analysis:** Simulates trades based on the generated signals, calculates daily and cumulative PNL, and computes performance metrics.

## Getting Started:

1. **Installation:** Ensure you have the required libraries installed: `yfinance`, `pandas`, `numpy`, `statsmodels`, `itertools`. 
2. **Configuration:** Adjust parameters in the `PairTradingStrategy` class constructor, such as `start`, `end`, `window`, `sd`, `rsi_threshold`, `transaction_cost_rate`, `shares_per_position`.
3. **Execution:**  Run the `Generalized_ALGO.py` script. The output will include a list of cointegrated pairs, backtesting results, and performance analytics.

## Future Enhancements:

* **Advanced Risk Management:** Implement more sophisticated risk management techniques, such as stop-loss orders and position sizing based on volatility.
* **Dynamic Parameter Optimization:** Explore methods for dynamically adjusting parameters like the trading window and standard deviation multiplier based on market conditions.
* **Alternative Data Sources:** Integrate alternative data sources to potentially improve cointegration identification and trading signals.

## Disclaimer:

This code is for educational and research purposes only.  It should not be considered financial advice.  Use at your own risk.

