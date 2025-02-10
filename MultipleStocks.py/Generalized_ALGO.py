import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as stat
from itertools import combinations
import plotly.graph_objects as go


class PairTradingStrategy:
    def __init__(self, tickers, start, end, window=24, sd=2, rsi_threshold=40, transaction_cost_rate=0.0025, shares_per_position=100):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.window = window
        self.sd = sd
        self.rsi_threshold = rsi_threshold
        self.transaction_cost_rate = transaction_cost_rate
        self.shares_per_position = shares_per_position

        # Initialize dataframes
        self.data = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.df_stockA_log = pd.DataFrame(columns=['Time', 'Signal', 'Entry A', 'Exit A', 'Stock A PNL'])
        self.df_stockB_log = pd.DataFrame(columns=['Time', 'Signal', 'Entry B', 'Exit B', 'Stock B PNL'])
        

        #Internal parameters initialized 
        self.position = 0
        self.long = 0
        self.short = 0
        self.beta = None  # Store beta coefficient
   
    def download_data(self):
        try:
            # 1. Get S&P 500 tickers with sector information
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            it_tickers = table[table['GICS Sector'] == 'Information Technology']['Symbol'].tolist()
            
            # 2. Download all ticker data first
            prices = yf.download(it_tickers, start=self.start, end=self.end)['Close']
            
            # 3. Filter out tickers with insufficient or missing data***
            min_data_points = 251 # Set your desired threshold (adjust as needed)
            valid_tickers = []
            for ticker in it_tickers:
                if ticker in prices.columns and prices[ticker].notna().sum() >= min_data_points: # Check for existence and minimum data points
                    valid_tickers.append(ticker)
                else:
                    print(f"Ticker: {ticker}, Data Points: {prices[ticker].notna().sum() if ticker in prices.columns else 'Not Found'} - Removed (Not Enough Data).")

            prices = prices[valid_tickers] # Keep only valid tickers

            # 4. Prepare data
            prices.index = prices.index.date # Convert datetime index to just the date part
            self.data = prices.reset_index().rename(columns={'index': 'time'})
            self.tickers = valid_tickers  # Update the class attribute to include only downloaded tickers

        except Exception as e:
            print(f"Error downloading or processing ticker data: {e}")
            self.tickers = []
            self.data = pd.DataFrame()
    
    #calculating the percentage change in prices for the selected stocks aka daily returns
    def calculate_returns(self):
        try:  # Handle potential errors during percentage change calculation
            self.returns = self.data.copy()  # Create a copy to avoid modifying original data
            for ticker in self.tickers: #Iterate through all valid tickers
                self.returns[f'{ticker}_returns'] = self.returns.set_index('time')[ticker].pct_change().values #Compute percentage change for prices of each ticker and reassign using .values to avoid index alignment issues
            self.returns['time'] = pd.to_datetime(self.returns['time']) #Convert to DateTimeIndex
            self.returns = self.returns.set_index('time').dropna().reset_index() # Drop rows with NaN values after pct_change
        except KeyError as e:
            print(f"Error calculating returns: {e}") #Print error message if keyerror occurs such as when no tickers were found
            self.returns = pd.DataFrame()  # Set to empty DataFrame if error

    def bollinger_bands(self, spread_series, pair_name): #Modified to accept spread_series and pair_name
        # Create a DataFrame for this pair's Bollinger Bands
        bb_df = pd.DataFrame(index=spread_series.index)
        bb_df['SMA'] = spread_series.rolling(window=self.window).mean()
        bb_df['Upper'] = bb_df['SMA'] + (self.sd * spread_series.rolling(window=self.window).std()) #using spread_series for calculation. spread_cl is no longer an input
        bb_df['Lower'] = bb_df['SMA'] - (self.sd * spread_series.rolling(window=self.window).std())
        return bb_df  # Return the DataFrame
    
    def compute_rsi(self, spread_series, pair_name):
        # Create a new DataFrame for RSI
        rsi_df = pd.DataFrame(index=spread_series.index) # Initialize empty dataframe for RSI values with correct index
        delta = spread_series.diff()  # Calculate the difference for spread_series
        gain = delta.where(delta > 0, 0)  
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(self.window).mean()
        avg_loss = loss.rolling(self.window).mean()
        rsi_df[f'Spread_RSI_{pair_name}'] = 100 * avg_gain / (avg_loss + avg_gain)
        return rsi_df  # Return the RSI DataFrame
    
    def find_cointegrated_pairs(self):
        cointegrated_pairs = []
        for ticker1, ticker2 in combinations(self.tickers, 2):
            try:
                # 1. Ensure both tickers have data before proceeding
                if ticker1 not in self.returns.columns or ticker2 not in self.returns.columns:
                    print(f"Skipping pair {ticker1}_{ticker2} due to missing data.")
                    continue  # Skip to the next pair

                if self.returns[ticker1].isnull().all() or self.returns[ticker2].isnull().all(): #Check if any columns are entirely NaN before proceeding
                    print(f"Skipping pair {ticker1}_{ticker2} due to all NaN values in a column.")
                    continue

                # Correct usage of .loc[] for label-based indexing
                OLS_method = stat.OLS(self.returns.loc[:, ticker1], stat.add_constant(self.returns.loc[:, ticker2])).fit()  
                
                # 2. Check for NaN or empty residuals before ADF test
                residuals = OLS_method.resid
                if residuals.isnull().all() or len(residuals) == 0:
                    print(f"Skipping pair {ticker1}_{ticker2} due to NaN or empty residuals.")
                    continue
                
                
                adf_test = adfuller(residuals)
                p_value = adf_test[1]

                if adf_test[0] <= adf_test[4]['10%'] and p_value <= 0.1:    
                    cointegrated_pairs.append((ticker1, ticker2, OLS_method.params.iloc[1], p_value))
            
            except (KeyError, ValueError, TypeError) as e:
                print(f"Error during cointegration test (pair {ticker1}_{ticker2}): {e}")
                continue

        # Sort by spread volatility (ascending), then p-value (ascending)
        cointegrated_pairs.sort(key=lambda x: x[3]) #order based on p-value
        return cointegrated_pairs[:24]
    
    def trade_strategy(self, spread_series, bb_df, rsi_df, pair_name):  # Accepts DataFrames
        #Shift values as before but use incoming series which has already been indexed
        prev_spread = spread_series.shift(1)
        prev_lower = bb_df['Lower'].shift(1)
        prev_upper = bb_df['Upper'].shift(1)



        buy_condition = (
            (prev_spread > prev_lower) & (spread_series < bb_df['Lower']) & (rsi_df[f'Spread_RSI_{pair_name}'] < self.rsi_threshold)
        )
        sell_condition = (
            (prev_spread < prev_upper) & (spread_series > bb_df['Upper']) & (rsi_df[f'Spread_RSI_{pair_name}'] > (100 - self.rsi_threshold))
        )


        signal_series = pd.Series(0, index=spread_series.index)  #Same index as spread_series for alignment
        signal_series.loc[buy_condition] = 1
        signal_series.loc[sell_condition] = -1


        return signal_series

    def compute_cumulative_pnl(self, combined_pnl):
        self.cumulative_pnl += combined_pnl  # Update the instance variable directly
        return self.cumulative_pnl
    
    def plot_bb_spread_signals(self, pair_name):
        if (
            not self.returns.empty and
            f'spread_{pair_name}' in self.returns.columns and
            f'signal_{pair_name}' in self.returns.columns
        ):
            spread_series = self.returns[f'spread_{pair_name}']
            bb_df = self.bollinger_bands(spread_series, pair_name) #Use updated bollinger_bands method
            signals = self.returns[f'signal_{pair_name}']

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=self.returns['time'], y=spread_series, mode='lines', name='Spread', line=dict(color='blue'))) # Spread

            fig.add_trace(go.Scatter(x=self.returns['time'], y=bb_df['SMA'], mode='lines', name='SMA', line=dict(color='orange'))) # SMA
            fig.add_trace(go.Scatter(x=self.returns['time'], y=bb_df['Upper'], mode='lines', name='Upper Band', line=dict(color='green'))) # Upper Band
            fig.add_trace(go.Scatter(x=self.returns['time'], y=bb_df['Lower'], mode='lines', name='Lower Band', line=dict(color='red')))  # Lower Band

            #Signals
            buy_signals = signals[signals == 1]
            sell_signals = signals[signals == -1]


            fig.add_trace(go.Scatter(x=self.returns.loc[buy_signals.index, 'time'], y=spread_series[buy_signals.index], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green')))
            fig.add_trace(go.Scatter(x=self.returns.loc[sell_signals.index, 'time'], y=spread_series[sell_signals.index], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red')))

            fig.update_layout(title=f'Spread with Bollinger Bands and Signals for {pair_name}',
                              xaxis_title='Time',
                              yaxis_title='Spread')
            fig.show()

        else:
            print(f"Data not available to plot for {pair_name}. Check if backtest has run and 'spread', 'SMA', 'Upper', 'Lower', and 'signal' columns exist.")


    def backtest(self):
        cointegrated_pairs = self.find_cointegrated_pairs()
        slippage = 0
        self.cumulative_pnl = 0.0  # Initialize cumulative PNL *before* the loop


        # Get pair names *first* for pre-allocation
        pair_names = [f"{ticker1}_{ticker2}" for ticker1, ticker2, *_ in cointegrated_pairs]

        # Efficiently pre-allocate spread and signal columns
        spread_cols = [f'spread_{pair_name}' for pair_name in pair_names]
        signal_cols = [f'signal_{pair_name}' for pair_name in pair_names]

        # Initialize with datetime objects to avoid comparison issues
        datetime_index = pd.to_datetime(self.returns['time'])
        self.returns = pd.concat([self.returns.set_index('time'), pd.DataFrame(np.nan, index=datetime_index, columns=spread_cols + signal_cols)], axis=1).reset_index()

        # Initialize DataFrames *before* the loop
        
        self.df_pnl_log = pd.DataFrame({'Time': self.returns['time']}) #Initialize pnl_log with time before loop
        self.trade_log_df = pd.DataFrame(columns=['pair', 'position', 'combined_pnl', 'sum_pnl'])

        self.df_stockA_log = pd.DataFrame(columns=['Time', 'Signal', 'Entry A', 'Exit A', 'Stock A PNL'])  # Initialize empty
        self.df_stockB_log = pd.DataFrame(columns=['Time', 'Signal', 'Entry B', 'Exit B', 'Stock B PNL'])  # Initialize empty


        for ticker1, ticker2, beta, p_value in cointegrated_pairs:
            pair_name = f"{ticker1}_{ticker2}"
            daily_pnl = [] #Initialize list before to store each days pnl after all pair trades
            pnl_entries = []

            # Spread and Indicator Calculation
            spread_series = self.returns[f'{ticker1}_returns'] - beta * self.returns[f'{ticker2}_returns'] #Correctly calculates spread from the returns of each stock
            self.returns.loc[:, f'spread_{pair_name}'] = spread_series
            
            bb_df = self.bollinger_bands(spread_series, pair_name)
            rsi_df = self.compute_rsi(spread_series, pair_name)
            
            signals = self.trade_strategy(spread_series, bb_df, rsi_df, pair_name)
            self.returns.loc[:, f'signal_{pair_name}'] = signals
            
            self.position = 0

            #commented line below: to avoid generating 100 images
            #self.plot_bb_spread_signals(pair_name) #Plot after calculations.

            # Inner loop (trading logic)
            for i in range(1, len(self.returns)):
                signal = self.returns[f'signal_{pair_name}'].iloc[i]

                #Initialize before use
                transaction_cost = 0
                combined_pnl = 0
                cum_pnl = 0
                stockA_pnl = 0
                stockB_pnl = 0

                #open a long position (signal=1)
                if signal == 1 and self.position == 0:
                    self.entry_price_A = self.data[ticker1].iloc[i] * (1 + slippage) # Apply slippage to entry
                    self.entry_price_B = self.data[ticker2].iloc[i] * (1 + slippage) 
                    
                    self.position = 1 #Set position
                    transaction_cost = self.transaction_cost_rate * (self.entry_price_A + self.entry_price_B) * self.shares_per_position #Entry Cost
                
                #open a short position (signal=-1)
                elif signal == -1 and self.position == 0:
                    self.entry_price_A = self.data[ticker1].iloc[i] * (1 - slippage) # Slippage on short entry (subtract)
                    self.entry_price_B = self.data[ticker2].iloc[i] * (1 - slippage) 
                    self.position = -1 #Set position
                    transaction_cost = self.transaction_cost_rate * (self.entry_price_A + self.entry_price_B) * self.shares_per_position #Entry Cost


                #close a long position
                elif self.position == 1 and spread_series.iloc[i] >= bb_df['SMA'].iloc[i]:  # Use spread_series and bb_df['SMA']
                    self.exit_price_A = self.data[ticker1].iloc[i] * (1 - slippage)
                    self.exit_price_B = self.data[ticker2].iloc[i] * (1 - slippage)
                    stockA_pnl = (self.exit_price_A - self.entry_price_A) * self.shares_per_position
                    stockB_pnl = (self.entry_price_B - self.exit_price_B) * self.shares_per_position  # Correct calculation for Stock B PNL


                    combined_pnl = stockA_pnl + stockB_pnl  - transaction_cost       #Correct combined_pnl calculation
                    daily_pnl.append({'Time': self.returns['time'].iloc[i], f'Daily PNL_{pair_name}': combined_pnl}) #Append here, *after* combined_pnl is calculated

                    self.position = 0
                    transaction_cost = 0 #Reset transaction costs once the trade is done.

                    if combined_pnl != 0:  # ***Check and update logs *inside* the elif***
                        cum_pnl = self.compute_cumulative_pnl(combined_pnl)
                        new_trade_row = pd.DataFrame({  # Create and append to trade_log_df directly
                            # ... (same as before)
                        })
                        self.trade_log_df = pd.concat([self.trade_log_df, new_trade_row], ignore_index=True)
                        daily_pnl.append({'Time': self.returns['time'].iloc[i], 'Daily PNL': combined_pnl})
                        pnl_entries.append({'Time': self.returns['time'].iloc[i], 'Signal': signal, 'PNL': combined_pnl, f'Cummulative PNL_{pair_name}': cum_pnl})
                        

                    #close a short position
                elif self.position == -1 and spread_series.iloc[i] <= bb_df['SMA'].iloc[i]:  # Use spread_series and bb_df['SMA']
                    self.exit_price_A = self.data[ticker1].iloc[i] * (1 + slippage)
                    self.exit_price_B = self.data[ticker2].iloc[i] * (1 + slippage)
                    stockA_pnl = (self.entry_price_A - self.exit_price_A) * self.shares_per_position # Correct calculation for Stock A PNL
                    stockB_pnl = (self.exit_price_B - self.entry_price_B) * self.shares_per_position

                    combined_pnl = stockA_pnl + stockB_pnl - transaction_cost      #Correct combined_pnl calculation
                    daily_pnl.append({'Time': self.returns['time'].iloc[i], f'Daily PNL_{pair_name}': combined_pnl}) #Append here, *after* combined_pnl is calculated

                    self.position = 0
                    transaction_cost = 0 #Reset transaction costs once the trade is done.

                    if combined_pnl != 0:  # ***Check and update logs *inside* the elif***
                        cum_pnl = self.compute_cumulative_pnl(combined_pnl)
                        new_trade_row = pd.DataFrame({  # Create and append to trade_log_df directly
                            # ... (same as before)
                        })
                        self.trade_log_df = pd.concat([self.trade_log_df, new_trade_row], ignore_index=True)
                        daily_pnl.append({'Time': self.returns['time'].iloc[i], 'Daily PNL': combined_pnl})
                        pnl_entries.append({'Time': self.returns['time'].iloc[i], 'Signal': signal, 'PNL': combined_pnl, f'Cummulative PNL_{pair_name}': cum_pnl})
                        
            #DataFrame updates and total PNL calculations *after outer loop*
            if daily_pnl: #Update self.df_pnl_log outside inner loop
                daily_pnl_df = pd.DataFrame(daily_pnl)
                
                #remove overlapping columns, except for time
                common_cols = self.df_pnl_log.columns.intersection(daily_pnl_df.columns).tolist()
                common_cols.remove('Time')  # Keep 'Time' for merging
                daily_pnl_df = daily_pnl_df.drop(columns=common_cols, errors='ignore')

                self.df_pnl_log = pd.merge(self.df_pnl_log, daily_pnl_df, on='Time', how='outer')

        # Calculate daily and cumulative PNL *after* outer loop
        self.df_pnl_log['Daily Changes'] = self.df_pnl_log[[col for col in self.df_pnl_log.columns if 'Daily PNL' in col]].sum(axis=1, skipna=True)
        self.df_pnl_log['Total Cumulative PNL'] = self.df_pnl_log['Daily Changes'].cumsum()
        self.df_pnl_log = self.df_pnl_log[['Time', 'Daily Changes', 'Total Cumulative PNL']]  # Select only desired columns
        
        return self.df_pnl_log 

    def calculate_analytics(self, pnl_df):  # New method for analytics

        pnl_df.loc[:, 'Time'] = pd.to_datetime(pnl_df['Time'])
        pnl_df = pnl_df.set_index('Time')


        monthly_changes = pnl_df['Daily Changes'].resample('ME').sum() # Resample *filtered* daily changes to monthly
        monthly_returns = monthly_changes.pct_change() # Calculate Monthly Returns (pct_change) 
        monthly_returns_filtered = monthly_returns.replace([np.inf, -np.inf], np.nan).dropna()

        if monthly_returns_filtered.isna().all():
            print("All monthly returns are NaN. Cannot calculate Sharpe ratio.")
            sharpe_ratio = np.nan
        elif len(monthly_returns_filtered) <= 1:
            print("Not enough data points to compute standard deviation. Cannot calculate Sharpe ratio.")
            sharpe_ratio = np.nan
        elif monthly_returns_filtered.std() == 0:
             print("Monthly returns have zero standard deviation. Cannot calculate Sharpe ratio.")
             sharpe_ratio = np.nan
        else:
            sharpe_ratio = np.sqrt(12) * (monthly_returns_filtered.mean() / monthly_returns_filtered.std())
        
        # Max Drawdown Calculation
        cumulative_pnl = pnl_df['Daily Changes'].cumsum()
        peak = cumulative_pnl.cummax()
        drawdown = (cumulative_pnl - peak) / peak
        max_drawdown = drawdown.min()

        analytics_summary = {'Total PNL': pnl_df['Total Cumulative PNL'].iloc[-1], #Store Total PNL
                                'Sharpe Ratio': sharpe_ratio,
                                'Max Drawdown': max_drawdown} #Store calculated metrics in a dictionary for later access
        
        return analytics_summary


if __name__ == "__main__":
    strategy = PairTradingStrategy(tickers=[], start='2023-01-31', end='2024-12-31')
    strategy.download_data()
    strategy.calculate_returns()
    
    cointegrated_pairs_list = strategy.find_cointegrated_pairs() # Call directly
    cointegrated_pairs_df = pd.DataFrame(cointegrated_pairs_list, columns=['Ticker 1', 'Ticker 2', 'Beta', 'P-Value'])
    print(cointegrated_pairs_df)  # Print the DataFrame
    
    #total_pnl = strategy.backtest()
    #print(f"Total PNL (from main): {total_pnl}") #Verify access to the total_pnl value
   
    pnl_df = strategy.backtest()  # Run the backtest
    filtered_pnl_df = pnl_df[(pnl_df['Daily Changes'] != 0) & pnl_df['Daily Changes'].notna()]  #Filter *before*
    analytics = strategy.calculate_analytics(filtered_pnl_df) # Call calculate_analytics with df_pnl_log

    if analytics: # Check if analytics is not None (in case of errors)
        print("\nAnalytics:")
        print(f"Sharpe Ratio: {analytics['Sharpe Ratio']}")
        print(f"Total PNL: {analytics['Total PNL']}")  # Access and print Total PNL
        print(f"Max Drawdown: {analytics['Max Drawdown']}")
    print(strategy.df_pnl_log)   # Print the updated DataFrame

    print("\nFiltered Daily Changes (non-zero):")
    print(filtered_pnl_df[['Time', 'Daily Changes']]) # Print filtered DataFrame with 'Time' and 'Daily Changes'.

    



    