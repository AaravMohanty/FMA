# Import necessary libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import yfinance as yf

# Function to fetch hourly stock data using yfinance
def get_minutely_stock_data(ticker):
    # Fetches hourly stock data for the specified ticker symbol.
    stock = yf.Ticker(ticker)
    # Fetch today's data, on every 1 minute interval    
    data = stock.history(interval='1m', period='1d') 
    # Save the data to CSV and label it as ticker_hourly_data.csv
    data.to_csv(f'{ticker}_hourly_data.csv')  

def smoothed_factorial_moving_average_series(data, period, smoothing_factor=0.5):
    fma_values = []
    # iterate through the data
    for i in range(len(data)):
        # only calculate the fma after a certain period otherwise 
        # it'll be too premature when calculating and not give good calcs
        if i >= period - 1:
            # create a window of data for the specified period
            window_data = data[i - period + 1:i + 1]
            # Use logarithm of factorial to reduce extreme differences
            weights = np.array([math.log(math.factorial(j+1)) for j in range(period)])
            # Apply smoothing factor. raising it to the power to make the graph smooth
            weights = np.power(weights, smoothing_factor)
            # normalizing weights so it all adds up to 1
            weights = weights / np.sum(weights)
            # adding each data point/price with the respective weight to get FMA
            fma = np.sum(window_data * weights)
            # add the fma to the list
            fma_values.append(fma)
        else:
            fma_values.append(np.nan)
    return fma_values

# Function to calculate the Relative Strength Index (RSI)
def RSI(data):
    dataClose = data["Close"]
    dataOpen = data["Open"]
    priceChange = dataClose - dataOpen
    # create gain and loss variables to get the return
    gains = np.where(priceChange > 0, priceChange, 0)
    losses = np.where(priceChange < 0, -priceChange, 0)
    # get avg gain and avg loss
    avg_gain = pd.Series(gains).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(losses).rolling(window=14, min_periods=1).mean()
    # calculate rs accordingly
    rs = avg_gain / avg_loss
    # use rs to calculate rsi
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Function to calculate Bollinger Bands
def bollinger_bands(data, period=20, multiplier=2):
    dataOpen = data["Open"]
    sma = dataOpen.rolling(window=period).mean()
    std = dataOpen.rolling(window=period).std()
    
    upper_band = sma + (multiplier * std)
    lower_band = sma - (multiplier * std)
    
    return upper_band, sma, lower_band

# Function to calculate maximum drawdown based on buy and sell prices
def calculate_max_drawdown_portfolio(buy_prices, sell_prices):
    if not buy_prices or not sell_prices:
        return 0  # No trades made, so no drawdown
    
    # Find the highest sell price (peak) and the lowest buy price (trough) after the peak
    peak = max(sell_prices)
    trough = min(buy_prices)
    
    # Calculate the maximum drawdown ratio
    max_drawdown_ratio = (peak - trough) / peak
    
    return max_drawdown_ratio

def calculate_sharpe_ratio(profit, buy_price):
    return_rate = (profit - buy_price) / buy_price

    # Subtract the risk-free return (currently set to 0.0483)
    risk_free_return = 0.0483
    excess_return = return_rate - risk_free_return

    # For an individual trade, standard deviation would be the price itself
    sharpe_ratio = excess_return / buy_price
    
    return sharpe_ratio

# Main Program
# Set the stock symbol and load data
ticker_name = input("Enter Stock Ticker: ")
hourly_data = get_minutely_stock_data(ticker_name)

# Load the saved data from the CSV file
filepath = f'{ticker_name}_hourly_data.csv'
data = pd.read_csv(filepath)

# Set the period (in minutes) for FMA and Bollinger Bands
fma_period = 20
bb_period = 20

# Calculate Bollinger Bands
upper_band, middle_band, lower_band = bollinger_bands(data, bb_period)

# Calculate FMA for the 'Open' column
data[f'FMA_Open{fma_period}'] = smoothed_factorial_moving_average_series(data['Open'], fma_period, 0.9)

# Calculate RSI
rsi_values = RSI(data)
data['RSI'] = rsi_values

# Initialize a column for trading signals
data['Signal'] = ''

# Lists to store buy and sell prices
buy_prices = []
sell_prices = []
sharpe_ratios = []

canSell = False  # Track when buying is allowed
# Loop to define buying and selling conditions based on the FMA, Bollinger Bands, and RSI
for i in range(len(data)):\
    # add to buy prices if it hits all the signal conditions
    if (data['Open'][i] < data[f'FMA_Open{fma_period}'][i] and
        data['RSI'][i] < 30 and
        data['Open'][i] <= lower_band[i] and not canSell):
        data.loc[i, 'Signal'] = 'Buy'
        canSell = True
        buy_prices.append(float(data['Open'][i]))  # Store buy price
    # add to sell prices if it hits all the signal conditions
    elif (data['Open'][i] > data[f'FMA_Open{fma_period}'][i] and
          data['RSI'][i] > 70 and
          data['Open'][i] >= upper_band[i] and canSell):
        data.loc[i, 'Signal'] = 'Sell'
        canSell = False
        sell_price = float(data['Open'][i])
        sell_prices.append(sell_price)  # Store sell price
    # Calculate profit for this trade
        buy_price = buy_prices[-1]
        profit = sell_price - buy_price

        # Calculate Sharpe Ratio for this trade
        sharpe_ratio = calculate_sharpe_ratio(profit, buy_price)
        sharpe_ratios.append(sharpe_ratio)

# Output buy and sell prices
print("\nAll Buy Signals:", buy_prices)
print("All Sell Signals:", sell_prices)

# If there are more buy prices than sell prices, remove the last buy
if len(buy_prices) > len(sell_prices):
    buy_prices.pop()

if buy_prices and sell_prices:
    # Calculate total buy and sell prices
    total_buy_prices = sum(buy_prices)
    total_sell_prices = sum(sell_prices)

    print("\nProfit: ", total_sell_prices - total_buy_prices)

    max_dd_ratio = calculate_max_drawdown_portfolio(buy_prices, sell_prices)

    # Output the Maximum Drawdown Ratio
    print("\nMaximum Drawdown Ratio: ", max_dd_ratio)

    # Calculate and output the Sharpe Ratio
    opening_price = data['Open'][0]
    profit = total_sell_prices - total_buy_prices  # Ensure profit is calculated from valid sell prices
    sharpe_ratio = calculate_sharpe_ratio(profit, buy_prices[-1])
    print("\nAverage Sharpe Ratio: ", np.mean(sharpe_ratios))
else:
    # Calculate total buy and sell prices
    total_buy_prices = sum(buy_prices)
    total_sell_prices = sum(sell_prices)

    print("\nProfit: N/A. No trades made.")

    max_dd_ratio = calculate_max_drawdown_portfolio(buy_prices, sell_prices)

    # Output the Maximum Drawdown Ratio
    print("\nMaximum Drawdown Ratio: N/A")

    print("\nAverage Sharpe Ratio: N/A")

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(14, 18), sharex=True)

# Plot Price Chart with Bollinger Bands, FMA, EMA, and SMA
axs[0].plot(data['Open'], label='Open Price', color='blue')
axs[0].plot(upper_band, label='Upper Bollinger Band', color='red', linestyle='--')
axs[0].plot(lower_band, label='Lower Bollinger Band', color='green', linestyle='--')
axs[0].plot(data[f'FMA_Open{fma_period}'], label=f'FMA (Period: {fma_period})', color='purple')

axs[0].set_title('Price Chart with Bollinger Bands, FMA, EMA, and SMA')
axs[0].legend()

# Plot RSI
axs[1].plot(data['RSI'], label='RSI', color='magenta')
axs[1].axhline(70, color='red', linestyle='--', label='Overbought (70)')
axs[1].axhline(30, color='green', linestyle='--', label='Oversold (30)')
axs[1].set_title('Relative Strength Index (RSI)')
axs[1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()