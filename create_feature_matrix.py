#!/usr/bin/env python3
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

def create_feature_matrix():
    """Create the initial feature matrix for Solana price prediction"""
    
    print("Creating feature matrix for Solana price prediction...")
    print("=" * 60)
    
    # Step 1: Create date range from July 23, 2023 to July 14, 2025
    start_date = datetime(2023, 7, 23)
    end_date = datetime(2025, 7, 14)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create the initial dataframe with dates
    feature_matrix = pd.DataFrame({
        'date': date_range.strftime('%Y-%m-%d')
    })
    
    print(f"Created date range: {len(feature_matrix)} days from {start_date.date()} to {end_date.date()}")
    
    # Step 2: Load SOL price data
    print("Loading SOL price data...")
    sol_data = pd.read_csv('price_data/price_data_sol.csv')
    sol_data['Date'] = pd.to_datetime(sol_data['Date'])
    
    # Sort by date (oldest first) for proper calculation
    sol_data = sol_data.sort_values('Date').reset_index(drop=True)
    
    print(f"SOL data loaded: {len(sol_data)} records from {sol_data['Date'].min().date()} to {sol_data['Date'].max().date()}")
    
    # Step 2.1: Load BTC price data
    print("Loading BTC price data...")
    btc_data = pd.read_csv('price_data/price_data_btc.csv')
    btc_data['Date'] = pd.to_datetime(btc_data['Date'])
    btc_data = btc_data.sort_values('Date').reset_index(drop=True)
    
    print(f"BTC data loaded: {len(btc_data)} records from {btc_data['Date'].min().date()} to {btc_data['Date'].max().date()}")
    
    # Step 2.2: Load ETH price data
    print("Loading ETH price data...")
    eth_data = pd.read_csv('price_data/price_data_eth.csv')
    eth_data['Date'] = pd.to_datetime(eth_data['Date'])
    eth_data = eth_data.sort_values('Date').reset_index(drop=True)
    
    print(f"ETH data loaded: {len(eth_data)} records from {eth_data['Date'].min().date()} to {eth_data['Date'].max().date()}")
    
    # Step 3: Calculate 1-day returns (percentage change from previous close)
    sol_data['sol_return_1d'] = sol_data['Close'].pct_change() * 100
    
    # Step 3.0: Calculate ACTUAL next-day return (what actually happens the next day)
    # This is what we would gain/lose if we traded based on today's prediction
    sol_data['sol_actual_next_day_return'] = sol_data['Close'].pct_change().shift(-1) * 100
    
    # Step 3.1: Calculate 3-day returns (percentage change from 3 days ago)
    sol_data['sol_return_3d'] = sol_data['Close'].pct_change(periods=3) * 100
    
    # Step 3.2: Calculate 7-day returns (percentage change from 7 days ago)
    sol_data['sol_return_7d'] = sol_data['Close'].pct_change(periods=7) * 100
    
    # Step 3.3: Calculate BTC returns
    btc_data['btc_return_1d'] = btc_data['Close'].pct_change() * 100
    btc_data['btc_return_3d'] = btc_data['Close'].pct_change(periods=3) * 100
    btc_data['btc_return_7d'] = btc_data['Close'].pct_change(periods=7) * 100
    
    # Step 3.4: Calculate ETH returns
    eth_data['eth_return_1d'] = eth_data['Close'].pct_change() * 100
    eth_data['eth_return_3d'] = eth_data['Close'].pct_change(periods=3) * 100
    eth_data['eth_return_7d'] = eth_data['Close'].pct_change(periods=7) * 100
    
    # Step 3.5: Calculate 7-day volatility (rolling standard deviation of daily returns)
    sol_data['sol_volatility_7d'] = sol_data['sol_return_1d'].rolling(window=7).std()
    
    # Step 3.6: Calculate 14-day RSI (Relative Strength Index)
    sol_data['sol_rsi_14'] = ta.momentum.RSIIndicator(close=sol_data['Close'], window=14).rsi()
    
    # Step 3.7: Calculate MACD histogram (12/26/9 standard parameters)
    macd = ta.trend.MACD(close=sol_data['Close'], window_slow=26, window_fast=12, window_sign=9)
    sol_data['sol_macd_histogram'] = macd.macd_diff()
    
    # Step 3.8: Calculate Simple Moving Averages
    sol_data['sol_sma_7'] = sol_data['Close'].rolling(window=7).mean()
    sol_data['sol_sma_14'] = sol_data['Close'].rolling(window=14).mean()
    
    # Step 3.9: Calculate SMA-based features
    # Feature 1: sol_close / sol_sma_7
    sol_data['sol_close_sma7_ratio'] = sol_data['Close'] / sol_data['sol_sma_7']
    
    # Feature 2: sol_sma_7 / sol_sma_14
    sol_data['sol_sma7_sma14_ratio'] = sol_data['sol_sma_7'] / sol_data['sol_sma_14']
    
    # Feature 3: % difference from SMA7
    sol_data['sol_price_dev_from_sma7'] = (sol_data['Close'] - sol_data['sol_sma_7']) / sol_data['sol_sma_7']
    
    # Step 4: Calculate next-day target variable
    # Shift returns by -1 to get next day's return for prediction
    sol_data['next_day_return'] = sol_data['sol_return_1d'].shift(-1)
    
    # Create target variable based on next day's return
    # -1: down (< -2%), 0: neutral (-2% to +2%), 1: up (> +2%)
    def classify_movement(return_pct):
        if pd.isna(return_pct):
            return np.nan
        elif return_pct > 2.0:
            return 1  # Up
        elif return_pct < -2.0:
            return -1  # Down
        else:
            return 0  # Neutral
    
    sol_data['target_next_day'] = sol_data['next_day_return'].apply(classify_movement)
    
    # Step 4.1: Calculate rolling mean of previous 2 days' target values (with shift to prevent leakage)
    sol_data['target_next_day_rolling_mean_2d'] = sol_data['target_next_day'].shift(1).rolling(window=2).mean()
    
    # Step 5: Merge with feature matrix
    # Convert date column to datetime for merging
    feature_matrix['date_dt'] = pd.to_datetime(feature_matrix['date'])
    
    # Merge SOL data with feature matrix
    feature_matrix = feature_matrix.merge(
        sol_data[['Date', 'Close', 'sol_actual_next_day_return', 'sol_return_1d', 'sol_return_3d', 'sol_return_7d', 'sol_volatility_7d', 'sol_rsi_14', 'sol_macd_histogram', 'sol_close_sma7_ratio', 'sol_sma7_sma14_ratio', 'sol_price_dev_from_sma7', 'target_next_day', 'target_next_day_rolling_mean_2d']], 
        left_on='date_dt', 
        right_on='Date', 
        how='left'
    )
    
    # Merge BTC closing prices and returns
    feature_matrix = feature_matrix.merge(
        btc_data[['Date', 'Close', 'btc_return_1d', 'btc_return_3d', 'btc_return_7d']], 
        left_on='date_dt', 
        right_on='Date', 
        how='left',
        suffixes=('', '_btc')
    )
    
    # Merge ETH closing prices and returns
    feature_matrix = feature_matrix.merge(
        eth_data[['Date', 'Close', 'eth_return_1d', 'eth_return_3d', 'eth_return_7d']], 
        left_on='date_dt', 
        right_on='Date', 
        how='left',
        suffixes=('', '_eth')
    )
    
    # Clean up columns and rename appropriately
    feature_matrix = feature_matrix.drop(['date_dt', 'Date', 'Date_btc', 'Date_eth'], axis=1)
    feature_matrix = feature_matrix.rename(columns={
        'Close': 'sol_close',
        'Close_btc': 'btc_close', 
        'Close_eth': 'eth_close'
    })
    
    # Calculate relative price ratios
    feature_matrix['sol_price_relative_to_btc'] = feature_matrix['sol_close'] / feature_matrix['btc_close']
    feature_matrix['sol_price_relative_to_eth'] = feature_matrix['sol_close'] / feature_matrix['eth_close']
    
    # Reorder columns to group all 1d, 3d, 7d returns together
    column_order = ['date', 'sol_close', 'sol_actual_next_day_return', 'btc_close', 'eth_close', 'sol_return_1d', 'btc_return_1d', 'eth_return_1d', 'sol_return_3d', 'btc_return_3d', 'eth_return_3d', 'sol_return_7d', 'btc_return_7d', 'eth_return_7d', 'sol_volatility_7d', 'sol_price_relative_to_btc', 'sol_price_relative_to_eth', 'sol_rsi_14', 'sol_macd_histogram', 'sol_close_sma7_ratio', 'sol_sma7_sma14_ratio', 'sol_price_dev_from_sma7', 'target_next_day_rolling_mean_2d', 'target_next_day']
    feature_matrix = feature_matrix[column_order]
    
    # Step 6: Display summary statistics
    print("\nFeature Matrix Summary:")
    print(f"Total rows: {len(feature_matrix)}")
    print(f"SOL price data available: {feature_matrix['sol_close'].notna().sum()} days")
    print(f"SOL actual next day returns: {feature_matrix['sol_actual_next_day_return'].notna().sum()} days")
    print(f"BTC price data available: {feature_matrix['btc_close'].notna().sum()} days")
    print(f"ETH price data available: {feature_matrix['eth_close'].notna().sum()} days")
    print(f"SOL 1-day returns calculated: {feature_matrix['sol_return_1d'].notna().sum()} days")
    print(f"BTC 1-day returns calculated: {feature_matrix['btc_return_1d'].notna().sum()} days")
    print(f"ETH 1-day returns calculated: {feature_matrix['eth_return_1d'].notna().sum()} days")
    print(f"SOL 3-day returns calculated: {feature_matrix['sol_return_3d'].notna().sum()} days")
    print(f"BTC 3-day returns calculated: {feature_matrix['btc_return_3d'].notna().sum()} days")
    print(f"ETH 3-day returns calculated: {feature_matrix['eth_return_3d'].notna().sum()} days")
    print(f"SOL 7-day returns calculated: {feature_matrix['sol_return_7d'].notna().sum()} days")
    print(f"BTC 7-day returns calculated: {feature_matrix['btc_return_7d'].notna().sum()} days")
    print(f"ETH 7-day returns calculated: {feature_matrix['eth_return_7d'].notna().sum()} days")
    print(f"7-day volatility calculated: {feature_matrix['sol_volatility_7d'].notna().sum()} days")
    print(f"SOL/BTC ratio calculated: {feature_matrix['sol_price_relative_to_btc'].notna().sum()} days")
    print(f"SOL/ETH ratio calculated: {feature_matrix['sol_price_relative_to_eth'].notna().sum()} days")
    print(f"14-day RSI calculated: {feature_matrix['sol_rsi_14'].notna().sum()} days")
    print(f"MACD histogram calculated: {feature_matrix['sol_macd_histogram'].notna().sum()} days")
    print(f"SMA close/7d ratio calculated: {feature_matrix['sol_close_sma7_ratio'].notna().sum()} days")
    print(f"SMA 7d/14d ratio calculated: {feature_matrix['sol_sma7_sma14_ratio'].notna().sum()} days")
    print(f"Price deviation from SMA7 calculated: {feature_matrix['sol_price_dev_from_sma7'].notna().sum()} days")
    print(f"Target variable available: {feature_matrix['target_next_day'].notna().sum()} days")
    print(f"Target 2d rolling mean calculated: {feature_matrix['target_next_day_rolling_mean_2d'].notna().sum()} days")
    
    print("\nTarget variable distribution:")
    target_counts = feature_matrix['target_next_day'].value_counts().sort_index()
    for target, count in target_counts.items():
        if target == -1:
            label = "Down (< -2%)"
        elif target == 0:
            label = "Neutral (-2% to +2%)"
        elif target == 1:
            label = "Up (> +2%)"
        else:
            label = "Unknown"
        print(f"  {label}: {count} days")
    
    print(f"\nReturn statistics:")
    print("Actual next-day returns (for trading simulation):")
    print(feature_matrix['sol_actual_next_day_return'].describe())
    print("\nSOL 1-day returns:")
    print(feature_matrix['sol_return_1d'].describe())
    print("\nBTC 1-day returns:")
    print(feature_matrix['btc_return_1d'].describe())
    print("\nETH 1-day returns:")
    print(feature_matrix['eth_return_1d'].describe())
    print("\nSOL 3-day returns:")
    print(feature_matrix['sol_return_3d'].describe())
    print("\nBTC 3-day returns:")
    print(feature_matrix['btc_return_3d'].describe())
    print("\nETH 3-day returns:")
    print(feature_matrix['eth_return_3d'].describe())
    print("\nSOL 7-day returns:")
    print(feature_matrix['sol_return_7d'].describe())
    print("\nBTC 7-day returns:")
    print(feature_matrix['btc_return_7d'].describe())
    print("\nETH 7-day returns:")
    print(feature_matrix['eth_return_7d'].describe())
    print("\n7-day volatility:")
    print(feature_matrix['sol_volatility_7d'].describe())
    
    print(f"\nRelative price statistics:")
    print("SOL/BTC ratio:")
    print(feature_matrix['sol_price_relative_to_btc'].describe())
    print("\nSOL/ETH ratio:")
    print(feature_matrix['sol_price_relative_to_eth'].describe())
    
    print(f"\nTechnical indicator statistics:")
    print("14-day RSI:")
    print(feature_matrix['sol_rsi_14'].describe())
    print("\nMACD histogram:")
    print(feature_matrix['sol_macd_histogram'].describe())
    
    print(f"\nSMA-based feature statistics:")
    print("SOL Close/SMA7 ratio:")
    print(feature_matrix['sol_close_sma7_ratio'].describe())
    print("\nSMA7/SMA14 ratio:")
    print(feature_matrix['sol_sma7_sma14_ratio'].describe())
    print("\nSOL price deviation from SMA7:")
    print(feature_matrix['sol_price_dev_from_sma7'].describe())
    
    print(f"\nTarget-based feature statistics:")
    print("Target 2d rolling mean:")
    print(feature_matrix['target_next_day_rolling_mean_2d'].describe())
    
    # Step 7: Save the feature matrix
    output_file = 'feature_matrix.csv'
    feature_matrix.to_csv(output_file, index=False)
    print(f"\n✓ Feature matrix saved to: {output_file}")
    
    # Display first few rows
    print(f"\nFirst row of feature matrix:")
    print(feature_matrix.head(1).to_string(index=False))
    
    return feature_matrix

if __name__ == "__main__":
    feature_matrix = create_feature_matrix() 