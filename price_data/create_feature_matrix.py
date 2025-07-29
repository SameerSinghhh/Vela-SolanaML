#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_date_range():
    """Create date range from July 20, 2023 to July 14, 2025"""
    start_date = datetime(2023, 7, 20)
    end_date = datetime(2025, 7, 14)
    
    date_range = []
    current_date = start_date
    
    while current_date <= end_date:
        date_range.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    return date_range

def load_sol_data():
    """Load and prepare SOL price data"""
    print("Loading SOL price data...")
    
    try:
        sol_df = pd.read_csv('price_data_sol.csv')
        sol_df['Date'] = pd.to_datetime(sol_df['Date'])
        
        # Sort by date to ensure proper order
        sol_df = sol_df.sort_values('Date')
        
        print(f"SOL data loaded: {len(sol_df)} rows")
        print(f"Date range: {sol_df['Date'].min()} to {sol_df['Date'].max()}")
        
        return sol_df
    
    except Exception as e:
        print(f"Error loading SOL data: {e}")
        return None

def calculate_returns_and_targets(df):
    """Calculate 1-day returns and next-day target classifications"""
    print("Calculating returns and target variables...")
    
    # Calculate 1-day return as percentage change
    df['sol_return_1d'] = df['Close'].pct_change() * 100
    
    # Calculate next day's return for target variable
    df['next_day_return'] = df['sol_return_1d'].shift(-1)
    
    # Create target variable based on next day's return
    # -1: down (<-2%), 0: neutral (±2%), 1: up (>+2%)
    conditions = [
        df['next_day_return'] < -2,    # Down
        df['next_day_return'] > 2,     # Up
    ]
    choices = [-1, 1]
    
    df['target_next_day_direction'] = np.select(conditions, choices, default=0)
    
    # Remove the temporary next_day_return column
    df = df.drop('next_day_return', axis=1)
    
    return df

def create_feature_matrix():
    """Create the main feature matrix"""
    print("Creating feature matrix...")
    print("=" * 50)
    
    # Create date range
    dates = create_date_range()
    print(f"Created date range: {len(dates)} days from {dates[0]} to {dates[-1]}")
    
    # Create initial DataFrame with dates
    feature_matrix = pd.DataFrame({'Date': dates})
    feature_matrix['Date'] = pd.to_datetime(feature_matrix['Date'])
    
    # Load SOL data
    sol_df = load_sol_data()
    if sol_df is None:
        print("Failed to load SOL data. Exiting.")
        return
    
    # Merge with feature matrix (left join to keep all dates)
    feature_matrix = feature_matrix.merge(
        sol_df[['Date', 'Close']], 
        on='Date', 
        how='left'
    )
    
    # Calculate returns and targets
    feature_matrix = calculate_returns_and_targets(feature_matrix)
    
    # Keep only the columns we want for now
    columns_to_keep = ['Date', 'sol_return_1d', 'target_next_day_direction']
    feature_matrix = feature_matrix[columns_to_keep]
    
    # Convert Date back to string format for consistency
    feature_matrix['Date'] = feature_matrix['Date'].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    output_file = 'feature_matrix.csv'
    feature_matrix.to_csv(output_file, index=False)
    
    print(f"\n✓ Feature matrix created and saved to {output_file}")
    print(f"Shape: {feature_matrix.shape}")
    print(f"Columns: {list(feature_matrix.columns)}")
    
    # Show some statistics
    print("\nTarget variable distribution:")
    target_counts = feature_matrix['target_next_day_direction'].value_counts().sort_index()
    for target, count in target_counts.items():
        direction = {-1: 'Down (<-2%)', 0: 'Neutral (±2%)', 1: 'Up (>+2%)'}[target]
        print(f"  {target}: {count} days ({direction})")
    
    # Show first and last few rows
    print(f"\nFirst 5 rows:")
    print(feature_matrix.head())
    print(f"\nLast 5 rows:")
    print(feature_matrix.tail())
    
    return feature_matrix

if __name__ == "__main__":
    feature_matrix = create_feature_matrix() 