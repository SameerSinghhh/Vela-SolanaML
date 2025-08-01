#!/usr/bin/env python3
import pandas as pd
import os
from datetime import datetime

def convert_date(date_str):
    """Convert date from 'Jul 15, 2025' format to 'YYYY-MM-DD' format"""
    try:
        # Parse the date string and convert to standard format
        date_obj = datetime.strptime(date_str.strip(), '%b %d, %Y')
        return date_obj.strftime('%Y-%m-%d')
    except:
        return date_str.strip()  # Return original if parsing fails

def clean_crypto_file(input_filename, output_filename):
    """Clean up crypto CSV files (BTC, ETH, SOL) with standard format"""
    print(f"Cleaning crypto file {input_filename} -> {output_filename}...")
    
    try:
        # Read the file with tab separator
        df = pd.read_csv(input_filename, sep='\t')
        
        # Clean up column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Standardize the Date column
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(convert_date)
        
        # Clean up numeric columns - remove commas and convert to float
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                # Remove commas and convert to numeric
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Save as proper CSV with comma separators
        df.to_csv(output_filename, index=False)
        print(f"✓ Successfully cleaned {output_filename}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        
    except Exception as e:
        print(f"✗ Error cleaning {input_filename}: {str(e)}")

def clean_sp_file(input_filename, output_filename):
    """Clean up SP file with different format"""
    print(f"Cleaning SP file {input_filename} -> {output_filename}...")
    
    try:
        # The SP file has a space-padded header, so we need to handle it specially
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        
        # Parse the header manually - it's space-separated
        header_line = lines[0].strip()
        # Split by multiple spaces to get columns
        import re
        header_cols = re.split(r'\s{2,}', header_line)
        print(f"  Detected columns: {header_cols}")
        
        # Read the data rows (skip header) with tab separator
        df = pd.read_csv(input_filename, sep='\t', skiprows=1, header=None)
        
        print(f"  Data has {len(df.columns)} columns")
        
        # The data actually has 7 columns: Date, Price, Open, High, Low, Empty, Change %
        # We'll assign proper column names and drop the empty column
        if len(df.columns) == 7:
            df.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Empty', 'Change %']
            # Drop the empty column
            df = df.drop('Empty', axis=1)
        else:
            # Fallback - use what we detect from header
            df.columns = header_cols[:len(df.columns)]
        
        # Standardize the Date column
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(convert_date)
        
        # Clean up numeric columns - remove commas and convert to float
        numeric_columns = ['Price', 'Open', 'High', 'Low']
        for col in numeric_columns:
            if col in df.columns:
                # Remove commas and convert to numeric
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Clean up the Change % column - keep as string but clean it
        if 'Change %' in df.columns:
            df['Change %'] = df['Change %'].astype(str).str.strip()
        
        # Save as proper CSV with comma separators
        df.to_csv(output_filename, index=False)
        print(f"✓ Successfully cleaned {output_filename}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        
    except Exception as e:
        print(f"✗ Error cleaning {input_filename}: {str(e)}")

def clean_dxy_vix_file(input_filename, output_filename):
    """Clean up DXY/VIX CSV files to match SOL format"""
    print(f"Cleaning DXY/VIX file {input_filename} -> {output_filename}...")
    
    try:
        df = pd.read_csv(input_filename, sep='\t')
        df.columns = df.columns.str.strip()
        
        # Standardize the Date column
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(convert_date)
        
        # Clean up numeric columns - remove commas and convert to numeric
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle Volume column - replace '-' with '0' and clean
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].astype(str).str.replace('-', '0')
            df['Volume'] = df['Volume'].str.replace(',', '')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['Volume'] = df['Volume'].fillna(0).astype(int)
        
        # Keep original date order (most recent to oldest, same as other files)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        df.to_csv(output_filename, index=False)
        print(f"✓ Successfully cleaned {output_filename}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        
    except Exception as e:
        print(f"✗ Error cleaning {input_filename}: {str(e)}")

def main():
    print("Starting CSV cleanup process...")
    print("Reading from raw_ files and outputting to cleaned versions")
    print("=" * 60)
    
    # Clean crypto files (BTC, ETH, SOL) - they have the same format
    crypto_files = [
        ('raw_price_data_btc.csv', 'price_data_btc.csv'),
        ('raw_price_data_eth.csv', 'price_data_eth.csv'),
        ('raw_price_data_sol.csv', 'price_data_sol.csv')
    ]
    
    for input_file, output_file in crypto_files:
        if os.path.exists(input_file):
            clean_crypto_file(input_file, output_file)
        else:
            print(f"✗ File not found: {input_file}")
    
    # Clean SP file separately - it has different format
    sp_input = 'raw_price_data_sp.csv'
    sp_output = 'price_data_sp.csv'
    if os.path.exists(sp_input):
        clean_sp_file(sp_input, sp_output)
    else:
        print(f"✗ File not found: {sp_input}")
    
    # Clean DXY and VIX files - they have similar format to crypto files but different structure
    dxy_vix_files = [
        ('raw_price_data_dxy.csv', 'price_data_dxy.csv'),
        ('raw_price_data_vix.csv', 'price_data_vix.csv')
    ]
    
    for input_file, output_file in dxy_vix_files:
        if os.path.exists(input_file):
            clean_dxy_vix_file(input_file, output_file)
        else:
            print(f"✗ File not found: {input_file}")
    
    print("=" * 60)
    print("CSV cleanup completed!")
    print("Original files preserved with 'raw_' prefix")
    print("Cleaned files: BTC, ETH, SOL, SP500, DXY, VIX")

if __name__ == "__main__":
    main() 