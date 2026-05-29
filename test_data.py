import yfinance as yf
import pandas as pd
import numpy as np

# Download Apple stock data
df = yf.download('AAPL', period='1mo', interval='1d')

# --- THE CRITICAL FIX ---
df = df.reset_index()        # Moves 'Datetime' from index to a column
df = df.rename(columns={'Datetime': 'Date'}) # Renames it to 'Date'

# Select the columns
test_data = df[['Date', 'Close']].copy()

# Rename to fit DataFill (x and y)
test_data.columns = ['x', 'y']

# Create missing gaps for testing
test_data.iloc[5, 1] = np.nan
test_data.iloc[12, 1] = np.nan

# Save to actual CSV
test_data.to_csv('test_stock_data.csv', index=False)

print("Done! 'test_stock_data.csv' created.")
