import pandas as pd
import numpy as np

def calculate_correlations_and_changes(data, lags):
    """Calculate correlations and changes for given data."""
    correlations = []
    for lag in lags:
        correlation = data['KOFTrGIdf'].corr(data['v2x_polyarchy'].shift(lag))
        correlations.append(correlation)
        print(f'Lag {lag} year(s): R = {correlation:.3f}')

    max_corr = max((c for c in correlations if not pd.isna(c)), key=abs)
    max_lag = lags[correlations.index(max_corr)]
    print(f'\nStrongest correlation at {max_lag} year lag: R = {max_corr:.3f}')

    return max_corr, max_lag

def find_significant_periods(data, max_lag, threshold):
    """Find periods of significant democracy-led growth."""
    print("\nSignificant democracy-led growth periods:")
    print("Years where democracy growth preceded trade growth:")

    # Vectorized operations for finding significant periods
    years = data.loc[data.index[:-max_lag], 'year']
    dem_changes = data['democracy_change'].iloc[:-max_lag]
    future_trades = data['trade_change'].shift(-max_lag).iloc[:-max_lag]

    mask = (dem_changes > threshold) & (future_trades > 0)
    significant_periods = pd.DataFrame({
        'year': years[mask],
        'dem_change': dem_changes[mask],
        'future_trade': future_trades[mask]
    })

    for _, row in significant_periods.iterrows():
        print(f"  {int(row['year'])}: Democracy growth: {row['dem_change']:.3f}, "
              f"Led to trade growth: {row['future_trade']:.3f} after {max_lag} years")

def analyze_group(data, name, lags):
    """Analyze a group of data (global or category)."""
    print(f'\nAnalyzing {name}:')
    print('-' * 40)

    # Calculate changes using vectorized operations
    data['democracy_change'] = data['v2x_polyarchy'].diff()
    data['trade_change'] = data['KOFTrGIdf'].diff()

    max_corr, max_lag = calculate_correlations_and_changes(data, lags)
    threshold = data['democracy_change'].std()
    find_significant_periods(data, max_lag, threshold)

    return max_corr, max_lag

def perform_time_lag_analysis():
    # Load data more efficiently by specifying dtypes
    excel_file = pd.ExcelFile('data/democracy_trade_analysis.xlsx')
    df = pd.concat([pd.read_excel(excel_file, sheet_name=sheet) 
                   for sheet in excel_file.sheet_names], ignore_index=True)

    lags = range(1, 6)

    print("\n=== Time Lag Analysis Results ===")
    print("================================")

    # Global analysis
    global_avg = df.groupby('year', as_index=False).agg({
        'v2x_polyarchy': 'mean',
        'KOFTrGIdf': 'mean'
    })
    analyze_group(global_avg, "Global Average", lags)

    # Category analysis
    categories = ['Liberal Democracy', 'Electoral Democracy', 
                 'Electoral Autocracy', 'Closed Autocracy']

    print("\n=== Category Analysis Results ===")
    print("================================")

    for category in categories:
        category_avg = df[df['Category'] == category].groupby('year', as_index=False).agg({
            'v2x_polyarchy': 'mean',
            'KOFTrGIdf': 'mean'
        })
        analyze_group(category_avg, category, lags)

if __name__ == "__main__":
    perform_time_lag_analysis()