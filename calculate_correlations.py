
import pandas as pd
import numpy as np

def calculate_correlations_and_changes(data, lags):
    """Calculate correlations and changes for given data."""
    correlations = []
    for lag in lags:
        correlation = data['KOFEcGI'].corr(data['v2x_polyarchy'].shift(lag))
        correlations.append(correlation)
        print(f'Lag {lag} year(s): R = {correlation:.3f}')

    valid_correlations = [c for c in correlations if not pd.isna(c)]
    if not valid_correlations:
        print("\nNo valid correlations found for this country")
        return None, None
        
    max_corr = max(valid_correlations, key=abs)
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
    data['trade_change'] = data['KOFEcGI'].diff()

    max_corr, max_lag = calculate_correlations_and_changes(data, lags)
    threshold = data['democracy_change'].std()
    find_significant_periods(data, max_lag, threshold)

    return max_corr, max_lag

def analyze_country(data, country_name, lags):
    """Analyze a single country's data."""
    significant_periods = []
    max_corr, max_lag = calculate_correlations_and_changes(data, lags)
    
    if max_corr is None or max_lag is None:
        return significant_periods
        
    threshold = data['democracy_change'].std()
    
    # Find significant periods
    years = data.loc[data.index[:-max_lag], 'year']
    dem_changes = data['democracy_change'].iloc[:-max_lag]
    future_trades = data['trade_change'].shift(-max_lag).iloc[:-max_lag]
    
    mask = (dem_changes > threshold) & (future_trades > 0)
    for year, dem_change, trade_change in zip(years[mask], dem_changes[mask], future_trades[mask]):
        significant_periods.append({
            'country': country_name,
            'year': int(year),
            'democracy_growth': dem_change,
            'trade_growth': trade_change,
            'lag_years': max_lag,
            'correlation': max_corr
        })
    
    return significant_periods

def perform_time_lag_analysis():
    # Load data
    excel_file = pd.ExcelFile('data/democracy_trade_analysis.xlsx')
    df = pd.concat([pd.read_excel(excel_file, sheet_name=sheet) 
                   for sheet in excel_file.sheet_names], ignore_index=True)

    lags = range(1, 6)
    all_significant_periods = []

    print("\n=== Time Lag Analysis Results ===")
    print("================================")

    # Analyze each country
    for country in df['country_name'].unique():
        country_data = df[df['country_name'] == country].copy()
        if len(country_data) > 10:  # Ensure enough data points
            print(f"\nAnalyzing {country}...")
            
            # Calculate changes
            country_data['democracy_change'] = country_data['v2x_polyarchy'].diff()
            country_data['trade_change'] = country_data['KOFEcGI'].diff()
            
            # Get significant periods
            periods = analyze_country(country_data, country, lags)
            all_significant_periods.extend(periods)

    # Sort and display top 10 instances
    print("\n=== Top 10 Instances of Democracy-Led Trade Growth ===")
    print("====================================================")
    
    # Sort by democracy growth magnitude
    sorted_periods = sorted(all_significant_periods, 
                          key=lambda x: x['democracy_growth'], 
                          reverse=True)[:10]
    
    # Print results in aligned format
    print("\n{:<20} {:<6} {:>12} {:>12} {:>8} {:>12}".format(
        "Country", "Year", "Dem Growth", "Trade Growth", "Lag", "Correlation"))
    print("-" * 75)
    
    for period in sorted_periods:
        print("{:<20} {:<6d} {:>12.3f} {:>12.3f} {:>8d} {:>12.3f}".format(
            period['country'][:19],
            period['year'],
            period['democracy_growth'],
            period['trade_growth'],
            period['lag_years'],
            period['correlation']
        ))

if __name__ == "__main__":
    perform_time_lag_analysis()
