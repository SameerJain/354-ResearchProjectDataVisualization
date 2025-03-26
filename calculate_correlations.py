
import pandas as pd
import numpy as np

# Load data from all sheets
excel_file = pd.ExcelFile('data/democracy_trade_analysis.xlsx')
sheets = excel_file.sheet_names
df = pd.concat([pd.read_excel(excel_file, sheet_name=sheet) for sheet in sheets], ignore_index=True)

# Print unique regime types in the dataset
print("\nRegime types in dataset:", df['Regime_Type'].unique())

# Calculate correlations for each country
correlations = {}
for country in df['country_name'].unique():
    country_data = df[df['country_name'] == country]
    if not country_data.empty:
        valid_data = country_data[['v2x_polyarchy', 'KOFTrGIdf']].dropna()
        correlation = valid_data['v2x_polyarchy'].corr(valid_data['KOFTrGIdf']) if len(valid_data) > 1 else None
        regime_type = country_data['Regime_Type'].iloc[0] if not country_data['Regime_Type'].isna().all() else 'Unknown'
        correlations[country] = {'r': correlation, 'regime': regime_type}
                
print(f"\nProcessed {len(correlations)} countries")

# Group by regime type and sort by correlation
regime_groups = {}
for country, data in correlations.items():
    regime = data['regime']
    if regime not in regime_groups:
        regime_groups[regime] = []
    regime_groups[regime].append((country, data['r']))

# Print results
if regime_groups:
    # Filter out nan keys and sort valid regime types
    valid_regimes = [r for r in regime_groups.keys() if pd.notnull(r)]
    for regime in sorted(valid_regimes):
        print(f"\n{regime}:")
        print("-" * len(regime + ":"))
        
        # Sort countries, putting None values at the end
        valid_data = [(c, r) for c, r in regime_groups[regime] if r is not None]
        none_data = [(c, r) for c, r in regime_groups[regime] if r is None]
        
        # Sort valid data by correlation value
        sorted_valid = sorted(valid_data, key=lambda x: x[1], reverse=True)
        
        # Print all data
        for country, r in sorted_valid + none_data:
            if r is None:
                print(f"{country}: insufficient data")
            else:
                print(f"{country}: {r:.3f}")
else:
    print("No valid correlations found. Please check your data.")
