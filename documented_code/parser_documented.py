"""
PSC 354 Research Project - Democracy and Trade Analysis
Documentation of Functions and Variables

FUNCTIONS:
----------
1. get_status(row)
   Purpose: Determines data presence status for each row
   Parameters: row - DataFrame row
   Returns: String indicating data status ('both', 'missing_kof', 'missing_polyarchy', 'missing_both')
   Used: To track missing data in the merged dataset

2. get_most_recent_regime(country_data)
   Purpose: Finds most recent valid regime type for a country
   Parameters: country_data - DataFrame subset for specific country
   Returns: String of regime type or 'Uncategorized'
   Used: For categorizing countries based on their most recent regime classification

MAJOR VARIABLES:
---------------
Data Loading:
- kof: DataFrame containing KOF Trade Globalization Index data
- vdem: DataFrame containing V-Dem democracy data
- all_codes: Set of country codes present in both datasets
- full_years: DataFrame with all year entries for each country

Data Processing:
- merged: Main DataFrame containing combined KOF and V-Dem data
- regime_mapping: Dictionary mapping numeric codes to regime type names
- special_country_mappings: Dictionary of special country code to name mappings
- categories: Dictionary storing most recent regime type for each country

Visualization:
- all_regime_data: List of DataFrames containing aggregated data by regime type
- regime_colors: Dictionary mapping regime types to color codes for plots

DATAFRAME COLUMNS:
-----------------
merged DataFrame columns:
- country_code: Unique identifier for each country
- year: Year of observation (1970-2020)
- country_name: Full name of country
- v2x_polyarchy: Democracy score from V-Dem
- v2x_regime: Numeric regime classification
- KOFTrGIdf: Trade openness score
- data_status: Indicates presence/absence of data
- Regime_Type: Text description of regime type
- Category: Country's classification based on most recent regime

EXCEL FORMATTING:
----------------
Format objects for Excel output:
- header_format: Grey background, bold text for headers
- missing_data_format: Light red background for missing data
- separator_format: Light blue background for country separators

PLOTTING PARAMETERS:
------------------
Individual country plots:
- fig: Figure object (12x6 size)
- ax1: Primary axis for democracy score
- ax2: Secondary axis for trade openness
- line1: Blue line for democracy score
- line2: Green line for trade openness

Aggregate plots:
- fig: Figure object (20x15 size)
- axes: 2x2 subplot array
- y1_min/max: Democracy score range
- y2_min/max: Trade openness range

FILE STRUCTURE:
--------------
Output directories:
- plots/aggregate: Contains regime-level plots
- plots/individual: Contains country-specific plots
- democracy_trade_analysis.xlsx: Excel output with formatted data

DATA PROCESSING STEPS:
--------------------
1. Load raw data (KOF and V-Dem)
2. Clean and standardize country codes
3. Create complete year range
4. Merge datasets
5. Add regime classifications
6. Generate data quality report
7. Export to Excel
8. Create visualizations

VISUALIZATION TYPES:
------------------
1. Individual country plots:
   - Democracy score vs Trade openness
   - Regime type timeline
   - Missing data highlighting

2. Aggregate plots:
   - Regime-level trends
   - Combined 4-panel plot
   - Consistent scaling across plots

ERROR HANDLING:
-------------
- Handles missing data in visualizations
- Accounts for empty country datasets
- Manages regime transitions
- Handles special country names

The script processes democracy and trade data, performing:
1. Data integration and cleaning
2. Quality assessment
3. Regime classification
4. Visualization
5. Excel report generation

Main outputs are:
1. Detailed Excel workbook
2. Country-specific plots
3. Regime-level visualizations
4. Data quality report
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Create directory structure for storing plots
os.makedirs('plots', exist_ok=True)
os.makedirs('plots/aggregate', exist_ok=True)  # For regime-level plots
os.makedirs('plots/individual', exist_ok=True)  # For country-level plots

# STEP 1: Load and preprocess KOF Trade Globalization Index data
# KOF data contains trade openness metrics for countries over time
kof = pd.read_excel('KOFGI_2024_public.xlsx', usecols=['code', 'year', 'KOFTrGIdf'])
kof = kof.rename(columns={'code': 'country_code'})  # Standardize country code column name
kof['country_code'] = kof['country_code'].astype(str).str.strip()  # Clean country codes
kof = kof[kof['year'].between(1970, 2020)]  # Filter for desired year range

# STEP 2: Load and preprocess V-Dem democracy data
# V-Dem data contains democracy scores and regime classifications
vdem = pd.read_csv('V-Dem-CY-Core-v15.csv', 
                   usecols=['country_text_id', 'country_name', 'year', 'v2x_regime', 'v2x_polyarchy'])
vdem = vdem.rename(columns={'country_text_id': 'country_code'})  # Standardize country code column name
vdem['country_code'] = vdem['country_code'].astype(str).str.strip()  # Clean country codes
vdem = vdem[vdem['year'].between(1970, 2020)]  # Filter for desired year range

# Get intersection of countries present in both datasets
all_codes = set(vdem['country_code'].unique()) & set(kof['country_code'].unique())

# STEP 3: Create complete year range for each country
# This ensures we have entries for all years, even if data is missing
full_years = pd.DataFrame([(c, y) for c in all_codes for y in range(1970, 2021)],
                          columns=['country_code', 'year'])

# Clean and prepare datasets for merging
vdem_clean = vdem[['country_code', 'country_name', 'year', 'v2x_polyarchy', 'v2x_regime']]
kof_clean = kof[['country_code', 'year', 'KOFTrGIdf']]

# Merge datasets with the complete year range
merged = full_years \
    .merge(vdem_clean, on=['country_code', 'year'], how='left') \
    .merge(kof_clean, on=['country_code', 'year'], how='left')

# STEP 4: Create data presence indicator
def get_status(row):
    """Determine which types of data are present/missing for each row"""
    poly = pd.notnull(row['v2x_polyarchy'])  # Check if democracy score exists
    kof = pd.notnull(row['KOFTrGIdf'])      # Check if trade openness exists
    if poly and kof:
        return 'both'
    elif poly:
        return 'missing_kof'
    elif kof:
        return 'missing_polyarchy'
    else:
        return 'missing_both'

merged['data_status'] = merged.apply(get_status, axis=1)

# STEP 5: Add regime classifications
# Map numeric regime codes to descriptive names
regime_mapping = {
    0: 'Closed Autocracy',
    1: 'Electoral Autocracy',
    2: 'Electoral Democracy',
    3: 'Liberal Democracy'
}

# Special mappings for countries with non-standard names
special_country_mappings = {
    'AFG': 'Afghanistan',
    'ARE': 'United Arab Emirates',
    # ... [other mappings]
    'KGZ': 'Kyrgyzstan'
}

# Map regime types and apply special country names
merged['Regime_Type'] = merged['v2x_regime'].map(regime_mapping)
merged.loc[merged['country_code'].isin(special_country_mappings.keys()), 'country_name'] = \
    merged.loc[merged['country_code'].isin(special_country_mappings.keys()), 'country_code'].map(special_country_mappings)

# Function to determine most recent regime type for categorization
def get_most_recent_regime(country_data):
    """Look backwards from 2020 until finding a valid regime type"""
    for year in range(2020, 1969, -1):
        year_data = country_data[country_data['year'] == year]
        if not year_data.empty and pd.notnull(year_data['v2x_regime'].iloc[0]):
            return regime_mapping.get(year_data['v2x_regime'].iloc[0])
    return 'Uncategorized'

# Apply categorization based on most recent regime type
categories = {}
for country in merged['country_code'].unique():
    country_data = merged[merged['country_code'] == country].sort_values('year', ascending=False)
    categories[country] = get_most_recent_regime(country_data)

merged['Category'] = merged['country_code'].map(categories)

# STEP 6: Generate data quality report
print("\n[DATA QUALITY REPORT]")
print("=" * 80)

for code in sorted(all_codes):
    sub = merged[merged['country_code'] == code]
    print("\n" + "-" * 80)
    print(f"Country: {code} ({code})")
    print("-" * 80)

    # Calculate missing years and coverage percentages
    poly_missing_years = sub[sub['v2x_polyarchy'].isnull()]['year'].tolist()
    kof_missing_years = sub[sub['KOFTrGIdf'].isnull()]['year'].tolist()
    total_years = len(sub)
    poly_coverage = ((total_years - len(poly_missing_years)) / total_years) * 100
    kof_coverage = ((total_years - len(kof_missing_years)) / total_years) * 100

    # Print data quality warnings and statistics
    if poly_missing_years or kof_missing_years:
        country_name = vdem[vdem['country_code'] == code]['country_name'].iloc[0] if not vdem[vdem['country_code'] == code].empty else code
        print(f"\n[WARNING] Data quality issues for {country_name} ({code})")
        print("-" * 80)
        print(f"Data Coverage:")
        print(f"  Democracy Score: {poly_coverage:.1f}% ({len(poly_missing_years)} missing years)")
        print(f"  Trade Openness: {kof_coverage:.1f}% ({len(kof_missing_years)} missing years)")

        # Print detailed missing data information
        if len(poly_missing_years) == total_years:
            print(f"  ⚠️ NO democracy score data available for this country")
        elif poly_missing_years:
            print(f"  Missing democracy score data for years: {sorted(poly_missing_years)}")

        if len(kof_missing_years) == total_years:
            print(f"  ⚠️ NO trade openness data available for this country")
        elif kof_missing_years:
            print(f"  Missing trade openness data for years: {sorted(kof_missing_years)}")

        # Identify patterns in missing data
        if len(poly_missing_years) > 0:
            earliest_poly = min(poly_missing_years) if poly_missing_years else None
            latest_poly = max(poly_missing_years) if poly_missing_years else None
            if earliest_poly and latest_poly:
                if latest_poly - earliest_poly + 1 == len(poly_missing_years):
                    print(f"  Note: Missing democracy data appears to be a continuous gap ({earliest_poly}-{latest_poly})")

        if len(kof_missing_years) > 0:
            earliest_kof = min(kof_missing_years) if kof_missing_years else None
            latest_kof = max(kof_missing_years) if kof_missing_years else None
            if earliest_kof and latest_kof:
                if latest_kof - earliest_kof + 1 == len(kof_missing_years):
                    print(f"  Note: Missing trade data appears to be a continuous gap ({earliest_kof}-{latest_kof})")

# Print summary of countries by regime category
print("\n[INFO] Countries by Regime Category")
print("=" * 80)
for category in sorted(merged['Category'].unique()):
    print(f"\n{category}")
    print("-" * len(category))
    category_countries = merged[merged['Category'] == category][['country_code', 'country_name']].drop_duplicates()
    for _, row in category_countries.sort_values('country_name').iterrows():
        country_name = vdem[vdem['country_code'] == row['country_code']]['country_name'].iloc[0] if not vdem[vdem['country_code'] == row['country_code']].empty else row['country_code']
        print(f"{country_name} ({row['country_code']})")

# STEP 7: Export data to Excel with formatting
merged = merged.sort_values(by=['Category', 'country_name', 'year'])
with pd.ExcelWriter("democracy_trade_analysis.xlsx", engine='xlsxwriter') as writer:
    # Define Excel formatting styles
    header_format = writer.book.add_format({
        'bold': True,
        'bg_color': '#D9D9D9',
        'border': 1
    })
    missing_data_format = writer.book.add_format({
        'bg_color': '#FFD9D9'  # Light red for missing data
    })
    separator_format = writer.book.add_format({
        'bg_color': '#E6F3FF'  # Light blue for separators
    })

    # Create separate sheets for each regime type
    for regime in merged['Category'].dropna().unique():
        df = merged[merged['Category'] == regime]
        
        # Add spacing between countries
        rows_with_spaces = []
        current_country = None
        is_separator = []
        
        for _, row in df.iterrows():
            if current_country != row['country_name'] and current_country is not None:
                rows_with_spaces.append([None] * len(df.columns))
                is_separator.append(True)
            rows_with_spaces.append(row.tolist())
            is_separator.append(False)
            current_country = row['country_name']

        # Write data to Excel with formatting
        df_with_spaces = pd.DataFrame(rows_with_spaces, columns=df.columns)
        df_with_spaces.to_excel(writer, sheet_name=regime[:31], index=False)
        
        worksheet = writer.sheets[regime[:31]]
        
        # Format headers
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Format data cells
        for row_num in range(1, len(df_with_spaces) + 1):
            for col_num in range(len(df.columns)):
                cell_value = df_with_spaces.iloc[row_num-1, col_num]
                if is_separator[row_num-1]:
                    worksheet.write(row_num, col_num, '', separator_format)
                elif pd.isna(cell_value):
                    worksheet.write(row_num, col_num, '', missing_data_format)
        
        # Auto-adjust column widths
        for idx, col in enumerate(df.columns):
            series = df[col]
            max_length = max(
                series.astype(str).apply(len).max(),
                len(str(series.name))
            ) + 2
            worksheet.set_column(idx, idx, max_length)

print("[INFO] Exported to 'democracy_trade_analysis.xlsx'")

# STEP 8: Create visualizations
print("\n[INFO] Generating charts...")

# Calculate overall min/max values for consistent scaling across plots
all_regime_data = []
for regime in sorted(merged['Regime_Type'].dropna().unique()):
    regime_data = merged[merged['Regime_Type'] == regime].groupby('year').agg({
        'v2x_polyarchy': 'mean',
        'KOFTrGIdf': 'mean'
    }).reset_index()
    all_regime_data.append(regime_data)

# Create individual regime plots
for regime, regime_data in zip(sorted(merged['Regime_Type'].dropna().unique()), all_regime_data):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    # Plot democracy score and trade openness
    line1 = ax1.plot(regime_data['year'], regime_data['v2x_polyarchy'],
                    color='blue', linewidth=2, label='Democracy Score')
    line2 = ax2.plot(regime_data['year'], regime_data['KOFTrGIdf'],
                    color='green', linewidth=2, label='Trade Openness')

    # Configure plot aesthetics
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Democracy Score', color='blue')
    ax2.set_ylabel('Trade Openness (KOF)', color='green')
    ax1.set_title(f'{regime}: Democracy vs Trade Openness')
    ax1.set_xlim(1970, 2020)
    ax1.grid(True, alpha=0.3)

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'plots/aggregate/democracy_vs_trade_{regime.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create combined plot of all regime types
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.ravel()

for idx, (regime, regime_data) in enumerate(zip(sorted(merged['Regime_Type'].dropna().unique()), all_regime_data)):
    ax1 = axes[idx]
    ax2 = ax1.twinx()

    # Plot democracy score and trade openness
    line1 = ax1.plot(regime_data['year'], regime_data['v2x_polyarchy'],
                    color='blue', linewidth=2, label='Democracy Score')
    line2 = ax2.plot(regime_data['year'], regime_data['KOFTrGIdf'],
                    color='green', linewidth=2, label='Trade Openness')

    # Configure plot aesthetics
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Democracy Score', color='blue')
    ax2.set_ylabel('Trade Openness (KOF)', color='green')
    ax1.set_title(regime)
    ax1.set_xlim(1970, 2020)
    ax1.grid(True, alpha=0.3)

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

plt.suptitle('Democracy vs Trade Openness by Regime Type (1970-2020)', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig('plots/aggregate/democracy_vs_trade_combined.png', dpi=300, bbox_inches='tight')
plt.close()

# Create individual country plots
for country_code in all_codes:
    country_data = merged[merged['country_code'] == country_code].sort_values('year')
    if country_data.empty:
        continue

    country_name = country_data['country_name'].iloc[0]
    regime_type = country_data['Regime_Type'].iloc[0]

    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot democracy score and trade openness
    line1 = ax1.plot(country_data['year'], country_data['v2x_polyarchy'],
                    color='blue', linewidth=2, label='Democracy Score')
    line2 = ax2.plot(country_data['year'], country_data['KOFTrGIdf'],
                    color='green', linewidth=2, label='Trade Openness')

    # Highlight missing data periods
    for year in range(1970, 2021):
        year_data = country_data[country_data['year'] == year]
        if year_data.empty or year_data['v2x_polyarchy'].isnull().any() or year_data['KOFTrGIdf'].isnull().any():
            ax1.axvspan(year-0.5, year+0.5, color='red', alpha=0.2)

    # Add regime type indicator line
    regime_colors = {
        'Liberal Democracy': '#2ecc71',
        'Electoral Democracy': '#3498db',
        'Electoral Autocracy': '#e74c3c',
        'Closed Autocracy': '#2c3e50'
    }

    # Plot regime changes over time
    y_pos = ax1.get_ylim()[0] - 0.05
    prev_regime = None
    start_year = None

    for year in range(1970, 2021):
        year_data = country_data[country_data['year'] == year]
        if not year_data.empty and pd.notnull(year_data['Regime_Type'].iloc[0]):
            current_regime = year_data['Regime_Type'].iloc[0]

            if current_regime != prev_regime:
                if prev_regime is not None:
                    ax1.hlines(y=y_pos, xmin=start_year, xmax=year,
                             colors=regime_colors.get(prev_regime, '#888888'),
                             linewidth=4)
                start_year = year
                prev_regime = current_regime

    # Draw final regime segment
    if prev_regime is not None:
        ax1.hlines(y=y_pos, xmin=start_year, xmax=2020,
                  colors=regime_colors.get(prev_regime, '#888888'),
                  linewidth=4)

    # Configure plot aesthetics
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Democracy Score', color='blue')
    ax2.set_ylabel('Trade Openness (KOF)', color='green')
    ax1.set_title(f'{country_name} ({regime_type}): Democracy vs Trade Openness')
    ax1.set_xlim(1970, 2020)
    ax1.grid(True, alpha=0.3)

    # Add comprehensive legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]

    # Add regime types to legend
    for regime_type, color in regime_colors.items():
        regime_patch = plt.Rectangle((0, 0), 1, 1, fc=color)
        lines.append(regime_patch)
        labels.append(regime_type)

    ax1.legend(lines, labels, loc='upper left', title='Legend',
               framealpha=0.7, prop={'size': 8}, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(f'plots/individual/democracy_vs_trade_{country_code}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("[INFO] Regime-level plot saved as 'plots/aggregate/democracy_vs_trade_by_regime.png'")
print("[INFO] Individual country plots saved in 'plots/individual' directory")
print("[INFO] Done!")