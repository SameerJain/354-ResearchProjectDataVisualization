import pandas as pd
import matplotlib.pyplot as plt
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)
os.makedirs('plots/aggregate', exist_ok=True)
os.makedirs('plots/individual', exist_ok=True)

# ---------------------------
# STEP 1: Load KOF Excel
# ---------------------------
kof = pd.read_excel('data/KOFGI_2024_public.xlsx',
                    usecols=['code', 'year', 'KOFTrGIdf'])
kof = kof.rename(columns={'code': 'country_code'})
kof['country_code'] = kof['country_code'].astype(str).str.strip()
kof = kof[kof['year'].between(1970, 2020)]

# ---------------------------
# STEP 2: Load V-Dem CSV
# ---------------------------
vdem = pd.read_csv('data/V-Dem-CY-Core-v15.csv',
                   usecols=[
                       'country_text_id', 'country_name', 'year', 'v2x_regime',
                       'v2x_polyarchy'
                   ])
vdem = vdem.rename(columns={'country_text_id': 'country_code'})
vdem['country_code'] = vdem['country_code'].astype(str).str.strip()
vdem = vdem[vdem['year'].between(1970, 2020)]

# Get all unique country codes from both datasets
all_codes = set(vdem['country_code'].unique()) & set(
    kof['country_code'].unique())

# ---------------------------
# STEP 3: Create complete 1970–2020 year set per country
# ---------------------------
full_years = pd.DataFrame([(c, y) for c in all_codes
                           for y in range(1970, 2021)],
                          columns=['country_code', 'year'])

# Merge V-Dem and KOF separately
vdem_clean = vdem[[
    'country_code', 'country_name', 'year', 'v2x_polyarchy', 'v2x_regime'
]]
kof_clean = kof[['country_code', 'year', 'KOFTrGIdf']]

merged = full_years \
    .merge(vdem_clean, on=['country_code', 'year'], how='left') \
    .merge(kof_clean, on=['country_code', 'year'], how='left')


# ---------------------------
# STEP 4: Detect and label data presence
# ---------------------------
def get_status(row):
    poly = pd.notnull(row['v2x_polyarchy'])
    kof = pd.notnull(row['KOFTrGIdf'])
    if poly and kof:
        return 'both'
    elif poly:
        return 'missing_kof'
    elif kof:
        return 'missing_polyarchy'
    else:
        return 'missing_both'


merged['data_status'] = merged.apply(get_status, axis=1)

# ---------------------------
# STEP 5: Add regime type and category based on most recent classification
# ---------------------------
# First map numeric codes to regime types
regime_mapping = {
    0: 'Closed Autocracy',
    1: 'Electoral Autocracy',
    2: 'Electoral Democracy',
    3: 'Liberal Democracy'
}

# Special country code mappings
special_country_mappings = {
    'AFG': 'Afghanistan',
    'ARE': 'United Arab Emirates',
    'UZB': 'Uzbekistan',
    'AZE': 'Azerbaijan',
    'KAZ': 'Kazakhstan',
    'MNE': 'Montenegro',
    'TJK': 'Tajikistan',
    'TKM': 'Turkmenistan',
    'LVA': 'Latvia',
    'EST': 'Estonia',
    'SVN': 'Slovenia',
    'SVK': 'Slovakia',
    'HRV': 'Croatia',
    'UKR': 'Ukraine',
    'MDA': 'Moldova',
    'MKD': 'North Macedonia',
    'LTU': 'Lithuania',
    'GEO': 'Georgia',
    'BIH': 'Bosnia and Herzegovina',
    'ARM': 'Armenia',
    'BLR': 'Belarus',
    'KGZ': 'Kyrgyzstan'
}

# Map the actual regime type for each year
merged['Regime_Type'] = merged['v2x_regime'].map(regime_mapping)

# Apply special country name mappings
merged.loc[merged['country_code'].isin(special_country_mappings.keys()), 'country_name'] = \
    merged.loc[merged['country_code'].isin(special_country_mappings.keys()), 'country_code'].map(special_country_mappings)


# Get the most recent regime type for categorization
def get_most_recent_regime(country_data):
    # Look backwards from 2020 until we find a valid regime type
    for year in range(2020, 1969, -1):
        year_data = country_data[country_data['year'] == year]
        if not year_data.empty and pd.notnull(year_data['v2x_regime'].iloc[0]):
            return regime_mapping.get(year_data['v2x_regime'].iloc[0])
    return 'Uncategorized'  # If no regime type found


# Apply categorization
categories = {}
for country in merged['country_code'].unique():
    country_data = merged[merged['country_code'] == country].sort_values(
        'year', ascending=False)
    categories[country] = get_most_recent_regime(country_data)

merged['Category'] = merged['country_code'].map(categories)

# ---------------------------
# STEP 6: Check for missing years and data quality
# ---------------------------
print("\n[DATA QUALITY REPORT]")
print("=" * 80)

# Get country names from V-Dem dataset
#country_names = vdem[['country_code', 'country_name']].drop_duplicates().set_index('country_code')['country_name'].to_dict()

for code in sorted(all_codes):
    sub = merged[merged['country_code'] == code]
    #country_name = country_names.get(code, code)
    print("\n" + "-" * 80)
    print(f"Country: {code} ({code})")  #Temporary fix, replaced below
    print("-" * 80)

    # Get specific years with missing data
    poly_missing_years = sub[sub['v2x_polyarchy'].isnull()]['year'].tolist()
    kof_missing_years = sub[sub['KOFTrGIdf'].isnull()]['year'].tolist()

    # Calculate coverage percentages
    total_years = len(sub)
    poly_coverage = (
        (total_years - len(poly_missing_years)) / total_years) * 100
    kof_coverage = ((total_years - len(kof_missing_years)) / total_years) * 100

    if poly_missing_years or kof_missing_years:
        country_name = vdem[vdem['country_code'] == code]['country_name'].iloc[
            0] if not vdem[vdem['country_code'] == code].empty else code
        print(f"\n[WARNING] Data quality issues for {country_name} ({code})")
        print("-" * 80)
        print(f"Data Coverage:")
        print(
            f"  Democracy Score: {poly_coverage:.1f}% ({len(poly_missing_years)} missing years)"
        )
        print(
            f"  Trade Openness: {kof_coverage:.1f}% ({len(kof_missing_years)} missing years)"
        )

        if len(poly_missing_years) == total_years:
            print(f"  ⚠️ NO democracy score data available for this country")
        elif poly_missing_years:
            print(
                f"  Missing democracy score data for years: {sorted(poly_missing_years)}"
            )

        if len(kof_missing_years) == total_years:
            print(f"  ⚠️ NO trade openness data available for this country")
        elif kof_missing_years:
            print(
                f"  Missing trade openness data for years: {sorted(kof_missing_years)}"
            )

        # Identify patterns in missing data
        if len(poly_missing_years) > 0:
            earliest_poly = min(
                poly_missing_years) if poly_missing_years else None
            latest_poly = max(
                poly_missing_years) if poly_missing_years else None
            if earliest_poly and latest_poly:
                if latest_poly - earliest_poly + 1 == len(poly_missing_years):
                    print(
                        f"  Note: Missing democracy data appears to be a continuous gap ({earliest_poly}-{latest_poly})"
                    )

        if len(kof_missing_years) > 0:
            earliest_kof = min(
                kof_missing_years) if kof_missing_years else None
            latest_kof = max(kof_missing_years) if kof_missing_years else None
            if earliest_kof and latest_kof:
                if latest_kof - earliest_kof + 1 == len(kof_missing_years):
                    print(
                        f"  Note: Missing trade data appears to be a continuous gap ({earliest_kof}-{latest_kof})"
                    )

print("\n[INFO] Countries by Regime Category")
print("=" * 80)
for category in sorted(merged['Category'].unique()):
    print(f"\n{category}")
    print("-" * len(category))
    category_countries = merged[merged['Category'] == category][[
        'country_code', 'country_name'
    ]].drop_duplicates()
    for _, row in category_countries.sort_values('country_name').iterrows():
        country_name = vdem[
            vdem['country_code'] ==
            row['country_code']]['country_name'].iloc[0] if not vdem[
                vdem['country_code'] ==
                row['country_code']].empty else row['country_code']
        print(f"{country_name} ({row['country_code']})")

# ---------------------------
# STEP 7: Sort and Export to Excel
# ---------------------------
merged = merged.sort_values(by=['Category', 'country_name', 'year'])
with pd.ExcelWriter("data/democracy_trade_analysis.xlsx",
                    engine='xlsxwriter') as writer:
    # Create formats
    header_format = writer.book.add_format({
        'bold': True,
        'bg_color': '#D9D9D9',
        'border': 1
    })
    missing_data_format = writer.book.add_format({
        'bg_color': '#FFD9D9'  # Light red
    })
    separator_format = writer.book.add_format({
        'bg_color': '#E6F3FF'  # Light blue
    })

    for regime in merged['Category'].dropna().unique():
        df = merged[merged['Category'] == regime]

        rows_with_spaces = []
        current_country = None
        is_separator = []

        for _, row in df.iterrows():
            if current_country != row[
                    'country_name'] and current_country is not None:
                rows_with_spaces.append([None] * len(df.columns))
                is_separator.append(True)
            rows_with_spaces.append(row.tolist())
            is_separator.append(False)
            current_country = row['country_name']

        df_with_spaces = pd.DataFrame(rows_with_spaces, columns=df.columns)
        df_with_spaces.to_excel(writer, sheet_name=regime[:31], index=False)

        worksheet = writer.sheets[regime[:31]]

        # Format headers
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # Format data cells and highlight missing data
        for row_num in range(1, len(df_with_spaces) + 1):
            for col_num in range(len(df.columns)):
                cell_value = df_with_spaces.iloc[row_num - 1, col_num]
                if is_separator[row_num - 1]:
                    worksheet.write(row_num, col_num, '', separator_format)
                elif pd.isna(cell_value):
                    worksheet.write(row_num, col_num, '', missing_data_format)

        # Auto-adjust column widths
        for idx, col in enumerate(df.columns):
            series = df[col]
            max_length = max(
                series.astype(str).apply(len).max(), len(str(series.name))) + 2
            worksheet.set_column(idx, idx, max_length)

print("[INFO] Exported to 'democracy_trade_analysis.xlsx'")

# ---------------------------
# STEP 8: Create Visualizations
# ---------------------------
print("\n[INFO] Generating charts...")
# Calculate overall min/max values for consistent scaling
all_regime_data = []
for regime in sorted(merged['Regime_Type'].dropna().unique()):
    regime_data = merged[merged['Regime_Type'] == regime].groupby('year').agg({
        'v2x_polyarchy':
        'mean',
        'KOFTrGIdf':
        'mean'
    }).reset_index()
    all_regime_data.append(regime_data)

y1_min = min(df['v2x_polyarchy'].min() for df in all_regime_data)
y1_max = max(df['v2x_polyarchy'].max() for df in all_regime_data)
y2_min = min(df['KOFTrGIdf'].min() for df in all_regime_data)
y2_max = max(df['KOFTrGIdf'].max() for df in all_regime_data)

# Create individual regime plots for aggregate folder
for regime, regime_data in zip(sorted(merged['Regime_Type'].dropna().unique()),
                               all_regime_data):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    line1 = ax1.plot(regime_data['year'],
                     regime_data['v2x_polyarchy'],
                     color='blue',
                     linewidth=2,
                     label='Democracy Score')
    line2 = ax2.plot(regime_data['year'],
                     regime_data['KOFTrGIdf'],
                     color='green',
                     linewidth=2,
                     label='Trade Openness')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Democracy Score', color='blue')
    ax2.set_ylabel('Trade Openness (KOF)', color='green')
    ax1.set_title(f'{regime}: Democracy vs Trade Openness')

    ax1.set_xlim(1970, 2020)
    ax1.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(
        f'plots/aggregate/democracy_vs_trade_{regime.replace(" ", "_")}.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

# Create combined plot
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.ravel()

for idx, (regime, regime_data) in enumerate(
        zip(sorted(merged['Regime_Type'].dropna().unique()), all_regime_data)):
    ax1 = axes[idx]
    ax2 = ax1.twinx()

    line1 = ax1.plot(regime_data['year'],
                     regime_data['v2x_polyarchy'],
                     color='blue',
                     linewidth=2,
                     label='Democracy Score')
    line2 = ax2.plot(regime_data['year'],
                     regime_data['KOFTrGIdf'],
                     color='green',
                     linewidth=2,
                     label='Trade Openness')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Democracy Score', color='blue')
    ax2.set_ylabel('Trade Openness (KOF)', color='green')
    ax1.set_title(regime)

    ax1.set_xlim(1970, 2020)
    ax1.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

plt.suptitle('Democracy vs Trade Openness by Regime Type (1970-2020)',
             y=1.02,
             fontsize=16)
plt.tight_layout()
plt.savefig('plots/aggregate/democracy_vs_trade_combined.png',
            dpi=300,
            bbox_inches='tight')
plt.close()

# Individual country plots
for country_code in all_codes:
    country_data = merged[merged['country_code'] == country_code].sort_values(
        'year')
    if country_data.empty:
        continue

    country_name = country_data['country_name'].iloc[0]
    regime_type = country_data['Regime_Type'].iloc[0]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    line1 = ax1.plot(country_data['year'],
                     country_data['v2x_polyarchy'],
                     color='blue',
                     linewidth=2,
                     label='Democracy Score')
    line2 = ax2.plot(country_data['year'],
                     country_data['KOFTrGIdf'],
                     color='green',
                     linewidth=2,
                     label='Trade Openness')

    for year in range(1970, 2021):
        year_data = country_data[country_data['year'] == year]
        if year_data.empty or year_data['v2x_polyarchy'].isnull().any(
        ) or year_data['KOFTrGIdf'].isnull().any():
            ax1.axvspan(year - 0.5, year + 0.5, color='red', alpha=0.2)

    # Add regime type indicator line
    regime_colors = {
        'Liberal Democracy': '#2ecc71',
        'Electoral Democracy': '#3498db',
        'Electoral Autocracy': '#e74c3c',
        'Closed Autocracy': '#2c3e50'
    }

    # Get year-by-year regime types
    y_pos = ax1.get_ylim()[0] - 0.05  # Position slightly below x-axis
    prev_regime = None
    start_year = None

    for year in range(1970, 2021):
        year_data = country_data[country_data['year'] == year]
        if not year_data.empty and pd.notnull(
                year_data['Regime_Type'].iloc[0]):
            current_regime = year_data['Regime_Type'].iloc[0]

            if current_regime != prev_regime:
                if prev_regime is not None:
                    ax1.hlines(y=y_pos,
                               xmin=start_year,
                               xmax=year,
                               colors=regime_colors.get(
                                   prev_regime, '#888888'),
                               linewidth=4)
                start_year = year
                prev_regime = current_regime

    # Draw final segment
    if prev_regime is not None:
        ax1.hlines(y=y_pos,
                   xmin=start_year,
                   xmax=2020,
                   colors=regime_colors.get(prev_regime, '#888888'),
                   linewidth=4)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Democracy Score', color='blue')
    ax2.set_ylabel('Trade Openness (KOF)', color='green')
    ax1.set_title(
        f'{country_name} ({regime_type}): Democracy vs Trade Openness')

    ax1.set_xlim(1970, 2020)
    ax1.grid(True, alpha=0.3)

    # Add legend with all information
    lines = line1 + line2
    labels = [l.get_label() for l in lines]

    # Add all regime types to legend
    for regime_type, color in regime_colors.items():
        regime_patch = plt.Rectangle((0, 0), 1, 1, fc=color)
        lines.append(regime_patch)
        labels.append(regime_type)

    ax1.legend(lines,
               labels,
               loc='upper left',
               title='Legend',
               framealpha=0.7,
               prop={'size': 8},
               title_fontsize=9)

    plt.tight_layout()
    plt.savefig(f'plots/individual/democracy_vs_trade_{country_code}.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()

print(
    "[INFO] Regime-level plot saved as 'plots/aggregate/democracy_vs_trade_by_regime.png'"
)
print("[INFO] Individual country plots saved in 'plots/individual' directory")
print("[INFO] Done!")
