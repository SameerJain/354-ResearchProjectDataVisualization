'''
equation for the correlation coefficient: https://numpy.org/doc/2.2/reference/generated/numpy.corrcoef.html
slope is calculated by polyfit function, then poly1d turns it into a callable function: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html

'''

'''
1. IMPORT necessary libraries (dash, pandas, numpy, plotly)

2. LOAD DATA
   - Read Excel file with democracy and trade data
   - Extract unique countries and regime categories
   - Create dropdown options list combining categories and countries

3. INITIALIZE Dash app with dark theme

4. DEFINE app layout
   - Create container with dark blue background
   - Add header
   - Create row with two columns
     - Left column: Country dropdown and time series graph
     - Right column: Category comparison graph
   - Create second row
     - Full-width correlation scatter plot

5. DEFINE correlation scatter plot callback:
   - INPUT: selected country/category from dropdown
   - FUNCTION update_scatter():
     - IF selection is a category average
       - Filter data for that category
       - Calculate means by year
     - ELSE
       - Filter data for selected country
     - Create scatter plot with points colored by year
     - Calculate trend line coefficients using linear regression
     - Calculate correlation coefficient
     - Add trend line to plot
     - Update layout with titles and styling
     - RETURN figure

6. DEFINE country time series callback:
   - INPUT: selected country/category from dropdown
   - FUNCTION update_country_graph():
     - Filter data similar to previous callback
     - Create figure with dual y-axes
     - Identify and highlight missing data periods
     - Add democracy score line (primary y-axis)
     - Add trade openness line (secondary y-axis)
     - Add regime type indicators at bottom
     - Update layout
     - RETURN figure

7. DEFINE category comparison callback:
   - INPUT: selected country (not used)
   - FUNCTION update_category_graph():
     - Create 2×2 grid of subplots
     - FOR each regime category:
       - Calculate averages by year
       - Determine subplot position
       - Add democracy score line
       - Add trade openness line
     - Update layout
     - RETURN figure

8. RUN the server
'''
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------
# DATA LOADING AND PREPARATION
# ----------------------

def load_data():
    """Load and prepare data from Excel file."""
    excel_file = pd.ExcelFile('data/democracy_trade_analysis.xlsx')
    df = pd.concat([
        pd.read_excel(excel_file, sheet_name=sheet)
        for sheet in excel_file.sheet_names
    ], ignore_index=True)
    return df


def prepare_dropdown_options(df):
    """Create dropdown options with global and category averages at top."""
    countries = sorted(df['country_name'].dropna().unique())
    categories = sorted(df['Category'].dropna().unique())

    dropdown_options = [
        {'label': '[Global Average] All Countries', 'value': 'global_average'}
    ] + [
        {'label': f'[Average] {category}', 'value': f'avg_{category}'}
        for category in categories
    ] + [
        {'label': country, 'value': country}
        for country in countries
    ]

    return dropdown_options, categories


# ----------------------
# DATA FILTERING
# ----------------------

def filter_data(df, selected):
    """Filter data based on dropdown selection."""
    if selected == 'global_average':
        # Handle global average
        data = df.groupby('year').agg({
            'v2x_polyarchy': 'mean',
            'KOFTrGIdf': 'mean',
            'Regime_Type': lambda x: x.mode()[0] if not x.empty else None
        }).reset_index()
        title = "Global Average"
    elif selected.startswith('avg_'):
        # Handle category average
        category = selected[4:]  # Remove 'avg_' prefix
        data = df[df['Category'] == category].groupby('year').agg({
            'v2x_polyarchy': 'mean',
            'KOFTrGIdf': 'mean',
            'Regime_Type': lambda x: x.mode()[0] if not x.empty else None
        }).reset_index()
        title = f"Average: {category}"
    else:
        # Handle individual country
        data = df[df['country_name'] == selected]
        title = selected

    return data, title


# ----------------------
# VISUALIZATION FUNCTIONS
# ----------------------

def create_correlation_plot(data, title):
    """Create scatter plot with correlation analysis."""
    fig = go.Figure()

    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=data['v2x_polyarchy'],
            y=data['KOFTrGIdf'],
            mode='markers+text',
            marker=dict(
                size=10,
                color=data['year'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Year")
            ),
            text=data['year'],
            textposition="top center",
            name='Years'
        )
    )

    # Add trend line and calculate statistics
    coefficients = np.polyfit(data['v2x_polyarchy'], data['KOFTrGIdf'], 1)
    slope = coefficients[0]
    polynomial = np.poly1d(coefficients)
    x_range = np.linspace(data['v2x_polyarchy'].min(), data['v2x_polyarchy'].max(), 100)

    # Calculate correlation coefficient
    correlation = np.corrcoef(data['v2x_polyarchy'], data['KOFTrGIdf'])[0, 1]

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=polynomial(x_range),
            mode='lines',
            name=f'R: {correlation:.2f}, Slope: {slope:.2f}',
            line=dict(color='red', dash='dash'),
            hovertemplate='Slope: %{customdata[0]:.2f}, R: %{customdata[1]:.2f}<extra></extra>',
            customdata=np.column_stack((np.full(len(x_range), slope), np.full(len(x_range), correlation)))
        )
    )

    # Update layout
    fig.update_layout(
        title=f'{title}: Democracy vs Trade Correlation',
        xaxis_title='Democracy Score',
        yaxis_title='Trade Openness',
        template="plotly_dark",
        paper_bgcolor='#172a45',
        plot_bgcolor='#172a45',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)  # Add more bottom margin
    )

    return fig


def find_missing_data_ranges(data):
    """Identify continuous ranges of missing data."""
    missing_ranges = []
    start_year = None

    for year in range(1970, 2021):
        year_data = data[data['year'] == year]
        is_missing = (
            year_data.empty or
            year_data['v2x_polyarchy'].isnull().any() or
            year_data['KOFTrGIdf'].isnull().any()
        )

        if is_missing and start_year is None:
            start_year = year
        elif not is_missing and start_year is not None:
            missing_ranges.append((start_year, year))
            start_year = None

    if start_year is not None:
        missing_ranges.append((start_year, 2021))

    return missing_ranges


def create_time_series(data, title):
    """Create time series with dual y-axes and regime indicators."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Highlight missing data
    missing_ranges = find_missing_data_ranges(data)
    for start, end in missing_ranges:
        fig.add_vrect(
            x0=start - 0.5,
            x1=end - 0.5,
            fillcolor="rgba(255, 0, 0, 0.15)",
            layer="below",
            line_width=0,
            name="Missing Data"
        )

    # Add democracy score line
    fig.add_trace(
        go.Scatter(
            x=data['year'],
            y=data['v2x_polyarchy'],
            name="Democracy Score",
            line=dict(color='#00bfff')
        ),
        secondary_y=False
    )

    # Add trade openness line
    fig.add_trace(
        go.Scatter(
            x=data['year'],
            y=data['KOFTrGIdf'],
            name="Trade Openness",
            line=dict(color='green')
        ),
        secondary_y=True
    )

    # Add regime indicators only for individual country views
    if not title.startswith('Average:') and title != "Global Average":
        regime_colors = {
            'Liberal Democracy': '#2ecc71',
            'Electoral Democracy': '#3498db',
            'Electoral Autocracy': '#e74c3c',
            'Closed Autocracy': '#2c3e50'
        }

        for regime_type in regime_colors:
            regime_years = data[data['Regime_Type'] == regime_type]['year']
            if not regime_years.empty:
                fig.add_trace(
                    go.Scatter(
                        x=regime_years,
                        y=[0] * len(regime_years),
                        name=regime_type,
                        mode='lines',
                        line=dict(color=regime_colors[regime_type], width=15),
                        showlegend=True
                    ),
                    secondary_y=False
                )

    # Update layout
    fig.update_layout(
        title=f"{title}: Democracy and Trade Analysis",
        template="plotly_dark",
        paper_bgcolor='#172a45',
        plot_bgcolor='#172a45',
        font=dict(color='#ffffff'),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)
    )

    return fig


def create_category_overview(df, categories):
    """Create 2x2 grid showing time series for all categories."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=categories,
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}]
        ]
    )

    for idx, category in enumerate(categories, 1):
        category_data = df[df['Category'] == category].groupby('year').agg({
            'v2x_polyarchy': 'mean',
            'KOFTrGIdf': 'mean'
        }).reset_index()

        row = (idx - 1) // 2 + 1
        col = (idx - 1) % 2 + 1

        # Add democracy score line
        fig.add_trace(
            go.Scatter(
                x=category_data['year'],
                y=category_data['v2x_polyarchy'],
                name="Democracy Score" if idx == 1 else None,
                line=dict(color='#00bfff'),
                showlegend=(idx == 1)
            ),
            row=row, col=col, secondary_y=False
        )

        # Add trade openness line
        fig.add_trace(
            go.Scatter(
                x=category_data['year'],
                y=category_data['KOFTrGIdf'],
                name="Trade Openness" if idx == 1 else None,
                line=dict(color='green'),
                showlegend=(idx == 1)
            ),
            row=row, col=col, secondary_y=True
        )

    # Update layout
    fig.update_layout(
        height=652,
        showlegend=True,
        template="plotly_dark",
        paper_bgcolor='#172a45',
        plot_bgcolor='#172a45',
        font=dict(color='#ffffff'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )

    return fig


# ----------------------
# APP INITIALIZATION
# ----------------------

# Load data
df = load_data()
dropdown_options, categories = prepare_dropdown_options(df)

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# ----------------------
# APP LAYOUT
# ----------------------

app.layout = dbc.Container([
    html.Div(
        style={'backgroundColor': '#0a192f', 'padding': '20px', 'minHeight': '100vh'},
        children=[
            # Header
            html.H1(
                "Democracy vs. Trade Openness Research - PSC354 Group 9",
                className="text-center my-4",
                style={
                    'color': '#64ffda',
                    'fontFamily': 'Helvetica, Arial, sans-serif',
                    'letterSpacing': '1px'
                }
            ),

            # Top row with country/category selection and graphs
            dbc.Row([
                # Left column
                dbc.Col([
                    # Dropdown card
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(
                                style={
                                    'backgroundColor': '#172a45',
                                    'padding': '15px',
                                    'borderRadius': '5px'
                                },
                                children=[
                                    html.H4("Select Country/Category"),
                                    dcc.Dropdown(
                                        id='country-dropdown',
                                        options=dropdown_options,
                                        value='global_average',
                                        style={
                                            'backgroundColor': '#1a365d',
                                            'color': 'white'
                                        },
                                        className='dropdown-dark',
                                    )
                                ]
                            )
                        ])
                    ], className="mb-3"),

                    # Country graph card
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(
                                style={
                                    'backgroundColor': '#172a45',
                                    'padding': '15px',
                                    'borderRadius': '5px'
                                },
                                children=[
                                    dcc.Graph(id='country-graph')
                                ]
                            )
                        ])
                    ])
                ], width=6),

                # Right column
                dbc.Col([
                    # Category graph card
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(
                                style={
                                    'backgroundColor': '#172a45',
                                    'padding': '15px',
                                    'borderRadius': '5px'
                                },
                                children=[
                                    dcc.Graph(id='category-graph')
                                ]
                            )
                        ])
                    ])
                ], width=6)
            ]),

            # Bottom row with correlation scatter plot
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(
                                style={
                                    'backgroundColor': '#172a45',
                                    'padding': '15px',
                                    'borderRadius': '5px'
                                },
                                children=[
                                    dcc.Graph(id='correlation-scatter')
                                ]
                            )
                        ])
                    ])
                ], width=12)
            ], className="mt-3")
        ]
    )
], fluid=True, style={'backgroundColor': '#0a192f', 'padding': '0'})


# ----------------------
# CALLBACKS
# ----------------------

@app.callback(
    Output('correlation-scatter', 'figure'),
    Input('country-dropdown', 'value')
)
def update_scatter(selected):
    data, title = filter_data(df, selected)
    data = data.dropna(subset=['v2x_polyarchy', 'KOFTrGIdf'])
    return create_correlation_plot(data, title)


@app.callback(
    Output('country-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_country_graph(selected):
    data, title = filter_data(df, selected)
    return create_time_series(data, title)


@app.callback(
    Output('category-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_category_graph(_):
    return create_category_overview(df, categories)


# ----------------------
# RUN APP
# ----------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0')