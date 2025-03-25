
"""
Democracy vs Trade Openness Analysis Dashboard
-------------------------------------------
This Dash application visualizes the relationship between democracy scores and trade openness
across different countries and regime categories from 1970-2020.

Key Features:
- Interactive country selection dropdown
- Individual country analysis with regime type indicators
- Aggregate analysis by regime category
- Missing data visualization
- Responsive dark theme design

Data Sources:
- V-Dem Dataset: Democracy scores and regime classifications
- KOF Dataset: Trade openness metrics
"""

# Import required libraries
import dash  # Core Dash library for web applications
import dash_bootstrap_components as dbc  # Bootstrap components for styling
from dash import dcc, html, Input, Output  # Dash components and callback decorators
import pandas as pd  # Data manipulation library
import plotly.graph_objects as go  # Plotting library
from plotly.subplots import make_subplots  # For creating subplots

# Load and prepare the dataset
excel_file = pd.ExcelFile('democracy_trade_analysis.xlsx')  # Load Excel file
# Combine all sheets into a single DataFrame
df = pd.concat([pd.read_excel(excel_file, sheet_name=sheet) 
                for sheet in excel_file.sheet_names], ignore_index=True)
# Extract unique country names and categories for dropdown and plotting
countries = sorted(df['country_name'].dropna().unique())
categories = sorted(df['Category'].dropna().unique())

# Initialize Dash application with dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Define application layout using Bootstrap components
app.layout = dbc.Container([
    html.Div(style={
        'backgroundColor': '#0a192f',  # Dark blue background
        'padding': '20px',
        'minHeight': '100vh'  # Full viewport height
    }, children=[
        # Application title
        html.H1("Democracy vs. Trade Openness Research - PSC354 Group 9", 
                className="text-center my-4",
                style={
                    'color': '#64ffda',  # Teal text color
                    'fontFamily': 'Helvetica, Arial, sans-serif',
                    'letterSpacing': '1px'
                }),

        # Main content row with two columns
        dbc.Row([
            # Left column: Country-specific analysis
            dbc.Col([
                # Country selection dropdown card
                dbc.Card([
                    dbc.CardBody([
                        html.Div(style={
                            'backgroundColor': '#172a45',
                            'padding': '15px',
                            'borderRadius': '5px'
                        }, children=[
                            html.H4("Select Country"),
                            dcc.Dropdown(
                                id='country-dropdown',
                                options=[{'label': c, 'value': c} for c in countries],
                                value=countries[0],  # Default to first country
                                style={
                                    'backgroundColor': '#1a365d',
                                    'color': 'white'
                                },
                                className='dropdown-dark',
                            )
                        ])
                    ])
                ], className="mb-3"),
                # Country-specific graph card
                dbc.Card([
                    dbc.CardBody([
                        html.Div(style={
                            'backgroundColor': '#172a45',
                            'padding': '15px',
                            'borderRadius': '5px'
                        }, children=[
                            dcc.Graph(id='country-graph')
                        ])
                    ])
                ])
            ], width=6),

            # Right column: Category analysis
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(style={
                            'backgroundColor': '#172a45',
                            'padding': '15px',
                            'borderRadius': '5px'
                        }, children=[
                            dcc.Graph(id='category-graph')
                        ])
                    ])
                ])
            ], width=6)
        ])
    ])
], fluid=True, style={'backgroundColor': '#0a192f', 'padding': '0'})

# Callback for updating country-specific graph
@app.callback(
    Output('country-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_country_graph(country):
    """
    Updates the country-specific graph based on dropdown selection.
    
    Parameters:
    - country (str): Selected country name from dropdown
    
    Returns:
    - plotly.graph_objects.Figure: Updated figure with country data
    """
    # Filter data for selected country
    data = df[df['country_name'] == country]

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Identify and track missing data ranges
    missing_ranges = []
    start_year = None

    # Scan through years to find continuous ranges of missing data
    for year in range(1970, 2021):
        year_data = data[data['year'] == year]
        is_missing = (year_data.empty or 
                     year_data['v2x_polyarchy'].isnull().any() or 
                     year_data['KOFTrGIdf'].isnull().any())

        if is_missing and start_year is None:
            start_year = year
        elif not is_missing and start_year is not None:
            missing_ranges.append((start_year, year))
            start_year = None

    # Handle case where missing data extends to end of range
    if start_year is not None:
        missing_ranges.append((start_year, 2021))

    # Add visual indicators for missing data periods
    for start, end in missing_ranges:
        fig.add_vrect(
            x0=start-0.5,
            x1=end-0.5,
            fillcolor="rgba(255, 0, 0, 0.15)",  # Light red
            layer="below",
            line_width=0,
            name="Missing Data"
        )

    # Add democracy score trace
    fig.add_trace(
        go.Scatter(
            x=data['year'],
            y=data['v2x_polyarchy'],
            name="Democracy Score",
            line=dict(color='#00bfff')  # Light blue
        ),
        secondary_y=False
    )

    # Add trade openness trace
    fig.add_trace(
        go.Scatter(
            x=data['year'],
            y=data['KOFTrGIdf'],
            name="Trade Openness",
            line=dict(color='green')
        ),
        secondary_y=True
    )

    # Define colors for different regime types
    regime_colors = {
        'Liberal Democracy': '#2ecc71',     # Green
        'Electoral Democracy': '#3498db',    # Blue
        'Electoral Autocracy': '#e74c3c',    # Red
        'Closed Autocracy': '#2c3e50'       # Dark blue
    }

    # Add regime type indicators
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

    # Update layout with styling
    fig.update_layout(
        title=dict(
            text=f"{country}: Democracy and Trade Analysis",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        template="plotly_dark",
        paper_bgcolor='#172a45',
        plot_bgcolor='#172a45',
        font=dict(color='#ffffff'),
        height=500,
        margin=dict(t=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

# Callback for updating category analysis graph
@app.callback(
    Output('category-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_category_graph(_):
    """
    Updates the regime category analysis graph.
    This graph shows aggregate trends for each regime category.
    
    Parameters:
    - _ (any): Unused parameter (maintains callback context)
    
    Returns:
    - plotly.graph_objects.Figure: Updated figure with category data
    """
    # Create 2x2 subplot grid for each regime category
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=categories,
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )

    # Generate plots for each category
    for idx, category in enumerate(categories, 1):
        # Calculate average scores per year for the category
        category_data = df[df['Category'] == category].groupby('year').agg({
            'v2x_polyarchy': 'mean',
            'KOFTrGIdf': 'mean'
        }).reset_index()

        # Calculate subplot position
        row = (idx - 1) // 2 + 1
        col = (idx - 1) % 2 + 1

        # Add democracy score trace
        fig.add_trace(
            go.Scatter(
                x=category_data['year'],
                y=category_data['v2x_polyarchy'],
                name="Democracy Score" if idx == 1 else None,  # Legend only for first subplot
                line=dict(color='#00bfff'),
                showlegend=(idx == 1)
            ),
            row=row, col=col, secondary_y=False
        )

        # Add trade openness trace
        fig.add_trace(
            go.Scatter(
                x=category_data['year'],
                y=category_data['KOFTrGIdf'],
                name="Trade Openness" if idx == 1 else None,  # Legend only for first subplot
                line=dict(color='green'),
                showlegend=(idx == 1)
            ),
            row=row, col=col, secondary_y=True
        )

    # Update layout with styling
    fig.update_layout(
        height=600,
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

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0')  # Make accessible on all network interfaces
