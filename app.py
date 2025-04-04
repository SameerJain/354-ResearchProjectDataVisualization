import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
REGIME_COLORS = {
    'Liberal Democracy': '#2ecc71',
    'Electoral Democracy': '#3498db',
    'Electoral Autocracy': '#e74c3c',
    'Closed Autocracy': '#8e44ad'
}

PLOT_STYLE = {
    'template': "plotly_dark",
    'paper_bgcolor': '#172a45',
    'plot_bgcolor': '#172a45',
    'height': 500,
    'legend': {
        'orientation': "h",
        'yanchor': "bottom",
        'y': -0.3,
        'xanchor': "center",
        'x': 0.5
    },
    'margin': {'b': 100}
}

def load_data():
    """Load and prepare data from Excel file."""
    df = pd.concat([
        pd.read_excel('data/democracy_trade_analysis.xlsx', sheet_name=sheet)
        for sheet in pd.ExcelFile('data/democracy_trade_analysis.xlsx').sheet_names
    ])
    return df.dropna(subset=['country_name', 'Category'])

def create_dropdown_options(df):
    """Create dropdown options with special entries at top."""
    countries = sorted(df['country_name'].dropna().unique())
    categories = sorted(df['Category'].dropna().unique())

    return ([
        {'label': '[Global Average] All Countries', 'value': 'global_average'},
        {'label': '[Comparison] Trade by Regime Type', 'value': 'regime_comparison'}
    ] + [
        {'label': f'[Average] {category}', 'value': f'avg_{category}'}
        for category in categories
    ] + [
        {'label': country, 'value': country}
        for country in countries
    ], categories)

def get_filtered_data(df, selected):
    """Filter data based on selection and return with appropriate title."""
    if selected == 'global_average':
        data = df.groupby('year').agg({
            'v2x_polyarchy': 'mean',
            'KOFEcGI': 'mean',
            'Regime_Type': lambda x: x.mode().iloc[0] if not x.empty else None
        }).reset_index()
        title = "Global Average"
    elif selected.startswith('avg_'):
        category = selected[4:]
        data = df[df['Regime_Type'] == category].groupby('year').agg({
            'v2x_polyarchy': 'mean',
            'KOFEcGI': 'mean',
            'Regime_Type': lambda x: x.mode().iloc[0] if not x.empty else None
        }).reset_index()
        title = f"Average: {category}"
    elif selected == 'regime_comparison':
        data = df.groupby(['year', 'Regime_Type'])['KOFEcGI'].mean().reset_index()
        title = "Trade Openness by Regime Type"
    else:
        data = df[df['country_name'] == selected]
        title = selected

    return data, title

def create_correlation_plot(data, title):
    """Create scatter plot with correlation analysis."""
    valid_data = data.dropna(subset=['v2x_polyarchy', 'KOFEcGI'])

    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=valid_data['v2x_polyarchy'],
        y=valid_data['KOFEcGI'],
        mode='markers+text',
        marker=dict(
            size=10,
            color=valid_data['year'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Year")
        ),
        text=valid_data['year'],
        textposition="top center",
        name='Years'
    ))

    # Add trend line
    if len(valid_data) > 1:
        coefficients = np.polyfit(valid_data['v2x_polyarchy'], valid_data['KOFEcGI'], 1)
        correlation = np.corrcoef(valid_data['v2x_polyarchy'], valid_data['KOFEcGI'])[0, 1]
        x_range = np.linspace(valid_data['v2x_polyarchy'].min(), valid_data['v2x_polyarchy'].max(), 100)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=coefficients[0] * x_range + coefficients[1],
            mode='lines',
            name=f'Correlation: {correlation:.2f}',
            line=dict(color='red', dash='dash'),
            hovertemplate='R = %{customdata:.2f}<extra></extra>',
            customdata=np.full(len(x_range), correlation)
        ))

    fig.update_layout(
        title=f'{title}: Democracy vs Trade Correlation',
        xaxis_title='Democracy Score',
        yaxis_title='Trade Openness',
        **PLOT_STYLE
    )

    return fig

def create_time_series(data, title):
    """Create time series plot with appropriate view based on selection."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if title == "Trade Openness by Regime Type":
        for regime in sorted(REGIME_COLORS.keys()):
            regime_data = data[data['Regime_Type'] == regime]
            if not regime_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=regime_data['year'],
                        y=regime_data['KOFEcGI'],
                        name=regime,
                        line=dict(color=REGIME_COLORS[regime])
                    ),
                    secondary_y=False
                )
    else:
        fig.add_trace(
            go.Scatter(
                x=data['year'],
                y=data['v2x_polyarchy'],
                name="Democracy Score",
                line=dict(color='#00bfff')
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=data['year'],
                y=data['KOFEcGI'],
                name="Trade Openness",
                line=dict(color='green')
            ),
            secondary_y=True
        )

        # Add regime indicators for individual countries
        if not any(x in title for x in ['Average:', 'Global Average', 'Trade Openness by Regime Type']):
            for regime_type, color in REGIME_COLORS.items():
                regime_years = data[data['Regime_Type'] == regime_type]['year']
                if not regime_years.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=regime_years,
                            y=[0] * len(regime_years),
                            name=regime_type,
                            mode='lines',
                            line=dict(color=color, width=15)
                        ),
                        secondary_y=False
                    )

    fig.update_layout(
        title=f"{title}: Democracy and Trade Analysis",
        **PLOT_STYLE
    )

    return fig

def create_category_overview(df, categories):
    """Create 2x2 grid showing time series for all categories."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=categories,
        specs=[[{"secondary_y": True}] * 2] * 2
    )

    for idx, category in enumerate(categories, 1):
        row, col = (idx - 1) // 2 + 1, (idx - 1) % 2 + 1

        category_data = df[df['Regime_Type'] == category].groupby('year').agg({
            'v2x_polyarchy': 'mean',
            'KOFEcGI': 'mean'
        }).reset_index()

        # Add traces for democracy score and trade openness
        for trace_idx, (metric, color) in enumerate([('v2x_polyarchy', '#00bfff'), ('KOFEcGI', 'green')]):
            fig.add_trace(
                go.Scatter(
                    x=category_data['year'],
                    y=category_data[metric],
                    name=["Democracy Score", "Trade Openness"][trace_idx] if idx == 1 else None,
                    line=dict(color=color),
                    showlegend=(idx == 1)
                ),
                row=row, col=col,
                secondary_y=bool(trace_idx)
            )

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

def create_app_layout():
    """Create the application layout."""
    return dbc.Container([
        html.Div(
            style={'backgroundColor': '#0a192f', 'padding': '20px', 'minHeight': '100vh'},
            children=[
                html.H1(
                    "Democracy vs. Trade Openness Research - PSC354 Group 9",
                    className="text-center my-4",
                    style={'color': '#64ffda', 'fontFamily': 'Helvetica, Arial, sans-serif', 'letterSpacing': '1px'}
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                    style={'backgroundColor': '#172a45', 'padding': '15px', 'borderRadius': '5px'},
                                    children=[
                                        html.H4("Select Country/Category"),
                                        dcc.Dropdown(
                                            id='country-dropdown',
                                            options=dropdown_options,
                                            value='global_average',
                                            style={'backgroundColor': '#1a365d', 'color': 'white'},
                                            className='dropdown-dark',
                                        )
                                    ]
                                )
                            ])
                        ], className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                    style={'backgroundColor': '#172a45', 'padding': '15px', 'borderRadius': '5px'},
                                    children=[dcc.Graph(id='country-graph')]
                                )
                            ])
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                    style={'backgroundColor': '#172a45', 'padding': '15px', 'borderRadius': '5px'},
                                    children=[dcc.Graph(id='category-graph')]
                                )
                            ])
                        ])
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                    style={'backgroundColor': '#172a45', 'padding': '15px', 'borderRadius': '5px'},
                                    children=[dcc.Graph(id='correlation-scatter')]
                                )
                            ])
                        ])
                    ], width=12)
                ], className="mt-3")
            ]
        )
    ], fluid=True, style={'backgroundColor': '#0a192f', 'padding': '0'})

# Initialize the application
df = load_data()
dropdown_options, categories = create_dropdown_options(df)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.layout = create_app_layout()

# Callbacks
@app.callback(
    Output('correlation-scatter', 'figure'),
    Input('country-dropdown', 'value')
)
def update_scatter(selected):
    data, title = get_filtered_data(df, selected)
    return create_correlation_plot(data, title)

@app.callback(
    Output('country-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_country_graph(selected):
    data, title = get_filtered_data(df, selected)
    return create_time_series(data, title)

@app.callback(
    Output('category-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_category_graph(_):
    return create_category_overview(df, categories)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
