import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data():
    """Load and prepare data from Excel file."""
    excel_file = pd.ExcelFile('data/democracy_trade_analysis.xlsx')
    return pd.concat([
        pd.read_excel(excel_file, sheet_name=sheet)
        for sheet in excel_file.sheet_names
    ], ignore_index=True)

def prepare_dropdown_options(df):
    """Create dropdown options with global and category averages at top."""
    countries = sorted(df['country_name'].dropna().unique())
    categories = sorted(df['Category'].dropna().unique())

    return ([
        {'label': '[Global Average] All Countries', 'value': 'global_average'}
    ] + [
        {'label': f'[Average] {category}', 'value': f'avg_{category}'}
        for category in categories
    ] + [
        {'label': country, 'value': country}
        for country in countries
    ], categories)

def filter_data(df, selected):
    """Filter data based on dropdown selection."""
    if selected == 'global_average':
        data = df.groupby('year', as_index=False).agg({
            'v2x_polyarchy': 'mean',
            'KOFTrGIdf': 'mean',
            'Regime_Type': lambda x: x.mode().iloc[0] if not x.empty else None
        })
        title = "Global Average"
    elif selected.startswith('avg_'):
        category = selected[4:]
        data = df[df['Regime_Type'] == category].groupby('year', as_index=False).agg({
            'v2x_polyarchy': 'mean',
            'KOFTrGIdf': 'mean',
            'Regime_Type': lambda x: x.mode().iloc[0] if not x.empty else None
        })
        title = f"Average: {category}"
    else:
        data = df[df['country_name'] == selected]
        title = selected

    return data, title

def create_correlation_plot(data, title):
    """Create scatter plot with correlation analysis."""
    fig = go.Figure()

    valid_data = data.dropna(subset=['v2x_polyarchy', 'KOFTrGIdf'])

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=valid_data['v2x_polyarchy'],
        y=valid_data['KOFTrGIdf'],
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
    coefficients = np.polyfit(valid_data['v2x_polyarchy'], valid_data['KOFTrGIdf'], 1)
    correlation = np.corrcoef(valid_data['v2x_polyarchy'], valid_data['KOFTrGIdf'])[0, 1]
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
        margin=dict(b=100)
    )

    return fig

def create_time_series(data, title):
    """Create time series with dual y-axes and regime indicators."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add democracy score line
    fig.add_trace(
        go.Scatter(x=data['year'], y=data['v2x_polyarchy'],
                  name="Democracy Score", line=dict(color='#00bfff')),
        secondary_y=False
    )

    # Add trade openness line
    fig.add_trace(
        go.Scatter(x=data['year'], y=data['KOFTrGIdf'],
                  name="Trade Openness", line=dict(color='green')),
        secondary_y=True
    )

    # Add regime indicators for individual countries
    if not title.startswith('Average:') and title != "Global Average":
        regime_colors = {
            'Liberal Democracy': '#2ecc71',
            'Electoral Democracy': '#3498db',
            'Electoral Autocracy': '#e74c3c',
            'Closed Autocracy': '#2c3e50'
        }

        for regime_type, color in regime_colors.items():
            regime_years = data[data['Regime_Type'] == regime_type]['year']
            if not regime_years.empty:
                fig.add_trace(
                    go.Scatter(x=regime_years, y=[0] * len(regime_years),
                             name=regime_type, mode='lines',
                             line=dict(color=color, width=15)),
                    secondary_y=False
                )

    fig.update_layout(
        title=f"{title}: Democracy and Trade Analysis",
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
        margin=dict(b=100)
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
        row = (idx - 1) // 2 + 1
        col = (idx - 1) % 2 + 1

        category_data = df[df['Regime_Type'] == category].groupby('year').agg({
            'v2x_polyarchy': 'mean',
            'KOFTrGIdf': 'mean'
        }).reset_index()

        fig.add_trace(
            go.Scatter(x=category_data['year'], y=category_data['v2x_polyarchy'],
                      name="Democracy Score" if idx == 1 else None,
                      line=dict(color='#00bfff'),
                      showlegend=(idx == 1)),
            row=row, col=col, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=category_data['year'], y=category_data['KOFTrGIdf'],
                      name="Trade Openness" if idx == 1 else None,
                      line=dict(color='green'),
                      showlegend=(idx == 1)),
            row=row, col=col, secondary_y=True
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

# Initialize data and app
df = load_data()
dropdown_options, categories = prepare_dropdown_options(df)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# App layout
app.layout = dbc.Container([
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

# Callbacks
@app.callback(
    Output('correlation-scatter', 'figure'),
    Input('country-dropdown', 'value')
)
def update_scatter(selected):
    data, title = filter_data(df, selected)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0')