import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load and prepare data
excel_file = pd.ExcelFile('data/democracy_trade_analysis.xlsx')
df = pd.concat([pd.read_excel(excel_file, sheet_name=sheet) 
                for sheet in excel_file.sheet_names], ignore_index=True)
countries = sorted(df['country_name'].dropna().unique())
categories = sorted(df['Category'].dropna().unique())

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# App layout
app.layout = dbc.Container([
    html.Div(style={'backgroundColor': '#0a192f', 'padding': '20px', 'minHeight': '100vh'}, children=[
        html.H1("Democracy vs. Trade Openness Research - PSC354 Group 9", 
                className="text-center my-4",
                style={'color': '#64ffda', 'fontFamily': 'Helvetica, Arial, sans-serif', 'letterSpacing': '1px'}),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(style={'backgroundColor': '#172a45', 'padding': '15px', 'borderRadius': '5px'}, children=[
                    html.H4("Select Country"),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': c, 'value': c} for c in countries],
                        value=countries[0],
                        style={
                            'backgroundColor': '#1a365d',
                            'color': 'white'
                        },
                        className='dropdown-dark',
                    )
                    ])
                ])
            ], className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.Div(style={'backgroundColor': '#172a45', 'padding': '15px', 'borderRadius': '5px'}, children=[
                    dcc.Graph(id='country-graph')
                    ])
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(style={'backgroundColor': '#172a45', 'padding': '15px', 'borderRadius': '5px'}, children=[
                    dcc.Graph(id='category-graph')
                    ])
                ])
            ])
        ], width=6)
    ])
])
], fluid=True, style={'backgroundColor': '#0a192f', 'padding': '0'})

@app.callback(
    Output('country-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_country_graph(country):
    data = df[df['country_name'] == country]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Identify continuous ranges of missing data
    missing_ranges = []
    start_year = None

    for year in range(1970, 2021):
        year_data = data[data['year'] == year]
        is_missing = year_data.empty or year_data['v2x_polyarchy'].isnull().any() or year_data['KOFTrGIdf'].isnull().any()

        if is_missing and start_year is None:
            start_year = year
        elif not is_missing and start_year is not None:
            missing_ranges.append((start_year, year))
            start_year = None

    if start_year is not None:
        missing_ranges.append((start_year, 2021))

    # Add missing data highlighting for each continuous range
    for start, end in missing_ranges:
        fig.add_vrect(
            x0=start-0.5,
            x1=end-0.5,
            fillcolor="rgba(255, 0, 0, 0.15)",
            layer="below",
            line_width=0,
            name="Missing Data"
        )

    fig.add_trace(
        go.Scatter(x=data['year'], y=data['v2x_polyarchy'],
                  name="Democracy Score", line=dict(color='#00bfff')),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=data['year'], y=data['KOFTrGIdf'],
                  name="Trade Openness", line=dict(color='green')),
        secondary_y=True
    )

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

    fig.update_layout(
        title=f"{country}: Democracy and Trade Analysis",
        template="plotly_dark",
        paper_bgcolor='#172a45',
        plot_bgcolor='#172a45',
        font=dict(color='#ffffff'),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

@app.callback(
    Output('category-graph', 'figure'),
    Input('country-dropdown', 'value')
)
def update_category_graph(_):
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=categories,
                       specs=[[{"secondary_y": True}, {"secondary_y": True}],
                            [{"secondary_y": True}, {"secondary_y": True}]])

    for idx, category in enumerate(categories, 1):
        category_data = df[df['Category'] == category].groupby('year').agg({
            'v2x_polyarchy': 'mean',
            'KOFTrGIdf': 'mean'
        }).reset_index()

        row = (idx - 1) // 2 + 1
        col = (idx - 1) % 2 + 1

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

if __name__ == '__main__':
    app.run(host='0.0.0.0')