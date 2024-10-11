# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[
    html.H1(
        'SpaceX Launch Records Dashboard',
        style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}
    ),
    
    # TASK 1: Add a dropdown list to enable Launch Site selection
    dcc.Dropdown(
        id='site-dropdown',
        options=[
            {'label': 'All Sites', 'value': 'ALL'}
        ] + [{'label': site, 'value': site} for site in spacex_df['Launch Site'].unique()],
        value='ALL',
        placeholder="Select a Launch Site here",
        searchable=True
    ),
    
    # Pie chart
    html.Div(dcc.Graph(id='success-pie-chart')),
    
    # Add a break
    html.Br(),
    
    # Payload range selector
    html.P("Payload range (Kg):"),
    
    # TASK 3: Add a slider to select payload range
    dcc.RangeSlider(
        id='payload-slider',
        min=0,
        max=10000,
        step=1000,
        value=[min_payload, max_payload]
    ),
    
    # TASK 4: Add a scatter chart
    html.Div(dcc.Graph(id='success-payload-scatter-chart'))
])

# TASK 2: Callback for pie chart
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    if entered_site == 'ALL':
        # Calculate total success launches for each launch site
        success_counts = spacex_df[spacex_df['class'] == 1]['Launch Site'].value_counts()

        # Create a pie chart with success counts for each launch site
        fig = px.pie(
            names=success_counts.index,  # Launch site names
            values=success_counts.values,  # Success counts
            title='Total Success Launches by Launch Site'
        )
        return fig
    else:
        # Filter the dataframe for the selected site
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        success_count = filtered_df['class'].value_counts().get(1, 0)
        fail_count = filtered_df['class'].value_counts().get(0, 0)

        # Create a pie chart for the selected site
        fig = px.pie(
            names=['Success', 'Failure'],
            values=[success_count, fail_count],
            title=f'Total Success Launches for {entered_site}'
        )
        return fig

# TASK 4: Callback for scatter plot
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def get_scatter_chart(entered_site, payload_range):
    if entered_site == 'ALL':
        filtered_df = spacex_df
    else:
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
    
    # Filter based on payload range
    mask = (filtered_df['Payload Mass (kg)'] >= payload_range[0]) & \
           (filtered_df['Payload Mass (kg)'] <= payload_range[1])
    filtered_df = filtered_df[mask]
    
    fig = px.scatter(filtered_df, 
                    x='Payload Mass (kg)', 
                    y='class',
                    color='Booster Version Category',
                    title='Correlation between Payload and Success for Selected Site(s)')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server()