import pandas as pd
import plotly.express as px
import dash
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
from dash import html
from dash import dcc

# Load data from CSV
data = pd.read_csv('../../../Datasets/source/diabetes_prediction_dataset.csv')

# Preprocess data
data = pd.get_dummies(data, columns=['gender', 'smoking_history'])
data = data.drop(data[data['age'] < 2].index)
data = data.drop(data[data['bmi'] > 50].index)

data.rename(columns={'smoking_history_No Info': 'smoke_no_info',
                     'smoking_history_never': 'never_smoke',
                     'smoking_history_current': 'current_smoke',
                     'smoking_history_ever': 'ever_smoke',
                     'smoking_history_former': 'former_smoke',
                     'smoking_history_not current': 'not_current',
                     'blood_glucose_level': 'glucose_level'}, inplace=True)

# Calculate correlation with diabetes
correlation_with_diabetes = data.drop(columns=['diabetes']).corrwith(data['diabetes'])

# Filter the correlation matrix to keep only correlations related to diabetes
correlation_matrix_diabetes = data[['diabetes'] + list(correlation_with_diabetes.index)].corr().round(2)

# Calculate the count of people with diabetes at each age group
diabetes_age_counts = data[data['diabetes'] == 1]['age'].value_counts().reset_index()
diabetes_age_counts.columns = ['age', 'diabetes_count']

# Create age bins
age_bins = data['age'].value_counts(bins=10)

# Define custom labels for the legend based on age ranges
age_pie_labels = [(int(age_range.left), int(age_range.right)) for age_range in age_bins.index]
age_pie_labels_str = [f"{lower} - {upper}" for lower, upper in age_pie_labels]

""" Creating plots using plotly.express """

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

# Create a bubble chart for diabetes count at each age group
bubble_chart = px.scatter(diabetes_age_counts, x='age', y='diabetes_count', size='diabetes_count',
                          labels={'age': 'Age', 'diabetes_count': 'Diabetes Count'},
                          title='Diabetes Count by Age', size_max=30, template='plotly_white',
                          color_discrete_sequence=px.colors.qualitative.Pastel2)

# Bar Chart for Diabetes Comparison
bar_chart = px.bar(data_frame=data, x=data['diabetes'].value_counts().index, y=data['diabetes'].value_counts().values,
                   text=data['diabetes'].value_counts().values,
                   labels={'x': 'Diabetes', 'y': 'Count'},
                   color=data['diabetes'].value_counts().values,
                   title='Comparison between diabetes 1 and 0',
                   template='simple_white')

# Initial Correlation Bar Chart
correlation_chart_bar = px.bar(x=correlation_with_diabetes.index, y=correlation_with_diabetes.values,
                               title='Correlation between diabetes and other columns',
                               color=correlation_with_diabetes.values,
                               width=1000,
                               height=400,
                               template='simple_white',
                               color_continuous_scale=px.colors.sequential.Emrld
                               )

# Correlation Heatmap focused on diabetes
correlation_chart_heatmap = px.imshow(correlation_matrix_diabetes, labels=dict(),
                                      color_continuous_scale='Spectral',
                                      title='Correlation heatmap',
                                      width=800,
                                      height=800,
                                      text_auto=True
                                      )
# Age Pie Chart
age_pie_chart = px.pie(names=age_pie_labels_str, values=age_bins.values,
                       title='Pie chart for age',
                       hole=0.3,
                       color_discrete_sequence=px.colors.qualitative.Pastel)

age_pie_chart.update_traces(hovertext=age_pie_labels_str)

age_pie_chart.update_layout(
    legend=dict(
        title="  Age",
        x=0.7,
        y=0.5,
    )
)

# Scatter Plot for Age vs. BMI:
scatter_plot = px.scatter(data_frame=data, x='age', y='bmi', color='diabetes', title='Age vs. BMI', template='plotly_white')

# Separate box plots for BMI, HbA1c level, and blood glucose level.
box_plot_bmi = px.box(data_frame=data, x='diabetes', y='bmi', color='diabetes', title='BMI', template='plotly_white')
box_plot_hba1c = px.box(data_frame=data, x='diabetes', y='HbA1c_level', color='diabetes', title='HbA1c level', template='plotly_white')
box_plot_glucose = px.box(data_frame=data, x='diabetes', y='glucose_level', color='diabetes',
                          title='Blood glucose level', template='plotly_white')

# Define dropdown options
dropdown_options_box = [
    {'label': 'BMI', 'value': 'bmi'},
    {'label': 'HbA1c level', 'value': 'HbA1c_level'},
    {'label': 'Blood glucose level', 'value': 'glucose_level'}
]

dropdown_options_distribution = [
    {'label': 'Age', 'value': 'age'},
    {'label': 'BMI', 'value': 'bmi'},
    {'label': 'HbA1c level', 'value': 'HbA1c_level'},
    {'label': 'Blood glucose level', 'value': 'glucose_level'}
]

# Initial Histogram
initial_histogram = px.histogram(data_frame=data, x='age', title='Histogram', template='plotly_white', nbins=15, text_auto=True, color='diabetes')
# Initial dropdown for boxes
initial_box = px.box(data_frame=data, x='diabetes', y='bmi', color='diabetes', title='BMI', template='plotly_white')

# Layout
app.layout = html.Div([

    html.H1('Diabetes Prediction Data Visualization',
            style={
                'textAlign': 'center',
                'margin-bottom': '20px',
                'font-size': '40px',
                'margin-top': '30px',
            }
            ),

    html.H6('This dashboard is created to visualize the data used for diabetes prediction'
            ' and for better understanding of the data',
            style={
                'textAlign': 'center',
                'margin-bottom': '20px',
                'font-size': '20px',
                'margin-top': '10px',
            }
            ),

    dbc.Card(dcc.Graph(id='bar-chart', figure=bar_chart), body=True, style={'margin-bottom': '20px'}),

    dbc.Card(dcc.Graph(id='age-pie-chart', figure=age_pie_chart), body=True, style={'margin-bottom': '20px'}),

    dbc.Card(dcc.Graph(id='bubble-chart', figure=bubble_chart), body=True, style={'margin-bottom': '20px'}),

    dbc.Card(dcc.Graph(id='scatter-plot', figure=scatter_plot), body=True, style={'margin-bottom': '20px'}),

    dcc.Dropdown(
        id='correlation-chart-dropdown',
        options=[
            {'label': 'Bar', 'value': 'bar_chart'},
            {'label': 'Heatmap', 'value': 'heatmap'}
        ],
        value='bar_chart',
        clearable=False,
        style={'textAlign': 'center', 'width': '40%', 'margin': '2%', 'margin-bottom': '5px'}
    ),

    dbc.Card(dcc.Graph(id='correlation-chart'), body=True, style={'margin-bottom': '20px',
                                                                  'align-items': 'center'}),

    dcc.Dropdown(
        id='variable-dropdown',
        options=dropdown_options_box,
        value='bmi',
        clearable=False,
        style={'textAlign': 'center', 'width': '40%', 'margin': '2%', 'margin-bottom': '5px'},
    ),

    dbc.Card(dcc.Graph(id='box-plot', figure=initial_box), body=True, style={'margin-bottom': '20px'}),

    dcc.Dropdown(
        id='variable-dropdown-distribution',
        options=dropdown_options_distribution,
        value='age',
        clearable=False,
        style={'textAlign': 'center', 'width': '40%', 'margin': '2%', 'margin-bottom': '5px'},
    ),

    dbc.Card(dcc.Graph(id='histogram', figure=initial_histogram), body=True, style={'margin-bottom': '20px'})
])


# Callback function for updating the correlation chart based on dropdown selection
@app.callback(
    Output('correlation-chart', 'figure'),
    [Input('correlation-chart-dropdown', 'value')]
)
def update_correlation_chart(selected_chart):
    if selected_chart == 'bar_chart':
        return correlation_chart_bar
    elif selected_chart == 'heatmap':
        return correlation_chart_heatmap


# Callback function for updating the box plot
@app.callback(
    Output('box-plot', 'figure'),
    [Input('variable-dropdown', 'value')]
)
def update_box_plot(selected_variable):
    fig = px.box(data, x='diabetes', y=selected_variable, color='diabetes',
                 title=f'{selected_variable.capitalize()}', template='plotly_white', color_discrete_sequence=px.colors.qualitative.Pastel2)
    return fig


# Callback function for updating the histogram
@app.callback(
    Output('histogram', 'figure'),
    [Input('variable-dropdown-distribution', 'value')]
)
def update_histogram(selected_variable):
    fig = px.histogram(data, x=selected_variable, nbins=15, title=f'Distribution of {selected_variable.capitalize()}', text_auto=True, template='plotly_white', color='diabetes')
    return fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
