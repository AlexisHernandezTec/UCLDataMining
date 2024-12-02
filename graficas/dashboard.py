from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

app = Dash(__name__)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = pd.read_csv("../datasets/ucl-finals-juntado.csv", index_col=0)

app.layout = html.Div(style={'backgroundColor':colors["background"]},children=[
                          html.H1(
                          children='UCL Dashboard',
                          style={
                              'color': colors["text"],
                              'textAlign': 'center'
                              }
                      ),
                      html.Div(
                          children="Dash: A web application framework for you data.",
                          style={
                              'color': colors["text"],
                              }
                    )
])
if __name__ == '__main__':
    app.run(debug=True)
#end if 