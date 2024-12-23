from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

app = Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor':colors["background"]},children=[
                          html.H1(
                          children='Hello Dash',
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
                    ),
                    dcc.Graph(
                        id='example-graph-2',
                        figure=fig
                    )
])

if __name__ == '__main__':
    app.run(debug=True)
#end if    