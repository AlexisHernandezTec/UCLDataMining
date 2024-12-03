from dash import Dash, dcc, html
from graficas.heatmapChart import plot_heatmap
from helpers.formatDataset import formatDSEncoder
from helpers.formatDataset import applyLinearRegression
from graficas.scatterChart import plot_real_vs_predicted
from graficas.pieChart import plot_pie_chart
import plotly.express as px
import pandas as pd

app = Dash(__name__)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = pd.read_csv("datasets/ucl-finals-juntado.csv", index_col=0)
ds1 = formatDSEncoder(df)
regressor, y_test, y_pred, sc = applyLinearRegression(ds1)
mapaCalor = plot_heatmap(ds1.corr())
scatterChart = plot_real_vs_predicted(y_test, y_pred)
pieChart = plot_pie_chart(pd.DataFrame({'Categoría': ['A', 'B', 'C'], 'Valores': [10, 20, 30]}))


app.layout = html.Div(
    style={
        'backgroundColor': colors["background"],
        'width': '100vw',         # Ancho de toda la ventana
        'height': '100%',         # Ajusta la altura al contenido
        'margin': '0',            # Sin márgenes externos
        'padding': '20px',        # Espaciado interno
        'overflowX': 'auto'       # Habilita el desplazamiento horizontal
    },
    children=[
        html.H1(
            children='UCL Dashboard',
            style={
                'color': colors["text"],
                'textAlign': 'center',
                'width': '100%'     # Asegura que el encabezado ocupe todo el ancho
            }
        ),
        html.Div(
            children="Dash: A web application framework for your data.",
            style={
                'color': colors["text"],
                'textAlign': 'center',
                'width': '100%'     # Igual que el encabezado
            }
        ),
        html.Div(
            style={
                'display': 'flex',          # Activa el diseño flex
                'flexDirection': 'row',     # Alinea los elementos horizontalmente
                'justifyContent': 'flex-start',  # Inicia los gráficos desde la izquierda
                'alignItems': 'center',     # Centra los gráficos verticalmente
                'gap': '20px',              # Espacio entre gráficos
                'overflowX': 'auto'         # Permite desplazamiento horizontal si es necesario
            },
            children=[
                dcc.Graph(
                    id='mapa-de-calor',
                    figure=mapaCalor
                ),
                dcc.Graph(
                    id='scatter-chart',
                    figure=scatterChart
                ),
                dcc.Graph(
                    id='pie-chart',
                    figure=pieChart
                )
            ]
        )
    ]
)



if __name__ == '__main__':
    app.run(debug=True)
#end if 