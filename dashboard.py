from dash import Dash, dcc, html
from graficas.pieChart import plot_pie_chart
from graficas.barchart import plot_bar_chart
from graficas.scatterChart import plot_scatter_chart
from helpers.predictFields import predictSingleField
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Dash(__name__, external_stylesheets=[
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap'
])

ds = pd.read_csv("datasets/ucl-finals-juntado.csv", index_col=0)
equipos = ds["Equipo"].unique()
nacionalidades = ds["Nacionalidad"].unique()
marcadores = ds["score"].unique()
formasDeGanar = ds["winning-way"].unique()
estadios = ds["stadium"].unique()

prediccionesEquipos = predictSingleField(ds, predicVar="winner",predictColumns="Equipo",predictColumns2D=["Equipo"],uniquePredictValues=equipos, testSize=0.2)
prediccionesNacionalidades = predictSingleField(ds, predicVar="winner",predictColumns="Nacionalidad",predictColumns2D=["Nacionalidad"],uniquePredictValues=nacionalidades, testSize=0.2)
prediccionesMarcador = predictSingleField(ds, predicVar="winner",predictColumns="score",predictColumns2D=["score"],uniquePredictValues=marcadores, testSize=0.2)
prediccionesPerdedores = predictSingleField(ds,predicVar="runner-up",predictColumns="Equipo",predictColumns2D=["Equipo"],uniquePredictValues=equipos, testSize=0.2)
prediccionesFormasDeGanar = predictSingleField(ds,predicVar="winner",predictColumns="winning-way",predictColumns2D=["winning-way"],uniquePredictValues=formasDeGanar, testSize=0.2)
prediccionesEstadios = predictSingleField(ds,predicVar="winner",predictColumns="stadium",predictColumns2D=["stadium"],uniquePredictValues=estadios, testSize=0.2)

equipos_df = pd.DataFrame(list(prediccionesEquipos.items()), columns=['Equipo', 'Valor'])
nacionalidades_df = pd.DataFrame(list(prediccionesNacionalidades.items()), columns=['Nacionalidad', 'Valor'])
marcadores_df = pd.DataFrame(list(prediccionesMarcador.items()), columns=['Marcador', 'Valor'])
perdedores_df = pd.DataFrame(list(prediccionesPerdedores.items()), columns=['Equipo', 'Valor'])
formasDeGanar_df = pd.DataFrame(list(prediccionesFormasDeGanar.items()), columns=['Forma de ganar', 'Valor'])
estadios_df = pd.DataFrame(list(prediccionesEstadios.items()), columns=['Estadio', 'Valor'])

teamsChart = plot_pie_chart(equipos_df,"Equipo", "Valor", "Predicción de equipo ganador")
nationalitiesChart = plot_bar_chart(nacionalidades_df,"Nacionalidad", "Valor", "Predicción de nacionalidad ganador")
scoresChart = plot_scatter_chart(data=marcadores_df,x="Marcador", y="Valor", title="Predicción de marcador ganador")
runnerUpsChart = plot_pie_chart(perdedores_df,"Equipo", "Valor", "Predicción del equipo perdedor")
winningWaysChart = plot_bar_chart(formasDeGanar_df,"Forma de ganar", "Valor", "Predicción de forma de ganar")
stadiumsChart = plot_scatter_chart(data=estadios_df,x="Estadio", y="Valor", title="Predicción de estadio ganador")

app.layout = html.Div(
    style={
        'backgroundColor': '#E2E8F0',  # Color de fondo
        'width': '100vw',
        'height': '100%',
        'margin': '0',
        'padding': '20px',
        'overflowX': 'auto',
        'borderRadius': '12px'
    },
    children=[
        html.H1(
            children='Muestreo de predicciones de la UEFA Champions League',
            style={
                'color': '#2D3748',
                'textAlign': 'center',
                'width': '100%',
                'fontFamily': 'Arial, sans-serif',  # Usando Arial como fuente
                'marginBottom': '10px'  # Reducir el espacio debajo del H1
            }
        ),
        html.Div(
            children="En este dashboard se muestran diferentes tipos de predicciones con algunas de las variables más significativas",
            style={
                'color': '#2D3748',
                'textAlign': 'center',
                'width': '100%',
                'marginBottom': '40px',  # Reducir el espacio debajo del H2
                'fontFamily': 'Arial, sans-serif'  # Usando Arial como fuente
            }
        ),
        
        # Primer conjunto de gráficos
        html.Div(
            style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'flex-start',
                'alignItems': 'center',
                'gap': '20px',
                'overflowX': 'auto',
                'marginBottom': '40px'
            },
            children=[
                html.Div(
                    style={
                        'backgroundColor': '#FFFFFF',   # Fondo blanco para el div
                        'borderRadius': '12px',         # Esquinas redondeadas
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',  # Sombra ligera
                        'padding': '20px'               # Espaciado dentro del div
                    },
                    children=dcc.Graph(
                        id='teams-chart',
                        figure=teamsChart
                    )
                ),
                html.Div(
                    style={
                        'backgroundColor': '#FFFFFF',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                        'padding': '20px'
                    },
                    children=dcc.Graph(
                        id='nationalities-chart',
                        figure=nationalitiesChart
                    )
                ),
                html.Div(
                    style={
                        'backgroundColor': '#FFFFFF',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                        'padding': '20px'
                    },
                    children=dcc.Graph(
                        id='score-chart',
                        figure=scoresChart
                    )
                )
            ]
        ),
        
        # Segundo conjunto de gráficos
        html.Div(
            style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'flex-start',
                'alignItems': 'center',
                'gap': '20px',
                'overflowX': 'auto'
            },
            children=[
                html.Div(
                    style={
                        'backgroundColor': '#FFFFFF',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                        'padding': '20px'
                    },
                    children=dcc.Graph(
                        id='runner-ups-chart',
                        figure=runnerUpsChart
                    )
                ),
                html.Div(
                    style={
                        'backgroundColor': '#FFFFFF',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                        'padding': '20px'
                    },
                    children=dcc.Graph(
                        id='winning-ways-chart',
                        figure=winningWaysChart
                    )
                ),
                html.Div(
                    style={
                        'backgroundColor': '#FFFFFF',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                        'padding': '20px'
                    },
                    children=dcc.Graph(
                        id='stadiums-chart',
                        figure=stadiumsChart
                    )
                )
            ]
        )
    ]
)





if __name__ == '__main__':
    app.run(debug=True)
#end if 