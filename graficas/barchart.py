import plotly.express as px

def plot_bar_chart(data, name="Equipo", value="Valor", title="Distribución de Datos"):
    fig = px.bar(data, x=name, y=value, title=title)
    return fig
#end def