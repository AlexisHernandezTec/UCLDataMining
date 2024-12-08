import plotly.express as px

def plot_pie_chart(data,name = "Equipo", value = "Valor", title = "Distribución de Datos"):
    fig = px.pie(data, names=name, values=value, title=title)
    return fig
#end defx