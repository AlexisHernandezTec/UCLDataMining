import plotly.express as px

def plot_scatter_chart(data, x="X", y="Y", title="Gráfico de Dispersión"):
    fig = px.scatter(data, x=x, y=y, title=title)
    return fig
#end def