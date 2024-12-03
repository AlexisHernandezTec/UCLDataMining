import plotly.express as px

def plot_pie_chart(data):
    """Genera un gráfico circular."""
    fig = px.pie(data, names='Categoría', values='Valores', title="Distribución de Datos")
    return fig
#end def