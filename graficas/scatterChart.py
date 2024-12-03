import plotly.graph_objects as go

def plot_real_vs_predicted(y_test, y_pred):
    """Genera un gráfico de dispersión de valores reales vs predichos."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Datos'))
    fig.update_layout(
        title="Real vs Predicted",
        xaxis_title="Valores Reales",
        yaxis_title="Valores Predichos"
    )
    return fig
#end def
