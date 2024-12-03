import plotly.express as px

def plot_heatmap(correlation_matrix):
    """Genera un heatmap compatible con Dash."""
    fig = px.imshow(
        correlation_matrix,
        color_continuous_scale='magma',
        title="Heatmap de Correlación"
    )
    fig.update_layout(
        title_font_size=24,
        coloraxis_colorbar=dict(title="Correlación")
    )
    return fig
#end def