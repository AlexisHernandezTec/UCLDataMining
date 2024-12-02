# model_visuals.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_heatmap(correlation_matrix, title="Heatmap de Correlaciones"):
    """Grafica un heatmap de la matriz de correlación."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.show()

def plot_real_vs_predicted(y_test, y_pred, title="Comparación de Valores Reales vs Predichos"):
    """Grafica valores reales contra valores predichos."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label="Predicciones")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal (y_test = y_pred)")
    plt.title(title)
    plt.xlabel("Valores Reales (y_test)")
    plt.ylabel("Valores Predichos (y_pred)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pie_chart(score, title="Efectividad del Modelo"):
    """Grafica un gráfico circular de precisión y error."""
    error = 1 - score
    labels = ['Precisión del Modelo', 'Error']
    sizes = [score, error]
    colors = ['#4CAF50', '#FF5722']
    explode = (0.1, 0)

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode)
    plt.title(title)
    plt.show()

def plot_feature_importance(model, feature_names, title="Importancia de las Características"):
    """Grafica la importancia de las características en el modelo."""
    feature_importance = pd.Series(model.coef_, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    feature_importance.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(title)
    plt.ylabel("Peso del Coeficiente")
    plt.xlabel("Características")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_residual_distribution(y_test, y_pred, title="Distribución de los Residuos"):
    """Grafica la distribución de los residuos."""
    residuos = y_test - y_pred

    plt.figure(figsize=(10, 6))
    sns.histplot(residuos, kde=True, color='purple', bins=30, alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', label="Cero Residuo")
    plt.title(title)
    plt.xlabel("Residuos (y_test - y_pred)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
