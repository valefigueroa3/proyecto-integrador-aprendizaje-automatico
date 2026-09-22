from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# 5. Resultados de scikit-learn

Este capítulo consolida las seis ejecuciones sobre la misma prueba común. Las métricas de clasificación usan el umbral predeterminado de cada modelo: 0,5 para probabilidades y 0 para `decision_function` de `LinearSVC`. El AUC utiliza la puntuación continua y no depende de un umbral."""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 5.1 Modelos incluidos y qué aprende cada uno

| Modelo | Idea principal |
|:--|:--|
| Regresión logística | Combina las variables con pesos y transforma el resultado en una probabilidad de default. |
| Árbol de decisión | Divide sucesivamente los préstamos mediante reglas sobre variables, hasta formar grupos con resultados más parecidos. |
| Bosque aleatorio | Promedia muchos árboles entrenados con variaciones de filas y variables para reducir la inestabilidad de un árbol individual. |
| HistGradientBoosting | Construye árboles por etapas; cada nueva etapa corrige parte de los errores de las anteriores. |
| LinearSVC | Busca una frontera lineal que separe las clases con el mayor margen; su puntuación no es una probabilidad calibrada. |
| GaussianNB | Calcula la probabilidad de cada clase suponiendo distribuciones gaussianas y que las variables son condicionalmente independientes. |

`best_params` y `best_cv_auc` resumen la búsqueda sobre los datos de entrenamiento y sus folds. Las métricas de prueba se calculan después, sobre las filas reservadas; esa prueba no participa en la selección de parámetros.

`C` regula cuánto se penalizan los coeficientes en los modelos lineales: valores menores aplican una regularización más fuerte. `max_depth` limita cuántas decisiones puede encadenar un árbol; `n_estimators` indica cuántos árboles reúne el bosque; y `max_iter` fija las etapas del boosting. La validación cruzada elige entre los valores probados, no entre todos los posibles."""))

cells.append(nbf.v4.new_code_cell(
"""from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RESULTS_DIR = PROJECT_DIR / 'results' / 'sklearn'
PRED_DIR = PROJECT_DIR / 'artifacts' / 'sklearn' / 'predictions'
FIGURES_DIR = PROJECT_DIR / 'figures' / 'sklearn'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

model_names = [
    'logistic_regression', 'decision_tree', 'random_forest',
    'hist_gradient_boosting', 'linear_svc', 'gaussian_nb',
]
labels = {
    'logistic_regression': 'Regresión logística',
    'decision_tree': 'Árbol de decisión',
    'random_forest': 'Bosque aleatorio',
    'hist_gradient_boosting': 'HistGradientBoosting',
    'linear_svc': 'SVM lineal',
    'gaussian_nb': 'Naive Bayes gaussiano',
}

records = []
for model in model_names:
    with open(RESULTS_DIR / f'metrics_{model}.json', encoding='utf-8') as stream:
        record = json.load(stream)
    record['nombre'] = labels[model]
    records.append(record)

metrics = pd.DataFrame(records).set_index('model')
metrics = metrics[['nombre', 'implementation', 'best_params', 'best_cv_auc', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision', 'fit_cv_seconds', 'predict_seconds', 'tn', 'fp', 'fn', 'tp']]
display(metrics)
metrics.to_csv(RESULTS_DIR / 'comparacion_metricas_sklearn.csv', encoding='utf-8-sig')"""
))

cells.append(nbf.v4.new_markdown_cell(
"""La tabla reúne el mejor AUC de validación cruzada, las métricas en la prueba y los tiempos. `HistGradientBoosting` logra el AUC mayor (0,7132), seguido por regresión logística (0,7098) y bosque aleatorio (0,7080); las diferencias entre estos tres son pequeñas. `GaussianNB` alcanza un recall más alto (0,5513) pero menor precisión (0,3133) y AUC (0,6668): detecta más préstamos castigados a costa de marcar más préstamos pagados como riesgo. `LinearSVC` obtiene AUC de 0,4759 y, al umbral predeterminado, predice los 269.062 préstamos como pagados: su accuracy de 0,8004 solo refleja la clase mayoritaria y su recall es cero. El modelo con mayor AUC ordena mejor los casos en general, pero no necesariamente ofrece el mejor balance de errores al umbral usado."""))

cells.append(nbf.v4.new_markdown_cell(
"""El AUC es la métrica de selección solicitada y AUC-PR presta atención a la clase minoritaria. Como `Fully Paid` representa cerca de 80 % de los datos, la accuracy se interpreta junto con recall, F1 y la matriz de confusión."""
))

cells.append(nbf.v4.new_code_cell(
"""metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
fig, ax = plt.subplots(figsize=(13, 6))
metrics[metric_cols].plot(kind='bar', ax=ax)
ax.set_title('Comparación de métricas en la prueba común')
ax.set_ylabel('Valor')
ax.set_xlabel('Modelo')
ax.set_ylim(0, 1)
ax.tick_params(axis='x', rotation=35)
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'metricas_sklearn.png', dpi=150, bbox_inches='tight')
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""En el gráfico se comparan seis preguntas distintas: `accuracy` es la proporción de aciertos; `precision` indica qué proporción de las alertas de default eran correctas; `recall` mide cuántos defaults se detectaron; F1 resume el equilibrio entre precisión y recall; AUC ROC evalúa el ordenamiento para múltiples umbrales; y AUC-PR resume precisión y recall en la clase positiva. La altura de `accuracy` cercana a 0,80 no basta para escoger el modelo: con la prevalencia observada, un predictor que siempre diga “pagado” ya se acerca a ese valor."""))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, model in zip(axes.flat, model_names):
    pred = pd.read_parquet(PRED_DIR / f'test_{model}.parquet')
    ConfusionMatrixDisplay.from_predictions(
        pred['default'], pred['prediction'],
        display_labels=['Fully Paid', 'Charged Off'],
        cmap='Blues', colorbar=False, ax=ax,
    )
    ax.set_title(labels[model])
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'matrices_confusion_sklearn.png', dpi=150, bbox_inches='tight')
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""Las matrices muestran directamente los tipos de error. `HistGradientBoosting` detecta 2.529 de los 53.712 préstamos castigados de prueba (recall 4,7 %) y deja pasar 51.183. `GaussianNB` detecta 29.613 (recall 55,1 %), pero genera 64.897 falsas alarmas entre los préstamos pagados. `LinearSVC` clasifica todos como pagados. Estos resultados corresponden a los umbrales predeterminados; la decisión de umbral debe ajustarse según el costo relativo de dejar pasar un default frente a revisar un préstamo que sí se pagaría."""))

cells.append(nbf.v4.new_code_cell(
"""fig, ax = plt.subplots(figsize=(10, 8))
for model in model_names:
    pred = pd.read_parquet(PRED_DIR / f'test_{model}.parquet')
    RocCurveDisplay.from_predictions(
        pred['default'], pred['score'], name=f\"{labels[model]} (AUC={metrics.loc[model, 'roc_auc']:.4f})\", ax=ax
    )
ax.set_title('Curvas ROC de los seis modelos de scikit-learn')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'curvas_roc_sklearn.png', dpi=150, bbox_inches='tight')
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""Cada curva ROC representa la relación entre la tasa de verdaderos positivos y la de falsas alarmas al variar el umbral. Una curva más cercana a la esquina superior izquierda y con mayor AUC indica mejor capacidad de ordenar un préstamo castigado por encima de uno pagado. El gráfico explica por qué un modelo puede tener AUC razonable y recall muy bajo al umbral fijo: el ranking y el punto operativo son cuestiones diferentes."""))

cells.append(nbf.v4.new_code_cell(
"""timing = metrics[['fit_cv_seconds', 'predict_seconds']].copy()
timing['tiempo_total_segundos'] = timing.sum(axis=1)
timing['modelo'] = [labels[m] for m in timing.index]
display(timing.sort_values('tiempo_total_segundos'))
timing.to_csv(RESULTS_DIR / 'tiempos_sklearn.csv', encoding='utf-8-sig')"""
))

cells.append(nbf.v4.new_markdown_cell(
"""El tiempo total suma ajuste con validación cruzada y predicción en prueba, pero no incluye la lectura ni el preprocesamiento. `GaussianNB` fue el más rápido (3,9 s), mientras que `LinearSVC` tardó cerca de 894,4 s y el árbol 571,8 s. Estos valores dependen de los grids y del hardware local; sirven para dimensionar esta ejecución y se compararán con PySpark usando la misma definición de tiempo en el capítulo final."""))

cells.append(nbf.v4.new_markdown_cell(
"""## 5.2 Nota sobre equivalencia y costo computacional

La guía permite sustituir `GradientBoostingClassifier` por `HistGradientBoostingClassifier` cuando el costo sea prohibitivo con el dataset completo. Se documenta esa sustitución. Para el bosque se usaron 10 árboles y se compararon las profundidades 10 y 15, porque los ensayos con 50 árboles no fueron viables en el hardware disponible; esta decisión debe quedar explícita al interpretar la comparación con PySpark."""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

output = Path("notebooks/05_resultados_sklearn.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, output)
print(f"Creado: {output}")
