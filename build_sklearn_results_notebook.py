from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# 5. Resultados de scikit-learn

Este capítulo consolida las seis ejecuciones sobre la misma prueba común. Las métricas de clasificación usan el umbral predeterminado de cada modelo: 0,5 para probabilidades y 0 para `decision_function` de `LinearSVC`. El AUC utiliza la puntuación continua y no depende de un umbral."""
))

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
"""La mayor AUC ROC corresponde a `HistGradientBoostingClassifier` (0,7132), seguido por regresión logística (0,7098) y bosque aleatorio (0,7080). La diferencia entre los tres primeros es pequeña en términos absolutos, por lo que la selección final se complementa con las pruebas pareadas del capítulo 7. El recall permanece bajo para estos modelos con el umbral predeterminado; una aplicación operativa requeriría elegir el umbral según el costo de falsos negativos."""))

cells.append(nbf.v4.new_markdown_cell(
"""El AUC es la métrica de selección solicitada. Accuracy debe interpretarse con cuidado porque la clase `Fully Paid` representa aproximadamente 80 % de la población; por eso también se reportan recall, F1 y AUC-PR."""
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

cells.append(nbf.v4.new_code_cell(
"""timing = metrics[['fit_cv_seconds', 'predict_seconds']].copy()
timing['tiempo_total_segundos'] = timing.sum(axis=1)
timing['modelo'] = [labels[m] for m in timing.index]
display(timing.sort_values('tiempo_total_segundos'))
timing.to_csv(RESULTS_DIR / 'tiempos_sklearn.csv', encoding='utf-8-sig')"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 5.1 Nota sobre equivalencia y costo computacional

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
