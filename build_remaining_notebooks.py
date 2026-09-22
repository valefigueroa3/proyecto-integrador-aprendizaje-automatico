from pathlib import Path
import nbformat as nbf


def write(path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


spark_cells = [
    nbf.v4.new_markdown_cell("""# 6. Modelado con PySpark ML

Se reutiliza la misma población elegible y la misma partición estratificada de la sección anterior. El entrenamiento está implementado en `run_spark_pipeline.py`: transforma numéricas y categóricas con `Imputer`, `StringIndexer`, `OneHotEncoder`, `VectorAssembler` y `StandardScaler`; después deja el `DataFrame` cacheado tras `VectorAssembler` y ajusta seis estimadores mediante `ParamGridBuilder` y `CrossValidator`. No se usa `toPandas` para la base: solo se materializan las predicciones de prueba, necesarias para las pruebas pareadas."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RESULTS_DIR = PROJECT_DIR / 'results' / 'spark'
ART_DIR = PROJECT_DIR / 'artifacts' / 'spark'
models = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosted_tree', 'linear_svc', 'naive_bayes']
labels = {
    'logistic_regression': 'Regresión logística', 'decision_tree': 'Árbol de decisión',
    'random_forest': 'Bosque aleatorio', 'gradient_boosted_tree': 'GBT',
    'linear_svc': 'SVM lineal', 'naive_bayes': 'Naive Bayes gaussiano',
}
rows = []
for model in models:
    file = RESULTS_DIR / f'metrics_{model}.json'
    if file.exists():
        row = json.loads(file.read_text(encoding='utf-8'))
        row['nombre'] = labels[model]
        rows.append(row)
metrics_spark = pd.DataFrame(rows).set_index('model')
display(metrics_spark)
metrics_spark.to_csv(RESULTS_DIR / 'comparacion_metricas_spark.csv', encoding='utf-8-sig')"""),
    nbf.v4.new_markdown_cell("""En PySpark, el mayor AUC ROC corresponde a `GBTClassifier` (0,7127), seguido de regresión logística (0,7099) y bosque aleatorio (0,7068). La comparación se realiza sobre la misma prueba común y con el mismo criterio AUC; las diferencias de recall y F1 muestran que el umbral predeterminado produce comportamientos distintos frente a la clase minoritaria."""),
    nbf.v4.new_code_cell("""if len(metrics_spark):
    ax = metrics_spark['roc_auc'].sort_values().plot(kind='barh', figsize=(9, 5), color='#2a9d8f')
    ax.set_title('AUC ROC de PySpark ML en la prueba común')
    ax.set_xlabel('AUC ROC')
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(PROJECT_DIR / 'figures' / 'spark_auc.png', dpi=150); plt.show()"""),
    nbf.v4.new_markdown_cell("""La ejecución se configuró con dos folds y grids reducidos para que el cálculo fuera viable en el equipo disponible; esto no modifica la población ni la partición. El bosque usa 10 árboles y el GBT 50 iteraciones, de forma análoga a las adaptaciones documentadas para scikit-learn. Spark se ejecuta con una ruta temporal corta (`C:\\jtmp`) en Windows por el límite de rutas de sockets de Java."""),
]
write(Path('notebooks/06_modelado_spark.ipynb'), spark_cells)

stats_cells = [
    nbf.v4.new_markdown_cell("""# 7. Comparaciones estadísticas e interpretabilidad

Las comparaciones son pareadas porque todos los modelos predicen exactamente las mismas observaciones de prueba. `run_statistical_tests.py` calcula DeLong para diferencias de AUC, McNemar exacta para las decisiones binarias y un bootstrap pareado de la diferencia de AUC. Los p-valores de cada familia se ajustan con Holm. El bootstrap usa 20 réplicas multinomiales sobre las observaciones completas; el número se limita por el coste de ordenar repetidamente 269.062 predicciones."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
STATS_DIR = PROJECT_DIR / 'results' / 'estadisticas'
for name in ['sklearn', 'spark']:
    file = STATS_DIR / f'comparaciones_{name}.csv'
    if file.exists():
        print(name)
        display(pd.read_csv(file).sort_values('delong_holm_p_value').head(10))"""),
    nbf.v4.new_markdown_cell("""## 7.1 LIME

La explicación local se genera para `HistGradientBoostingClassifier`, que fue el mejor modelo de scikit-learn por AUC ROC. Se usan 5.000 observaciones como fondo de perturbación de LIME y 800 perturbaciones alrededor de un préstamo con `Charged Off`; este fondo solo sirve para interpretar el modelo y no cambia el entrenamiento."""),
    nbf.v4.new_code_cell("""lime_file = PROJECT_DIR / 'results' / 'lime' / 'explicacion_local_hist_gradient_boosting.csv'
if lime_file.exists():
    lime_table = pd.read_csv(lime_file)
    display(lime_table.head(12))"""),
    nbf.v4.new_markdown_cell("""La salida de LIME es una explicación local, no una medida de importancia global ni una relación causal. En el caso analizado, las condiciones de plazo, vivienda, ingreso, FICO, DTI e interés son las variables que más desplazan la puntuación hacia `Charged Off` o `Fully Paid` alrededor de esa observación."""),
    nbf.v4.new_code_cell("""from IPython.display import Image, display
lime_fig = PROJECT_DIR / 'figures' / 'lime' / 'lime_hist_gradient_boosting.png'
if lime_fig.exists(): display(Image(filename=str(lime_fig)))"""),
]
write(Path('notebooks/07_estadisticas_lime.ipynb'), stats_cells)
print('Notebooks restantes creados')
