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

Se reutiliza la misma población elegible y la misma partición estratificada definida en el capítulo 3. El ajuste se ejecuta con `run_spark_pipeline.py`; esta página presenta los pasos del pipeline, la búsqueda realizada y sus resultados. La prueba común contiene 1.076.248 observaciones de entrenamiento y 269.062 de prueba.

## 6.1 Preprocesamiento distribuido

Los transformadores se ajustan con las filas de entrenamiento. `StringIndexer` convierte cada categoría en un código y `OneHotEncoder` crea indicadores, para que el código no se interprete como un orden numérico. El vector de características transformado se almacena en caché para reutilizarlo durante el ajuste de los estimadores y sus folds. El escalador usa `withMean=False` para conservar la representación dispersa.

| Tipo de variable | Transformación |
|:--|:--|
| Numéricas | Imputación por mediana con `Imputer` y estandarización con `StandardScaler` |
| Categóricas | `StringIndexer` con categorías inválidas admitidas y codificación `OneHotEncoder` |
| Ensamble | `VectorAssembler` combina columnas numéricas y codificadas en `features` |
| Caché | Vectores de entrenamiento y prueba cacheados después de completar el pipeline de transformaciones |

Las predicciones distribuidas de prueba se guardan para calcular las comparaciones pareadas. Solo ese conjunto de predicciones se materializa localmente para persistirlo; la base de entrenamiento no se convierte con `toPandas()` ni se recoge en el driver.

## 6.2 Modelos y búsqueda de hiperparámetros

`ParamGridBuilder` genera las combinaciones y `CrossValidator` las evalúa con AUC ROC en dos folds. Se usa paralelismo 1 para mantener estable el consumo de memoria en el equipo local.

| Modelo | Parámetros evaluados |
|:--|:--|
| Regresión logística | `regParam`: 0,000001 y 0,0001; `elasticNetParam`: 0; `maxIter`: 100 |
| Árbol de decisión | `maxDepth`: 5 y 10 |
| Bosque aleatorio | `numTrees`: 10; `maxDepth`: 10 y 15 |
| Gradient boosted tree | `maxIter`: 50; `maxDepth`: 3 y 5 |
| SVM lineal | `regParam`: 0,000001 y 0,0001; `maxIter`: 100 |
| Naive Bayes gaussiano | `smoothing`: 0,1 y 1,0 |

`maxDepth` limita la complejidad del árbol; `numTrees` fija cuántos árboles contiene el bosque; `maxIter` limita iteraciones o etapas de boosting; `regParam` controla la penalización de los modelos lineales; y `smoothing` estabiliza las varianzas de Naive Bayes. `elasticNetParam: 0` deja la regresión logística con penalización L2. La validación cruzada escoge entre los valores listados, y luego cada combinación elegida se evalúa en la prueba reservada.

La configuración no replica exactamente la búsqueda de scikit-learn: allí se usaron tres folds para los modelos con grid; PySpark usa dos. Naive Bayes en scikit-learn se ajustó sin búsqueda de hiperparámetros. Estas diferencias se consideran al interpretar los tiempos y al comparar modelos de nombre equivalente."""),
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
    nbf.v4.new_markdown_cell("""El mayor AUC ROC corresponde a `GBTClassifier` (0,7127), seguido de regresión logística (0,7099) y bosque aleatorio (0,7068). El árbol de decisión obtiene 0,4939, cercano al azar, y Naive Bayes 0,5255. `LinearSVC` alcanza un AUC de 0,6301, pero al umbral que produjo `prediction` no identificó positivos: sus 269.062 decisiones fueron “pagado”. Esto muestra que el AUC mide el ordenamiento de puntuaciones y no la calidad de una única decisión al umbral predeterminado. Naive Bayes invierte el perfil: detecta 55,6 % de los defaults, pero con 31,3 % de precisión y 66,8 % de accuracy, debido a muchas falsas alarmas. La columna `best_params` señala la combinación elegida con validación cruzada en entrenamiento; `roc_auc` y las demás métricas evalúan el modelo ya elegido en la prueba común. La métrica preferible depende del costo de cada tipo de error."""),
    nbf.v4.new_code_cell("""if len(metrics_spark):
    ax = metrics_spark['roc_auc'].sort_values().plot(kind='barh', figsize=(9, 5), color='#2a9d8f')
    ax.set_title('AUC ROC de PySpark ML en la prueba común')
    ax.set_xlabel('AUC ROC')
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(PROJECT_DIR / 'figures' / 'spark_auc.png', dpi=150); plt.show()"""),
    nbf.v4.new_markdown_cell("""La gráfica ordena los modelos por capacidad de ranking y no por cantidad de aciertos a un solo umbral. `GBTClassifier` obtiene el mayor AUC del entorno; su ventaja sobre regresión logística es pequeña (cerca de 0,0029). La diferencia grande entre los AUC de los árboles de decisión de ambos entornos se analizará con las pruebas pareadas y debe leerse junto con la advertencia de que sus implementaciones y búsquedas no son idénticas."""),
    nbf.v4.new_markdown_cell("""## 6.3 Tiempo de ejecución y alcance de la medición

Los tiempos reportados corresponden al ajuste y validación con `CrossValidator`, seguidos de la predicción sobre prueba. No incluyen la lectura del archivo ni el tiempo de inicio de Spark. Se usaron grids acotados, dos folds, 10 árboles en Random Forest y 50 iteraciones en GBT para que el cálculo completo fuera viable; no se redujo el número de filas. En Windows, Spark se ejecuta con una ruta temporal corta (`C:\\jtmp`) por el límite de rutas de sockets de Java."""),
    nbf.v4.new_code_cell("""spark_timing = metrics_spark[['fit_cv_seconds', 'predict_seconds']].copy()
spark_timing['total_seconds'] = spark_timing.sum(axis=1)
display(spark_timing.sort_values('total_seconds'))"""),
    nbf.v4.new_markdown_cell("""El ajuste y validación de GBT es la etapa más costosa (885,6 s), seguido por el bosque aleatorio (398,5 s); Naive Bayes tarda cerca de 72,6 s sumando ajuste y predicción. La tabla permite ver que el tiempo no se deduce del AUC: un modelo más lento no necesariamente obtiene la mejor métrica para todas las necesidades. La suma de estos seis tiempos se compara con scikit-learn en el capítulo 8, bajo las mismas exclusiones de lectura, preprocesamiento e inicio del entorno."""),
]
write(Path('notebooks/06_modelado_spark.ipynb'), spark_cells)

stats_cells = [
    nbf.v4.new_markdown_cell("""# 7. Comparaciones estadísticas e interpretabilidad

Las comparaciones son pareadas porque todos los modelos predicen exactamente las mismas observaciones de prueba. `run_statistical_tests.py` compara el mejor AUC contra sus alternativas dentro de cada entorno y también contrasta las seis familias de modelos entre scikit-learn y PySpark. Se aplican DeLong a las puntuaciones continuas, McNemar exacta a las decisiones y bootstrap pareado a las diferencias de AUC. La corrección de Holm se aplica por separado a cada familia de pruebas. El bootstrap usa 100 réplicas sobre todas las observaciones; su resolución es limitada y se interpreta junto con los intervalos de confianza.

DeLong pregunta si dos modelos asignan puntuaciones con una capacidad de ordenamiento distinta en los mismos préstamos. McNemar se fija en las decisiones al umbral usado y cuenta en qué casos solo uno de los modelos acierta. El bootstrap remuestrea en pareja las filas de prueba para estimar un intervalo de la diferencia entre AUC. Holm ajusta los p-valores cuando se hacen varias comparaciones y así reduce falsos hallazgos por multiplicidad."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
STATS_DIR = PROJECT_DIR / 'results' / 'estadisticas'
for name in ['sklearn', 'spark']:
    file = STATS_DIR / f'comparaciones_{name}.csv'
    if file.exists():
        print(name)
        table = pd.read_csv(file).sort_values('delong_holm_p_value')
        table = table.rename(columns={
            'model_1': 'Mejor modelo', 'model_2': 'Alternativa',
            'delong_delta_auc': 'Δ AUC', 'delong_holm_p_value': 'p DeLong (Holm)',
            'mcnemar_holm_p_value': 'p McNemar (Holm)',
            'bootstrap_ci95_low': 'IC bootstrap 95 % inferior',
            'bootstrap_ci95_high': 'IC bootstrap 95 % superior',
            'bootstrap_holm_p_value': 'p bootstrap (Holm)',
        })
        shown = table[[
            'Mejor modelo', 'Alternativa', 'Δ AUC', 'p DeLong (Holm)',
            'p McNemar (Holm)', 'IC bootstrap 95 % inferior',
            'IC bootstrap 95 % superior', 'p bootstrap (Holm)',
        ]].round(5)
        p_columns = ['p DeLong (Holm)', 'p McNemar (Holm)', 'p bootstrap (Holm)']
        for column in p_columns:
            shown[column] = shown[column].map(
                lambda value: '<1e-5' if pd.notna(value) and value < 1e-5 else f'{value:.5f}'
            )
        display(shown)"""),
    nbf.v4.new_markdown_cell("""En scikit-learn, HistGradientBoosting presenta el mayor AUC y DeLong con Holm distingue su puntuación de cada alternativa, incluida regresión logística (diferencia de 0,00339). McNemar también encuentra diferencia en las decisiones frente a regresión logística, aunque es cercana al límite de 0,05 (p ajustado 0,04542). En PySpark, GBT es el mejor por AUC; DeLong lo separa de sus alternativas, pero McNemar no distingue sus decisiones de las de regresión logística (p ajustado 0,17243). Los intervalos bootstrap sin ajustar no incluyen cero, pero los p-valores bootstrap corregidos son 0,09901 en las dos familias. Con solo 100 réplicas, la resolución del bootstrap no alcanza para declarar significación tras Holm. Estas tablas comparan al modelo líder con cada alternativa; no son una prueba de todas las parejas posibles."""),
    nbf.v4.new_markdown_cell("""## 7.1 Comparación entre entornos

La tabla siguiente compara implementaciones de la misma familia sobre idénticas filas de prueba. `Δ AUC` se define como AUC de scikit-learn menos AUC de PySpark: un valor positivo favorece scikit-learn. Los valores ajustados de Holm permiten identificar diferencias estadísticas sin ignorar que se hicieron seis comparaciones."""),
    nbf.v4.new_code_cell("""cross_file = STATS_DIR / 'comparaciones_entornos.csv'
if cross_file.exists():
    cross = pd.read_csv(cross_file).sort_values('delong_holm_p_value')
    cross_table = cross.rename(columns={
        'familia_modelo': 'Familia de modelo',
        'delong_auc_1': 'AUC scikit-learn', 'delong_auc_2': 'AUC PySpark',
        'delong_delta_auc': 'Δ AUC (sklearn − Spark)',
        'delong_holm_p_value': 'p DeLong (Holm)',
        'mcnemar_holm_p_value': 'p McNemar (Holm)',
        'bootstrap_ci95_low': 'IC bootstrap 95 % inferior',
        'bootstrap_ci95_high': 'IC bootstrap 95 % superior',
        'bootstrap_holm_p_value': 'p bootstrap (Holm)',
    })
    shown_cross = cross_table[[
        'Familia de modelo', 'AUC scikit-learn', 'AUC PySpark',
        'Δ AUC (sklearn − Spark)', 'p DeLong (Holm)', 'p McNemar (Holm)',
        'IC bootstrap 95 % inferior', 'IC bootstrap 95 % superior',
        'p bootstrap (Holm)',
    ]].round(5)
    p_columns = ['p DeLong (Holm)', 'p McNemar (Holm)', 'p bootstrap (Holm)']
    for column in p_columns:
        shown_cross[column] = shown_cross[column].map(
            lambda value: '<1e-5' if pd.notna(value) and value < 1e-5 else f'{value:.5f}'
        )
    display(shown_cross)"""),
    nbf.v4.new_markdown_cell("""La diferencia de regresión logística entre entornos es −0,00003 y no resulta significativa con DeLong (p ajustado 0,19163). En cambio, DeLong detecta diferencias para árbol (+0,21008 a favor de scikit-learn), SVM lineal (−0,15429, a favor de Spark), Naive Bayes (+0,14123), bosque aleatorio (+0,00121) y boosting (+0,00050). Las diferencias de árbol, SVM y Naive Bayes son grandes en AUC; las de bosque y boosting son muy pequeñas aunque detectables en esta prueba extensa. McNemar solo señala diferencia de decisiones para Naive Bayes tras Holm. Para las seis comparaciones, el bootstrap con 100 réplicas no produce p-valores ajustados menores que 0,05; sus intervalos y valores se leen con la resolución limitada documentada."""),
    nbf.v4.new_markdown_cell("""La significación estadística no mide por sí sola el valor práctico de la diferencia. Las implementaciones, los grids y el número de folds no son idénticos; una diferencia pequeña puede resultar detectable con una prueba de 269.062 observaciones y, aun así, tener un efecto operativo limitado. McNemar puede discrepar de DeLong porque evalúa clases predichas a un umbral, mientras que DeLong compara el ordenamiento de las puntuaciones."""),
    nbf.v4.new_markdown_cell("""## 7.2 Interpretabilidad con LIME

La explicación local se genera para `HistGradientBoostingClassifier`, que fue el mejor modelo de scikit-learn por AUC ROC. Se usan 5.000 observaciones como fondo de perturbación de LIME y 800 perturbaciones alrededor de un préstamo con `Charged Off`; este fondo solo sirve para interpretar el modelo y no cambia el entrenamiento."""),
    nbf.v4.new_code_cell("""lime_file = PROJECT_DIR / 'results' / 'lime' / 'explicacion_local_hist_gradient_boosting.csv'
if lime_file.exists():
    lime_table = pd.read_csv(lime_file)
    display(lime_table.head(12))"""),
    nbf.v4.new_markdown_cell("""En la tabla, el signo se refiere a la probabilidad local de `Charged Off` para el caso seleccionado: valores positivos la empujan hacia default y valores negativos la alejan. En este préstamo, vivir en arriendo, tener ingreso menor o una tasa más alta aportan peso positivo; un plazo de 36 meses, FICO mayor que 714, DTI moderado y monto menor aportan peso negativo. Son contribuciones del modelo alrededor de este único caso, no efectos causales ni reglas universales. La salida de LIME no mide importancia global y puede variar con el préstamo, el conjunto de referencia y las perturbaciones. En un entorno distribuido se materializa solo el caso y una muestra pequeña; no se explica todo el clúster automáticamente."""),
    nbf.v4.new_code_cell("""from IPython.display import Image, display
lime_fig = PROJECT_DIR / 'figures' / 'lime' / 'lime_hist_gradient_boosting.png'
if lime_fig.exists(): display(Image(filename=str(lime_fig)))"""),
]
write(Path('notebooks/07_estadisticas_lime.ipynb'), stats_cells)
print('Notebooks restantes creados')
