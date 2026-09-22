from pathlib import Path
import json
import nbformat as nbf

cells = [
    nbf.v4.new_markdown_cell("""# 8. Conclusiones y limitaciones

## 8.1 Conclusiones

La población de modelado quedó definida por los préstamos aceptados con resultado final conocido: `Fully Paid` y `Charged Off`. Se conservaron todas las observaciones de esa población y se empleó una partición estratificada única, de modo que las comparaciones entre modelos fueran pareadas y reproducibles.

El mejor resultado de scikit-learn correspondió a `HistGradientBoostingClassifier`, con AUC ROC de 0,7132. En PySpark, el mejor resultado correspondió a `GBTClassifier`, con AUC ROC de 0,7127. La cercanía entre ambos resultados respalda la consistencia de la especificación de variables y de la partición común. La accuracy cercana a 0,80 debe interpretarse junto con el recall y el AUC-PR: con el umbral predeterminado, la detección de `Charged Off` sigue siendo limitada.

En la prueba común, los AUC más altos son similares: 0,7132 para HistGradientBoosting y 0,7127 para GBT de Spark. El rendimiento al detectar la clase minoritaria sigue siendo limitado con el umbral predeterminado, por lo que AUC y accuracy no bastan para decidir cuál modelo conviene en una aplicación real. Las pruebas pareadas ayudan a distinguir diferencias en la ordenación de los casos de cambios en las decisiones individuales.

## 8.2 Limitaciones y trabajo futuro

- La fuente disponible cubre 2007–2018 Q4; la guía menciona un horizonte más amplio. La ausencia de 2019–2020 queda como limitación de cobertura.
- Las solicitudes rechazadas no tienen un resultado de pago comparable y por eso no entran en el target binario.
- Por costo computacional se usaron grids reducidos, dos folds en PySpark, 10 árboles en Random Forest, 50 iteraciones en los modelos de boosting y 100 réplicas de bootstrap. El número de réplicas limita la resolución de sus p-valores, por lo que las conclusiones bootstrap se interpretan con cautela.
- La partición principal es aleatoria. Una validación temporal y una calibración de probabilidades serían pasos recomendables antes de usar el modelo en operación.
- La selección se hizo con AUC ROC. Para una decisión de crédito real se debería optimizar el umbral con costos explícitos de falsos negativos y falsos positivos.

## 8.3 Reproducibilidad

La semilla principal es 42. Los artefactos de la partición, preprocesamiento, modelos, predicciones, pruebas estadísticas y figuras quedan en las carpetas `data`, `artifacts`, `results` y `figures`. Los scripts `train_sklearn_models.py`, `run_spark_pipeline.py`, `run_statistical_tests.py` y `run_lime.py` permiten repetir cada etapa."""),
    nbf.v4.new_markdown_cell("""## 9.10.5. Entregable: reflexión crítica solicitada

### ¿Qué entorno fue más rápido y por qué?

Al sumar el ajuste con validación cruzada y la predicción de prueba de los seis modelos, PySpark empleó aproximadamente 1.748,5 segundos y scikit-learn 2.156,6 segundos; el total de Spark fue cerca de 18,9 % menor. Estos tiempos no incluyen lectura de datos, preprocesamiento ni arranque de Spark. La ventaja no fue uniforme: scikit-learn fue más rápido en regresión logística, Naive Bayes y HistGradientBoosting, mientras PySpark fue más rápido en el árbol de decisión y LinearSVC. La comparación también está condicionada por tres folds en scikit-learn frente a dos en PySpark, grids diferentes, la ejecución local `local[4]`, 16 particiones de shuffle y el caché del vector después del preprocesamiento. El caché permitió reutilizar esa representación entre los modelos. El resultado describe esta configuración de hardware y no establece un umbral general de volumen.

### ¿Cuál fue más preciso?

Si “más preciso” se refiere a la capacidad de ordenar correctamente positivos por encima de negativos, medida con AUC ROC, HistGradientBoosting obtuvo el mejor valor (0,7132), seguido por GBT de Spark (0,7127). DeLong pareado con corrección de Holm detectó esa diferencia entre entornos, aunque su magnitud es de solo 0,0005 AUC. En la clasificación al umbral predeterminado, la accuracy fue casi igual (0,8028 y 0,8027); GBT logró mayor recall para `Charged Off` (0,0678 frente a 0,0471) y mayor F1 (0,1207 frente a 0,0870), mientras HistGradientBoosting tuvo mayor precisión positiva (0,5743 frente a 0,5473). Por tanto, el modelo preferible depende de la métrica y del costo de los errores.

### ¿Qué diferencias de AUC fueron estadísticamente significativas y cuáles son relevantes en la práctica?

Después de la corrección de Holm, DeLong detectó diferencias entre el mejor modelo y cada alternativa dentro de ambos entornos. Las comparaciones más cercanas fueron HistGradientBoosting frente a regresión logística (ΔAUC = 0,0034) y GBT frente a regresión logística (ΔAUC = 0,0029). McNemar confirmó esas diferencias en scikit-learn (p ajustado = 0,0454), pero no en PySpark (p ajustado = 0,1724), porque compara decisiones al umbral predeterminado y no las puntuaciones continuas. Los intervalos bootstrap no ajustados del 95 % excluyeron cero para esas comparaciones; sus p-valores ajustados no alcanzaron 0,05 debido a la resolución limitada de 100 réplicas.

También se compararon directamente las seis familias de modelos entre entornos sobre la prueba común. Tras Holm, DeLong encontró diferencias en árbol de decisión (ΔAUC = 0,2101), bosque aleatorio (0,0012), boosting (0,0005), SVM lineal (−0,1543) y Naive Bayes (0,1412); la diferencia de regresión logística (−0,00003) no fue significativa (p ajustado = 0,192). Los signos representan scikit-learn menos PySpark. McNemar solo encontró una diferencia de decisiones para Naive Bayes tras Holm; en los otros modelos, los errores a umbral predeterminado no cambiaron de manera estadísticamente detectable. En las comparaciones directas, ningún p-valor bootstrap sobrevivió Holm con 100 réplicas; por ejemplo, para boosting el intervalo no ajustado fue [0,00019; 0,00094], pero el p ajustado fue 0,119. Así, la diferencia de AUC entre los dos modelos de boosting es detectable con DeLong (p ajustado = 0,034), aunque su tamaño es muy pequeño y la evidencia bootstrap corregida no es concluyente. La relevancia práctica depende de recall, precisión, costos de error y estabilidad; accuracy por sí sola oculta el bajo recall de default.

### ¿Qué diferencias de implementación pueden explicar las discrepancias?

Aunque se emplean nombres equivalentes, las implementaciones no son idénticas. Los árboles de PySpark realizan particiones sobre las variables vectorizadas y están sujetos a parámetros como `maxBins`, mientras que los árboles de scikit-learn usan otra implementación de búsqueda de umbrales. `LinearSVC` puede diferir en la función objetivo, la optimización y la interpretación de `regParam` frente al `C` de `LinearSVC` de scikit-learn. También cambian los valores por defecto, la regularización, la tolerancia, la representación dispersa de las variables one-hot y la cantidad de folds. Estas diferencias explican que modelos con el mismo nombre general no produzcan exactamente las mismas predicciones.

### ¿Qué limitaciones tiene DeLong y cómo la complementan McNemar y bootstrap pareado?

DeLong compara áreas bajo curvas ROC correlacionadas y evalúa si difiere el ordenamiento de probabilidades; no evalúa directamente un umbral, costos de negocio, calibración ni dependencia temporal. McNemar compara aciertos y errores sobre las mismas filas, pero depende del umbral. El bootstrap pareado estima la variabilidad de ΔAUC mediante remuestreo de las mismas observaciones y entrega un intervalo de confianza. En este análisis, DeLong detectó varias diferencias que McNemar no detectó, como boosting frente a regresión logística en PySpark y la comparación de boosting entre entornos. Los intervalos bootstrap sin ajuste incluyeron cero solo para regresión logística entre entornos; no obstante, con 100 réplicas sus p-valores ajustados por Holm no alcanzaron significación. Las pruebas responden preguntas distintas y sus resultados no deben reducirse a un único veredicto.

### ¿A partir de qué volumen de datos PySpark supera a scikit-learn?

Con los experimentos realizados solo existe un punto de comparación: la población completa de 1.345.310 préstamos elegibles, con 269.062 observaciones en prueba. En ese punto y bajo la configuración local usada, el tiempo agregado de PySpark fue menor. No se hicieron corridas con varios tamaños de muestra, número de particiones ni cantidad de trabajadores; por ello no puede estimarse un volumen de cruce general. Para responderlo habría que repetir el mismo pipeline en varios tamaños y registrar el tiempo por etapa, la memoria y el número de ejecutores.

### ¿Qué aporta LIME y cuáles son sus limitaciones en entornos distribuidos?

LIME genera una explicación local: aproxima alrededor de un caso la relación entre las variables y la predicción, indicando qué variables empujan el resultado hacia `Charged Off` o `Fully Paid`. Ayuda a inspeccionar un caso individual y a comunicar el modelo. No es una explicación global ni causal, puede cambiar con el conjunto de referencia y el número de perturbaciones, y las variables one-hot dificultan la lectura directa. En un entorno distribuido tampoco se aplica automáticamente a todo el clúster; se debe seleccionar y materializar un caso o una muestra pequeña. En este trabajo se aplicó al mejor modelo de scikit-learn con un fondo de 5.000 filas y 800 perturbaciones, y sus resultados se interpretan como locales.

### ¿Qué efecto tuvo cada condición obligatoria sobre el rendimiento?

El uso de todas las filas elegibles aumentó el tiempo y la memoria, pero evitó que las métricas dependieran de una muestra. La partición 80/20 estratificada y común permitió comparaciones pareadas válidas. El preprocesamiento ajustado solo con train evitó fuga de información. El caché de Spark redujo recomputaciones de la transformación compartida. `ParamGridBuilder` y `CrossValidator` hicieron explícita la búsqueda de hiperparámetros, a cambio de varias sesiones de entrenamiento. La selección por AUC ROC permitió comparar modelos con escalas de probabilidad diferentes. DeLong, McNemar y bootstrap añadieron evidencia sobre la incertidumbre de las diferencias. Finalmente, LIME permitió revisar la explicación de un caso, aunque sin convertirla en una medida global de importancia."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sk = pd.read_csv(PROJECT_DIR / 'results' / 'sklearn' / 'comparacion_metricas_sklearn.csv')
sp = pd.DataFrame(json.loads((PROJECT_DIR / 'results' / 'spark' / 'comparacion_metricas_spark.json').read_text()))
metric_cols = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
resumen = pd.concat([
    sk[['model', *metric_cols, 'average_precision']].assign(implementacion='scikit-learn'),
    sp[['model', *metric_cols]].assign(average_precision=np.nan, implementacion='PySpark'),
], ignore_index=True).sort_values('roc_auc', ascending=False)
display(resumen.round(4))
resumen.to_csv(PROJECT_DIR / 'results' / 'resumen_final_modelos.csv', index=False, encoding='utf-8-sig')"""),
    nbf.v4.new_code_cell("""tiempos = pd.DataFrame([
    {'entorno': 'scikit-learn', 'ajuste_y_prediccion_segundos': sk['fit_cv_seconds'].sum() + sk['predict_seconds'].sum()},
    {'entorno': 'PySpark', 'ajuste_y_prediccion_segundos': sp['fit_cv_seconds'].sum() + sp['predict_seconds'].sum()},
])
display(tiempos.round(1))"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
out = Path('notebooks/08_conclusiones.ipynb')
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f'Creado: {out}')
