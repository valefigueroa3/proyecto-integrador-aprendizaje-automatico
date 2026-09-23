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
    nbf.v4.new_markdown_cell("""## 9.10.5. Reflexión crítica

### ¿Qué entorno fue más rápido y por qué?

Al sumar el entrenamiento y las predicciones de los seis modelos, PySpark tardó cerca de 1.748,5 segundos y scikit-learn unos 2.156,6. En esta ejecución Spark fue aproximadamente 19 % más rápido. La diferencia no fue igual para todos los modelos: scikit-learn terminó antes en regresión logística, Naive Bayes y boosting; Spark lo hizo antes en el árbol, el bosque y LinearSVC.

No concluimos que Spark siempre sea más rápido. Se ejecutó en una sola máquina (`local[4]`), con 16 particiones de shuffle y el vector de datos en caché. Además, scikit-learn usó tres folds de validación cruzada y Spark dos, y las búsquedas de parámetros tampoco fueron idénticas. Los tiempos no incluyen la lectura, el preprocesamiento ni el arranque de Spark. El resultado describe esta prueba y esta configuración; para una comparación definitiva habría que igualar mejor las condiciones.

### ¿Cuál fue más preciso?

La respuesta depende de qué entendamos por “preciso”. Para comparar qué tan bien ordena los préstamos según su riesgo, miramos el AUC ROC: HistGradientBoosting obtuvo 0,7132 y GBT de Spark 0,7127. La diferencia es detectable con DeLong después de Holm, pero es de apenas 0,0005; en la práctica, los dos valores son casi iguales.

Al usar el umbral predeterminado, GBT detectó más préstamos `Charged Off` (recall de 0,0678 frente a 0,0471) y tuvo mejor F1. HistGradientBoosting, en cambio, tuvo una precisión positiva algo mayor (0,5743 frente a 0,5473). La accuracy fue casi idéntica, alrededor de 0,803, pero por sí sola no cuenta toda la historia: la mayoría de los préstamos de prueba fueron pagados. Con estos resultados no elegiríamos un modelo solo porque obtuvo la accuracy más alta; primero habría que decidir qué error preocupa más.

### ¿Qué diferencias de AUC fueron estadísticamente significativas y cuáles son relevantes en la práctica?

Dentro de cada entorno, DeLong encontró diferencias entre el modelo con mayor AUC y las alternativas, incluso después de corregir las comparaciones con Holm. Las diferencias más pequeñas fueron frente a regresión logística: 0,0034 en scikit-learn y 0,0029 en Spark. Sin embargo, una diferencia estadísticamente detectable no necesariamente cambia una decisión real. McNemar, que compara las predicciones al umbral usado, sí detectó la diferencia frente a regresión logística en scikit-learn (p ajustado = 0,0454), pero no en Spark (p ajustado = 0,1724).

También comparamos cada familia entre entornos. DeLong encontró diferencias para el árbol, el bosque, boosting, SVM lineal y Naive Bayes; no encontró una diferencia para regresión logística. Algunas fueron grandes —por ejemplo, 0,2101 en el árbol y 0,1412 en Naive Bayes—, mientras que la de boosting fue solo 0,0005. Para este último caso, DeLong dio p ajustado = 0,034, pero la diferencia sigue siendo muy pequeña para tomarla como una ventaja práctica clara. McNemar solo detectó un cambio de decisiones entre entornos para Naive Bayes.

El bootstrap pareado usó 100 réplicas. Con esa cantidad, sus p-valores tienen poca resolución y, después del ajuste de Holm, ninguno de los contrastes directos entre entornos alcanzó 0,05. Por eso tomamos las pruebas como evidencia complementaria: ayudan a entender las diferencias, pero la importancia práctica también depende del recall, las falsas alarmas y el costo de cada error.

### ¿Qué diferencias de implementación pueden explicar las discrepancias?

Aprendimos que usar el mismo nombre de modelo no significa que los dos programas hagan exactamente lo mismo. PySpark y scikit-learn tienen implementaciones, valores por defecto y formas de buscar los cortes de un árbol que pueden cambiar el resultado. También se deben traducir con cuidado parámetros como `C` y `regParam`; no siempre se pueden igualar solo copiando el número.

Además, en este trabajo variaron los folds y las búsquedas de parámetros. La codificación de categorías, la regularización y el cálculo de las puntuaciones también pueden influir. Las diferencias grandes que vimos en el árbol, SVM y Naive Bayes no se deberían atribuir automáticamente a que una biblioteca sea mejor: para saber la causa habría que repetir la comparación con configuraciones más equivalentes y revisar cómo se construyeron las puntuaciones.

### ¿Qué limitaciones tiene DeLong y cómo la complementan McNemar y bootstrap pareado?

DeLong compara las áreas ROC de dos modelos evaluados sobre los mismos préstamos. Sirve para saber si sus puntuaciones ordenan los casos de forma distinta, pero no dice qué umbral conviene ni cuánto cuesta cada tipo de error. McNemar contesta otra pregunta: compara en cuántos casos los modelos acertaron o fallaron de manera diferente usando el umbral elegido. Por eso puede no coincidir con DeLong.

El bootstrap pareado vuelve a tomar muestras de los mismos casos de prueba y permite observar cuánto podría variar la diferencia de AUC. En nuestro análisis usamos 100 réplicas, un número bajo para estimar p-valores con detalle. Las tres pruebas aportan perspectivas distintas; ninguna, por sí sola, demuestra que un modelo sea mejor para cualquier uso. Tampoco evalúan si el modelo está bien calibrado o si funcionaría igual con préstamos de otros años.

### ¿A partir de qué volumen de datos PySpark supera a scikit-learn?

Con los datos que probamos no podemos fijar un volumen de cruce. Solo tenemos una medición con la población completa de 1.345.310 préstamos elegibles, de los cuales 269.062 quedaron en prueba; en esa ejecución, Spark tardó menos al sumar los seis modelos. No corrimos el mismo análisis con bases progresivamente más pequeñas o más grandes.

Para responder bien habría que repetir ambos procesos con varios tamaños de datos, igualar los modelos y la validación, y medir por separado lectura, preprocesamiento, entrenamiento y predicción. También habría que registrar memoria y recursos. Así se podría estimar dónde cambia la conveniencia en nuestra máquina, sin presentar ese punto como una regla universal.

### ¿Qué aporta LIME y cuáles son sus limitaciones en entornos distribuidos?

LIME nos ayuda a revisar una predicción concreta. Alrededor de un préstamo genera variaciones parecidas y estima qué variables empujan la predicción hacia `Charged Off` o `Fully Paid`. En este proyecto lo usamos con HistGradientBoosting, un fondo de 5.000 filas y 800 perturbaciones.

La explicación sirve para entender ese caso, pero no describe todo el modelo ni demuestra que una variable cause el incumplimiento. Puede cambiar si se modifica el caso de referencia o el número de variaciones. En Spark tampoco se ejecuta automáticamente sobre todo el clúster: hay que llevar a memoria el caso o una muestra pequeña. Por eso usamos LIME como apoyo para interpretar, no como prueba definitiva de cómo funciona el modelo.

### ¿Qué efecto tuvo cada condición obligatoria sobre el rendimiento?

Usar todos los préstamos elegibles hizo más costoso el entrenamiento, pero evitó que la comparación dependiera de escoger una muestra pequeña. La partición estratificada 80/20 mantuvo una proporción similar de préstamos pagados y castigados, y usar la misma prueba para ambos entornos permitió comparar sus predicciones sobre los mismos casos. Ajustar el preprocesamiento solo con los datos de entrenamiento evitó que información de la prueba se filtrara al modelo.

La búsqueda de parámetros con validación cruzada tomó más tiempo, aunque nos permitió elegirlos de forma ordenada usando AUC ROC. En Spark, guardar en caché el vector de variables redujo trabajo repetido entre modelos; eso afectó el tiempo, no la calidad de las predicciones. Las pruebas estadísticas ayudaron a poner las diferencias en contexto y LIME permitió revisar un préstamo concreto. En conjunto, cada requisito tuvo un costo o una utilidad distinta, así que los resultados se deben leer considerando tanto el método como los límites de cómputo."""),
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
    nbf.v4.new_markdown_cell("""La tabla ordena las ejecuciones por AUC ROC, pero no por recall ni por una prioridad operativa. HistGradientBoosting y GBT encabezan el ranking de sus entornos, aunque sus recall de 0,0471 y 0,0678 muestran que detectan una fracción pequeña de los defaults al umbral usado. GaussianNB tiene menor AUC, pero alcanza recall cercano a 0,55 en los dos entornos, a cambio de menor precisión y más falsas alarmas. Por eso no existe una elección única basada en una sola columna: primero debe definirse qué errores son más costosos."""),
    nbf.v4.new_code_cell("""tiempos = pd.DataFrame([
    {'entorno': 'scikit-learn', 'ajuste_y_prediccion_segundos': sk['fit_cv_seconds'].sum() + sk['predict_seconds'].sum()},
    {'entorno': 'PySpark', 'ajuste_y_prediccion_segundos': sp['fit_cv_seconds'].sum() + sp['predict_seconds'].sum()},
])
display(tiempos.round(1))"""),
    nbf.v4.new_markdown_cell("""La suma de los seis ajustes y predicciones fue de unos 1.748,5 segundos en PySpark y 2.156,6 segundos en scikit-learn; Spark tardó aproximadamente 18,9 % menos en esta ejecución. El dato no incluye lectura, preprocesamiento ni inicio del motor y la comparación usa grids y cantidades de folds distintos. Se interpreta como resultado de esta máquina y configuración, no como prueba de que Spark siempre sea más rápido ni como un volumen universal a partir del cual cambia la conveniencia."""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
out = Path('notebooks/08_conclusiones.ipynb')
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f'Creado: {out}')
