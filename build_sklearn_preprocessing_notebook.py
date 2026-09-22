from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# 4. Preprocesamiento con scikit-learn

Este capítulo transforma la base de modelado usando la partición común. Todos los componentes que aprenden parámetros —imputadores, escalador y codificador— se ajustan exclusivamente con entrenamiento. El conjunto de prueba permanece aislado hasta la evaluación final."""
))

cells.append(nbf.v4.new_code_cell(
"""from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_PATH = PROJECT_DIR / 'data' / 'processed' / 'lending_club_model_base.parquet'
ARTIFACTS_DIR = PROJECT_DIR / 'artifacts' / 'sklearn'
RESULTS_DIR = PROJECT_DIR / 'results' / 'preprocessing'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_DIR / 'data' / 'processed' / 'manifest.json', encoding='utf-8') as stream:
    manifest = json.load(stream)

numeric_features = manifest['numeric_features']
categorical_features = manifest['categorical_features']
features = numeric_features + categorical_features
print(f'Variables numéricas: {numeric_features}')
print(f'Variables categóricas: {categorical_features}')"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 4.1 Lectura de la base y separación mediante `split`

No se llama nuevamente a `train_test_split`. La columna `split` contiene la asignación única creada en el capítulo anterior."""
))

cells.append(nbf.v4.new_code_cell(
"""df = pd.read_parquet(DATA_PATH, columns=['id', 'default', 'split'] + features)

train = df.loc[df['split'] == 'train'].copy()
test = df.loc[df['split'] == 'test'].copy()

X_train_raw = train[features]
y_train = train['default'].to_numpy(dtype=np.int8)
id_train = train['id'].astype(str).to_numpy()

X_test_raw = test[features]
y_test = test['default'].to_numpy(dtype=np.int8)
id_test = test['id'].astype(str).to_numpy()

assert len(train) == manifest['rows_train']
assert len(test) == manifest['rows_test']
assert df['id'].is_unique

print(f'Train: {X_train_raw.shape} | default: {y_train.mean():.4%}')
print(f'Test:  {X_test_raw.shape} | default: {y_test.mean():.4%}')"""
))

cells.append(nbf.v4.new_markdown_cell(
"""Las proporciones de `Charged Off` son prácticamente iguales en entrenamiento y prueba (19,9626 % y 19,9627 %). Por eso la evaluación no parte de una prueba artificialmente más sencilla o difícil. La división se toma de la columna `split` ya creada, de modo que los dos entornos conservan exactamente las mismas observaciones."""))

cells.append(nbf.v4.new_markdown_cell(
"""## 4.2 Transformadores

- Numéricas: imputación por mediana y estandarización.
- Categóricas: categoría explícita `Missing` y one-hot encoding.
- Categorías desconocidas en prueba: se ignoran de forma segura mediante `handle_unknown='ignore'`.

El `Pipeline` que aparece dentro de cada grupo solo organiza transformaciones. La búsqueda de hiperparámetros de los modelos se realizará después con `GridSearchCV` sobre matrices ya transformadas, tal como exige la guía."""
))

cells.append(nbf.v4.new_code_cell(
"""numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features),
    ],
    sparse_threshold=1.0,
    verbose_feature_names_out=False,
)

fit_start = time.perf_counter()
X_train = preprocessor.fit_transform(X_train_raw)
fit_transform_seconds = time.perf_counter() - fit_start

transform_start = time.perf_counter()
X_test = preprocessor.transform(X_test_raw)
test_transform_seconds = time.perf_counter() - transform_start

X_train = sparse.csr_matrix(X_train, dtype=np.float32)
X_test = sparse.csr_matrix(X_test, dtype=np.float32)
feature_names = preprocessor.get_feature_names_out()

assert X_train.shape[0] == len(y_train)
assert X_test.shape[0] == len(y_test)
assert X_train.shape[1] == X_test.shape[1] == len(feature_names)
assert np.isfinite(X_train.data).all()
assert np.isfinite(X_test.data).all()

print(f'X_train: {X_train.shape}, nnz={X_train.nnz:,}')
print(f'X_test:  {X_test.shape}, nnz={X_test.nnz:,}')
print(f'Ajuste + transformación train: {fit_transform_seconds:.2f} s')
print(f'Transformación test: {test_transform_seconds:.2f} s')"""
))

cells.append(nbf.v4.new_markdown_cell(
"""El preprocesamiento genera 100 columnas: cinco numéricas y 95 indicadores one-hot para las categorías. La matriz es dispersa: cada fila solo activa unas pocas categorías, por lo que guardar los ceros explícitamente desperdiciaría memoria. Se conservaron 1.076.248 filas de entrenamiento y 269.062 de prueba; el ajuste y la transformación de entrenamiento tomaron 10,24 segundos y aplicar las reglas ya aprendidas a prueba tomó 1,31 segundos."""))

cells.append(nbf.v4.new_markdown_cell("## 4.3 Parámetros aprendidos únicamente de entrenamiento"))

cells.append(nbf.v4.new_code_cell(
"""numeric_imputer = preprocessor.named_transformers_['numeric'].named_steps['imputer']
scaler = preprocessor.named_transformers_['numeric'].named_steps['scaler']
onehot = preprocessor.named_transformers_['categorical'].named_steps['onehot']

numeric_parameters = pd.DataFrame({
    'variable': numeric_features,
    'mediana_imputacion_train': numeric_imputer.statistics_,
    'media_scaler_train': scaler.mean_,
    'escala_train': scaler.scale_,
})
display(numeric_parameters)
numeric_parameters.to_csv(
    RESULTS_DIR / 'parametros_numericos_train.csv', index=False, encoding='utf-8-sig'
)

category_summary = pd.DataFrame({
    'variable': categorical_features,
    'n_categorias_aprendidas_train': [len(values) for values in onehot.categories_],
    'categorias': [' | '.join(map(str, values)) for values in onehot.categories_],
})
display(category_summary)
category_summary.to_csv(
    RESULTS_DIR / 'categorias_aprendidas_train.csv', index=False, encoding='utf-8-sig'
)"""
))

cells.append(nbf.v4.new_markdown_cell(
"""La tabla numérica documenta los valores que aprendió el preprocesador solo con entrenamiento: por ejemplo, los faltantes de ingreso se reemplazan por 65.000, los de monto por 12.000 y los de DTI por 17,62. La media y la escala se usan para estandarizar las cinco variables. La tabla categórica confirma qué niveles conoce el codificador; una categoría nueva que aparezca en prueba no detiene el proceso y se representa sin activar indicadores aprendidos. Ninguno de esos parámetros se estima a partir de la prueba."""))

cells.append(nbf.v4.new_markdown_cell("## 4.4 Persistencia de matrices y metadatos"))

cells.append(nbf.v4.new_code_cell(
"""sparse.save_npz(ARTIFACTS_DIR / 'X_train.npz', X_train, compressed=True)
sparse.save_npz(ARTIFACTS_DIR / 'X_test.npz', X_test, compressed=True)
np.save(ARTIFACTS_DIR / 'y_train.npy', y_train)
np.save(ARTIFACTS_DIR / 'y_test.npy', y_test)
np.save(ARTIFACTS_DIR / 'id_train.npy', id_train)
np.save(ARTIFACTS_DIR / 'id_test.npy', id_test)
np.save(ARTIFACTS_DIR / 'feature_names.npy', feature_names)
joblib.dump(preprocessor, ARTIFACTS_DIR / 'preprocessor.joblib', compress=3)

timings = pd.DataFrame([{
    'entorno': 'scikit-learn',
    'etapa': 'preprocesamiento',
    'fit_transform_train_segundos': fit_transform_seconds,
    'transform_test_segundos': test_transform_seconds,
    'n_train': len(y_train),
    'n_test': len(y_test),
    'n_features_salida': len(feature_names),
}])
timings.to_csv(RESULTS_DIR / 'tiempos_preprocesamiento_sklearn.csv', index=False, encoding='utf-8-sig')

artifact_checks = {
    'X_train_shape': list(sparse.load_npz(ARTIFACTS_DIR / 'X_train.npz').shape),
    'X_test_shape': list(sparse.load_npz(ARTIFACTS_DIR / 'X_test.npz').shape),
    'y_train_rows': int(len(np.load(ARTIFACTS_DIR / 'y_train.npy'))),
    'y_test_rows': int(len(np.load(ARTIFACTS_DIR / 'y_test.npy'))),
    'feature_count': int(len(np.load(ARTIFACTS_DIR / 'feature_names.npy', allow_pickle=True))),
}
with open(ARTIFACTS_DIR / 'preprocessing_manifest.json', 'w', encoding='utf-8') as stream:
    json.dump(artifact_checks, stream, indent=2)

display(timings)
display(pd.DataFrame({'feature': feature_names}).head(30))
print('Todos los artefactos fueron reabiertos y verificados.')"""
))

cells.append(nbf.v4.new_markdown_cell(
"""Las matrices, etiquetas, identificadores y nombres de variables se guardan por separado para conservar la correspondencia entre filas y predicciones. El preprocesador también se serializa para poder aplicar exactamente las mismas transformaciones en una ejecución futura. La lectura de vuelta verifica dimensiones, número de etiquetas y variables; el resultado confirma que los artefactos se escribieron íntegramente y están listos para el entrenamiento."""))

cells.append(nbf.v4.new_markdown_cell(
"""## 4.5 Consideraciones para los modelos

- Regresión logística y LinearSVC utilizarán directamente las matrices escaladas.
- Los árboles son invariantes a transformaciones monótonas de escala, por lo que pueden usar la misma representación para mantener una entrada común.
- `GaussianNB` requiere una matriz densa; su entrenamiento se manejará de manera específica y se vigilará el uso de memoria.
- `GradientBoostingClassifier` puede resultar costoso con el conjunto completo. Si resulta prohibitivo, se documentará la sustitución permitida por `HistGradientBoostingClassifier`.
- La codificación se ajustó una sola vez usando entrenamiento. El conjunto de prueba no participó en ninguna estadística aprendida."""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

output = Path("notebooks/04_preprocesamiento_sklearn.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, output)
print(f"Creado: {output}")
