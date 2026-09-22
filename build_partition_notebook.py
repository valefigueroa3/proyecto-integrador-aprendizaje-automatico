from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# 3. Partición común y base de modelado

La prueba de DeLong y las comparaciones pareadas requieren que todos los modelos se evalúen sobre exactamente las mismas observaciones. En este capítulo se crea una sola partición 80/20, estratificada por clase y con semilla fija. La asignación se guarda en Parquet y será leída tanto por scikit-learn como por PySpark."""
))

cells.append(nbf.v4.new_code_cell(
"""from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

candidates = [
    Path('../accepted_2007_to_2018Q4.csv'),
    Path('../../accepted_2007_to_2018Q4.csv'),
    Path('Tarea1/accepted_2007_to_2018Q4.csv'),
]
CSV_PATH = next((p.resolve() for p in candidates if p.exists()), None)
if CSV_PATH is None:
    raise FileNotFoundError('No se encontró accepted_2007_to_2018Q4.csv')

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
RESULTS_DIR = PROJECT_DIR / 'results' / 'preprocessing'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
print(f'Python: {platform.python_version()} | pandas: {pd.__version__} | sklearn: {sklearn.__version__}')"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 3.1 Variables y control de fuga de información

La base conserva variables disponibles al momento de la solicitud u originación. Se excluyen explícitamente resultados posteriores como pagos acumulados, recuperaciones, saldo pendiente, fecha o monto del último pago y estado de hardship. También se excluyen campos de texto libre o identificadores que no deben actuar como predictores."""
))

cells.append(nbf.v4.new_code_cell(
"""numeric_features = [
    'loan_amnt', 'int_rate', 'fico_range_high', 'annual_inc', 'dti',
]
categorical_features = [
    'emp_length', 'purpose', 'home_ownership', 'addr_state',
    'verification_status', 'grade', 'term',
]
metadata_cols = ['id', 'loan_status', 'issue_d']
selected_cols = metadata_cols + numeric_features + categorical_features

post_outcome_examples = [
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
    'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
    'next_pymnt_d', 'last_fico_range_high', 'last_fico_range_low',
    'debt_settlement_flag', 'settlement_status', 'settlement_amount',
]

feature_catalog = pd.DataFrame({
    'variable': numeric_features + categorical_features,
    'tipo': ['numérica'] * len(numeric_features) + ['categórica'] * len(categorical_features),
    'momento': 'Disponible al originarse el préstamo',
})
display(feature_catalog)
feature_catalog.to_csv(RESULTS_DIR / 'catalogo_variables_modelado.csv', index=False, encoding='utf-8-sig')"""
))

cells.append(nbf.v4.new_markdown_cell("## 3.2 Carga completa de la población elegible"))

cells.append(nbf.v4.new_code_cell(
"""target_map = {'Fully Paid': 0, 'Charged Off': 1}
parts = []

for chunk in pd.read_csv(CSV_PATH, usecols=selected_cols, chunksize=150_000, low_memory=False):
    numeric_id = chunk['id'].astype('string').str.fullmatch(r'\\d+', na=False)
    eligible = numeric_id & chunk['loan_status'].isin(target_map)
    if eligible.any():
        part = chunk.loc[eligible].copy()
        part['id'] = part['id'].astype('string')
        part['default'] = part['loan_status'].map(target_map).astype('int8')
        parts.append(part)

model_base = pd.concat(parts, ignore_index=True)
del parts

assert len(model_base) == 1_345_310, 'Cambió el número esperado de préstamos elegibles.'
assert model_base['id'].is_unique, 'El identificador id debe ser único.'
assert set(model_base['default'].unique()) == {0, 1}

print(f'Filas: {len(model_base):,}')
print(f'IDs únicos: {model_base.id.nunique():,}')
display(model_base.head())"""
))

cells.append(nbf.v4.new_markdown_cell("## 3.3 Creación de la partición estratificada"))

cells.append(nbf.v4.new_code_cell(
"""indices = np.arange(len(model_base))
train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=model_base['default'].to_numpy(),
)

split_values = np.full(len(model_base), 'train', dtype=object)
split_values[test_idx] = 'test'

common_split = model_base[['id', 'default']].copy()
common_split['split'] = split_values

assert len(train_idx) + len(test_idx) == len(model_base)
assert np.intersect1d(train_idx, test_idx).size == 0
assert common_split['id'].is_unique
assert not common_split['split'].isna().any()

split_summary = (
    common_split.groupby(['split', 'default'])
    .size().rename('n').reset_index()
)
split_summary['porcentaje_dentro_split'] = (
    100 * split_summary['n'] / split_summary.groupby('split')['n'].transform('sum')
)
display(split_summary)
split_summary.to_csv(RESULTS_DIR / 'resumen_particion.csv', index=False, encoding='utf-8-sig')"""
))

cells.append(nbf.v4.new_markdown_cell("## 3.4 Persistencia y comprobación de reproducibilidad"))

cells.append(nbf.v4.new_code_cell(
"""split_path = PROCESSED_DIR / 'common_split.parquet'
base_path = PROCESSED_DIR / 'lending_club_model_base.parquet'

common_split.to_parquet(split_path, index=False, compression='snappy')
model_base_with_split = model_base.merge(
    common_split[['id', 'split']], on='id', how='left', validate='one_to_one'
)
model_base_with_split.to_parquet(base_path, index=False, compression='snappy')

# Reapertura: valida los artefactos realmente escritos, no solo los objetos en memoria.
split_check = pd.read_parquet(split_path)
base_check = pd.read_parquet(base_path, columns=['id', 'default', 'split'])

assert split_check.equals(common_split)
assert len(base_check) == len(model_base)
assert base_check['id'].is_unique
assert base_check['split'].value_counts().to_dict() == common_split['split'].value_counts().to_dict()

manifest = {
    'source_file': CSV_PATH.name,
    'eligible_statuses': target_map,
    'random_state': RANDOM_STATE,
    'test_size': TEST_SIZE,
    'rows_total': int(len(common_split)),
    'rows_train': int((common_split['split'] == 'train').sum()),
    'rows_test': int((common_split['split'] == 'test').sum()),
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'split_file': split_path.name,
    'model_base_file': base_path.name,
}
with open(PROCESSED_DIR / 'manifest.json', 'w', encoding='utf-8') as stream:
    json.dump(manifest, stream, ensure_ascii=False, indent=2)

print(f'Partición guardada: {split_path} ({split_path.stat().st_size / 1024**2:.2f} MiB)')
print(f'Base guardada: {base_path} ({base_path.stat().st_size / 1024**2:.2f} MiB)')
display(split_check.head())"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 3.5 Uso obligatorio en ambos entornos

**scikit-learn:** leer `lending_club_model_base.parquet` y separar mediante la columna `split`. Los imputadores, codificadores y escaladores se ajustarán solo sobre `split == 'train'`.

**PySpark:** leer la misma base Parquet o unir el CSV con `common_split.parquet` mediante `id`, y filtrar por `split`. No se utilizará `randomSplit`, porque produciría observaciones diferentes.

La base Parquet conserva todas las filas elegibles. La reducción de columnas responde a la selección documentada de predictores y a la exclusión de fuga de información; no es muestreo."""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

output = Path("notebooks/03_particion_comun.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, output)
print(f"Creado: {output}")
