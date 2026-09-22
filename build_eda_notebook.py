from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# 2. Análisis exploratorio de datos

Este capítulo analiza la población completa de préstamos con resultado final conocido. Los cálculos descriptivos y las pruebas estadísticas utilizan todas las observaciones elegibles; no se realiza muestreo. Se priorizan variables conocidas al momento de originar el préstamo, porque las variables de pagos posteriores producirían fuga de información."""
))

cells.append(nbf.v4.new_code_cell(
"""from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style='whitegrid', context='notebook')
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

candidates = [
    Path('../accepted_2007_to_2018Q4.csv'),
    Path('../../accepted_2007_to_2018Q4.csv'),
    Path('Tarea1/accepted_2007_to_2018Q4.csv'),
]
CSV_PATH = next((p.resolve() for p in candidates if p.exists()), None)
if CSV_PATH is None:
    raise FileNotFoundError('No se encontró accepted_2007_to_2018Q4.csv')

PROJECT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
FIGURES_DIR = PROJECT_DIR / 'figures' / 'eda'
RESULTS_DIR = PROJECT_DIR / 'results' / 'eda'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f'Fuente: {CSV_PATH}')"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2.1 Variables seleccionadas

Las variables principales provienen de la guía. Se agregan `grade`, `term`, `issue_d` y `verification_status` por su utilidad descriptiva. No se usan variables como `total_pymnt`, `recoveries`, `last_pymnt_d` o `out_prncp`, porque solo se conocen después de otorgar el préstamo y revelarían directamente su resultado."""
))

cells.append(nbf.v4.new_code_cell(
"""numeric_cols = [
    'loan_amnt', 'int_rate', 'annual_inc', 'dti',
    'fico_range_high',
]
categorical_cols = [
    'emp_length', 'purpose', 'home_ownership', 'addr_state',
    'verification_status', 'grade', 'term',
]
usecols = ['id', 'loan_status', 'issue_d'] + numeric_cols + categorical_cols
target_map = {'Fully Paid': 0, 'Charged Off': 1}

parts = []
for chunk in pd.read_csv(CSV_PATH, usecols=usecols, chunksize=150_000, low_memory=False):
    valid_id = chunk['id'].astype('string').str.fullmatch(r'\\d+', na=False)
    eligible = valid_id & chunk['loan_status'].isin(target_map)
    if eligible.any():
        part = chunk.loc[eligible].copy()
        part['default'] = part['loan_status'].map(target_map).astype('int8')
        parts.append(part)

df = pd.concat(parts, ignore_index=True)
del parts
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')

print(f'Filas elegibles: {len(df):,}')
print(f'Columnas analizadas: {df.shape[1]}')
display(df.head())
display(df.tail())
df.info(memory_usage='deep')"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2.2 Calidad, tipos y valores faltantes

Se revisan tipos, cobertura, número de categorías y valores faltantes antes de definir el preprocesamiento. La tabla se calcula sobre todas las observaciones elegibles; por tanto, sirve para identificar problemas de calidad sin usar todavía información de la partición de prueba para ajustar parámetros."""))

cells.append(nbf.v4.new_code_cell(
"""quality = pd.DataFrame({
    'tipo': df.dtypes.astype(str),
    'n_no_nulos': df.notna().sum(),
    'n_nulos': df.isna().sum(),
    'pct_nulos': 100 * df.isna().mean(),
    'n_unicos': df.nunique(dropna=True),
}).sort_values('pct_nulos', ascending=False)
display(quality)
quality.to_csv(RESULTS_DIR / 'calidad_variables.csv', encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(10, 6))
missing_plot = quality.loc[quality['pct_nulos'] > 0].sort_values('pct_nulos')
ax.barh(missing_plot.index, missing_plot['pct_nulos'], color='#4C78A8')
ax.set(title='Porcentaje de valores faltantes', xlabel='Porcentaje', ylabel='Variable')
ax.axvline(30, color='darkorange', linestyle='--', label='Umbral orientativo: 30 %')
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'valores_faltantes.png', dpi=150, bbox_inches='tight')
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2.3 Distribución de la variable objetivo

La distribución de `default` determina el grado de desbalance y orienta la selección de métricas. Se reportan ambas clases antes de entrenar para evitar interpretar una accuracy alta como evidencia suficiente de capacidad predictiva."""))

cells.append(nbf.v4.new_code_cell(
"""target_distribution = (
    df['default'].value_counts().sort_index().rename('n').to_frame()
    .assign(porcentaje=lambda x: 100 * x['n'] / x['n'].sum())
)
target_distribution.index = ['Fully Paid (0)', 'Charged Off (1)']
display(target_distribution)
target_distribution.to_csv(RESULTS_DIR / 'distribucion_target.csv', encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(data=df, x='default', hue='default', palette=['#59A14F', '#E15759'], legend=False, ax=ax)
ax.set(title='Distribución de la variable objetivo', xlabel='Default', ylabel='Número de préstamos')
ax.set_xticks([0, 1], ['Fully Paid (0)', 'Charged Off (1)'])
for container in ax.containers:
    ax.bar_label(container, fmt='{:,.0f}')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'distribucion_target.png', dpi=150, bbox_inches='tight')
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""La clase de default es minoritaria, pero representa cerca de una quinta parte de la población. La partición deberá ser estratificada y la evaluación no debe depender solo de accuracy; se reportarán recall, F1, AUC ROC y AUC-PR."""
))

cells.append(nbf.v4.new_markdown_cell("## 2.4 Análisis unidimensional de variables numéricas"))

cells.append(nbf.v4.new_code_cell(
"""desc = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
desc['mediana'] = df[numeric_cols].median()
desc['IQR'] = desc['75%'] - desc['25%']
desc['limite_inferior_IQR'] = desc['25%'] - 1.5 * desc['IQR']
desc['limite_superior_IQR'] = desc['75%'] + 1.5 * desc['IQR']
desc['n_outliers_IQR'] = {
    col: int(((df[col] < desc.loc[col, 'limite_inferior_IQR']) |
              (df[col] > desc.loc[col, 'limite_superior_IQR'])).sum())
    for col in numeric_cols
}
desc['pct_outliers_IQR'] = 100 * desc['n_outliers_IQR'] / desc['count']
desc['asimetria'] = df[numeric_cols].skew()
display(desc)
desc.to_csv(RESULTS_DIR / 'resumen_numericas.csv', encoding='utf-8-sig')"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(14, 4 * len(numeric_cols)))
for i, col in enumerate(numeric_cols):
    values = df[col].dropna()
    sns.histplot(values, bins=50, ax=axes[i, 0], color='#4C78A8')
    axes[i, 0].set_title(f'Distribución de {col}')
    sns.boxplot(x=values, ax=axes[i, 1], color='#F28E2B', showfliers=True)
    axes[i, 1].set_title(f'Boxplot de {col}')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'distribuciones_numericas.png', dpi=150, bbox_inches='tight')
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""La asimetría y el porcentaje de outliers se revisan para decidir transformaciones. `annual_inc` presenta una cola derecha marcada. En la implementación final se conserva la especificación común de imputación por mediana y estandarización; no se aplica `log1p`, de modo que el efecto de la transformación no se mezcla con la comparación entre modelos."""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2.5 Análisis unidimensional de variables categóricas

Para cada variable se cuentan categorías y se marca la proporción de niveles raros. Esta revisión permite decidir si una categoría explícita para faltantes y `handle_unknown='ignore'` son suficientes para el preprocesamiento."""))

cells.append(nbf.v4.new_code_cell(
"""categorical_tables = {}
for col in categorical_cols:
    table = df[col].fillna('<MISSING>').value_counts(dropna=False).rename('n').to_frame()
    table['porcentaje'] = 100 * table['n'] / table['n'].sum()
    table['categoria_rara_menor_1pct'] = table['porcentaje'] < 1
    categorical_tables[col] = table
    print(f'\\n{col}: {len(table)} categorías')
    display(table.head(20))
    table.to_csv(RESULTS_DIR / f'frecuencias_{col}.csv', encoding='utf-8-sig')"""
))

cells.append(nbf.v4.new_code_cell(
"""plot_cols = ['emp_length', 'purpose', 'home_ownership', 'verification_status', 'grade', 'term']
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
for ax, col in zip(axes.flat, plot_cols):
    order = df[col].fillna('<MISSING>').value_counts().index
    sns.countplot(data=df.assign(**{col: df[col].fillna('<MISSING>')}), y=col, order=order, ax=ax, color='#4C78A8')
    ax.set_title(f'Distribución de {col}')
    ax.set_xlabel('Número de préstamos')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'distribuciones_categoricas.png', dpi=150, bbox_inches='tight')
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2.6 Variables numéricas frente a default

Se comparan las distribuciones de cada variable entre préstamos pagados y castigados. La prueba de Mann–Whitney se utiliza por la asimetría observada, mientras que la correlación punto-biserial resume la dirección y magnitud de la relación con la clase."""))

cells.append(nbf.v4.new_code_cell(
"""numeric_tests = []
for col in numeric_cols:
    clean = df[[col, 'default']].dropna()
    paid = clean.loc[clean['default'] == 0, col]
    defaulted = clean.loc[clean['default'] == 1, col]
    # Con este tamaño, se usa Mann-Whitney por robustez ante asimetría y outliers.
    statistic, p_value = stats.mannwhitneyu(paid, defaulted, alternative='two-sided')
    r_pb, p_pb = stats.pointbiserialr(clean['default'], clean[col])
    numeric_tests.append({
        'variable': col,
        'media_paid': paid.mean(),
        'media_default': defaulted.mean(),
        'diferencia_medias': defaulted.mean() - paid.mean(),
        'mediana_paid': paid.median(),
        'mediana_default': defaulted.median(),
        'mannwhitney_U': statistic,
        'p_mannwhitney': p_value,
        'correlacion_punto_biserial': r_pb,
        'p_punto_biserial': p_pb,
    })
numeric_tests = pd.DataFrame(numeric_tests).set_index('variable')
display(numeric_tests)
numeric_tests.to_csv(RESULTS_DIR / 'pruebas_numericas_vs_default.csv', encoding='utf-8-sig')"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 4 * len(numeric_cols)))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(data=df, x='default', y=col, hue='default', palette=['#59A14F', '#E15759'], legend=False, ax=ax, showfliers=False)
    ax.set_title(f'{col} según resultado del préstamo (sin mostrar puntos extremos)')
    ax.set_xticks([0, 1], ['Fully Paid', 'Charged Off'])
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'boxplots_numericas_vs_default.png', dpi=150, bbox_inches='tight')
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2.7 Variables categóricas frente a default

La prueba chi-cuadrado evalúa asociación global y V de Cramer aporta una medida de tamaño del efecto. Las tasas por categoría se muestran como apoyo descriptivo; no se convierten directamente en reglas de clasificación."""))

cells.append(nbf.v4.new_code_cell(
"""categorical_tests = []
default_rate_tables = {}
for col in categorical_cols:
    series = df[col].fillna('<MISSING>')
    contingency = pd.crosstab(series, df['default'])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt((chi2 / n) / min_dim) if min_dim > 0 else np.nan
    categorical_tests.append({
        'variable': col,
        'chi2': chi2,
        'grados_libertad': dof,
        'p_value': p_value,
        'V_Cramer': cramers_v,
    })
    rates = df.assign(_category=series).groupby('_category')['default'].agg(['count', 'mean'])
    rates['tasa_default_pct'] = 100 * rates['mean']
    rates = rates.sort_values('tasa_default_pct', ascending=False)
    default_rate_tables[col] = rates
    rates.to_csv(RESULTS_DIR / f'tasa_default_{col}.csv', encoding='utf-8-sig')

categorical_tests = pd.DataFrame(categorical_tests).set_index('variable')
display(categorical_tests)
categorical_tests.to_csv(RESULTS_DIR / 'chi2_categoricas_vs_default.csv', encoding='utf-8-sig')
display(default_rate_tables['grade'])"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, col in zip(axes.flat, ['grade', 'term', 'home_ownership', 'verification_status']):
    rates = default_rate_tables[col].sort_values('tasa_default_pct')
    ax.barh(rates.index.astype(str), rates['tasa_default_pct'], color='#E15759')
    ax.set_title(f'Tasa de default por {col}')
    ax.set_xlabel('Default (%)')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'tasas_default_categoricas.png', dpi=150, bbox_inches='tight')
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2.8 Correlación y posibles redundancias

La matriz de Pearson se usa para detectar redundancia lineal entre numéricas y una posible relación excesiva con el target. El umbral `|r| > 0.7` se toma como señal exploratoria, no como criterio único para eliminar variables."""))

cells.append(nbf.v4.new_code_cell(
"""corr = df[numeric_cols + ['default']].corr(method='pearson')
display(corr)
corr.to_csv(RESULTS_DIR / 'correlaciones_pearson.csv', encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax)
ax.set_title('Matriz de correlaciones de Pearson')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'correlaciones_pearson.png', dpi=150, bbox_inches='tight')
plt.show()

high_corr = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack().rename('r').reset_index()
        .query('abs(r) > 0.7')
)
display(high_corr if not high_corr.empty else pd.DataFrame({'resultado': ['No se encontraron pares con |r| > 0.7.']}))"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2.9 Evolución temporal

Se agrupan los préstamos por año de originación para observar cambios en volumen y tasa de default. Este análisis no modifica la partición aleatoria, pero advierte que una validación temporal adicional podría ser necesaria para medir estabilidad fuera del periodo observado."""))

cells.append(nbf.v4.new_code_cell(
"""temporal = (
    df.dropna(subset=['issue_d'])
      .assign(anio=lambda x: x['issue_d'].dt.year)
      .groupby('anio')['default']
      .agg(n='size', tasa_default='mean')
)
temporal['tasa_default_pct'] = 100 * temporal['tasa_default']
display(temporal)
temporal.to_csv(RESULTS_DIR / 'evolucion_temporal.csv', encoding='utf-8-sig')

fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.bar(temporal.index, temporal['n'], color='#4C78A8', alpha=0.7)
ax1.set(xlabel='Año de originación', ylabel='Número de préstamos')
ax2 = ax1.twinx()
ax2.plot(temporal.index, temporal['tasa_default_pct'], color='#E15759', marker='o')
ax2.set_ylabel('Default (%)')
ax1.set_title('Volumen y tasa de default por año de originación')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'evolucion_temporal.png', dpi=150, bbox_inches='tight')
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2.10 Resumen ejecutivo y decisiones preliminares

Los resultados anteriores deben interpretarse atendiendo magnitud y relevancia práctica. Con más de 1,3 millones de observaciones, valores p extremadamente pequeños son esperables incluso cuando las diferencias son modestas.

Decisiones para la siguiente etapa:

- Mantener la partición estratificada común 80/20 mediante `id` y semilla fija.
- Ajustar imputadores, codificadores y escaladores solo con entrenamiento.
- Tratar las categorías ausentes explícitamente y evaluar la agrupación de categorías con menos de 1 %.
- Mantener `annual_inc` con imputación por mediana y estandarización en la comparación principal; `log1p` queda como análisis de sensibilidad futuro.
- Conservar métricas sensibles al desbalance, en especial recall, F1, AUC ROC y AUC-PR.
- Excluir todas las variables posteriores al otorgamiento para evitar fuga de información.
- Revisar estabilidad temporal, pues la composición de préstamos y su madurez cambian entre años."""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

output = Path("notebooks/02_eda.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, output)
print(f"Creado: {output}")
