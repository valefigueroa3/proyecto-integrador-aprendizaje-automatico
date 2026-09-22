# Proyecto integrador de aprendizaje automático

Comparación de modelos de clasificación de default en scikit-learn y PySpark usando
el archivo completo `accepted_2007_to_2018Q4.csv`.

## Datos

El archivo original debe permanecer en:

```text
Tarea1/accepted_2007_to_2018Q4.csv
```

No se incluye una copia dentro del proyecto. El universo de modelado se define con los
préstamos cuyo resultado final es `Fully Paid` o `Charged Off`. Esta selección define la
población compatible con la variable objetivo; no corresponde a muestreo.

## Orden de trabajo

1. `notebooks/01_carga_y_auditoria.ipynb`: carga, validación del archivo y definición del target.
2. EDA unidimensional y bidimensional.
3. Preprocesamiento y partición común.
4. Modelado con scikit-learn.
5. Modelado con PySpark.
6. DeLong, McNemar y bootstrap pareado.
7. Interpretabilidad con LIME.
8. Comparación y conclusiones.

Los modelos se pueden reproducir con `train_sklearn_models.py` y
`run_spark_pipeline.py`. Después de tener las predicciones, las pruebas pareadas
se generan con `run_statistical_tests.py` y la explicación local con `run_lime.py`.
En Windows, Spark se ejecuta con `TEMP` y `TMP` apuntando a una carpeta corta
como `C:\\jtmp` para evitar el límite de rutas de sockets de Java.

## Entorno

Crear el entorno desde esta carpeta:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Abrir los notebooks:

```powershell
.venv/Scripts/jupyter-lab.exe
```

Construir el libro HTML:

```powershell
.venv/Scripts/jupyter-book.exe build .
```

El libro generado queda en `_build/html/index.html`.

## Publicación con GitHub y Jupyter Book

La entrega se publica como un repositorio de GitHub. Se deben subir los notebooks,
scripts, configuración, resultados, figuras, artefactos procesados, `README.md` y
la carpeta `.github/workflows`. El CSV original de 1,7 GB, `.venv`, `.uv-cache` y
`_build` no se suben.

Después del primer `push` a la rama `main` (o `master`), la acción
`.github/workflows/deploy-book.yml` construye el libro con Jupyter Book y lo
publica en GitHub Pages. En el repositorio se debe abrir **Settings → Pages** y
seleccionar **Source: GitHub Actions**. La dirección pública aparecerá en
**Actions** y en **Settings → Pages**.

Para crear la copia local del repositorio y publicarla después de crear un
repositorio vacío en GitHub:

```powershell
cd Tarea1/proyecto
git init
git add .
git commit -m "Proyecto integrador de aprendizaje automático"
git branch -M main
git remote add origin https://github.com/USUARIO/NOMBRE-DEL-REPOSITORIO.git
git push -u origin main
```

El enlace que se entrega al profesor debe ser el de GitHub Pages; el enlace del
repositorio también puede incluirse para que se revisen los notebooks y scripts.
