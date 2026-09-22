# Predicción de default en Lending Club

Este Jupyter Book documenta la construcción y comparación de seis clasificadores en
scikit-learn y PySpark sobre el conjunto completo de préstamos elegibles de Lending Club.

El archivo fuente disponible corresponde a préstamos aceptados entre 2007 y el cuarto
trimestre de 2018. La población de modelado se limita a préstamos con resultado final
conocido: `Fully Paid` o `Charged Off`. No se realiza muestreo de esta población.

El proyecto comparará desempeño predictivo, tiempo de cómputo e inferencia estadística
mediante las pruebas de DeLong, McNemar y bootstrap pareado. Finalmente, se explicarán
predicciones locales con LIME.

