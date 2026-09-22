# Predicción de default en Lending Club

Este libro presenta el desarrollo y la comparación de modelos para clasificar préstamos
de Lending Club como pagados o castigados. El análisis se realizó con el conjunto completo
de préstamos elegibles y se documentan las decisiones para que el proceso pueda revisarse
y reproducirse.

## Alcance del proyecto

| Elemento | Definición |
|:--|:--|
| **Fuente** | Préstamos aceptados, 2007–2018 Q4 |
| **Población analizada** | 1.345.310 préstamos con resultado final conocido |
| **Variable objetivo** | `default = 1` para `Charged Off`; `default = 0` para `Fully Paid` |
| **Partición** | Entrenamiento 80 % y prueba 20 %, estratificada y compartida por ambos entornos |
| **Herramientas** | scikit-learn y PySpark ML |
| **Evaluación** | AUC ROC, métricas de clasificación, tiempos y pruebas estadísticas pareadas |
| **Interpretabilidad** | Explicación local con LIME |

## Recorrido

El libro comienza auditando la fuente y describiendo las variables, define una partición
común y documenta el preprocesamiento. Luego presenta el ajuste de seis clasificadores en
cada entorno, evalúa sus resultados y aplica comparaciones pareadas con DeLong, McNemar y
bootstrap. El capítulo final sintetiza los hallazgos, limitaciones y reflexiones críticas
solicitadas para la entrega.
