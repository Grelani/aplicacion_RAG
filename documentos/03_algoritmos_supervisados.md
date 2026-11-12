# Algoritmos de aprendizaje supervisado

Documento de práctica original para la aplicación RAG.

## K vecinos más cercanos (KNN)

KNN predice la salida de una observación comparándola con los ejemplos de entrenamiento más cercanos de acuerdo con una métrica de distancia. En clasificación, la categoría suele definirse mediante el voto de los k vecinos más próximos. En regresión, se puede promediar el valor de esos vecinos.

El parámetro k controla cuántos vecinos participan en la predicción. Con k pequeño, el resultado puede ser sensible al ruido; con k grande, se suavizan las decisiones y pueden perderse patrones locales. Para el ejemplo didáctico del curso, utilizaremos k = 5, sin afirmar que ese valor sea universalmente óptimo.

KNN depende fuertemente del escalado de variables cuando se utilizan distancias euclidianas. La selección de k debe realizarse con datos de validación, no con datos de prueba.

## Árboles de decisión

Un árbol de decisión utiliza reglas sucesivas para dividir los datos en grupos relativamente homogéneos. Cada nodo interno contiene una condición y cada hoja representa una predicción. Los árboles son interpretables, pero si crecen sin restricciones pueden sobreajustarse.

Para controlar la complejidad, se usan opciones como profundidad máxima, número mínimo de ejemplos por hoja y poda.

## Naive Bayes

Naive Bayes utiliza el teorema de Bayes y supone independencia condicional entre las variables predictoras dada la clase. Aunque este supuesto puede no cumplirse estrictamente, el método suele ser una línea base útil y rápida, especialmente en ciertas tareas de clasificación de texto.

## Regresión logística

A pesar de su nombre, la regresión logística se usa habitualmente para clasificación. En su versión binaria calcula una probabilidad de pertenencia a una clase mediante una función logística. La decisión final depende de un umbral de clasificación, que puede modificarse según el costo de los errores.

## Selección del modelo

No existe un algoritmo que siempre sea superior. La selección debe considerar el tipo y volumen de datos, calidad de etiquetas, posibilidad de explicar predicciones, tiempo de cómputo y medidas de evaluación relevantes. Conviene comparar modelos con la misma partición de datos y un protocolo reproducible.
