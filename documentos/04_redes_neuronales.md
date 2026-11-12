# Redes neuronales y entrenamiento

Texto didáctico original. La descripción presenta conceptos básicos y no pretende abarcar todas las arquitecturas.

## Neurona artificial

Una neurona artificial calcula una combinación ponderada de entradas, añade un sesgo y aplica una función de activación. Una red neuronal suele contener una capa de entrada, capas ocultas y una capa de salida. Durante el entrenamiento se ajustan pesos y sesgos para reducir una función de pérdida.

## Funciones de activación

La función ReLU, unidad lineal rectificada, se define como max(0, x). Devuelve cero si la entrada es negativa y la entrada misma si es positiva. Es habitual en capas ocultas porque es sencilla y ayuda a introducir no linealidad.

La función sigmoide transforma valores reales al intervalo entre 0 y 1, por lo que se usa en determinados problemas de salida binaria. Softmax convierte un conjunto de puntuaciones en valores no negativos que suman uno y se emplea con frecuencia en clasificación multiclase.

## Propagación hacia adelante

En el paso forward, los datos atraviesan las capas y se obtiene una predicción. Después se calcula la pérdida, que cuantifica la discrepancia entre predicción y objetivo.

## Retropropagación

La retropropagación, o backpropagation, aplica la regla de la cadena para calcular cómo cambia la pérdida respecto a los pesos de la red. El optimizador usa estos gradientes para modificar los parámetros. El descenso del gradiente y Adam son ejemplos de estrategias de optimización.

## Épocas, lotes y tasa de aprendizaje

Una época corresponde a un recorrido por los ejemplos del conjunto de entrenamiento. Un lote o batch es un subconjunto procesado antes de actualizar los parámetros. La tasa de aprendizaje determina, en términos generales, el tamaño de los cambios de los parámetros durante la optimización.

## Redes convolucionales

Las redes convolucionales procesan estructuras como imágenes aplicando filtros que detectan patrones locales. Las primeras capas pueden captar bordes o texturas; capas posteriores pueden combinar patrones de mayor complejidad. El pooling o las convoluciones con stride pueden reducir la resolución espacial y modificar el campo receptivo efectivo.

## Transfer learning

El aprendizaje por transferencia reutiliza un modelo previamente entrenado para una tarea nueva. Se puede utilizar como extractor de características o ajustar parte de sus parámetros mediante fine-tuning. Suele ser útil cuando no se dispone de un gran conjunto de datos propio.
