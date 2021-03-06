# Inicio
- [Bases](#bases)
    + [Variables](#variables)
- [Regresion Lineal](#regresion-lineal)
    + [Descomposicion Cholesky o LU](#descomposicion-cholesky-o-lu)
    + [Rateos de aprendizaje](#rateos-de-aprendizaje)
    + [Regresion Lineal Deming](#regresion-lineal-deming)
- [Regresion Logistica](#regresion-logistica)
- [SVM](#svm)
<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Instalacion

INstaacion **Tensorflow 2.0**
```sh
conda create --name py3-TF2.0 python=3
conda activate py3-TF2.0
conda install tensorflow
pip install --upgrade tensorflow
pip install ipykernel
```

# Bases
<img src="images/6.png"><br />
Introduce el concepto de TPU (Tensor Processing Units) que mejora la performance<br />
<img src="images/7.png"><br />

**Se enfoca en redes neuronales**<br />
TensorFlow es mejor para redes neuronales que scikit learn<br />
TF introduce en nuevas versiones a Keras, en el 2015 era muy complejo y tenia una curva de aprendizaje complicada<br />
TensorFlow 2 es basicamente Keras, toma lo mejor de Keras y TF1<br />
Tensorflow no maneja los datos en xlsx o csv, porque necesita manejarlo como tensores, en multiple dimensiones, una solucion es NPZ files<br />
Existen 11 pasos:
- Importación o generación del conjunto de datos.
- Transformación y normalización de los datos.
- Dividir el conjunto de datos en conjunto de entrenamiento, de validación y de test.
- Definir los hiperparámetros del algoritmo
- Inicializar variables y placeholders
- Definir la estructura del modelo del algoritmo.
- Declarar la función de pérdidas (loss function)
- Inicializar y entrenar el modelo anterior.
- Evaluación del modelo
- Ajustar los hiper parámetros
- Publicar (subir a producción) y predecir nuevos resultados
```py

x = tf.constant(30) # Constante

 # variable, float32, siempre vectores de 3 coordenadas x,y,z
x_input = tf.placeholder(tf.float32, [None, 3]) # vectores de 3
y_input = tf.placeholder(tf.float32, [None, 5]) # vectores de 5

# add -> sumar
# multiply -> multiplicar
y_pred = tf.add(tf.multiply(m_matrix, x_input), n_vector)

# Rellena con ceros 3 filas, 4 columnas y 6 de profundidad
zero_t = tf.zeros([3, 4, 6])

# Relleno con 1988
filled_t = tf.fill([4,5,2], 1988)

# Vector de ceros con el mismo tamaño que cte_t
zero_sim = tf.zeros_like(cte_t)

# Vector de uno con el mismo tamaño que cte_t
ones_sim = tf.ones_like(cte_t)
```


### Variables

```py
# Convertir a variable
tf.Variable(rand_norm_t)

# Convertir a tensor
tf.convert_to_tensor(1988)
```

# Regresion Lineal

At * A -> Siempre da una matriz cuadrada<br />

De todas las rectas posibles cual es la que minimiza mas la distancia<br />
<img src="images/3.png"><br />

### Regresion Lineal Multiple
<img src="images/6.png"><br />
<img src="images/7.png"><br />

### Descomposicion Cholesky o LU
El problema esta cuando hay matrices muy grandes, eso nos ayuda a tener mayo eficiencia que el <br />
$$x = (A^TA)^{-1}A^Tb$$ <br />
Obtener un conjunto de matrices a partir de una matriz y operar con ellas, aca es donde viene LU<br />
<img src="images/1.png"><br />

### Rateos de aprendizaje
<img src="images/2.png"><br />

### Regresion Lineal Deming
Lo que lo diferencia de la RL es la tecnica que utilizareos,
minimizaermos la proyeccion con respecto a la recta, minimizamos la recta en perpendicular, **NO MINIMIZA EL ERROR EN Y, MINIMIZA EL ERROR TANTO EN X COMO EN Y**<br />
<img src="images/4.png"><br />

# Regresion Logistica
Probabilidad de pertenecer a un conjunto de datos, es si o no<br />

# SVM
Seria lo que engloba la parte de la linea de la regresion, separaremos distancia entre las categorias para minimizar el error<br />
<img src="images/5.png"><br />