# Resumen Librerías
Notas para futuras consultas

## NumPy
Para arreglos multidimensionales

*pip install numpy*

*import numpy as np*

**np.array()** crea un arreglo n-dimensional

*a = np.array([1, 2, 3, 4])*

**np.arange(start, stop, step)** crea un array con valores igualmente espaciados

*np.arange(0, 10, 2)  # array([0, 2, 4, 6, 8])*

**np.linspace(start, stop, num)** crea un arreglo con 'num' valores, igualmente espaciados entre 'start' y 'stop'

*np.linspace(0, 1, 5)  # array([0. , 0.25, 0.5 , 0.75, 1. ])*

**np.zeros(shape)** y **np.ones(shape)** crean un array lleno de ceros y unos, respectivamente, de la forma 'shape'

*np.zeros((2, 3))  # array([[0., 0., 0.], [0., 0., 0.]])*

**np.eye(n)** crea una matriz identidad de tamaño n

*np.eye(3)  # array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])*

Las operaciones aritméticas se realizan elemento a elemento:

*a = np.array([1, 2, 3])*

*b = np.array([4, 5, 6])*

*c = a + b  # array([5, 7, 9])*

**np.sin()**, **np.cos()**, **np.exp()**, **np.sqrt()**, etc.

*np.sqrt([1, 4, 9])  # array([1., 2., 3.])*

Maneja indexación y slicing para acceder a subarreglos

*a = np.array([1, 2, 3, 4, 5])*

*a[1:4]  # array([2, 3, 4])*

**mean()**, promedio; **std()**, desviación estándar; **sum()** suma de los elementos

*a = np.array([1, 2, 3])*

*a.mean()  # 2.0*


## Matplotlib
Para visualización de gráficos en 2D (de líneas, barras, dispersión, histogramas, etc.)

*pip install matplotlib*

*import matplotlib.pyplot as plt* 

(**pyplot** ofrece una interfaz similar a MATLAB)

*x = [1, 2, 3, 4, 5]*

*y = [2, 3, 5, 7, 11]*

Gráfico de línea.

*plt.plot(x, y)*

*plt.plot(x, y, marker='o')  # marker='o' añade puntos en el gráfico*

Gráfico de barras

*plt.bar(x, y)*

Gráfico de dispersión

*plt.scatter(x, y)*

Histograma

*plt.hist(x, bins=6)  # Crear un histograma con 6 bins de x*

Nombrar etiquetas, título y mostrar gráfico

*plt.xlabel('X-axis')*

*plt.ylabel('Y-axis')*

*plt.title('Simple Plot')*

*plt.show()*

Para personalización de gráficos:

Leyendas

*plt.plot(x, y, label='Line')*

*plt.legend()  # Añadir una leyenda*

Colores y estilos

*plt.plot(x, y, color='red', linestyle='--', marker='x')  # Personalización del color y estilo de línea*

Subgráficas

*fig, axs = plt.subplots(2, 2)  # Crear una figura con 4 subgráficas*
*axs[0, 0].plot(x, y)*

*axs[0, 1].bar(x, y)*

*axs[1, 0].scatter(x, y)*

*axs[1, 1].hist(data, bins=6)*

*plt.show()*

Guardar gráficos

*plt.plot(x, y)*

*plt.savefig('plot.png')  # Guarda el gráfico como un archivo PNG*


## Pandas
Proporciona estructuras de datos y herramientas de análisis para trabajar con datos tabulares. Usa principalmente las estructuras 'Series' y 'DataFrame'

*pip install pandas*

*import pandas as pd*

Series

*s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])*

*print(s)*

output:

*a    1*

*b    2*

*c    3*

*d    4*

*e    5*

*dtype: int64*

DataFrame

*data = {*

*.    'Nombre': ['Ana', 'Luis', 'Carlos'],*

*.    'Edad': [23, 34, 45],*

*.    'Ciudad': ['México', 'Monterrey', 'Guadalajara']*

*}*

*df = pd.DataFrame(data)*

*print(df)*

output:

*.   Nombre  Edad      Ciudad*

*0     Ana    23      México*

*1    Luis    34    Monterrey*

*2  Carlos    45  Guadalajara*

Operaciones básicas:

Leer datos

*df = pd.read_csv('archivo.csv')*

Escribir datos

*df.to_csv('archivo.csv', index=False)*

Seleccionar una o más columnas

*df[['Nombre', 'Edad']]*

Seleccionar una fila por índice (si df tiene index)

*df.loc[b]  # Selecciona la fila nombre b*

Seleccionar una fila por posición:

*df.iloc[0]  # Selecciona la fila posición 0*

Filtrado de datos

*df[df['Edad'] > 30]  # Filtra las filas donde la edad es mayor de 30*

Obtener estadísticas descriptivas

*df.describe()*

Agregar y eliminar columnas

*df['NuevaColumna'] = [1, 2, 3]  # Agregar una nueva columna*

*df = df.drop('NuevaColumna', axis=1)  # Eliminar una columna*

Agrupación y agregación

*df_grouped = df.groupby('Ciudad').mean()  # Agrupar por 'Ciudad' y calcular la media*

Obtener datos

*df_sorted = df.sort_values(by='Edad', ascending=False)  # Ordenar por la columna 'Edad' en orden descendente*


## TensorFlow
Para el desarrollo y la ejecución de modelos de aprendizaje automático (machine learning) y redes neuronales profundas (deep learning)

*pip install tensorflow*

*import tensorflow as tf*

Tensor. Arreglo multidimensional que representa datos

*# Crear un tensor constante*

*tensor = tf.constant([[1, 2], [3, 4]])*

*print(tensor)*

output:

*<tf.Tensor: shape=(2, 2), dtype=int32, numpy=*

*array([[1, 2],*
*.       [3, 4]], dtype=int32)>*

Modelo secuencial. Ideal para redes neuronales con una única secuencia de capas

*from tensorflow.keras.models import Sequential*

*from tensorflow.keras.layers import Dense*

*model = Sequential([*

*.    Dense(10, activation='relu', input_shape=(8,)),*

*.    Dense(1, activation='sigmoid')*

*])*

Modelo funcional. Más flexible, permite crear modelos más complejos y redes neuronales con múltiples entradas y salidas

*from tensorflow.keras.layers import Input, Dense*

*from tensorflow.keras.models import Model*

*inputs = Input(shape=(8,))*

*x = Dense(10, activation='relu')(inputs)*

*outputs = Dense(1, activation='sigmoid')(x)*

*model = Model(inputs=inputs, outputs=outputs)*

Compilación del modelo. Define el optimizador, la función de pérdida y las métricas para el entrenamiento

*model.compile(optimizer='adam',*

*.              loss='binary_crossentropy',*

*.              metrics=['accuracy'])*

Entrenamiento del modelo. Ajusta el modelo a los datos de entrenamiento

*model.fit(x_train, y_train, epochs=10, batch_size=32)*

Evaluación. Evalúa el modelo con datos de prueba para obtener métricas de rendimiento

*loss, accuracy = model.evaluate(x_test, y_test)*

*print(f'Loss: {loss}, Accuracy: {accuracy}')*

Predicción. Utiliza el modelo para hacer predicciones sobre nuevos datos

*predictions = model.predict(x_new)*

**Tutoriales en su sitio web**


## PyTorch
Para el desarrollo y la ejecución de modelos de aprendizaje automático (machine learning) y redes neuronales profundas (deep learning)

*pip install torch*

*import torch*

Tensor. Arreglo multidimensional que representa datos

Creación de tensores
*tensor = torch.tensor([[1, 2], [3, 4]])*

*print(tensor)*

output:

*tensor([[1, 2],*

*.        [3, 4]])*

Autograd. Sistema de diferenciación automática esencial para entrenar redes neuronales

Gradientes

*x = torch.tensor(2.0, requires_grad=True)*

*y = x**2*

*y.backward()*

*print(x.grad)  # Imprime el gradiente de y respecto a x*

output:

*tensor(4.)*

**torch.nn** para construir modelos de redes neuronales

*import torch.nn as nn*

*class SimpleNN(nn.Module):*

*.    def __init__(self):*

*.        super(SimpleNN, self).__init__()*

*.        self.fc1 = nn.Linear(784, 128)*

*.        self.fc2 = nn.Linear(128, 64)*

*.        self.fc3 = nn.Linear(64, 10)*



*.    def forward(self, x):*

*.        x = torch.relu(self.fc1(x))*

*.        x = torch.relu(self.fc2(x))*

*.        x = torch.softmax(self.fc3(x), dim=1)*

*.        return x*

*model = SimpleNN()*

Definición del optimizador y la función de pérdida

*import torch.optim as optim*

*criterion = nn.CrossEntropyLoss()*

*optimizer = optim.Adam(model.parameters(), lr=0.001)*

Bucle de entrenamiento

*for epoch in range(5):*

*.    optimizer.zero_grad()   # Limpiar los gradientes*

*.    output = model(x_train) # Pasar los datos a través del modelo*

*.    loss = criterion(output, y_train) # Calcular la pérdida*

*.    loss.backward() # Hacer retropropagación*

*.    optimizer.step() # Actualizar los pesos*

*.    print(f'Epoch {epoch+1}, Loss: {loss.item()}')*

Evaluación

*model.eval()  # Establecer el modo de evaluación*

*with torch.no_grad():  # Desactivar autograd para la evaluación*

*.    output = model(x_test)*

*.    predicted = torch.argmax(output, dim=1)*

*.    accuracy = (predicted == y_test).float().mean()*

*.    print(f'Accuracy: {accuracy.item()}')*

Predicción

*model.eval()*

*with torch.no_grad():*

*.    prediction = model(x_new)*

*.    print(prediction)*

**Tutoriales en su sitio web**
