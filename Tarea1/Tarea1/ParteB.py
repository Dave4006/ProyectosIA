# %% [markdown]
# # Importar Modulos

# %%
import pandas as pd
import numpy as np
import random
import plotly.express as px
import tensorflow as tf

from sklearn.utils import shuffle
import tensorflow_addons as tfa
from IPython.display import display

# %% [markdown]
# # Importación de Datos

# %%
#el conj de datos va de 0 a 2623 o sea 2624 datos
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.3f}".format

# Recompila la información de un archivo CSV y lo guarda en un arreglo
data_copy = np.loadtxt("lp5.csv", delimiter=",", dtype=str)

#  Contador de filas en el CSV
cont = 16
labels = []
numbers = []
# Arreglo de ceros con las dimensiones de los datos
features= np.zeros((164,15,6))
test_features= np.zeros((164,15,6))

# Ciclo que recorre todos los datos del CSV y guarda los títulos de los datos cuando el contador es 16,
# guarda los números en un arreglo aparte y reestablece el contador a su valor original cuando es igual a 0.
for i in range(len(data_copy)):
    if(cont == 16):
        labels.append(data_copy[i])
    if(cont<16):
        numbers.append(data_copy[i])
    cont -= 1
    if(cont == 0):
        cont = 16
        
test_numbers = np.array(numbers).astype(float)

# Añade el +- 5% de ruido a los datos.
test_numbers = test_numbers + test_numbers*random.uniform(-0.05, 0.05)

# Ciclos anidados que recorren todas las dimensiones del arreglo de ceros,
# agrupa los datos numéricos en conjuntos con forma 15x6 
cont = 0
for i in range(164):
    for j in range(15):
        for z in range(6):
            features[i][j][z] = numbers[cont][z]
            test_features[i][j][z]= test_numbers[cont][z]
        cont += 1
        
# Convierte los datos obtenidos del primer ciclo en un dataframe
labels_df = pd.DataFrame(labels, columns=['labels','1','2','3','4','5'])

display(labels_df)
display(features)
display(test_features)

# %% [markdown]
# # Manejo de Datos y Normalización

# %%
# Diccionario que codifica los datos categóricos a un conjunto de números idóneo para la capa de salida.
classes_dict = {'normal':'1 0 0 0 0', #1
        'collision_in_tool':'0 1 0 0 0', #2
        'collision_in_part':'0 0 1 0 0', #3
        'bottom_collision':'0 0 0 1 0', #4
        'bottom_obstruction':'0 0 0 0 1'} #5

# Se reemplazan los datos por lo del diccionario
labels_df = labels_df.replace({'labels':classes_dict})
# Se separan los datos de manera que se aisle un caractér por columna.
labels_df[['1', '2', '3', '4', '5']] = labels_df['labels'].str.split(' ', 4, expand= True)
# Se eliminan las columnas, solo dejando las columnas llamadas "1", "2", "3", "4", "5"
labels_df = labels_df.loc[:,['1', '2', '3', '4', '5']]
# Se convierten los valores de un tipo String a uno Entero para poder ser utilizados por la red neuronal.
labels_df[['1', '2', '3', '4', '5']]=labels_df[['1', '2', '3', '4', '5']].astype(str).astype(int)

'''/* 
Function: norm

Normaliza los datos a través de una función min-max entre 0 y 1 para ser alimentados a la red.

Parameters:

    x - Datos a normalizar.
    
Returns:

    Los datos alimentados ya normalizados.
*/'''

def norm(x):
    x_min = x.min()
    x_max = x.max()
    range = x_max - x_min  #min max entre 0 y 1
    return((x-x_min)/(range))

# Normaliza las características.
train_features = norm(features)
test_features = norm(test_features)
# Split para entrenamiento y validacion, con 20% para testeo y se aleatorizan.
train_labels = labels_df
test_labels = labels_df

train_features, train_labels = shuffle(train_features, train_labels, random_state=0)
test_features, test_labels = shuffle(test_features, test_labels, random_state= 0)

# %% [markdown]
# # Creación del Modelo

# %%
'''/* 
Function: my_model

Genera el modelo categórico convolucional y lo compila. El modelo consiste de lo siguiente:
    - Una capa de entradas de forma (15,6,1) que es igual a las dimensiones de la matriz de datos.
    
    - Una capa de convolución de dos dimensiones que genera 8 filtros, tiene un kernel 3x3, 
        función de activación ReLu y un padding para que mantenga las dimensiones de la entrada.
    - Una capa de convolución de dos dimensiones que genera 8 filtros, tiene un kernel 3x3, 
        función de activación ReLu y un padding para que mantenga las dimensiones de la entrada.
    - Una capa de Max Pooling con un pool de 2x2, un salto de 2 casillas y sin padding.
    - Una capa de Dropout razón con una razón de 0.25

    - Una capa de convolución de dos dimensiones que genera 16 filtros, tiene un kernel 3x3, 
        función de activación ReLu y un padding para que mantenga las dimensiones de la entrada.
    - Una capa de convolución de dos dimensiones que genera 16 filtros, tiene un kernel 3x3, 
        función de activación ReLu y un padding para que mantenga las dimensiones de la entrada.
    - Una capa de Max Pooling con un pool de 2x2, un salto de 2 casillas y sin padding.
    - Una capa de Dropout razón con una razón de 0.457.
    
    - Una capa de flatten para pasar de varias dimensiones a una.
    - Una capa oculta con 24 neuronas y con una función de activación ReLu.
    - Una capa de salida con 5 neuronas. Una por categoría.
    
    - Utiliza ADAM como optimizador.
    - La función de pérdida es la Entropía Cruzada Categórica.
    - La métrica de evaluación es la exactitud.
    
Parameters:

    my_learning_rate - Razón de aprendizaje.
    
Returns:

    El modelo neuronal ya creado.
*/'''

def my_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape = (15,6,1), padding='same'), # cant de filtros, dimensiones del kernel, kernel entre más pequeño mejor y se prefiere un número impar
        tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2), strides= 2, padding= 'valid'), # dimensiones del pooling, stride
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2), strides= 2, padding='valid'), # dimensiones del pooling, stride
        tf.keras.layers.Dropout(0.457),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    return model

# %% [markdown]
# # Entrenamiento del Modelo

# %%
'''/* 
Function: train_model

Alimenta los datos correspondientes a las características y etiquetas al modelo y conduce el proceso de entrenamiento y validación.
Además de esto, también le establece al modelo la cantidad de ciclos de entrenamiento y el tamaño del batch.
Por último crea un subconjunto de validación con un 25% de los datos de entrenamiento.
    
Parameters:

    model - El modelo creado anteriormente.
    features - Conjunto que contiene las características.
    labels - Conjunto que contiene las etiquetas.
    epochs - Cantidad de ciclos de entrenamiento
    batch_size - Tamaño del batch.
    
Returns:

    Los resultados del entrenamiento.
*/'''


def train_model(model, features, labels, epochs, batch_size):
    history = model.fit(
        x = features,
        y = labels,
        epochs= epochs,
        batch_size= batch_size,
        validation_split= 0.25
    )
    # Guarda los resultados obtenidos del proceso de entrenamiento y validación en un DataFrame.
    # Estos resultados son las pérdidas y el número del ciclo correspondiente.
    hist= pd.DataFrame(history.history)
    # Añade al DataFrame la información sobre los ciclos.
    hist['epoch'] = history.epoch
    
    return hist

# %% [markdown]
# # Visualizaciones

# %%
'''/* 
Function: loss_curve

Grafica las curvas de pérdida correspondientes al entrenamiento y la validación.
    
Parameters:

    history - Resultados provenientes del proceso de entrenamiento.
    
Returns:

    Las gráficas ya creadas.
*/'''

def loss_curves(history):
    hist = history
    # Cambia los títulos de cada columna que contiene los datos de exactitud por una versión más legible
    labels = {"loss":"Training Loss", "val_loss":"Validation Loss"}
    hist.rename(columns = labels, inplace = True)
    
    # Crea la figura, establece los títulos de eje y la paleta de colors
    fig = px.line(hist, x='epoch', y=['Training Loss', 'Validation Loss'],
                title='Gráficas de Pérdida de Entrenamiento y Evaluación',
                labels={"epoch": "Epoch", "value":"Binary Cross Entropy", "variable":"Curvas de Pérdida"},
                color_discrete_map={
                "Training Loss": "#46039f", "Validation Loss": "#fb9f3a"})
    # Actualiza el tema de la gráfica.
    fig.update_layout(template='plotly_white')
    fig.show()
    
'''/* 
Function: accuracy_curve

Grafica las curvas de exactitud correspondientes al entrenamiento y la validación.
    
Parameters:

    history - Resultados provenientes del proceso de entrenamiento.
    
Returns:

    Las gráficas ya creadas.
*/'''
    
def accuracy_curve(history):
    hist = history
    # Cambia los títulos de cada columna que contiene los datos de exactitud por una versión más legible
    labels = {"val_accuracy":"Exactitud de Validación", "accuracy":"Exactitud de Entrenamiento"}
    hist.rename(columns = labels, inplace = True)
    
    # Crea la figura, establece los títulos de eje y la paleta de colors
    fig = px.line(hist, x='epoch', y=['Exactitud de Entrenamiento', 'Exactitud de Validación'],
                title='Gráficas de Exactitud',
                labels={"epoch": "Epoch", "value":"Exactitud", "variable":"Curvas de Exactitud"},
                color_discrete_map={
                "Training Loss": "#46039f", "Validation Loss": "#fb9f3a"})
    # Actualiza el tema de la gráfica.
    fig.update_layout(template='plotly_white')
    fig.show()

# %% [markdown]
# # Se corren las funciones

# %%
# Hiperparámetros
learning_rate = 0.001
epochs = 350
batch_size = 5
# Llama a la función para crear el modelo y lo guarda.
model = my_model(learning_rate)
# Invoca a la función de entrenamiento y guarda los resultados.
history= train_model(model, train_features, train_labels, epochs, batch_size)
# Llama a la función de las gráficas.
loss_curves(history)
accuracy_curve(history)


# %% [markdown]
# # Predicciones

# %%
# Hace predicciones usando el conjunto de datos de prueba.
pd.options.display.float_format = "{:.0f}".format
predictions = model.predict(test_features)

# For que itera por filas las predicciones, encuentra el valor máximo y lo cambia por 1. Si este es diferente que 1.
for i in range(len(predictions)):
    if(predictions[i].max() != 1):
        row, col = np.where(predictions == predictions[i].max())
        predictions[i][col[0]] = 1
#Cambia todos los números que no son 1 por 0.
predictions[predictions!=1] = 0 
    
# Convierte el arreglo de predicciones en un DataFrame
predictions_df = pd.DataFrame(predictions, columns=['normal', 'collision_in_tool', 'collision_in_part', 'bottom_collision', 'bottom_obstruction'])

# Crea las matrices de confusión para 5 clases.
CM = tfa.metrics.MultiLabelConfusionMatrix(num_classes=5)
# Se le pasan los datos 
CM.update_state(test_labels, predictions_df)
# Se guarda el resultado
result = CM.result().numpy()

# Grafica matriz de confusión para la clase normal.
normal = px.imshow(result[0], 
                labels = dict(x='normal (Predicted)', y='normal'),
                x = ['Negative', '-Positive'], y=['True', 'False'],text_auto=True, color_continuous_scale='greys')
normal.update_xaxes(side="top")
normal.show()

# Grafica matriz de confusión para la clase collision_in_tool.
collision_in_tool = px.imshow(result[1], 
                labels = dict(x='collision_in_tool (Predicted)', y='collision_in_tool'),
                x = ['Negative', '-Positive'], y=['True', 'False'], text_auto=True, color_continuous_scale='greys')
collision_in_tool.update_xaxes(side="top")
collision_in_tool.show()

# Grafica matriz de confusión para la clase collision_in_part.
collision_in_part = px.imshow(result[2], 
                labels = dict(x='collision_in_part (Predicted)', y='collision_in_part'),
                x = ['Negative', '-Positive'], y=['True', 'False'], text_auto=True, color_continuous_scale='greys')
collision_in_part.update_xaxes(side="top")
collision_in_part.show()

# Grafica matriz de confusión para la clase bottom_collision.
bottom_collision = px.imshow(result[3], 
                labels = dict(x='bottom_collision (Predicted)', y='bottom_collision'),
                x = ['Negative', '-Positive'], y=['True', 'False'], text_auto=True, color_continuous_scale='greys')
bottom_collision.update_xaxes(side="top")
bottom_collision.show()

# Grafica matriz de confusión para la clase bottom_obstruction.
bottom_obstruction = px.imshow(result[4], 
                labels = dict(x='bottom_obstruction (Predicted)', y='bottom_obstruction'),
                x = ['Negative', '-Positive'], y=['True', 'False'], text_auto=True, color_continuous_scale='greys')
bottom_obstruction.update_xaxes(side="top")
bottom_obstruction.show()

model.save('ModeloConvolucion')


