import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
print('Python version:', sys.version.split(' ')[0])

import tensorflow as tf
print('TensorFlow version:', tf.__version__)

import keras
from keras import layers
from keras import ops

def plots(history):
  plt.figure(figsize=(14,4))
  plt.subplot(1,2,1)
  plt.plot(history.history['loss'], '.-', label='Train loss')
  if 'val_loss' in history.history.keys():
    plt.plot(history.history['val_loss'], '.-', label='Val loss')
  plt.xlabel('Epochs');
  plt.legend();
  plt.grid();
  plt.subplot(1,2,2)
  plt.plot(history.history['accuracy'], '.-', label='Train accuracy')
  plt.xlabel('Epochs');
  if 'val_accuracy' in history.history.keys():
    plt.plot(history.history['val_accuracy'], '.-', label='Val accuracy')
  plt.legend();
  plt.grid();

# 1. Treinando a partir do zero
# MNIST
# Assim como no exercício anterior, carregue o conjunto MNIST e separe as últimas 5000 imagens como conjunto de validação. 
# No entanto, desta vez não realize qualquer pré-processamento nas imagens (como escalonamento); 
# isto será feito internamente no modelo depois.

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

# Atributos do dataset
num_classes = 10
input_shape = (28, 28, 1)

# Separar um conjunto de 5000 amostras para validação
x_val = x_train[-5000:] # 5000 amostras finais
y_val = y_train[-5000:] 
x_train = x_train[:-5000] # conjunto menos as 5000 amostras finais
y_train = y_train[:-5000]
x_train.dtype  

# Garantir input shape
x_train = np.expand_dims(x_train, -1)
x_val = np.expand_dims(x_val, -1)
x_test = np.expand_dims(x_test, -1)
print(f"x_train shape: {x_train.shape}")
print(f"{x_train.shape[0]} train samples of shape {x_train.shape[-3:]}")
print(f"{x_val.shape[0]} val samples of shape {x_val.shape[-3:]}")
print(f"{x_test.shape[0]} test samples of shape {x_test.shape[-3:]}")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train.shape

# 2. Usando o Keras, construa uma rede neural com pelo menos uma camada convolucional (tf.keras.layers.Conv2D) e 
# confirme que não há nenhum erro de definição. Organize seu código em uma função de criação do modelo, conforme o 
# exemplo abaixo (dê o nome que preferir). Utilize camadas de escalonamento e reshape conforme necessário.

# Dicas
# Funções úteis: tf.keras.layers.Rescaling, tf.keras.layers.Reshape

# Camadas convolucionais 2D exigem que a entrada seja um tensor 3D, sendo o último eixo correspondente ao número de 
# canais (no caso, apenas 1, para uma imagem em tons de cinza).

def make_model():
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
)
    return model

model = make_model()
model.summary()

# 3. Desenvolva (i.e., aprimore a arquitetura) e treine sua rede (a partir do zero), tentando conseguir uma acurácia de validação de pelo menos 99.2%.
# (Lembre que usando apenas camadas densas é difícil conseguir uma acurácia muito superior a 98%.) Em seguida, calcule a acurácia no conjunto de teste.
batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {round(score[1]*100, 2)} %")

# (OPCIONAL) Por que o uso de Dropout faz com que o desempenho de treinamento comece bastante inferior ao de validação?
# Porque o dropout é ativado apenas no treinamento e desativado na validação.

# Dicas
# Parta da arquitetura deste tutorial (com os devidos ajustes feitos no item anterior) e adicione uma camada densa com um número suficiente de unidades. Lembre-se de (ao contrário do tutorial) trazer para dentro do modelo qualquer pré-processamento necessário.
# Visualize os gráficos do treinamento usando a função plots fornecida (ou a ferramenta TensorBoard).
# Ao usar camadas convolucionais com GPU, a execução paralelizada torna impossível garantir a reproducibilidade, portanto, não perca tempo com isso.


