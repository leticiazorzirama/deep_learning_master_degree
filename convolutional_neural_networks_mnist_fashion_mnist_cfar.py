# Importações

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
print('Python version:', sys.version.split(' ')[0])

import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import keras
from keras import layers
from keras import ops
from keras import datasets

# Utilidades

def plots1(history):
  plt.figure(figsize=(14,4))
  plt.subplot(1,2,1)
  plt.plot(history.history['loss'], '.-', label='Train loss')
  if 'val_loss' in history.history.keys():
    plt.plot(history.history['val_loss'], '.-', label='Val loss')
  plt.xlabel('Epochs')
  plt.legend()
  plt.grid()
  plt.subplot(1,2,2)
  plt.plot(history.history['accuracy'], '.-', label='Train accuracy')
  plt.xlabel('Epochs')
  if 'val_accuracy' in history.history.keys():
    plt.plot(history.history['val_accuracy'], '.-', label='Val accuracy')
  plt.legend()
  plt.grid()

def plots2(history, model=model, training=training, batch_size=batch_size, epochs=epochs):
    """
    Plots training and validation loss/accuracy curves with dynamic title and hyperparameters.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Main Title & Subtitle Setup ---
    # Main title using model name
    fig.suptitle(f"Model {model} | Training {training}", fontsize=16, fontweight='bold', y=1.03)
    
    # Subtitle using batch size and epochs
    plt.figtext(
        0.5, 0.96, 
        f"Batch Size = {batch_size} | Epochs = {epochs}", 
        ha="center", fontsize=11, color="dimgray", style="italic"
    )

    # --- Loss Plot ---
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', linewidth=2)
    ax1.set_title('Loss Curve', fontsize=12, pad=10)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Accuracy Plot ---
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

    ax2.plot(history.history[acc_key], label='Train Accuracy', linewidth=2)
    if val_acc_key in history.history:
        ax2.plot(history.history[val_acc_key], label='Validation Accuracy', linestyle='--', linewidth=2)
    ax2.set_title('Accuracy Curve', fontsize=12, pad=10)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

# ===============
#  MNIST dataset
# ===============
# 1. Treinando a partir do zero
# Assim como no exercício anterior, carregue o conjunto MNIST e separe as últimas 5000 imagens como conjunto de validação. 
# No entanto, desta vez não realize qualquer pré-processamento nas imagens (como escalonamento); 
# isto será feito internamente no modelo depois.

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

# Atributos do dataset
print(f"Train images Min: {x_train.min()}")
print(f"Train images Max: {x_train.max()}")

# Quantidade de classes e contagem de cada
classes, counts = np.unique(y_train, return_counts=True)

for cls, count in zip(classes, counts):
    print(f"Class {cls}: {count} samples")

# Separar um conjunto de 5000 amostras para validação
x_val = x_train[-5000:] # 5000 amostras finais
y_val = y_train[-5000:] 
x_train = x_train[:-5000] # conjunto menos as 5000 amostras finais
y_train = y_train[:-5000]
x_train.dtype  
input_shape = x_train.shape[-2:]

# Redimensionar as dimensões de x externamente
# x_train = np.expand_dims(x_train, -1)
# x_val = np.expand_dims(x_val, -1)
# x_test = np.expand_dims(x_test, -1)
# print(f"x_train shape: {x_train.shape}")
# print(f"{x_train.shape[0]} train samples of shape {x_train.shape[-3:]}")
# print(f"{x_val.shape[0]} val samples of shape {x_val.shape[-3:]}")
# print(f"{x_test.shape[0]} test samples of shape {x_test.shape[-3:]}")

# Converter vetores y para matrizes binárias
y_train = keras.utils.to_categorical(y_train, len(classes))
y_test = keras.utils.to_categorical(y_test, len(classes))
y_train.shape

# 2. Usando o Keras, construa uma rede neural com pelo menos uma camada convolucional (tf.keras.layers.Conv2D) e 
# confirme que não há nenhum erro de definição. Organize seu código em uma função de criação do modelo, conforme o 
# exemplo abaixo (dê o nome que preferir). Utilize camadas de escalonamento e reshape conforme necessário.

# Dicas
# Funções úteis: tf.keras.layers.Rescaling, tf.keras.layers.Reshape

# Camadas convolucionais 2D exigem que a entrada seja um tensor 3D, sendo o último eixo correspondente ao número de 
# canais (no caso, apenas 1, para uma imagem em tons de cinza).

# ==========
#  MODELO 1
# ==========
model = "1"

def make_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            layers.Reshape((28, 28, -1)), # Redimensionar as dimensões de x internamente
            layers.Rescaling(scale=1./255), # Rescalonar para [0, 1]
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(len(classes), activation="softmax"),
        ]
)
    return model

model1 = make_model()
model1.summary()

# 3. Desenvolva (i.e., aprimore a arquitetura) e treine sua rede (a partir do zero), tentando conseguir uma acurácia de validação de pelo menos 99.2%.
# (Lembre que usando apenas camadas densas é difícil conseguir uma acurácia muito superior a 98%.) 
# Em seguida, calcule a acurácia no conjunto de teste.

# Treinamento 1
training = 1
batch_size = 128
epochs = 15 

model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Resultados
history = model1.history
score = model1.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# 98.90% para acurácia de validação
# 99.12% para a acurácia de teste

# Treinamento 2
training = 2
batch_size = 128
epochs = 30 # Épocas aumentadas de 15 para 30 para aumentar a acurácia de validação

model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Resultados
history = model1.history
score = model1.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# 99.45% para acurácia de validação
# 99.23% para a acurácia de teste

# 4. (OPCIONAL) Por que o uso de Dropout faz com que o desempenho de treinamento comece bastante inferior ao de validação?
# Porque o dropout é ativado apenas no treinamento e desativado na validação.

# Dicas
# Parta da arquitetura deste tutorial (com os devidos ajustes feitos no item anterior) e adicione uma camada densa com um número suficiente de unidades. Lembre-se de (ao contrário do tutorial) trazer para dentro do modelo qualquer pré-processamento necessário.
# Visualize os gráficos do treinamento usando a função plots fornecida (ou a ferramenta TensorBoard).
# Ao usar camadas convolucionais com GPU, a execução paralelizada torna impossível garantir a reproducibilidade, portanto, não perca tempo com isso.

# ==========
#  MODELO 2
# ==========
model = "2"

def make_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            layers.Reshape((28, 28, -1)), # Redimensionar as dimensões de x internamente
            layers.Rescaling(scale=1./255), # Rescalonar para [0, 1]
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"), # Camada adicionada 
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(len(classes), activation="softmax"),
        ]
)
    return model

model2 = make_model()
model2.summary()

# Treinamento 1
training = 1
batch_size = 128
epochs = 30

model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Resultados
history = model2.history
score = model2.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# 99.33% para acurácia de validação
# 99.04% para a acurácia de teste

# Treinamento 2
training = 2
batch_size = 128
epochs = 45

model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Resultados
history = model2.history
score = model2.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") # 98.9%

# 99.69% para acurácia de validação
# 98.99% para a acurácia de teste

# ======================
#  FASHION-MNIST dataset
# ======================

# 5. (OPCIONAL) Repita para o conjunto Fashion-MNIST, o qual também está disponível no Keras. 
# Nesse caso, é suficiente aproveitar a mesma arquitetura do modelo e apenas (se necessário) alterar a taxa de aprendizado e o número de épocas. 
# Sem muito esforço é possível conseguir uma acurácia de validação de 92% (em comparação com 87% para uma rede densa). 
# Se desejar, visualize algumas imagens do conjunto de treinamento e algumas predições erradas no conjunto de teste.

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

# Converter vetores y para matrizes binárias
y_train = keras.utils.to_categorical(y_train, len(classes))
y_test = keras.utils.to_categorical(y_test, len(classes))
y_train.shape

# Atributos do dataset
print(f"Train images Min: {x_train.min()}")
print(f"Train images Max: {x_train.max()}")

# Quantidade de classes e contagem de cada
classes, counts = np.unique(y_train, return_counts=True)

for cls, count in zip(classes, counts):
    print(f"Class {cls}: {count} samples")

# Visualização do dataset
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])

plt.tight_layout()
plt.show()

# =================================
#  MODELO 3 - idêntico ao modelo 2
# =================================
model = "3"

def make_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            layers.Reshape((28, 28, -1)), # Redimensionar as dimensões de x internamente
            layers.Rescaling(scale=1./255), # Rescalonar para [0, 1]
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"), 
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(len(classes), activation="softmax"),
        ]
)
    return model

model3 = make_model()
model3.summary()

# Treinamento 1
training = 1
batch_size = 128
epochs = 45

model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model3.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Resultados
history = model3.history
score = model3.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# 90.09% para acurácia de validação
# 89.58% para a acurácia de teste

# TO DO 
# Plots for all