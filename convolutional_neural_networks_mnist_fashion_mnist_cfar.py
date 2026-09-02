# TO DOs
# Transferir para jupyter notebook
# GPU
# Plotagens com métricas de treino e validação 

# Importações

import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

print('Python version:', sys.version.split(' ')[0])

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import data as tf_data

print('TensorFlow version:', tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import keras
from keras import callbacks, layers
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input

# Funções de plotagens

def plots1(history):
  plt.figure(figsize=(14,4))
  plt.subplot(1,2,1)
  plt.plot(history.history['loss'], '.-', label='Train loss')
  if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], '.-', label='Val loss')
  plt.xlabel('Epochs')
  plt.legend()
  plt.grid()
  plt.subplot(1,2,2)
  plt.plot(history.history['accuracy'], '.-', label='Train accuracy')
  plt.xlabel('Epochs')
  if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], '.-', label='Val accuracy')
  plt.legend()
  plt.grid()

def plots2(history, model, training, batch_size, epochs):
    """
    Plots training and validation loss/accuracy curves with dynamic title and hyperparameters.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Main Title & Subtitle Setup ---
    # Main title using model version
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

# Carregar dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

# Atributos do dataset
print(f"Train images Min: {x_train.min()}")
print(f"Train images Max: {x_train.max()}")

# Salvar quantidade de classes e contagem de instâncias por classe
classes, counts = np.unique(y_train, return_counts=True)

for cls, count in zip(classes, counts):
    print(f"Class {cls}: {count} samples")

# Separar um conjunto de 5000 amostras para validação
x_val = x_train[-5000:] # 5000 amostras finais
y_val = y_train[-5000:] 
x_train = x_train[:-5000] # conjunto menos as 5000 amostras finais
y_train = y_train[:-5000]
print(x_train.dtype)
input_shape = x_train.shape[-2:]

# Redimensionar as dimensões de x externamente (só para salvar aqui, porque está sendo feito internamente nos modelos)
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
print(y_train.shape)

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
            layers.Reshape((28, 28, -1)), # Redimensionar subconjuntos de x
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

# 3. Desenvolva (i.e., aprimore a arquitetura) e treine sua rede (a partir do zero), tentando conseguir uma acurácia de validação de pelo menos 99.2%.
# (Lembre que usando apenas camadas densas é difícil conseguir uma acurácia muito superior a 98%.) 
# Em seguida, calcule a acurácia no conjunto de teste.

# TREINAMENTO 1
training = 1

# Construir a arquitetura
model1 = make_model()
model1.summary()

# Hiperparâmetros
batch_size = 128
epochs = 15 
model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinamento
model1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Resultados
history = model1.history
score = model1.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# Visualização dos resultados
mod11_plots1 = plots1(history)
mod11_plots2 = plots2(history, model, training, batch_size, epochs)

# TREINAMENTO 2
training = 2

# Construir a arquitetura
model1 = make_model()
model1.summary()

# Hiperparâmetros
batch_size = 128
epochs = 30 # Épocas aumentadas de 15 para 30 para aumentar a acurácia de validação e tentar chegar nos 99.2%
model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinamento
model1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Resultados
history = model1.history
score = model1.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# Visualização dos resultados
mod12_plots1 = plots1(history)
mod12_plots2 = plots2(history, model, training, batch_size, epochs)

# 4. (OPCIONAL) Por que o uso de Dropout faz com que o desempenho de treinamento comece bastante inferior ao de validação?
# RESPOSTA: Porque o dropout é ativado apenas no treinamento e desativado na validação.

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
            layers.Reshape((28, 28, -1)), # Redimensionar subconjuntos de x
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

# TREINAMENTO 1
training = 1

# Construir a arquitetura
model2 = make_model()
model2.summary()

# Hiperparâmetros
batch_size = 128
epochs = 30
model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinamento
model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Resultados
history = model2.history
score = model2.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# Visualização dos resultados
mod21_plots1 = plots1(history)
mod21_plots2 = plots2(history, model, training, batch_size, epochs)

# TREINAMENTO 2
training = 2

# Construir a arquitetura
model2 = make_model()
model2.summary()

# Hiperparâmetros
batch_size = 128
epochs = 45
model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinamento
model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Resultados
history = model2.history
score = model2.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# Visualização dos resultados
mod22_plots1 = plots1(history)
mod22_plots2 = plots2(history, model, training, batch_size, epochs)

# ======================
#  FASHION-MNIST dataset
# ======================

# 5. (OPCIONAL) Repita para o conjunto Fashion-MNIST, o qual também está disponível no Keras. 
# Nesse caso, é suficiente aproveitar a mesma arquitetura do modelo e apenas (se necessário) alterar a taxa de aprendizado e o número de épocas. 
# Sem muito esforço é possível conseguir uma acurácia de validação de 92% (em comparação com 87% para uma rede densa). 
# Se desejar, visualize algumas imagens do conjunto de treinamento e algumas predições erradas no conjunto de teste.

# Carregar dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

# Atributos do dataset
print(f"Train images Min: {x_train.min()}")
print(f"Train images Max: {x_train.max()}")

# Salvar quantidade de classes e contagem de instâncias por classe
classes, counts = np.unique(y_train, return_counts=True)

for cls, count in zip(classes, counts):
    print(f"Class {cls}: {count} samples")

# Converter vetores y para matrizes binárias
y_train = keras.utils.to_categorical(y_train, len(classes))
y_test = keras.utils.to_categorical(y_test, len(classes))
print(y_train.shape)

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

# TREINAMENTO 1
training = 1

# Construir a arquitetura
model3 = make_model()
model3.summary()

# Hiperparâmetros
batch_size = 128
epochs = 45
model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinamento
model3.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Resultados
history = model3.history
score = model3.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# Visualização dos resultados
mod31_plots1 = plots1(history)
mod31_plots2 = plots2(history, model, training, batch_size, epochs)

# TREINAMENTO 2
training = 2

# Construir a arquitura
model3 = make_model()
model3.summary()

# Hiperparâmetros
batch_size = 128
epochs = 90
model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinamento
model3.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Resultados
history = model3.history
score = model3.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# Visualização dos resultados
mod32_plots1 = plots1(history)
mod32_plots2 = plots2(history, model, training, batch_size, epochs)

# Sinal de sobreajuste a partir da época 40

# ==================
#  CIFAR-10 dataset
# ==================

# Carregar dataset
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
print('x_train.dtype:', x_train.dtype)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Atributos do dataset
print(f"Train images Min: {x_train.min()}")
print(f"Train images Max: {x_train.max()}")

# Salvar quantidade de classes e contagem de instâncias por classe
classes, counts = np.unique(y_train, return_counts=True)

for cls, count in zip(classes, counts):
    print(f"Class {cls}: {count} samples")

# Redimensionr y para um tensor 1D com valores em [0, 1, ..., n_classes-1]
# Par que se possa usar a perda sparse_categorical_crossentropy
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# Subconjunto de validação
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=5000, shuffle=False)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)

# Visualização do dataset
plt.figure(figsize=(12,6))
for i in range(5):
  for c in range(10):
    plt.subplot(5, 10, 10*i+c+1)
    img = x_train[y_train == c][i]
    plt.imshow(img)
    if i == 0:
      plt.title(f'y = {c}')
    plt.axis('off')    

# 6. Inicialmente, apenas converta a mesma arquitetura utilizada no MNIST para o formato das imagens do CIFAR-10 e treine o modelo. 
# Note que agora não é mais necessário usar uma camada Reshape. Certifique-se de escolher um batch size e taxa de aprendizado apropriadas. 
# Observe que é difícil obter uma acurácia de validação superior a 73%.

# =================================
#  MODELO 4 - idêntico ao modelo 3
# =================================
model = "4"

def make_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
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

# TREINAMENTO 1
training = 1

# Construir a arquitetura
model4 = make_model()
model4.summary()

# Hiperparâmetros
batch_size = 128
epochs = 45
model4.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Treinamento
model4.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# Resultados
history = model4.history
score = model4.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# Visualização dos resultados
mod41_plots1 = plots1(history)
mod41_plots2 = plots2(history, model, training, batch_size, epochs)

# 7. Por que você acha que isso acontece? Explique.
# Porque o CIFAR-10 é um dataset mais complexo que o MNIST e o Fashion-MNIST em 
# termos de dimensões das imagens (32 x 32 x 3) e a arquitetura já se torna simples para
# ser utilizada no CIFAR-10. 
# As imagens do CIFAR-10, inclusive, além de coloridas, são mais heterogêneas, apresentando cenários mais
# próximos do mundo real. Soma-se ainda o fato das imagens serem de baixa resolução. 
# As imagens dos datasets MNIST e Fashion-MNIST, diferentemente, além de monocromáticas, 
# são mais homogêneas e abstratas.

# Perturbação dos dados
data_augmentation = keras.Sequential(
    [
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(mode='horizontal'),
    ],
    name='data_augmentation',
)

# Visualização da perturbação dos dados
plt.figure(figsize=(10, 6))
i = 0
for j in range(15):
  img = data_augmentation(x_train[[i]])[0].numpy()
  plt.subplot(3, 5, j+1)
  plt.imshow(img.astype('uint8'))
  plt.axis('off')

# ==========
#  MODELO 5 
# ==========
model = 5

# Modelo baseline
# Princípios gerais dos modelos VGG
# Blocos: filtros 3 x 3 e max pooling 2 x 2
# Blocos empilhados com número crescente de filtros
# Padding para garantir que dimensões compatíveis dos features maps e inputs
def make_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Rescaling(scale=1./255), # Rescalonar para [0, 1],
            data_augmentation,
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform', padding='same'),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform', padding='same'),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform', padding='same'),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform', padding='same'),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer='he_uniform', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
            layers.Dense(len(classes), activation="softmax"),
            ]
)
    return model
            
# TREINAMENTO 1
training = 1

# Construir a arquitetura
model5 = make_model()
model5.summary()

# Hiperparâmetros
batch_size = 128
epochs = 45
opt = keras.optimizers.Adam(learning_rate=0.01)
model5.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Treinamento
model5.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Resultados
history = model5.history
score = model5.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# Visualização dos resultados
mod51_plots1 = plots1(history)
mod51_plots2 = plots2(history, model, training, batch_size, epochs)

# TREINAMENTO 2
training = 2

# Construir a arquitetura
model5 = make_model()
model5.summary()

# Hiperparâmetros
batch_size = 128
epochs = 90 

opt = keras.optimizers.Adam(learning_rate=0.01)
model5.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Adicionar callback de decaimento
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="accuracy",
    factor=0.1,
    patience=10,
    min_lr=0.0,
)

# Treinamento
model5.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr])

# Resultados
history = model5.history
score = model5.evaluate(x_test, y_test, verbose=0)
print(f"Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# Visualização dos resultados
mod52_plots1 = plots1(history)
mod52_plots2 = plots2(history, model, training, batch_size, epochs)

# =================================
#  TRANSFER LEARNING e FINE-TUNING 
# =================================

# ======================
#  CATS VS. DOG dataset
# ======================

# Carregar dataset
train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # Reserve 10% for validation and 10% for test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Include labels
)

print(f"Number of training samples: {train_ds.cardinality()}")
print(f"Number of validation samples: {validation_ds.cardinality()}")
print(f"Number of test samples: {test_ds.cardinality()}")

# Dimensões de uma imagem
for i, (image, label) in enumerate(train_ds.take(1)):
    print(image[0].shape)

# Visualizar amostras do dataset
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))
    plt.axis("off")

# Redimensionar os dados
resize_fn = keras.layers.Resizing(150, 150)

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))

# Atributos do dataset - dimensões de uma amostra após redimensionar
for i, (image, label) in enumerate(train_ds.take(1)):
    print(image[0].shape)

# Perturbação dos dados
augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

# Batch e prefetching ?
batch_size = 64

train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()

# Visualizar amostra do dataset de treino após perturbação
for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(np.expand_dims(first_image, 0))
        plt.imshow(np.array(augmented_image[0]).astype("int32"))
        plt.title(int(labels[0]))
        plt.axis("off")

# ==========
#  MODELO 6
# ==========
model = 6

# Treinamento 1
training = 1

# Carregar modelo Xception pré-treinado
base_model = keras.applications.Xception(
    weights="imagenet",  # Carregar pesos do modelo treinado com o ImageNet
    input_shape=(150, 150, 3),
    include_top=False, # Não incluir o classificador no topo da arquitetura
) 

# Congelar
base_model.trainable = False

# Criar um novo modelo no topo da arquitetura
inputs = keras.Input(shape=(150, 150, 3))

# Xception requer que o input seja de [-1., +1] e offset de 1
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)

# O modelo base contém camadas de batch normalization que devem ser mantidas 
# no modo de inferência quando a arquitetura for descongelada para o fine-tuning
# Verificar se o modelo está rodando no modo de inferência
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  
outputs = keras.layers.Dense(1)(x)
model6 = keras.Model(inputs, outputs)

# Visualizar a arquitetura com o modelo base congelado (Trainable = N)
model6.summary(show_trainable=True)

# Hiperparâmetros
epochs = 2
model6.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# Treinar apenas o topo do modelo
model6.fit(train_ds, epochs=epochs, validation_data=validation_ds, shuffle=False)

# Resultados
history = model6.history
score = model6.evaluate(test_ds, verbose=0)
print(f"Transfer-learning \n Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Transfer-learning \n Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %") 

# FINE-TUNING
# Descongelar o modelo
# Irá continuar rodando no modo inferência

# Descongelar
base_model.trainable = True

# Visualizar a arquitetura com o modelo base treinável (Trainable = Y)
model6.summary(show_trainable=True)

# Hiperparâmetros
epochs = 1
model6.compile(
    optimizer=keras.optimizers.Adam(1e-5),  
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# Treinar apenas o topo do modelo
model6.fit(train_ds, epochs=epochs, validation_data=validation_ds, shuffle=False)

# Resultados
history = model6.history
score = model6.evaluate(test_ds, verbose=0)
print(f"Fine-tuning after Transfer-learning \n Model {model} | Training {training} - Test loss: {score[0]}")
print(f"Fine-tuning after Transfer-learning \n Model {model} | Training {training} - Test accuracy: {round(score[1]*100, 2)} %")

