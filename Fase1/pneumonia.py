import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# CONFIGURAÇÃO DE AMBIENTE E CAMINHOS

CAMINHO_BASE = r'tech_challenge_img\chest_xray'

train_dir = os.path.join(CAMINHO_BASE, 'train')
test_dir = os.path.join(CAMINHO_BASE, 'test')

if not os.path.exists(train_dir):
    print(f'Caminho de treino: {train_dir} não encontrado')
else:
    print(f"Diretório de treino encontrado: {train_dir}")

print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))


# 1. PRÉ PROCESSAMENTO
# Redes neurais não enxergam imagens, apenas matrizes e numeros
# Normalizei os numeros (Dividindo por 255) para facilitar o cálculo.
# Aplicando também Data Augmentation no treino. Para criar variações artificiais
# (zoom, giros de leve) para evitar Overfitting.

#Dimensões para todas as imagens padrão

IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32 #numero de imagens para processar por vez


print('PReparando geradores de Imagens')

#Configurações para dados de treino.
train_datagen = ImageDataGenerator(
    rescale=1./255,     #Normaliza pixel de 0-255 para 0-1
    rotation_range=20,  #Rotaciona levemente a imagem
    zoom_range=0.2,     #aplicando zoom aleatório
    shear_range=0.2,    #Deformação geometrica
    horizontal_flip=True,   #Espeçha horizontalmente (pulmão esquerdo virando direto)
    fill_mode='nearest'     #preenche pixels varios criados pela rotação
)


#Configuração para teste sem Augmentation
#A ideia é saber como o modelo se sai em imagens reais e originais


test_datagen = ImageDataGenerator(rescale=1./255)

print('Carregando imagens de treino...')
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',    #Binary, pois tenho apenas duas classes: Normal e Pneumonia
    color_mode='rgb'        #3 canais por compatibilidade padrão
)

print("Carregando imagens de teste..")
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 2. CONSTRUÇÃO DA ARQUITETURA CNN
#Usei uma arquitetura sequencial clássica para Visão Computacional
# - Conv2D: Filtros que deslizam sobre a imagen detectando padrões (bordas, manchas)
# - MaxPooling2D: Reduz o tamanho da imagem mantendo apenas a informação mais forte
# diminuição do custo computacional:
# - Flatten: Transforma a matriz 2D em um vetor longo para entrar na parte densa
# - Dropout: Desliga aleatóriamente neurônios durante o treino para forçar a rede a
#aprender caminhos alternativbos (reduz overfitting)


model = Sequential([
    #Primeira camada Convolucional
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    #Segunda camada (mais filtros para mais detalhes complexos)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    #Terceira Camada
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    #Camada de classificação
    Flatten(),
    Dense(128, activation='relu'),  #Camada densa com 128 neuronios
    Dropout(0.5),                   #Ignora 50% dos neuronios a cada passo do treino
    Dense(1, activation='sigmoid')  #Saída únida (1 a 1). <0.5 = Normal, >0.5 = Pneumonia
])

#Compilação:
# - Optimizer Adam: Mais versatil, ajustando a taxa de aprendizado sozinho
# - Loss Binary Crossentropy: A função de pedrã padrão para classificação binãria

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


print('\nResumo do modelo criado:')
model.summary()


# 3. TREINAMENTO

#Ajustando o peso da rede neural comparando predições com rotuloes reais
#epochs = olha o dataset inteiro 8 vezes
#step_per_epoch = Quantos lotes formam uma epoca - BATCH_SIZE (32)


EPOCHS = 8

print(f"\nIniciando treinamento por {EPOCHS} épocas...")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=test_generator.samples
)

# Salvando o modelo
model.save('modelo_pneumonia_cnn.keras')
print('\nModelo Salvo')

# 4. AVALIAÇÃO E VISUALIZAÇÃO DOS RESULTADOS
# Analisei graficamente se houve aprendizado constante e se não houve overfitting

#Extraindo dados do histórico

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))


#Acurácia
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia no Treino')
plt.plot(epochs_range, val_acc, label='Acurácia na Validação')
plt.legend(loc='lower right')
plt.title('Performance: Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia (0-1)')

#Graf. de fperda (menoir melhor)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Erro no Treino')
plt.plot(epochs_range, val_loss, label='Erro na Validação')
plt.legend(loc='upper right')
plt.title('Performance: Erro (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Valor do Erro')

plt.tight_layout()
plt.show()

#Avaliação numérica
print("\n--- Avaliação Final no Conjunto de Teste ---")
results = model.evaluate(test_generator)
print(f"Perda (Loss) Final: {results[0]:.4f}")
print(f"Acurácia (Accuracy) Final: {results[1]*100:.2f}%")

# CONCLUSÃO TÉCNICA
# usei no modelo redes neurais convolucionais para extrair padrões de opacidade pulmonar típicos da pneumonia.
# Se a acurácia de validação estiver acima de 80-85%, o modelo é considerado funcional para triagem preliminar.
# O uso de Dropout ajudou a manter a generalização, evitando que o modelo apenas memorizasse os raios-X de treino.