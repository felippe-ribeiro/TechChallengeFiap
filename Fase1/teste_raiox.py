import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import random

CAMINHO_BASE = r'tech_challenge_img\chest_xray'
TEST_DIR = os.path.join(CAMINHO_BASE, 'test')
MODELO_PATH = 'modelo_pneumonia_cnn.keras'

IMG_HEIGHT, IMG_WIDTH = 150, 150 

try:
    model = tf.keras.models.load_model(MODELO_PATH)
    print('Modelo carredado :)')

except OSError:
    print(f'Erro nao encotnrado arquivo: {MODELO_PATH}.')
    exit()

def carregar_e_preparar_imagem(caminho_img):
    #Carregar imagem no tamanho certo
    img = image.load_img(caminho_img, target_size=(IMG_HEIGHT, IMG_WIDTH))

    #Converter para array
    img_array = image.img_to_array(img)

    #normalizando igual o treinamento
    img_array = img_array / 255.0

    #Dimensão extrpa para representar o batch
    #pois minha rede espera receber um lote de imagens mesmo que seja apenas 1
    #Ex: (150,150,3) --> para (1,150,150,3)
    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array

def testar_aleatorio():
    #entre a pasta normal e com pneumonia
    categoria = random.choice(['NORMAL', 'PNEUMONIA'])
    pasta_escolhida = os.path.join(TEST_DIR, categoria)

    #pega 1 img
    arquivos = os.listdir(pasta_escolhida)
    arquivo_img = random.choice(arquivos)
    caminho_completo = os.path.join(pasta_escolhida, arquivo_img)

    img_original, img_processada = carregar_e_preparar_imagem(caminho_completo)

    #PREVISAO
    #Vai retornar numero entre 0 e 1
    predicao = model.predict(img_processada, verbose=0)
    score = predicao[0][0]

    if score > 0.5:
        resultado = 'PNEUMONIA'
        confianca = score * 100
        cor = 'red'
    else:
        resultado = 'NORMAL'
        confianca = (1 - score) * 100
        cor = 'green'

    plt.figure(figsize=(6, 6))
    plt.imshow(img_original)
    plt.axis('off')
    plt.title(f"Real: {categoria}\n Prev. IA: {resultado} ({confianca:.2f}%)", color=cor, fontsize=14, fontweight='bold')
    plt.show()

    print(f'\n--- Resultado ---')
    print(f"Imagem Real: {categoria}")
    print(f"Previsão da IA: {resultado}")
    print(f"Confiançã: {confianca:.2f}%")


while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    input("Enter para testar imagem nova ou ctrl+c e enter para sair")
    try:
        testar_aleatorio()
    except Exception as e:
        print(f"erro ao carregar a imagem: {e}")
        break