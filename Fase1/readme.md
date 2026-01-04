# **Tech Challenge - Fase 1: Sistema Inteligente de Suporte ao Diagnóstico**
Bem-vindo ao repositório do meu projeto para o Tech Challenge da Fase 1.

O objetivo deste desafio foi desenvolver um sistema de triagem hospitalar que utiliza Inteligência Artificial para apoiar a equipe médica. O projeto foi dividido em duas frentes: uma análise preditiva baseada em dados clínicos (Diabetes) e um módulo de visão computacional para análise de imagens (Pneumonia).

# **Estrutura do Projeto**
O projeto pode ser executado de duas formas: via Google Colab (recomendado para visualização rápida) ou Localmente via scripts Python.

**Sistema_de_Suporte_ao_Diagnóstico_de_Diabetes.ipynb:** Notebook completo com toda a narrativa de Data Science, gráficos interativos e comparação de modelos.

**analise_diabetes.py:** Versão em script para rodar a análise de dados tabulares no terminal local.

**pneumonia.py:** Script do desafio extra (Visão Computacional) utilizando Redes Neurais (CNN).

**diabetes.csv:** Dataset utilizado na primeira parte.

Como o dataset para o projeto de visão computacional é muito grande vou disponibilizar o link para download:
[Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


# **Como Executar**
Pré-requisitos
Se for rodar localmente, deixei um arquivo requeriments.txt disponível:

```bash
#!/bin/bash
pip install -r requirements.txt

```

## 1. Módulo de Diabetes (Dados Tabulares)
Nesta etapa, o objetivo é prever se um paciente tem diabetes com base em exames clínicos.

**Opção A:** Google Colab (Notebook) Basta abrir o arquivo .ipynb no Google Colab, fazer o upload do diabetes.csv na aba de arquivos e rodar todas as células. O notebook contém explicações detalhadas célula a célula.

**Opção B**: Python Local Certifique-se de que o arquivo diabetes.csv está na mesma pasta do script e execute:

```bash
#!/bin/bash
python analise_diabetes.py

```

O script abrirá janelas com os gráficos de distribuição e correlação. Feche as janelas para que o script continue o treinamento dos modelos.
Ao termino ele gera os arquivos pkl para usar o teste_diabetes.py
Esse script que implementei permite interagir com o modelo treinado, simulando um médico inserindo dados de um novo paciente em tempo real.

**Executar o simulador:**

```bash
python teste_diabetes.py
```
**Abaixo exemplos de dados que simulam diabeticos e não diabeticos:**

**Diabetico:** Gravidez: 6 | Glicose: 148 | Pressão: 72 | Pele: 35 | Insulina: 0 | IMC: 33.6 | Pedigree: 0.627 | Idade: 50

**Não Diabetes:** Gravidez: 1 | Glicose: 85 | Pressão: 66 | Pele: 29 | Insulina: 0 | IMC: 26.6 | Pedigree: 0.351 | Idade: 31

## 2. Módulo de Pneumonia (Visão Computacional - Bônus)
Nesta etapa, utilizamos Redes Neurais Convolucionais (CNN) para detectar pneumonia em Raio-X de tórax.

**Baixe o dataset no Kaggle:** Chest X-Ray Images (Pneumonia).

Extraia os arquivos. A estrutura deve ficar: pasta/chest_xray/train e pasta/chest_xray/test.

Abra o arquivo **pneumonia.py** e edite a variável CAMINHO_BASE na linha 15 para apontar para onde você salvou as imagens no computador.

**Execute:**

```bash
python pneumonia.py
```

O treinamento pode demorar alguns minutos dependendo da sua máquina.

## **Detalhes Técnicos e Metodologia**
**Parte 1:** Predição de Diabetes
Utilizei o dataset do Pima Indians Diabetes Database. Durante a Análise Exploratória, notei algo crítico:

Variáveis como Glucose, Insulin e BMI possuíam valores iguais a **0 (zero)**, o que é biologicamente impossível.

**Solução:** Em vez de remover essas linhas (o que reduziria ia muito a base), tratei esses zeros como missing values (imaginei que os pacientes não informaram ou só nao preencheram mesmo) e os preenchi com a mediana da coluna. Isso tornou o modelo muito mais robusto.

**Testei três algoritmos:**

**KNN** (K-Nearest Neighbors)

**SVM** (Support Vector Machine) - Kernel Linear

**Random Forest**

**Resultado:** O Random Forest apresentou o melhor desempenho geral na métrica **AUC (Area Under Curve)**, lidando bem com a complexidade dos dados.

**Parte 2:** Detecção de Pneumonia para o desafio de imagem, construí uma Rede Neural Convolucional (CNN) do zero utilizando **TensorFlow/Keras**.

**Arquitetura:** 3 camadas de Convolução + MaxPooling, seguidas de camadas densas.

**Prevenção de Overfitting:** O dataset de Raio-X não é gigantesco, então apliquei Data Augmentation (rotações, zoom e espelhamento nas imagens de treino) e camadas de Dropout na rede. Isso garante que a IA aprenda a identificar a doença e não apenas "decore" as imagens.
Usei essa técnica após alguns testes mal sucedidos.
Usei como referência inicial o seguinte artigo: https://www.datacamp.com/pt/tutorial/complete-guide-data-augmentation

**Resultados Esperados,** ao rodar os projetos, você verá:

**Gráficos de Distribuição:** Mostran o desbalanceamento das classes.

**Matrizes de Confusão:** Detalha os Falsos Positivos e Falsos Negativos (crítico na área médica).

**Curvas ROC:** Compara a performance dos algoritmos.

**Gráficos de Acurácia/Perda (Loss):** Mostra a evolução do aprendizado da Rede Neural época por época.

### **Teste Rápido**
Após treinar o modelo ou realizar o download do meu modelo pronto. Utilize o script de teste para verificar diagnósticos em imagens aleatórias:

1. Garantir que o arquivo `modelo_pneumonia_cnn.keras` foi gerado.
2. Depois Execute:
```bash
python teste_raiox.py
```
