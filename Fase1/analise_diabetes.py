import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score

sns.set_theme(style="whitegrid")

print("\n--- 1. Carregando Dataset ---")
try:
    df = pd.read_csv('diabetes.csv')
    print(f"Dataset carregado com sucesso! Dimensões: {df.shape}")
    print("\nPrimeiras linhas:")
    print(df.head())
except FileNotFoundError:
    print("Arquivo 'diabetes.csv' não encontrado.")
    exit()

print("\n--- 2. Análise Exploratória (Feche as janelas dos gráficos para continuar) ---")

# Gráfico 1: Distribuição das Classes (Balanceamento)
# Verificar se tem classes desbalanceadas (muitos saudáveis e poucos doentes), o que pode enviesar o modelo.
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df, palette='viridis')
plt.title('Distribuição: 0 (Não Diabético) vs 1 (Diabético)')
plt.xlabel('Diagnóstico')
plt.ylabel('Quantidade')
print("Exibindo gráfico de distribuição...")
plt.show()

# Gráfico 2: Correlação e Boxplots
# Entender quais variáveis (Glicose, IMC, Idade) mais impactam o resultado.
# Boxplots vai ajudar a ver a separação das medianas entre os grupos.
features_interesse = ['Glucose', 'BMI', 'Age', 'Insulin']
plt.figure(figsize=(12, 8))
for i, col in enumerate(features_interesse):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='Outcome', y=col, data=df, palette='coolwarm')
    plt.title(f'{col} por Diagnóstico')
plt.tight_layout()
print("Exibindo boxplots das variáveis principais...")
plt.show()

print("\n--- 3. Tratamento de Dados e Normalização ---")

# Tratamento de Zeros 
# Variáveis como Glicose e Pressão não podem ser 0. Significa que falta dados
cols_erradas = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_erradas] = df[cols_erradas].replace(0, np.nan)

# Preenchendo com a mediana
print("Substituindo valores 0  pela mediana da coluna.")
imputer = SimpleImputer(strategy='median')
df[cols_erradas] = imputer.fit_transform(df[cols_erradas])

# Separação
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Divisão Treino e Teste
# Stratify=y garante a proporção
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalização (StandardScaler)
# Colocar na mesma escala numerica
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Dados normalizados e divididos.")

print("\n--- 4. Treinamento dos Modelos ---")

modelos = {
    "KNN (Vizinhos Próximos)": KNeighborsClassifier(n_neighbors=9),
    "SVM (Kernel Linear)": SVC(kernel='linear', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

resultados = {} 

for nome, modelo in modelos.items():
    print(f"Treinando {nome}...")
    modelo.fit(X_train_scaled, y_train)
    
    y_pred = modelo.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    resultados[nome] = modelo 
    
    print(f"-> Acurácia: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 40)

print("\n--- 5. Gerando Gráficos de Avaliação ---")

# Plot da Curva ROC
# Mostra o trade-off entre sensibilidade e especificidade.
# O modelo com maior área sob a curva é o melhor generalista.
plt.figure(figsize=(10, 6))
for nome, modelo in resultados.items():
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{nome} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (50%)')
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdadeiros Positivos (Recall)')
plt.title('Comparação de Desempenho (Curva ROC)')
plt.legend()
print("Exibindo Curva ROC...")
plt.show()

# Plot das Matrizes de Confusão
# Identificar exatamente onde o modelo erra (Falso Negativo vs Falso Positivo)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (nome, modelo) in enumerate(resultados.items()):
    y_pred = modelo.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Matriz: {nome}')
    axes[i].set_xlabel('Predito')
    axes[i].set_ylabel('Real')

plt.tight_layout()
print("Exibindo Matrizes de Confusão...")
plt.show()


print("Salvando o Modelo")
modelo = resultados["Random Forest"]

try:
    joblib.dump(modelo, 'modelo_diabetes_rf.pkl')
    joblib.dump(scaler, 'scaler_diabetes.pkl')
    print("Arquivos salvos com sucesso:")
    print(" - modelo_diabetes_rf.pkl (A Inteligência Artificial)")
    print(" - scaler_diabetes.pkl (A Régua de normalização)")
except Exception as e:
    print("Erro ao salvar o modelo: {e}")
