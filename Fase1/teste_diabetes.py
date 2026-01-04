import joblib
import numpy as np
import os

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

print("Teste de Diabetes")

try:
    modelo = joblib.load('modelo_diabetes_rf.pkl')
    scaler = joblib.load('scaler_diabetes.pkl')
    print("Carregado..\n")
except FileNotFoundError:
    print('Arquivos nao encontrados.')
    exit()

def fazer_diagnostico():
    print("\nInserir dados:\n")
    try:
        gravidez = float(input("1. Número de gravidezes (ex: 0, 1, 2...): "))
        glicose  = float(input("2. Glicose (mg/dL) (ex: 100, 150...): "))
        pressao  = float(input("3. Pressão Sanguínea (mm Hg) (ex: 70, 80...): "))
        pele     = float(input("4. Espessura da pele (mm) (ex: 20, 30...): "))
        insulina = float(input("5. Insulina (mu U/ml) (ex: 80, 150...): "))
        imc      = float(input("6. IMC (Índice de Massa Corporal) (ex: 25.5, 30.0...): "))
        pedigree = float(input("7. Histórico Familiar (DiabetesPedigree) (ex: 0.5): "))
        idade    = float(input("8. Idade (anos): "))
    except ValueError:
        print("\nDigite apenas numeros e ponto para decimais.")
        return
    
    novos_dados = np.array([[gravidez, glicose, pressao, pele, insulina, imc, pedigree, idade]])
    dados_escalados = scaler.transform(novos_dados)
    #Previsao
    predicao = modelo.predict(dados_escalados)
    probabilidade = modelo.predict_proba(dados_escalados)

    resultado = "DIABETICO" if predicao[0] == 1 else "NÃO DIABETICO"
    chance_diabetes = probabilidade[0][1] * 100

    print("\n" + "-"*40)
    print(f"RESULTADO: {resultado}")
    print(f"Probabilidade calculada de diabetes: {chance_diabetes:.2f}%")
    print("-"*40)

while True:
    fazer_diagnostico()
    continuar = input("\nDeseja testar outro paciente? (s/n): ").lower()
    if continuar != 's':
        break
    limpar_tela()