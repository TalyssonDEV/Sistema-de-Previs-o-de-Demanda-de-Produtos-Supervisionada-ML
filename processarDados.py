import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def carregar_dados(caminho):
    dados = pd.read_csv(caminho)
    return dados


def filtrar_dados(dados):
    dados['tipo'] = dados['tipo'].astype(str).str.strip().str.lower()
    return dados[dados['tipo'] == 'real'].copy()


def criar_lags(dados, quantidade_lags):
    for i in range(1, quantidade_lags + 1):
        dados[f'lag_{i}'] = dados['Demand'].shift(i)

    colunas_lag = [f'lag_{i}' for i in range(1, quantidade_lags + 1)]
    dados = dados.dropna(subset=colunas_lag)

    return dados


def dividir_treino_teste(dados, proporcao=0.8):
    tamanho_treino = int(len(dados) * proporcao)
    treino = dados.iloc[:tamanho_treino]
    teste = dados.iloc[tamanho_treino:]
    return treino, teste


def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    return mae, rmse, mape

def main():

    # Carregar e preparar dados
    dados = carregar_dados('database/hist_prev_gasolina.csv')
    dados = filtrar_dados(dados)

    dados['date'] = pd.to_datetime(dados['date'])
    dados = dados.sort_values('date')

    dados['Demand'] = pd.to_numeric(dados['Demand'], errors='coerce')
    dados = dados.dropna(subset=['Demand'])

    dados = criar_lags(dados, quantidade_lags=6)

    features = [f'lag_{i}' for i in range(1, 7)]

    treino, teste = dividir_treino_teste(dados)

    X_train = treino[features]
    y_train = treino['Demand']
    X_test = teste[features]
    y_test = teste['Demand']

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    resultados = {}


    # Baseline
    y_pred_baseline = teste['lag_1']
    resultados['Baseline'] = calcular_metricas(y_test, y_pred_baseline)


    # Regressão Linear

    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train_scaled, y_train)
    y_pred_lr = modelo_lr.predict(X_test_scaled)
    resultados['Linear Regression'] = calcular_metricas(y_test, y_pred_lr)


    #  KNN
    modelo_knn = KNeighborsRegressor(n_neighbors=5)
    modelo_knn.fit(X_train_scaled, y_train)
    y_pred_knn = modelo_knn.predict(X_test_scaled)
    resultados['KNN'] = calcular_metricas(y_test, y_pred_knn)


    # MLP
    modelo_mlp = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        max_iter=3000,
        early_stopping=True,
        random_state=42
    )
    modelo_mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = modelo_mlp.predict(X_test_scaled)
    resultados['MLP'] = calcular_metricas(y_test, y_pred_mlp)

 
    #  Mostrar Tabela 
    tabela_resultados = pd.DataFrame(resultados, index=['MAE', 'RMSE', 'MAPE']).T

    print("\n===== COMPARAÇÃO DE MODELOS =====")
    print(tabela_resultados.round(2))

    melhor_modelo = tabela_resultados['MAPE'].idxmin()
    print(f"\n🏆 Melhor modelo baseado em MAPE: {melhor_modelo}")


    #  Gráfico Comparativo
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 6))

    plt.plot(teste['date'], y_test, label='Real', linewidth=2)
    plt.plot(teste['date'], y_pred_baseline, label='Baseline', linestyle=':')
    plt.plot(teste['date'], y_pred_lr, label='Linear', linestyle='--')
    plt.plot(teste['date'], y_pred_knn, label='KNN', linestyle='-.')
    plt.plot(teste['date'], y_pred_mlp, label='MLP', alpha=0.7)

    plt.title('Comparação de Modelos - Previsão de Demanda', fontsize=16)
    plt.xlabel('Data')
    plt.ylabel('Demand')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
