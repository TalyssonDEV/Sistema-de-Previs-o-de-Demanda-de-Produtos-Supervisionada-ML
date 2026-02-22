import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def carregar_dados_e_exogenas(caminho_principal, caminho_exogenas):
    dados = pd.read_csv(caminho_principal)
    exogenas = pd.read_csv(caminho_exogenas)
    
    # Garantir que a coluna date esteja no mesmo formato para o merge
    dados['date'] = pd.to_datetime(dados['date'])
    exogenas['date'] = pd.to_datetime(exogenas['date'])
    
    # Fazer o merge usando date como chave (left join para manter todas as datas originais)
    dados = pd.merge(dados, exogenas, on='date', how='left')
    
    # Preencher valores nulos das exógenas
    dados = dados.ffill().bfill()
    
    return dados


def filtrar_dados(dados):
    dados['tipo'] = dados['tipo'].astype(str).str.strip().str.lower()
    return dados[dados['tipo'] == 'real'].copy()


def criar_lags(dados, quantidade_lags):
    for i in range(1, quantidade_lags + 1):
        dados[f'lag_{i}'] = dados['Demand'].shift(i)

    # Criar médias móveis baseadas nos lags recentes para suavizar a curva de aprendizado
    dados['rolling_mean_3'] = dados['lag_1'].rolling(window=3).mean()
    dados['rolling_mean_6'] = dados['lag_1'].rolling(window=6).mean()

    colunas_lag = [f'lag_{i}' for i in range(1, quantidade_lags + 1)] + ['rolling_mean_3', 'rolling_mean_6']
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

def processar_produto(caminho_dados, nome_produto, usar_exogenas=False):
    print(f"\n{'='*40}")
    print(f" PROCESSANDO PRODUTO: {nome_produto}")
    print(f"{'='*40}")

    if usar_exogenas:
        dados = carregar_dados_e_exogenas(caminho_dados, 'database/middle_exogenas_historico.csv')
    else:
        dados = pd.read_csv(caminho_dados)
        dados['date'] = pd.to_datetime(dados['date'])

    dados = filtrar_dados(dados)

    dados = dados.sort_values('date')

    dados['Demand'] = pd.to_numeric(dados['Demand'], errors='coerce')
    dados = dados.dropna(subset=['Demand'])

    dados['ano'] = dados['date'].dt.year
    dados['mes'] = dados['date'].dt.month
    dados['mes_sin'] = np.sin(2 * np.pi * dados['mes'] / 12.0)
    dados['mes_cos'] = np.cos(2 * np.pi * dados['mes'] / 12.0)

    dados = criar_lags(dados, quantidade_lags=6)

    # lags + rolling means + datas + exogenas
    features = [f'lag_{i}' for i in range(1, 7)] + ['rolling_mean_3', 'rolling_mean_6', 'ano', 'mes_sin', 'mes_cos']
    if usar_exogenas:
        features += ['WTI_real', 'Industrial_Production', 'Freight_Transp', 'Unemployment', 'CPI']

    treino, teste = dividir_treino_teste(dados)

    X_train = treino[features]
    y_train = treino['Demand']
    X_test = teste[features]
    y_test = teste['Demand']

    # Normalização
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    resultados = {}


    #  KNN
    # Utilizando métrica 'manhattan' (p=1) que performa discretamente melhor em séries contínuas isoladas
    modelo_knn = KNeighborsRegressor(n_neighbors=3, weights='distance', p=1)
    modelo_knn.fit(X_train_scaled, y_train)
    y_pred_knn = modelo_knn.predict(X_test_scaled)

    y_pred_knn = np.maximum(y_pred_knn, 0)
    resultados['KNN'] = calcular_metricas(y_test, y_pred_knn)


    # MLP
    # Escalonar também a variável alvo (y)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    modelo_mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=5000,
        activation='relu',
        solver='lbfgs', 
        alpha=0.001, 
        random_state=42
    )
    modelo_mlp.fit(X_train_scaled, y_train_scaled)
    y_pred_mlp_scaled = modelo_mlp.predict(X_test_scaled)
    
    # Inverter a transformação para voltar à escala original
    y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).ravel()
    y_pred_mlp = np.maximum(y_pred_mlp, 0)
    resultados['MLP'] = calcular_metricas(y_test, y_pred_mlp)

 
    #  Mostrar Tabela 
    tabela_resultados = pd.DataFrame(resultados, index=['MAE', 'RMSE', 'MAPE']).T

    print(f"\n===== COMPARAÇÃO DE MODELOS ({nome_produto}) =====")
    print(tabela_resultados.round(2))

    # Escolher o melhor modelo entre KNN e MLP
    modelos_foco = ['KNN', 'MLP']
    melhor_modelo = tabela_resultados.loc[modelos_foco, 'MAPE'].idxmin()
    print(f"\nMelhor modelo baseado em MAPE: {melhor_modelo}")


    #  Gráfico Comparativo (Linhas temporal)
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 6))

    plt.plot(teste['date'], y_test, label='Real', linewidth=2)
    plt.plot(teste['date'], y_pred_knn, label='KNN', linestyle='-.')
    plt.plot(teste['date'], y_pred_mlp, label='MLP', alpha=0.7, color='purple')

    plt.title(f'Comparação de Modelos - Previsão de Demanda ({nome_produto})', fontsize=16)
    plt.xlabel('Data')
    plt.ylabel('Demand')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Gráfico 2: "Matriz de Confusão" de Regressão (Dispersão Real x Previsto para o MLP)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_mlp, alpha=0.6, color='purple', label='Precisação (MLP)')
    
    # Linha ideal - os pontos devem cair entre esses valores
    min_val = min(y_test.min(), y_pred_mlp.min())
    max_val = max(y_test.max(), y_pred_mlp.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Linha de Perfeição')
    
    plt.title(f'Real vs Previsto ("Matriz de Acertos") - MLP ({nome_produto})', fontsize=14)
    plt.xlabel('Valores Reais (Demand)')
    plt.ylabel('Valores Previstos (Demand)')
    plt.legend()
    plt.tight_layout()


def main():
    processar_produto('database/hist_prev_gasolina.csv', 'Gasolina', usar_exogenas=False)
    processar_produto('database/hist_prev_middle.csv', 'Diesel', usar_exogenas=False)
    plt.show()

if __name__ == '__main__':
    main()
