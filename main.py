import time
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt

# funcao para obter dados historicos 
@st.cache_data(show_spinner=False)
def get_historical_data(symbol="bitcoin", days=30):
    time.sleep(1) 
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily",
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "prices" in data and data["prices"]:
            prices = data["prices"]
            df = pd.DataFrame(prices, columns=["timestamp", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            df["close"] = df["close"].astype(float)
            return df[["timestamp", "close"]]
        else:
            raise ValueError("A resposta da API não contém 'prices'.")
    except Exception as e:
        st.error(f"Erro ao obter dados históricos: {e}")
        return pd.DataFrame()

# funcao de pre-processamento
def preprocess_data(df, sequence_length=30):

    if len(df) < sequence_length + 1:
        raise ValueError(f"Dados insuficientes: {len(df)} dias para sequence_length={sequence_length}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_close"] = scaler.fit_transform(df["close"].values.reshape(-1, 1))

    data = df["scaled_close"].values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler

# funcao para criar o modelo LSTM
def criar_modelo(input_shape):

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# obter dados em tempo real 
@st.cache_data(show_spinner=False)
def get_real_time_price(symbol="bitcoin"):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": symbol,
            "vs_currencies": "usd"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if symbol in data and "usd" in data[symbol]:
            return float(data[symbol]["usd"])
        else:
            raise ValueError(f"Dados ausentes para o símbolo '{symbol}'.")
    except Exception as e:
        st.error(f"Erro ao obter preço em tempo real: {e}")
        return None

st.title("Análise e Previsão do Bitcoin com CoinGecko API")

# preco em tempo real
st.header("Preço em Tempo Real")
real_time_price = get_real_time_price()
if real_time_price:
    st.metric(label="Preço Atual do Bitcoin (USD)", value=f"${real_time_price:,.2f}")
else:
    st.warning("Não foi possível obter o preço em tempo real.")

st.header("Dados Históricos do Bitcoin")
sequence_length = 30
days = st.slider(
    "Número de dias a carregar:",
    min_value=sequence_length, 
    max_value=365,
    value=30,
    step=1
)

st.write("Carregando dados históricos")
with st.spinner("Carregando..."):
    df = get_historical_data(symbol="bitcoin", days=days)

if not df.empty:
    try:
        X_train, y_train, X_test, y_test, scaler = preprocess_data(df, sequence_length=sequence_length)
        st.write(f"Dados pré-processados com sucesso! {len(X_train)} amostras para treino, {len(X_test)} para teste.")
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        # criando e treinando o modelo
        modelo = criar_modelo(input_shape=(X_train.shape[1], X_train.shape[2]))
        modelo.fit(X_train, y_train, epochs=20, batch_size=32)

        # avaliar o modelo
        loss = modelo.evaluate(X_test, y_test)
        st.write(f"Erro médio quadrático nos dados de teste: {loss:.4f}")

        # fazendo previsoes
        previsoes = modelo.predict(X_test)

        # reescalando previsoes para o formato original
        previsoes_reescaladas = scaler.inverse_transform(previsoes)
        y_test_reescalado = scaler.inverse_transform(y_test.reshape(-1, 1))

        # resultados no Streamlit
        st.subheader("Comparação entre valores reais e previstos")
        fig, ax = plt.subplots()
        ax.plot(y_test_reescalado, label="Real", color="blue")
        ax.plot(previsoes_reescaladas, label="Previsto", color="red")
        ax.legend()
        st.pyplot(fig)
        st.subheader("Previsões (reescaladas)")
        st.write(pd.DataFrame({
            "Real": y_test_reescalado.flatten(),
            "Previsto": previsoes_reescaladas.flatten()
        }))
    except ValueError as ve:
        st.error(str(ve))
else:
    st.error("Não foi possível carregar os dados históricos.")