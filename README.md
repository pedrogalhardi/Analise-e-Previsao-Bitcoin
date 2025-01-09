# Análise e Previsão do Bitcoin

Este projeto utiliza a biblioteca Streamlit para criar uma aplicação interativa que analisa e prevê os preços do Bitcoin. Ele integra dados históricos e atuais utilizando a API CoinGecko, e emprega um modelo LSTM (Long Short-Term Memory) para previsão de séries temporais.

## Requisitos

Certifique-se de ter o Python 3.8 ou superior instalado em seu ambiente.

## Configuração do Ambiente

1. Clone este repositório:
   ```bash
   git clone https://github.com/pedrogalhardi/Analise-e-Previsao-Bitcoin.git
   cd Analise-e-Previsao-Bitcoin
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   ```

3. Ative o ambiente virtual:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/MacOS**:
     ```bash
     source venv/bin/activate
     ```

4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Execução do Projeto

1. Inicie a aplicação Streamlit:
   ```bash
   streamlit run main.py
   ```

2. Acesse a aplicação no navegador através do link exibido no terminal, geralmente: `http://localhost:8501/`

## Estrutura do Projeto

- `main.py`: Arquivo principal da aplicação Streamlit.
- `requirements.txt`: Lista de dependências necessárias para o projeto.
- `README.md`: Documentação do projeto.

## Funcionalidades

1. **Análise em Tempo Real**: Exibe o preço atual do Bitcoin utilizando a API CoinGecko.
2. **Análise Histórica**: Carrega dados históricos para visualização e análise.
3. **Previsão com LSTM**: Utiliza um modelo LSTM para prever o comportamento do Bitcoin com base em dados históricos.

## Dependências

As principais dependências incluem:

- `streamlit`
- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Para uma lista completa, consulte o arquivo `requirements.txt`.
