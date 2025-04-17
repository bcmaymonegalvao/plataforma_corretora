# importar as bibliotecas
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import timedelta

# criar as funções de carregamento de dados
    # cotações do ITAU - ITUB4 - 2010 a 2024
@st.cache_data
def carregar_dados(empresas):
    texto_tickers = " ".join(empresas)
    dados_acao = yf.Tickers(texto_tickers)
    cotacoes_acao = dados_acao.history(period = "1d", start = "2010-01-01", end = "2024-07-01")
    print(cotacoes_acao)
    cotacoes_acao = cotacoes_acao["Close"]
    return cotacoes_acao

def carregar_tickers_acoes():
    url = "https://raw.githubusercontent.com/bcmaymonegalvao/plataforma_corretora/refs/heads/main/curso_streamlit/src/IBOV.csv"
    base_tickers = pd.read_csv(url, sep = ";")
    tickers = list(base_tickers["Código"])
    tickers = [item + ".SA" for item in tickers]
    return tickers


acoes = carregar_tickers_acoes()
dados = carregar_dados(acoes)

# criar a interface do streamlit
st.write("""
# App Preço de ações
O gráfico abaixo representa a evolução do preço das ações da BOVESPA ao longo dos anos
""")#markdown

# prepara as visualizações = filtros
lista_acoes = st.sidebar.multiselect("Escolha as acoes para visualizar", dados.columns)
if lista_acoes:
    dados = dados[lista_acoes]
    if len(lista_acoes) == 1:
        acao_unica = lista_acoes[0]
        dados = dados.rename(columns = {acao_unica: "Close"})

# filtro de datas
data_inicial = dados.index.min().to_pydatetime()
data_final =  dados.index.max().to_pydatetime()
intervalo_data = st.sidebar.slider("Selecione o período",
                                    min_value = data_inicial,
                                    max_value = data_final,
                                    value = (data_inicial, data_final),
                                    step = timedelta(days = 1))
print(intervalo_data)
dados = dados.loc[intervalo_data[0]:intervalo_data[1]]  

# criar o gráfico
st.line_chart(dados)

# calculo de performance
texto_performance_ativos = ""

if len(lista_acoes) == 0:
    lista_acoes = list(dados.columns)
elif len(lista_acoes) == 1:
    dados = dados.rename(columns = {"Close" :acao_unica})

carteira = [1000 for acao in lista_acoes]
total_inicial_carteira = sum(carteira)

for i, acao in enumerate(lista_acoes):
    performance_ativo = dados[acao].iloc[-1] / dados[acao].iloc[0] - 1
    performance_ativo = float(performance_ativo)

    carteira[i] = carteira[i] * (1 + performance_ativo)

    if performance_ativo > 0:
        texto_performance_ativos = texto_performance_ativos + f"  \n{acao}: :green[{performance_ativo: .1%}]"
    elif performance_ativo < 0:
        texto_performance_ativos = texto_performance_ativos + f"  \n{acao}: :red[{performance_ativo: .1%}]"    
    else:
        texto_performance_ativos = texto_performance_ativos + f"  \n{acao}: {performance_ativo: .1%}"

total_final_carteira = sum(carteira)
performance_carteira = total_final_carteira / total_inicial_carteira - 1

if performance_carteira > 0:
    texto_performance_carteira = f"Performance da carteira com todos os ativos: :green[{performance_ativo:.1%}]"
elif performance_ativo < 0:
    texto_performance_carteira = f"Performance da carteira com todos os ativos: :red[{performance_ativo:.1%}]"    
else:
    texto_performance_carteira = f"Performance da carteira com todos os ativos: {performance_ativo:.1%}"

st.write(f"""
### Performance dos Ativos
Essa foi a performance de cada ativo no período selecionado

{texto_performance_ativos}                

{texto_performance_carteira}
""")