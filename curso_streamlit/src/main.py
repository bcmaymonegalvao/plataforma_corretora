# importar as bibliotecas
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

# Configuração da página para melhor performance
st.set_page_config(layout="wide")

# Funções com cache para melhor performance
@st.cache_data
def carregar_tickers_acoes():
    url = "https://raw.githubusercontent.com/bcmaymonegalvao/plataforma_corretora/refs/heads/main/curso_streamlit/src/IBOV.csv"
    base_tickers = pd.read_csv(url, sep=";", encoding="cp1252")
    bom_cp1252 = ''.join(chr(code) for code in (239, 187, 191))
    base_tickers.columns = (
        base_tickers.columns.str.replace("\ufeff", "", regex=False)
        .str.replace(bom_cp1252, "", regex=False)
        .str.strip()
    )
    codigo_coluna = base_tickers.columns[0]
    tickers = (
        base_tickers[codigo_coluna]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return [f"{ticker}.SA" for ticker in tickers if ticker]

@st.cache_data(ttl=3600)  # Cache por 1 hora para evitar chamadas repetidas ao Yahoo Finance
def carregar_dados(empresas):
    if not empresas:  # Retorna DataFrame vazio se nenhuma ação for passada
        return pd.DataFrame()
    
    texto_tickers = " ".join(empresas)
    dados_acao = yf.Tickers(texto_tickers)
    cotacoes_acao = dados_acao.history(period="1d", start="2010-01-01", end="2024-07-01")["Close"]
    return cotacoes_acao

# Interface do usuário
st.write("""
# App Preço de Ações
O gráfico abaixo representa a evolução do preço das ações da BOVESPA ao longo dos anos
""")

# Carrega os tickers uma vez (com cache)
acoes = carregar_tickers_acoes()

# Sidebar com controles
with st.sidebar:
    st.header("Filtros")
    lista_acoes = st.multiselect("Escolha as ações para visualizar", acoes)
    
    # Só mostra o slider de datas se houver ações selecionadas
    if lista_acoes:
        dados = carregar_dados(lista_acoes)
        
        if not dados.empty:
            data_inicial = dados.index.min().to_pydatetime()
            data_final = dados.index.max().to_pydatetime()
            
            intervalo_data = st.slider(
                "Selecione o período",
                min_value=data_inicial,
                max_value=data_final,
                value=(data_inicial, data_final),
                step=timedelta(days=1)
            )
            
            dados = dados.loc[intervalo_data[0]:intervalo_data[1]]
            
            # Renomeia coluna se apenas uma ação selecionada
            if len(lista_acoes) == 1:
                dados = dados.rename(columns={lista_acoes[0]: "Close"})

# Exibição do gráfico
# Exibição do gráfico
if not lista_acoes:
    st.info("ℹ️ Selecione pelo menos uma ação no menu à esquerda para visualizar o gráfico")
    st.line_chart(pd.DataFrame())  # Gráfico vazio
else:
    if dados.empty:
        st.warning("⚠️ Não foi possível carregar dados para as ações selecionadas")
    else:
        # Criar um DataFrame combinado com preços e EMAs
        dados_plot = pd.DataFrame()
        
        for acao in dados.columns:
            # Adiciona o preço original
            dados_plot[f'{acao}'] = dados[acao]
            
            # Calcula a EMA de 9 dias
            dados_plot[f'{acao}_EMA9'] = dados[acao].ewm(span=9, adjust=False).mean()
        
        # Plotar o gráfico com ambas as linhas
        st.line_chart(dados_plot)
        

# Cálculo de performance (só executa se houver ações selecionadas)
if lista_acoes and not dados.empty:
    texto_performance_ativos = ""
    carteira = [1000 for _ in lista_acoes]
    total_inicial_carteira = sum(carteira)
    
    for i, acao in enumerate(lista_acoes):
        col_name = "Close" if len(lista_acoes) == 1 else acao
        try:
            performance_ativo = dados[col_name].iloc[-1] / dados[col_name].iloc[0] - 1
            performance_ativo = float(performance_ativo)
            
            carteira[i] = carteira[i] * (1 + performance_ativo)
            
            if performance_ativo > 0:
                texto_performance_ativos += f"  \n{acao}: :green[{performance_ativo:.1%}]"
            elif performance_ativo < 0:
                texto_performance_ativos += f"  \n{acao}: :red[{performance_ativo:.1%}]"
            else:
                texto_performance_ativos += f"  \n{acao}: {performance_ativo:.1%}"
        except KeyError:
            texto_performance_ativos += f"  \n{acao}: Dados indisponíveis"
    
    total_final_carteira = sum(carteira)
    performance_carteira = total_final_carteira / total_inicial_carteira - 1
    
    if performance_carteira > 0:
        texto_performance_carteira = f"Performance da carteira:  \n:green[{performance_carteira:.1%}]"
    elif performance_carteira < 0:
        texto_performance_carteira = f"Performance da carteira:  \n:red[{performance_carteira:.1%}]"
    else:
        texto_performance_carteira = f"Performance da carteira:  \n{performance_carteira:.1%}"
    
    st.write(f"""
    ### Performance dos Ativos
    Essa foi a performance de cada ativo no período selecionado
    
    {texto_performance_ativos}
    
    {texto_performance_carteira}
    """)
