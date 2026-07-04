import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import time


# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

st.set_page_config(
    page_title="Regressão Linear em Ações",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .small-text {
            color: #666;
            font-size: 0.9rem;
        }
        .ux-card {
            padding: 1rem;
            border-radius: 0.8rem;
            border: 1px solid #e6e6e6;
            background-color: #fafafa;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DADOS AUXILIARES DE UX
# ============================================================

TICKERS = {
    "Apple — AAPL": "AAPL",
    "Microsoft — MSFT": "MSFT",
    "Alphabet/Google — GOOGL": "GOOGL",
    "Amazon — AMZN": "AMZN",
    "Tesla — TSLA": "TSLA",
    "Meta — META": "META",
    "Netflix — NFLX": "NFLX",
    "Petrobras — PETR4.SA": "PETR4.SA",
    "Vale — VALE3.SA": "VALE3.SA",
    "Itaú — ITUB4.SA": "ITUB4.SA",
    "Bradesco — BBDC4.SA": "BBDC4.SA",
    "B3 — B3SA3.SA": "B3SA3.SA",
    "Siemens — SIE.DE": "SIE.DE",
    "BMW — BMW.DE": "BMW.DE",
    "Airbus — AIR.PA": "AIR.PA",
    "LVMH — MC.PA": "MC.PA",
    "Santander — SAN.MC": "SAN.MC",
}

FEATURES_CONFIG = {
    "Dia": {
        "coluna": "Dia",
        "descricao": "Representa a passagem do tempo na série histórica."
    },
    "Volume normalizado": {
        "coluna": "Volume_Norm",
        "descricao": "Volume negociado padronizado para comparação com outras variáveis."
    },
    "Volatilidade": {
        "coluna": "Volatilidade",
        "descricao": "Variação recente dos retornos, calculada em janela móvel de 20 dias."
    },
    "Média móvel de 7 dias": {
        "coluna": "MA7",
        "descricao": "Preço médio dos últimos 7 pregões."
    },
    "Média móvel de 21 dias": {
        "coluna": "MA21",
        "descricao": "Preço médio dos últimos 21 pregões."
    },
    "Range diário": {
        "coluna": "Range",
        "descricao": "Diferença entre preço máximo e mínimo do dia."
    },
}


# ============================================================
# FUNÇÕES
# ============================================================

@st.cache_data(ttl=7200)
def carregar_dados_yfinance(ticker, anos=5, max_retries=3):
    """
    Carrega dados históricos do Yahoo Finance com retry automático.
    Retorna: DataFrame, mensagem_erro
    """

    for tentativa in range(max_retries):
        try:
            end = datetime.today()
            start = end - timedelta(days=365 * anos)

            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                if tentativa < max_retries - 1:
                    time.sleep(2 ** tentativa)
                    continue
                return pd.DataFrame(), "Nenhum dado foi retornado para esse ticker."

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df.columns = [
                col.capitalize() if isinstance(col, str) else col
                for col in df.columns
            ]

            colunas_necessarias = {"Close", "High", "Low"}
            if not colunas_necessarias.issubset(set(df.columns)):
                return pd.DataFrame(), "Os dados retornados não possuem as colunas necessárias."

            df["Dia"] = range(len(df))
            df["Retorno"] = df["Close"].pct_change()

            if "Volume" in df.columns and df["Volume"].std() > 0:
                df["Volume_Norm"] = (df["Volume"] - df["Volume"].mean()) / df["Volume"].std()
            else:
                df["Volume_Norm"] = 0

            df["Volatilidade"] = df["Retorno"].rolling(window=20).std()
            df["MA7"] = df["Close"].rolling(window=7).mean()
            df["MA21"] = df["Close"].rolling(window=21).mean()
            df["Range"] = df["High"] - df["Low"]

            df = df.dropna()

            return df, None

        except Exception as e:
            error_msg = str(e)

            if (
                "Rate" in error_msg
                or "429" in error_msg
                or "Too Many Requests" in error_msg
            ):
                if tentativa < max_retries - 1:
                    time.sleep(2 ** (tentativa + 1))
                    continue
                return (
                    pd.DataFrame(),
                    "O Yahoo Finance bloqueou temporariamente as requisições. "
                    "Aguarde alguns minutos e tente novamente."
                )

            if tentativa < max_retries - 1:
                time.sleep(2 ** tentativa)
                continue

            return pd.DataFrame(), error_msg

    return pd.DataFrame(), "Erro desconhecido ao carregar os dados."


def calcular_metricas(y_test, y_pred, coeficientes=None, intercepto=None):
    mse = mean_squared_error(y_test, y_pred)
    return {
        "predicoes": y_pred,
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "coeficientes": coeficientes,
        "intercepto": intercepto
    }


def treinar_modelos(X_train, y_train, X_test, y_test):
    modelos = {}
    resultados = {}

    modelos_base = {
        "Linear Simples": LinearRegression(),
        "Ridge — regularização L2": Ridge(alpha=1.0),
        "Lasso — regularização L1": Lasso(alpha=0.1, max_iter=10000),
        "ElasticNet — L1 + L2": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    }

    for nome, modelo in modelos_base.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        modelos[nome] = modelo
        resultados[nome] = calcular_metricas(
            y_test,
            y_pred,
            coeficientes=modelo.coef_,
            intercepto=modelo.intercept_
        )

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    modelo_poly = LinearRegression()
    modelo_poly.fit(X_train_poly, y_train)
    y_pred_poly = modelo_poly.predict(X_test_poly)

    modelos["Polinomial — grau 2"] = (poly, modelo_poly)
    resultados["Polinomial — grau 2"] = calcular_metricas(
        y_test,
        y_pred_poly,
        coeficientes=modelo_poly.coef_,
        intercepto=modelo_poly.intercept_
    )

    return modelos, resultados


def interpretar_r2(r2):
    if r2 >= 0.85:
        return "Ótimo ajuste aos dados de teste."
    if r2 >= 0.65:
        return "Bom ajuste, mas ainda há erro relevante."
    if r2 >= 0.40:
        return "Ajuste moderado. Use com cautela."
    return "Baixo ajuste. O modelo pode não estar capturando bem o comportamento da ação."


def criar_df_metricas(resultados):
    metricas_df = pd.DataFrame({
        "Modelo": list(resultados.keys()),
        "R²": [resultados[m]["r2"] for m in resultados],
        "RMSE": [resultados[m]["rmse"] for m in resultados],
        "MAE": [resultados[m]["mae"] for m in resultados],
        "MSE": [resultados[m]["mse"] for m in resultados],
    })

    metricas_df["Interpretação"] = metricas_df["R²"].apply(interpretar_r2)
    return metricas_df.sort_values("R²", ascending=False)


def plotar_historico(dados, acao, split_date):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["Close"],
            mode="lines",
            name="Preço de fechamento"
        )
    )

    fig.add_vline(
        x=split_date,
        line_dash="dash",
        annotation_text="Início do teste",
        annotation_position="top"
    )

    fig.update_layout(
        title=f"Histórico de preços — {acao}",
        xaxis_title="Data",
        yaxis_title="Preço de fechamento",
        hovermode="x unified",
        height=500
    )

    return fig


def plotar_predicoes(y_test, y_pred, modelo_nome):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=y_test,
            mode="lines",
            name="Valor real"
        )
    )

    fig.add_trace(
        go.Scatter(
            y=y_pred,
            mode="lines",
            name="Predição"
        )
    )

    fig.update_layout(
        title=f"Valores reais vs predições — {modelo_nome}",
        xaxis_title="Observação no conjunto de teste",
        yaxis_title="Preço",
        hovermode="x unified",
        height=450
    )

    return fig


def plotar_scatter_real_predito(y_test, y_pred, modelo_nome):
    df_plot = pd.DataFrame({
        "Valor real": y_test,
        "Predição": y_pred
    })

    fig = px.scatter(
        df_plot,
        x="Valor real",
        y="Predição",
        title=f"Predições vs valores reais — {modelo_nome}",
        height=450
    )

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Predição perfeita"
        )
    )

    return fig


def plotar_residuos(y_test, y_pred):
    residuos = y_test - y_pred

    df_residuos = pd.DataFrame({
        "Observação": range(len(residuos)),
        "Resíduo": residuos,
        "Predição": y_pred
    })

    fig = px.scatter(
        df_residuos,
        x="Predição",
        y="Resíduo",
        title="Resíduos vs valores preditos",
        height=450
    )

    fig.add_hline(y=0, line_dash="dash")

    return fig, residuos


# ============================================================
# CABEÇALHO
# ============================================================

st.title("📈 Análise de Regressão Linear em Ações")

st.markdown(
    """
    Analise dados históricos de ações, compare modelos de regressão e visualize
    a qualidade das predições de forma guiada.

    <span class="small-text">
    Uso educacional. Este app não é recomendação de investimento.
    </span>
    """,
    unsafe_allow_html=True
)

with st.expander("Como usar este app em 3 passos"):
    st.markdown(
        """
        1. Escolha uma ação e configure o período histórico.
        2. Selecione quais variáveis deseja usar no modelo.
        3. Clique em **Executar análise** e compare os resultados nas abas.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configurações da análise")

    modo_iniciante = st.toggle(
        "Modo iniciante",
        value=True,
        help="Mostra mais explicações e reduz a carga técnica da interface."
    )

    ticker_label = st.selectbox(
        "Ação",
        options=list(TICKERS.keys()),
        help="Escolha a ação que será analisada."
    )

    ticker = TICKERS[ticker_label]

    anos_historico = st.slider(
        "Período histórico",
        min_value=1,
        max_value=5,
        value=5,
        help="Quantidade de anos de dados históricos usados na análise."
    )

    test_size_percent = st.slider(
        "Dados reservados para teste",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentual final da série histórica usado para testar o modelo."
    )

    st.divider()

    st.subheader("Variáveis do modelo")

    features_selecionadas_labels = st.multiselect(
        "Selecione as variáveis explicativas",
        options=list(FEATURES_CONFIG.keys()),
        default=[
            "Dia",
            "Volume normalizado",
            "Volatilidade",
            "Média móvel de 7 dias",
            "Média móvel de 21 dias",
            "Range diário"
        ],
        help="Essas variáveis serão usadas para tentar explicar o preço de fechamento."
    )

    if modo_iniciante:
        with st.expander("O que são essas variáveis?"):
            for nome, cfg in FEATURES_CONFIG.items():
                st.markdown(f"**{nome}:** {cfg['descricao']}")

    st.divider()

    executar = st.button(
        "🚀 Executar análise",
        type="primary",
        use_container_width=True
    )

    limpar_cache = st.button(
        "Limpar cache dos dados",
        use_container_width=True,
        help="Útil se o Yahoo Finance retornar dados inconsistentes ou antigos."
    )

    if limpar_cache:
        st.cache_data.clear()
        st.success("Cache limpo. Execute a análise novamente.")


# ============================================================
# ESTADO INICIAL
# ============================================================

if not executar:
    st.info("Configure a análise na barra lateral e clique em **Executar análise**.")
    st.stop()


# ============================================================
# VALIDAÇÕES DE ENTRADA
# ============================================================

if not features_selecionadas_labels:
    st.warning("Selecione pelo menos uma variável explicativa para treinar os modelos.")
    st.stop()

features_disponiveis = [
    FEATURES_CONFIG[label]["coluna"]
    for label in features_selecionadas_labels
]

test_size = test_size_percent / 100


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

with st.status("Preparando análise...", expanded=True) as status:
    st.write(f"Carregando dados históricos de **{ticker_label}**...")
    dados, erro = carregar_dados_yfinance(ticker, anos=anos_historico)

    if erro:
        status.update(label="Não foi possível concluir a análise.", state="error")

        st.error("Não foi possível carregar os dados da ação selecionada.")

        st.markdown(
            """
            **O que você pode tentar:**

            - aguardar alguns minutos e executar novamente;
            - escolher outra ação;
            - limpar o cache na barra lateral;
            - verificar sua conexão.
            """
        )

        with st.expander("Detalhes técnicos do erro"):
            st.code(erro)

        st.stop()

    st.write("Validando dados e variáveis selecionadas...")

    colunas_faltantes = [
        col for col in features_disponiveis
        if col not in dados.columns
    ]

    if colunas_faltantes:
        status.update(label="A análise foi interrompida.", state="error")
        st.error("Algumas variáveis selecionadas não estão disponíveis nos dados.")
        st.write(colunas_faltantes)
        st.stop()

    if len(dados) < 100:
        status.update(label="Dados insuficientes.", state="error")
        st.warning(
            "Foram encontradas poucas observações para uma análise confiável. "
            "Tente aumentar o período histórico ou escolher outra ação."
        )
        st.stop()

    st.write("Separando treino e teste...")

    X = dados[features_disponiveis].values
    y = dados["Close"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write("Treinando modelos...")
    modelos, resultados = treinar_modelos(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test
    )

    metricas_df = criar_df_metricas(resultados)
    melhor_modelo = metricas_df.iloc[0]["Modelo"]

    status.update(label="Análise concluída.", state="complete")


# ============================================================
# RESUMO EXECUTIVO
# ============================================================

st.success(f"Análise concluída para **{ticker_label}**.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Observações", f"{len(dados):,}".replace(",", "."))

with col2:
    st.metric("Variáveis usadas", len(features_disponiveis))

with col3:
    st.metric("Melhor modelo", melhor_modelo)

with col4:
    st.metric("Melhor R²", f"{metricas_df.iloc[0]['R²']:.4f}")

if modo_iniciante:
    st.info(
        f"O melhor modelo nesta execução foi **{melhor_modelo}**. "
        f"Interpretação: {interpretar_r2(metricas_df.iloc[0]['R²'])}"
    )


# ============================================================
# ABAS
# ============================================================

tab_resumo, tab_modelos, tab_diagnostico, tab_dados, tab_ajuda = st.tabs(
    [
        "📊 Resumo",
        "🔍 Modelos",
        "🧪 Diagnóstico",
        "🗃️ Dados",
        "📚 Ajuda"
    ]
)


# ============================================================
# TAB 1 — RESUMO
# ============================================================

with tab_resumo:
    st.header("Resumo da análise")

    col1, col2 = st.columns([2, 1])

    with col1:
        split_date = dados.index[len(X_train)]
        fig_historico = plotar_historico(dados, ticker, split_date)
        st.plotly_chart(fig_historico, use_container_width=True)

    with col2:
        st.subheader("Configuração usada")

        st.write(f"**Ação:** {ticker_label}")
        st.write(f"**Período:** {anos_historico} ano(s)")
        st.write(f"**Treino:** {100 - test_size_percent}%")
        st.write(f"**Teste:** {test_size_percent}%")

        with st.expander("Variáveis selecionadas"):
            for label in features_selecionadas_labels:
                st.markdown(f"- **{label}**")

    st.subheader("Ranking dos modelos")

    st.dataframe(
        metricas_df.style.format(
            {
                "R²": "{:.4f}",
                "RMSE": "{:.2f}",
                "MAE": "{:.2f}",
                "MSE": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    csv_metricas = metricas_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Baixar métricas em CSV",
        data=csv_metricas,
        file_name=f"metricas_{ticker}.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 2 — MODELOS
# ============================================================

with tab_modelos:
    st.header("Análise dos modelos")

    modelo_analise = st.selectbox(
        "Escolha um modelo para analisar",
        options=list(resultados.keys()),
        index=list(resultados.keys()).index(melhor_modelo),
        help="Você pode comparar visualmente o desempenho de cada modelo."
    )

    resultado = resultados[modelo_analise]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R²", f"{resultado['r2']:.4f}")

    with col2:
        st.metric("RMSE", f"{resultado['rmse']:.2f}")

    with col3:
        st.metric("MAE", f"{resultado['mae']:.2f}")

    with col4:
        st.metric("MSE", f"{resultado['mse']:.2f}")

    if modo_iniciante:
        with st.expander("Como interpretar essas métricas?"):
            st.markdown(
                """
                - **R²:** indica quanto da variação do preço o modelo conseguiu explicar. Quanto maior, melhor.
                - **RMSE:** erro médio em escala próxima ao preço. Quanto menor, melhor.
                - **MAE:** erro médio absoluto. É mais fácil de interpretar que o MSE.
                - **MSE:** erro quadrático médio. Penaliza erros grandes com mais força.
                """
            )

    col1, col2 = st.columns(2)

    with col1:
        fig_pred = plotar_predicoes(
            y_test,
            resultado["predicoes"],
            modelo_analise
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with col2:
        fig_scatter = plotar_scatter_real_predito(
            y_test,
            resultado["predicoes"],
            modelo_analise
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    if modelo_analise != "Polinomial — grau 2":
        st.subheader("Coeficientes do modelo")

        coef_df = pd.DataFrame({
            "Variável": features_selecionadas_labels,
            "Coluna técnica": features_disponiveis,
            "Coeficiente": resultado["coeficientes"]
        })

        coef_df["Importância absoluta"] = np.abs(coef_df["Coeficiente"])
        coef_df = coef_df.sort_values("Importância absoluta", ascending=False)

        st.dataframe(
            coef_df.style.format(
                {
                    "Coeficiente": "{:.4f}",
                    "Importância absoluta": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        fig_coef = px.bar(
            coef_df,
            x="Coeficiente",
            y="Variável",
            orientation="h",
            title="Peso das variáveis no modelo",
            height=450
        )

        fig_coef.add_vline(x=0)
        st.plotly_chart(fig_coef, use_container_width=True)

    else:
        st.info(
            "O modelo polinomial cria combinações entre variáveis. "
            "Por isso, os coeficientes são menos diretos de interpretar nesta interface."
        )


# ============================================================
# TAB 3 — DIAGNÓSTICO
# ============================================================

with tab_diagnostico:
    st.header("Diagnóstico dos erros")

    modelo_diag = st.selectbox(
        "Modelo para diagnóstico",
        options=list(resultados.keys()),
        index=list(resultados.keys()).index(melhor_modelo),
        key="modelo_diag"
    )

    resultado_diag = resultados[modelo_diag]

    fig_residuos, residuos = plotar_residuos(
        y_test,
        resultado_diag["predicoes"]
    )

    st.plotly_chart(fig_residuos, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Média dos resíduos", f"{np.mean(residuos):.4f}")

    with col2:
        st.metric("Desvio dos resíduos", f"{np.std(residuos):.4f}")

    with col3:
        st.metric("Maior erro absoluto", f"{np.max(np.abs(residuos)):.2f}")

    with st.expander("Histograma e Q-Q plot dos resíduos"):
        hist_fig = px.histogram(
            x=residuos,
            nbins=30,
            title="Distribuição dos resíduos",
            labels={"x": "Resíduo", "y": "Frequência"},
            height=400
        )

        hist_fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(hist_fig, use_container_width=True)

        qq = stats.probplot(residuos, dist="norm")
        qq_df = pd.DataFrame({
            "Quantis teóricos": qq[0][0],
            "Quantis observados": qq[0][1]
        })

        qq_fig = px.scatter(
            qq_df,
            x="Quantis teóricos",
            y="Quantis observados",
            title="Q-Q Plot dos resíduos",
            height=400
        )

        st.plotly_chart(qq_fig, use_container_width=True)

    if modo_iniciante:
        st.info(
            "Resíduos são os erros do modelo. Em uma boa regressão, eles devem ficar "
            "distribuídos em torno de zero, sem padrão claro."
        )


# ============================================================
# TAB 4 — DADOS
# ============================================================

with tab_dados:
    st.header("Dados utilizados")

    st.markdown(
        """
        Esta seção ajuda a verificar transparência e consistência dos dados usados no treinamento.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Data inicial", dados.index.min().strftime("%d/%m/%Y"))

    with col2:
        st.metric("Data final", dados.index.max().strftime("%d/%m/%Y"))

    with col3:
        st.metric("Linhas após limpeza", len(dados))

    st.subheader("Amostra dos dados")

    st.dataframe(
        dados.tail(100),
        use_container_width=True
    )

    csv_dados = dados.to_csv().encode("utf-8")

    st.download_button(
        "Baixar dados tratados em CSV",
        data=csv_dados,
        file_name=f"dados_tratados_{ticker}.csv",
        mime="text/csv"
    )

    with st.expander("Estatísticas descritivas"):
        st.dataframe(
            dados.describe().T,
            use_container_width=True
        )


# ============================================================
# TAB 5 — AJUDA
# ============================================================

with tab_ajuda:
    st.header("Ajuda e documentação")

    st.subheader("O que este app faz?")

    st.markdown(
        """
        O app baixa dados históricos de ações pelo Yahoo Finance, cria variáveis derivadas
        e treina modelos de regressão para estimar o preço de fechamento no conjunto de teste.
        """
    )

    st.subheader("Modelos disponíveis")

    st.markdown(
        """
        - **Linear Simples:** modelo base de regressão linear.
        - **Ridge:** regressão linear com penalização L2, útil para reduzir instabilidade.
        - **Lasso:** regressão linear com penalização L1, podendo reduzir coeficientes pouco úteis.
        - **ElasticNet:** combinação de L1 e L2.
        - **Polinomial grau 2:** adiciona relações não lineares simples entre variáveis.
        """
    )

    st.subheader("Limitações importantes")

    st.warning(
        """
        Preços de ações são séries temporais complexas e influenciadas por eventos externos.
        Um bom R² histórico não garante capacidade preditiva futura.
        Use este app para estudo de regressão, análise exploratória e comparação de modelos,
        não para tomada direta de decisão financeira.
        """
    )

    st.subheader("Como a UX foi melhorada com Nielsen?")

    st.markdown(
        """
        - **Status visível:** o app informa cada etapa da execução.
        - **Prevenção de erros:** valida variáveis, dados mínimos e falhas de carregamento.
        - **Reconhecimento:** mostra nomes amigáveis das ações e explicações das variáveis.
        - **Controle do usuário:** análise só roda após clique no botão principal.
        - **Ajuda contextual:** sliders, seletores e métricas possuem explicações.
        - **Design minimalista:** informações técnicas ficam em abas e expanders.
        - **Recuperação de erro:** mensagens explicam o que aconteceu e o que fazer.
        """
    )
