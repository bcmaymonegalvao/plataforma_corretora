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
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Regressão Linear em Ações",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# TEMA VISUAL — COLORMAP PLASMA
# ============================================================

PLASMA = {
    "deep": "#0d0887",
    "purple": "#5b02a3",
    "magenta": "#9a179b",
    "pink": "#cb4679",
    "coral": "#ed7953",
    "orange": "#fb9f3a",
    "yellow": "#fdca26",
    "lime": "#f0f921",
    "background": "#f8f7fc",
    "surface": "#ffffff",
    "text": "#1f1f29",
    "muted": "#6b6478",
    "border": "#ece7f5",
}

PLASMA_SEQUENCE = [
    PLASMA["deep"],
    PLASMA["purple"],
    PLASMA["magenta"],
    PLASMA["pink"],
    PLASMA["coral"],
    PLASMA["orange"],
    PLASMA["yellow"],
]


def aplicar_tema_plasma():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(240, 249, 33, 0.12), transparent 28%),
                    radial-gradient(circle at top right, rgba(203, 70, 121, 0.10), transparent 30%),
                    linear-gradient(180deg, #fbf9ff 0%, {PLASMA["background"]} 100%);
            }}

            .block-container {{
                padding-top: 1.6rem;
                padding-bottom: 2.5rem;
                max-width: 1280px;
            }}

            section[data-testid="stSidebar"] {{
                background:
                    linear-gradient(
                        180deg,
                        rgba(13, 8, 135, 0.98) 0%,
                        rgba(91, 2, 163, 0.97) 52%,
                        rgba(154, 23, 155, 0.95) 100%
                    );
            }}

            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] span {{
                color: #ffffff;
            }}

            section[data-testid="stSidebar"] div[data-baseweb="select"] span {{
                color: {PLASMA["text"]} !important;
            }}

            section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
                background-color: rgba(255, 255, 255, 0.96);
                border-radius: 12px;
            }}

            section[data-testid="stSidebar"] input {{
                color: {PLASMA["text"]} !important;
            }}

            .stButton > button {{
                border-radius: 14px;
                border: 1px solid rgba(253, 202, 38, 0.45);
                background: linear-gradient(
                    90deg,
                    {PLASMA["magenta"]},
                    {PLASMA["coral"]},
                    {PLASMA["orange"]}
                );
                color: white;
                font-weight: 800;
                padding: 0.65rem 1rem;
                box-shadow: 0 8px 22px rgba(203, 70, 121, 0.25);
                transition: all 0.2s ease-in-out;
            }}

            .stButton > button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 10px 26px rgba(203, 70, 121, 0.36);
                border-color: {PLASMA["yellow"]};
                color: white;
            }}

            button[data-baseweb="tab"] {{
                border-radius: 999px;
                padding: 0.5rem 1rem;
                font-weight: 700;
            }}

            button[data-baseweb="tab"][aria-selected="true"] {{
                background: linear-gradient(90deg, {PLASMA["deep"]}, {PLASMA["magenta"]});
                color: white;
            }}

            div[data-testid="stDataFrame"] {{
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid {PLASMA["border"]};
                box-shadow: 0 8px 24px rgba(13, 8, 135, 0.06);
            }}

            .hero-card {{
                padding: 1.7rem 1.8rem;
                border-radius: 26px;
                background:
                    linear-gradient(
                        120deg,
                        rgba(13, 8, 135, 0.97),
                        rgba(154, 23, 155, 0.92),
                        rgba(237, 121, 83, 0.90)
                    ),
                    radial-gradient(circle at right, rgba(240, 249, 33, 0.28), transparent 30%);
                color: white;
                box-shadow: 0 18px 42px rgba(13, 8, 135, 0.22);
                margin-bottom: 1.2rem;
            }}

            .hero-title {{
                font-size: 2.15rem;
                font-weight: 850;
                margin-bottom: 0.35rem;
                letter-spacing: -0.03em;
            }}

            .hero-subtitle {{
                font-size: 1.02rem;
                line-height: 1.55;
                max-width: 920px;
                opacity: 0.95;
                margin-bottom: 0;
            }}

            .metric-card {{
                min-height: 132px;
                padding: 1.05rem 1.1rem;
                border-radius: 22px;
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid {PLASMA["border"]};
                box-shadow: 0 10px 28px rgba(13, 8, 135, 0.08);
                position: relative;
                overflow: hidden;
                margin-bottom: 0.9rem;
            }}

            .metric-card::before {{
                content: "";
                position: absolute;
                inset: 0 auto 0 0;
                width: 7px;
                background: var(--accent);
            }}

            .metric-card::after {{
                content: "";
                position: absolute;
                width: 110px;
                height: 110px;
                right: -45px;
                top: -45px;
                border-radius: 50%;
                background: var(--accent-soft);
            }}

            .metric-icon {{
                font-size: 1.45rem;
                margin-bottom: 0.35rem;
                position: relative;
                z-index: 2;
            }}

            .metric-label {{
                color: {PLASMA["muted"]};
                font-size: 0.82rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.25rem;
                position: relative;
                z-index: 2;
            }}

            .metric-value {{
                color: {PLASMA["text"]};
                font-size: 1.45rem;
                font-weight: 850;
                line-height: 1.15;
                letter-spacing: -0.03em;
                position: relative;
                z-index: 2;
                word-break: break-word;
            }}

            .metric-help {{
                color: {PLASMA["muted"]};
                font-size: 0.83rem;
                margin-top: 0.45rem;
                line-height: 1.35;
                position: relative;
                z-index: 2;
            }}

            .section-card {{
                padding: 1.2rem 1.25rem;
                border-radius: 22px;
                background: rgba(255, 255, 255, 0.90);
                border: 1px solid {PLASMA["border"]};
                box-shadow: 0 10px 28px rgba(13, 8, 135, 0.07);
                margin-bottom: 1rem;
            }}

            .section-title {{
                display: flex;
                align-items: center;
                gap: 0.55rem;
                color: {PLASMA["text"]};
                font-size: 1.35rem;
                font-weight: 850;
                margin-bottom: 0.2rem;
                letter-spacing: -0.02em;
            }}

            .section-subtitle {{
                color: {PLASMA["muted"]};
                font-size: 0.95rem;
                margin-bottom: 0.5rem;
            }}

            .badge {{
                display: inline-block;
                padding: 0.28rem 0.65rem;
                border-radius: 999px;
                background: linear-gradient(90deg, {PLASMA["deep"]}, {PLASMA["magenta"]});
                color: white;
                font-weight: 800;
                font-size: 0.78rem;
                letter-spacing: 0.02em;
            }}

            .best-model-card {{
                padding: 1.2rem 1.25rem;
                border-radius: 22px;
                background:
                    linear-gradient(135deg, rgba(240, 249, 33, 0.18), rgba(251, 159, 58, 0.14)),
                    #ffffff;
                border: 1px solid rgba(253, 202, 38, 0.55);
                box-shadow: 0 10px 28px rgba(251, 159, 58, 0.14);
                margin: 0.8rem 0 1.1rem 0;
            }}

            .best-model-title {{
                color: {PLASMA["deep"]};
                font-size: 1rem;
                font-weight: 850;
                margin-bottom: 0.25rem;
            }}

            .best-model-value {{
                color: {PLASMA["magenta"]};
                font-size: 1.45rem;
                font-weight: 900;
                letter-spacing: -0.03em;
            }}

            .small-muted {{
                color: {PLASMA["muted"]};
                font-size: 0.9rem;
                line-height: 1.45;
            }}

            .info-box {{
                padding: 1rem 1.15rem;
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.92);
                border-left: 7px solid {PLASMA["orange"]};
                border-top: 1px solid {PLASMA["border"]};
                border-right: 1px solid {PLASMA["border"]};
                border-bottom: 1px solid {PLASMA["border"]};
                box-shadow: 0 8px 24px rgba(13, 8, 135, 0.06);
                color: {PLASMA["text"]};
                margin-bottom: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


def hero_plasma():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">📈 Análise de Regressão Linear em Ações</div>
            <p class="hero-subtitle">
                Compare modelos, acompanhe métricas de erro e visualize o comportamento histórico
                das ações em uma interface mais clara, interativa e visualmente consistente.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card(
    label,
    value,
    help_text="",
    icon="📌",
    color="#9a179b",
    soft_color="rgba(154, 23, 155, 0.12)"
):
    st.markdown(
        f"""
        <div class="metric-card" style="--accent: {color}; --accent-soft: {soft_color};">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_header(title, subtitle="", icon="✨"):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{icon} {title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def best_model_card(modelo, r2, interpretacao):
    st.markdown(
        f"""
        <div class="best-model-card">
            <div class="badge">🏆 Melhor modelo</div>
            <div class="best-model-title" style="margin-top: 0.7rem;">
                Modelo com maior R² no conjunto de teste
            </div>
            <div class="best-model-value">{modelo}</div>
            <div class="small-muted">
                R² = <strong>{r2:.4f}</strong>. {interpretacao}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def info_box(texto):
    st.markdown(
        f"""
        <div class="info-box">
            {texto}
        </div>
        """,
        unsafe_allow_html=True
    )


def estilizar_figura_plasma(fig, titulo=None, altura=None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
        font=dict(
            family="Arial, sans-serif",
            color=PLASMA["text"],
            size=13
        ),
        title=dict(
            text=titulo if titulo else fig.layout.title.text,
            font=dict(size=20, color=PLASMA["deep"]),
            x=0.02
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=70, b=40),
    )

    if altura:
        fig.update_layout(height=altura)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(13, 8, 135, 0.08)",
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(13, 8, 135, 0.08)",
        zeroline=False
    )

    return fig


# ============================================================
# DADOS AUXILIARES
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
# FUNÇÕES DE DADOS E MODELAGEM
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


# ============================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================

def plotar_historico(dados, acao, split_date):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["Close"],
            mode="lines",
            name="Preço de fechamento",
            line=dict(color=PLASMA["magenta"], width=3),
            fill="tozeroy",
            fillcolor="rgba(154, 23, 155, 0.08)"
        )
    )

    fig.add_shape(
        type="line",
        x0=split_date,
        x1=split_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            color=PLASMA["orange"],
            width=2,
            dash="dash"
        )
    )

    fig.add_annotation(
        x=split_date,
        y=1,
        xref="x",
        yref="paper",
        text="Início do teste",
        showarrow=False,
        yshift=18,
        font=dict(color=PLASMA["orange"], size=12)
    )

    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Preço de fechamento",
        hovermode="x unified"
    )

    return estilizar_figura_plasma(
        fig,
        titulo=f"Histórico de preços — {acao}",
        altura=500
    )


def plotar_predicoes(y_test, y_pred, modelo_nome):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=y_test,
            mode="lines",
            name="Valor real",
            line=dict(color=PLASMA["deep"], width=3)
        )
    )

    fig.add_trace(
        go.Scatter(
            y=y_pred,
            mode="lines",
            name="Predição",
            line=dict(color=PLASMA["orange"], width=3)
        )
    )

    fig.update_layout(
        xaxis_title="Observação no conjunto de teste",
        yaxis_title="Preço",
        hovermode="x unified"
    )

    return estilizar_figura_plasma(
        fig,
        titulo=f"Valores reais vs predições — {modelo_nome}",
        altura=450
    )


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

    fig.update_traces(
        marker=dict(
            color=PLASMA["magenta"],
            size=8,
            opacity=0.72,
            line=dict(width=0.8, color="white")
        )
    )

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Predição perfeita",
            line=dict(color=PLASMA["orange"], dash="dash", width=3)
        )
    )

    return estilizar_figura_plasma(fig, altura=450)


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

    fig.update_traces(
        marker=dict(
            color=PLASMA["pink"],
            size=8,
            opacity=0.72,
            line=dict(width=0.8, color="white")
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=PLASMA["orange"]
    )

    return estilizar_figura_plasma(fig, altura=450), residuos


def plotar_ranking_modelos(metricas_df):
    fig = px.bar(
        metricas_df.sort_values("R²", ascending=True),
        x="R²",
        y="Modelo",
        orientation="h",
        color="R²",
        color_continuous_scale="Plasma",
        text="R²",
        title="Comparação dos modelos por R²"
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="R² no conjunto de teste",
        yaxis_title="Modelo"
    )

    return estilizar_figura_plasma(fig, altura=430)


def plotar_coeficientes(coef_df):
    fig = px.bar(
        coef_df,
        x="Coeficiente",
        y="Variável",
        orientation="h",
        color="Coeficiente",
        color_continuous_scale="Plasma",
        title="Peso das variáveis no modelo",
        height=450
    )

    fig.add_vline(
        x=0,
        line_color=PLASMA["deep"],
        line_width=2
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title="Coeficiente"),
        xaxis_title="Coeficiente padronizado",
        yaxis_title="Variável"
    )

    return estilizar_figura_plasma(fig, altura=450)


# ============================================================
# APLICAÇÃO DO TEMA E CABEÇALHO
# ============================================================

aplicar_tema_plasma()
hero_plasma()

st.markdown(
    """
    <span class="small-muted">
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
    st.header("⚙️ Configurações")

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
        "🧹 Limpar cache dos dados",
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
    info_box(
        "Configure a análise na barra lateral e clique em "
        "<strong>Executar análise</strong> para iniciar."
    )
    st.stop()


# ============================================================
# VALIDAÇÕES
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
# PIPELINE PRINCIPAL
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
    metric_card(
        label="Observações",
        value=f"{len(dados):,}".replace(",", "."),
        help_text="Linhas usadas após limpeza e criação das variáveis.",
        icon="🗂️",
        color=PLASMA["deep"],
        soft_color="rgba(13, 8, 135, 0.12)"
    )

with col2:
    metric_card(
        label="Variáveis",
        value=str(len(features_disponiveis)),
        help_text="Quantidade de atributos usados pelos modelos.",
        icon="🧩",
        color=PLASMA["purple"],
        soft_color="rgba(91, 2, 163, 0.12)"
    )

with col3:
    metric_card(
        label="Melhor modelo",
        value=melhor_modelo,
        help_text="Modelo com maior R² no conjunto de teste.",
        icon="🏆",
        color=PLASMA["magenta"],
        soft_color="rgba(154, 23, 155, 0.13)"
    )

with col4:
    metric_card(
        label="Melhor R²",
        value=f"{metricas_df.iloc[0]['R²']:.4f}",
        help_text="Quanto maior, melhor o ajuste no teste.",
        icon="📊",
        color=PLASMA["orange"],
        soft_color="rgba(251, 159, 58, 0.15)"
    )

best_model_card(
    melhor_modelo,
    metricas_df.iloc[0]["R²"],
    interpretar_r2(metricas_df.iloc[0]["R²"])
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
# ABA 1 — RESUMO
# ============================================================

with tab_resumo:
    section_header(
        title="Resumo da análise",
        subtitle="Visão geral dos dados, configuração utilizada e comparação inicial dos modelos.",
        icon="📊"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        split_date = dados.index[len(X_train)]
        fig_historico = plotar_historico(dados, ticker, split_date)
        st.plotly_chart(fig_historico, use_container_width=True)

    with col2:
        section_header(
            title="Configuração usada",
            subtitle="Parâmetros escolhidos para esta execução.",
            icon="⚙️"
        )

        metric_card(
            label="Ação",
            value=ticker,
            help_text=ticker_label,
            icon="🏢",
            color=PLASMA["deep"],
            soft_color="rgba(13, 8, 135, 0.12)"
        )

        metric_card(
            label="Período",
            value=f"{anos_historico} ano(s)",
            help_text=f"Treino: {100 - test_size_percent}% | Teste: {test_size_percent}%",
            icon="🕒",
            color=PLASMA["pink"],
            soft_color="rgba(203, 70, 121, 0.13)"
        )

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

    fig_ranking = plotar_ranking_modelos(metricas_df)
    st.plotly_chart(fig_ranking, use_container_width=True)

    csv_metricas = metricas_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Baixar métricas em CSV",
        data=csv_metricas,
        file_name=f"metricas_{ticker}.csv",
        mime="text/csv"
    )


# ============================================================
# ABA 2 — MODELOS
# ============================================================

with tab_modelos:
    section_header(
        title="Análise dos modelos",
        subtitle="Compare métricas, predições e coeficientes do modelo selecionado.",
        icon="🔍"
    )

    modelo_analise = st.selectbox(
        "Escolha um modelo para analisar",
        options=list(resultados.keys()),
        index=list(resultados.keys()).index(melhor_modelo),
        help="Você pode comparar visualmente o desempenho de cada modelo."
    )

    resultado = resultados[modelo_analise]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            label="R²",
            value=f"{resultado['r2']:.4f}",
            help_text="Capacidade explicativa do modelo.",
            icon="📈",
            color=PLASMA["deep"],
            soft_color="rgba(13, 8, 135, 0.12)"
        )

    with col2:
        metric_card(
            label="RMSE",
            value=f"{resultado['rmse']:.2f}",
            help_text="Erro médio penalizando erros grandes.",
            icon="📉",
            color=PLASMA["magenta"],
            soft_color="rgba(154, 23, 155, 0.13)"
        )

    with col3:
        metric_card(
            label="MAE",
            value=f"{resultado['mae']:.2f}",
            help_text="Erro médio absoluto.",
            icon="🎯",
            color=PLASMA["coral"],
            soft_color="rgba(237, 121, 83, 0.14)"
        )

    with col4:
        metric_card(
            label="MSE",
            value=f"{resultado['mse']:.2f}",
            help_text="Erro quadrático médio.",
            icon="🧮",
            color=PLASMA["orange"],
            soft_color="rgba(251, 159, 58, 0.15)"
        )

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

        fig_coef = plotar_coeficientes(coef_df)
        st.plotly_chart(fig_coef, use_container_width=True)

    else:
        st.info(
            "O modelo polinomial cria combinações entre variáveis. "
            "Por isso, os coeficientes são menos diretos de interpretar nesta interface."
        )


# ============================================================
# ABA 3 — DIAGNÓSTICO
# ============================================================

with tab_diagnostico:
    section_header(
        title="Diagnóstico dos erros",
        subtitle="Avalie resíduos, dispersão dos erros e possíveis padrões não capturados pelo modelo.",
        icon="🧪"
    )

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
        metric_card(
            label="Média dos resíduos",
            value=f"{np.mean(residuos):.4f}",
            help_text="Idealmente, deve ficar próximo de zero.",
            icon="⚖️",
            color=PLASMA["purple"],
            soft_color="rgba(91, 2, 163, 0.12)"
        )

    with col2:
        metric_card(
            label="Desvio dos resíduos",
            value=f"{np.std(residuos):.4f}",
            help_text="Indica dispersão dos erros.",
            icon="🌊",
            color=PLASMA["pink"],
            soft_color="rgba(203, 70, 121, 0.13)"
        )

    with col3:
        metric_card(
            label="Maior erro absoluto",
            value=f"{np.max(np.abs(residuos)):.2f}",
            help_text="Maior diferença entre real e predito.",
            icon="🚨",
            color=PLASMA["orange"],
            soft_color="rgba(251, 159, 58, 0.15)"
        )

    with st.expander("Histograma e Q-Q plot dos resíduos"):
        hist_fig = px.histogram(
            x=residuos,
            nbins=30,
            title="Distribuição dos resíduos",
            labels={"x": "Resíduo", "y": "Frequência"},
            height=400
        )

        hist_fig.update_traces(
            marker=dict(
                color=PLASMA["magenta"],
                line=dict(color="white", width=0.6)
            )
        )

        hist_fig.add_vline(
            x=0,
            line_dash="dash",
            line_color=PLASMA["orange"]
        )

        hist_fig = estilizar_figura_plasma(hist_fig, altura=400)
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

        qq_fig.update_traces(
            marker=dict(
                color=PLASMA["coral"],
                size=8,
                opacity=0.72,
                line=dict(width=0.8, color="white")
            )
        )

        qq_fig = estilizar_figura_plasma(qq_fig, altura=400)
        st.plotly_chart(qq_fig, use_container_width=True)

    if modo_iniciante:
        st.info(
            "Resíduos são os erros do modelo. Em uma boa regressão, eles devem ficar "
            "distribuídos em torno de zero, sem padrão claro."
        )


# ============================================================
# ABA 4 — DADOS
# ============================================================

with tab_dados:
    section_header(
        title="Dados utilizados",
        subtitle="Verifique o período, a quantidade de registros e a amostra dos dados tratados.",
        icon="🗃️"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            label="Data inicial",
            value=dados.index.min().strftime("%d/%m/%Y"),
            help_text="Primeira data disponível após tratamento.",
            icon="📅",
            color=PLASMA["deep"],
            soft_color="rgba(13, 8, 135, 0.12)"
        )

    with col2:
        metric_card(
            label="Data final",
            value=dados.index.max().strftime("%d/%m/%Y"),
            help_text="Última data disponível no conjunto.",
            icon="📆",
            color=PLASMA["magenta"],
            soft_color="rgba(154, 23, 155, 0.13)"
        )

    with col3:
        metric_card(
            label="Linhas tratadas",
            value=str(len(dados)),
            help_text="Registros após remoção de valores ausentes.",
            icon="🧹",
            color=PLASMA["orange"],
            soft_color="rgba(251, 159, 58, 0.15)"
        )

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
# ABA 5 — AJUDA
# ============================================================

with tab_ajuda:
    section_header(
        title="Ajuda e documentação",
        subtitle="Entenda os modelos, as métricas e as limitações da análise.",
        icon="📚"
    )

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

    st.subheader("Como interpretar as métricas?")

    st.markdown(
        """
        - **R²:** mede a capacidade explicativa do modelo. Quanto maior, melhor.
        - **RMSE:** mede o erro médio com penalização maior para erros grandes.
        - **MAE:** mede o erro médio absoluto.
        - **MSE:** mede o erro quadrático médio.
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

    st.subheader("Melhorias de UX aplicadas")

    st.markdown(
        """
        - **Visibilidade do status:** o app informa carregamento, validação e treinamento.
        - **Prevenção de erros:** valida variáveis, quantidade mínima de dados e falhas de download.
        - **Reconhecimento:** usa nomes amigáveis das ações e explicações das variáveis.
        - **Controle do usuário:** a análise só roda após o botão principal.
        - **Ajuda contextual:** sliders, seletores, métricas e abas explicam sua função.
        - **Design visual:** cards, containers, gradientes e gráficos seguem o padrão Plasma.
        """
    )
