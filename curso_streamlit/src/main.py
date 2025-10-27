import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf  # Substituído pandas_datareader
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configuração da página
st.set_page_config(
    page_title="Análise de Regressão Linear - BOVESPA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e descrição
st.title("📈 Análise de Regressão Linear - Ações Globais")
st.markdown("""
Este aplicativo realiza análise completa de regressão linear sobre dados de ações históricos.
Explore diferentes modelos, visualize métricas de performance e entenda os resultados através de gráficos explicativos.
""")

@st.cache_data
def carregar_tickers_acoes():
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", # EUA
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "B3SA3.SA", # Brasil
        "SIE.DE", "BMW.DE", "AIR.PA", "MC.PA", "SAN.MC" # Europa
    ]

@st.cache_data(ttl=3600)
def carregar_dados_yfinance(ticker):
    try:
        end = datetime.today()
        start = end - timedelta(days=1800)
        
        # Usar yfinance diretamente
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if df.empty:
            st.warning(f"Nenhum dado retornado para {ticker}")
            return pd.DataFrame()
        
        # Ajustar nomes de colunas (yfinance usa maiúsculas)
        df.columns = df.columns.str.capitalize()
        
        # Feature engineering
        df['Dia'] = range(len(df))
        df['Retorno'] = df['Close'].pct_change()
        df['Volume_Norm'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
        df['Volatilidade'] = df['Retorno'].rolling(window=20).std()
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        df['Range'] = df['High'] - df['Low']
        df = df.dropna()
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao baixar dados de {ticker}: {str(e)}")
        return pd.DataFrame()

def treinar_modelos(X_train, y_train, X_test, y_test):
    modelos = {}
    resultados = {}
    
    # 1. Regressão Linear Simples
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    modelos['Linear Simples'] = lr
    resultados['Linear Simples'] = {
        'predicoes': y_pred_lr,
        'mse': mean_squared_error(y_test, y_pred_lr),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'mae': mean_absolute_error(y_test, y_pred_lr),
        'r2': r2_score(y_test, y_pred_lr),
        'coeficientes': lr.coef_,
        'intercepto': lr.intercept_
    }
    
    # 2. Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    modelos['Ridge (L2)'] = ridge
    resultados['Ridge (L2)'] = {
        'predicoes': y_pred_ridge,
        'mse': mean_squared_error(y_test, y_pred_ridge),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'mae': mean_absolute_error(y_test, y_pred_ridge),
        'r2': r2_score(y_test, y_pred_ridge),
        'coeficientes': ridge.coef_,
        'intercepto': ridge.intercept_
    }
    
    # 3. Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    modelos['Lasso (L1)'] = lasso
    resultados['Lasso (L1)'] = {
        'predicoes': y_pred_lasso,
        'mse': mean_squared_error(y_test, y_pred_lasso),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
        'mae': mean_absolute_error(y_test, y_pred_lasso),
        'r2': r2_score(y_test, y_pred_lasso),
        'coeficientes': lasso.coef_,
        'intercepto': lasso.intercept_
    }
    
    # 4. ElasticNet
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic.fit(X_train, y_train)
    y_pred_elastic = elastic.predict(X_test)
    modelos['ElasticNet'] = elastic
    resultados['ElasticNet'] = {
        'predicoes': y_pred_elastic,
        'mse': mean_squared_error(y_test, y_pred_elastic),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_elastic)),
        'mae': mean_absolute_error(y_test, y_pred_elastic),
        'r2': r2_score(y_test, y_pred_elastic),
        'coeficientes': elastic.coef_,
        'intercepto': elastic.intercept_
    }
    
    # 5. Polinomial
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train)
    y_pred_poly = lr_poly.predict(X_test_poly)
    modelos['Polinomial (grau 2)'] = (poly, lr_poly)
    resultados['Polinomial (grau 2)'] = {
        'predicoes': y_pred_poly,
        'mse': mean_squared_error(y_test, y_pred_poly),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_poly)),
        'mae': mean_absolute_error(y_test, y_pred_poly),
        'r2': r2_score(y_test, y_pred_poly),
        'coeficientes': lr_poly.coef_,
        'intercepto': lr_poly.intercept_
    }
    
    return modelos, resultados

# Carregar tickers globais
acoes = carregar_tickers_acoes()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    acao_selecionada = st.selectbox("Escolha uma ação", acoes, index=0)
    st.subheader("Divisão dos Dados")
    test_size = st.slider("Tamanho do conjunto de teste (%)", 10, 40, 20) / 100
    st.subheader("Features para o Modelo")
    usar_volume = st.checkbox("Usar Volume Normalizado", value=True)
    usar_volatilidade = st.checkbox("Usar Volatilidade", value=True)
    usar_mas = st.checkbox("Usar Médias Móveis", value=True)
    usar_range = st.checkbox("Usar Range (High-Low)", value=True)

if acao_selecionada:
    dados = carregar_dados_yfinance(acao_selecionada)
    
    if dados.empty:
        st.error("❌ Não foi possível carregar dados para esta ação.")
    else:
        # Preparar features
        features_disponiveis = ['Dia']
        if usar_volume:
            features_disponiveis.append('Volume_Norm')
        if usar_volatilidade:
            features_disponiveis.append('Volatilidade')
        if usar_mas:
            features_disponiveis.extend(['MA7', 'MA21'])
        if usar_range:
            features_disponiveis.append('Range')
        
        X = dados[features_disponiveis].values
        y = dados['Close'].values
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Normalizar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Treinar modelos
        modelos, resultados = treinar_modelos(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Criar abas
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visão Geral", 
            "🔍 Análise dos Modelos", 
            "📈 Visualizações", 
            "📚 Documentação"
        ])
        
        # TAB 1: VISÃO GERAL
        with tab1:
            st.header("Visão Geral dos Dados e Modelos")
            
            st.success("✅ Dados carregados via yfinance")
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Observações", len(dados))
                st.metric("Dados de Treino", len(X_train))
            
            with col2:
                st.metric("Dados de Teste", len(X_test))
                st.metric("Número de Features", X.shape[1])
            
            with col3:
                st.metric("Preço Mínimo", f"$ {dados['Close'].min():.2f}")
                st.metric("Preço Máximo", f"$ {dados['Close'].max():.2f}")
            
            with col4:
                st.metric("Preço Médio", f"$ {dados['Close'].mean():.2f}")
                st.metric("Desvio Padrão", f"$ {dados['Close'].std():.2f}")
            
            # Gráfico de preço histórico
            st.subheader("Histórico de Preços")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dados.index, dados['Close'], label='Preço de Fechamento', linewidth=2)
            
            # Destacar divisão treino/teste
            split_date = dados.index[len(X_train)]
            ax.axvline(x=split_date, color='red', linestyle='--', label='Divisão Treino/Teste')
            
            ax.set_xlabel('Data', fontsize=12)
            ax.set_ylabel('Preço ($)', fontsize=12)
            ax.set_title(f'Histórico de Preços - {acao_selecionada}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Comparação de métricas
            st.subheader("Comparação de Performance dos Modelos")
            
            metricas_df = pd.DataFrame({
                'Modelo': list(resultados.keys()),
                'R² Score': [resultados[m]['r2'] for m in resultados],
                'RMSE': [resultados[m]['rmse'] for m in resultados],
                'MAE': [resultados[m]['mae'] for m in resultados],
                'MSE': [resultados[m]['mse'] for m in resultados]
            })
            
            # Destacar melhor modelo
            melhor_modelo = metricas_df.loc[metricas_df['R² Score'].idxmax(), 'Modelo']
            st.success(f"🏆 Melhor Modelo (maior R²): **{melhor_modelo}**")
            
            st.dataframe(
                metricas_df.style.highlight_max(axis=0, subset=['R² Score'], color='lightgreen')
                                .highlight_min(axis=0, subset=['RMSE', 'MAE', 'MSE'], color='lightgreen')
                                .format({'R² Score': '{:.4f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'MSE': '{:.2f}'})
            )
        
        # TAB 2: ANÁLISE DOS MODELOS
        with tab2:
            st.header("Análise Detalhada dos Modelos")
            
            modelo_analise = st.selectbox("Selecione o modelo para análise", list(resultados.keys()))
            resultado = resultados[modelo_analise]
            
            # Métricas do modelo selecionado
            st.subheader(f"Métricas - {modelo_analise}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R² Score", f"{resultado['r2']:.4f}")
            
            with col2:
                st.metric("RMSE", f"{resultado['rmse']:.2f}")
            
            with col3:
                st.metric("MAE", f"{resultado['mae']:.2f}")
            
            with col4:
                st.metric("MSE", f"{resultado['mse']:.2f}")
            
            # Gráfico de Predições vs Real
            st.subheader("Predições vs Valores Reais")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scatter plot
            ax1.scatter(y_test, resultado['predicoes'], alpha=0.6, edgecolors='k')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Predição Perfeita')
            ax1.set_xlabel('Valores Reais ($)', fontsize=11)
            ax1.set_ylabel('Predições ($)', fontsize=11)
            ax1.set_title(f'{modelo_analise} - Predições vs Real', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Linha do tempo
            ax2.plot(range(len(y_test)), y_test, label='Real', linewidth=2)
            ax2.plot(range(len(y_test)), resultado['predicoes'], 
                    label='Predição', linewidth=2, alpha=0.7)
            ax2.set_xlabel('Observação', fontsize=11)
            ax2.set_ylabel('Preço ($)', fontsize=11)
            ax2.set_title(f'{modelo_analise} - Série Temporal', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Análise de Resíduos
            st.subheader("Análise de Resíduos")
            
            residuos = y_test - resultado['predicoes']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Histograma
            ax1.hist(residuos, bins=30, edgecolor='black', alpha=0.7)
            ax1.axvline(x=0, color='red', linestyle='--')
            ax1.set_xlabel('Resíduos ($)', fontsize=11)
            ax1.set_ylabel('Frequência', fontsize=11)
            ax1.set_title('Distribuição dos Resíduos', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Q-Q Plot
            stats.probplot(residuos, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Resíduos vs Predições
            ax3.scatter(resultado['predicoes'], residuos, alpha=0.6, edgecolors='k')
            ax3.axhline(y=0, color='red', linestyle='--')
            ax3.set_xlabel('Valores Preditos ($)', fontsize=11)
            ax3.set_ylabel('Resíduos ($)', fontsize=11)
            ax3.set_title('Resíduos vs Predições', fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            # Resíduos no tempo
            ax4.plot(range(len(residuos)), residuos, alpha=0.7)
            ax4.axhline(y=0, color='red', linestyle='--')
            ax4.set_xlabel('Observação', fontsize=11)
            ax4.set_ylabel('Resíduos ($)', fontsize=11)
            ax4.set_title('Resíduos ao Longo do Tempo', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Coeficientes
            if modelo_analise != 'Polinomial (grau 2)':
                st.subheader("Coeficientes do Modelo")
                
                coef_df = pd.DataFrame({
                    'Feature': features_disponiveis,
                    'Coeficiente': resultado['coeficientes']
                })
                coef_df['Importância Abs'] = np.abs(coef_df['Coeficiente'])
                coef_df = coef_df.sort_values('Importância Abs', ascending=False)
                
                st.dataframe(coef_df.style.format({'Coeficiente': '{:.4f}', 'Importância Abs': '{:.4f}'}))
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['green' if x > 0 else 'red' for x in coef_df['Coeficiente']]
                ax.barh(coef_df['Feature'], coef_df['Coeficiente'], color=colors, alpha=0.7)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.set_xlabel('Valor do Coeficiente', fontsize=11)
                ax.set_title('Importância das Features', fontsize=12)
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
        
        # TAB 3: VISUALIZAÇÕES
        with tab3:
            st.header("Visualizações Comparativas")
            
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.flatten()
            
            for idx, (nome, resultado) in enumerate(resultados.items()):
                ax = axes[idx]
                ax.scatter(y_test, resultado['predicoes'], alpha=0.6, edgecolors='k', s=30)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', lw=2)
                ax.set_xlabel('Real ($)', fontsize=10)
                ax.set_ylabel('Predição ($)', fontsize=10)
                ax.set_title(f'{nome}\nR²={resultado["r2"]:.4f}', fontsize=11)
                ax.grid(True, alpha=0.3)
            
            fig.delaxes(axes[-1])
            plt.tight_layout()
            st.pyplot(fig)
        
        # TAB 4: DOCUMENTAÇÃO
        with tab4:
            st.header("📚 Documentação")
            st.markdown("""
            ### Sobre o Aplicativo
            
            Este aplicativo utiliza **yfinance** para obter dados de ações em tempo real e implementa
            5 diferentes modelos de regressão linear para análise e previsão.
            
            ### Modelos Implementados
            - **Linear Simples**: Regressão OLS básica
            - **Ridge (L2)**: Regularização L2
            - **Lasso (L1)**: Regularização L1 com seleção de features
            - **ElasticNet**: Combinação de L1 e L2
            - **Polinomial**: Features polinomiais de grau 2
            
            ### Métricas
            - **R²**: Coeficiente de determinação (0-1, maior melhor)
            - **RMSE**: Raiz do erro quadrático médio
            - **MAE**: Erro absoluto médio
            - **MSE**: Erro quadrático médio
            """)

# Rodapé
st.markdown("---")
st.markdown("""
**Análise de Regressão Linear para Ações**  
*Este aplicativo é para fins educacionais. Não constitui recomendação de investimento.*
""")
