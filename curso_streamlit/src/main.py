import streamlit as st
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
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
st.title("📈 Análise de Regressão Linear - Ações Globais/Stooq")
st.markdown("""
Este aplicativo realiza análise completa de regressão linear sobre dados de ações históricos, utilizando dados reais das principais bolsas do mundo via Stooq, sem necessidade de API Key!
Explore diferentes modelos, visualize métricas de performance e entenda os resultados através de gráficos explicativos.
""")

@st.cache_data
def carregar_tickers_acoes():
    # Exemplo com algumas ações globais e brasileiras (Stooq usa formatos diferentes)
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", # EUA
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "B3SA3.SA", # Brasil
        "SIE.DE", "BMW.DE", "AIR.PA", "MC.PA", "SAN.MC" # Europa: Alemanha(EUA.DE), França, Espanha
    ]

@st.cache_data(ttl=3600)
def carregar_dados_stooq(ticker):
    try:
        # Ajuste datas se necessário
        end = datetime.today()
        start = end - timedelta(days=1800)
        df = pdr.DataReader(ticker, data_source='stooq', start=start, end=end)
    except Exception as e:
        st.error(f"Erro ao baixar dados de {ticker}: {str(e)}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    # Ajusta índice se necessário (para Streamlit)
    df = df.sort_index()
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
    dados = carregar_dados_stooq(acao_selecionada)
    if dados.empty:
        st.error("❌ Não foi possível carregar dados para esta ação.")
    else:
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
            
            # Informação sobre fonte dos dados
            if os.path.exists(f"cache_cotacoes/{acao_selecionada}_cotacoes.csv"):
                st.info("📁 Dados carregados do cache local")
            else:
                st.warning("🔄 Dados demonstrativos sendo utilizados devido a limitações da API")
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Observações", len(dados))
                st.metric("Dados de Treino", len(X_train))
            
            with col2:
                st.metric("Dados de Teste", len(X_test))
                st.metric("Número de Features", X.shape[1])
            
            with col3:
                st.metric("Preço Mínimo", f"R$ {dados['Close'].min():.2f}")
                st.metric("Preço Máximo", f"R$ {dados['Close'].max():.2f}")
            
            with col4:
                st.metric("Preço Médio", f"R$ {dados['Close'].mean():.2f}")
                st.metric("Desvio Padrão", f"R$ {dados['Close'].std():.2f}")
            
            # Gráfico de preço histórico
            st.subheader("Histórico de Preços")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dados.index, dados['Close'], label='Preço de Fechamento', linewidth=2)
            
            # Destacar divisão treino/teste
            split_date = dados.index[len(X_train)]
            ax.axvline(x=split_date, color='red', linestyle='--', label='Divisão Treino/Teste')
            
            ax.set_xlabel('Data', fontsize=12)
            ax.set_ylabel('Preço (R$)', fontsize=12)
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
                with st.expander("📖 O que é R² Score?"):
                    st.markdown("""
                    **R² (Coeficiente de Determinação)** mede a proporção da variância na variável dependente 
                    que é previsível a partir das variáveis independentes.
                    
                    - **R² = 1**: Modelo perfeito
                    - **R² = 0**: Modelo não explica a variância
                    - **R² < 0**: Modelo pior que a média
                    
                    **Interpretação**: Um R² de 0.85 significa que 85% da variação nos preços é explicada pelo modelo.
                    """)
            
            with col2:
                st.metric("RMSE", f"{resultado['rmse']:.2f}")
                with st.expander("📖 O que é RMSE?"):
                    st.markdown("""
                    **RMSE (Root Mean Squared Error)** é a raiz quadrada da média dos erros ao quadrado.
                    
                    - Penaliza erros maiores mais fortemente
                    - Mesma unidade da variável target (R$)
                    - Quanto menor, melhor
                    
                    **Interpretação**: RMSE de 5.0 significa erro médio de R$ 5,00 nas previsões.
                    """)
            
            with col3:
                st.metric("MAE", f"{resultado['mae']:.2f}")
                with st.expander("📖 O que é MAE?"):
                    st.markdown("""
                    **MAE (Mean Absolute Error)** é a média dos valores absolutos dos erros.
                    
                    - Tratamento linear dos erros
                    - Mesma unidade da variável target (R$)
                    - Mais robusto a outliers que RMSE
                    
                    **Interpretação**: MAE de 3.5 significa erro médio absoluto de R$ 3,50.
                    """)
            
            with col4:
                st.metric("MSE", f"{resultado['mse']:.2f}")
                with st.expander("📖 O que é MSE?"):
                    st.markdown("""
                    **MSE (Mean Squared Error)** é a média dos erros ao quadrado.
                    
                    - Penaliza erros grandes
                    - Sempre positivo
                    - Sensível a outliers
                    
                    **Interpretação**: Quanto menor, melhor o ajuste do modelo.
                    """)
            
            # Gráfico de Predições vs Real
            st.subheader("Predições vs Valores Reais")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scatter plot
            ax1.scatter(y_test, resultado['predicoes'], alpha=0.6, edgecolors='k')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Predição Perfeita')
            ax1.set_xlabel('Valores Reais (R$)', fontsize=11)
            ax1.set_ylabel('Predições (R$)', fontsize=11)
            ax1.set_title(f'{modelo_analise} - Predições vs Real', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Linha do tempo
            ax2.plot(range(len(y_test)), y_test, label='Real', linewidth=2)
            ax2.plot(range(len(y_test)), resultado['predicoes'], 
                    label='Predição', linewidth=2, alpha=0.7)
            ax2.set_xlabel('Observação', fontsize=11)
            ax2.set_ylabel('Preço (R$)', fontsize=11)
            ax2.set_title(f'{modelo_analise} - Série Temporal', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Análise de Resíduos
            st.subheader("Análise de Resíduos")
            
            residuos = y_test - resultado['predicoes']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Histograma dos resíduos
            ax1.hist(residuos, bins=30, edgecolor='black', alpha=0.7)
            ax1.axvline(x=0, color='red', linestyle='--')
            ax1.set_xlabel('Resíduos (R$)', fontsize=11)
            ax1.set_ylabel('Frequência', fontsize=11)
            ax1.set_title('Distribuição dos Resíduos', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # 2. Q-Q Plot
            stats.probplot(residuos, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 3. Resíduos vs Valores Preditos
            ax3.scatter(resultado['predicoes'], residuos, alpha=0.6, edgecolors='k')
            ax3.axhline(y=0, color='red', linestyle='--')
            ax3.set_xlabel('Valores Preditos (R$)', fontsize=11)
            ax3.set_ylabel('Resíduos (R$)', fontsize=11)
            ax3.set_title('Resíduos vs Predições', fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            # 4. Resíduos ao longo do tempo
            ax4.plot(range(len(residuos)), residuos, alpha=0.7)
            ax4.axhline(y=0, color='red', linestyle='--')
            ax4.set_xlabel('Observação', fontsize=11)
            ax4.set_ylabel('Resíduos (R$)', fontsize=11)
            ax4.set_title('Resíduos ao Longo do Tempo', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            with st.expander("📖 Como interpretar a Análise de Resíduos?"):
                st.markdown("""
                A **Análise de Resíduos** verifica as suposições da regressão linear:
                
                1. **Distribuição dos Resíduos**: Deve ser aproximadamente normal (formato de sino)
                2. **Q-Q Plot**: Pontos devem seguir a linha diagonal (normalidade)
                3. **Resíduos vs Predições**: Devem estar distribuídos aleatoriamente em torno de zero (homocedasticidade)
                4. **Resíduos ao Longo do Tempo**: Não deve haver padrões (independência)
                
                **Problemas comuns:**
                - Padrão em formato de funil → heterocedasticidade
                - Padrões sistemáticos → modelo não linear adequado
                - Outliers → observações atípicas influenciando o modelo
                """)
            
            # Coeficientes do modelo
            if modelo_analise != 'Polinomial (grau 2)':
                st.subheader("Coeficientes do Modelo")
                
                coef_df = pd.DataFrame({
                    'Feature': features_disponiveis,
                    'Coeficiente': resultado['coeficientes']
                })
                coef_df['Importância Abs'] = np.abs(coef_df['Coeficiente'])
                coef_df = coef_df.sort_values('Importância Abs', ascending=False)
                
                st.dataframe(coef_df.style.format({'Coeficiente': '{:.4f}', 'Importância Abs': '{:.4f}'}))
                
                # Gráfico de importância
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['green' if x > 0 else 'red' for x in coef_df['Coeficiente']]
                ax.barh(coef_df['Feature'], coef_df['Coeficiente'], color=colors, alpha=0.7)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.set_xlabel('Valor do Coeficiente', fontsize=11)
                ax.set_title('Importância das Features', fontsize=12)
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
                
                with st.expander("📖 Como interpretar os Coeficientes?"):
                    st.markdown("""
                    Os **Coeficientes** indicam a relação entre cada feature e o preço:
                    
                    - **Positivo**: Aumento na feature → aumento no preço
                    - **Negativo**: Aumento na feature → diminuição no preço
                    - **Magnitude**: Quanto maior (em módulo), maior o impacto
                    
                    **Importante**: Os coeficientes são para features normalizadas, 
                    então representam o impacto de uma mudança de 1 desvio padrão na feature.
                    """)
        
        # TAB 3: VISUALIZAÇÕES
        with tab3:
            st.header("Visualizações Comparativas")
            
            # Comparação de todos os modelos
            st.subheader("Comparação Visual dos Modelos")
            
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.flatten()
            
            for idx, (nome, resultado) in enumerate(resultados.items()):
                ax = axes[idx]
                ax.scatter(y_test, resultado['predicoes'], alpha=0.6, edgecolors='k', s=30)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', lw=2)
                ax.set_xlabel('Real (R$)', fontsize=10)
                ax.set_ylabel('Predição (R$)', fontsize=10)
                ax.set_title(f'{nome}\nR²={resultado["r2"]:.4f}', fontsize=11)
                ax.grid(True, alpha=0.3)
            
            # Remover subplot extra
            fig.delaxes(axes[-1])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Gráfico de barras das métricas
            st.subheader("Comparação de Métricas")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            modelos_nomes = list(resultados.keys())
            
            # R²
            r2_scores = [resultados[m]['r2'] for m in modelos_nomes]
            axes[0, 0].bar(modelos_nomes, r2_scores, color='steelblue', alpha=0.7)
            axes[0, 0].set_ylabel('R² Score', fontsize=11)
            axes[0, 0].set_title('R² Score por Modelo', fontsize=12)
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # RMSE
            rmse_scores = [resultados[m]['rmse'] for m in modelos_nomes]
            axes[0, 1].bar(modelos_nomes, rmse_scores, color='coral', alpha=0.7)
            axes[0, 1].set_ylabel('RMSE (R$)', fontsize=11)
            axes[0, 1].set_title('RMSE por Modelo', fontsize=12)
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # MAE
            mae_scores = [resultados[m]['mae'] for m in modelos_nomes]
            axes[1, 0].bar(modelos_nomes, mae_scores, color='lightgreen', alpha=0.7)
            axes[1, 0].set_ylabel('MAE (R$)', fontsize=11)
            axes[1, 0].set_title('MAE por Modelo', fontsize=12)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # MSE
            mse_scores = [resultados[m]['mse'] for m in modelos_nomes]
            axes[1, 1].bar(modelos_nomes, mse_scores, color='mediumpurple', alpha=0.7)
            axes[1, 1].set_ylabel('MSE', fontsize=11)
            axes[1, 1].set_title('MSE por Modelo', fontsize=12)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Boxplot dos erros
            st.subheader("Distribuição dos Erros por Modelo")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            erros_modelos = [np.abs(y_test - resultados[m]['predicoes']) for m in modelos_nomes]
            
            bp = ax.boxplot(erros_modelos, labels=modelos_nomes, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.set_ylabel('Erro Absoluto (R$)', fontsize=11)
            ax.set_title('Distribuição dos Erros Absolutos por Modelo', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
        
        # TAB 4: DOCUMENTAÇÃO
        with tab4:
            st.header("📚 Documentação e Conceitos")
            
            st.markdown("""
            ## Modelos Implementados
            
            Este aplicativo implementa 5 diferentes técnicas de regressão linear:
            """)
            
            with st.expander("1️⃣ Regressão Linear Simples"):
                st.markdown("""
                ### Regressão Linear Simples
                
                **Descrição**: Método clássico que encontra a melhor linha reta para ajustar os dados.
                
                **Fórmula**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
                
                **Características**:
                - Método dos Mínimos Quadrados Ordinários (OLS)
                - Minimiza a soma dos quadrados dos resíduos
                - Sem regularização
                - Pode sofrer de overfitting com muitas features
                
                **Quando usar**: 
                - Relação linear clara entre variáveis
                - Número de features não é muito grande
                - Não há multicolinearidade severa
                """)
            
            with st.expander("2️⃣ Regressão Ridge (L2)"):
                st.markdown("""
                ### Regressão Ridge (L2 Regularization)
                
                **Descrição**: Adiciona penalização L2 (soma dos quadrados dos coeficientes) à função de custo.
                
                **Fórmula**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε + α∑βᵢ²
                
                **Características**:
                - Reduz coeficientes mas não os zera
                - Lida bem com multicolinearidade
                - Parâmetro α controla a regularização
                - Preserva todas as features
                
                **Quando usar**:
                - Muitas features correlacionadas
                - Suspeita de overfitting
                - Todas as features podem ser relevantes
                """)
            
            with st.expander("3️⃣ Regressão Lasso (L1)"):
                st.markdown("""
                ### Regressão Lasso (L1 Regularization)
                
                **Descrição**: Adiciona penalização L1 (soma dos valores absolutos dos coeficientes) à função de custo.
                
                **Fórmula**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε + α∑|βᵢ|
                
                **Características**:
                - Pode zerar coeficientes (seleção automática de features)
                - Produz modelos esparsos
                - Útil para seleção de features
                - Menos estável que Ridge
                
                **Quando usar**:
                - Muitas features irrelevantes
                - Necessidade de seleção automática de features
                - Interpretabilidade é importante
                """)
            
            with st.expander("4️⃣ ElasticNet (L1 + L2)"):
                st.markdown("""
                ### ElasticNet (L1 + L2 Regularization)
                
                **Descrição**: Combina penalizações L1 e L2, balanceando as vantagens de ambas.
                
                **Fórmula**: y = β₀ + β₁x₁ + ... + βₙxₙ + ε + α[ρ∑|βᵢ| + (1-ρ)∑βᵢ²]
                
                **Características**:
                - Combina seleção de features (L1) e estabilidade (L2)
                - Dois parâmetros: α (regularização) e ρ (balanceamento L1/L2)
                - Mais flexível que Ridge ou Lasso isoladamente
                - Funciona bem com grupos de features correlacionadas
                
                **Quando usar**:
                - Grupos de features correlacionadas
                - Necessidade de seleção de features e estabilidade
                - Quando nem Ridge nem Lasso funcionam bem sozinhos
                """)
            
            with st.expander("5️⃣ Regressão Polinomial"):
                st.markdown("""
                ### Regressão Polinomial
                
                **Descrição**: Estende a regressão linear criando features polinomiais das variáveis originais.
                
                **Exemplo (grau 2)**: y = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₂² + β₅x₁x₂ + ε
                
                **Características**:
                - Captura relações não-lineares
                - Aumenta significativamente o número de features
                - Risco alto de overfitting
                - Pode modelar curvas complexas
                
                **Quando usar**:
                - Relações não-lineares evidentes nos dados
                - Poucos dados de treinamento (evitar graus altos)
                - Combinado com regularização para evitar overfitting
                """)
            
            # ... resto da documentação igual ao código anterior ...
            
            st.markdown("""
            ## Sobre os Dados
            
            ### Fonte dos Dados
            - **Primária**: Yahoo Finance via yfinance
            - **Cache Local**: Dados salvos após primeiro download
            - **Fallback**: Dados sintéticos quando API não funciona
            
            ### Tratamento de Erros
            - Sistema de retry com delay progressivo
            - Headers customizados para evitar bloqueio
            - Cache local para reduzir chamadas à API
            - Dados demonstrativos como última opção
            
            ### Limitações
            - Dados podem estar desatualizados devido a limitações da API
            - Dados sintéticos são apenas para demonstração
            - Performance real pode variar com dados atualizados
            """)

# Rodapé
st.markdown("---")
st.markdown("""
**Desenvolvido por Bruno Galvão**  
**Análise de Regressão Linear para Ações da BOVESPA**  
*Este aplicativo é para fins educacionais e de análise. Não constitui recomendação de investimento.*

⚠️ **Nota sobre Dados**: Devido a limitações recentes da API do Yahoo Finance, o aplicativo pode utilizar dados em cache ou demonstrativos.
""")
