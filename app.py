import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ==========================
# CONFIGURAÇÃO GERAL
# ==========================

st.set_page_config(
    page_title="Simulador de Risco - Pareto Truncada",
    layout="centered"
)

DAYS_PER_YEAR = 365  # Base anual fixa

st.title("Simulação Monte Carlo - Riscos Operacionais")
st.write("Preencha os parâmetros abaixo e clique em **Rodar Simulação**.")
st.caption("A simulação considera automaticamente 365 dias por ano.")

# ==========================
# ENTRADAS DO USUÁRIO
# ==========================

attacks_per_day = st.number_input(
    "Ataques por dia",
    min_value=1,
    value=100
)

successful_attacks_year = st.number_input(
    "Quantidade de ataques com sucesso no ano",
    min_value=0,
    value=60,
    step=1,
    help="Total esperado de ataques bem-sucedidos por ano"
)

xm = st.number_input(
    "Perda mínima - xm (R$)",
    min_value=0.0,
    value=100_000.0,
    step=10_000.0
)

xaverage = st.number_input(
    "Perda média (R$)",
    min_value=xm + 1,
    value=1_000_000.0,
    step=50_000.0
)

x_max = st.number_input(
    "Perda máxima (R$)",
    min_value=xaverage,
    value=300_000_000.0,
    step=1_000_000.0
)

T = st.number_input(
    "Número de simulações Monte Carlo",
    min_value=100,
    value=10_000,
    step=1_000
)

seed = st.number_input(
    "Semente aleatória",
    value=42
)

st.divider()

# ==========================
# BOTÃO DE EXECUÇÃO
# ==========================
rodar = st.button("Rodar Simulação")

# ==========================
# PROCESSAMENTO
# ==========================
if rodar:

    # ---------- frequência ----------
    n_attacks = attacks_per_day * DAYS_PER_YEAR

    if successful_attacks_year > n_attacks:
        st.error(
            "Erro: A quantidade de ataques com sucesso no ano "
            "não pode ser maior que o número total de ataques."
        )
        st.stop()

    p_success = successful_attacks_year / n_attacks
    lambda_success = successful_attacks_year  # parâmetro da Poisson

    # Exibição dos parâmetros de frequência
    st.write(f"**Total de ataques no ano:** {n_attacks:,}")


    # ---------- severidade (Pareto Truncada) ----------
    alpha = xaverage / (xaverage - xm)

    # ---------- simulação ----------
    rng = np.random.default_rng(int(seed))
    N_success = rng.poisson(lam=lambda_success, size=int(T))

    losses = np.zeros(int(T))

    for i, n in enumerate(N_success):
        if n > 0:
            draws = xm * (1 + rng.pareto(alpha, size=n))
            draws = np.clip(draws, xm, x_max)
            losses[i] = draws.sum()

    # ---------- métricas ----------
    mean_loss = losses.mean()
    median_loss = np.median(losses)   # >>> MEDIANA ADICIONADA <<<

    var_95 = np.percentile(losses, 95)
    var_99 = np.percentile(losses, 99)
    var_995 = np.percentile(losses, 99.5)

    threshold = var_995
    es_995 = losses[losses >= threshold].mean()

    prob_large_year = (losses >= 260_000_000).mean()

    # ==========================
    # RESULTADOS
    # ==========================
    st.subheader("Resultados da Simulação")

    col1, col2 = st.columns(2)

    col1.metric("Média Anual", f"R$ {mean_loss:,.0f}")
    col1.metric("Mediana Anual", f"R$ {median_loss:,.0f}")
    col1.metric("VaR 95%", f"R$ {var_95:,.0f}")

    col2.metric("VaR 99%", f"R$ {var_99:,.0f}")
    col2.metric("VaR 99.5%", f"R$ {var_995:,.0f}")
    col2.metric("ES 99.5%", f"R$ {es_995:,.0f}")

    st.write(
        f"**Probabilidade anual de perda >= R$ 260 milhões:** "
        f"{prob_large_year:.4%}"
    )

    # ==========================
    # GRÁFICO
    # ==========================
    st.subheader("Distribuição das Perdas Anuais")

    fig, ax = plt.subplots()
    ax.hist(losses, bins=100)

    # Formatação do eixo X em R$ milhões
    formatter = mtick.FuncFormatter(lambda x, pos: f"R$ {x/1e6:,.0f} mi")
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel("Perda Anual (R$ milhões)")
    ax.set_ylabel("Frequência")

    st.pyplot(fig)

