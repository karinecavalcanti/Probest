"""
Projeto COE241 / COS868 - Probest - Parte 1
Análise Estatística, MLE e Inferência Bayesiana em medições de rede.
"""

# ============================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÃO
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split


# ============================================================
# 2. CARREGAR E PRE-PROCESSAR O DATASET
# ============================================================
df = pd.read_csv("data/ndt_tests_corrigido.csv")

# padronizar colunas (ajuste conforme seu CSV)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]


# limpeza
num_cols = ["download_throughput_bps", "upload_throughput_bps", "rtt_download_sec", "rtt_upload_sec", "packet_loss_percent"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=num_cols)

print("Dimensões do dataset:", df.shape)
print(df.head())


# ============================================================
# 3. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ============================================================
def resumo_estatistico(series):
    return {
        "média": series.mean(),
        "mediana": series.median(),
        "variância": series.var(ddof=1),
        "desvio_std": series.std(ddof=1),
        "q0.9": series.quantile(0.9),
        "q0.99": series.quantile(0.99)
    }

vars_interesse = ["download_throughput_bps", "upload_throughput_bps", "rtt_download_sec", "rtt_upload_sec", "packet_loss_percent"]

# estatísticas gerais
estatisticas = {v: resumo_estatistico(df[v]) for v in vars_interesse}
print("\n=== Estatísticas gerais ===")

for v, s in estatisticas.items():
    print(v, s)


# estatisticas por cliente e servidor
estat_cliente = df.groupby("client")[vars_interesse].agg(["mean", "median", "var", "std"])
estat_servidor = df.groupby("server")[vars_interesse].agg(["mean", "median", "var","std"])
estat_cliente.to_csv("outputs/tabelas/estat_por_cliente.csv")
estat_servidor.to_csv("outputs/tabelas/estat_por_servidor.csv")



# ============================================================
# 3.1 SELEÇÃO DE CLIENTES/SERVIDORES DISTINTOS
# ============================================================

# Analisar diferenças entre clientes
client_stats = df.groupby("client")["download_throughput_bps"].agg(["mean", "std"])
print("\n=== Estatísticas de throughput por cliente ===")
print(client_stats.sort_values("mean", ascending=False))

# Selecionar dois clientes com comportamentos distintos
# Exemplo: maior vs menor throughput, ou maior vs menor variabilidade
cliente_alto = client_stats.nlargest(1, "mean").index[0]
cliente_baixo = client_stats.nsmallest(1, "mean").index[0]

print(f"\nClientes selecionados para análise:")
print(f"Cliente alto desempenho: {cliente_alto}")
print(f"Cliente baixo desempenho: {cliente_baixo}")

# ============================================================
# 3.2 GRÁFICOS COMPARATIVOS PARA CLIENTES SELECIONADOS
# ============================================================

# Filtrar dados dos clientes selecionados
df_clientes_sel = df[df["client"].isin([cliente_alto, cliente_baixo])]

# Histogramas comparativos
for v in vars_interesse:
    plt.figure(figsize=(10, 6))
    for cliente in [cliente_alto, cliente_baixo]:
        dados_cliente = df_clientes_sel[df_clientes_sel["client"] == cliente][v]
        plt.hist(dados_cliente, alpha=0.6, label=cliente, bins=20, density=True)
    plt.title(f"Histograma Comparativo - {v}")
    plt.legend()
    plt.savefig(f"outputs/figuras/hist_comp_{v}.png")
    plt.close()

# Boxplots comparativos
for v in vars_interesse:
    plt.figure(figsize=(8, 6))
    dados_box = [df_clientes_sel[df_clientes_sel["client"] == c][v] for c in [cliente_alto, cliente_baixo]]
    plt.boxplot(dados_box, labels=[cliente_alto, cliente_baixo])
    plt.title(f"Boxplot - {v}")
    plt.savefig(f"outputs/figuras/boxplot_{v}.png")
    plt.close()

# Scatter plots comparativos
plt.figure(figsize=(10, 6))
cores = {"cliente_alto": "blue", "cliente_baixo": "red"}
for cliente, cor in zip([cliente_alto, cliente_baixo], ["blue", "red"]):
    dados_cliente = df_clientes_sel[df_clientes_sel["client"] == cliente]
    plt.scatter(dados_cliente["rtt_download_sec"], 
                dados_cliente["download_throughput_bps"], 
                alpha=0.6, label=cliente, c=cor)
plt.xlabel("RTT Download (sec)")
plt.ylabel("Throughput Download (bps)")
plt.legend()
plt.title("Scatter RTT vs Throughput - Clientes Selecionados")
plt.savefig("outputs/figuras/scatter_clientes_sel.png")
plt.close()

# ============================================================
# 3.3 MODELAGEM PARA CLIENTES SELECIONADOS
# ============================================================

print(f"\n=== MODELAGEM PARA CLIENTE {cliente_alto} ===")
dados_cliente_alto = df[df["client"] == cliente_alto]

# RTT - Normal
rtt_alto = dados_cliente_alto["rtt_download_sec"].dropna()
mu_alto, sigma_alto = stats.norm.fit(rtt_alto)
print(f"RTT Cliente {cliente_alto}: mu={mu_alto:.4f}, sigma={sigma_alto:.4f}")

# Throughput - Gamma
th_alto = dados_cliente_alto["download_throughput_bps"].dropna()
th_alto = th_alto[th_alto > 0]
k_alto, _, scale_alto = stats.gamma.fit(th_alto, floc=0)
print(f"Throughput Cliente {cliente_alto}: shape={k_alto:.4f}, scale={scale_alto:.4f}")

print(f"\n=== MODELAGEM PARA CLIENTE {cliente_baixo} ===")
dados_cliente_baixo = df[df["client"] == cliente_baixo]

# RTT - Normal  
rtt_baixo = dados_cliente_baixo["rtt_download_sec"].dropna()
mu_baixo, sigma_baixo = stats.norm.fit(rtt_baixo)
print(f"RTT Cliente {cliente_baixo}: mu={mu_baixo:.4f}, sigma={sigma_baixo:.4f}")

# Throughput - Gamma
th_baixo = dados_cliente_baixo["download_throughput_bps"].dropna()
th_baixo = th_baixo[th_baixo > 0]
k_baixo, _, scale_baixo = stats.gamma.fit(th_baixo, floc=0)
print(f"Throughput Cliente {cliente_baixo}: shape={k_baixo:.4f}, scale={scale_baixo:.4f}")


# ============================================================
# 4. MÁXIMA VEROSSIMILHANÇA (MLE)
# ============================================================

# --- RTT (Normal) ---
rtt = df["rtt_download_sec"].dropna()
mu_mle, sigma_mle = stats.norm.fit(rtt)
print(f"\nRTT MLE: mu={mu_mle:.4f}, sigma={sigma_mle:.4f}")

# diagnóstico gráfico
x = np.linspace(rtt.min(), rtt.max(), 200)
plt.hist(rtt, bins=30, density=True, alpha=0.5)
plt.plot(x, stats.norm.pdf(x, mu_mle, sigma_mle), 'r')
plt.title("RTT - Ajuste Normal (MLE)")
plt.savefig("C:/Users/Julia/Desktop/compsoc/outputs/figuras/mle_rtt.png")
plt.close()

# --- Throughput (Gamma) ---
th = df["download_throughput_bps"].dropna()
th = th[th > 0]  # 🔹 remove valores negativos ou zero
k_mle, loc, scale_mle = stats.gamma.fit(th, floc=0)

print(f"Throughput Gamma MLE: shape={k_mle:.4f}, scale={scale_mle:.4f}, rate={1/scale_mle:.4f}")

# gráfico
x = np.linspace(th.min(), th.max(), 200)
plt.hist(th, bins=30, density=True, alpha=0.5)
plt.plot(x, stats.gamma.pdf(x, k_mle, scale=scale_mle), 'r')
plt.title("Throughput - Ajuste Gamma (MLE)")
plt.savefig("C:/Users/Julia/Desktop/compsoc/outputs/figuras/mle_throughput.png")
plt.close()

# --- Perda (proporção) ---
p_mle = df["packet_loss_percent"].mean()
print(f"Perda (MLE) p={p_mle:.6f}")

# ============================================================
# 5. INFERÊNCIA BAYESIANA (analítica)
# ============================================================

# Dividir treino e teste
train, test = train_test_split(df, test_size=0.3, random_state=42)

# ----- (1) Normal–Normal (RTT) -----
r_train = train["rtt_download_sec"].dropna()
n = len(r_train)
rbar = r_train.mean()
sigma2 = sigma_mle**2
mu0 = 0
tau0_2 = 1e6

tau_n2 = 1 / (1/tau0_2 + n/sigma2)
mu_n = tau_n2 * (mu0/tau0_2 + n*rbar/sigma2)
media_pred_rtt = mu_n
var_pred_rtt = sigma2 + tau_n2
print(f"\nPosterior RTT: mu_n={mu_n:.4f}, var_n={tau_n2:.6f}")

# comparação com dados de teste
mean_test_rtt = test["rtt_download_sec"].mean()
print(f"Média teste RTT={mean_test_rtt:.4f}, preditiva={media_pred_rtt:.4f}")

# ----- (2) Beta–Binomial (Perda) -----
nt = 1000
loss_train = train["packet_loss_percent"].dropna()
xt = (loss_train * nt).round().astype(int)
xtot = xt.sum()
ntot = nt * len(xt)
a0, b0 = 1.0, 1.0
an = a0 + xtot
bn = b0 + (ntot - xtot)
p_post_mean = an / (an + bn)
print(f"\nPosterior Beta-Binomial: a_n={an:.1f}, b_n={bn:.1f}, média posterior={p_post_mean:.6f}")

# ----- (3) Gama–Gama (Throughput) -----
y_train = train["download_throughput_bps"].dropna()
n = len(y_train)
k = k_mle
a0, b0 = 1.0, 1.0
an = a0 + n*k
bn = b0 + y_train.sum()
E_beta = an / bn
media_pred_throughput = k * bn / (an - 1)
print(f"\nPosterior Gama-Gama: an={an:.2f}, bn={bn:.2f}, E[β]={E_beta:.6f}")
print(f"Média preditiva throughput={media_pred_throughput:.4f}")

# ============================================================
# 6. COMPARAÇÃO MLE vs BAYES
# ============================================================

comparacao = pd.DataFrame({
    "Variável": ["RTT (µ)", "Perda (p)", "Throughput (β)"],
    "MLE": [mu_mle, p_mle, 1/scale_mle],
    "Bayes (Posterior Mean)": [mu_n, p_post_mean, E_beta]
})
comparacao.to_csv("C:/Users/Julia/Desktop/compsoc/outputs/tabelas/comparacao_mle_bayes.csv", index=False)
print("\n=== Comparação MLE vs Bayes ===")
print(comparacao)

# ============================================================
# 7. GRÁFICOS PREDITIVOS
# ============================================================
x = np.linspace(test["rtt_download_sec"].min(), test["rtt_download_sec"].max(), 200)
plt.hist(test["rtt_download_sec"], bins=30, density=True, alpha=0.5, label="dados teste")
plt.plot(x, stats.norm.pdf(x, mu_n, np.sqrt(var_pred_rtt)), 'r', label="predictiva bayes")
plt.legend()
plt.title("Posterior Predictive RTT")
plt.savefig("C:/Users/Julia/Desktop/compsoc/outputs/figuras/predictiva_rtt.png")
plt.close()

print("\nConcluído. Gráficos e tabelas salvos em 'outputs/'.")