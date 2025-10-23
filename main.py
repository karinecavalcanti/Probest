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

# criar pastas de saída
os.makedirs("outputs/figuras", exist_ok=True)
os.makedirs("outputs/tabelas", exist_ok=True)

# ============================================================
# 2. CARREGAR E PRÉ-PROCESSAR O DATASET
# ============================================================
df = pd.read_csv("data/ndt_tests_corrigido.csv")

# padronizar colunas (ajuste conforme seu CSV)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# converter timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# converter perdas em proporção
if "packet_loss_percent" in df.columns:
    df["packet_loss"] = df["packet_loss_percent"] / 100.0

# converter throughput em Mbps
df["download_mbps"] = df["download_throughput_bps"] / 1e6
df["upload_mbps"] = df["upload_throughput_bps"] / 1e6

# limpeza
num_cols = ["download_mbps", "upload_mbps", "rtt_download_sec", "rtt_upload_sec", "packet_loss"]
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
        "n": series.count(),
        "média": series.mean(),
        "mediana": series.median(),
        "variância": series.var(ddof=1),
        "desvio_std": series.std(ddof=1),
        "q0.9": series.quantile(0.9),
        "q0.99": series.quantile(0.99),
        "min": series.min(),
        "max": series.max()
    }

vars_interesse = ["download_mbps", "upload_mbps", "rtt_download_sec", "rtt_upload_sec", "packet_loss"]

# estatísticas gerais
estatisticas = {v: resumo_estatistico(df[v]) for v in vars_interesse}
print("\n=== Estatísticas gerais ===")
for v, s in estatisticas.items():
    print(v, s)

# estatísticas por cliente e servidor
estat_cliente = df.groupby("client")[vars_interesse].agg(["mean", "median", "std", "count"])
estat_servidor = df.groupby("server")[vars_interesse].agg(["mean", "median", "std", "count"])
estat_cliente.to_csv("outputs/tabelas/estat_por_cliente.csv")
estat_servidor.to_csv("outputs/tabelas/estat_por_servidor.csv")

# gráficos principais
for v in vars_interesse:
    plt.figure()
    sns.histplot(df[v], kde=True)
    plt.title(f"Histograma - {v}")
    plt.savefig(f"outputs/figuras/hist_{v}.png")
    plt.close()

sns.scatterplot(x="rtt_download_sec", y="download_mbps", data=df)
plt.title("Scatter RTT vs Throughput (download)")
plt.savefig("outputs/figuras/scatter_rtt_throughput.png")
plt.close()

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
plt.savefig("outputs/figuras/mle_rtt.png")
plt.close()

# --- Throughput (Gamma) ---
th = df["download_mbps"].dropna()
th = th[th > 0]  # 🔹 remove valores negativos ou zero
k_mle, loc, scale_mle = stats.gamma.fit(th, floc=0)

print(f"Throughput Gamma MLE: shape={k_mle:.4f}, scale={scale_mle:.4f}, rate={1/scale_mle:.4f}")

# gráfico
x = np.linspace(th.min(), th.max(), 200)
plt.hist(th, bins=30, density=True, alpha=0.5)
plt.plot(x, stats.gamma.pdf(x, k_mle, scale=scale_mle), 'r')
plt.title("Throughput - Ajuste Gamma (MLE)")
plt.savefig("outputs/figuras/mle_throughput.png")
plt.close()

# --- Perda (proporção) ---
p_mle = df["packet_loss"].mean()
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
loss_train = train["packet_loss"].dropna()
xt = (loss_train * nt).round().astype(int)
xtot = xt.sum()
ntot = nt * len(xt)
a0, b0 = 1.0, 1.0
an = a0 + xtot
bn = b0 + (ntot - xtot)
p_post_mean = an / (an + bn)
print(f"\nPosterior Beta-Binomial: a_n={an:.1f}, b_n={bn:.1f}, média posterior={p_post_mean:.6f}")

# ----- (3) Gama–Gama (Throughput) -----
y_train = train["download_mbps"].dropna()
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
comparacao.to_csv("outputs/tabelas/comparacao_mle_bayes.csv", index=False)
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
plt.savefig("outputs/figuras/predictiva_rtt.png")
plt.close()

print("\nConcluído. Gráficos e tabelas salvos em 'outputs/'.")
