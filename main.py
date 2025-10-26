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
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, gamma, beta, lognorm, weibull_min


# ============================================================
# 2. CARREGAR E PRE-PROCESSAR O DATASET
# ============================================================
try:
    df = pd.read_csv("data/ndt_tests_corrigido.csv")
except FileNotFoundError:
    print("ERRO: Arquivo 'ndt_tests_corrigido.csv' não encontrado. Verifique o caminho.")
    exit()

# padronizar colunas (ajuste conforme seu CSV)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# limpeza
num_cols = ["download_throughput_bps", "upload_throughput_bps", "rtt_download_sec", "rtt_upload_sec", "packet_loss_percent"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=num_cols)

print("Dimensões do dataset:", df.shape)
print(df.head())

# Seleção de clientes (necessário para Seção 4 e 5)
client_stats = df.groupby("client")["download_throughput_bps"].agg(["mean", "std"])
cliente_alto = client_stats.nlargest(1, "mean").index[0]
cliente_baixo = client_stats.nsmallest(1, "mean").index[0]

vars_interesse = ["download_throughput_bps", "upload_throughput_bps", "rtt_download_sec", "rtt_upload_sec", "packet_loss_percent"]
nome_eixo_x = {
    "download_throughput_bps": "Throughput Download (bps)",
    "upload_throughput_bps": "Throughput Upload (bps)",
    "rtt_download_sec": "RTT Download (s)",
    "rtt_upload_sec": "RTT Upload (s)",
    "packet_loss_percent": "Perda de Pacotes (%)"
}

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


print("\n=== Calculando estatísticas por cliente e servidor ===")

estatisticas_desejadas = ["mean", "median", "var", "std", lambda x: x.quantile(0.9), lambda x: x.quantile(0.99)]

estat_cliente = df.groupby("client")[vars_interesse].agg(estatisticas_desejadas)
estat_servidor = df.groupby("server")[vars_interesse].agg(estatisticas_desejadas)

# Renomear colunas para ficar claro
novas_colunas = []
for col in estat_cliente.columns:
    if col[1] == '<lambda_0>':
        novas_colunas.append(f"{col[0]}_q0.9")
    elif col[1] == '<lambda_1>':
        novas_colunas.append(f"{col[0]}_q0.99")
    else:
        novas_colunas.append(f"{col[0]}_{col[1]}")

estat_cliente.columns = novas_colunas
estat_servidor.columns = novas_colunas

# Salvar
estat_cliente.to_csv("outputs/tabelas/estat_por_cliente.csv")
estat_servidor.to_csv("outputs/tabelas/estat_por_servidor.csv")


# Analisar diferenças entre clientes
client_stats = df.groupby("client")["download_throughput_bps"].agg(["mean", "std"])
print("\n=== Estatísticas de throughput por cliente ===")
print(client_stats.sort_values("mean", ascending=False))

# Selecionar dois clientes com comportamentos distintos
# Exemplo: maior vs menor throughput, ou maior vs menor variabilidade
cliente_alto = client_stats.nlargest(1, "mean").index[0]
cliente_baixo = client_stats.nsmallest(1, "mean").index[0]


# Filtrar dados dos clientes selecionados
df_clientes_sel = df[df["client"].isin([cliente_alto, cliente_baixo])]


# Histograma cliente 1 separado

for v in vars_interesse:

    plt.figure(figsize=(6,6))
    dados_a = df_clientes_sel[df_clientes_sel["client"] == "client01"][v].dropna()
    
    plt.hist(dados_a, bins=30, alpha=0.5, edgecolor='black', linewidth=1.2, label=cliente_alto)
    plt.xlabel(f"{v}")
    plt.ylabel("Frequência")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"outputs/figuras/hist_comp_client01 - {v}.png")

# Histograma cliente 10 separado

for v in vars_interesse:

    plt.figure(figsize=(6,6))
    dados_a = df_clientes_sel[df_clientes_sel["client"] == "client10"][v].dropna()
    
    plt.hist(dados_a, bins=30, alpha=0.5, edgecolor='black', linewidth=1.2, label=cliente_alto)
    plt.xlabel(f"{v}")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"outputs/figuras/hist_comp_client10 - {v}.png")

#=====HISTOGRAMAS=======

# download_throughput_bps
plt.figure(figsize=(6, 6))
todos_dados = df_clientes_sel[df_clientes_sel["client"].isin([cliente_alto, cliente_baixo])][vars_interesse[0]]
bins = np.histogram_bin_edges(todos_dados, bins=30) 
for cliente in [cliente_alto, cliente_baixo]:
    dados_cliente = df_clientes_sel[df_clientes_sel["client"] == cliente][vars_interesse[0]]
    plt.hist(
        dados_cliente,
        bins=bins,
        alpha=0.6,
        label=cliente,
        density=False,
        edgecolor='black',
        linewidth=1.2
    )

plt.xlabel("Dowload Throughput (bps)") 
plt.ylabel("Frequency")                      
plt.title(f"Dowload Throughput")
plt.legend()
plt.yscale("log")
plt.savefig(f"outputs/figuras/hist_Dowload_Throughput.png")
plt.close()

# upload_throughput_bps
plt.figure(figsize=(6, 6))
todos_dados = df_clientes_sel[df_clientes_sel["client"].isin([cliente_alto, cliente_baixo])][vars_interesse[1]]
bins = np.histogram_bin_edges(todos_dados, bins=30) 
for cliente in [cliente_alto, cliente_baixo]:
    dados_cliente = df_clientes_sel[df_clientes_sel["client"] == cliente][vars_interesse[1]]
    plt.hist(
        dados_cliente,
        bins=bins,
        alpha=0.6,
        label=cliente,
        density=False,
        edgecolor='black',
        linewidth=1.2
    )

plt.xlabel("Upload Throughput (bps)") 
plt.ylabel("Frequency")                      
plt.title(f"Upload Throughput")
plt.legend()
plt.yscale("log")
plt.savefig(f"outputs/figuras/hist_UploadThroughput.png")
plt.close()

# rtt_download_sec
plt.figure(figsize=(6, 6))
todos_dados = df_clientes_sel[df_clientes_sel["client"].isin([cliente_alto, cliente_baixo])][vars_interesse[2]]
bins = np.histogram_bin_edges(todos_dados, bins=30) 
for cliente in [cliente_alto, cliente_baixo]:
    dados_cliente = df_clientes_sel[df_clientes_sel["client"] == cliente][vars_interesse[2]]
    plt.hist(
        dados_cliente,
        bins=bins,
        alpha=0.6,
        label=cliente,
        density=False,
        edgecolor='black',
        linewidth=1.2
    )

plt.xlabel("RTT Download (s)") 
plt.ylabel("Frequency")                      
plt.title(f"RTT Download")
plt.legend()
plt.yscale("log")
plt.savefig(f"outputs/figuras/hist_RTT_Download.png")
plt.close()

# rtt_upload_sec
plt.figure(figsize=(6, 6))
todos_dados = df_clientes_sel[df_clientes_sel["client"].isin([cliente_alto, cliente_baixo])][vars_interesse[3]]
bins = np.histogram_bin_edges(todos_dados, bins=30) 
for cliente in [cliente_alto, cliente_baixo]:
    dados_cliente = df_clientes_sel[df_clientes_sel["client"] == cliente][vars_interesse[3]]
    plt.hist(
        dados_cliente,
        bins=bins,
        alpha=0.6,
        label=cliente,
        density=False,
        edgecolor='black',
        linewidth=1.2
    )

plt.xlabel("RTT Upload (s)") 
plt.ylabel("Frequency")                      
plt.title(f"RTT Download")
plt.legend()
plt.yscale("log")
plt.savefig(f"outputs/figuras/hist_RTT_Upload.png")
plt.close()

# packet_loss_percent
plt.figure(figsize=(6, 6))
todos_dados = df_clientes_sel[df_clientes_sel["client"].isin([cliente_alto, cliente_baixo])][vars_interesse[4]]
bins = np.histogram_bin_edges(todos_dados, bins=30) 
for cliente in [cliente_alto, cliente_baixo]:
    dados_cliente = df_clientes_sel[df_clientes_sel["client"] == cliente][vars_interesse[4]]
    plt.hist(
        dados_cliente,
        bins=bins,
        alpha=0.6,
        label=cliente,
        density=False,
        edgecolor='black',
        linewidth=1.2
    )

plt.xlabel("Packet Loss Percent (%)") 
plt.ylabel("Frequency")                      
plt.title(f"Packet Loss Percent")
plt.legend()
plt.yscale("log")
plt.savefig(f"outputs/figuras/hist_Packet_Loss_Percent.png")
plt.close()

#=====BOXPLOT=======

# Boxplot download_throughput_bps
plt.figure(figsize=(6, 5))

dados_box = [
    df_clientes_sel[df_clientes_sel["client"] == cliente_alto][vars_interesse[0]],
    df_clientes_sel[df_clientes_sel["client"] == cliente_baixo][vars_interesse[0]]
]

plt.boxplot(
    dados_box,
    labels=[cliente_alto, cliente_baixo],
    patch_artist=True,
    boxprops=dict(facecolor='plum', color='black'),
    medianprops=dict(color='orange', linewidth=1.5),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=4)
)

plt.ylabel("Dowload Throughput (bps)")
plt.title("Dowload Throughput")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/figuras/boxplot_Dowload_Throughput.png")

# Boxplot upload_throughput_bps
plt.figure(figsize=(6, 5))

dados_box = [
    df_clientes_sel[df_clientes_sel["client"] == cliente_alto][vars_interesse[1]],
    df_clientes_sel[df_clientes_sel["client"] == cliente_baixo][vars_interesse[1]]
]

plt.boxplot(
    dados_box,
    labels=[cliente_alto, cliente_baixo],
    patch_artist=True,
    boxprops=dict(facecolor='plum', color='black'),
    medianprops=dict(color='orange', linewidth=1.5),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=4)
)

plt.ylabel("Upload Throughput (bps)")
plt.title("Upload Throughput")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/figuras/boxplot_Upload_Throughput.png")

# Boxplot rtt_download_sec
plt.figure(figsize=(6, 5))

dados_box = [
    df_clientes_sel[df_clientes_sel["client"] == cliente_alto][vars_interesse[2]],
    df_clientes_sel[df_clientes_sel["client"] == cliente_baixo][vars_interesse[2]]
]

plt.boxplot(
    dados_box,
    labels=[cliente_alto, cliente_baixo],
    patch_artist=True,
    boxprops=dict(facecolor='plum', color='black'),
    medianprops=dict(color='orange', linewidth=1.5),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=4)
)

plt.ylabel("RTT Download (s)")
plt.title("RTT Download")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/figuras/boxplot_RTT_Download.png")

# Boxplot rtt_upload_sec
plt.figure(figsize=(6, 5))

dados_box = [
    df_clientes_sel[df_clientes_sel["client"] == cliente_alto][vars_interesse[3]],
    df_clientes_sel[df_clientes_sel["client"] == cliente_baixo][vars_interesse[3]]
]

plt.boxplot(
    dados_box,
    labels=[cliente_alto, cliente_baixo],
    patch_artist=True,
    boxprops=dict(facecolor='plum', color='black'),
    medianprops=dict(color='orange', linewidth=1.5),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=4)
)

plt.ylabel("RTT Upload (s)")
plt.title("RTT Upload")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/figuras/boxplot_RTT_Upload.png")

# Boxplot packet_loss_percent
plt.figure(figsize=(6, 5))

dados_box = [
    df_clientes_sel[df_clientes_sel["client"] == cliente_alto][vars_interesse[4]],
    df_clientes_sel[df_clientes_sel["client"] == cliente_baixo][vars_interesse[4]]
]

plt.boxplot(
    dados_box,
    labels=[cliente_alto, cliente_baixo],
    patch_artist=True,
    boxprops=dict(facecolor='plum', color='black'),
    medianprops=dict(color='orange', linewidth=1.5),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=4)
)

plt.ylabel("Packet Loss Percent (%)")
plt.title("Packet Loss Percent")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("outputs/figuras/boxplot_Packet_Loss_Percent.png")


# Scatter plots comparativos
plt.figure(figsize=(10, 6))
cores = {"cliente_alto": "blue", "cliente_baixo": "orange"}
for cliente, cor in zip([cliente_alto, cliente_baixo], ["blue", "orange"]):
    dados_cliente = df_clientes_sel[df_clientes_sel["client"] == cliente]
    plt.scatter(dados_cliente["rtt_download_sec"], 
                dados_cliente["download_throughput_bps"], 
                alpha=0.6, label=cliente, c=cor)
plt.xlabel("RTT Download (sec)")
plt.ylabel("Throughput Download (bps)")
plt.legend()
plt.title("RTT Download vs Throughput Download")
plt.savefig("outputs/figuras/scatter_clientes_sel.png")
plt.close()

# 

# ============================================================
# 3.1 DEFINIÇÃO DE MODELOS (CONJUGADOS PARA MLE)
# ============================================================

print(f"\n=== DEFINIÇÃO DOS MODELOS PARAMÉTRICOS (CONJUGADOS) ===")

# Modelos fixos: LogNormal (RTT), Gamma (Throughput), Beta (Loss).
modelos_fixos = {
    "rtt_download_sec": "LogNormal", 
    "rtt_upload_sec": "LogNormal", 
    "download_throughput_bps": "Gamma", 
    "upload_throughput_bps": "Gamma", 
    "packet_loss_percent": "Beta" 
}

modelos_clientes = {cliente_alto: modelos_fixos, cliente_baixo: modelos_fixos}

for cliente in [cliente_alto, cliente_baixo]:
    print(f"\nModelos DEFINITIVOS para {cliente}:")
    for variavel, modelo in modelos_clientes[cliente].items():
        print(f"  {variavel}: {modelo}")

# ============================================================
# 4. MÁXIMA VEROSSIMILHANÇA (MLE) E AVALIAÇÃO DE AJUSTE
# ============================================================
print("\n=== 4. MÁXIMA VEROSSIMILHANÇA (MLE) E AVALIAÇÃO DE AJUSTE ===")

parametros_mle = {}

# --- INÍCIO DO PROCESSO MLE ---
for cliente in [cliente_alto, cliente_baixo]:
    print(f"\n--- CLIENTE: {cliente} ---")
    dados_cliente = df[df["client"] == cliente]
    modelos = modelos_clientes[cliente]
    parametros_mle[cliente] = {}
    
    for variavel, modelo in modelos.items():
        series = dados_cliente[variavel].dropna()
        if len(series) == 0:
            continue
            
        print(f"\n{variavel} ({modelo}):")
        
        is_beta = modelo == "Beta"
        
        # Variáveis genéricas
        x, pdf, legenda_mle = None, None, ""

        # AJUSTE MLE
        if modelo == "LogNormal":
            # --- MLE de LogNormal (RTT) ---
            series_pos = series[series > 0]
            if len(series_pos) == 0: continue
            
            s, loc, scale = lognorm.fit(series_pos, floc=0)
            parametros_mle[cliente][variavel] = (s, loc, scale)
            
            mu_log = np.log(scale)       # Média no log-espaço (µ)
            sigma_log = s                # Desvio padrão no log-espaço (σ)
            
            x_min, x_max = series_pos.min(), series_pos.max()
            x = np.linspace(x_min, x_max, 200)
            pdf = lognorm.pdf(x, s, loc=loc, scale=scale)
            
            # >>> APLICANDO .3F AQUI <<<
            legenda_mle = f"LogNormal MLE: µ={mu_log:.3f}, σ={sigma_log:.3f}"
            print(f"  θˆMLE (s, scale): ({s:.3f}, {scale:.3f})") # Alterado para 3 casas
            
        elif modelo == "Gamma":
            # --- MLE de Gamma (Throughput) ---
            series_pos = series[series > 0]
            if len(series_pos) == 0: continue
                
            k, loc, scale = gamma.fit(series_pos, floc=0)
            parametros_mle[cliente][variavel] = (k, loc, scale)
            
            x_min, x_max = series_pos.min(), series_pos.max()
            x = np.linspace(x_min, x_max, 200)
            pdf = gamma.pdf(x, k, loc=loc, scale=scale)
             
            legenda_mle = f"Gamma MLE: k={k:.3f}, θ̂={scale:.2e}" # <-- FORÇADO .2e para 3.13e+08
            print(f"  θˆMLE (k, θ̂): ({k:.3f}, {scale:.2e})") # Também formatado no console

        elif is_beta:
            # --- MLE de Beta (Packet Loss) ---
            series_prop = series / 100.0
            series_prop = series_prop[(series_prop > 0) & (series_prop < 1)]
            if len(series_prop) == 0: continue
                
            alpha, beta_param, loc, scale = beta.fit(series_prop, floc=0, fscale=1)
            parametros_mle[cliente][variavel] = (alpha, beta_param, loc, scale)
            
            x_prop = np.linspace(0.001, 0.999, 200) 
            pdf_prop = beta.pdf(x_prop, alpha, beta_param)
            
            # >>> APLICANDO .3F AQUI <<<
            legenda_mle = f"Beta MLE: α={alpha:.3f}, β={beta_param:.3f}"
            print(f"  θˆMLE (α, β): ({alpha:.3f}, {beta_param:.3f})") # Alterado para 3 casas

        # ===================================
        # AVALIAÇÃO DO AJUSTE (GRÁFICOS)
        # ===================================
        
        # Define os rótulos de eixo para o gráfico atual (usa nome_eixo_x definido na Seção 3)
        label_x = nome_eixo_x.get(variavel, variavel)
        label_y_hist = "Densidade de Probabilidade"
        label_y_qq = "Valores Ordenados"
        
        # --------------------------------------------------------
        # 1️⃣ HISTOGRAMA + FUNÇÃO DENSIDADE (PDF)
        # --------------------------------------------------------
        plt.figure(figsize=(10, 6)) 
        plt.hist(series, bins=30, density=True, alpha=0.6, label="Dados reais (histograma)")

        if is_beta:
            # Plota a PDF da Beta (escala 0-100)
            plt.plot(x_prop * 100, pdf_prop / 100.0, 'r-', lw=2, label=legenda_mle)
        else:
            # Plota a PDF de LogNormal ou Gamma
            plt.plot(x, pdf, 'r-', lw=2, label=legenda_mle)

        plt.title(f"Histograma + PDF ({modelo}) - {variavel} - {cliente}")
        plt.xlabel(label_x)
        plt.ylabel(label_y_hist)
        # Ajuste de Legenda
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1, fontsize=9)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.savefig(f"outputs/figuras/mle/histpdf_{cliente}_{variavel}.png")
        plt.close()

        # --------------------------------------------------------
        # 2️⃣ QQ PLOT - Dados vs Quantis Teóricos
        # --------------------------------------------------------
        plt.figure(figsize=(6, 6))
        
        dist_map = {
            "LogNormal": lognorm,
            "Gamma": gamma,
            "Beta": beta
        }
        dist_obj = dist_map.get(modelo)
        
        if modelo == "LogNormal":
            s, _, _ = parametros_mle[cliente][variavel]
            sparams_plot = (s,)
            series_plot = series[series > 0]
        elif modelo == "Gamma":
            k, _, _ = parametros_mle[cliente][variavel]
            sparams_plot = (k,)
            series_plot = series[series > 0]
        elif is_beta:
            alpha, beta_param, _, _ = parametros_mle[cliente][variavel]
            sparams_plot = (alpha, beta_param)
            series_plot = series_prop 

        stats.probplot(series_plot, dist=dist_obj, sparams=sparams_plot, plot=plt)
        plt.title(f"QQ Plot - {variavel} ({modelo}) - {cliente}")

        plt.xlabel("Quantis Teóricos")
        plt.ylabel(label_y_qq)
        plt.tight_layout()
        plt.savefig(f"outputs/figuras/mle/qqplot_{cliente}_{variavel}.png")
        plt.close()

print("Gráficos MLE gerados e salvos em 'outputs/figuras/mle/'.")   
            


# ============================================================
# 5. INFERÊNCIA BAYESIANA (ANALÍTICA CONJUGADA) — CORRIGIDA
# ============================================================
print("\n=== 5. INFERÊNCIA BAYESIANA (ANALÍTICA CONJUGADA) ===")
print("Modelos usados: LogNormal (Normal-Normal no log-space), Gamma (Gama-Gama), Beta (Beta-Binomial).")

# Divisão treino/teste
train, test = train_test_split(df, test_size=0.3, random_state=42)
resultados_mle_comparacao = {}

# Função auxiliar
def estimate_gamma_shape_k(series):
    try:
        k, _, _ = stats.gamma.fit(series[series > 0], floc=0)
        return k
    except Exception:
        return 1.0

for cliente in [cliente_alto, cliente_baixo]:
    print(f"\n--- CLIENTE: {cliente} ---")
    train_c = train[train["client"] == cliente]
    test_c = test[test["client"] == cliente]
    mle_params = parametros_mle.get(cliente, {})

    # ================================================================
    # Caso A: Packet Loss (Beta-Binomial)
    # ================================================================
    loss_var = "packet_loss_percent"
    loss_train = train_c[loss_var].dropna() / 100.0  # converter para [0,1]

    # Priors levemente informativas: Beta(1,1)
    a0_loss, b0_loss = 1.0, 1.0

    # Somatório direto em proporções (sem N_ensaios artificial)
    an_loss = a0_loss + (loss_train.sum() * len(loss_train))
    bn_loss = b0_loss + (len(loss_train) - loss_train.sum() * len(loss_train))

    # Posterior esperada
    p_post_mean = an_loss / (an_loss + bn_loss)
    E_pred_loss = p_post_mean * 100.0
    mean_test_loss = test_c[loss_var].mean()

    print("\n[A] Packet Loss (Beta-Binomial Corrigido):")
    print(f"  PRIOR: Beta(α0={a0_loss:.1f}, β0={b0_loss:.1f}) (Uniforme)")
    print(f"  POSTERIOR: Beta(αn={an_loss:.2f}, βn={bn_loss:.2f})")
    print(f"  E[p|x]: {p_post_mean:.6f} → {E_pred_loss:.4f}% preditivo")
    print(f"  Média Real (Teste): {mean_test_loss:.4f}%")

    alpha_mle, beta_mle, _, _ = mle_params.get(loss_var, (1, 1, 0, 1))
    E_mle_loss = (alpha_mle / (alpha_mle + beta_mle)) * 100.0

    resultados_mle_comparacao[cliente, loss_var] = {
        "E_MLE": E_mle_loss,
        "E_Bayes": E_pred_loss,
        "Média Teste": mean_test_loss,
        "Parâmetro": "p (Prob. Média)"
    }

    # ================================================================
    # Caso B: RTT Download/Upload (LogNormal com Normal-Normal)
    # ================================================================
    for rtt_var in ["rtt_download_sec", "rtt_upload_sec"]:
        r_train = train_c[rtt_var].dropna()
        r_train_log = np.log(r_train[r_train > 0])

        n = len(r_train_log)
        rbar_log = r_train_log.mean()

        # Obter σ e μ no log-space a partir do MLE
        s_mle, _, scale_mle = mle_params.get(rtt_var, (1, 0, 1))
        mu_mle = np.log(scale_mle)
        sigma2_log = s_mle**2

        # Prior centrada no MLE e fracamente informativa
        mu0 = mu_mle
        tau0_2 = 1.0  # menor do que 1e6 → ainda "larga", mas numérica estável

        # Posterior
        tau_n2 = 1 / (1/tau0_2 + n/sigma2_log)
        mu_n = tau_n2 * (mu0/tau0_2 + n*rbar_log/sigma2_log)

        # Média preditiva no espaço original
        var_pred_rtt_log = sigma2_log + tau_n2
        E_pred_rtt = np.exp(mu_n + 0.5 * var_pred_rtt_log)

        mean_test_rtt = test_c[rtt_var].mean()
        var_test_rtt = test_c[rtt_var].var(ddof=1)

        E_mle_rtt = np.exp(mu_mle + 0.5 * sigma2_log)

        print(f"\n[B] {rtt_var} (LogNormal / Normal-Normal Corrigido):")
        print(f"  MLE: μ_log={mu_mle:.4f}, σ_log²={sigma2_log:.6f}")
        print(f"  PRIOR: Normal(µ0={mu0:.4f}, τ0²={tau0_2:.4f})")
        print(f"  POSTERIOR: Normal(µn={mu_n:.4f}, τn²={tau_n2:.6f})")
        print(f"  E[R_novo|r]: {E_pred_rtt:.4f}s | Média Teste: {mean_test_rtt:.4f}s")

        resultados_mle_comparacao[cliente, rtt_var] = {
            "E_MLE": E_mle_rtt,
            "E_Bayes": E_pred_rtt,
            "Média Teste": mean_test_rtt,
            "Parâmetro": "Média E[Y]"
        }

    # ================================================================
    # Caso C: Throughput (Gamma-Gamma)
    # ================================================================
    for tp_var in ["download_throughput_bps", "upload_throughput_bps"]:
        y_train = train_c[tp_var].dropna()
        y_sum = y_train.sum()

        k_fixo = estimate_gamma_shape_k(y_train)  # shape (k)
        a0_gama, b0_gama = 1.0, 0.001  # prior leve

        # Posterior (em termos de taxa β)
        an_gama = a0_gama + len(y_train) * k_fixo
        bn_gama = b0_gama + y_sum  # coerente com β = 1/θ

        # Média preditiva no espaço original
        E_pred_tp = k_fixo * (bn_gama / (an_gama - 1)) if an_gama > 1 else np.nan
        var_pred_tp = (
            (k_fixo * bn_gama * (an_gama + k_fixo - 1))
            / ((an_gama - 1) ** 2 * (an_gama - 2))
            if an_gama > 2 else np.nan
        )

        mean_test_tp = test_c[tp_var].mean()
        var_test_tp = test_c[tp_var].var(ddof=1)

        k_mle, _, scale_mle = mle_params.get(tp_var, (1, 0, 1))
        E_mle_tp = k_mle * scale_mle

        print(f"\n[C] {tp_var} (Gamma-Gamma Corrigido, k={k_fixo:.3f}):")
        print(f"  PRIOR: Gamma(a0={a0_gama:.1f}, b0={b0_gama:.3f})")
        print(f"  POSTERIOR: Gamma(an={an_gama:.2f}, bn={bn_gama:.2f})")
        print(f"  E[Y_novo|y]: {E_pred_tp:.2f} | Média Teste: {mean_test_tp:.2f}")
        print(f"  Var Preditiva: {var_pred_tp:.2e} | Var Real: {var_test_tp:.2e}")

        resultados_mle_comparacao[cliente, tp_var] = {
            "E_MLE": E_mle_tp,
            "E_Bayes": E_pred_tp,
            "Média Teste": mean_test_tp,
            "Parâmetro": "Média E[Y]"
        }


# ============================================================
# 6. COMPARAÇÃO MLE vs BAYES (E Geração da Tabela Final)
# ============================================================
# ... (O código desta seção gera a tabela final para o relatório)

# ============================================================
# 6. COMPARAÇÃO MLE vs BAYES
# ============================================================
print("\n=== 6. COMPARAÇÃO MLE vs BAYES ===")

comparacoes_list = []
for (cliente, var), data in resultados_mle_comparacao.items():
    comparacoes_list.append({
        "Cliente": cliente,
        "Variável": var,
        "Parâmetro": data["Parâmetro"],
        "E_MLE (do Modelo Ajustado)": data["E_MLE"],
        "E_Bayes (Preditivo)": data["E_Bayes"],
        "Média Teste (Real)": data["Média Teste"]
    })

df_comparacao = pd.DataFrame(comparacoes_list)
df_comparacao.to_csv("outputs/tabelas/comparacao_mle_bayes.csv", index=False)
print("\nDataFrame de Comparação (MLE vs Bayes vs Teste):")
print(df_comparacao)

print("\nConcluído. Gráficos e tabelas salvos em 'outputs/'.")