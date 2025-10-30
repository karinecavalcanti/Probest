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

# padronizar colunas
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# limpeza
num_cols = ["download_throughput_bps", "upload_throughput_bps", "rtt_download_sec", "rtt_upload_sec", "packet_loss_percent"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=num_cols)

print("Dimensões do dataset:", df.shape)
print(df.head())

# Seleção de clientes 
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
plt.title(f"RTT Upload")
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
            
            legenda_mle = f"LogNormal MLE: µ={mu_log:.3f}, σ={sigma_log:.3f}"
            print(f"   θˆMLE (s, scale): ({s:.3f}, {scale:.3f})") 
            
        elif modelo == "Gamma":
            # --- MLE de Gamma (Throughput) ---
            series_pos = series[series > 0]
            if len(series_pos) == 0: continue
                
            k, loc, scale = gamma.fit(series_pos, floc=0)
            parametros_mle[cliente][variavel] = (k, loc, scale)
            
            x_min, x_max = series_pos.min(), series_pos.max()
            x = np.linspace(x_min, x_max, 200)
            pdf = gamma.pdf(x, k, loc=loc, scale=scale)
              
            legenda_mle = f"Gamma MLE: k={k:.3f}, θ̂={scale:.2e}" 
            print(f"   θˆMLE (k, θ̂): ({k:.3f}, {scale:.2e})") 

        elif is_beta:
            # --- MLE de Beta (Packet Loss) ---
            series_prop = series / 100.0
            series_prop = series_prop[(series_prop > 0) & (series_prop < 1)]
            if len(series_prop) == 0: continue
                
            alpha, beta_param, loc, scale = beta.fit(series_prop, floc=0, fscale=1)
            parametros_mle[cliente][variavel] = (alpha, beta_param, loc, scale)
            
            x_prop = np.linspace(0.001, 0.999, 200) 
            pdf_prop = beta.pdf(x_prop, alpha, beta_param)
            
            legenda_mle = f"Beta MLE: α={alpha:.3f}, β={beta_param:.3f}"
            print(f"   θˆMLE (α, β): ({alpha:.3f}, {beta_param:.3f})")

        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 
        
        # Define os rótulos de eixo para o gráfico atual 
        label_x = nome_eixo_x.get(variavel, variavel)
        label_y_hist = "Densidade de Probabilidade"
        label_y_qq = "Valores Ordenados"
        
        # --------------------------------------------------------
        # SUBPLOT 1: HISTOGRAMA + FUNÇÃO DENSIDADE (PDF)
        # --------------------------------------------------------
        ax_hist = axes[0]
        ax_hist.hist(series, bins=30, density=True, alpha=0.6, label="Dados reais (histograma)")

        if is_beta:
            # Plota a PDF da Beta (escala 0-100)
            ax_hist.plot(x_prop * 100, pdf_prop / 100.0, 'r-', lw=2, label=legenda_mle)
        else:
            # Plota a PDF de LogNormal ou Gamma
            ax_hist.plot(x, pdf, 'r-', lw=2, label=legenda_mle)

        ax_hist.set_title(f"Histograma + PDF ({modelo})", fontsize=12)
        ax_hist.set_xlabel(label_x, fontsize=10)
        ax_hist.set_ylabel(label_y_hist, fontsize=10)
        ax_hist.legend(loc='upper right', fontsize=9)
        ax_hist.grid(axis='y', linestyle='--', alpha=0.6)


        # --------------------------------------------------------
        # SUBPLOT 2: QQ PLOT - Dados vs Quantis Teóricos
        # --------------------------------------------------------
        ax_qq = axes[1]
        
        dist_map = {
            "LogNormal": lognorm,
            "Gamma": gamma,
            "Beta": beta
        }
        dist_obj = dist_map.get(modelo)
        
        sparams_plot = None
        series_plot = None
        
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

        # stats.probplot plota diretamente no eixo ax_qq
        stats.probplot(series_plot, dist=dist_obj, sparams=sparams_plot, plot=ax_qq)
        ax_qq.set_title(f"QQ Plot ({modelo})", fontsize=12)

        ax_qq.set_xlabel("Quantis Teóricos", fontsize=10)
        ax_qq.set_ylabel(label_y_qq, fontsize=10)
        
        # Título geral para a figura inteira
        fig.suptitle(f"Ajuste MLE - {variavel} - Cliente: {cliente}", fontsize=14, fontweight='bold')
        
        # Ajusta o layout para evitar sobreposição
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajuste para acomodar o suptitle

        plt.savefig(f"outputs/figuras/mle/ajuste_mle_{cliente}_{variavel}.png")
        plt.close(fig) # Fecha a figura para liberar memória

print("Gráficos MLE (Histograma+PDF e QQ Plot lado a lado) gerados e salvos em 'outputs/figuras/mle/'.")

# ============================================================
# FUNÇÃO AUXILIAR PARA ESTIMAR FORMA DA GAMMA
# ============================================================

def estimate_gamma_shape_k(series):
    """Estima o parâmetro de forma (k) do Gamma usando Method of Moments"""
    try:
        mean_val = series.mean()
        var_val = series.var(ddof=1)
        
        if var_val > 0:
            k_estimate = (mean_val ** 2) / var_val
            return max(k_estimate, 0.1)
        else:
            return 1.0
    except Exception:
        return 1.0

# ============================================================
# 5. INFERÊNCIA BAYESIANA (ANALÍTICA CONJUGADA)
# ============================================================
print("\n=== 5. INFERÊNCIA BAYESIANA (ANALÍTICA CONJUGADA) ===")
print("Modelos usados: LogNormal (Normal-Normal no log-space), Gamma (Gama-Gama), Beta (Beta-Binomial Aproximado).")

# Definir priors conjugadas REALISTAS
priors_conjugadas = {
    "LogNormal": {
        "tipo": "Normal-Normal",
        "prior_mu": "Normal", 
        "hiperparametros": {"mu0": -3.0, "tau0_2": 1.0},
    },
    "Gamma": {
        "tipo": "Gamma-Gamma",
        "prior_beta": "Gamma", 
        "hiperparametros": {"alpha0": 2.0, "beta0": 2e-8},
    },
    "Beta": {
        "tipo": "Beta-Binomial", 
        "prior_p": "Beta", 
        "hiperparametros": {"alpha0": 1, "beta0": 1}, 
    }
}

print("\n=== PRIORS CONJUGADAS DEFINIDAS ===")
for modelo, config in priors_conjugadas.items():
    print(f"{modelo}: {config['tipo']}")
    print(f" Hiperparâmetros: {config['hiperparametros']}")
    print()

# Requisitos: Divisão treino/teste (70/30)
train, test = train_test_split(df, test_size=0.3, random_state=42)
resultados_mle_comparacao = {}

# --- PARÂMETRO NECESSÁRIO PARA CORREÇÃO DO PACKET LOSS ---
# O modelo Beta-Binomial exige contagens inteiras. Assumimos um número razoável de pacotes por teste.
N_PACKETS_ASSUMIDOS = 1000 # Assumindo 1000 pacotes em cada medição de perda

for cliente in [cliente_alto, cliente_baixo]:
    print(f"\n--- CLIENTE: {cliente} ---")
    train_c = train[train["client"] == cliente]
    test_c = test[test["client"] == cliente]
    mle_params = parametros_mle.get(cliente, {})

    # ================================================================
    # Caso A: Packet Loss (Beta-Binomial) 
    # ================================================================
    loss_var = "packet_loss_percent"
    loss_train = train_c[loss_var].dropna() / 100.0 # converter para [0,1]
    
    # Prior (Uniforme = Beta(1,1))
    prior_config = priors_conjugadas["Beta"]
    a0_loss = prior_config["hiperparametros"]["alpha0"] # 1.0
    b0_loss = prior_config["hiperparametros"]["beta0"]  # 1.0

    # O modelo Beta-Binomial conjuga a probabilidade de sucesso p.
    # Assumimos N_PACKETS_ASSUMIDOS pacotes por teste.
    
    # Contagem de pacotes perdidos em cada teste
    contagem_perdas = np.round(loss_train * N_PACKETS_ASSUMIDOS)
    
    # Contagens agregadas totais
    S_sucessos = contagem_perdas.sum() # Total de pacotes perdidos (Sucessos no modelo Binomial)
    N_total_testes = len(loss_train) * N_PACKETS_ASSUMIDOS # Total de pacotes enviados
    F_falhas = N_total_testes - S_sucessos # Total de pacotes não perdidos (Falhas)

    an_loss = a0_loss + S_sucessos
    bn_loss = b0_loss + F_falhas

    # 1. Média Posterior E[p|x] (Média da Probabilidade de Perda)
    p_post_mean = an_loss / (an_loss + bn_loss)
    # 2. Variância Posterior Var[p|x]
    var_post_p = (an_loss * bn_loss) / ((an_loss + bn_loss)**2 * (an_loss + bn_loss + 1))
    
    # 3. Média Preditiva E[Y_novo|y] = E[p|x]
    E_pred_loss = p_post_mean 
    # 4. Variância Preditiva Var[Y_novo|y]
    var_pred_loss = var_post_p 
    
    # 5. Média Real (Teste)
    mean_test_loss = test_c[loss_var].mean() / 100.0 # Converter para proporção
    # 6. Variância Real (Teste)
    var_test_loss = test_c[loss_var].var(ddof=1) / (100.0**2) # Variância em proporção

    print(f"\n[A] Packet Loss:")
    print(f" Prior: {prior_config['tipo']} | Assumindo N_PACKETS={N_PACKETS_ASSUMIDOS}")
    print(f" Hiperparâmetros: {prior_config['hiperparametros']}")
    print(f" E[p|x]: {p_post_mean:.6f} | Var[p|x]: {var_post_p:.2e}")
    print(f" E Preditiva: {E_pred_loss:.6f} | Var Preditiva: {var_pred_loss:.2e}") # p_post_mean e var_post_p são para proporção [0,1]
    print(f" E Real (Teste): {mean_test_loss:.6f} | Var Real (Teste): {var_test_loss:.2e}") # Usado proporção [0,1] para consistência interna

    resultados_mle_comparacao[cliente, loss_var] = {
        "Parâmetro": "p (Prob. Média)",
        "E_Post": p_post_mean,
        "Var_Post": var_post_p,
        "E_Pred": E_pred_loss,
        "Var_Pred": var_pred_loss,
        "E_Teste": mean_test_loss,
        "Var_Teste": var_test_loss
    }

    # ================================================================
    # Caso B: RTT Download/Upload (LogNormal com Normal-Normal)
    # ================================================================
    for rtt_var in ["rtt_download_sec", "rtt_upload_sec"]:
        r_train = train_c[rtt_var].dropna()
        r_train_log = np.log(r_train[r_train > 0])

        n = len(r_train_log)
        if n == 0:
            continue
            
        rbar_log = r_train_log.mean()

        # Parâmetros fixados do MLE
        s_mle, _, scale_mle = mle_params.get(rtt_var, (1, 0, 1))
        sigma2_log = s_mle**2 
        
        if sigma2_log == 0 or sigma2_log < 1e-10:
             sigma2_log = r_train_log.var(ddof=1)
        if sigma2_log == 0: 
             continue

        prior_config = priors_conjugadas["LogNormal"]
        mu0 = prior_config["hiperparametros"]["mu0"]# -3.0
        tau0_2 = prior_config["hiperparametros"]["tau0_2"] # 1.0

        # Posterior (do parâmetro mu)
        tau_n2 = 1 / (1/tau0_2 + n/sigma2_log)
        mu_n = tau_n2 * (mu0/tau0_2 + n*rbar_log/sigma2_log)
        
        # 1. Média Posterior E[Y|x] (no espaço original)
        E_post_orig = np.exp(mu_n + 0.5 * tau_n2)
        # 2. Variância Posterior Var[Y|x] (no espaço original)
        Var_post_orig = (np.exp(tau_n2) - 1) * np.exp(2*mu_n + tau_n2)
        
        # 3. Média Preditiva E[R_novo|r]
        var_pred_rtt_log = sigma2_log + tau_n2
        E_pred_rtt = np.exp(mu_n + 0.5 * var_pred_rtt_log)
        
        # 4. Variância Preditiva Var[R_novo|r]
        var_pred_rtt = (E_pred_rtt**2) * (np.exp(var_pred_rtt_log) - 1)

        # 5. Média Real (Teste)
        mean_test_rtt = test_c[rtt_var].mean()
        # 6. Variância Real (Teste)
        var_test_rtt = test_c[rtt_var].var(ddof=1)

        print(f"\n[B] {rtt_var}:")
        print(f" Prior: {prior_config['tipo']}")
        print(f" Hiperparâmetros: {prior_config['hiperparametros']}")
        print(f" E Posterior: {E_post_orig:.6f}s | Var Posterior: {Var_post_orig:.2e}")
        print(f" E Preditiva: {E_pred_rtt:.6f}s | Var Preditiva: {var_pred_rtt:.2e}")
        print(f" E Real (Teste): {mean_test_rtt:.6f}s | Var Real (Teste): {var_test_rtt:.2e}")

        resultados_mle_comparacao[cliente, rtt_var] = {
            "Parâmetro": "Média E[Y]",
            "E_Post": E_post_orig, 
            "Var_Post": Var_post_orig, 
            "E_Pred": E_pred_rtt,
            "Var_Pred": var_pred_rtt,
            "E_Teste": mean_test_rtt,
            "Var_Teste": var_test_rtt
        }

    # ================================================================
    # Caso C: Throughput (Gamma-Gamma) 
    # ================================================================
    for tp_var in ["download_throughput_bps", "upload_throughput_bps"]:
        y_train = train_c[tp_var].dropna()
        if len(y_train) == 0:
            continue
            
        y_sum = y_train.sum()
        
        # k fixo (estimado por MLE) 
        k_fixo = estimate_gamma_shape_k(y_train) 
        if k_fixo <= 0: 
            continue

        # Prior (Gamma)
        prior_config = priors_conjugadas["Gamma"]
        a0_gama = prior_config["hiperparametros"]["alpha0"] # 2.0
        b0_gama = prior_config["hiperparametros"]["beta0"]  # 2e-8

        # Posterior (em termos de taxa β)
        an_gama = a0_gama + len(y_train) * k_fixo
        bn_gama = b0_gama + y_sum

        # Média e Variância do parâmetro de taxa beta
        E_post_beta = an_gama / bn_gama
        Var_post_beta = an_gama / (bn_gama**2)
        
        # 3. Média Preditiva E[Y_novo|y]
        if an_gama > 1:
            E_pred_tp = k_fixo / E_post_beta 
        else:
            E_pred_tp = np.nan
            
        # 4. Variância Preditiva Var[Y_novo|y]
        if an_gama > 2:
            var_pred_tp = (
                (k_fixo * bn_gama * (an_gama + k_fixo - 1))
                / ((an_gama - 1) ** 2 * (an_gama - 2))
            )
        else:
            var_pred_tp = np.nan

        # 5. Média Real (Teste)
        mean_test_tp = test_c[tp_var].mean()
        # 6. Variância Real (Teste)
        var_test_tp = test_c[tp_var].var(ddof=1)

        print(f"\n[C] {tp_var} (k={k_fixo:.3f}):")
        print(f" Prior: {prior_config['tipo']}")
        print(f" Hiperparâmetros: {prior_config['hiperparametros']}")
        print(f" E[β|x]: {E_post_beta:.2e} | Var[β|x]: {Var_post_beta:.2e}")
        print(f" E Preditiva: {E_pred_tp:.2f} | Var Preditiva: {var_pred_tp:.2e}")
        print(f" E Real (Teste): {mean_test_tp:.2f} | Var Real (Teste): {var_test_tp:.2e}")

        resultados_mle_comparacao[cliente, tp_var] = {
            "Parâmetro": "Taxa Beta",
            "E_Post": E_post_beta, 
            "Var_Post": Var_post_beta, 
            "E_Pred": E_pred_tp,
            "Var_Pred": var_pred_tp,
            "E_Teste": mean_test_tp,
            "Var_Teste": var_test_tp
        }
        
# ============================================================
# 6. COMPARAÇÃO MLE vs BAYES 
# ============================================================
print("\n" + "="*80)
print("COMPARAÇÃO MLE vs BAYES - RESULTADOS ORGANIZADOS (CORREÇÃO: MÉDIA VS MÉDIA)")
print("="*80)

comparacao_mle_bayes_list = []

for cliente in [cliente_alto, cliente_baixo]:
    print(f"\n{' CLIENTE: ' + cliente + ' ':=^80}")
    
    # Cabeçalho da tabela
    print(f"\n{'Variável':<25} {'MLE (Média)':<15} {'Bayes (E_Pred)':<15} {'Diferença':<12} {'Efeito Prior':<15}")
    print("-" * 80)
    
    for variavel in vars_interesse:
        # Obter parâmetros MLE
        mle_params_cliente = parametros_mle.get(cliente, {})
        
        if variavel in mle_params_cliente:
            params_mle = mle_params_cliente[variavel]
            
            # 1. Calcular ESTIMATIVA PONTUAL MLE (MÉDIA DA DISTRIBUIÇÃO)
            if "rtt" in variavel:
                s, loc, scale = params_mle
                # Média E[Y] = exp(mu_log + sigma_log^2 / 2)
                theta_mle = lognorm.mean(s, loc, scale)
                unidade = "s"
            elif "throughput" in variavel:
                k, loc, scale = params_mle
                # Média E[Y] = k * scale
                theta_mle = gamma.mean(k, loc, scale)
                unidade = "bps"
            elif "loss" in variavel:
                alpha, beta_p, loc, scale = params_mle
                # Média E[p] = alpha / (alpha + beta) * 100
                theta_mle = beta.mean(alpha, beta_p, loc, scale) * 100
                unidade = "%"
            else:
                theta_mle = np.nan
                unidade = ""
            
            # Obter estimativa Bayesiana (E_Posterior da seção 5)
            bayes_data = resultados_mle_comparacao.get((cliente, variavel), {})
            
            # 2. Obter ESTIMATIVA PONTUAL BAYES (MÉDIA DA POSTERIOR DA VARIÁVEL)
            if "rtt" in variavel:
                # E_Post (E[Y|x]) no espaço original
                theta_bayes = bayes_data.get("E_Post", np.nan) 
            elif "throughput" in variavel:
                theta_bayes = bayes_data.get("E_Pred", np.nan) 
            elif "loss" in variavel:
                # E_Post (E[p|x]) em Proporção, converter para %
                theta_bayes = bayes_data.get("E_Post", np.nan) * 100
            else:
                theta_bayes = np.nan
            
            
            if not np.isnan(theta_mle) and not np.isnan(theta_bayes):
                diff_abs = abs(theta_bayes - theta_mle)
                diff_rel = (diff_abs / theta_mle) * 100 if theta_mle != 0 else np.nan
                
                # Classificar efeito da prior
                if diff_rel < 5:
                    efeito_prior = "Pequeno"
                elif diff_rel < 20:
                    efeito_prior = "Moderado"
                else:
                    efeito_prior = "Grande"
                
                # Formatar valores para exibição (Ajustado para maior precisão em RTT)
                if "throughput" in variavel:
                    theta_mle_str = f"{theta_mle:.2e}"
                    theta_bayes_str = f"{theta_bayes:.2e}"
                    diff_str = f"{diff_abs:.2e}"
                elif "loss" in variavel:
                    theta_mle_str = f"{theta_mle:.4f}"
                    theta_bayes_str = f"{theta_bayes:.4f}"
                    diff_str = f"{diff_abs:.4f}"
                else: # RTT
                    theta_mle_str = f"{theta_mle:.6f}"
                    theta_bayes_str = f"{theta_bayes:.6f}"
                    diff_str = f"{diff_abs:.6f}"
                
                # Imprimir linha da tabela
                print(f"{variavel:<25} {theta_mle_str:<15} {theta_bayes_str:<15} {diff_str:<12} {efeito_prior:<15}")
                comparacao_mle_bayes_list.append({
                    "Cliente": cliente,
                    "Variável": variavel,
                    "Unidade": unidade,
                    "Theta_MLE": theta_mle,
                    "Theta_Bayes": theta_bayes,
                    "Diferença_Absoluta": diff_abs,
                    "Diferença_Relativa_%": diff_rel,
                    "Efeito_Prior": efeito_prior
                })

print("\n" + "="*80)

# Salvar comparação detalhada em CSV
if comparacao_mle_bayes_list:
    df_comparacao_detalhada = pd.DataFrame(comparacao_mle_bayes_list)
    df_comparacao_detalhada.to_csv("outputs/tabelas/comparacao_mle_bayes_detalhada.csv", index=False)

    print(f"\nArquivo salvo: outputs/tabelas/comparacao_mle_bayes_detalhada.csv")