# %% [markdown]
# # Trabalho 02 — Deep Learning (PPGCA/Unisinos, 2026/2)
# **Detecção de *prompt injection* com um MLP sobre TF-IDF: do backpropagation à mão aos otimizadores**
#
# Aluno: Epitácio Neto · Professor: Felipe André Zeiser
#
# **Problema.** Aplicações com modelos de linguagem recebem texto não confiável (o *prompt* do usuário). Um *prompt injection*
# tenta fazer o modelo ignorar as instruções originais (p.ex. "ignore as instruções anteriores e..."). O objetivo é um detector
# **barato e rápido** que classifique cada texto como *seguro* (0) ou *injeção* (1) e funcione como primeiro filtro.
#
# **Escopo e relação com as aulas.** O trabalho aplica, em uma tarefa real, o que foi visto: texto → tokens → TF-IDF (Aula 6);
# rede densa (MLP) com ReLU/sigmoide e perda de entropia cruzada binária (Aulas 1 e 2); backpropagation à mão e verificação
# numérica (Aulas 2 e 3); comparação dos otimizadores SGD, Momentum, RMSProp e Adam (Aula 4); escolha de arquitetura justificada
# frente a um modelo pré-treinado (Aulas 3 e 5). Apenas **datasets públicos** são usados.
#
# **Como executar.** `python trabalho2_dl.py` (CPU, ~10 min) ou abrir o notebook. Os datasets são baixados do Hugging Face Hub
# (`deepset/prompt-injections`, `neuralchemy/Prompt-injection-dataset` config `core`, `xTRam1/safe-guard-prompt-injection`).
# Resultados em `resultados/` e figuras em `figuras/`. Um laboratório web interativo, que repete os experimentos no navegador,
# acompanha o trabalho (ver README do repositório).
#
# **Protocolo.** Treino/validação fixos (80/20 do treino do deepset, estratificado); **todas as escolhas** (taxa de aprendizado,
# tamanho da camada oculta, época de parada) usam só a validação; o teste do deepset (em-distribuição) e dois conjuntos
# fora-da-distribuição (neuralchemy, xTRam1) são apenas relatados. 5 sementes (0 a 4); média ± desvio-padrão.

# %%
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(AQUI))
import injecao as I
RAIZ = AQUI.parent if (AQUI.parent / "resultados").exists() else AQUI
RES, FIG = RAIZ / "resultados", RAIZ / "figuras"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
sys.path.insert(0, "/home/ubuntu/meridianseal/unisinos/tools/templates")
try:
    from plot_theme import aplicar, salvar, COR
    aplicar()
except Exception:
    import matplotlib; matplotlib.use("Agg")
    COR = {"azul": "#1D2369", "azul2": "#4A5AB8", "vermelho": "#C0392B", "cinza": "#666666", "claro": "#E9ECF3", "verde": "#2E7D5B"}
    def aplicar(): pass
    def salvar(fig, p): fig.tight_layout(); fig.savefig(p, dpi=150, bbox_inches="tight")
import matplotlib.pyplot as plt
import torch, sklearn
SEMENTES = [0, 1, 2, 3, 4]
EPOCAS, BATCH = 40, 32
INFO = {"python": sys.version.split()[0], "torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
        "epocas": EPOCAS, "batch": BATCH, "sementes": SEMENTES}
def salvar_json(obj, nome): (RES / nome).write_text(json.dumps(obj, ensure_ascii=False, indent=1))

# %% [markdown]
# ## 1. Dados
# O deepset (546 treino / 116 teste) é pequeno e **bilíngue** (inglês e alemão). Os conjuntos fora-da-distribuição servem para
# medir quanto o detector depende do vocabulário visto no treino.

# %%
D = I.carregar_dados()
(tr_t, tr_y), (va_t, va_y) = I.dividir_treino_val(*D["treino_total"])
te_t, te_y = D["teste"]
CONJ = {"deepset (teste)": D["teste"], "neuralchemy (OOD)": D["neuralchemy"], "xTRam1 (OOD)": D["xtram1"]}
tab = [{"conjunto": "treino", "n": len(tr_y), "injecao": int(tr_y.sum())}, {"conjunto": "validação", "n": len(va_y), "injecao": int(va_y.sum())}]
tab += [{"conjunto": k, "n": len(v[1]), "injecao": int(v[1].sum())} for k, v in CONJ.items()]
tab_dados = pd.DataFrame(tab); tab_dados["pct_injecao"] = (100 * tab_dados.injecao / tab_dados.n).round(1)
print(tab_dados.to_string(index=False)); tab_dados.to_csv(RES / "tab_dados.csv", index=False)
print("Exemplos:", [(t[:70], int(l)) for t, l in list(zip(tr_t, tr_y))[:3]])

# %% [markdown]
# ## 2. Texto → vetor (Aula 6)
# Tokenização por palavras em minúsculas, vocabulário dos 2.000 termos mais frequentes (documento-frequência ≥ 2) e
# **TF-IDF** com normalização L2, implementado à mão para que o laboratório em JavaScript produza os mesmos vetores.

# %%
V = I.Vetorizador(max_vocab=2000, min_df=2).ajustar(tr_t)
Xtr, Xva = V.transformar(tr_t), V.transformar(va_t)
XC = {k: V.transformar(v[0]) for k, v in CONJ.items()}
print("vocabulário:", len(V.vocab), "| tamanho do vetor:", Xtr.shape[1], "| 10 termos mais frequentes:", V.vocab[:10])
# verificação cruzada com o TfidfVectorizer do scikit-learn (mesma fórmula): diferença máxima deve ser ~0
from sklearn.feature_extraction.text import TfidfVectorizer
sk = TfidfVectorizer(tokenizer=I.tokenizar, lowercase=False, token_pattern=None, vocabulary=V.indice, norm="l2", smooth_idf=True)
sk.fit(tr_t)
dif_sk = float(np.abs(sk.transform(va_t).toarray() - Xva).max())
print("diferença máxima vs scikit-learn:", dif_sk); INFO["dif_tfidf_sklearn"] = dif_sk

# %% [markdown]
# ## 3. Backpropagation à mão (Aulas 2 e 3)
# Rede 2-camadas: `z1 = X·W1 + b1 → h = ReLU(z1) → z2 = h·W2 + b2 → p = σ(z2)`, perda BCE.
# Com BCE + sigmoide, ∂L/∂z2 = (p − y)/n; a regra da cadeia propaga de trás para frente. A verificação compara os gradientes
# escritos à mão com os do autograd do PyTorch (devem coincidir até a precisão numérica).

# %%
rng = np.random.RandomState(0)
H0 = 16
P = {"W1": rng.randn(Xtr.shape[1], H0) * np.sqrt(2 / Xtr.shape[1]), "b1": np.zeros(H0), "W2": rng.randn(H0, 1) * np.sqrt(2 / H0), "b2": np.zeros(1)}
xb, yb = Xtr[:64], tr_y[:64].astype(float)
g_mao, perda_mao = I.mlp_gradientes(P, xb, yb)
tp = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in P.items()}
z = torch.relu(torch.tensor(xb) @ tp["W1"] + tp["b1"]) @ tp["W2"] + tp["b2"]
perda_t = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.tensor(yb).view(-1, 1)); perda_t.backward()
dif_g = {k: float(np.abs(g_mao[k] - tp[k].grad.numpy()).max()) for k in P}
print("perda (mão):", round(perda_mao, 8), "| perda (torch):", round(float(perda_t), 8)); print("erro máx. dos gradientes:", dif_g)
INFO["gradcheck_max_abs_err"] = max(dif_g.values())
# um passo de gradiente descendente à mão reduz a perda (Aula 3)?
P2 = {k: P[k] - 0.5 * g_mao[k] for k in P}
print("perda antes/depois de 1 passo (lr=0,5):", round(perda_mao, 5), "→", round(I.mlp_gradientes(P2, xb, yb)[1], 5))

# %% [markdown]
# ## 4. Linha de base e escolha da taxa de aprendizado por otimizador (Aula 4)
# Linha de base: regressão logística sobre o mesmo TF-IDF (Aula 6: baselines simples seguem válidos). Para cada otimizador,
# uma grade de taxas de aprendizado, 3 sementes, **escolhida pela perda de validação**.

# %%
from sklearn.linear_model import LogisticRegression
base_val = {}
for C in (0.1, 1, 10, 100):
    lr_ = LogisticRegression(C=C, max_iter=2000).fit(Xtr, tr_y)
    p = lr_.predict_proba(Xva)[:, 1]; base_val[C] = I.bce(p, va_y)
C_best = min(base_val, key=base_val.get); print("LogReg: perda de validação por C:", {k: round(v, 4) for k, v in base_val.items()}, "→ C =", C_best)
LR_GRID = {"SGD": [0.01, 0.1, 1.0, 3.0, 10.0], "SGD+Momentum": [0.001, 0.01, 0.1, 0.3, 1.0],
           "RMSProp": [1e-4, 1e-3, 1e-2, 1e-1], "Adam": [1e-4, 1e-3, 1e-2, 1e-1]}
H_PADRAO = 32
linhas = []
t0 = time.time()
for otim, grade in LR_GRID.items():
    for lr in grade:
        for s in SEMENTES[:3]:
            m, cv, ep = I.treinar_mlp(Xtr, tr_y, Xva, va_y, H=H_PADRAO, otim=otim, lr=lr, epocas=EPOCAS, batch=BATCH, semente=s)
            linhas.append({"otimizador": otim, "lr": lr, "semente": s, "val_loss": min(cv["val"]), "melhor_epoca": ep})
grade_df = pd.DataFrame(linhas)
resumo_lr = grade_df.groupby(["otimizador", "lr"]).agg(val_loss=("val_loss", "mean"), melhor_epoca=("melhor_epoca", "mean")).reset_index()
resumo_lr.to_csv(RES / "tab_grade_lr.csv", index=False)
LR_BEST = {o: float(g.loc[g.val_loss.idxmin(), "lr"]) for o, g in resumo_lr.groupby("otimizador")}
print(resumo_lr.round(4).to_string(index=False)); print("melhores taxas (validação):", LR_BEST, f"[{time.time()-t0:.0f}s]")

# %% [markdown]
# ## 5. Experimento principal: SGD × Momentum × RMSProp × Adam (5 sementes)
# **Hipótese (fixada antes dos resultados, Aula 4):** com a taxa certa para cada um, os métodos adaptativos (RMSProp/Adam)
# atingem menor perda de validação em menos épocas do que o SGD puro; e a taxa ótima do SGD é uma ordem de grandeza (ou mais)
# maior que a do Adam.

# %%
def metr_conj(model, limiar=0.5):
    return {k: I.metricas(CONJ[k][1], I.prever(model, XC[k]), limiar) for k in CONJ}
curvas, rows = {}, []
for otim in LR_GRID:
    cs = []
    for s in SEMENTES:
        m, cv, ep = I.treinar_mlp(Xtr, tr_y, Xva, va_y, H=H_PADRAO, otim=otim, lr=LR_BEST[otim], epocas=EPOCAS, batch=BATCH, semente=s)
        cs.append(cv)
        mm = metr_conj(m)
        for k, v in mm.items():
            rows.append({"otimizador": otim, "semente": s, "conjunto": k, "melhor_epoca": ep, "val_loss": min(cv["val"]), **v})
    curvas[otim] = {k: np.array([c[k] for c in cs]).tolist() for k in cs[0]}
principal = pd.DataFrame(rows); principal.to_csv(RES / "principal_otimizadores.csv", index=False)
salvar_json(curvas, "curvas_otimizadores.json")
def agg(df, cols, by):
    out = []
    for key, g in df.groupby(by, sort=False):
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        for c in cols:
            mu, sd, lo, hi = I.ic95(g[c]); r[c], r[c + "_dp"], r[c + "_lo"], r[c + "_hi"] = mu, sd, lo, hi
        out.append(r)
    return pd.DataFrame(out)
t_otim = agg(principal, ["f1", "auc", "acc", "prec", "rec", "val_loss", "melhor_epoca"], ["otimizador", "conjunto"])
t_otim["lr"] = t_otim.otimizador.map(LR_BEST); t_otim.to_csv(RES / "tab_otimizadores.csv", index=False)
print(t_otim[t_otim.conjunto == "deepset (teste)"][["otimizador", "lr", "val_loss", "melhor_epoca", "f1", "f1_dp", "auc", "auc_dp"]].round(4).to_string(index=False))
# perda de validação em épocas marcadas (velocidade de convergência)
marc = [1, 3, 5, 10, 20, 40]
conv = pd.DataFrame([{"otimizador": o, **{f"ep{e}": float(np.mean(np.array(c["val"])[:, e - 1])) for e in marc}} for o, c in curvas.items()])
conv.to_csv(RES / "tab_convergencia.csv", index=False); print(conv.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
cores = {"SGD": COR["cinza"], "SGD+Momentum": COR["azul2"], "RMSProp": COR["verde"], "Adam": COR["vermelho"]}
for o, c in curvas.items():
    for a, k, tt in ((ax[0], "treino", "Perda de treino"), (ax[1], "val", "Perda de validação")):
        arr = np.array(c[k]); mu, sd = arr.mean(0), arr.std(0, ddof=1); x = np.arange(1, len(mu) + 1)
        a.plot(x, mu, color=cores[o], label=f"{o} (lr={LR_BEST[o]:g})"); a.fill_between(x, mu - sd, mu + sd, color=cores[o], alpha=0.15); a.set_title(tt); a.set_xlabel("época")
ax[0].set_ylabel("entropia cruzada binária"); ax[1].legend(fontsize=7.5)
salvar(fig, FIG / "fig1_otimizadores.png")

# %% [markdown]
# ## 6. Capacidade da rede: tamanho da camada oculta (subajuste × sobreajuste)
# Adam com a melhor taxa; H ∈ {0 (regressão logística), 2, 8, 32, 128, 512}. Contagem de parâmetros: V·H + H + H + 1.

# %%
rows = []
for H in (0, 2, 8, 32, 128, 512):
    for s in SEMENTES:
        m, cv, ep = I.treinar_mlp(Xtr, tr_y, Xva, va_y, H=H, otim="Adam", lr=LR_BEST["Adam"], epocas=EPOCAS, batch=BATCH, semente=s)
        mm = metr_conj(m)
        rows.append({"H": H, "semente": s, "parametros": (Xtr.shape[1] + 1) if H == 0 else I.n_parametros(Xtr.shape[1], H), "melhor_epoca": ep,
                     "treino_loss": cv["treino"][ep - 1], "val_loss": cv["val"][ep - 1], "f1_teste": mm["deepset (teste)"]["f1"],
                     "auc_teste": mm["deepset (teste)"]["auc"], "f1_nc": mm["neuralchemy (OOD)"]["f1"], "f1_xt": mm["xTRam1 (OOD)"]["f1"]})
cap = pd.DataFrame(rows); cap.to_csv(RES / "capacidade_runs.csv", index=False)
t_cap = agg(cap, ["treino_loss", "val_loss", "f1_teste", "auc_teste", "f1_nc", "f1_xt", "melhor_epoca"], ["H"]); t_cap["parametros"] = cap.groupby("H", sort=False).parametros.first().values
t_cap.to_csv(RES / "tab_capacidade.csv", index=False)
print(t_cap[["H", "parametros", "treino_loss", "val_loss", "f1_teste", "f1_teste_dp", "auc_teste"]].round(4).to_string(index=False))
H_BEST = int(t_cap.loc[t_cap[t_cap.H > 0].val_loss.idxmin(), "H"]); print("H escolhido pela validação:", H_BEST)
fig, ax = plt.subplots(figsize=(5.2, 3.3))
xs = np.arange(len(t_cap)); ax.plot(xs, t_cap.treino_loss, "o-", color=COR["azul"], label="treino"); ax.plot(xs, t_cap.val_loss, "s-", color=COR["vermelho"], label="validação")
ax.set_xticks(xs); ax.set_xticklabels([("0 (logística)" if h == 0 else str(h)) for h in t_cap.H], fontsize=7.5); ax.set_xlabel("neurônios na camada oculta (H)"); ax.set_ylabel("perda na melhor época"); ax.legend(); ax.set_title("Capacidade × generalização")
salvar(fig, FIG / "fig2_capacidade.png")

# %% [markdown]
# ## 7. Representação do texto (Aula 6): BoW × TF-IDF × TF-IDF com bigramas

# %%
rows = []
for nome, kw in (("BoW binário", dict(modo="bow")), ("TF-IDF (unigramas)", dict()), ("TF-IDF (uni+bigramas)", dict(bigramas=True))):
    Vz = I.Vetorizador(max_vocab=2000, min_df=2, **kw).ajustar(tr_t)
    a, b = Vz.transformar(tr_t), Vz.transformar(va_t)
    XCz = {k: Vz.transformar(v[0]) for k, v in CONJ.items()}
    for s in SEMENTES:
        m, cv, ep = I.treinar_mlp(a, tr_y, b, va_y, H=H_BEST, otim="Adam", lr=LR_BEST["Adam"], epocas=EPOCAS, batch=BATCH, semente=s)
        mm = {k: I.metricas(CONJ[k][1], I.prever(m, XCz[k])) for k in CONJ}
        rows.append({"representacao": nome, "semente": s, "val_loss": min(cv["val"]), "f1_teste": mm["deepset (teste)"]["f1"], "auc_teste": mm["deepset (teste)"]["auc"],
                     "f1_nc": mm["neuralchemy (OOD)"]["f1"], "f1_xt": mm["xTRam1 (OOD)"]["f1"]})
rep = pd.DataFrame(rows); rep.to_csv(RES / "representacao_runs.csv", index=False)
t_rep = agg(rep, ["val_loss", "f1_teste", "auc_teste", "f1_nc", "f1_xt"], ["representacao"]); t_rep.to_csv(RES / "tab_representacao.csv", index=False)
print(t_rep[["representacao", "val_loss", "f1_teste", "f1_teste_dp", "auc_teste", "f1_nc", "f1_xt"]].round(4).to_string(index=False))

# %% [markdown]
# ## 8. Modelo final, comparação e generalização fora da distribuição
# Configuração final escolhida **apenas pela validação**: Adam, H e taxa acima. Comparação com a regressão logística e com os
# números do Trabalho 01 (DistilBERT/BERT-base, mesmo teste do deepset, 3 sementes), lidos de `trabalho1/resultados/`.

# %%
rows, final_models = [], []
for s in SEMENTES:
    m, cv, ep = I.treinar_mlp(Xtr, tr_y, Xva, va_y, H=H_BEST, otim="Adam", lr=LR_BEST["Adam"], epocas=EPOCAS, batch=BATCH, semente=s)
    final_models.append(m)
    for k, v in metr_conj(m).items(): rows.append({"modelo": f"MLP (H={H_BEST}, Adam)", "semente": s, "conjunto": k, **v})
lr_ = LogisticRegression(C=C_best, max_iter=2000).fit(Xtr, tr_y)
for k in CONJ: rows.append({"modelo": "Regressão logística + TF-IDF", "semente": 0, "conjunto": k, **I.metricas(CONJ[k][1], lr_.predict_proba(XC[k])[:, 1])})
final = pd.DataFrame(rows); final.to_csv(RES / "final_runs.csv", index=False)
t_final = agg(final, ["f1", "auc", "acc", "prec", "rec", "fpr"], ["modelo", "conjunto"]); t_final.to_csv(RES / "tab_final.csv", index=False)
print(t_final[["modelo", "conjunto", "f1", "f1_dp", "auc", "prec", "rec", "fpr"]].round(3).to_string(index=False))
T1 = Path("/home/ubuntu/meridianseal/unisinos/dl/trabalho1/resultados/tab_e2_desempenho.csv")
ref_t1 = pd.read_csv(T1)
ref_t1 = ref_t1[ref_t1.detector.isin(["DistilBERT", "BERT-base", "Regra"])][["detector", "conjunto", "f1", "f1_dp", "auc", "n_sementes"]]
ref_t1.to_csv(RES / "ref_trabalho1.csv", index=False); print(ref_t1.round(2).to_string(index=False))
# matriz de confusão da semente 0 no teste
mc = metr_conj(final_models[0])["deepset (teste)"]; INFO["matriz_confusao_s0"] = {k: mc[k] for k in ("tp", "fp", "fn", "tn")}

# %% [markdown]
# ## 9. Interpretação e custo
# Importância de cada termo por **oclusão**: queda média da probabilidade de injeção quando o termo é removido do texto (textos
# de injeção do treino+teste do deepset em que o termo aparece). E latência de inferência na CPU (tokenização + TF-IDF + MLP).

# %%
m0 = final_models[0]
def prob_texto(t): return float(I.prever(m0, V.transformar([t]))[0])
alvo = [t for t, y in zip(tr_t + te_t, list(tr_y) + list(te_y)) if y == 1]
imp = {}
for t in alvo:
    base = prob_texto(t); toks = list(dict.fromkeys(I.tokenizar(t)))
    for w in toks:
        if w not in V.indice: continue
        t2 = " ".join(x for x in I.tokenizar(t) if x != w)
        imp.setdefault(w, []).append(base - prob_texto(t2))
topo = sorted(((w, float(np.mean(v)), len(v)) for w, v in imp.items() if len(v) >= 5), key=lambda x: -x[1])[:12]
pd.DataFrame(topo, columns=["termo", "queda_media_prob", "n_textos"]).to_csv(RES / "tab_termos_importantes.csv", index=False)
print(topo)
amostra = (te_t * 10)[:500]
t0 = time.perf_counter()
for t in amostra: prob_texto(t)
lat_ms = (time.perf_counter() - t0) / len(amostra) * 1000
INFO["latencia_mlp_ms_por_texto"] = lat_ms; INFO["parametros_mlp"] = I.n_parametros(Xtr.shape[1], H_BEST); INFO["H_final"] = H_BEST; INFO["lr_melhor"] = LR_BEST
INFO["vocab"] = len(V.vocab); INFO["C_logreg"] = C_best
print(f"latência MLP: {lat_ms:.3f} ms/texto | parâmetros: {INFO['parametros_mlp']:,}")

# %%
# Figura 3: generalização (F1) por conjunto
fig, ax = plt.subplots(figsize=(6.4, 3.4))
nomes = ["MLP (H=%d, Adam)" % H_BEST, "Regressão logística + TF-IDF"]
conjs = list(CONJ)
w = 0.2
for i, nm in enumerate(nomes):
    vals = [t_final[(t_final.modelo == nm) & (t_final.conjunto == c)].f1.iloc[0] for c in conjs]
    dp = [t_final[(t_final.modelo == nm) & (t_final.conjunto == c)].f1_dp.iloc[0] for c in conjs]
    ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, yerr=dp, color=[COR["azul"], COR["azul2"]][i], label=nm, capsize=2)
for i, (det, cor) in enumerate((("DistilBERT", COR["vermelho"]), ("BERT-base", COR["verde"]))):
    vals = []
    for c in ["deepset (teste, em-distribuição)", "neuralchemy (OOD)", "xTRam1 (OOD, sintético)"]:
        r = ref_t1[(ref_t1.detector == det) & (ref_t1.conjunto == c)]; vals.append(r.f1.iloc[0] / 100)
    ax.bar(np.arange(3) + (i + 0.5 - 0.0) * w, vals, w, color=cor, label=f"{det} (Trabalho 01)")
ax.set_xticks(range(3)); ax.set_xticklabels(["deepset (teste)", "neuralchemy (OOD)", "xTRam1 (OOD)"], fontsize=8); ax.set_ylabel("F1"); ax.set_ylim(0, 1.32); ax.legend(fontsize=7, ncol=2, loc="upper right")
ax.set_title("Generalização: F1 por conjunto")
salvar(fig, FIG / "fig3_generalizacao.png")
salvar_json(INFO, "info.json")
print("OK — resultados em", RES)
