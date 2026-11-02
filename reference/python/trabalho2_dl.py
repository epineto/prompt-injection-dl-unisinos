# %% [markdown]
# # Trabalho 02 — Deep Learning (PPGCA/Unisinos, 2026/2)
# **Detecção de *prompt injection* em inglês e português com um MLP sobre TF-IDF: do backpropagation à mão aos otimizadores**
#
# Aluno: Epitácio Neto · Professor: Felipe André Zeiser
#
# **Problema.** Aplicações com modelos de linguagem recebem texto não confiável (o *prompt* do usuário). Um *prompt injection*
# tenta fazer o modelo ignorar as instruções originais (p.ex. "ignore as instruções anteriores e..."). O objetivo é um detector
# **barato e rápido** que classifique cada texto como *seguro* (0) ou *injeção* (1) e funcione como primeiro filtro.
#
# **Duas línguas.** O detector é treinado com texto em **inglês** (parte em inglês do deepset/prompt-injections) **e em português
# do Brasil** (tradução alinhada, texto a texto, do mesmo conjunto; um texto e a sua tradução ficam sempre do mesmo lado da
# divisão, sem vazamento). **Observação sobre os dados:** o deepset original mistura inglês e alemão (e poucas outras línguas); os
# textos em alemão e em outras línguas foram **descartados, não convertidos** (restam 330 textos em inglês: 269 de treino e 61 de
# teste). A tradução para o português é automática, por modelo de linguagem, verificada por amostragem; não é fala nativa. Como
# verificação nativa em português, usa-se o teste do `ai-safety-brazil/prompt-injection-br-synthetic` (sintético, CC-BY-4.0) como
# conjunto fora da distribuição.
#
# **Escopo e relação com as aulas.** O trabalho aplica, em uma tarefa real, o que foi visto: texto → tokens → TF-IDF (Aula 6);
# rede densa (MLP) com ReLU/sigmoide e perda de entropia cruzada binária (Aulas 1 e 2); backpropagation à mão e verificação
# numérica (Aulas 2 e 3); comparação dos otimizadores SGD, Momentum, RMSProp e Adam (Aula 4); escolha de arquitetura justificada
# frente a um modelo pré-treinado (Aulas 3 e 5). Apenas **datasets públicos** são usados.
#
# **Como executar.** `python trabalho2_dl.py` (CPU, ~10 min) ou abrir o notebook. Os datasets são baixados do Hugging Face Hub
# (`deepset/prompt-injections`, `neuralchemy/Prompt-injection-dataset` config `core`, `xTRam1/safe-guard-prompt-injection`,
# `ai-safety-brazil/prompt-injection-br-synthetic`); a tradução (`deepset_pt.json`) vem de `dados/` ou do repositório do laboratório.
# Resultados em `resultados/` e figuras em `figuras/`. Um laboratório web interativo, que repete os experimentos no navegador,
# acompanha o trabalho (ver README do repositório).
#
# **Protocolo.** Treino/validação fixos (80/20 do treino do deepset, estratificado, aplicado ao original e à tradução); **todas as
# escolhas** (taxa de aprendizado, tamanho da camada oculta, época de parada) usam só a validação bilíngue; os testes do deepset
# (EN e PT, em-distribuição) e três conjuntos fora-da-distribuição (neuralchemy, xTRam1, BR-synthetic) são apenas relatados.
# 5 sementes (0 a 4); média ± desvio-padrão.

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
# O deepset original (546 treino / 116 teste) mistura inglês e alemão (e poucas outras línguas). **Os textos em alemão e em outras
# línguas foram descartados** (não convertidos): detecção de língua + marcadores lexicais + revisão manual; restam **330 textos em
# inglês** (269 treino / 61 teste). A tradução desses textos para o português (pt-BR) dobra o corpus. Os conjuntos fora-da-distribuição servem para medir quanto o detector depende do vocabulário visto no treino.

# %%
D = I.carregar_dados()
itr, iva = I.dividir_treino_val(D["treino_total"][1])
tr0 = [D["treino_total"][0][i] for i in itr]; trp = [D["treino_total_pt"][i] for i in itr]; tr_y1 = D["treino_total"][1][itr]
va0 = [D["treino_total"][0][i] for i in iva]; vap = [D["treino_total_pt"][i] for i in iva]; va_y1 = D["treino_total"][1][iva]
MODOS = {"bilingue": (tr0 + trp, np.concatenate([tr_y1, tr_y1]), va0 + vap, np.concatenate([va_y1, va_y1])),
         "orig": (tr0, tr_y1, va0, va_y1), "pt": (trp, tr_y1, vap, va_y1)}
tr_t, tr_y, va_t, va_y = MODOS["bilingue"]
CONJ = {"deepset EN (teste)": D["teste"], "deepset PT (teste)": D["teste_pt"], "neuralchemy (OOD)": D["neuralchemy"],
        "xTRam1 (OOD)": D["xtram1"], "BR-synthetic (PT nativo, OOD)": D["br_nativo"]}
CONJ_BIL = "bilíngue (teste EN + PT)"
tab = [{"conjunto": "treino bilíngue (EN + PT)", "n": len(tr_y), "injecao": int(tr_y.sum())}, {"conjunto": "validação bilíngue", "n": len(va_y), "injecao": int(va_y.sum())}]
tab += [{"conjunto": k, "n": len(v[1]), "injecao": int(v[1].sum())} for k, v in CONJ.items()]
tab_dados = pd.DataFrame(tab); tab_dados["pct_injecao"] = (100 * tab_dados.injecao / tab_dados.n).round(1)
print(tab_dados.to_string(index=False)); tab_dados.to_csv(RES / "tab_dados.csv", index=False)
print("Exemplos (original → tradução):", [(a[:50], b[:50]) for a, b in list(zip(tr0, trp))[:3]])
# Verificação simples da tradução: proporção de textos idênticos ao original (esperado baixo; os já em português ou só código ficam iguais)
ident = float(np.mean([a.strip() == b.strip() for a, b in zip(D["treino_total"][0], D["treino_total_pt"])]))
print("textos de treino idênticos ao original após a tradução:", round(100 * ident, 1), "%"); INFO["traducao_identicos_pct"] = 100 * ident

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
# ## 5. Experimento principal: SGD × Momentum × RMSProp × Adam (5 sementes, treino bilíngue)
# **Hipótese H1 (fixada antes dos resultados, Aula 4):** com a taxa certa para cada um, os métodos adaptativos (RMSProp/Adam)
# atingem a melhor perda de validação em menos épocas do que o SGD puro (e, como expectativa auxiliar, com perda menor); e a taxa
# ótima do SGD é uma ordem de grandeza (ou mais) maior que a do Adam.

# %%
EN, PT_ = "deepset EN (teste)", "deepset PT (teste)"
def metr_de_probs(P, limiar=0.5):
    """Métricas por conjunto a partir das probabilidades P[conjunto]; acrescenta o teste bilíngue (EN + PT concatenados)."""
    out = {k: I.metricas(CONJ[k][1], P[k], limiar) for k in CONJ}
    out[CONJ_BIL] = I.metricas(np.concatenate([CONJ[EN][1], CONJ[PT_][1]]), np.concatenate([P[EN], P[PT_]]), limiar)
    return out
def metr_conj(model, limiar=0.5, XCd=None):
    XCd = XC if XCd is None else XCd
    return metr_de_probs({k: I.prever(model, XCd[k]) for k in CONJ}, limiar)
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
print(t_otim[t_otim.conjunto.isin([EN, PT_, CONJ_BIL])][["otimizador", "conjunto", "lr", "val_loss", "melhor_epoca", "f1", "f1_dp", "auc", "auc_dp"]].round(4).to_string(index=False))
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
# **Escolha de H:** menor H cuja perda de validação fica a ≤ 5% da mínima (regra de parcimônia, ver a célula abaixo).

# %%
rows = []
for H in (0, 2, 8, 32, 128, 512):
    for s in SEMENTES:
        m, cv, ep = I.treinar_mlp(Xtr, tr_y, Xva, va_y, H=H, otim="Adam", lr=LR_BEST["Adam"], epocas=EPOCAS, batch=BATCH, semente=s)
        mm = metr_conj(m)
        rows.append({"H": H, "semente": s, "parametros": (Xtr.shape[1] + 1) if H == 0 else I.n_parametros(Xtr.shape[1], H), "melhor_epoca": ep,
                     "treino_loss": cv["treino"][ep - 1], "val_loss": cv["val"][ep - 1], "f1_en": mm[EN]["f1"], "f1_pt": mm[PT_]["f1"], "f1_bil": mm[CONJ_BIL]["f1"],
                     "auc_bil": mm[CONJ_BIL]["auc"], "f1_nc": mm["neuralchemy (OOD)"]["f1"], "f1_xt": mm["xTRam1 (OOD)"]["f1"], "f1_br": mm["BR-synthetic (PT nativo, OOD)"]["f1"]})
cap = pd.DataFrame(rows); cap.to_csv(RES / "capacidade_runs.csv", index=False)
t_cap = agg(cap, ["treino_loss", "val_loss", "f1_en", "f1_pt", "f1_bil", "auc_bil", "f1_nc", "f1_xt", "f1_br", "melhor_epoca"], ["H"]); t_cap["parametros"] = cap.groupby("H", sort=False).parametros.first().values
t_cap.to_csv(RES / "tab_capacidade.csv", index=False)
print(t_cap[["H", "parametros", "treino_loss", "val_loss", "f1_en", "f1_pt", "f1_bil", "f1_bil_dp", "auc_bil"]].round(4).to_string(index=False))
# Regra de parcimônia (adotada APÓS observar a curva, pois as perdas de validação de H=32 a 512 diferem muito pouco): menor H > 0 cuja
# perda de validação fica no máximo 5% acima da mínima. Sem essa regra, o mínimo estrito escolheria H=512 (≈955 mil parâmetros).
_c = t_cap[t_cap.H > 0]; H_MIN_ESTRITO = int(_c.loc[_c.val_loss.idxmin(), "H"])
H_BEST = int(_c[_c.val_loss <= 1.05 * _c.val_loss.min()].H.min())
INFO["H_minimo_estrito"] = H_MIN_ESTRITO; print("H de menor perda de validação:", H_MIN_ESTRITO, "| H escolhido (parcimônia, ≤5% da mínima):", H_BEST)
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
        mm = metr_conj(m, XCd=XCz)
        rows.append({"representacao": nome, "semente": s, "val_loss": min(cv["val"]), "f1_en": mm[EN]["f1"], "f1_pt": mm[PT_]["f1"], "f1_bil": mm[CONJ_BIL]["f1"], "auc_bil": mm[CONJ_BIL]["auc"],
                     "f1_nc": mm["neuralchemy (OOD)"]["f1"], "f1_xt": mm["xTRam1 (OOD)"]["f1"], "f1_br": mm["BR-synthetic (PT nativo, OOD)"]["f1"]})
rep = pd.DataFrame(rows); rep.to_csv(RES / "representacao_runs.csv", index=False)
t_rep = agg(rep, ["val_loss", "f1_en", "f1_pt", "f1_bil", "auc_bil", "f1_nc", "f1_xt", "f1_br"], ["representacao"]); t_rep.to_csv(RES / "tab_representacao.csv", index=False)
print(t_rep[["representacao", "val_loss", "f1_en", "f1_pt", "f1_bil", "f1_bil_dp", "auc_bil", "f1_nc", "f1_xt"]].round(4).to_string(index=False))

# %% [markdown]
# ## 8. Línguas: treinar só em EN × só em PT × bilíngue
# **Hipótese H4 (fixada antes dos resultados):** um modelo treinado só em EN generaliza mal para o português (vocabulário
# disjunto, exceto termos compartilhados), e o treino bilíngue recupera o desempenho em português sem perder em inglês.
# Mesmo protocolo (Adam, H e taxa escolhidos antes), vetorizador ajustado em cada corpus de treino, 5 sementes.

# %%
rows = []
for modo, (a_t, a_y, b_t, b_y) in MODOS.items():
    Vm = I.Vetorizador(max_vocab=2000, min_df=2).ajustar(a_t)
    A, B = Vm.transformar(a_t), Vm.transformar(b_t)
    XCm = {k: Vm.transformar(v[0]) for k, v in CONJ.items()}
    for s in SEMENTES:
        m, cv, ep = I.treinar_mlp(A, a_y, B, b_y, H=H_BEST, otim="Adam", lr=LR_BEST["Adam"], epocas=EPOCAS, batch=BATCH, semente=s)
        for k, v in metr_conj(m, XCd=XCm).items():
            rows.append({"treino": modo, "vocab": len(Vm.vocab), "semente": s, "conjunto": k, "melhor_epoca": ep, **v})
idi = pd.DataFrame(rows); idi.to_csv(RES / "idiomas_runs.csv", index=False)
t_idi = agg(idi, ["f1", "auc", "prec", "rec"], ["treino", "conjunto"]); t_idi["vocab"] = t_idi.treino.map(idi.groupby("treino").vocab.first())
t_idi.to_csv(RES / "tab_idiomas.csv", index=False)
print(t_idi[t_idi.conjunto.isin([EN, PT_, CONJ_BIL, "BR-synthetic (PT nativo, OOD)"])][["treino", "vocab", "conjunto", "f1", "f1_dp", "auc", "rec"]].round(3).to_string(index=False))
cols = [EN, PT_, "BR-synthetic (PT nativo, OOD)"]; nomes_t = {"orig": "só EN", "pt": "só PT", "bilingue": "bilíngue"}
mat = np.array([[t_idi[(t_idi.treino == m_) & (t_idi.conjunto == c)].f1.iloc[0] for c in cols] for m_ in MODOS])
fig, ax = plt.subplots(figsize=(5.4, 2.9)); im = ax.imshow(mat, vmin=0.3, vmax=1, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(["teste EN", "teste PT\n(trad.)", "BR-synth.\n(PT nativo)"], fontsize=8); ax.set_yticks(range(3)); ax.set_yticklabels([f"treino {nomes_t[m_]}" for m_ in MODOS], fontsize=8)
for i_ in range(3):
    for j_ in range(3): ax.text(j_, i_, f"{mat[i_, j_]:.2f}".replace(".", ","), ha="center", va="center", color="white" if mat[i_, j_] > 0.7 else COR["azul"], fontsize=10, fontweight="bold")
ax.grid(False); ax.set_title("F1 por língua de treino × teste"); salvar(fig, FIG / "fig4_idiomas.png")

# %% [markdown]
# ## 9. Modelo final, comparação e generalização fora da distribuição
# Configuração final escolhida **apenas pela validação**: Adam, H e taxa acima, treino bilíngue. Comparação com a regressão
# logística e com os números do Trabalho 01 (DistilBERT/BERT-base, mesmo teste do deepset, restrito aos textos em inglês, 3 sementes), recalculados de
# `trabalho1/resultados/` (probabilidades por texto; só existem para o inglês e para os conjuntos fora da distribuição em inglês).

# %%
rows, final_models = [], []
for s in SEMENTES:
    m, cv, ep = I.treinar_mlp(Xtr, tr_y, Xva, va_y, H=H_BEST, otim="Adam", lr=LR_BEST["Adam"], epocas=EPOCAS, batch=BATCH, semente=s)
    final_models.append(m)
    for k, v in metr_conj(m).items(): rows.append({"modelo": f"MLP (H={H_BEST}, Adam)", "semente": s, "conjunto": k, **v})
lr_ = LogisticRegression(C=C_best, max_iter=2000).fit(Xtr, tr_y)
for k, v in metr_de_probs({k: lr_.predict_proba(XC[k])[:, 1] for k in CONJ}).items(): rows.append({"modelo": "Regressão logística + TF-IDF", "semente": 0, "conjunto": k, **v})
final = pd.DataFrame(rows); final.to_csv(RES / "final_runs.csv", index=False)
t_final = agg(final, ["f1", "auc", "acc", "prec", "rec", "fpr"], ["modelo", "conjunto"]); t_final.to_csv(RES / "tab_final.csv", index=False)
print(t_final[["modelo", "conjunto", "f1", "f1_dp", "auc", "prec", "rec", "fpr"]].round(3).to_string(index=False))
T1 = Path("/home/ubuntu/meridianseal/unisinos/dl/trabalho1/resultados/tab_e2_desempenho.csv")
if not T1.exists(): T1 = AQUI.parent / "resultados" / "ref_trabalho1.csv"  # (cópia só com os valores; sem as probabilidades por texto o recálculo é pulado)
ref_t1 = pd.read_csv(T1)
ref_t1 = ref_t1[ref_t1.detector.isin(["DistilBERT", "BERT-base", "Regra"])][["detector", "conjunto", "f1", "f1_dp", "auc", "n_sementes"]]
# O Trabalho 01 avaliou o teste completo do deepset (116 textos, EN/DE). Para comparar na MESMA base, F1/AUC do DistilBERT e do
# BERT-base são recalculados só nos textos em inglês (61) a partir das probabilidades por texto guardadas no Trabalho 01.
import glob as _glob
_ie = D["indices_en"][1]; _yfull = np.array(__import__("datasets").load_dataset("deepset/prompt-injections")["test"]["label"])
_lin = []
for _det, _pat in (("DistilBERT", "det__DistilBERT__deepset__s*.json"), ("BERT-base", "det__BERT_base__deepset__s*.json")):
    _fs = []
    for _f in sorted(_glob.glob(str(T1.parent / "e2" / _pat))):
        _p = np.array(json.load(open(_f))["probs"]["deepset_teste"])[_ie]; _m = I.metricas(_yfull[_ie], _p); _fs.append((100 * _m["f1"], 100 * _m["auc"]))
    if _fs:
        _a = np.array(_fs); _lin.append({"detector": _det, "conjunto": "deepset (teste, só inglês)", "f1": _a[:, 0].mean(), "f1_dp": _a[:, 0].std(ddof=1), "auc": _a[:, 1].mean(), "n_sementes": len(_fs)})
if _lin: ref_t1 = pd.concat([ref_t1, pd.DataFrame(_lin)], ignore_index=True)
ref_t1.to_csv(RES / "ref_trabalho1.csv", index=False); print(ref_t1.round(2).to_string(index=False))
for nm_, key_ in (("en", EN), ("pt", PT_)):
    mc = metr_conj(final_models[0])[key_]; INFO[f"matriz_confusao_s0_{nm_}"] = {k: mc[k] for k in ("tp", "fp", "fn", "tn")}

# %% [markdown]
# ## 10. Interpretação e custo
# Importância de cada termo por **oclusão**: queda média da probabilidade de injeção quando o termo é removido do texto (textos
# de injeção de treino e teste, nas duas línguas, em que o termo aparece). E latência de inferência na CPU (tokenização + TF-IDF + MLP).

# %%
m0 = final_models[0]
def prob_texto(t): return float(I.prever(m0, V.transformar([t]))[0])
te_t = CONJ[EN][0] + CONJ[PT_][0]; te_y = np.concatenate([CONJ[EN][1], CONJ[PT_][1]])
alvo = [t for t, y in zip(tr_t + te_t, list(tr_y) + list(te_y)) if y == 1]
imp = {}
for t in alvo:
    base = prob_texto(t); toks = list(dict.fromkeys(I.tokenizar(t)))
    for w in toks:
        if w not in V.indice: continue
        t2 = " ".join(x for x in I.tokenizar(t) if x != w)
        imp.setdefault(w, []).append(base - prob_texto(t2))
topo = sorted(((w, float(np.mean(v)), len(v)) for w, v in imp.items() if len(v) >= 5), key=lambda x: -x[1])[:16]
pd.DataFrame(topo, columns=["termo", "queda_media_prob", "n_textos"]).to_csv(RES / "tab_termos_importantes.csv", index=False)
print(topo)
amostra = (te_t * 3)[:500]
t0 = time.perf_counter()
for t in amostra: prob_texto(t)
lat_ms = (time.perf_counter() - t0) / len(amostra) * 1000
INFO["latencia_mlp_ms_por_texto"] = lat_ms; INFO["parametros_mlp"] = I.n_parametros(Xtr.shape[1], H_BEST); INFO["H_final"] = H_BEST; INFO["lr_melhor"] = LR_BEST
INFO["vocab"] = len(V.vocab); INFO["C_logreg"] = C_best
print(f"latência MLP: {lat_ms:.3f} ms/texto | parâmetros: {INFO['parametros_mlp']:,}")

# %%
# Figura 3: generalização (F1) por conjunto
conjs = [EN, PT_, "neuralchemy (OOD)", "xTRam1 (OOD)", "BR-synthetic (PT nativo, OOD)"]
rot = ["deepset\nEN", "deepset\nPT (trad.)", "neuralchemy\n(OOD, EN)", "xTRam1\n(OOD, EN)", "BR-synth.\n(OOD, PT nativo)"]
fig, ax = plt.subplots(figsize=(7.2, 3.5)); w = 0.26
# MLP só EN (do experimento de línguas) ao lado do bilíngue e da logística
series = [(f"MLP bilíngue (H={H_BEST}, Adam)", [t_final[(t_final.modelo == f"MLP (H={H_BEST}, Adam)") & (t_final.conjunto == c)] for c in conjs], COR["azul"]),
          ("MLP só EN", [t_idi[(t_idi.treino == "orig") & (t_idi.conjunto == c)] for c in conjs], COR["cinza"]),
          ("Regressão logística bilíngue", [t_final[(t_final.modelo == "Regressão logística + TF-IDF") & (t_final.conjunto == c)] for c in conjs], COR["azul2"])]
for i, (nm, dfs, cor) in enumerate(series):
    ax.bar(np.arange(5) + (i - 1) * w, [d.f1.iloc[0] for d in dfs], w, yerr=[d.f1_dp.iloc[0] for d in dfs], color=cor, label=nm, capsize=2)
mapa_t1 = {EN: "deepset (teste, só inglês)", "neuralchemy (OOD)": "neuralchemy (OOD)", "xTRam1 (OOD)": "xTRam1 (OOD, sintético)"}
for det, mk, cor in (("DistilBERT", "D", COR["vermelho"]), ("BERT-base", "s", COR["verde"])):
    xs_, ys_ = [], []
    for ci, c in enumerate(conjs):
        if c in mapa_t1:
            xs_.append(ci); ys_.append(ref_t1[(ref_t1.detector == det) & (ref_t1.conjunto == mapa_t1[c])].f1.iloc[0] / 100)
    ax.scatter(xs_, ys_, marker=mk, s=38, color=cor, zorder=5, edgecolor="white", label=f"{det} (Trabalho 01, EN)")
ax.set_xticks(range(5)); ax.set_xticklabels(rot, fontsize=7.5); ax.set_ylabel("F1"); ax.set_ylim(0, 1.3); ax.legend(fontsize=6.8, ncol=2, loc="upper right"); ax.set_title("Generalização: F1 por conjunto")
salvar(fig, FIG / "fig3_generalizacao.png")
salvar_json(INFO, "info.json")
print("OK — resultados em resultados/")
