"""Biblioteca do Trabalho 02 (Deep Learning, PPGCA/Unisinos 2026/2): detecção de prompt injection com MLP sobre TF-IDF.

Contém (i) carga e divisão dos dados, (ii) tokenização e TF-IDF implementados à mão (para que o laboratório web em
JavaScript reproduza exatamente os mesmos números), (iii) um MLP em NumPy com backpropagation escrito à mão (Aulas 2 e 3),
(iv) o mesmo MLP em PyTorch com os quatro otimizadores da Aula 4 e (v) métricas.
"""
import math
import os
import re
import time

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# 1. Dados
# ----------------------------------------------------------------------------------------------------------------------
SEMENTE_DIVISAO = 2026


URL_PT = "https://raw.githubusercontent.com/epineto/prompt-injection-dl-unisinos/main/data/deepset_pt.json"


def carregar_pt():
    """Tradução para português (pt-BR) do deepset/prompt-injections, alinhada item a item com o original (ver dados/LEIA-ME).
    Procura o arquivo localmente (dados/, ../dados/, data/) e, se não achar (p.ex. no Colab), baixa do repositório."""
    import json
    from pathlib import Path
    for base in (Path.cwd(), Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()):
        for rel in ("dados/deepset_pt.json", "../dados/deepset_pt.json", "data/deepset_pt.json", "../data/deepset_pt.json", "deepset_pt.json"):
            f = (base / rel).resolve()
            if f.exists():
                return json.loads(f.read_text())
    import urllib.request
    return json.loads(urllib.request.urlopen(URL_PT, timeout=60).read().decode("utf-8"))


def carregar_dados(n_xtram=1000, br=True):
    """Baixa os datasets do Hugging Face Hub (ou usa o cache local; defina HF_HUB_OFFLINE=1 para forçar o modo offline).
    Em-distribuição: deepset (EN/DE) e a sua tradução para português (alinhada, mesma divisão). Fora da distribuição: neuralchemy,
    xTRam1 (inglês) e, se br=True, o teste do ai-safety-brazil/prompt-injection-br-synthetic (português nativo, sintético)."""
    from datasets import load_dataset
    ds = load_dataset("deepset/prompt-injections")
    nc = load_dataset("neuralchemy/Prompt-injection-dataset", "core")["test"]
    xt = load_dataset("xTRam1/safe-guard-prompt-injection")["test"]
    pt = carregar_pt()
    assert len(pt["treino"]) == len(ds["train"]) and len(pt["teste"]) == len(ds["test"]), "tradução desalinhada com o deepset"
    rng = np.random.RandomState(SEMENTE_DIVISAO)
    # amostra estratificada e determinística do xTRam1 (mesma regra do Trabalho 01: n=1000)
    y = np.array(xt["label"])
    idx = []
    for c in (0, 1):
        ic = np.where(y == c)[0]
        k = int(round(n_xtram * len(ic) / len(y)))
        idx += list(rng.choice(ic, k, replace=False))
    idx = sorted(idx)
    d = {
        "treino_total": (list(ds["train"]["text"]), np.array(ds["train"]["label"])),
        "treino_total_pt": list(pt["treino"]),
        "teste": (list(ds["test"]["text"]), np.array(ds["test"]["label"])),
        "teste_pt": (list(pt["teste"]), np.array(ds["test"]["label"])),
        "neuralchemy": (list(nc["text"]), np.array(nc["label"])),
        "xtram1": ([xt["text"][i] for i in idx], y[idx]),
    }
    if br:
        brd = load_dataset("ai-safety-brazil/prompt-injection-br-synthetic")["test"]
        d["br_nativo"] = (list(brd["text"]), np.array(brd["label"]))
    return d


def dividir_treino_val(y, frac_val=0.2, semente=SEMENTE_DIVISAO):
    """Divisão estratificada fixa treino/validação, devolvida como ÍNDICES (para aplicar o mesmo corte ao original e à tradução:
    um texto e a sua tradução ficam sempre do mesmo lado, sem vazamento). A validação serve a TODAS as escolhas; o teste só ao final."""
    rng = np.random.RandomState(semente)
    iv = []
    for c in (0, 1):
        ic = np.where(y == c)[0]
        iv += list(rng.choice(ic, int(round(frac_val * len(ic))), replace=False))
    iv = np.array(sorted(iv))
    sv = set(iv.tolist())
    it = np.array([i for i in range(len(y)) if i not in sv])
    return it, iv


# ----------------------------------------------------------------------------------------------------------------------
# 2. Tokenização e TF-IDF (à mão; o JavaScript do laboratório faz exatamente o mesmo)
# ----------------------------------------------------------------------------------------------------------------------
_RE_TOKEN = re.compile(r"\w+")


def tokenizar(texto, bigramas=False):
    t = _RE_TOKEN.findall(texto.lower())
    if bigramas:
        t = t + [t[i] + " " + t[i + 1] for i in range(len(t) - 1)]
    return t


class Vetorizador:
    """modo: 'tfidf' (padrão), 'bow' (contagem binária). idf = ln((1+N)/(1+df)) + 1; linhas normalizadas em L2."""

    def __init__(self, max_vocab=2000, min_df=2, bigramas=False, modo="tfidf"):
        self.max_vocab, self.min_df, self.bigramas, self.modo = max_vocab, min_df, bigramas, modo

    def ajustar(self, textos):
        df = {}
        for t in textos:
            for w in set(tokenizar(t, self.bigramas)):
                df[w] = df.get(w, 0) + 1
        itens = [(w, c) for w, c in df.items() if c >= self.min_df]
        itens.sort(key=lambda x: (-x[1], x[0]))           # desempate alfabético (determinístico)
        itens = itens[: self.max_vocab]
        self.vocab = [w for w, _ in itens]
        self.indice = {w: i for i, w in enumerate(self.vocab)}
        n = len(textos)
        self.idf = np.array([math.log((1 + n) / (1 + c)) + 1 for _, c in itens], dtype=np.float64)
        return self

    def transformar(self, textos):
        X = np.zeros((len(textos), len(self.vocab)), dtype=np.float64)
        for r, t in enumerate(textos):
            for w in tokenizar(t, self.bigramas):
                j = self.indice.get(w)
                if j is not None:
                    X[r, j] += 1.0
        if self.modo == "bow":
            X = (X > 0).astype(np.float64)
        else:
            X = X * self.idf
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return X / nrm


# ----------------------------------------------------------------------------------------------------------------------
# 3. MLP em NumPy com backpropagation escrito à mão (Aulas 2 e 3)
# ----------------------------------------------------------------------------------------------------------------------
def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mlp_forward(P, X):
    """X(n,V) -> z1=X@W1+b1 -> h=ReLU(z1) -> z2=h@W2+b2 -> p=sigmoide(z2)."""
    z1 = X @ P["W1"] + P["b1"]
    h = np.maximum(z1, 0.0)
    z2 = h @ P["W2"] + P["b2"]
    return z1, h, z2, sigmoide(z2)


def mlp_gradientes(P, X, y):
    """Regra da cadeia à mão. Com BCE+sigmoide, dL/dz2 = (p - y)/n (Aula 2)."""
    n = X.shape[0]
    z1, h, z2, p = mlp_forward(P, X)
    dz2 = (p - y.reshape(-1, 1)) / n
    g = {"W2": h.T @ dz2, "b2": dz2.sum(0)}
    dh = dz2 @ P["W2"].T
    dz1 = dh * (z1 > 0)
    g["W1"] = X.T @ dz1
    g["b1"] = dz1.sum(0)
    return g, bce(p.ravel(), y)


def n_parametros(V, H):
    """V*H + H (camada oculta) + H*1 + 1 (saída). Mesmo raciocínio dos 641 parâmetros da Aula 4."""
    return V * H + H + H + 1


# ----------------------------------------------------------------------------------------------------------------------
# 4. MLP em PyTorch com os otimizadores da Aula 4
# ----------------------------------------------------------------------------------------------------------------------
OTIMIZADORES = {
    "SGD": lambda ps, lr: __import__("torch").optim.SGD(ps, lr=lr),
    "SGD+Momentum": lambda ps, lr: __import__("torch").optim.SGD(ps, lr=lr, momentum=0.9),
    "RMSProp": lambda ps, lr: __import__("torch").optim.RMSprop(ps, lr=lr, alpha=0.9, eps=1e-8),
    "Adam": lambda ps, lr: __import__("torch").optim.Adam(ps, lr=lr, betas=(0.9, 0.999), eps=1e-8),
}


def criar_mlp(V, H, semente):
    import torch
    torch.manual_seed(semente)
    if H == 0:   # regressão logística (sem camada oculta)
        return torch.nn.Sequential(torch.nn.Linear(V, 1))
    return torch.nn.Sequential(torch.nn.Linear(V, H), torch.nn.ReLU(), torch.nn.Linear(H, 1))


def metricas(y, p, limiar=0.5):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    pred = (p >= limiar).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan")
    return {"acc": (tp + tn) / len(y), "prec": prec, "rec": rec, "f1": f1, "auc": auc, "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def treinar_mlp(Xtr, ytr, Xva, yva, H=32, otim="Adam", lr=1e-3, epocas=40, batch=32, semente=0, patience=None):
    """Treina com mini-batch e devolve o modelo na MELHOR época (menor perda de validação) e as curvas por época."""
    import copy
    import torch
    torch.set_num_threads(1)
    m = criar_mlp(Xtr.shape[1], H, semente)
    opt = OTIMIZADORES[otim](m.parameters(), lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xv = torch.tensor(Xva, dtype=torch.float32); yv = torch.tensor(yva, dtype=torch.float32).view(-1, 1)
    g = torch.Generator().manual_seed(semente)
    curva = {"treino": [], "val": [], "val_acc": [], "val_f1": []}
    melhor, melhor_ep, estado = float("inf"), 0, copy.deepcopy(m.state_dict())
    for ep in range(1, epocas + 1):
        m.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            lossf(m(Xt[b]), yt[b]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            lt = float(lossf(m(Xt), yt)); zv = m(Xv); lv = float(lossf(zv, yv))
            pv = torch.sigmoid(zv).numpy().ravel()
        mt = metricas(yva, pv)
        curva["treino"].append(lt); curva["val"].append(lv); curva["val_acc"].append(mt["acc"]); curva["val_f1"].append(mt["f1"])
        if lv < melhor - 1e-9:
            melhor, melhor_ep, estado = lv, ep, copy.deepcopy(m.state_dict())
        if patience and ep - melhor_ep >= patience:
            break
    m.load_state_dict(estado)
    m.eval()
    return m, curva, melhor_ep


def prever(m, X):
    import torch
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X, dtype=torch.float32))).numpy().ravel()


def ic95(vals):
    """Média, desvio-padrão (ddof=1) e IC95% t-Student entre sementes."""
    from scipy import stats
    v = np.asarray(vals, dtype=float)
    mu, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    h = float(stats.t.ppf(0.975, len(v) - 1) * sd / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return mu, sd, mu - h, mu + h
