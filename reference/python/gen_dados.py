"""Exporta os dados, o vocabulário, o modelo pré-treinado e os resultados do Trabalho 02 para o laboratório web.

Uso (a partir da raiz do repositório):  python reference/python/gen_dados.py
Gera: data/dataset.json, data/vocab.json, data/modelo.json, data/resultados.json.
Requer numpy, pandas, torch, scikit-learn e datasets (com os três conjuntos no cache do Hugging Face).
A divisão treino/validação/teste e a amostragem são determinísticas (semente 2026), as mesmas do trabalho.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(AQUI))
import injecao as I  # noqa: E402

RES = Path(os.environ.get("RESULTADOS", RAIZ / "reference" / "resultados"))
DATA = RAIZ / "data"
DATA.mkdir(exist_ok=True)
N_AMOSTRA = 300
H, LR, EPOCAS, BATCH, SEMENTE = 32, 0.01, 40, 32, 0


def amostra_estratificada(textos, y, n, semente=I.SEMENTE_DIVISAO):
    """n textos, proporção de classes preservada, ordem original mantida (determinístico)."""
    rng = np.random.RandomState(semente)
    y = np.asarray(y)
    idx = []
    for c in (0, 1):
        ic = np.where(y == c)[0]
        k = int(round(n * len(ic) / len(y)))
        idx += list(rng.choice(ic, k, replace=False))
    idx = sorted(idx)
    return [textos[i] for i in idx], [int(y[i]) for i in idx]


def arred(a, d=6):
    """Arredonda para d algarismos significativos (lista de floats)."""
    return [float(f"{float(x):.{d}g}") for x in np.asarray(a).ravel()]


def ler_csv(nome):
    with open(RES / nome, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    D = I.carregar_dados()
    (tr_t, tr_y), (va_t, va_y) = I.dividir_treino_val(*D["treino_total"])
    te_t, te_y = D["teste"]
    nc_t, nc_y = amostra_estratificada(D["neuralchemy"][0], D["neuralchemy"][1], N_AMOSTRA)
    xt_t, xt_y = amostra_estratificada(D["xtram1"][0], D["xtram1"][1], N_AMOSTRA)
    assert (len(tr_t), len(va_t), len(te_t)) == (436, 110, 116)

    dataset = {
        "meta": {"semente_divisao": I.SEMENTE_DIVISAO, "amostra": N_AMOSTRA,
                 "nota": "treino/validação/teste do deepset com as mesmas divisões do trabalho (436/110/116); neuralchemy e xTRam1 são amostras estratificadas e determinísticas de 300 textos."},
        "fontes": {
            "deepset": {"nome": "deepset/prompt-injections", "url": "https://huggingface.co/datasets/deepset/prompt-injections", "licenca": "Apache-2.0 (conforme o cartão do conjunto)"},
            "neuralchemy": {"nome": "neuralchemy/Prompt-injection-dataset (core, test)", "url": "https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset", "licenca": "Apache-2.0"},
            "xtram1": {"nome": "xTRam1/safe-guard-prompt-injection (test)", "url": "https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection", "licenca": "não declarada no cartão do conjunto; sintético; amostra de 300 textos apenas para demonstração acadêmica"},
        },
        "treino": {"texto": tr_t, "y": [int(v) for v in tr_y]},
        "validacao": {"texto": va_t, "y": [int(v) for v in va_y]},
        "teste": {"texto": te_t, "y": [int(v) for v in te_y]},
        "neuralchemy": {"texto": nc_t, "y": nc_y},
        "xtram1": {"texto": xt_t, "y": xt_y},
    }
    (DATA / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    V = I.Vetorizador(max_vocab=2000, min_df=2).ajustar(tr_t)
    n = len(V.vocab)
    assert n == 983
    vocab = {"termos": V.vocab, "idf": [float(f"{x:.12g}") for x in V.idf], "n_documentos": len(tr_t), "min_df": 2, "max_vocab": 2000}
    (DATA / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # --- modelo pré-treinado: Adam, H=32, lr=0,01, semente 0 (o modelo final dos resultados) ---
    Xtr, Xva, Xte = V.transformar(tr_t), V.transformar(va_t), V.transformar(te_t)
    m, curva, ep = I.treinar_mlp(Xtr, tr_y, Xva, va_y, H=H, otim="Adam", lr=LR, epocas=EPOCAS, batch=BATCH, semente=SEMENTE)
    mt = I.metricas(te_y, I.prever(m, Xte))
    assert (mt["tp"], mt["fp"], mt["fn"], mt["tn"]) == (50, 1, 10, 55), mt  # igual à matriz de confusão dos resultados
    W1 = m[0].weight.detach().numpy().T      # (V, H)
    b1 = m[0].bias.detach().numpy()
    W2 = m[2].weight.detach().numpy().ravel()  # (H,)
    b2 = m[2].bias.detach().numpy()
    modelo = {"V": n, "H": H, "otimizador": "Adam", "lr": LR, "semente": SEMENTE, "epocas": EPOCAS, "batch": BATCH, "melhor_epoca": int(ep),
              "n_parametros": I.n_parametros(n, H), "precisao": "float32 arredondado a 6 algarismos significativos",
              "W1": arred(W1), "b1": arred(b1), "W2": arred(W2), "b2": arred(b2),
              "teste_python": {k: mt[k] for k in ("f1", "auc", "tp", "fp", "fn", "tn")}}
    (DATA / "modelo.json").write_text(json.dumps(modelo, separators=(",", ":")), encoding="utf-8")
    print("modelo: melhor época", ep, "| teste", {k: round(v, 4) if isinstance(v, float) else v for k, v in mt.items()})

    # --- resultados.json (a partir dos CSV do trabalho; nenhum número é calculado aqui além de médias das curvas) ---
    info = json.loads((RES / "info.json").read_text())
    curvas = json.loads((RES / "curvas_otimizadores.json").read_text())
    medias = {}
    for o, c in curvas.items():
        medias[o] = {k: [float(f"{x:.5g}") for x in np.array(c[k]).mean(0)] for k in ("treino", "val")}
    t_otim = [r for r in ler_csv("tab_otimizadores.csv")]
    otim = {}
    for r in t_otim:
        o = otim.setdefault(r["otimizador"], {"lr": num(r["lr"]), "val_loss": num(r["val_loss"]), "melhor_epoca": num(r["melhor_epoca"]), "melhor_epoca_dp": num(r["melhor_epoca_dp"]), "f1": {}, "auc": {}})
        o["f1"][r["conjunto"]] = [num(r["f1"]), num(r["f1_dp"])]
        o["auc"][r["conjunto"]] = [num(r["auc"]), num(r["auc_dp"])]
    t_final = ler_csv("tab_final.csv")
    final = [{"modelo": r["modelo"], "conjunto": r["conjunto"], "f1": num(r["f1"]), "f1_dp": num(r["f1_dp"]), "auc": num(r["auc"]), "prec": num(r["prec"]), "rec": num(r["rec"]), "fpr": num(r["fpr"])} for r in t_final]
    final_s0 = [{"conjunto": r["conjunto"], **{k: num(r[k]) for k in ("acc", "prec", "rec", "f1", "auc")}, **{k: int(r[k]) for k in ("tp", "fp", "fn", "tn")}}
                for r in ler_csv("final_runs.csv") if r["semente"] == "0" and r["modelo"].startswith("MLP")]
    lat = {r["detector"]: r for r in ler_csv("t1_tab_e2_latencia.csv")}
    vel = {r["modelo"]: r for r in ler_csv("t1_tab_velocidade.csv")}
    custo = {"mlp": {"parametros": info["parametros_mlp"], "ms_por_texto": info["latencia_mlp_ms_por_texto"]}}
    for k, nome in (("DistilBERT", "distilbert"), ("BERT-base", "bert_base")):
        custo[nome] = {"p50_ms": float(lat[k]["p50_ms"]), "tamanho_mb": float(lat[k]["tamanho_MB"]), "params_m": float(vel[k]["parâmetros obtidos (M)"])}
    custo["mlp"]["tamanho_mb_json"] = round((DATA / "modelo.json").stat().st_size / 1e6, 2)
    custo["fonte_transformers"] = "Trabalho 01 (dl/trabalho1/resultados/tab_e2_latencia.csv; mesmo servidor, CPU)"
    res = {
        "info": info,
        "dados": [{"conjunto": r["conjunto"], "n": int(r["n"]), "injecao": int(r["injecao"]), "pct": num(r["pct_injecao"])} for r in ler_csv("tab_dados.csv")],
        "otimizadores": otim,
        "curvas_medias": medias,
        "convergencia": [{k: (v if k == "otimizador" else num(v)) for k, v in r.items()} for r in ler_csv("tab_convergencia.csv")],
        "grade_lr": [{"otimizador": r["otimizador"], "lr": num(r["lr"]), "val_loss": num(r["val_loss"]), "melhor_epoca": num(r["melhor_epoca"])} for r in ler_csv("tab_grade_lr.csv")],
        "capacidade": [{k: num(r[k]) for k in ("H", "parametros", "treino_loss", "val_loss", "val_loss_dp", "f1_teste", "f1_teste_dp", "auc_teste", "f1_nc", "f1_xt", "melhor_epoca")} for r in ler_csv("tab_capacidade.csv")],
        "representacao": [{"representacao": r["representacao"], **{k: num(r[k]) for k in ("val_loss", "f1_teste", "f1_teste_dp", "auc_teste", "f1_nc", "f1_xt")}} for r in ler_csv("tab_representacao.csv")],
        "final": final,
        "final_semente0": final_s0,
        "trabalho1": [{"detector": r["detector"], "conjunto": r["conjunto"], "f1": num(r["f1"]), "f1_dp": num(r["f1_dp"]), "auc": num(r["auc"]), "n_sementes": int(r["n_sementes"])} for r in ler_csv("ref_trabalho1.csv")],
        "termos": [{"termo": r["termo"], "queda": num(r["queda_media_prob"]), "n": int(r["n_textos"])} for r in ler_csv("tab_termos_importantes.csv")],
        "custo": custo,
    }
    (DATA / "resultados.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for f in ("dataset", "vocab", "modelo", "resultados"):
        print(f, (DATA / f"{f}.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
