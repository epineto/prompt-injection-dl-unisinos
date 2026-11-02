<p align="center"><img src="assets/logo_unisinos.png" alt="Unisinos" width="200"></p>

# Laboratório de Deep Learning: detectando prompt injection com um MLP

**🇧🇷 Português** · [🇬🇧 English below](#english)

[![Testes](https://github.com/epineto/prompt-injection-dl-unisinos/actions/workflows/test.yml/badge.svg)](https://github.com/epineto/prompt-injection-dl-unisinos/actions/workflows/test.yml)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-green.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/demo-GitHub%20Pages-1D2369.svg)](https://epineto.github.io/prompt-injection-dl-unisinos/)

**Demonstração: <https://epineto.github.io/prompt-injection-dl-unisinos/>**

![Página inicial em português](docs/img/desktop-pt-intro.png)

## O que é
Aplicação web interativa (100% no navegador, sem dependências e sem build) que acompanha o **Trabalho 02 de Deep Learning 2026/2** (PPGCA/Unisinos): detecção de *prompt injection* **em inglês, alemão e português** com um MLP sobre TF-IDF, usado como primeiro filtro barato e rápido. O treino é bilíngue: o deepset/prompt-injections (EN/DE) e a sua tradução para o português do Brasil (**tradução automática por LLM, verificada por amostragem; não é fala nativa**). O laboratório permite testar textos (EN, DE e PT) com o modelo final do trabalho, treinar uma rede do zero no navegador (escolhendo as línguas de treino), comparar SGD, Momentum, RMSProp e Adam e acompanhar o backpropagation de uma rede 2-2-1 com todos os números visíveis. O núcleo (`src/core/`) reproduz o código Python de `reference/python/` e é verificado contra ele nos testes.

O idioma padrão é o português (pt-BR); o botão **EN** troca para inglês. Campos numéricos aceitam vírgula ou ponto decimal.

## Seções
Barra fixa no topo, com roteamento por âncora:

| Âncora | O que faz |
|---|---|
| `#intro` | Problema, pipeline texto → tokens → TF-IDF → MLP → probabilidade e atalhos |
| `#testar` | Modelo pré-treinado (Adam, H = 32, taxa 0,001, semente 0, treino bilíngue) em tempo real, com exemplos em EN, DE e PT: probabilidade, medidor, limiar, destaque por palavra (oclusão), o que o modelo vê, matrizes de confusão, F1 e AUC por língua (EN/DE, PT, bilíngue) e fora da distribuição (neuralchemy, xTRam1, BR-synthetic) |
| `#treinar` | Treino do zero em Web Worker: línguas de treino (bilíngue, só EN/DE ou só PT), otimizador, taxa, H, lote, épocas e semente; curvas ao vivo, melhor época (parada antecipada), contagem de parâmetros e métricas finais |
| `#otimizadores` | SGD × Momentum × RMSProp × Adam, mesma semente, taxas do trabalho (1; 0,1; 0,01; 0,001), multiplicador ×0,1 / ×1 / ×10 e a opção "Adam com lr 0,01" |
| `#backprop` | Rede 2-2-1 (XOR): forward, perda, regra da cadeia e atualização passo a passo, com fronteira de decisão |
| `#resultados` | Tabelas e gráficos do relatório (otimizadores, escolha da taxa, capacidade, representação, matriz língua de treino × teste, generalização, custo) e mapa relatório → laboratório |
| `#sobre` | Como rodar, estrutura, tabela Aula → seção, limitações, créditos e licenças |

Roteiro de uso em 20 s por seção: [docs/NAVEGACAO.md](docs/NAVEGACAO.md).

## Como rodar localmente
O app é estático (módulos ES e Web Workers), mas precisa de um servidor HTTP:

```bash
npm run serve            # equivale a: python3 -m http.server 8000
# abra http://localhost:8000
```

> Abrir `index.html` por `file://` **não funciona**: navegadores bloqueiam módulos ES e Web Workers nesse modo.

O GitHub Pages publica a **raiz** do repositório (todos os caminhos são relativos).

## Testes
```bash
npm test                 # node --test tests/  (Node >= 18; sem dependências)
```
Os testes verificam, contra o Python (`tests/reference.json`, gerado por `reference/python/gen_reference.py`): tokenização (hash de todos os textos, inclusive português com acentos e cedilha), TF-IDF, forward, gradientes (também por diferenças finitas), um e três passos dos quatro otimizadores (tolerância 1e-5), mulberry32, métricas (inclusive AUC com empates), a rede 2-2-1, o modelo pré-treinado exportado (60 textos, nas três línguas), os vocabulários por língua de treino, a leitura de números com vírgula, a integridade dos JSON de dados e a paridade dos dicionários PT/EN. A integração contínua (`.github/workflows/test.yml`) roda `npm test` com Node 20 a cada push.

## Estrutura
```
index.html  style.css  404.html  manifest.webmanifest
assets/      logo, ícones, imagem social
data/        dataset.json, vocab.json, modelo.json, resultados.json, deepset_pt.json (tradução)
notebook/    notebook executado do trabalho
src/core/    rng, texto (tokenização e TF-IDF), mlp, otimizadores, treino, métricas, xor, números
src/ui/      interface por seção; src/worker.js (treino em Web Worker)
src/i18n/    pt.js e en.js
tests/       testes (node --test) e reference.json
reference/   python/ (injecao.py, trabalho2_dl.py, geradores) e resultados/ (CSV do trabalho)
docs/        REPRODUCAO.md, REPRODUCTION.md, NAVEGACAO.md, img/
tools/       capturas.py (capturas de tela com Playwright)
```

## Relatório → laboratório
| Item do relatório | Onde |
|---|---|
| Dados e protocolo (872/220; testes EN/DE e PT; OOD) | `#resultados`, `#testar` |
| TF-IDF à mão e backpropagation à mão (verificados contra scikit-learn e autograd) | `#testar`, `#backprop`, `npm test` |
| Comparação de otimizadores (Figura 1) | `#otimizadores`, `#resultados` |
| Capacidade (Figura 2) e contagem de parâmetros | `#treinar`, `#resultados` |
| Línguas de treino × teste (Figura 4, hipótese H4) | `#treinar`, `#resultados` |
| Generalização frente a DistilBERT/BERT-base (Figura 3) e custo | `#resultados` |
| Interpretação por oclusão | `#testar`, `#resultados` |

Mapa completo, com os valores esperados: [docs/REPRODUCAO.md](docs/REPRODUCAO.md).

## Limites conhecidos
- **O português é tradução automática** (LLM, verificada por amostragem), não fala nativa. O teste PT é a tradução do teste EN/DE (em-distribuição, mais fácil); no português nativo (BR-synthetic, textos longos de contexto mais instrução) o modelo não generaliza: classifica tudo como injeção (F1 0,571).
- O treino no navegador usa inicialização He e embaralhamento mulberry32; o PyTorch usa outros. Os números ficam próximos dos do Python, não idênticos; a paridade numérica é verificada nos otimizadores, gradientes e no modelo pré-treinado.
- Os conjuntos fora da distribuição no laboratório são amostras estratificadas (300 textos do neuralchemy e do xTRam1; 200 do BR-synthetic); os números do conjunto completo (Python) aparecem em separado.
- O vocabulário (1.863 termos no treino bilíngue) vem do treino; termos novos, outros idiomas e ofuscação são ignorados.
- A oclusão é uma explicação local e aproximada; com a probabilidade saturada, remover um termo muda pouco.
- Com 872 amostras de treino (436 textos e as suas traduções), o ganho do MLP sobre a regressão logística é modesto (cerca de 4 pontos de F1).
- Somente conjuntos públicos são usados; nada é afirmado sobre resultados da tese.

## Dados e licenças
- [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections): Apache-2.0 (conforme o cartão do conjunto).
- `data/deepset_pt.json`: tradução para o português do Brasil do deepset (546 treino + 116 teste), automática por LLM, verificada por amostragem (cerca de 24 correções); **não é fala nativa**. Obra derivada do deepset sob Apache-2.0, com atribuição ao deepset.
- [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) (core, teste): Apache-2.0; amostra de 300 textos.
- [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) (teste): licença não declarada no cartão; conjunto sintético; amostra de 300 textos incluída apenas para demonstração acadêmica. Antes de redistribuir, conferir a licença no cartão do conjunto.
- [ai-safety-brazil/prompt-injection-br-synthetic](https://huggingface.co/datasets/ai-safety-brazil/prompt-injection-br-synthetic) (teste): CC-BY-4.0, **com atribuição** a ai-safety-brazil; amostra estratificada de 200 textos (português nativo, sintético).

## Créditos
Epitácio Vicente do Nascimento Neto, Programa de Pós-Graduação em Computação Aplicada (PPGCA), Unisinos. Disciplina Deep Learning 2026/2, Prof. Felipe André Zeiser. Código sob licença [MIT](LICENSE).

## Como citar
Ver [CITATION.cff](CITATION.cff).

---

# English

**Deep Learning Lab: detecting prompt injection with an MLP.** A static web app (100% in the browser, no dependencies, no build) that accompanies Deep Learning Assignment 02 (PPGCA/Unisinos, 2026/2). It lets you test texts in English, German or Portuguese with the assignment's final model (MLP over TF-IDF trained on bilingual data: deepset EN/DE plus its **automatic, LLM-made Portuguese translation, not native speech**), train a network from scratch in a Web Worker (choosing the training languages), compare SGD, Momentum, RMSProp and Adam, and watch backpropagation on a 2-2-1 network with every number visible. Click **EN** in the header for the English interface.

![Home page in English](docs/img/desktop-en-intro.png)

| Anchor | What it does |
|---|---|
| `#intro` | Problem, pipeline text → tokens → TF-IDF → MLP → probability |
| `#testar` | Pre-trained model in real time (EN, DE, PT examples), per-word occlusion, confusion matrices per language and out of distribution |
| `#treinar` | Live training from scratch, training-language selector (bilingual / EN/DE only / PT only), early stopping, parameter count |
| `#otimizadores` | Four optimizers side by side (rates 1; 0.1; 0.01; 0.001), ×0.1 / ×1 / ×10 multiplier and an "Adam with lr 0.01" option |
| `#backprop` | XOR with a 2-2-1 network, step by step |
| `#resultados` | Report tables and charts (incl. the training-language × test matrix), report → lab map |
| `#sobre` | How to run, structure, lecture → section table, limitations |

**Run locally:** `npm run serve` (a static HTTP server is required; `file://` does not work), then open <http://localhost:8000>. **Test:** `npm test` (Node 18+, no dependencies). **Reproduce the report:** [docs/REPRODUCTION.md](docs/REPRODUCTION.md). **Navigation guide:** [docs/NAVEGACAO.md](docs/NAVEGACAO.md).

**Known limits:** browser training differs from PyTorch in initialization and shuffling (close, not identical numbers); the Portuguese is automatic translation, not native speech, and the model does not generalize to native Portuguese (BR-synthetic: everything labelled injection, F1 0.571); out-of-distribution sets are samples here (300, 300 and 200 texts; full-set Python numbers are shown separately); the MLP’s gain over logistic regression is modest (about 4 F1 points); only public datasets are used. **Licenses:** deepset and neuralchemy Apache-2.0 (the Portuguese translation is a derivative work of deepset, with attribution); xTRam1 declares no license on its card (check before redistributing the sample); BR-synthetic (ai-safety-brazil) CC-BY-4.0, with attribution.

**Credits:** Epitácio Vicente do Nascimento Neto, PPGCA/Unisinos; Prof. Felipe André Zeiser. License: [MIT](LICENSE). Citation: [CITATION.cff](CITATION.cff).
