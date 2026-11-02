# Reprodução dos resultados do relatório no laboratório

[English version](REPRODUCTION.md) · [README](../README.md)

Este guia mapeia cada tabela, figura e número do Trabalho 02 de Deep Learning à seção do laboratório (<https://epineto.github.io/prompt-injection-dl-unisinos/>) e informa os valores esperados. Os valores de referência vêm de `reference/resultados/` (gerados por `reference/python/trabalho2_dl.py`) e estão em `data/resultados.json`.

## Tolerâncias
O treino no navegador usa inicialização He e embaralhamento por mulberry32; o PyTorch usa outra inicialização e outro gerador. Por isso, **o que é determinístico coincide** (tokenização, TF-IDF, forward, gradientes, um passo de cada otimizador, modelo pré-treinado; ver `npm test`) e **o que depende de sorteio coincide estatisticamente**: esperam-se valores dentro do desvio entre sementes, não idênticos.

## Mapa de correspondência

| Item do trabalho | Onde no laboratório | Como obter | Valor esperado |
|---|---|---|---|
| Dados: 436 treino, 110 validação, 116 teste (deepset); OOD neuralchemy e xTRam1 | `#resultados` → Dados | Abrir a seção | 162, 41 e 60 injeções; 37,2%, 37,3%, 51,7% |
| Vocabulário de 983 termos; TF-IDF igual ao scikit-learn (dif. ≈ 1e-16) | `#testar` → O que o modelo vê; `tests/texto.test.js` | `npm test` | 983 termos; vetores idênticos aos do Python (tolerância 1e-9) |
| Gradientes à mão = autograd (erro ≈ 1e-16) | `#backprop`; `tests/mlp.test.js`, `tests/xor.test.js` | `npm test` | Gradientes iguais ao Python e às diferenças finitas |
| Contagem de parâmetros V·H+H+H+1 = 31.521 (H = 32) | `#treinar` → Parâmetros | Alterar H | 31.521 para H = 32; 1.971 para H = 2 |
| Modelo final (Adam, H = 32, taxa 0,01, semente 0): matriz de confusão TP 50, FP 1, FN 10, TN 55; F1 0,901 | `#testar` → deepset | Limiar 0,50 | Idêntico (modelo pré-treinado exportado) |
| OOD (semente 0, conjunto completo): neuralchemy F1 0,705; xTRam1 F1 0,538 | `#testar` (linha "Python, conjunto completo"); matrizes da amostra de 300 textos | Limiar 0,50 | Amostra: F1 próximo (0,702 e 0,559) |
| Figura 1: otimizadores (SGD 1,0; Momentum 0,1; RMSProp 0,01; Adam 0,01) | `#otimizadores` | Multiplicador ×1, semente 0, **Comparar** | RMSProp e Adam: melhor época por volta de 5 a 6; SGD e Momentum: por volta de 20 a 23; referência de 5 sementes na tabela inferior |
| Taxa grande/pequena demais (Aula 4) | `#otimizadores` | Multiplicador ×10 e ×0,1 | ×10: oscilação/divergência; ×0,1: convergência lenta |
| F1 de teste por otimizador (SGD 0,842; Momentum 0,840; RMSProp 0,868; Adam 0,873 ± 0,018) | `#resultados` → Otimizadores | Abrir a seção | Valores acima; no navegador, 1 semente: próximos |
| Figura 2: capacidade (H = 0, 2, 8, 32, 128, 512) | `#resultados` → Capacidade; `#treinar` | Em `#treinar`, variar H | F1 de teste entre 0,86 e 0,88; melhor perda de validação em H = 8 a 32; H = 512 pior (0,284) |
| Representação (BoW 0,873; TF-IDF 0,873; uni+bigramas 0,883) | `#resultados` → Representação | Abrir a seção | Valores acima (dentro do desvio); sem controle no navegador |
| Figura 3: generalização × DistilBERT/BERT-base (Trabalho 01) | `#resultados` → Generalização | Abrir a seção | MLP deepset 0,873; neuralchemy 0,712; xTRam1 0,534 |
| Custo: 31.521 parâmetros, ≈ 0,15 ms por texto, ≈ 340× mais rápido que o DistilBERT | `#resultados` → Custo | Abrir a seção | 0,15 ms contra 52,2 ms (p50) |
| Termos mais influentes por oclusão ("digo", "context", "alles", "code"…) | `#resultados` → Termos; `#testar` (destaque) | Colar um texto com esses termos | Tabela do trabalho; no destaque, a cor é a queda de probabilidade |
| Passo de gradiente descendente reduz a perda | `#backprop` | **Passo** | A perda média cai a cada passo com taxa adequada |

## Atalho de 1 minuto
1. `#testar`: exemplo "Injeção clássica (EN)" → probabilidade ≈ 99,8%.
2. `#treinar`: **Treinar** com os valores padrão (leva segundos) → melhor época por volta de 5.
3. `#otimizadores`: **Comparar** → adaptativos mais rápidos, não melhores.
4. `#backprop`: **Treinar** → a perda cai e a fronteira separa o XOR.

## Regerar dados e referências
```bash
python reference/python/gen_dados.py        # data/*.json (treina o modelo pré-treinado; requer torch, scikit-learn, datasets)
python reference/python/gen_reference.py    # tests/reference.json
npm test
```
