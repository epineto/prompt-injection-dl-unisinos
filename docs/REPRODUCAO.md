# Reprodução dos resultados do relatório no laboratório

[English version](REPRODUCTION.md) · [README](../README.md)

Este guia mapeia cada tabela, figura e número do Trabalho 02 de Deep Learning à seção do laboratório (<https://epineto.github.io/prompt-injection-dl-unisinos/>) e informa os valores esperados. Os valores de referência vêm de `reference/resultados/` (gerados por `reference/python/trabalho2_dl.py`) e estão em `data/resultados.json`.

## Tolerâncias
O treino no navegador usa inicialização He e embaralhamento por mulberry32; o PyTorch usa outra inicialização e outro gerador. Por isso, **o que é determinístico coincide** (tokenização, TF-IDF, forward, gradientes, um passo de cada otimizador, modelo pré-treinado; ver `npm test`) e **o que depende de sorteio coincide estatisticamente**: esperam-se valores dentro do desvio entre sementes, não idênticos.

## Mapa de correspondência

| Item do trabalho | Onde no laboratório | Como obter | Valor esperado |
|---|---|---|---|
| Dados: treino 872 (436 + 436 traduções), validação 220 (110 + 110), testes EN/DE 116 e PT 116 (deepset); OOD neuralchemy, xTRam1 e BR-synthetic | `#resultados` → Dados | Abrir a seção | 324 e 82 injeções (37,2% e 37,3%); 60 de 116 nos dois testes (51,7%) |
| Vocabulário de 1.863 termos (só EN/DE: 983; só PT: 928); TF-IDF igual ao scikit-learn (dif. ≈ 1,7e-16) | `#testar` → O que o modelo vê; `#treinar` (seletor de línguas); `tests/texto.test.js` | `npm test` | Vocabulários idênticos aos do Python (hash SHA-256); vetores iguais (tolerância 1e-9) |
| Tokenização igual em Python (`\w`) e JavaScript (`\p{L}`), com acentos, cedilha e til | `tests/texto.test.js` | `npm test` | Hash de todos os textos (EN, DE, PT e BR-synthetic) idêntico |
| Gradientes à mão = autograd (erro ≈ 5,6e-17) | `#backprop`; `tests/mlp.test.js`, `tests/xor.test.js` | `npm test` | Gradientes iguais ao Python e às diferenças finitas |
| Contagem de parâmetros V·H+H+H+1 = 59.681 (V = 1.863, H = 32) | `#treinar` → Parâmetros | Alterar H ou a língua de treino | 59.681 para H = 32 (bilíngue); 3.731 para H = 2; 31.521 não aparece mais (era o vocabulário só EN/DE) |
| Modelo final (Adam, H = 32, taxa 0,001, semente 0, treino bilíngue): EN/DE TP 49, FP 1, FN 11, TN 55 (F1 0,891); PT TP 55, FP 0, FN 5, TN 56 (F1 0,957) | `#testar` → matrizes EN/DE e PT | Limiar 0,50 | Idêntico (modelo pré-treinado exportado e reconferido no teste) |
| OOD (semente 0, conjunto completo): neuralchemy F1 0,750; xTRam1 F1 0,496; BR-synthetic F1 0,571 | `#testar` (linha "Python, conjunto completo"); matrizes das amostras (300, 300 e 200 textos) | Limiar 0,50 | Amostras: F1 próximo (0,757; 0,492; 0,571); no BR-synthetic, tudo é classificado como injeção |
| Escolha da taxa (grade, 3 sementes): SGD 1; Momentum 0,1; RMSProp 0,01; Adam 0,001 (Adam 0,01 empata em perda: 0,1898 contra 0,1889) | `#resultados` → Otimizadores (grade); `#otimizadores` | Marcar "Adam com lr 0,01" e **Comparar** | Adam 0,01 chega ao mínimo em poucas épocas (3,7 na média de 3 sementes) |
| Figura 1: otimizadores, melhor época (RMSProp 3,8; SGD 19,4; Momentum 16,0; Adam 30,4) | `#otimizadores` | Multiplicador ×1, semente 0, **Comparar** | RMSProp por volta de 4 épocas; SGD e Momentum por volta de 11 a 20; Adam por volta de 31; referência de 5 sementes na tabela inferior |
| Taxa grande/pequena demais (Aula 4) | `#otimizadores` | Multiplicador ×10 e ×0,1 | ×10: oscilação/divergência; ×0,1: convergência lenta |
| F1 de teste bilíngue por otimizador (SGD 0,851; Momentum 0,842; RMSProp 0,894; Adam 0,921) e por língua (EN/DE 0,785 a 0,883; PT 0,905 a 0,956) | `#resultados` → Otimizadores | Abrir a seção | Valores acima; no navegador, 1 semente: próximos |
| Hipótese H1: menor perda de validação do SGD/Momentum (0,157/0,159); razão de taxas SGD/Adam = 1.000 | `#resultados` → Otimizadores | Abrir a seção | "Menor perda" refutada; "mais rápidos" só em parte (RMSProp; Adam com taxa 0,01) |
| Figura 2: capacidade (H = 0, 2, 8, 32, 128, 512); regra de parcimônia | `#resultados` → Capacidade; `#treinar` | Em `#treinar`, variar H | F1 bilíngue 0,755 (H = 0), 0,864, 0,926, 0,921, 0,925 e 0,919; perda de validação 0,477 a 0,185; H = 32 pela regra de parcimônia (declarada como desvio do protocolo) |
| Representação (BoW 0,901; TF-IDF 0,921; uni+bigramas 0,883) | `#resultados` → Representação | Abrir a seção | Valores acima; sem controle no navegador |
| Figura 4 e hipótese H4: línguas de treino × teste | `#treinar` (seletor "Línguas de treino"); `#resultados` → Línguas | Treinar nas três opções e comparar os testes EN/DE e PT | Python: só EN/DE → PT 0,677; só PT → EN/DE 0,375; bilíngue → EN/DE 0,883 e PT 0,956; BR-synthetic 0,571 nos três |
| Figura 3: generalização × DistilBERT/BERT-base (Trabalho 01, só EN/DE) | `#resultados` → Generalização | Abrir a seção | MLP bilíngue: EN/DE 0,883; PT 0,956; neuralchemy 0,747; xTRam1 0,498; BR-synthetic 0,571 (AUC 0,642); logística bilíngue: 0,849; 0,909; 0,690; 0,507; 0,571 |
| Custo: 59.681 parâmetros, ≈ 0,15 ms por texto, ≈ 340× mais rápido que o DistilBERT | `#resultados` → Custo | Abrir a seção | 0,15 ms contra 52,2 ms (p50); o tempo varia por execução |
| Termos mais influentes por oclusão ("gere", "context", "digo", "imagine"…) | `#resultados` → Termos; `#testar` (destaque) | Colar um texto com esses termos | Tabela do trabalho; no destaque, a cor é a queda de probabilidade |
| Passo de gradiente descendente reduz a perda | `#backprop` | **Passo** | A perda média cai a cada passo com taxa adequada |

## Atalho de 1 minuto
1. `#testar`: exemplo "Injeção clássica (EN)" → probabilidade ≈ 99,4% (em português, "Injeção clássica (PT)": 99,8%).
2. `#treinar`: **Treinar** com os valores padrão (bilíngue, leva segundos) → F1 bilíngue ≈ 0,92; depois "Só EN/DE" e "Só PT" para ver a hipótese H4.
3. `#otimizadores`: **Comparar**, com e sem "Adam com lr 0,01" → o RMSProp (e o Adam com taxa 0,01) chegam ao mínimo em poucas épocas; não têm perda menor.
4. `#backprop`: **Treinar** → a perda cai e a fronteira separa o XOR.

## Dados bilíngues e a tradução
`data/deepset_pt.json` é a tradução alinhada para o português do Brasil do deepset/prompt-injections (546 textos de treino e 116 de teste), usada por `reference/python/injecao.py` e pelo notebook (URL raw do repositório, `https://raw.githubusercontent.com/epineto/prompt-injection-dl-unisinos/main/data/deepset_pt.json`). Foi feita por LLM e verificada por amostragem (cerca de 24 correções): **não é fala nativa**. É obra derivada do deepset (Apache-2.0), com atribuição. Licenças: deepset e neuralchemy sob Apache-2.0; xTRam1 sem licença declarada; BR-synthetic (`ai-safety-brazil/prompt-injection-br-synthetic`) sob CC-BY-4.0, com atribuição.

## Regerar dados e referências
```bash
# requer torch, scikit-learn, datasets e os quatro conjuntos no cache do Hugging Face (HF_HUB_OFFLINE=1 usa o cache)
python reference/python/gen_dados.py        # data/*.json (re-treina o modelo pré-treinado e confere as matrizes de confusão da semente 0)
python reference/python/gen_reference.py    # tests/reference.json
npm test
```

## Notebook
`notebook/Trabalho2-DL-Executado.ipynb` é o notebook executado do trabalho (mesmos números de `reference/resultados/`); `reference/python/trabalho2_dl.py` é o mesmo código em formato de script.
