# Reprodução dos resultados do relatório no laboratório

[English version](REPRODUCTION.md) · [README](../README.md)

Este guia mapeia cada tabela, figura e número do Trabalho 02 de Deep Learning à seção do laboratório (<https://epineto.github.io/prompt-injection-dl-unisinos/>) e informa os valores esperados. Os valores de referência vêm de `reference/resultados/` (gerados por `reference/python/trabalho2_dl.py`) e estão em `data/resultados.json`.

## Tolerâncias
O treino no navegador usa inicialização He e embaralhamento por mulberry32; o PyTorch usa outra inicialização e outro gerador. Por isso, **o que é determinístico coincide** (tokenização, TF-IDF, forward, gradientes, um passo de cada otimizador, modelo pré-treinado; ver `npm test`) e **o que depende de sorteio coincide estatisticamente**: esperam-se valores dentro do desvio entre sementes, não idênticos.

## Mapa de correspondência

| Item do trabalho | Onde no laboratório | Como obter | Valor esperado |
|---|---|---|---|
| Dados: treino 432 (216 + 216 traduções), validação 106 (53 + 53), testes EN 61 e PT 61 (deepset, só os textos em inglês); OOD neuralchemy (942), xTRam1 (1.000) e BR-synthetic (1.000) | `#resultados` → Dados | Abrir a seção | 148 e 36 injeções (34,3% e 34,0%); 33 de 61 nos dois testes (54,1%). **Testes de 61 textos: resultados indicativos** |
| Observação do deepset: o original mistura inglês e alemão (e poucas outras línguas); alemão e demais línguas foram **descartados, não convertidos**; restam 330 textos em inglês (269 de treino e 61 de teste) | `#resultados` → Dados; `#sobre` | Abrir a seção | Texto da observação; `data/deepset_pt.json` traz `indices_en` (269 e 61 posições) |
| Vocabulário de 788 termos (só EN: 375; só PT: 384); TF-IDF igual ao scikit-learn (dif. ≈ 1,7e-16) | `#testar` → O que o modelo vê; `#treinar` (seletor de línguas); `tests/texto.test.js` | `npm test` | Vocabulários idênticos aos do Python (hash SHA-256); vetores iguais (tolerância 1e-9) |
| Tokenização igual em Python (`\w`) e JavaScript (`\p{L}`), com acentos, cedilha e til | `tests/texto.test.js` | `npm test` | Hash de todos os textos (EN, PT e BR-synthetic) idêntico |
| Gradientes à mão = autograd (erro ≈ 2,8e-17) | `#backprop`; `tests/mlp.test.js`, `tests/xor.test.js` | `npm test` | Gradientes iguais ao Python e às diferenças finitas |
| Contagem de parâmetros V·H+H+H+1 = 6.321 (V = 788, H = 8) | `#treinar` → Parâmetros | Alterar H ou a língua de treino | 6.321 para H = 8 (bilíngue); 1.581 para H = 2; 25.281 para H = 32 |
| Modelo final (Adam, H = 8, taxa 0,001, semente 0, treino bilíngue): EN TP 25, FP 1, FN 8, TN 27 (F1 0,847); PT TP 27, FP 0, FN 6, TN 28 (F1 0,900) | `#testar` → matrizes EN e PT | Limiar 0,50 | Idêntico (modelo pré-treinado exportado e reconferido no teste) |
| OOD (semente 0, conjunto completo): neuralchemy F1 0,648; xTRam1 F1 0,571; BR-synthetic F1 0,568 | `#testar` (linha "Python, conjunto completo"); matrizes das amostras (300, 300 e 200 textos) | Limiar 0,50 | Amostras: F1 próximo (0,660; 0,547; 0,554); no BR-synthetic, quase tudo é classificado como injeção |
| Escolha da taxa (grade, 3 sementes): SGD 1; Momentum 0,1; RMSProp 0,001; Adam 0,001. Com lr 0,01: Adam melhor época 3,0 e perda 0,395; RMSProp 2,0 e 0,381 | `#resultados` → Otimizadores (grade); `#otimizadores` | Marcar "Adam e RMSProp com lr 0,01" e **Comparar** | Convergem em 2 a 3 épocas, mas com perda de validação maior (Adam 0,395 contra 0,362 com 0,001) |
| Figura 1: otimizadores, melhor época (SGD 14,0; Momentum 13,6; RMSProp 24,0; Adam 24,4) | `#otimizadores` | Multiplicador ×1, semente 0, **Comparar** | SGD e Momentum por volta de 12 a 14 épocas; RMSProp e Adam por volta de 24 (na semente 0 podem ir até 40); referência de 5 sementes na tabela inferior |
| Taxa grande/pequena demais (Aula 4) | `#otimizadores` | Multiplicador ×10 e ×0,1 | ×10: oscilação/divergência; ×0,1: convergência lenta |
| F1 de teste bilíngue (122) por otimizador (SGD 0,849; Momentum 0,864; RMSProp 0,883; Adam 0,893) e por língua (EN 0,806 a 0,867; PT 0,889 a 0,918) | `#resultados` → Otimizadores | Abrir a seção | Valores acima; no navegador, 1 semente: próximos |
| **Hipótese H1 (adaptativos convergem em menos épocas e com perda menor): REFUTADA** neste conjunto pequeno. Menor perda de validação: Momentum 0,309 e SGD 0,326 (RMSProp 0,380; Adam 0,363); razão de taxas SGD/Adam = 1.000 (confirmada) | `#resultados` → Otimizadores; `#otimizadores` (mensagem-chave) | Abrir a seção | SGD e Momentum chegam primeiro e com menor perda; adaptativos com F1 um pouco maior, dentro de 1 a 2 desvios |
| Figura 2: capacidade (H = 0, 2, 8, 32, 128, 512); regra de parcimônia | `#resultados` → Capacidade; `#treinar` | Em `#treinar`, variar H | F1 bilíngue 0,590 (H = 0, não convergiu), 0,873, 0,885, 0,893, 0,887 e 0,878; perda de validação 0,556; 0,501; 0,370; 0,363; 0,374; 0,385; H = 8 pela regra de parcimônia (menor H com perda até 5% acima da mínima; adotada depois de ver a curva, desvio declarado do protocolo; mínimo estrito em H = 32) |
| Logística (scikit-learn, C = 100) × MLP bilíngue, F1 (AUC) | `#resultados` → Capacidade e Generalização | Abrir a seção | EN 0,828 (0,955) × 0,859 (0,942); PT 0,881 (0,991) × 0,911 (0,998); bilíngue 0,855 × 0,885: ganho de cerca de 3 p.p. |
| Representação (BoW 0,880; TF-IDF 0,885; uni+bigramas 0,868) | `#resultados` → Representação | Abrir a seção | Valores acima (dentro do ruído); sem controle no navegador |
| Figura 4 e hipótese H4: línguas de treino × teste | `#treinar` (seletor "Línguas de treino"); `#resultados` → Línguas | Treinar nas três opções e comparar os testes EN e PT | Python: só EN → PT 0,268 (AUC 0,510); só PT → EN 0,364; bilíngue → EN 0,859 e PT 0,911; BR-synthetic ≈ 0,57 nos três. **H4 confirmada** |
| Figura 3: generalização × DistilBERT/BERT-base (Trabalho 01, só inglês: 61 textos do teste) | `#resultados` → Generalização | Abrir a seção | MLP bilíngue: EN 0,859; PT 0,911; neuralchemy 0,655; xTRam1 0,570; BR-synthetic 0,571 (AUC 0,581); logística: 0,828; 0,881; 0,653; 0,543; 0,571. DistilBERT 0,941 e BERT-base 0,979 no EN |
| Custo: 6.321 parâmetros (≈ 25 KB em float32), ≈ 0,18 ms por texto, ≈ 291 vezes mais rápido que o DistilBERT (52,2 ms ÷ 0,179 ms) | `#resultados` → Custo | Abrir a seção | 0,18 ms contra 52,2 ms (p50); o tempo varia por execução e por hardware |
| Termos mais influentes por oclusão ("diga"/"say", "antes"/"before", "você"/"you", "ignore", "tudo"; artefatos "2023", "n") | `#resultados` → Termos; `#testar` (destaque) | Colar um texto com esses termos | Tabela do trabalho; no destaque, a cor é a queda de probabilidade |
| Passo de gradiente descendente reduz a perda | `#backprop` | **Passo** | A perda média cai a cada passo com taxa adequada |

## Atalho de 1 minuto
1. `#testar`: exemplo "Injeção clássica (EN)" → probabilidade ≈ 91,5% (em português, "Injeção clássica (PT)": 90,7%; "Seguro (EN)": 8,1%).
2. `#treinar`: **Treinar** com os valores padrão (bilíngue, H = 8, leva segundos) → F1 bilíngue próximo de 0,88; depois "Só EN" e "Só PT" para ver a hipótese H4.
3. `#otimizadores`: **Comparar**, com e sem "Adam e RMSProp com lr 0,01" → SGD e Momentum atingem o mínimo primeiro e com menor perda (H1 refutada); com lr 0,01 os adaptativos chegam em 2 a 3 épocas, mas com perda pior.
4. `#backprop`: **Treinar** → a perda cai e a fronteira separa o XOR.

## Dados bilíngues e a tradução
**Observação sobre o deepset:** o conjunto original mistura inglês e alemão (e poucas outras línguas, como espanhol, croata e francês). Os textos em alemão e em outras línguas foram **descartados, não convertidos nem traduzidos** (detecção de língua, marcadores lexicais e revisão manual); restam 330 textos em inglês (269 de treino e 61 de teste).

`data/deepset_pt.json` é a tradução alinhada para o português do Brasil do deepset/prompt-injections completo (546 textos de treino e 116 de teste), com `indices_en` (as 269 e 61 posições dos textos em inglês efetivamente usados), lida por `reference/python/injecao.py` e pelo notebook (URL raw do repositório, `https://raw.githubusercontent.com/epineto/prompt-injection-dl-unisinos/main/data/deepset_pt.json`). Foi feita por modelo de linguagem (LLM) e verificada por amostragem (cerca de 24 correções): **não é fala nativa**. É obra derivada do deepset (Apache-2.0), com atribuição. Licenças: deepset e neuralchemy sob Apache-2.0; xTRam1 sem licença declarada; BR-synthetic (`ai-safety-brazil/prompt-injection-br-synthetic`) sob CC-BY-4.0, com atribuição.

## Regerar dados e referências
```bash
# requer torch, scikit-learn, datasets e os quatro conjuntos no cache do Hugging Face (HF_HUB_OFFLINE=1 usa o cache)
python reference/python/gen_dados.py        # data/*.json (re-treina o modelo pré-treinado e confere as matrizes de confusão da semente 0)
python reference/python/gen_reference.py    # tests/reference.json
npm test
```

## Notebook
`notebook/Trabalho2-DL-Executado.ipynb` é o notebook executado do trabalho (mesmos números de `reference/resultados/`); `reference/python/trabalho2_dl.py` é o mesmo código em formato de script.
