# Navegação do laboratório (roteiro para a apresentação)

Cada seção tem uma âncora fixa (hash) que pode ser aberta direto: `…/#intro`, `…/#testar`, etc. A barra do topo troca de seção; **PT/EN** troca o idioma.

| Âncora | O que é | O que mostrar em 20 s |
|---|---|---|
| `#intro` | Problema, pipeline texto → tokens → TF-IDF → MLP → probabilidade, números-chave | Ler as 3 linhas do problema e apontar o pipeline com o exemplo ao vivo (≈ 91% em inglês e em português) |
| `#testar` | Modelo pré-treinado em tempo real, oclusão por palavra, matrizes de confusão | Clicar em "Injeção clássica (EN)", "Injeção clássica (PT)" e "Seguro: receita (EN)" (duas línguas); mostrar a palavra colorida; mover o limiar; descer até as matrizes por língua e o BR-synthetic (quase tudo vira injeção) |
| `#treinar` | Treino do zero em Web Worker, curvas ao vivo, melhor época | **Treinar** com os padrões (bilíngue); apontar o ponto verde (parada antecipada) e o contador V·H+H+H+1 = 6.321 (H = 8); trocar "Línguas de treino" para "Só EN" e ver o F1 do teste PT cair (hipótese H4) |
| `#otimizadores` | SGD × Momentum × RMSProp × Adam, mesma semente | **Comparar**; mostrar que SGD e Momentum chegam ao mínimo antes (~12 a 14 épocas contra ~24) e com menor perda: a hipótese H1 foi refutada neste conjunto pequeno; marcar "Adam e RMSProp com lr 0,01" para ver os adaptativos chegarem em 2 a 3 épocas, mas com perda pior; depois ×10 para ver a instabilidade |
| `#backprop` | Rede 2-2-1 (XOR) com todos os números | **Passo** duas vezes (mostrar a regra da cadeia), depois **Treinar** e ver a fronteira se formar |
| `#resultados` | Tabelas e gráficos do relatório, custo, generalização, mapeamento relatório → laboratório | Mostrar a matriz língua × teste (H4), a Generalização (queda fora da distribuição; PT nativo sem generalização) e o Custo (~291 vezes mais rápido) e a observação do deepset (alemão descartado; testes de 61 textos: resultados indicativos) |
| `#sobre` | Como rodar, testes, aulas, limitações, licenças | Mostrar a tabela Aula → seção, o aviso (português = tradução automática) e a observação do deepset (alemão descartado, não convertido) e as licenças |

Dica: abrir o laboratório já na seção desejada pelo link com âncora (por exemplo, `https://epineto.github.io/prompt-injection-dl-unisinos/#otimizadores`).
