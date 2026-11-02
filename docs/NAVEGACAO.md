# Navegação do laboratório (roteiro para a apresentação)

Cada seção tem uma âncora fixa (hash) que pode ser aberta direto: `…/#intro`, `…/#testar`, etc. A barra do topo troca de seção; **PT/EN** troca o idioma.

| Âncora | O que é | O que mostrar em 20 s |
|---|---|---|
| `#intro` | Problema, pipeline texto → tokens → TF-IDF → MLP → probabilidade, números-chave | Ler as 3 linhas do problema e apontar o pipeline com o exemplo ao vivo (99,8%) |
| `#testar` | Modelo pré-treinado em tempo real, oclusão por palavra, matrizes de confusão | Clicar em "Injeção clássica (EN)", depois em "Seguro (EN)"; mostrar a palavra colorida; mover o limiar |
| `#treinar` | Treino do zero em Web Worker, curvas ao vivo, melhor época | **Treinar** com os padrões; apontar o ponto verde (parada antecipada) e o contador V·H+H+H+1 = 31.521 |
| `#otimizadores` | SGD × Momentum × RMSProp × Adam, mesma semente | **Comparar**; mostrar que adaptativos chegam ao mínimo em ~5 épocas contra ~23; depois ×10 para ver a instabilidade |
| `#backprop` | Rede 2-2-1 (XOR) com todos os números | **Passo** duas vezes (mostrar a regra da cadeia), depois **Treinar** e ver a fronteira se formar |
| `#resultados` | Tabelas e gráficos do relatório, custo, generalização, mapeamento relatório → laboratório | Descer até Generalização (queda fora da distribuição) e Custo (~340× mais rápido) |
| `#sobre` | Como rodar, testes, aulas, limitações, licenças | Mostrar a tabela Aula → seção e as limitações |

Dica: abrir o laboratório já na seção desejada pelo link com âncora (por exemplo, `https://epineto.github.io/prompt-injection-dl-unisinos/#otimizadores`).
