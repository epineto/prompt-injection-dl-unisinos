# Navegação do laboratório (roteiro para a apresentação)

Cada seção tem uma âncora fixa (hash) que pode ser aberta direto: `…/#intro`, `…/#testar`, etc. A barra do topo troca de seção; **PT/EN** troca o idioma.

| Âncora | O que é | O que mostrar em 20 s |
|---|---|---|
| `#intro` | Problema, pipeline texto → tokens → TF-IDF → MLP → probabilidade, números-chave | Ler as 3 linhas do problema e apontar o pipeline com o exemplo ao vivo (99,8% em português) |
| `#testar` | Modelo pré-treinado em tempo real, oclusão por palavra, matrizes de confusão | Clicar em "Injeção clássica (EN)", "Injeção clássica (PT)" e "Seguro (DE)" (três línguas); mostrar a palavra colorida; mover o limiar; descer até as matrizes por língua e o BR-synthetic (tudo vira injeção) |
| `#treinar` | Treino do zero em Web Worker, curvas ao vivo, melhor época | **Treinar** com os padrões (bilíngue); apontar o ponto verde (parada antecipada) e o contador V·H+H+H+1 = 59.681; trocar "Línguas de treino" para "Só EN/DE" e ver o F1 do teste PT cair (hipótese H4) |
| `#otimizadores` | SGD × Momentum × RMSProp × Adam, mesma semente | **Comparar**; mostrar que o RMSProp chega ao mínimo em ~4 épocas contra ~11 a 20 de SGD/Momentum; marcar "Adam com lr 0,01" para ver o Adam acelerar; depois ×10 para ver a instabilidade |
| `#backprop` | Rede 2-2-1 (XOR) com todos os números | **Passo** duas vezes (mostrar a regra da cadeia), depois **Treinar** e ver a fronteira se formar |
| `#resultados` | Tabelas e gráficos do relatório, custo, generalização, mapeamento relatório → laboratório | Mostrar a matriz língua × teste (H4), a Generalização (queda fora da distribuição; PT nativo sem generalização) e o Custo (~339 vezes mais rápido) |
| `#sobre` | Como rodar, testes, aulas, limitações, licenças | Mostrar a tabela Aula → seção, o aviso (português = tradução automática) e as licenças |

Dica: abrir o laboratório já na seção desejada pelo link com âncora (por exemplo, `https://epineto.github.io/prompt-injection-dl-unisinos/#otimizadores`).
