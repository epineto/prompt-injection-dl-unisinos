// Prepara as matrizes esparsas de todos os conjuntos a partir de dataset.json e vocab.json.
// treino: 'bil' (original EN + tradução PT, vocabulário de data/vocab.json), 'orig' (só EN) ou 'pt' (só PT);
// nos dois últimos o vocabulário é reajustado só nos textos escolhidos, como no experimento de línguas do Python.
import { criarVetorizador, vetorizar, ajustar } from './texto.js';

export const TREINOS = ['bil', 'orig', 'pt'];

// Nomes dos conjuntos de avaliação: en = teste deepset EN; pt = teste deepset traduzido; bil = os dois concatenados.
export const CONJUNTOS = ['en', 'pt', 'bil', 'neuralchemy', 'xtram1', 'br'];

function recortar(c, treino) {
  if (treino === 'bil') return c;
  const [a, b] = treino === 'orig' ? [0, c.n_orig] : [c.n_orig, c.texto.length];
  return { texto: c.texto.slice(a, b), y: c.y.slice(a, b) };
}

export function prepararDados(ds, vocab, modo = 'tfidf', treino = 'bil') {
  const tr = recortar(ds.treino, treino), va = recortar(ds.validacao, treino);
  const voc = treino === 'bil' ? vocab : ajustar(tr.texto, { maxVocab: vocab.max_vocab ?? 2000, minDf: vocab.min_df ?? 2 });
  const vet = criarVetorizador(voc);
  const v = (c) => ({ X: c.texto.map((t) => vetorizar(t, vet, modo)), y: Float64Array.from(c.y) });
  const a = v(tr), b = v(va), te = v(ds.teste), tp = v(ds.teste_pt), nc = v(ds.neuralchemy), xt = v(ds.xtram1), br = v(ds.br);
  return {
    V: vet.V, vet, treino,
    Xtr: a.X, ytr: a.y, Xva: b.X, yva: b.y,
    Xte: te.X, yte: te.y, Xtp: tp.X, ytp: tp.y, Xnc: nc.X, ync: nc.y, Xxt: xt.X, yxt: xt.y, Xbr: br.X, ybr: br.y,
  };
}

// Conjuntos de avaliação como lista [chave, X, y]; "bil" é o teste EN seguido do teste PT (122 textos).
export function conjuntosDeAvaliacao(d) {
  return [['en', d.Xte, d.yte], ['pt', d.Xtp, d.ytp], ['bil', [...d.Xte, ...d.Xtp], Float64Array.from([...d.yte, ...d.ytp])],
    ['neuralchemy', d.Xnc, d.ync], ['xtram1', d.Xxt, d.yxt], ['br', d.Xbr, d.ybr]];
}

export function modeloDeJson(m) {
  return { V: m.V, H: m.H, W1: Float64Array.from(m.W1), b1: Float64Array.from(m.b1), W2: Float64Array.from(m.W2), b2: Float64Array.from(m.b2) };
}
