// Prepara as matrizes esparsas de todos os conjuntos a partir de dataset.json e vocab.json.
import { criarVetorizador, vetorizar } from './texto.js';

export function prepararDados(ds, vocab, modo = 'tfidf') {
  const vet = criarVetorizador(vocab);
  const v = (c) => ({ X: c.texto.map((t) => vetorizar(t, vet, modo)), y: Float64Array.from(c.y) });
  const tr = v(ds.treino), va = v(ds.validacao), te = v(ds.teste), nc = v(ds.neuralchemy), xt = v(ds.xtram1);
  return { V: vet.V, vet, Xtr: tr.X, ytr: tr.y, Xva: va.X, yva: va.y, Xte: te.X, yte: te.y, Xnc: nc.X, ync: nc.y, Xxt: xt.X, yxt: xt.y };
}

export function modeloDeJson(m) {
  return { V: m.V, H: m.H, W1: Float64Array.from(m.W1), b1: Float64Array.from(m.b1), W2: Float64Array.from(m.W2), b2: Float64Array.from(m.b2) };
}
