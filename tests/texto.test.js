import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { tokenizar, ajustar, criarVetorizador, vetorizar, tokensComPosicao } from '../src/core/texto.js';
import { ref, dataset, vocab } from './util.js';

const TODOS = ['treino', 'validacao', 'teste', 'neuralchemy', 'xtram1'].flatMap((k) => dataset[k].texto);

test('tokenização coincide com o Python em TODOS os textos (hash SHA-256 e contagens)', () => {
  const h = createHash('sha256');
  let n = 0;
  for (const t of TODOS) { const tk = tokenizar(t); n += tk.length; h.update(tk.join(' ') + '\n'); }
  assert.equal(TODOS.length, ref.tokenizacao_todos.n_textos);
  assert.equal(n, ref.tokenizacao_todos.n_tokens);
  assert.equal(h.digest('hex'), ref.tokenizacao_todos.sha256);
});

test('tokenização: casos de borda (Unicode, underscore, vazio)', () => {
  assert.deepEqual(tokenizar('Größe: ÄÖÜ ß straße 123_abc'), ['größe', 'äöü', 'ß', 'straße', '123_abc']);
  assert.deepEqual(tokenizar(''), []);
  assert.deepEqual(tokenizar('a b c', true), ['a', 'b', 'c', 'a b', 'b c']);
});

test('TF-IDF: vetores esparsos iguais ao Python (≥ 20 textos, tolerância 1e-9)', () => {
  assert.ok(ref.tfidf.length >= 20);
  const vet = criarVetorizador(vocab);
  for (const r of ref.tfidf) {
    assert.deepEqual(tokenizar(r.texto), r.tokens, r.texto.slice(0, 40));
    const x = vetorizar(r.texto, vet);
    assert.deepEqual(Array.from(x.idx), r.x.idx);
    for (let k = 0; k < x.val.length; k++) assert.ok(Math.abs(x.val[k] - r.x.val[k]) < 1e-9);
    const norma = x.val.reduce((s, v) => s + v * v, 0);
    assert.ok(x.idx.length === 0 ? norma === 0 : Math.abs(norma - 1) < 1e-12, 'norma L2 unitária');
  }
});

test('vocabulário e idf reajustados no treino coincidem com data/vocab.json', () => {
  const v = ajustar(dataset.treino.texto, { maxVocab: 2000, minDf: 2 });
  assert.equal(v.termos.length, 983);
  assert.deepEqual(v.termos, vocab.termos);
  assert.equal(v.n_documentos, 436);
  for (let i = 0; i < v.idf.length; i++) assert.ok(Math.abs(v.idf[i] - vocab.idf[i]) < 1e-9);
});

test('tokensComPosicao devolve posições consistentes com tokenizar', () => {
  for (const t of dataset.teste.texto.slice(0, 40)) {
    const { base, tokens } = tokensComPosicao(t);
    assert.deepEqual(tokens.map((x) => x.token), tokenizar(t));
    for (const x of tokens) assert.equal(base.slice(x.ini, x.fim).toLowerCase(), x.token);
  }
});
