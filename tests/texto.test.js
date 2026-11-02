import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { tokenizar, ajustar, criarVetorizador, vetorizar, tokensComPosicao } from '../src/core/texto.js';
import { ref, dataset, vocab } from './util.js';

const sha = (termos) => createHash('sha256').update(termos.join('\n'), 'utf8').digest('hex');

const TODOS = ['treino', 'validacao', 'teste', 'teste_pt', 'neuralchemy', 'xtram1', 'br'].flatMap((k) => dataset[k].texto);

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
  assert.equal(v.termos.length, 1863);
  assert.deepEqual(v.termos, vocab.termos);
  assert.equal(v.n_documentos, 872);
  for (let i = 0; i < v.idf.length; i++) assert.ok(Math.abs(v.idf[i] - vocab.idf[i]) < 1e-9);
});

test('vocabulários por língua de treino (só EN/DE: 983; só PT: 928) coincidem com o Python', () => {
  const n = dataset.treino.n_orig;
  assert.equal(n, 436);
  for (const [nome, textos] of [['orig', dataset.treino.texto.slice(0, n)], ['pt', dataset.treino.texto.slice(n)]]) {
    const v = ajustar(textos, { maxVocab: 2000, minDf: 2 }), p = ref.vocab_por_idioma[nome];
    assert.equal(v.termos.length, p.n_termos);
    assert.equal(textos.length, p.n_textos);
    assert.equal(sha(v.termos), p.sha256, nome);
    assert.ok(Math.abs(v.idf.reduce((a, b) => a + b, 0) - p.idf_soma) < 1e-9);
  }
  assert.equal(ref.vocab_por_idioma.orig.n_termos, 983);
  assert.equal(ref.vocab_por_idioma.pt.n_termos, 928);
});

test('português com acentos, cedilha e til: JS (\\p{L}) = Python (\\w) nos casos de borda e em todos os textos PT', () => {
  assert.deepEqual(tokenizar('Ação não é coração: AÇÃO, ÚLTIMA, pingüim, órgão'), ['ação', 'não', 'é', 'coração', 'ação', 'última', 'pingüim', 'órgão']);
  const PT = [...dataset.teste_pt.texto, ...dataset.br.texto, ...dataset.treino.texto.slice(dataset.treino.n_orig)];
  const comAcento = PT.filter((x) => /[ãõçáéíóúâêôà]/i.test(x));
  assert.ok(comAcento.length > 100, 'há muitos textos PT com acentos');
  // os casos com acento estão no hash global; aqui só se confere que nenhuma letra acentuada é descartada (sem fragmentos soltos)
  for (const x of comAcento.slice(0, 200)) assert.ok(tokenizar(x).join('').length >= x.replace(/[^\p{L}\p{N}_]/gu, '').length);
  for (const r of ref.tfidf.filter((q) => /[ãõçáéíóúâêôà]/i.test(q.texto))) assert.deepEqual(tokenizar(r.texto), r.tokens);
});

test('tokensComPosicao devolve posições consistentes com tokenizar', () => {
  for (const t of dataset.teste.texto.slice(0, 40)) {
    const { base, tokens } = tokensComPosicao(t);
    assert.deepEqual(tokens.map((x) => x.token), tokenizar(t));
    for (const x of tokens) assert.equal(base.slice(x.ini, x.fim).toLowerCase(), x.token);
  }
});
