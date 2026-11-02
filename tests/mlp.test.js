import test from 'node:test';
import assert from 'node:assert/strict';
import { criarVetorizador, vetorizar } from '../src/core/texto.js';
import { forward, gradientes, perdaMedia, bce, nParametros, inicializar, prever } from '../src/core/mlp.js';
import { criarOtimizador, NOMES } from '../src/core/otimizadores.js';
import { mulberry32 } from '../src/core/rng.js';
import { ref, vocab, maxAbs } from './util.js';

const vet = criarVetorizador(vocab);
const M = ref.mlp;
const X = M.textos.map((t) => vetorizar(t, vet));
const y = Float64Array.from(M.y);
const pesos = () => ({ V: vet.V, H: M.H, W1: Float64Array.from(M.W1), b1: Float64Array.from(M.b1), W2: Float64Array.from(M.W2), b2: Float64Array.from(M.b2) });

test('contagem de parâmetros V·H+H+H+1 (Aula 4)', () => {
  assert.equal(nParametros(1863, 32), 59681);
  assert.equal(nParametros(1863, 2), 3731);
  assert.equal(nParametros(2, 2) , 2 * 2 + 2 + 2 + 1);
});

test('forward e perda coincidem com o Python', () => {
  const P = pesos();
  const p = X.map((x) => forward(P, x).p);
  assert.ok(maxAbs(p, M.p) < 1e-12);
  assert.ok(Math.abs(bce(p, y) - M.perda) < 1e-12);
  assert.ok(Math.abs(perdaMedia(P, X, y) - M.perda) < 1e-12);
});

test('gradientes à mão coincidem com o Python (tolerância 1e-9)', () => {
  const P = pesos();
  const { g, perda } = gradientes(P, X, y);
  assert.ok(Math.abs(perda - M.perda) < 1e-12);
  assert.ok(maxAbs(g.W1, M.g_W1) < 1e-9);
  assert.ok(maxAbs(g.b1, M.g_b1) < 1e-9);
  assert.ok(maxAbs(g.W2, M.g_W2) < 1e-9);
  assert.ok(maxAbs(g.b2, M.g_b2) < 1e-9);
});

test('gradientes conferem com diferenças finitas centrais', () => {
  const P = pesos();
  const { g } = gradientes(P, X, y);
  const h = 1e-6;
  const verificar = (campo, i) => {
    const a = P[campo][i];
    P[campo][i] = a + h; const lp = perdaMedia(P, X, y);
    P[campo][i] = a - h; const lm = perdaMedia(P, X, y);
    P[campo][i] = a;
    const num = (lp - lm) / (2 * h);
    assert.ok(Math.abs(num - g[campo][i]) < 1e-6, `${campo}[${i}] num=${num} ana=${g[campo][i]}`);
  };
  for (let i = 0; i < P.H; i++) { verificar('b1', i); verificar('W2', i); }
  verificar('b2', 0);
  // alguns pesos de W1 com gradiente não nulo e outros nulos
  const idxs = new Set();
  for (const x of X) for (const j of x.idx) for (let hh = 0; hh < P.H; hh++) idxs.add(j * P.H + hh);
  [...idxs].slice(0, 25).forEach((i) => verificar('W1', i));
  verificar('W1', 0);
});

test('mini-lote: gradiente de um subconjunto = média daquelas amostras', () => {
  const P = pesos();
  const ids = Int32Array.from([1, 3, 5]);
  const { g } = gradientes(P, X, y, ids);
  const h = 1e-6, i = X[1].idx[0] * P.H;
  const f = () => perdaMedia(P, Array.from(ids, (k) => X[k]), Array.from(ids, (k) => y[k]));
  P.W1[i] += h; const lp = f(); P.W1[i] -= 2 * h; const lm = f(); P.W1[i] += h;
  assert.ok(Math.abs((lp - lm) / (2 * h) - g.W1[i]) < 1e-6);
});

test('os 4 otimizadores reproduzem 3 passos do PyTorch (float64, tolerância 1e-5)', () => {
  for (const nome of NOMES) {
    const R = ref.otimizadores[nome];
    const P = pesos();
    const opt = criarOtimizador(nome, R.lr, P);
    for (let passo = 1; passo <= 3; passo++) {
      const { g } = gradientes(P, X, y);
      opt.passo(P, g);
      if (R.passos[passo]) {
        const S = R.passos[passo];
        for (const c of ['W1', 'b1', 'W2', 'b2']) assert.ok(maxAbs(P[c], S[c]) < 1e-5, `${nome} passo ${passo} ${c}: ${maxAbs(P[c], S[c])}`);
      }
    }
  }
});

test('semântica: SGD+Momentum acumula buf=μ·buf+g; Adam dá passo de ≈ lr no 1º passo', () => {
  const P = { V: 1, H: 1, W1: Float64Array.of(0), b1: Float64Array.of(0), W2: Float64Array.of(0), b2: Float64Array.of(0) };
  const g = { W1: Float64Array.of(2), b1: Float64Array.of(0), W2: Float64Array.of(0), b2: Float64Array.of(0) };
  const m = criarOtimizador('SGD+Momentum', 0.1, P);
  m.passo(P, g); m.passo(P, g);
  assert.ok(Math.abs(P.W1[0] - (-0.1 * 2 - 0.1 * (0.9 * 2 + 2))) < 1e-12);
  const Q = { ...P, W1: Float64Array.of(0) };
  criarOtimizador('Adam', 0.01, Q).passo(Q, g);
  assert.ok(Math.abs(Q.W1[0] + 0.01) < 1e-6);
});

test('inicialização He tem desvio ≈ sqrt(2/V) e é determinística por semente', () => {
  const a = inicializar(1863, 32, mulberry32(0)), b = inicializar(1863, 32, mulberry32(0));
  assert.deepEqual(a.W1, b.W1);
  const m = a.W1.reduce((s, v) => s + v, 0) / a.W1.length;
  const sd = Math.sqrt(a.W1.reduce((s, v) => s + (v - m) ** 2, 0) / a.W1.length);
  assert.ok(Math.abs(sd - Math.sqrt(2 / 1863)) < 0.1 * Math.sqrt(2 / 1863));
  assert.ok(a.b1.every((v) => v === 0));
});

test('prever devolve probabilidades em (0,1)', () => {
  const P = pesos();
  for (const p of prever(P, X)) assert.ok(p > 0 && p < 1);
});
