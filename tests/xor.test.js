import test from 'node:test';
import assert from 'node:assert/strict';
import { iniciar, clonar, forward, perdaMedia, gradienteAmostra, gradienteLote, atualizar, XOR_X, XOR_Y } from '../src/core/xor.js';
import { ref } from './util.js';

const Wref = () => ({ W1: ref.xor.W1, b1: ref.xor.b1.slice(), W2: ref.xor.W2.slice(), b2: ref.xor.b2[0] });

test('XOR 2-2-1: forward, perda e gradientes coincidem com o Python', () => {
  const W = Wref();
  XOR_X.forEach((x, i) => assert.ok(Math.abs(forward(W, x).p - ref.xor.p[i]) < 1e-12));
  assert.ok(Math.abs(perdaMedia(W) - ref.xor.perda) < 1e-12);
  const g = gradienteLote(W);
  for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) assert.ok(Math.abs(g.W1[a][b] - ref.xor.g_W1[a][b]) < 1e-12);
  for (let j = 0; j < 2; j++) { assert.ok(Math.abs(g.b1[j] - ref.xor.g_b1[j]) < 1e-12); assert.ok(Math.abs(g.W2[j] - ref.xor.g_W2[j]) < 1e-12); }
  assert.ok(Math.abs(g.b2 - ref.xor.g_b2[0]) < 1e-12);
});

test('XOR 2-2-1: gradientes por diferenças finitas (amostra e lote)', () => {
  const h = 1e-6;
  const W = iniciar(5);
  const partes = [['W1', 0, 0], ['W1', 0, 1], ['W1', 1, 0], ['W1', 1, 1], ['b1', 0], ['b1', 1], ['W2', 0], ['W2', 1], ['b2']];
  const get = (p) => (p.length === 3 ? W[p[0]][p[1]][p[2]] : p.length === 2 ? W[p[0]][p[1]] : W[p[0]]);
  const set = (p, v) => { if (p.length === 3) W[p[0]][p[1]][p[2]] = v; else if (p.length === 2) W[p[0]][p[1]] = v; else W[p[0]] = v; };
  const ana = (g, p) => (p.length === 3 ? g[p[0]][p[1]][p[2]] : p.length === 2 ? g[p[0]][p[1]] : g[p[0]]);
  for (const ids of [[0], [1], [2], [3], [0, 1, 2, 3]]) {
    const g = ids.length === 1 ? gradienteAmostra(W, XOR_X[ids[0]], XOR_Y[ids[0]]).g : gradienteLote(W, ids);
    for (const p of partes) {
      const a = get(p);
      set(p, a + h); const lp = perdaMedia(W, ids);
      set(p, a - h); const lm = perdaMedia(W, ids);
      set(p, a);
      assert.ok(Math.abs((lp - lm) / (2 * h) - ana(g, p)) < 1e-7, `${p.join('.')} ids=${ids}`);
    }
  }
});

test('XOR 2-2-1: a perda cai e a rede aprende o XOR (amostra a amostra, semente 1, lr 0,5; e lote, semente 2)', () => {
  const W = iniciar(1);
  const l0 = perdaMedia(W);
  for (let e = 0; e < 1000; e++) for (let i = 0; i < 4; i++) atualizar(W, gradienteAmostra(W, XOR_X[i], XOR_Y[i]).g, 0.5);
  assert.ok(perdaMedia(W) < l0 && perdaMedia(W) < 0.05, 'perda final ' + perdaMedia(W));
  XOR_X.forEach((x, i) => assert.equal(forward(W, x).p >= 0.5 ? 1 : 0, XOR_Y[i]));
  const B = iniciar(2);
  for (let e = 0; e < 3000; e++) atualizar(B, gradienteLote(B), 0.5);
  assert.ok(perdaMedia(B) < 0.05, 'perda final (lote) ' + perdaMedia(B));
});

test('XOR 2-2-1: algumas sementes ficam presas em um platô (3 de 4 acertos), o que o laboratório mostra', () => {
  const W = iniciar(3);
  for (let e = 0; e < 3000; e++) for (let i = 0; i < 4; i++) atualizar(W, gradienteAmostra(W, XOR_X[i], XOR_Y[i]).g, 0.5);
  assert.ok(perdaMedia(W) > 0.2);
});

test('XOR: clonar não compartilha estado', () => {
  const a = iniciar(2), b = clonar(a);
  b.W1[0][0] = 99;
  assert.notEqual(a.W1[0][0], 99);
});
