import test from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, permutacao } from '../src/core/rng.js';
import { metricas, auc, confusao } from '../src/core/metricas.js';
import { lerNumero, fmtNum } from '../src/core/numeros.js';
import { ref } from './util.js';

test('mulberry32 reproduz a implementação independente em Python', () => {
  const r = mulberry32(ref.rng.semente);
  for (const v of ref.rng.valores) assert.ok(Math.abs(r() - v) < 1e-15);
});

test('permutacao é uma permutação válida, determinística por semente', () => {
  const p = permutacao(100, mulberry32(3)), q = permutacao(100, mulberry32(3)), s = permutacao(100, mulberry32(4));
  assert.deepEqual(Array.from(p), Array.from(q));
  assert.notDeepEqual(Array.from(p), Array.from(s));
  assert.deepEqual(Array.from(p).sort((a, b) => a - b), Array.from({ length: 100 }, (_, i) => i));
});

test('métricas, AUC (com empates) e BCE coincidem com o Python/sklearn', () => {
  const { y, p, limiar, esperado } = ref.metricas;
  const m = metricas(y, p, limiar);
  for (const k of ['tp', 'fp', 'fn', 'tn']) assert.equal(m[k], esperado[k]);
  for (const k of ['acc', 'prec', 'rec', 'f1', 'auc', 'fpr']) assert.ok(Math.abs(m[k] - esperado[k]) < 1e-12, k);
});

test('métricas: casos simples', () => {
  assert.equal(auc([0, 1], [0.1, 0.9]), 1);
  assert.equal(auc([0, 1], [0.9, 0.1]), 0);
  assert.equal(auc([0, 1], [0.5, 0.5]), 0.5);
  assert.deepEqual(confusao([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], 0.5), { tp: 1, fp: 1, fn: 1, tn: 1 });
});

test('leitura de números com vírgula ou ponto', () => {
  assert.equal(lerNumero('0,01'), 0.01);
  assert.equal(lerNumero('0.01'), 0.01);
  assert.equal(lerNumero(' 1,5 '), 1.5);
  assert.equal(lerNumero('1.234,5'), 1234.5);
  assert.equal(lerNumero('1,234.5'), 1234.5);
  assert.equal(lerNumero('1e-3'), 0.001);
  assert.ok(Number.isNaN(lerNumero('abc')));
  assert.ok(Number.isNaN(lerNumero('')));
  assert.equal(fmtNum(0.873, 3, 'pt'), '0,873');
  assert.equal(fmtNum(0.873, 3, 'en'), '0.873');
});
