import test from 'node:test';
import assert from 'node:assert/strict';
import { criarVetorizador, vetorizar } from '../src/core/texto.js';
import { forward, prever, nParametros } from '../src/core/mlp.js';
import { prepararDados, modeloDeJson } from '../src/core/dados.js';
import { metricas } from '../src/core/metricas.js';
import { treinar, avaliarConjuntos } from '../src/core/treino.js';
import { ref, vocab, modelo, dataset, resultados } from './util.js';

test('modelo pré-treinado exportado reproduz as probabilidades do Python em ≥ 50 textos (1e-9)', () => {
  const vet = criarVetorizador(vocab), P = modeloDeJson(modelo);
  assert.ok(ref.pretreinado.textos.length >= 50);
  ref.pretreinado.textos.forEach((t, i) => assert.ok(Math.abs(forward(P, vetorizar(t, vet)).p - ref.pretreinado.p[i]) < 1e-9, `texto ${i}`));
});

test('modelo pré-treinado: matriz de confusão no teste deepset = resultados do trabalho (50/1/10/55)', () => {
  const D = prepararDados(dataset, vocab), P = modeloDeJson(modelo);
  const m = metricas(D.yte, prever(P, D.Xte));
  assert.deepEqual([m.tp, m.fp, m.fn, m.tn], [50, 1, 10, 55]);
  const s0 = resultados.final_semente0.find((r) => r.conjunto.startsWith('deepset'));
  assert.ok(Math.abs(m.f1 - s0.f1) < 1e-3, `F1 JS ${m.f1} vs Python ${s0.f1}`);
  assert.ok(Math.abs(m.auc - s0.auc) < 2e-3);
});

test('integridade dos dados exportados', () => {
  assert.deepEqual([dataset.treino.y.length, dataset.validacao.y.length, dataset.teste.y.length], [436, 110, 116]);
  assert.equal(dataset.neuralchemy.y.length, 300);
  assert.equal(dataset.xtram1.y.length, 300);
  for (const k of ['treino', 'validacao', 'teste', 'neuralchemy', 'xtram1']) {
    assert.equal(dataset[k].texto.length, dataset[k].y.length);
    assert.ok(dataset[k].y.every((v) => v === 0 || v === 1));
  }
  assert.equal(vocab.termos.length, 983);
  assert.equal(vocab.idf.length, 983);
  assert.equal(modelo.W1.length, modelo.V * modelo.H);
  assert.equal(modelo.W1.length + modelo.b1.length + modelo.W2.length + modelo.b2.length, nParametros(983, 32));
  assert.equal(resultados.info.parametros_mlp, 31521);
  assert.equal(resultados.info.vocab, 983);
  assert.ok(Object.values(dataset.fontes).every((f) => f.licenca && f.url));
  // proporção de injeção dos dados (tab_dados do trabalho)
  const inj = (k) => dataset[k].y.reduce((s, v) => s + v, 0);
  assert.deepEqual([inj('treino'), inj('validacao'), inj('teste')], [162, 41, 60]);
  // divisão estratificada sem sobreposição entre treino e validação
  const tr = new Set(dataset.treino.texto);
  assert.ok(dataset.validacao.texto.filter((t) => tr.has(t)).length <= 3, 'duplicatas exatas são raras');
});

test('resultados.json traz os números do trabalho (conferência por amostragem)', () => {
  const r = resultados;
  assert.equal(r.otimizadores.Adam.lr, 0.01);
  assert.equal(r.otimizadores.SGD.lr, 1);
  assert.ok(Math.abs(r.otimizadores.Adam.f1['deepset (teste)'][0] - 0.873) < 1e-3);
  assert.ok(Math.abs(r.otimizadores.SGD.f1['deepset (teste)'][0] - 0.842) < 1e-3);
  assert.ok(Math.abs(r.otimizadores.Adam.melhor_epoca - 5.2) < 1);
  assert.ok(Math.abs(r.otimizadores.SGD.melhor_epoca - 23.2) < 1e-6);
  assert.equal(r.capacidade.length, 6);
  assert.equal(r.representacao.length, 3);
  assert.equal(r.curvas_medias.Adam.val.length, 40);
  assert.deepEqual(['tp', 'fp', 'fn', 'tn'].map((k) => r.final_semente0[0][k]), [50, 1, 10, 55]);
  assert.ok(r.custo.distilbert.p50_ms > 50);
});

test('treino em JS: determinístico por semente, perda de treino cai e restaura a melhor época', async () => {
  const D = prepararDados(dataset, vocab);
  const cfg = { otimizador: 'Adam', lr: 0.01, H: 32, batch: 32, epocas: 12, semente: 0 };
  const a = await treinar(cfg, D), b = await treinar(cfg, D);
  assert.deepEqual(a.curva.val, b.curva.val);
  assert.ok(a.curva.treino[11] < a.curva.treino[0]);
  assert.equal(a.curva.val[a.melhorEp - 1], Math.min(...a.curva.val));
  const m = avaliarConjuntos(a.melhor, D);
  assert.ok(m.deepset.f1 > 0.7, 'F1 no teste ' + m.deepset.f1);
  const c = await treinar({ ...cfg, semente: 1 }, D);
  assert.notDeepEqual(a.curva.val, c.curva.val);
});

test('parar() interrompe o treino e devolve o melhor modelo até então', async () => {
  const D = prepararDados(dataset, vocab);
  let n = 0;
  const r = await treinar({ otimizador: 'SGD', lr: 1, H: 8, batch: 32, epocas: 40, semente: 0 }, D, { parar: () => ++n >= 3 });
  assert.equal(r.epocasFeitas, 3);
});
