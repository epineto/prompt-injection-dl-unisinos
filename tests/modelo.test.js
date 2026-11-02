import test from 'node:test';
import assert from 'node:assert/strict';
import { criarVetorizador, vetorizar } from '../src/core/texto.js';
import { forward, prever, nParametros } from '../src/core/mlp.js';
import { prepararDados, modeloDeJson } from '../src/core/dados.js';
import { metricas } from '../src/core/metricas.js';
import { treinar, avaliarConjuntos } from '../src/core/treino.js';
import { readFileSync } from 'node:fs';
import { ref, vocab, modelo, dataset, resultados } from './util.js';

test('modelo pré-treinado exportado reproduz as probabilidades do Python em ≥ 50 textos (1e-9)', () => {
  const vet = criarVetorizador(vocab), P = modeloDeJson(modelo);
  assert.ok(ref.pretreinado.textos.length >= 50);
  ref.pretreinado.textos.forEach((t, i) => assert.ok(Math.abs(forward(P, vetorizar(t, vet)).p - ref.pretreinado.p[i]) < 1e-9, `texto ${i}`));
});

test('modelo pré-treinado: matrizes de confusão (semente 0) = resultados do trabalho (EN 25/1/8/27; PT 27/0/6/28)', () => {
  const D = prepararDados(dataset, vocab), P = modeloDeJson(modelo);
  const en = metricas(D.yte, prever(P, D.Xte)), pt = metricas(D.ytp, prever(P, D.Xtp));
  assert.deepEqual([en.tp, en.fp, en.fn, en.tn], [25, 1, 8, 27]);
  assert.deepEqual([pt.tp, pt.fp, pt.fn, pt.tn], [27, 0, 6, 28]);
  assert.deepEqual(resultados.info.matriz_confusao_s0_en, { tp: 25, fp: 1, fn: 8, tn: 27 });
  assert.deepEqual(resultados.info.matriz_confusao_s0_pt, { tp: 27, fp: 0, fn: 6, tn: 28 });
  for (const [k, m] of [['deepset EN (teste)', en], ['deepset PT (teste)', pt]]) {
    const s0 = resultados.final_semente0.find((r) => r.conjunto === k);
    assert.ok(Math.abs(m.f1 - s0.f1) < 1e-3, `F1 JS ${m.f1} vs Python ${s0.f1}`);
    assert.ok(Math.abs(m.auc - s0.auc) < 2e-3);
  }
  // o conjunto bilíngue (122) e os fora da distribuição também batem com o Python (semente 0), nas amostras do laboratório
  const av = avaliarConjuntos(P, D), bil = resultados.final_semente0.find((r) => r.conjunto.startsWith('bilíngue'));
  assert.equal(av.bil.tp + av.bil.fp + av.bil.fn + av.bil.tn, 122);
  assert.ok(Math.abs(av.bil.f1 - bil.f1) < 1e-3);
  assert.equal(modelo.treino, 'bilíngue (EN + PT)');
  assert.deepEqual([modelo.otimizador, modelo.lr, modelo.H, modelo.semente], ['Adam', 0.001, 8, 0]);
});

test('integridade dos dados exportados', () => {
  assert.deepEqual([dataset.treino.y.length, dataset.validacao.y.length, dataset.teste.y.length, dataset.teste_pt.y.length], [432, 106, 61, 61]);
  assert.deepEqual([dataset.treino.n_orig, dataset.validacao.n_orig], [216, 53]);
  assert.equal(dataset.neuralchemy.y.length, 300);
  assert.equal(dataset.xtram1.y.length, 300);
  assert.equal(dataset.br.y.length, 200);
  for (const k of ['treino', 'validacao', 'teste', 'teste_pt', 'neuralchemy', 'xtram1', 'br']) {
    assert.equal(dataset[k].texto.length, dataset[k].y.length);
    assert.ok(dataset[k].y.every((v) => v === 0 || v === 1));
  }
  assert.equal(vocab.termos.length, 788);
  assert.equal(vocab.idf.length, 788);
  assert.equal(modelo.W1.length, modelo.V * modelo.H);
  assert.equal(modelo.W1.length + modelo.b1.length + modelo.W2.length + modelo.b2.length, nParametros(788, 8));
  assert.equal(nParametros(788, 8), 6321);
  assert.equal(resultados.info.parametros_mlp, 6321);
  assert.equal(resultados.info.vocab, 788);
  assert.ok(Object.values(dataset.fontes).every((f) => f.licenca && f.url));
  assert.match(dataset.fontes.br.licenca, /CC-BY-4\.0/);
  assert.match(dataset.fontes.deepset_pt.licenca, /Apache-2\.0/);
  // proporção de injeção (tab_dados do trabalho): 148/432 e 36/106 (34,3% e 34,0%), 33/61 (54,1%) nos dois testes
  const inj = (k) => dataset[k].y.reduce((s, v) => s + v, 0);
  assert.deepEqual([inj('treino'), inj('validacao'), inj('teste'), inj('teste_pt')], [148, 36, 33, 33]);
  // tradução alinhada: o rótulo da tradução é o do original (mesmo índice) e original e tradução ficam do mesmo lado
  for (const k of ['treino', 'validacao']) {
    const n = dataset[k].n_orig, y = dataset[k].y;
    assert.equal(y.length, 2 * n);
    assert.deepEqual(y.slice(0, n), y.slice(n));
  }
  assert.deepEqual(dataset.teste.y, dataset.teste_pt.y);
  // divisão sem sobreposição entre treino e validação
  // nenhum texto original (EN) de validação ou de teste aparece no treino; entre as traduções, textos curtos de originais
  // diferentes podem ter a mesma tradução (12 coincidências exatas entre treino e validação), por isso o limite
  const n0 = dataset.treino.n_orig, m0 = dataset.validacao.n_orig;
  const tr0 = new Set(dataset.treino.texto.slice(0, n0)), trp = new Set(dataset.treino.texto.slice(n0));
  assert.equal(dataset.validacao.texto.slice(0, m0).filter((t) => tr0.has(t)).length, 0, 'originais: sem duplicatas treino/validação');
  assert.equal(dataset.teste.texto.filter((t) => tr0.has(t)).length, 0);
  assert.equal(dataset.teste_pt.texto.filter((t) => trp.has(t)).length, 0);
  assert.ok(dataset.validacao.texto.slice(m0).filter((t) => trp.has(t)).length <= 15, 'traduções idênticas de originais diferentes são raras');
});

test('data/deepset_pt.json é a tradução completa (546 treino + 116 teste alinhados ao original completo) e bate com o dataset do laboratório', () => {
  const pt = JSON.parse(readFileSync(new URL('../data/deepset_pt.json', import.meta.url), 'utf8'));
  assert.equal(pt.treino.length, 546);
  assert.equal(pt.teste.length, 116);
  assert.equal(pt.indices_en.treino.length, 269);
  assert.equal(pt.indices_en.teste.length, 61);
  assert.deepEqual(pt.indices_en.teste.map((i) => pt.teste[i]), dataset.teste_pt.texto);
  const emTreino = new Set(pt.treino);
  assert.ok(dataset.treino.texto.slice(dataset.treino.n_orig).every((t) => emTreino.has(t)));
  assert.ok(dataset.validacao.texto.slice(dataset.validacao.n_orig).every((t) => emTreino.has(t)));
});

test('resultados.json traz os números do trabalho (conferência por amostragem)', () => {
  const r = resultados, F = { bil: 'bilíngue (teste EN + PT)', en: 'deepset EN (teste)', pt: 'deepset PT (teste)' };
  assert.equal(r.otimizadores.Adam.lr, 0.001);
  assert.equal(r.otimizadores.SGD.lr, 1);
  assert.equal(r.otimizadores['SGD+Momentum'].lr, 0.1);
  assert.equal(r.otimizadores.RMSProp.lr, 0.001);
  assert.ok(Math.abs(r.otimizadores.Adam.f1[F.bil][0] - 0.893) < 1e-3);
  assert.ok(Math.abs(r.otimizadores.SGD.f1[F.bil][0] - 0.849) < 1e-3);
  assert.ok(Math.abs(r.otimizadores.RMSProp.f1[F.pt][0] - 0.904) < 1e-3);
  assert.ok(Math.abs(r.otimizadores.RMSProp.melhor_epoca - 24.0) < 0.05);
  assert.ok(Math.abs(r.otimizadores.SGD.melhor_epoca - 14.0) < 0.05);
  assert.ok(Math.abs(r.otimizadores.Adam.melhor_epoca - 24.4) < 0.05);
  const g = (n, lr) => r.grade_lr.find((x) => x.otimizador === n && x.lr === lr);
  assert.ok(Math.abs(g('Adam', 0.01).melhor_epoca - 3.0) < 0.01);
  assert.ok(Math.abs(g('Adam', 0.01).val_loss - 0.3953) < 1e-4 && Math.abs(g('Adam', 0.001).val_loss - 0.3618) < 1e-4);
  assert.ok(Math.abs(g('RMSProp', 0.01).val_loss - 0.3814) < 1e-4 && Math.abs(g('RMSProp', 0.001).val_loss - 0.3784) < 1e-4);
  assert.deepEqual([g('RMSProp', 0.01).melhor_epoca, g('RMSProp', 0.001).melhor_epoca], [2, 24]);
  assert.equal(r.capacidade.length, 6);
  assert.deepEqual(r.capacidade.map((c) => c.H), [0, 2, 8, 32, 128, 512]);
  assert.equal(r.capacidade[5].parametros, 404481);
  assert.equal(r.representacao.length, 3);
  assert.equal(r.curvas_medias.Adam.val.length, 40);
  // matriz língua de treino × teste (Figura 4)
  const id = (tr, c) => r.idiomas.find((x) => x.treino === tr && x.conjunto === c);
  assert.ok(Math.abs(id('orig', F.pt).f1 - 0.268) < 1e-3);
  assert.ok(Math.abs(id('pt', F.en).f1 - 0.364) < 1e-3);
  assert.ok(Math.abs(id('bilingue', F.pt).f1 - 0.911) < 1e-3);
  assert.deepEqual([id('orig', F.en).vocab, id('pt', F.en).vocab, id('bilingue', F.en).vocab], [375, 384, 788]);
  assert.deepEqual(['tp', 'fp', 'fn', 'tn'].map((k) => r.final_semente0[0][k]), [25, 1, 8, 27]);
  const br = r.final.find((x) => x.modelo.startsWith('MLP') && x.conjunto.startsWith('BR-synthetic'));
  assert.ok(Math.abs(br.f1 - 0.571) < 1e-3 && br.rec > 0.98 && br.fpr > 0.98);
  assert.ok(r.custo.distilbert.p50_ms > 50);
  // nenhum número do trabalho monolíngue sobrevive nos dados exportados
  assert.ok(!JSON.stringify(r).includes('59681'));
  assert.notEqual(r.info.parametros_mlp, 31521);
});

test('treino em JS: determinístico por semente, perda de treino cai e restaura a melhor época', async () => {
  const D = prepararDados(dataset, vocab);
  const cfg = { otimizador: 'Adam', lr: 0.001, H: 8, batch: 32, epocas: 12, semente: 0 };
  const a = await treinar(cfg, D), b = await treinar(cfg, D);
  assert.deepEqual(a.curva.val, b.curva.val);
  assert.ok(a.curva.treino[11] < a.curva.treino[0]);
  assert.equal(a.curva.val[a.melhorEp - 1], Math.min(...a.curva.val));
  const m = avaliarConjuntos(a.melhor, D);
  assert.ok(m.bil.f1 > 0.6, 'F1 no teste ' + m.bil.f1);
  assert.deepEqual(Object.keys(m), ['en', 'pt', 'bil', 'neuralchemy', 'xtram1', 'br']);
  const c = await treinar({ ...cfg, semente: 1 }, D);
  assert.notDeepEqual(a.curva.val, c.curva.val);
});

test('parar() interrompe o treino e devolve o melhor modelo até então', async () => {
  const D = prepararDados(dataset, vocab);
  let n = 0;
  const r = await treinar({ otimizador: 'SGD', lr: 1, H: 8, batch: 32, epocas: 40, semente: 0 }, D, { parar: () => ++n >= 3 });
  assert.equal(r.epocasFeitas, 3);
});

test('seletor de línguas de treino: tamanhos de treino, vocabulário reajustado e treino curto em cada modo', async () => {
  const bil = prepararDados(dataset, vocab, 'tfidf', 'bil'), orig = prepararDados(dataset, vocab, 'tfidf', 'orig'), pt = prepararDados(dataset, vocab, 'tfidf', 'pt');
  assert.deepEqual([bil.Xtr.length, orig.Xtr.length, pt.Xtr.length], [432, 216, 216]);
  assert.deepEqual([bil.Xva.length, orig.Xva.length, pt.Xva.length], [106, 53, 53]);
  assert.deepEqual([bil.V, orig.V, pt.V], [788, 375, 384]);
  for (const D of [orig, pt]) assert.deepEqual([D.Xte.length, D.Xtp.length, D.Xbr.length], [61, 61, 200]);
  const cfg = { otimizador: 'Adam', lr: 0.001, H: 8, batch: 32, epocas: 15, semente: 0 };
  const ro = await treinar(cfg, orig), rp = await treinar(cfg, pt);
  const mo = avaliarConjuntos(ro.melhor, orig), mp = avaliarConjuntos(rp.melhor, pt);
  // H4 no navegador: o treino só EN vai pior no teste PT do que no EN, e o treino só PT vai pior no EN do que no PT
  assert.ok(mo.pt.f1 < mo.en.f1, `só EN: PT ${mo.pt.f1} < EN ${mo.en.f1}`);
  assert.ok(mp.en.f1 < mp.pt.f1, `só PT: EN ${mp.en.f1} < PT ${mp.pt.f1}`);
});

test('sem alemão nos dados exportados: observação do deepset presente e comparação com o Trabalho 01 só no recorte em inglês', () => {
  assert.match(dataset.meta.nota, /descartados, não convertidos/);
  assert.match(dataset.meta.nota, /330 textos em inglês/);
  assert.ok(!resultados.trabalho1.some((x) => x.conjunto === 'deepset (teste, em-distribuição)'), 'sem o teste EN/DE completo do Trabalho 01');
  assert.ok(resultados.trabalho1.some((x) => x.conjunto === 'deepset (teste, só inglês)'));
  assert.equal(resultados.dados.find((d) => d.conjunto.startsWith('deepset EN')).n, 61);
  const pt = JSON.parse(readFileSync(new URL('../data/deepset_pt.json', import.meta.url), 'utf8'));
  assert.match(pt.metodo, /modelo de linguagem \(LLM\)/);
  assert.match(pt.nota_indices_en, /descartados/);
});
