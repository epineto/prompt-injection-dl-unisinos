// Seção #treinar: treino do zero em Web Worker, com curvas ao vivo.
import { lerNumero } from '../core/numeros.js';
import { nParametros } from '../core/mlp.js';
import { $, COR } from './util.js';
import { t, fmt, fmtM, onLang } from './i18n.js';
import { GraficoLinhas } from './graficos.js';
import { iniciarTreino } from './executor.js';

export const PADRAO = { otimizador: 'Adam', lr: 0.01, H: 32, batch: 32, epocas: 40, semente: 0 };

export function lerConfig(ids) {
  const g = (k) => $('#' + ids[k]);
  const cfg = { otimizador: g('otim').value, lr: lerNumero(g('lr').value), H: Number(g('h').value), batch: Number(g('batch').value), epocas: Number(g('ep').value), semente: Number(g('seed').value) };
  const int = (v, a, b) => Number.isInteger(v) && v >= a && v <= b;
  const ok = cfg.lr > 0 && cfg.lr <= 100 && int(cfg.H, 1, 512) && int(cfg.batch, 1, 436) && int(cfg.epocas, 1, 200) && int(cfg.semente, 0, 9999);
  return ok ? cfg : null;
}

export function tabelaMetricas(r) {
  const nomes = [['deepset', 'testar.c_deepset'], ['neuralchemy', 'testar.c_nc'], ['xtram1', 'testar.c_xt']];
  let h = `<table class="tab"><thead><tr><th>${t('treinar.t_conj')}</th><th>F1</th><th>AUC</th><th>${t('treinar.t_prec')}</th><th>${t('treinar.t_rec')}</th><th>TP / FP / FN / TN</th></tr></thead><tbody>`;
  for (const [k, ch] of nomes) {
    const m = r.metricas[k];
    h += `<tr><th scope="row">${t(ch)}</th><td>${fmt(m.f1, 3)}</td><td>${fmt(m.auc, 3)}</td><td>${fmt(m.prec, 3)}</td><td>${fmt(m.rec, 3)}</td><td>${m.tp} / ${m.fp} / ${m.fn} / ${m.tn}</td></tr>`;
  }
  return h + '</tbody></table>';
}

export function iniciarTreinar(S) {
  const ids = { otim: 'tr-otim', lr: 'tr-lr', h: 'tr-h', batch: 'tr-batch', ep: 'tr-ep', seed: 'tr-seed' };
  const gPerda = new GraficoLinhas($('#tr-g-perda'), { fmtY: (v) => fmt(v, 2) });
  const gF1 = new GraficoLinhas($('#tr-g-f1'), { fmtY: (v) => fmt(v, 2) });
  let exec = null, c = null, ultimo = null;

  function params() {
    const H = Number($('#tr-h').value), V = S.vet.V;
    $('#tr-params').textContent = Number.isInteger(H) && H > 0 ? `V·H + H + H + 1 = ${V}·${H} + ${H} + ${H} + 1 = ${Number(nParametros(V, H)).toLocaleString(t('lang.loc'))}` : '–';
  }
  function desenhar() {
    if (!c) { gPerda.definir({ series: [], marcas: [], xMax: 40 }); gF1.definir({ series: [], marcas: [], xMax: 40, yMin: 0, yMax: 1 }); return; }
    const marcas = c.melhorEp ? [{ x: c.melhorEp, y: c.val[c.melhorEp - 1], cor: COR.verde }] : [];
    gPerda.definir({ xMax: c.total, yMin: 0, rotX: t('treinar.epoca'), rotY: t('treinar.perda'), marcas,
      series: [{ nome: 'treino', cor: COR.azul, y: c.treino }, { nome: 'val', cor: COR.verm, y: c.val }] });
    gF1.definir({ xMax: c.total, yMin: 0, yMax: 1, rotX: t('treinar.epoca'), rotY: 'F1', marcas: c.melhorEp ? [{ x: c.melhorEp, y: c.f1[c.melhorEp - 1], cor: COR.verde }] : [],
      series: [{ nome: 'f1', cor: COR.azul2, y: c.f1 }] });
    $('#tr-g-perda').setAttribute('aria-label', t('treinar.aria_perda', { n: c.val.length, t: fmt(c.treino[c.treino.length - 1], 3), v: fmt(c.val[c.val.length - 1], 3), m: c.melhorEp }));
  }
  function emExecucao(on) {
    $('#tr-go').disabled = on; $('#tr-stop').disabled = !on; $('#tr-reset').disabled = on;
    for (const id of Object.values(ids)) $('#' + id).disabled = on;
  }
  function barra(pct) { $('#tr-barra').setAttribute('aria-valuenow', String(Math.round(pct))); $('#tr-barra-i').style.width = pct + '%'; }

  $('#tr-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const cfg = lerConfig(ids);
    const av = $('#tr-aviso');
    if (!cfg) { av.hidden = false; av.textContent = t('treinar.invalido'); return; }
    av.hidden = true; $('#tr-res').hidden = true;
    c = { total: cfg.epocas, treino: [], val: [], f1: [], melhorEp: 0 };
    emExecucao(true); desenhar(); barra(0);
    $('#tr-status').textContent = t('treinar.iniciando');
    exec = iniciarTreino(cfg, (m) => {
      c.treino.push(m.treino); c.val.push(m.val); c.f1.push(m.val_f1); c.melhorEp = m.melhorEp;
      barra(100 * m.ep / cfg.epocas);
      const resta = (m.ms / m.ep) * (cfg.epocas - m.ep) / 1000;
      $('#tr-status').textContent = t('treinar.status', { ep: m.ep, n: cfg.epocas, s: fmt(m.ms / 1000, 1), r: fmt(resta, 1) });
      desenhar();
    });
    try {
      const r = await exec.promessa;
      ultimo = { r, cfg };
      barra(100);
      mostrarResultado();
    } catch (e) {
      av.hidden = false; av.textContent = t('erro.worker', { m: e.message });
    } finally { emExecucao(false); exec = null; }
  });
  $('#tr-stop').addEventListener('click', () => exec && exec.parar());
  $('#tr-reset').addEventListener('click', () => {
    $('#tr-otim').value = PADRAO.otimizador; $('#tr-lr').value = fmtM(PADRAO.lr, 4); $('#tr-h').value = PADRAO.H; $('#tr-batch').value = PADRAO.batch; $('#tr-ep').value = PADRAO.epocas; $('#tr-seed').value = PADRAO.semente; params();
  });
  $('#tr-h').addEventListener('input', params);

  function mostrarResultado() {
    if (!ultimo) return;
    const { r, cfg } = ultimo;
    const py = S.res.otimizadores.Adam;
    const f1u = r.metricasUltima.deepset.f1;
    $('#tr-res').hidden = false;
    $('#tr-res-txt').textContent = t('treinar.res_txt', {
      s: fmt(r.ms / 1000, 1), n: r.epocasFeitas, m: r.melhorEp, l: fmt(r.curva.val[r.melhorEp - 1], 3), p: Number(r.parametros).toLocaleString(t('lang.loc')),
      u: fmt(f1u, 3), f: fmt(r.metricas.deepset.f1, 3), pf: fmt(py.f1['deepset (teste)'][0], 3), pd: fmt(py.f1['deepset (teste)'][1], 3),
    }) + (r.interrompido ? ' ' + t('treinar.interrompido') : '');
    $('#tr-res-tab').innerHTML = tabelaMetricas(r);
    void cfg;
  }
  onLang(() => { params(); desenhar(); if (ultimo) mostrarResultado(); if (!exec && !c) $('#tr-status').textContent = t('treinar.pronto'); });
  params(); desenhar();
}
