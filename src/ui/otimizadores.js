// Seção #otimizadores: SGD × Momentum × RMSProp × Adam, mesma semente e mesmos dados, 4 workers em paralelo.
import { NOMES, LR_TRABALHO, LR_ADAM_RAPIDO } from '../core/otimizadores.js';
import { $, COR_OTIM, NOME_OTIM, NOME_PY } from './util.js';
import { t, fmt, fmtM, onLang } from './i18n.js';
import { GraficoLinhas } from './graficos.js';
import { iniciarTreino } from './executor.js';

export function iniciarOtimizadores(S) {
  const gT = new GraficoLinhas($('#ot-g-treino'), { fmtY: (v) => fmt(v, 2) });
  const gV = new GraficoLinhas($('#ot-g-val'), { fmtY: (v) => fmt(v, 2) });
  let execs = [], estado = null, ultimo = null;
  const YMAX = 1.5;

  function emExecucao(on) {
    $('#ot-go').disabled = on; $('#ot-stop').disabled = !on; $('#ot-seed').disabled = on; $('#ot-ep').disabled = on; $('#ot-adam-rapido').disabled = on;
    document.querySelectorAll('input[name="ot-mult"]').forEach((r) => { r.disabled = on; });
  }
  const mult = () => Number(document.querySelector('input[name="ot-mult"]:checked').value);

  function desenhar() {
    const total = estado ? estado.total : 40;
    const serie = (campo) => NOMES.map((o) => ({ nome: o, cor: COR_OTIM[o], y: estado ? estado.cv[o][campo] : [] }));
    const marcas = estado ? NOMES.filter((o) => estado.melhorEp[o]).map((o) => ({ x: estado.melhorEp[o], y: estado.cv[o].val[estado.melhorEp[o] - 1], cor: COR_OTIM[o] })) : [];
    gT.definir({ xMax: total, yMin: 0, yMax: YMAX, rotX: t('treinar.epoca'), rotY: t('treinar.perda'), series: serie('treino') });
    gV.definir({ xMax: total, yMin: 0, yMax: YMAX, rotX: t('treinar.epoca'), rotY: t('treinar.perda'), series: serie('val'), marcas });
    const lg = $('#ot-legenda');
    lg.innerHTML = NOMES.map((o) => `<span><i style="background:${COR_OTIM[o]}"></i>${NOME_OTIM[o]} (${t('otim.lr')} ${fmtM(lrDe(o), 4)})</span>`).join('') + `<span class="mini">${t('otim.corte', { y: fmt(YMAX, 1) })}</span>`;
  }
  const lrBase = (o, rapido) => (o === 'Adam' && rapido ? LR_ADAM_RAPIDO : LR_TRABALHO[o]);
  const rapido = () => $('#ot-adam-rapido').checked;
  const lrDe = (o) => lrBase(o, estado ? estado.rapido : rapido()) * (estado ? estado.mult : mult());

  function status() {
    const ul = $('#ot-status');
    ul.innerHTML = NOMES.map((o) => {
      const e = estado ? estado.cv[o].val.length : 0, total = estado ? estado.total : 0;
      return `<li><span class="sw" style="background:${COR_OTIM[o]}"></span><strong>${NOME_OTIM[o]}</strong> <span class="mini">${estado ? t('otim.ep_de', { e, n: total }) : ''}</span></li>`;
    }).join('');
  }

  $('#ot-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const semente = Number($('#ot-seed').value), epocas = Number($('#ot-ep').value), av = $('#ot-aviso');
    if (!Number.isInteger(semente) || semente < 0 || semente > 9999 || !Number.isInteger(epocas) || epocas < 1 || epocas > 200) { av.hidden = false; av.textContent = t('treinar.invalido'); return; }
    av.hidden = true; $('#ot-msg').innerHTML = ''; $('#ot-tab').innerHTML = '';
    const m = mult();
    const rap = rapido();
    estado = { total: epocas, mult: m, rapido: rap, cv: Object.fromEntries(NOMES.map((o) => [o, { treino: [], val: [] }])), melhorEp: {} };
    emExecucao(true); desenhar(); status();
    execs = NOMES.map((o) => iniciarTreino({ otimizador: o, lr: lrBase(o, rap) * m, H: 32, batch: 32, epocas, semente, treino: 'bil' }, (e) => {
      estado.cv[o].treino.push(e.treino); estado.cv[o].val.push(e.val); estado.melhorEp[o] = e.melhorEp;
      if (!estado._raf) estado._raf = requestAnimationFrame(() => { estado._raf = 0; desenhar(); status(); });
    }));
    try {
      const rs = await Promise.all(execs.map((x) => x.promessa));
      ultimo = { mult: m, rapido: rap, semente, rs: Object.fromEntries(NOMES.map((o, i) => [o, rs[i]])) };
      desenhar(); status(); tabela();
    } catch (e) { av.hidden = false; av.textContent = t('erro.worker', { m: e.message }); }
    finally { emExecucao(false); execs = []; }
  });
  $('#ot-stop').addEventListener('click', () => execs.forEach((x) => x.parar()));

  function tabela() {
    if (!ultimo) return;
    const { rs, mult: m } = ultimo;
    let h = `<table class="tab"><thead><tr><th>${t('otim.t_otim')}</th><th>${t('otim.lr')}</th><th>${t('otim.t_melhor_ep')}</th><th>${t('otim.t_melhor')}</th><th>${t('otim.t_final')}</th><th>${t('otim.t_f1')}</th><th>${t('otim.t_f1u')}</th></tr></thead><tbody>`;
    for (const o of NOMES) {
      const r = rs[o], vf = r.curva.val[r.curva.val.length - 1], vm = r.curva.val[r.melhorEp - 1];
      const cls = Number.isFinite(vf) && vf > vm * 1.1 ? ' class="piora"' : '';
      h += `<tr><th scope="row"><span class="sw" style="background:${COR_OTIM[o]}"></span>${NOME_OTIM[o]}</th><td>${fmtM(lrBase(o, ultimo.rapido) * m, 4)}</td><td>${r.melhorEp}</td><td>${fmt(vm, 3)}</td><td${cls}>${Number.isFinite(vf) ? fmt(vf, 3) : '∞'}</td><td>${fmt(r.metricas.bil.f1, 3)}</td><td>${fmt(r.metricasUltima.bil.f1, 3)}</td></tr>`;
    }
    $('#ot-tab').innerHTML = h + '</tbody></table>';
    const cl = Math.round((rs.SGD.melhorEp + rs['SGD+Momentum'].melhorEp) / 2);
    $('#ot-msg').innerHTML = `<p class="just"><strong>${t('otim.chave_t')}</strong> ${t('otim.chave_live', { rm: rs.RMSProp.melhorEp, ad: rs.Adam.melhorEp, cl, ep: ultimo.rs.SGD.epocasFeitas })}</p>`;
  }

  function referencia() {
    const o = S.res.otimizadores, F = NOME_PY;
    let h = `<table class="tab"><thead><tr><th>${t('otim.t_otim')}</th><th>${t('otim.lr')}</th><th>${t('otim.t_melhor_ep')}</th><th>${t('otim.t_melhor')}</th><th>F1 ${t('res.i_bil')}</th><th>F1 ${t('res.i_en')}</th><th>F1 ${t('res.i_pt')}</th><th>AUC ${t('res.i_bil')}</th></tr></thead><tbody>`;
    for (const n of NOMES) {
      const r = o[n], f = (k) => `${fmt(r.f1[F[k]][0], 3)} ± ${fmt(r.f1[F[k]][1], 3)}`;
      h += `<tr><th scope="row"><span class="sw" style="background:${COR_OTIM[n]}"></span>${NOME_OTIM[n]}</th><td>${fmtM(r.lr, 4)}</td><td>${fmt(r.melhor_epoca, 1)} ± ${fmt(r.melhor_epoca_dp, 1)}</td><td>${fmt(r.val_loss, 3)}</td><td>${f('bil')}</td><td>${f('en')}</td><td>${f('pt')}</td><td>${fmt(r.auc[F.bil][0], 3)}</td></tr>`;
    }
    $('#ot-ref').innerHTML = h + '</tbody></table>';
    const g = (n, lr) => S.res.grade_lr.find((x) => x.otimizador === n && x.lr === lr);
    $('#ot-ref-d').textContent = t('otim.ref_d', { rm: fmt(o.RMSProp.melhor_epoca, 1), cl: fmt((o.SGD.melhor_epoca + o['SGD+Momentum'].melhor_epoca) / 2, 0), ad: fmt(o.Adam.melhor_epoca, 0),
      v01: fmt(g('Adam', 0.01).val_loss, 3), v001: fmt(g('Adam', 0.001).val_loss, 3), e01: fmt(g('Adam', 0.01).melhor_epoca, 1), vs: fmt(o.SGD.val_loss, 3), vm: fmt(o['SGD+Momentum'].val_loss, 3) });
  }
  onLang(() => { desenhar(); status(); tabela(); referencia(); });
  desenhar(); status(); referencia();
}
