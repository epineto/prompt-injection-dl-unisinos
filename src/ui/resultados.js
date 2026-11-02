// Seção #resultados: tabelas e gráficos estáticos de data/resultados.json.
import { $, COR, COR_OTIM, NOME_OTIM, esc } from './util.js';
import { t, fmt, fmtM, onLang } from './i18n.js';
import { GraficoLinhas } from './graficos.js';
import { NOMES } from '../core/otimizadores.js';

const tab = (cab, linhas, cls = '') => `<div class="rolagem"><table class="tab ${cls}"><thead><tr>${cab.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>${linhas.map((l) => `<tr>${l.map((c, i) => (i === 0 ? `<th scope="row">${c}</th>` : `<td>${c}</td>`)).join('')}</tr>`).join('')}</tbody></table></div>`;
const pm = (m, d, dec = 3) => `${fmt(m, dec)}${d != null ? ' ± ' + fmt(d, dec) : ''}`;

export function iniciarResultados(S) {
  const r = S.res;
  let gCurva = null;

  function barras(grupos) { // grupos: [{rotulo, barras:[{nome, cor, v}]}]
    return grupos.map((g) => `<div class="bgrupo"><h4>${esc(g.rotulo)}</h4>${g.barras.map((b) => `<div class="b-lin"><span class="b-n">${esc(b.nome)}</span><span class="b-trilho"><i style="width:${Math.max(b.v, 0) * 100}%;background:${b.cor}"></i></span><span class="b-v">${fmt(b.v, 3)}</span></div>`).join('')}</div>`).join('');
  }

  function render() {
    const C = (k) => r.otimizadores[k];
    const D = 'deepset (teste)';
    let h = '';
    // dados
    h += `<div class="cartao"><h3>${t('res.dados')}</h3>${tab([t('res.d_conj'), 'n', t('res.d_inj'), t('res.d_pct')], r.dados.map((d) => [esc(d.conjunto), d.n, d.injecao, fmt(d.pct, 1) + '%']))}<p class="mini just">${t('res.dados_d')}</p></div>`;
    // otimizadores
    h += `<div class="cartao"><h3>${t('res.otim')}</h3><div class="grade2"><div><canvas id="res-g-curva" role="img" aria-label="${esc(t('res.otim_aria'))}"></canvas><div class="legenda">${NOMES.map((o) => `<span><i style="background:${COR_OTIM[o]}"></i>${NOME_OTIM[o]}</span>`).join('')}</div></div><div>${
      tab([t('otim.t_otim'), t('otim.lr'), t('otim.t_melhor_ep'), t('otim.t_melhor'), 'F1', 'AUC'], NOMES.map((o) => [`<span class="sw" style="background:${COR_OTIM[o]}"></span>${NOME_OTIM[o]}`, fmtM(C(o).lr, 4), pm(C(o).melhor_epoca, C(o).melhor_epoca_dp, 1), fmt(C(o).val_loss, 3), pm(...C(o).f1[D]), fmt(C(o).auc[D][0], 3)]))}</div></div>
      <p class="mini just">${t('res.otim_d', { ad: fmtM((C('RMSProp').melhor_epoca + C('Adam').melhor_epoca) / 2, 0), cl: fmtM((C('SGD').melhor_epoca + C('SGD+Momentum').melhor_epoca) / 2, 0), vs: fmt(C('SGD').val_loss, 3), va: fmt(C('Adam').val_loss, 3) })}</p>
      ${tab(['', 'ep 1', 'ep 3', 'ep 5', 'ep 10', 'ep 20', 'ep 40'], r.convergencia.map((c) => [`<span class="sw" style="background:${COR_OTIM[c.otimizador]}"></span>${NOME_OTIM[c.otimizador]}`, ...['ep1', 'ep3', 'ep5', 'ep10', 'ep20', 'ep40'].map((k) => fmt(c[k], 3))]))}
      <p class="mini just">${t('res.conv_d')}</p></div>`;
    // capacidade
    h += `<div class="cartao"><h3>${t('res.cap')}</h3>${tab(['H', t('res.params'), t('res.cap_treino'), t('res.cap_val'), 'F1 (deepset)', 'AUC', t('otim.t_melhor_ep')],
      r.capacidade.map((c) => [c.H === 0 ? t('res.logistica') : String(c.H), Number(c.parametros).toLocaleString(t('lang.loc')), fmt(c.treino_loss, 3), pm(c.val_loss, c.val_loss_dp), pm(c.f1_teste, c.f1_teste_dp), fmt(c.auc_teste, 3), fmt(c.melhor_epoca, 1)]))}<p class="mini just">${t('res.cap_d')}</p></div>`;
    // representação
    h += `<div class="cartao"><h3>${t('res.rep')}</h3>${tab([t('res.rep_nome'), t('res.cap_val'), 'F1 (deepset)', 'AUC', 'F1 neuralchemy', 'F1 xTRam1'],
      r.representacao.map((c) => [esc(c.representacao), fmt(c.val_loss, 3), pm(c.f1_teste, c.f1_teste_dp), fmt(c.auc_teste, 3), fmt(c.f1_nc, 3), fmt(c.f1_xt, 3)]))}<p class="mini just">${t('res.rep_d')}</p></div>`;
    // generalização
    const mlp = (c) => r.final.find((x) => x.modelo.startsWith('MLP') && x.conjunto === c), lg = (c) => r.final.find((x) => x.modelo.startsWith('Regress') && x.conjunto === c);
    const t1 = (d, c) => r.trabalho1.find((x) => x.detector === d && x.conjunto === c);
    const conj = [['deepset (teste)', 'deepset (teste, em-distribuição)', 'testar.c_deepset'], ['neuralchemy (OOD)', 'neuralchemy (OOD)', 'testar.c_nc'], ['xTRam1 (OOD)', 'xTRam1 (OOD, sintético)', 'testar.c_xt']];
    h += `<div class="cartao"><h3>${t('res.gen')}</h3><div class="barras">${barras(conj.map(([a, b, ch]) => ({ rotulo: t(ch), barras: [
      { nome: 'MLP (H=32, Adam)', cor: COR.azul, v: mlp(a).f1 }, { nome: t('res.logreg'), cor: COR.azul2, v: lg(a).f1 },
      { nome: 'DistilBERT (T1)', cor: COR.verm, v: t1('DistilBERT', b).f1 / 100 }, { nome: 'BERT-base (T1)', cor: COR.verde, v: t1('BERT-base', b).f1 / 100 }] })))}</div>
      ${tab([t('res.d_conj'), 'MLP F1', 'MLP AUC', t('res.logreg') + ' F1', 'DistilBERT F1', 'BERT-base F1'], conj.map(([a, b, ch]) => [t(ch), pm(mlp(a).f1, mlp(a).f1_dp), fmt(mlp(a).auc, 3), fmt(lg(a).f1, 3), fmt(t1('DistilBERT', b).f1 / 100, 3), fmt(t1('BERT-base', b).f1 / 100, 3)]))}
      <p class="mini just">${t('res.gen_d')}</p></div>`;
    // custo
    const cu = r.custo, razao = cu.distilbert.p50_ms / cu.mlp.ms_por_texto;
    h += `<div class="cartao"><h3>${t('res.custo')}</h3>${tab([t('res.c_modelo'), t('res.params'), t('res.c_ms'), t('res.c_tam')], [
      ['MLP (H=32)', Number(cu.mlp.parametros).toLocaleString(t('lang.loc')), fmt(cu.mlp.ms_por_texto, 2), fmt(cu.mlp.parametros * 4 / 1e6, 2) + ' (fp32)'],
      ['DistilBERT (T1)', fmt(cu.distilbert.params_m, 1) + ' M', fmt(cu.distilbert.p50_ms, 1) + ' (p50)', fmt(cu.distilbert.tamanho_mb, 0)],
      ['BERT-base (T1)', fmt(cu.bert_base.params_m, 1) + ' M', fmt(cu.bert_base.p50_ms, 1) + ' (p50)', fmt(cu.bert_base.tamanho_mb, 0)]])}
      <p class="just"><strong>${t('res.custo_razao', { x: fmt(razao, 0) })}</strong> ${t('res.custo_d')}</p></div>`;
    // termos
    h += `<div class="cartao"><h3>${t('res.termos')}</h3>${tab([t('res.termo'), t('res.queda'), t('res.n_textos')], r.termos.map((x) => [esc(x.termo), fmt(x.queda * 100, 1) + ' p.p.', x.n]))}<p class="mini just">${t('res.termos_d')}</p></div>`;
    // mapeamento
    const mapa = ['dados', 'tfidf', 'grad', 'otim', 'cap', 'rep', 'gen', 'custo', 'termos'].map((k) => [t('map.' + k + '.r'), t('map.' + k + '.l')]);
    h += `<div class="cartao"><h3>${t('res.mapa')}</h3>${tab([t('res.mapa_r'), t('res.mapa_l')], mapa, 'tab-txt')}<p class="mini just">${t('res.mapa_d')}</p></div>`;
    $('#res-corpo').innerHTML = h;
    const cv = $('#res-g-curva');
    gCurva = new GraficoLinhas(cv, { altura: 240, fmtY: (v) => fmt(v, 2) });
    gCurva.definir({ xMax: 40, yMin: 0, yMax: 1.1, rotX: t('treinar.epoca'), rotY: t('res.val_media'),
      series: NOMES.map((o) => ({ nome: o, cor: COR_OTIM[o], y: r.curvas_medias[o].val })) });
  }
  onLang(render);
  render();
}
