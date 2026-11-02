// Seção #resultados: tabelas e gráficos estáticos de data/resultados.json.
import { $, COR, COR_OTIM, NOME_OTIM, NOME_PY, CHAVE_CONJ, CHAVE_TREINO, TREINO_PY, esc } from './util.js';
import { t, fmt, fmtM, onLang } from './i18n.js';
import { GraficoLinhas } from './graficos.js';
import { NOMES } from '../core/otimizadores.js';
import { parametrosRef } from './ref_otim.js';

const tab = (cab, linhas, cls = '') => `<div class="rolagem"><table class="tab ${cls}"><thead><tr>${cab.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>${linhas.map((l) => `<tr>${l.map((c, i) => (i === 0 ? `<th scope="row">${c}</th>` : `<td>${c}</td>`)).join('')}</tr>`).join('')}</tbody></table></div>`;
const pm = (m, d, dec = 3) => `${fmt(m, dec)}${d != null ? ' ± ' + fmt(d, dec) : ''}`;
const sw = (o) => `<span class="sw" style="background:${COR_OTIM[o]}"></span>${NOME_OTIM[o]}`;

// Célula de mapa de calor (como a Figura 4): azul mais forte para F1 maior; texto claro sobre fundo escuro.
function celula(v) {
  const a = Math.max(0, Math.min(1, (v - 0.3) / 0.7));
  const cor = `rgb(${Math.round(233 - a * (233 - 29))},${Math.round(236 - a * (236 - 35))},${Math.round(243 - a * (243 - 105))})`;
  return `<td class="heat" style="background:${cor};color:${a > 0.55 ? '#fff' : '#1D2369'}">${fmt(v, 2)}</td>`;
}

export function iniciarResultados(S) {
  const r = S.res;

  function barras(grupos) { // grupos: [{rotulo, barras:[{nome, cor, v}]}]
    return grupos.map((g) => `<div class="bgrupo"><h4>${esc(g.rotulo)}</h4>${g.barras.map((b) => `<div class="b-lin"><span class="b-n">${esc(b.nome)}</span><span class="b-trilho"><i style="width:${Math.max(b.v, 0) * 100}%;background:${b.cor}"></i></span><span class="b-v">${fmt(b.v, 3)}</span></div>`).join('')}</div>`).join('');
  }

  const nomeDados = (n) => { // nomes do CSV (PT) -> idioma da interface
    if (n.startsWith('treino')) return t('res.d_treino');
    if (n.startsWith('valida')) return t('res.d_val');
    const k = Object.keys(NOME_PY).find((c) => NOME_PY[c] === n);
    return k ? t(CHAVE_CONJ[k]) : n;
  };

  function render() {
    const C = (k) => r.otimizadores[k], F = NOME_PY;
    const idi = (tr, c) => r.idiomas.find((x) => x.treino === tr && x.conjunto === F[c]);
    let h = '';
    // dados
    h += `<div class="cartao"><h3>${t('res.dados')}</h3>${tab([t('res.d_conj'), 'n', t('res.d_inj'), t('res.d_pct')], r.dados.map((d) => [esc(nomeDados(d.conjunto)), d.n, d.injecao, fmt(d.pct, 1) + '%']))}<p class="mini just">${t('res.dados_d')}</p></div>`;
    // línguas (H4)
    const cols = ['en', 'pt', 'bil', 'br'], cabI = [t('res.i_treino'), t('res.i_vocab'), t('res.i_en'), t('res.i_pt'), t('res.i_bil'), t('res.i_br')];
    const linhasI = ['orig', 'pt', 'bilingue'].map((tr) => [t(CHAVE_TREINO[Object.keys(TREINO_PY).find((k) => TREINO_PY[k] === tr)]), Number(idi(tr, 'en').vocab).toLocaleString(t('lang.loc')), ...cols.map((c) => idi(tr, c))]);
    h += `<div class="cartao"><h3>${t('res.idiomas')}</h3><div class="rolagem"><table class="tab"><thead><tr>${cabI.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>${linhasI.map((l) => `<tr><th scope="row">${l[0]}</th><td>${l[1]}</td>${l.slice(2).map((x) => celula(x.f1)).join('')}</tr>`).join('')}</tbody></table></div>
      <p class="mini just">${t('res.idiomas_d', { a: fmt(idi('orig', 'pt').f1, 3), b: fmt(idi('pt', 'en').f1, 3), c: pm(idi('bilingue', 'pt').f1, idi('bilingue', 'pt').f1_dp), d: fmt(idi('bilingue', 'en').f1, 3), e: fmt(idi('orig', 'en').f1, 3), f: fmt(idi('bilingue', 'br').f1, 2) })}</p></div>`;
    // grade de taxas de aprendizado: perda de validação média de 3 sementes; a menor de cada otimizador em negrito
    const gradeTab = () => {
      const lrs = [...new Set(r.grade_lr.map((g) => g.lr))].sort((a, b) => a - b);
      return tab([t('otim.t_otim'), ...lrs.map((lr) => fmtM(lr, 4))], NOMES.map((o) => {
        const gs = r.grade_lr.filter((g) => g.otimizador === o), min = Math.min(...gs.map((g) => g.val_loss));
        return [sw(o), ...lrs.map((lr) => { const g = gs.find((x) => x.lr === lr); return g ? (g.val_loss === min ? `<strong>${fmt(g.val_loss, 3)}</strong>` : fmt(g.val_loss, 3)) : '–'; })];
      }));
    };
    // otimizadores
    h += `<div class="cartao"><h3>${t('res.otim')}</h3><div class="grade2"><div><canvas id="res-g-curva" role="img" aria-label="${esc(t('res.otim_aria'))}"></canvas><div class="legenda">${NOMES.map((o) => `<span><i style="background:${COR_OTIM[o]}"></i>${NOME_OTIM[o]}</span>`).join('')}</div></div><div>${
      tab([t('otim.t_otim'), t('otim.lr'), t('otim.t_melhor_ep'), t('otim.t_melhor'), `F1 ${t('res.i_bil')}`, `F1 ${t('res.i_en')}`, `F1 ${t('res.i_pt')}`, 'AUC'], NOMES.map((o) => [sw(o), fmtM(C(o).lr, 4), pm(C(o).melhor_epoca, C(o).melhor_epoca_dp, 1), fmt(C(o).val_loss, 3), pm(...C(o).f1[F.bil]), pm(...C(o).f1[F.en]), pm(...C(o).f1[F.pt]), fmt(C(o).auc[F.bil][0], 3)]))}</div></div>
      <p class="mini just">${t('res.otim_d', parametrosRef(r))}</p>
      ${tab(['', 'ep 1', 'ep 3', 'ep 5', 'ep 10', 'ep 20', 'ep 40'], r.convergencia.map((c) => [sw(c.otimizador), ...['ep1', 'ep3', 'ep5', 'ep10', 'ep20', 'ep40'].map((k) => fmt(c[k], 3))]))}
      <p class="mini just">${t('res.conv_d', { r40: fmt(r.convergencia.find((c) => c.otimizador === 'RMSProp').ep40, 2) })}</p>
      <h4>${t('res.grade')}</h4>${gradeTab()}<p class="mini just">${t('res.grade_d', parametrosRef(r))}</p></div>`;
    // capacidade
    const cap = (H) => r.capacidade.find((c) => c.H === H), lgb = r.final.find((x) => x.modelo.startsWith('Regress') && x.conjunto === F.bil), mlb = r.final.find((x) => x.modelo.startsWith('MLP') && x.conjunto === F.bil);
    const f1s = [8, 32, 128, 512].map((H) => cap(H).f1_bil);
    h += `<div class="cartao"><h3>${t('res.cap')}</h3>${tab(['H', t('res.params'), t('res.cap_treino'), t('res.cap_val'), `F1 ${t('res.i_en')}`, `F1 ${t('res.i_pt')}`, `F1 ${t('res.i_bil')}`, 'AUC', t('otim.t_melhor_ep')],
      r.capacidade.map((c) => [c.H === 0 ? t('res.logistica') : String(c.H), Number(c.parametros).toLocaleString(t('lang.loc')), fmt(c.treino_loss, 3), pm(c.val_loss, c.val_loss_dp), fmt(c.f1_en, 3), fmt(c.f1_pt, 3), pm(c.f1_bil, c.f1_bil_dp), fmt(c.auc_bil, 3), fmt(c.melhor_epoca, 1)]))}<p class="mini just">${t('res.cap_d', {
      lo: fmt(Math.min(...[8, 32, 128, 512].map((H) => cap(H).f1_bil)), 2), hi: fmt(Math.max(...f1s), 2), f2: pm(cap(2).f1_bil, cap(2).f1_bil_dp), f0: fmt(cap(0).f1_bil, 3), l0: fmt(cap(0).val_loss, 3), v32: fmt(cap(32).val_loss, 3), v8: fmt(cap(8).val_loss, 3), np: Number(r.info.parametros_mlp).toLocaleString(t('lang.loc')),
      lg: fmt(lgb.f1, 3), dpp: fmt((mlb.f1 - lgb.f1) * 100, 0), amlp: fmt(mlb.auc, 3), alg: fmt(lgb.auc, 3) })}</p></div>`;
    // representação
    const rp = (n) => r.representacao.find((x) => x.representacao.startsWith(n));
    h += `<div class="cartao"><h3>${t('res.rep')}</h3>${tab([t('res.rep_nome'), t('res.cap_val'), `F1 ${t('res.i_en')}`, `F1 ${t('res.i_pt')}`, `F1 ${t('res.i_bil')}`, 'AUC', 'F1 neuralchemy', 'F1 xTRam1', 'F1 BR-synth.'],
      r.representacao.map((c) => [esc(c.representacao), fmt(c.val_loss, 3), fmt(c.f1_en, 3), fmt(c.f1_pt, 3), pm(c.f1_bil, c.f1_bil_dp), fmt(c.auc_bil, 3), fmt(c.f1_nc, 3), fmt(c.f1_xt, 3), fmt(c.f1_br, 3)]))}<p class="mini just">${t('res.rep_d', { tf: fmt(rp('TF-IDF (uni').f1_bil, 3), bow: fmt(rp('BoW').f1_bil, 3), bi: fmt(rp('TF-IDF (uni+').f1_bil, 3) })}</p></div>`;
    // generalização
    const mlp = (c) => r.final.find((x) => x.modelo.startsWith('MLP') && x.conjunto === F[c]), lg = (c) => r.final.find((x) => x.modelo.startsWith('Regress') && x.conjunto === F[c]);
    const t1n = { en: 'deepset (teste, só inglês)', neuralchemy: 'neuralchemy (OOD)', xtram1: 'xTRam1 (OOD, sintético)' };
    const t1 = (d, c) => (t1n[c] ? r.trabalho1.find((x) => x.detector === d && x.conjunto === t1n[c]) : null);
    const conj = ['en', 'pt', 'neuralchemy', 'xtram1', 'br'];
    const bs = (c) => {
      const b = [{ nome: t('res.mlp_bil'), cor: COR.azul, v: mlp(c).f1 }, { nome: t('res.mlp_en'), cor: COR.cinza, v: idi('orig', c).f1 }, { nome: t('res.logreg'), cor: COR.azul2, v: lg(c).f1 }];
      for (const [d, cor] of [['DistilBERT', COR.verm], ['BERT-base', COR.verde]]) if (t1(d, c)) b.push({ nome: d + ' (T1)', cor, v: t1(d, c).f1 / 100 });
      return b;
    };
    const t1c = (d, c) => (t1(d, c) ? fmt(t1(d, c).f1 / 100, 3) : '–');
    h += `<div class="cartao"><h3>${t('res.gen')}</h3><div class="barras">${barras(conj.map((c) => ({ rotulo: t(CHAVE_CONJ[c]), barras: bs(c) })))}</div>
      ${tab([t('res.d_conj'), 'MLP F1', 'MLP AUC', `${t('res.mlp_en')} F1`, `${t('res.logreg')} F1`, 'DistilBERT F1', 'BERT-base F1'], conj.map((c) => [t(CHAVE_CONJ[c]), pm(mlp(c).f1, mlp(c).f1_dp), fmt(mlp(c).auc, 3), fmt(idi('orig', c).f1, 3), fmt(lg(c).f1, 3), t1c('DistilBERT', c), t1c('BERT-base', c)]))}
      <p class="mini just">${t('res.gen_d', { br: fmt(mlp('br').f1, 3), abr: fmt(mlp('br').auc, 3) })}</p></div>`;
    // custo
    const cu = r.custo, razao = cu.distilbert.p50_ms / cu.mlp.ms_por_texto;
    h += `<div class="cartao"><h3>${t('res.custo')}</h3>${tab([t('res.c_modelo'), t('res.params'), t('res.c_ms'), t('res.c_tam')], [
      ['MLP (H=8)', Number(cu.mlp.parametros).toLocaleString(t('lang.loc')), fmt(cu.mlp.ms_por_texto, 2), fmt(cu.mlp.parametros * 4 / 1e6, 2) + ' (fp32)'],
      ['DistilBERT (T1)', fmt(cu.distilbert.params_m, 1) + ' M', fmt(cu.distilbert.p50_ms, 1) + ' (p50)', fmt(cu.distilbert.tamanho_mb, 0)],
      ['BERT-base (T1)', fmt(cu.bert_base.params_m, 1) + ' M', fmt(cu.bert_base.p50_ms, 1) + ' (p50)', fmt(cu.bert_base.tamanho_mb, 0)]])}
      <p class="just"><strong>${t('res.custo_razao', { x: fmt(razao, 0) })}</strong> ${t('res.custo_d', { mb: fmt(cu.mlp.tamanho_mb_json, 2) })}</p></div>`;
    // termos
    h += `<div class="cartao"><h3>${t('res.termos')}</h3>${tab([t('res.termo'), t('res.queda'), t('res.n_textos')], r.termos.map((x) => [esc(x.termo), fmt(x.queda * 100, 1) + ' p.p.', x.n]))}<p class="mini just">${t('res.termos_d')}</p></div>`;
    // mapeamento
    const mapa = ['dados', 'tfidf', 'grad', 'otim', 'cap', 'rep', 'idi', 'gen', 'custo', 'termos'].map((k) => [t('map.' + k + '.r'), t('map.' + k + '.l')]);
    h += `<div class="cartao"><h3>${t('res.mapa')}</h3>${tab([t('res.mapa_r'), t('res.mapa_l')], mapa, 'tab-txt')}<p class="mini just">${t('res.mapa_d')}</p></div>`;
    $('#res-corpo').innerHTML = h;
    const cv = $('#res-g-curva');
    const g = new GraficoLinhas(cv, { altura: 240, fmtY: (v) => fmt(v, 2) });
    g.definir({ xMax: 40, yMin: 0, yMax: 1.1, rotX: t('treinar.epoca'), rotY: t('res.val_media'),
      series: NOMES.map((o) => ({ nome: o, cor: COR_OTIM[o], y: r.curvas_medias[o].val })) });
  }
  onLang(render);
  render();
}
