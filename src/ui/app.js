// Ponto de entrada: i18n, roteamento por hash (#intro … #sobre), carga dos dados e inicialização das seções.
import * as I18N from './i18n.js';
import { $, $$, NOME_PY } from './util.js';
import { prepararDados, modeloDeJson, conjuntosDeAvaliacao } from '../core/dados.js';
import { criarVetorizador } from '../core/texto.js';
import { prever } from '../core/mlp.js';
import { iniciarTestar, analisar, exemploPadrao } from './testar.js';
import { iniciarTreinar } from './treinar.js';
import { iniciarOtimizadores } from './otimizadores.js';
import { iniciarBackprop } from './backprop.js';
import { iniciarResultados } from './resultados.js';
import { iniciarSobre } from './sobre.js';

const SECOES = ['intro', 'testar', 'treinar', 'otimizadores', 'backprop', 'resultados', 'sobre'];

function rota(primeira) {
  const h = decodeURIComponent(location.hash.slice(1));
  const id = SECOES.includes(h) ? h : 'intro';
  SECOES.forEach((s) => { const e = document.getElementById(s); e.classList.toggle('ativa', s === id); });
  $$('.nav a').forEach((a) => { const on = a.getAttribute('href') === '#' + id; if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  if (!primeira) { window.scrollTo({ top: 0 }); }
  window.dispatchEvent(new Event('resize'));
}

function introAoVivo(S) {
  const ex = exemploPadrao().texto;
  const a = analisar(ex, S);
  $('#pipe-texto').textContent = ex;
  $('#pipe-a').textContent = '"' + ex.slice(0, 26) + '…"';
  $('#pipe-b').textContent = a.toks.slice(0, 4).join(' · ') + ' …';
  $('#pipe-c').textContent = I18N.t('intro.pipe.c_ex', { n: a.x.idx.length, v: S.vet.V });
  $('#pipe-d').textContent = `${S.vet.V} → ${S.P.H} → 1`;
  const nV = Number(S.vet.V).toLocaleString(I18N.t('lang.loc'));
  $('#pipe-c-d').textContent = I18N.t('intro.pipe.c_d', { v: nV });
  $('#pipe-d-d').textContent = I18N.t('intro.pipe.d_d', { v: nV });
  const R = S.res;
  $('#hero-nums').innerHTML = [[Number(R.info.parametros_mlp).toLocaleString(I18N.t('lang.loc')), 'intro.n1'], [I18N.fmt(R.info.latencia_mlp_ms_por_texto, 2) + ' ms', 'intro.n2'], [I18N.fmt(R.otimizadores.Adam.f1[NOME_PY.bil][0], 3), 'intro.n3']]
    .map(([v, k]) => `<div><strong>${v}</strong><span>${I18N.t(k)}</span></div>`).join('');
  $('#pipe-e').textContent = I18N.fmt(a.p * 100, 1) + '%';
}

async function main() {
  I18N.iniciar();
  rota(true);
  window.addEventListener('hashchange', () => rota(false));
  try {
    const get = (n) => fetch(`data/${n}.json`).then((r) => { if (!r.ok) throw new Error(`data/${n}.json: HTTP ${r.status}`); return r.json(); });
    const [ds, vocab, modelo, res] = await Promise.all(['dataset', 'vocab', 'modelo', 'resultados'].map(get));
    const D = prepararDados(ds, vocab);
    const P = modeloDeJson(modelo);
    const S = { ds, vocab, vet: criarVetorizador(vocab), D, P, modelo, res, probs: Object.fromEntries(conjuntosDeAvaliacao(D).map(([k, X]) => [k, prever(P, X)])) };
    window.__lab = S;
    iniciarTestar(S); iniciarTreinar(S); iniciarOtimizadores(S); iniciarBackprop(); iniciarResultados(S); iniciarSobre(S);
    introAoVivo(S);
    I18N.onLang(() => introAoVivo(S));
    document.documentElement.dataset.pronto = '1';
  } catch (e) {
    const er = $('#erro-carga');
    er.hidden = false;
    er.textContent = I18N.t('erro.carga', { m: e.message });
    console.error(e);
  }
}
main();
