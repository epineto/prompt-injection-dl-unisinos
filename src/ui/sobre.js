// Seção #sobre: links, tabela Aula → seção, limitações e dados/licenças (textos vêm dos dicionários).
import { $, esc } from './util.js';
import { t, onLang } from './i18n.js';

const REPO = 'https://github.com/epineto/prompt-injection-dl-unisinos';
const DEMO = 'https://epineto.github.io/prompt-injection-dl-unisinos/';
const AULAS = [['1', '#treinar #testar'], ['2', '#backprop #treinar #testar'], ['3', '#backprop #treinar #resultados'], ['4', '#otimizadores #treinar'], ['5', '#testar #resultados'], ['6', '#intro #testar #resultados']];

export function iniciarSobre(S) {
  function render() {
    $('#sobre-links').innerHTML = [[REPO, 'sobre.l_repo'], [DEMO, 'sobre.l_demo'], [REPO + '/blob/main/docs/REPRODUCAO.md', 'sobre.l_reprod'], [REPO + '/blob/main/docs/NAVEGACAO.md', 'sobre.l_nav'],
      [REPO + '/blob/main/reference/python/trabalho2_dl.py', 'sobre.l_script'], [REPO + '/tree/main/notebook', 'sobre.l_nb']]
      .map(([h, k]) => `<li><a href="${h}" target="_blank" rel="noopener">${esc(t(k))}</a></li>`).join('');
    $('#sobre-aulas').innerHTML = `<thead><tr><th>${t('sobre.t_aula')}</th><th>${t('sobre.t_conteudo')}</th><th>${t('sobre.t_secao')}</th></tr></thead><tbody>${AULAS.map(([n, secs]) =>
      `<tr><th scope="row">${t('sobre.aula')} ${n}</th><td>${t('sobre.a' + n)}</td><td>${secs.split(' ').map((s) => `<a href="${s}">${s}</a>`).join(' ')}</td></tr>`).join('')}</tbody>`;
    $('#sobre-limites').innerHTML = [1, 2, 3, 4, 5, 6, 7].map((i) => `<li>${t('sobre.lim' + i)}</li>`).join('');
    const f = S.ds.fontes;
    $('#sobre-dados').innerHTML = ['deepset', 'neuralchemy', 'xtram1'].map((k) => `<li><a href="${f[k].url}" target="_blank" rel="noopener">${esc(f[k].nome)}</a>: ${esc(t('sobre.lic_' + k))}</li>`).join('');
  }
  onLang(render);
  render();
}
