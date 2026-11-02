export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export function el(tag, attrs = {}, filhos = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v; else if (k === 'html') e.innerHTML = v; else e.setAttribute(k, v); }
  for (const f of [].concat(filhos)) e.append(f);
  return e;
}
export const COR = { azul: '#1D2369', azul2: '#4A5AB8', verm: '#C0392B', verde: '#2E7D5B', cinza: '#666666', claro: '#E9ECF3' };
export const COR_OTIM = { 'SGD': '#666666', 'SGD+Momentum': '#4A5AB8', 'RMSProp': '#2E7D5B', 'Adam': '#C0392B' };
export const NOME_OTIM = { 'SGD': 'SGD', 'SGD+Momentum': 'SGD + Momentum', 'RMSProp': 'RMSProp', 'Adam': 'Adam' };
export const fmtInt = (n, lang) => Number(n).toLocaleString(lang === 'pt' ? 'pt-BR' : 'en-US', { maximumFractionDigits: 0 });
// Nomes dos conjuntos nos CSV/JSON do Python (resultados.json) para as chaves do laboratório.
export const NOME_PY = { en: 'deepset EN/DE (teste)', pt: 'deepset PT (teste)', bil: 'bilíngue (teste EN/DE + PT)', neuralchemy: 'neuralchemy (OOD)', xtram1: 'xTRam1 (OOD)', br: 'BR-synthetic (PT nativo, OOD)' };
export const CHAVE_CONJ = { en: 'testar.c_en', pt: 'testar.c_pt', bil: 'testar.c_bil', neuralchemy: 'testar.c_nc', xtram1: 'testar.c_xt', br: 'testar.c_br' };
// Treino do laboratório ('bil', 'orig', 'pt') -> nome na coluna "treino" de tab_idiomas.csv.
export const TREINO_PY = { bil: 'bilingue', orig: 'orig', pt: 'pt' };
export const CHAVE_TREINO = { bil: 'treinar.lang_bil', orig: 'treinar.lang_orig', pt: 'treinar.lang_pt' };
