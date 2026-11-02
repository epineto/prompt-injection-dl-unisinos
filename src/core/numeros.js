// Leitura de números digitados em pt-BR (vírgula decimal) ou en (ponto). Devolve NaN se inválido.
export function lerNumero(s) {
  if (typeof s === 'number') return s;
  let t = String(s ?? '').trim().replace(/\s+/g, '');
  if (t === '') return NaN;
  const virg = t.lastIndexOf(','), pto = t.lastIndexOf('.');
  if (virg >= 0 && pto >= 0) { // 1.234,5 (pt) ou 1,234.5 (en): o último separador é o decimal
    t = virg > pto ? t.replace(/\./g, '').replace(',', '.') : t.replace(/,/g, '');
  } else if (virg >= 0) {
    t = t.replace(',', '.');
  }
  return /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t) ? Number(t) : NaN;
}

export const fmtNum = (n, d = 2, lang = 'pt') => Number(n).toLocaleString(lang === 'pt' ? 'pt-BR' : 'en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
export const fmtMax = (n, max = 3, lang = 'pt') => Number(n).toLocaleString(lang === 'pt' ? 'pt-BR' : 'en-US', { maximumFractionDigits: max });
