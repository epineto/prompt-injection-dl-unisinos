// Gráficos próprios em canvas/SVG (sem bibliotecas). / Own canvas/SVG charts, no libraries.
import { COR } from './util.js';

const css = (n, f) => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || f;

// Gráfico de linhas com redesenho automático ao mudar de tamanho.
// dados: {series:[{nome, cor, y:[...], tracejado, largura}], marcas:[{x, y, cor}], xMax, yMin, yMax, rotX, rotY, fmtY}
export class GraficoLinhas {
  constructor(canvas, { altura = 230, fmtY = (v) => String(v) } = {}) {
    this.c = canvas; this.altura = altura; this.fmtY = fmtY; this.dados = { series: [], marcas: [] };
    this.c.style.width = '100%'; this.c.style.height = altura + 'px'; this.c.style.display = 'block';
    this._ro = new ResizeObserver(() => this.desenhar());
    this._ro.observe(canvas);
  }
  definir(dados) { this.dados = dados; this.desenhar(); }
  desenhar() {
    const c = this.c, w = c.clientWidth, h = this.altura;
    if (!w) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const { series, marcas = [], rotX = '', rotY = '' } = this.dados;
    const m = { l: 46, r: 10, t: 10, b: 32 };
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    let xMax = this.dados.xMax || 1, yMin = this.dados.yMin, yMax = this.dados.yMax;
    const todos = series.flatMap((s) => s.y.filter((v) => Number.isFinite(v)));
    if (yMin === undefined) yMin = 0;
    if (yMax === undefined) yMax = todos.length ? Math.max(...todos) * 1.08 : 1;
    if (!(yMax > yMin)) yMax = yMin + 1;
    const X = (x) => m.l + ((x - 1) / Math.max(xMax - 1, 1)) * pw;
    const Y = (y) => m.t + ph - ((Math.min(Math.max(y, yMin), yMax) - yMin) / (yMax - yMin)) * ph;
    g.font = '11px Arial, sans-serif'; g.lineWidth = 1;
    const txt = css('--texto-suave', '#4A4F6A'), borda = css('--borda', '#DDE2EE');
    g.fillStyle = txt; g.strokeStyle = borda; g.textAlign = 'right'; g.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const v = yMin + ((yMax - yMin) * i) / 4, y = Y(v);
      g.beginPath(); g.moveTo(m.l, y); g.lineTo(w - m.r, y); g.stroke();
      g.fillText(this.fmtY(v), m.l - 6, y);
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    const bruto = xMax / 6, mag = Math.pow(10, Math.floor(Math.log10(Math.max(bruto, 1))));
    const passo = Math.max(1, [1, 2, 5, 10].map((k) => k * mag).find((v) => v >= bruto) || mag * 10);
    const ticks = new Set([1, xMax]);
    for (let x = passo; x < xMax - passo / 3; x += passo) ticks.add(x);
    for (const x of ticks) { g.fillText(String(x), X(x), h - m.b + 5); }
    g.fillText(rotX, m.l + pw / 2, h - 14);
    g.save(); g.translate(11, m.t + ph / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'middle'; g.fillText(rotY, 0, 0); g.restore();
    g.strokeStyle = txt; g.beginPath(); g.moveTo(m.l, m.t); g.lineTo(m.l, m.t + ph); g.lineTo(w - m.r, m.t + ph); g.stroke();
    for (const s of series) {
      g.strokeStyle = s.cor; g.lineWidth = s.largura || 2; g.setLineDash(s.tracejado ? [5, 4] : []);
      g.beginPath();
      let ini = true;
      s.y.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        // valores acima de yMax são recortados na borda superior (divergência)
        if (ini) { g.moveTo(X(i + 1), Y(v)); ini = false; } else g.lineTo(X(i + 1), Y(v));
      });
      g.stroke(); g.setLineDash([]);
    }
    for (const mk of marcas) {
      g.fillStyle = mk.cor; g.strokeStyle = '#fff'; g.lineWidth = 2;
      g.beginPath(); g.arc(X(mk.x), Y(mk.y), 5, 0, 2 * Math.PI); g.fill(); g.stroke();
    }
  }
}

// Medidor semicircular (SVG) de 0 a 1 com marca do limiar.
export function gauge(p, limiar, rotulos = { zero: '0', um: '1' }) {
  const cx = 150, cy = 140, r = 110;
  const pt = (v, rr = r) => { const a = Math.PI * (1 - v); return [cx + rr * Math.cos(a), cy - rr * Math.sin(a)]; };
  const arco = (v0, v1, cor) => { const [x0, y0] = pt(v0), [x1, y1] = pt(v1); return `<path d="M${x0} ${y0} A${r} ${r} 0 0 1 ${x1} ${y1}" fill="none" stroke="${cor}" stroke-width="22"/>`; };
  const seg = 24;
  let arcos = '';
  for (let i = 0; i < seg; i++) {
    const v0 = i / seg, v1 = (i + 1) / seg, mid = (v0 + v1) / 2;
    const cor = mid < 0.5 ? mix([46, 125, 91], [212, 160, 23], mid / 0.5) : mix([212, 160, 23], [192, 57, 43], (mid - 0.5) / 0.5);
    arcos += arco(v0, v1 + 0.002, cor);
  }
  const [nx, ny] = pt(Math.min(Math.max(p, 0), 1), r - 18);
  const [t0x, t0y] = pt(limiar, r - 16), [t1x, t1y] = pt(limiar, r + 16);
  return `<svg viewBox="0 0 300 170" aria-hidden="true" focusable="false">${arcos}
    <line x1="${t0x}" y1="${t0y}" x2="${t1x}" y2="${t1y}" stroke="#000" stroke-width="4"/>
    <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="${COR.azul}" stroke-width="5" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="9" fill="${COR.azul}"/>
    <text x="${cx - r}" y="${cy + 22}" text-anchor="middle" font-size="12" fill="#4A4F6A">${rotulos.zero}</text>
    <text x="${cx + r}" y="${cy + 22}" text-anchor="middle" font-size="12" fill="#4A4F6A">${rotulos.um}</text></svg>`;
}
const mix = (a, b, t) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`;

// Cor de fundo para contribuição de um termo (queda de probabilidade). >0 vermelho (injeção), <0 verde-azulado (seguro).
export function corContribuicao(d, escala = 0.25) {
  const a = Math.min(Math.abs(d) / escala, 1);
  if (Math.abs(d) < 0.004) return 'transparent';
  return d > 0 ? `rgba(192,57,43,${0.12 + 0.48 * a})` : `rgba(46,125,91,${0.12 + 0.42 * a})`;
}
