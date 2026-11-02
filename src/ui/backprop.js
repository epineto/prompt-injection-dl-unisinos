// Seção #backprop: mini-rede 2-2-1 (XOR) com todos os números do ciclo forward → perda → gradiente → atualização.
import { iniciar, clonar, forward, perda, perdaMedia, gradienteAmostra, gradienteLote, atualizar, XOR_X, XOR_Y } from '../core/xor.js';
import { lerNumero } from '../core/numeros.js';
import { $, COR } from './util.js';
import { t, fmt, onLang } from './i18n.js';
import { GraficoLinhas } from './graficos.js';

const f4 = (v) => fmt(v, 4);
const sg = (v) => (v >= 0 ? '+' : '−') + fmt(Math.abs(v), 4);

export function iniciarBackprop() {
  const gPerda = new GraficoLinhas($('#bp-g-perda'), { altura: 170, fmtY: (v) => fmt(v, 2) });
  const cvF = $('#bp-g-fronteira');
  let W, hist, foco, ultimo, atualizacoes, rodando = false, raf = 0;

  const lr = () => lerNumero($('#bp-lr').value);
  const modo = () => Number($('#bp-modo').value);

  function reiniciar() {
    const s = Number($('#bp-seed').value);
    W = iniciar(Number.isInteger(s) && s >= 1 ? s : 1);
    hist = [perdaMedia(W)]; foco = 0; ultimo = null; atualizacoes = 0;
    parar(); desenhar();
  }

  function passo() {
    const eta = lr();
    const av = $('#bp-aviso');
    if (!(eta > 0 && eta <= 20)) { av.hidden = false; av.textContent = t('treinar.invalido'); return false; }
    av.hidden = true;
    const antes = clonar(W), m = modo();
    const idx = foco;
    const cadeia = gradienteAmostra(W, XOR_X[idx], XOR_Y[idx]);
    const g = m === 1 ? cadeia.g : gradienteLote(W);
    atualizar(W, g, eta);
    ultimo = { antes, depois: clonar(W), cadeia, g, eta, idx, modo: m, perdaAntes: hist[hist.length - 1] };
    if (m === 1) foco = (foco + 1) % 4;
    atualizacoes++;
    hist.push(perdaMedia(W));
    return true;
  }
  const epoca = () => { for (let i = 0; i < (modo() === 1 ? 4 : 1); i++) if (!passo()) return false; return true; };

  function parar() { rodando = false; cancelAnimationFrame(raf); const b = $('#bp-treinar'); b.setAttribute('aria-pressed', 'false'); b.textContent = t('bp.treinar'); }
  function ciclo() {
    if (!rodando) return;
    for (let i = 0; i < 6; i++) if (!epoca()) { parar(); return; }
    desenhar();
    if (hist[hist.length - 1] < 0.003 || hist.length > 40000) { parar(); desenhar(); return; }
    raf = requestAnimationFrame(ciclo);
  }

  // ---------- desenho ----------
  function rede() {
    const a = ultimo ? ultimo.antes : W, amostra = ultimo ? ultimo.idx : foco;
    const f = forward(a, XOR_X[amostra]);
    const P = { x: [[70, 60], [70, 190]], h: [[230, 60], [230, 190]], o: [350, 125] };
    const cor = (w) => (w >= 0 ? COR.azul2 : COR.verm), larg = (w) => 1 + Math.min(Math.abs(w), 3) * 1.6;
    let s = '<svg viewBox="0 0 430 260" role="img" aria-label="' + t('bp.rede_aria') + '">';
    const aresta = (p, q, w, frac) => {
      const lx = p[0] + (q[0] - p[0]) * frac, ly = p[1] + (q[1] - p[1]) * frac;
      return `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="${cor(w)}" stroke-width="${larg(w)}" opacity=".75"/><text x="${lx}" y="${ly - 4}" text-anchor="middle" class="w-t" stroke="#fff" stroke-width="3" paint-order="stroke">${sg(w)}</text>`;
    };
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) s += aresta(P.x[i], P.h[j], a.W1[i][j], i === j ? 0.5 : (i === 0 ? 0.3 : 0.7));
    for (let j = 0; j < 2; j++) s += aresta(P.h[j], P.o, a.W2[j], 0.5);
    const no = (p, rot, val, cl) => `<circle cx="${p[0]}" cy="${p[1]}" r="22" fill="#fff" stroke="${cl}" stroke-width="3"/><text x="${p[0]}" y="${p[1] - 2}" text-anchor="middle" class="n-t">${rot}</text><text x="${p[0]}" y="${p[1] + 12}" text-anchor="middle" class="n-v">${val}</text>`;
    s += no(P.x[0], 'x₁', fmt(XOR_X[amostra][0], 0), COR.cinza) + no(P.x[1], 'x₂', fmt(XOR_X[amostra][1], 0), COR.cinza);
    s += no(P.h[0], 'a₁', fmt(f.a1[0], 2), COR.azul) + no(P.h[1], 'a₂', fmt(f.a1[1], 2), COR.azul) + no(P.o, 'p', fmt(f.p, 2), f.p >= 0.5 ? COR.verm : COR.verde);
    s += `<text x="230" y="22" text-anchor="middle" class="b-t">b₁ = ${sg(a.b1[0])}</text><text x="230" y="250" text-anchor="middle" class="b-t">b₂ = ${sg(a.b1[1])}</text><text x="350" y="170" text-anchor="middle" class="b-t">b = ${sg(a.b2)}</text>`;
    s += `<text x="70" y="22" text-anchor="middle" class="b-t">${t('bp.entrada')}</text>`;
    $('#bp-rede').innerHTML = s + '</svg>';
    const bt = $('#bp-amostras');
    if (!bt.childElementCount) {
      XOR_X.forEach((x, i) => { const b = document.createElement('button'); b.type = 'button'; b.className = 'chip'; b.dataset.i = i; b.addEventListener('click', () => { foco = i; ultimo = null; desenhar(); }); bt.append(b); });
    }
    [...bt.children].forEach((b, i) => { b.textContent = `(${XOR_X[i][0]}, ${XOR_X[i][1]}) → ${XOR_Y[i]}`; b.setAttribute('aria-pressed', String(i === amostra)); });
  }

  function fronteira() {
    const w = Math.max(cvF.clientWidth, 120);
    const dpr = window.devicePixelRatio || 1;
    cvF.style.height = w + 'px';
    cvF.width = Math.round(w * dpr); cvF.height = Math.round(w * dpr);
    const g = cvF.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ext = [-0.25, 1.25], N = 64;
    const oc = document.createElement('canvas'); oc.width = N; oc.height = N;
    const og = oc.getContext('2d'), im = og.createImageData(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = ext[0] + ((i + 0.5) / N) * (ext[1] - ext[0]), y = ext[1] - ((j + 0.5) / N) * (ext[1] - ext[0]);
      const p = forward(W, [x, y]).p, al = 0.1 + 0.5 * Math.abs(p - 0.5) * 2, c = p >= 0.5 ? [192, 57, 43] : [46, 125, 91], k = (j * N + i) * 4;
      for (let q = 0; q < 3; q++) im.data[k + q] = Math.round(255 + (c[q] - 255) * al);
      im.data[k + 3] = 255;
    }
    og.putImageData(im, 0, 0);
    g.imageSmoothingEnabled = true; g.drawImage(oc, 0, 0, w, w);
    const px = (v) => ((v - ext[0]) / (ext[1] - ext[0])) * w, py = (v) => ((ext[1] - v) / (ext[1] - ext[0])) * w;
    XOR_X.forEach((x, i) => {
      g.fillStyle = XOR_Y[i] ? COR.verm : COR.verde; g.strokeStyle = '#fff'; g.lineWidth = 2.5;
      g.beginPath(); g.arc(px(x[0]), py(x[1]), 9, 0, 2 * Math.PI); g.fill(); g.stroke();
      g.fillStyle = '#fff'; g.font = 'bold 11px Arial'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(XOR_Y[i]), px(x[0]), py(x[1]));
    });
    g.fillStyle = '#4A4F6A'; g.font = '11px Arial'; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    g.fillText('x₁ →', w / 2, w - 4); g.save(); g.translate(10, w / 2); g.rotate(-Math.PI / 2); g.fillText('x₂ →', 0, 0); g.restore();
    cvF.setAttribute('aria-label', t('bp.fronteira_aria', { n: XOR_X.filter((x, i) => (forward(W, x).p >= 0.5 ? 1 : 0) === XOR_Y[i]).length }));
  }

  function numeros() {
    const box = $('#bp-num');
    if (!ultimo) { box.innerHTML = `<p class="just">${t('bp.sem_passo')}</p>`; return; }
    const { antes: a, depois: d, cadeia: c, g, eta, idx, modo: m } = ultimo, x = XOR_X[idx], y = XOR_Y[idx], f = c.f;
    const L = perda(f.p, y);
    const eq = (s) => `<div class="eq">${s}</div>`;
    let h = `<p class="just"><strong>${t('bp.amostra_foco')}</strong> x = (${x[0]}, ${x[1]}), y = ${y}. ${m === 1 ? t('bp.nota_amostra') : t('bp.nota_lote')}</p><div class="cols-num"><div>`;
    h += `<h4>1 · ${t('bp.fwd')}</h4>`;
    for (let j = 0; j < 2; j++) {
      h += eq(`z₁${'₁₂'[j]} = x₁·W₁${'₁₂'[0]}${'₁₂'[j]} + x₂·W₁${'₁₂'[1]}${'₁₂'[j]} + b₁${'₁₂'[j]} = ${x[0]}·(${sg(a.W1[0][j])}) + ${x[1]}·(${sg(a.W1[1][j])}) + (${sg(a.b1[j])}) = <b>${f4(f.z1[j])}</b>`);
      h += eq(`a${'₁₂'[j]} = σ(z₁${'₁₂'[j]}) = 1/(1 + e^(−z)) = <b>${f4(f.a1[j])}</b>`);
    }
    h += eq(`z₂ = a₁·W₂₁ + a₂·W₂₂ + b₂ = ${f4(f.a1[0])}·(${sg(a.W2[0])}) + ${f4(f.a1[1])}·(${sg(a.W2[1])}) + (${sg(a.b2)}) = <b>${f4(f.z2)}</b>`);
    h += eq(`p = σ(z₂) = <b>${f4(f.p)}</b>`);
    h += `<h4>2 · ${t('bp.perda_t')}</h4>`;
    h += eq(`L = −[y·ln p + (1−y)·ln(1−p)] = <b>${f4(L)}</b>`);
    h += eq(`${t('bp.perda_media')} = <b>${f4(ultimo.perdaAntes)}</b> → <b>${f4(hist[hist.length - 1])}</b>`);
    h += `</div><div><h4>3 · ${t('bp.cadeia')}</h4>`;
    h += eq(`∂L/∂z₂ = p − y = ${f4(f.p)} − ${y} = <b>${sg(c.dz2)}</b>`);
    for (let j = 0; j < 2; j++) h += eq(`∂L/∂W₂${'₁₂'[j]} = ∂L/∂z₂·a${'₁₂'[j]} = ${sg(c.dz2)}·${f4(f.a1[j])} = <b>${sg(c.g.W2[j])}</b>`);
    for (let j = 0; j < 2; j++) h += eq(`∂L/∂a${'₁₂'[j]} = ∂L/∂z₂·W₂${'₁₂'[j]} = ${sg(c.dz2)}·(${sg(a.W2[j])}) = ${sg(c.da1[j])}`);
    for (let j = 0; j < 2; j++) h += eq(`∂L/∂z₁${'₁₂'[j]} = ∂L/∂a${'₁₂'[j]}·a${'₁₂'[j]}(1−a${'₁₂'[j]}) = ${sg(c.da1[j])}·${f4(f.a1[j] * (1 - f.a1[j]))} = <b>${sg(c.dz1[j])}</b>`);
    h += eq(`∂L/∂W₁ᵢⱼ = xᵢ·∂L/∂z₁ⱼ ${t('bp.e_bias')}`);
    h += `</div></div><h4>4 · ${t('bp.upd')}: W ← W − η·∂L/∂W (η = ${fmt(eta, 3)}${m === 4 ? ', ' + t('bp.grad_medio') : ''})</h4>`;
    const linhas = [];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) linhas.push([`W₁${'₁₂'[i]}${'₁₂'[j]}`, a.W1[i][j], g.W1[i][j], d.W1[i][j]]);
    for (let j = 0; j < 2; j++) linhas.push([`b₁${'₁₂'[j]}`, a.b1[j], g.b1[j], d.b1[j]]);
    for (let j = 0; j < 2; j++) linhas.push([`W₂${'₁₂'[j]}`, a.W2[j], g.W2[j], d.W2[j]]);
    linhas.push(['b₂', a.b2, g.b2, d.b2]);
    h += `<div class="rolagem"><table class="tab tab-upd"><thead><tr><th>${t('bp.t_par')}</th><th>${t('bp.t_antes')}</th><th>∂L/∂W</th><th>${t('bp.t_depois')}</th></tr></thead><tbody>${linhas.map(([n, v, gg, nv]) => `<tr><th scope="row">${n}</th><td>${sg(v)}</td><td>${sg(gg)}</td><td><b>${sg(nv)}</b></td></tr>`).join('')}</tbody></table></div>`;
    box.innerHTML = h;
  }

  function resumo() {
    const acertos = XOR_X.filter((x, i) => (forward(W, x).p >= 0.5 ? 1 : 0) === XOR_Y[i]).length;
    $('#bp-resumo').textContent = t('bp.resumo', { n: atualizacoes, l: fmt(hist[hist.length - 1], 4), a: acertos });
  }

  function desenhar() {
    rede(); fronteira(); numeros(); resumo();
    gPerda.definir({ xMax: Math.max(hist.length, 2), yMin: 0, yMax: Math.max(0.75, hist[0] * 1.05), rotX: t('bp.g_x'), rotY: t('treinar.perda'), series: [{ nome: 'perda', cor: COR.verm, y: hist }] });
  }

  $('#bp-passo').addEventListener('click', () => { parar(); if (passo()) desenhar(); });
  $('#bp-epoca').addEventListener('click', () => { parar(); if (epoca()) desenhar(); });
  $('#bp-treinar').addEventListener('click', () => {
    if (rodando) { parar(); return; }
    if (!(lr() > 0)) { $('#bp-aviso').hidden = false; $('#bp-aviso').textContent = t('treinar.invalido'); return; }
    rodando = true; const b = $('#bp-treinar'); b.setAttribute('aria-pressed', 'true'); b.textContent = t('bp.pausar'); ciclo();
  });
  $('#bp-reset').addEventListener('click', reiniciar);
  $('#bp-seed').addEventListener('change', reiniciar);
  $('#bp-modo').addEventListener('change', () => { ultimo = null; desenhar(); });
  new ResizeObserver(() => fronteira()).observe(cvF);
  onLang(() => { desenhar(); const b = $('#bp-treinar'); b.textContent = rodando ? t('bp.pausar') : t('bp.treinar'); });
  reiniciar();
  return { reiniciar };
}
