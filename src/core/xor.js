// Mini-rede 2-2-1 com sigmoide nas duas camadas e perda BCE, para o problema XOR (Aulas 2 e 3).
// z1 = x·W1 + b1 ; a1 = σ(z1) ; z2 = a1·W2 + b2 ; p = σ(z2) ; L = -[y ln p + (1-y) ln(1-p)]
import { mulberry32, normal } from './rng.js';
import { sigmoide } from './mlp.js';

export const XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]];
export const XOR_Y = [0, 1, 1, 0];

export function iniciar(semente = 1, escala = 1) {
  const r = mulberry32(semente);
  return { W1: [[normal(r) * escala, normal(r) * escala], [normal(r) * escala, normal(r) * escala]], b1: [0, 0], W2: [normal(r) * escala, normal(r) * escala], b2: 0 };
}
export const clonar = (W) => ({ W1: W.W1.map((l) => l.slice()), b1: W.b1.slice(), W2: W.W2.slice(), b2: W.b2 });

export function forward(W, x) {
  const z1 = [0, 1].map((j) => x[0] * W.W1[0][j] + x[1] * W.W1[1][j] + W.b1[j]);
  const a1 = z1.map(sigmoide);
  const z2 = a1[0] * W.W2[0] + a1[1] * W.W2[1] + W.b2;
  return { z1, a1, z2, p: sigmoide(z2) };
}
export const perda = (p, y) => -(y * Math.log(Math.min(Math.max(p, 1e-12), 1 - 1e-12)) + (1 - y) * Math.log(1 - Math.min(Math.max(p, 1e-12), 1 - 1e-12)));
export const perdaMedia = (W, ids = [0, 1, 2, 3]) => ids.reduce((s, i) => s + perda(forward(W, XOR_X[i]).p, XOR_Y[i]), 0) / ids.length;

// Regra da cadeia passo a passo para UMA amostra (devolve cada fator, para exibir os números).
export function gradienteAmostra(W, x, y) {
  const f = forward(W, x);
  const dz2 = f.p - y;                                            // dL/dz2 (BCE + sigmoide)
  const gW2 = f.a1.map((a) => dz2 * a), gb2 = dz2;                // dL/dW2 = dz2 · a1
  const da1 = W.W2.map((w) => dz2 * w);                           // dL/da1 = dz2 · W2
  const dz1 = da1.map((d, j) => d * f.a1[j] * (1 - f.a1[j]));     // σ'(z) = a(1-a)
  const gW1 = [[x[0] * dz1[0], x[0] * dz1[1]], [x[1] * dz1[0], x[1] * dz1[1]]];
  return { f, dz2, da1, dz1, g: { W1: gW1, b1: dz1.slice(), W2: gW2, b2: gb2 } };
}

// Gradiente médio de um conjunto de amostras.
export function gradienteLote(W, ids = [0, 1, 2, 3]) {
  const s = { W1: [[0, 0], [0, 0]], b1: [0, 0], W2: [0, 0], b2: 0 };
  for (const i of ids) {
    const { g } = gradienteAmostra(W, XOR_X[i], XOR_Y[i]);
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) s.W1[a][b] += g.W1[a][b] / ids.length;
    for (let j = 0; j < 2; j++) { s.b1[j] += g.b1[j] / ids.length; s.W2[j] += g.W2[j] / ids.length; }
    s.b2 += g.b2 / ids.length;
  }
  return s;
}

// W ← W − lr·∂L/∂W (atualização in-place).
export function atualizar(W, g, lr) {
  for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) W.W1[a][b] -= lr * g.W1[a][b];
  for (let j = 0; j < 2; j++) { W.b1[j] -= lr * g.b1[j]; W.W2[j] -= lr * g.W2[j]; }
  W.b2 -= lr * g.b2;
}
