// MLP V -> H (ReLU) -> 1 (sigmoide), perda BCE, backpropagation à mão (Aulas 2 e 3).
// Entrada esparsa: cada amostra é {idx: Int32Array, val: Float64Array}. Pesos: W1 (V*H, linha = termo), b1 (H), W2 (H), b2 (1).
import { normal } from './rng.js';

export const nParametros = (V, H) => V * H + H + H + 1; // Aula 4: mesma conta dos 641 parâmetros

// Inicialização He/Kaiming normal (std = sqrt(2/fan_in)) nas duas camadas e vieses nulos.
// Obs.: o PyTorch usa Kaiming uniforme com a=sqrt(5) por padrão; a He normal é a escolha ensinada para ReLU (Aula 4).
export function inicializar(V, H, rng) {
  const W1 = new Float64Array(V * H), W2 = new Float64Array(H);
  const s1 = Math.sqrt(2 / V), s2 = Math.sqrt(2 / H);
  for (let i = 0; i < W1.length; i++) W1[i] = normal(rng) * s1;
  for (let i = 0; i < H; i++) W2[i] = normal(rng) * s2;
  return { V, H, W1, b1: new Float64Array(H), W2, b2: new Float64Array(1) };
}

export const copiar = (P) => ({ V: P.V, H: P.H, W1: P.W1.slice(), b1: P.b1.slice(), W2: P.W2.slice(), b2: P.b2.slice() });
export const sigmoide = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

// z1 = x·W1 + b1 ; h = ReLU(z1) ; z2 = h·W2 + b2 ; p = σ(z2)
export function forward(P, x, buf) {
  const H = P.H, z1 = buf ? buf.z1 : new Float64Array(H);
  for (let h = 0; h < H; h++) z1[h] = P.b1[h];
  for (let k = 0; k < x.idx.length; k++) {
    const base = x.idx[k] * H, v = x.val[k];
    for (let h = 0; h < H; h++) z1[h] += v * P.W1[base + h];
  }
  let z2 = P.b2[0];
  for (let h = 0; h < H; h++) if (z1[h] > 0) z2 += z1[h] * P.W2[h];
  return { z1, z2, p: sigmoide(z2) };
}

export const prever = (P, X) => { const buf = { z1: new Float64Array(P.H) }; return Float64Array.from(X, (x) => forward(P, x, buf).p); };

// Perda BCE pela forma estável em z (igual ao BCEWithLogitsLoss do PyTorch).
export const bceLogit = (z, y) => Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));

// BCE média sobre probabilidades, com recorte em [1e-12, 1-1e-12] (igual a injecao.bce).
export function bce(p, y, eps = 1e-12) {
  let s = 0;
  for (let i = 0; i < p.length; i++) { const q = Math.min(Math.max(p[i], eps), 1 - eps); s -= y[i] * Math.log(q) + (1 - y[i]) * Math.log(1 - q); }
  return s / p.length;
}

export function perdaMedia(P, X, y) {
  const buf = { z1: new Float64Array(P.H) };
  let s = 0;
  for (let i = 0; i < X.length; i++) s += bceLogit(forward(P, X[i], buf).z2, y[i]);
  return s / X.length;
}

export const novosGradientes = (P) => ({ W1: new Float64Array(P.W1.length), b1: new Float64Array(P.H), W2: new Float64Array(P.H), b2: new Float64Array(1) });

// Gradientes da perda média das amostras `ids` de X (regra da cadeia à mão):
// dL/dz2 = (p - y)/n ; dL/dW2 = h·dz2 ; dh = dz2·W2 ; dz1 = dh·1[z1>0] ; dL/dW1 = xᵀ·dz1.
export function gradientes(P, X, y, ids, g = novosGradientes(P)) {
  const H = P.H, n = ids ? ids.length : X.length;
  g.W1.fill(0); g.b1.fill(0); g.W2.fill(0); g.b2.fill(0);
  const z1buf = new Float64Array(H), dz1 = new Float64Array(H);
  let perda = 0;
  for (let t = 0; t < n; t++) {
    const i = ids ? ids[t] : t, x = X[i];
    const { z2, p } = forward(P, x, { z1: z1buf });
    perda += bceLogit(z2, y[i]);
    const dz2 = (p - y[i]) / n;
    g.b2[0] += dz2;
    for (let h = 0; h < H; h++) {
      if (z1buf[h] > 0) { g.W2[h] += z1buf[h] * dz2; dz1[h] = dz2 * P.W2[h]; } else dz1[h] = 0;
      g.b1[h] += dz1[h];
    }
    for (let k = 0; k < x.idx.length; k++) {
      const base = x.idx[k] * H, v = x.val[k];
      for (let h = 0; h < H; h++) g.W1[base + h] += v * dz1[h];
    }
  }
  return { g, perda: perda / n };
}
