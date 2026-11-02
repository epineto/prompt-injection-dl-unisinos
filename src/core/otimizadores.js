// Os quatro otimizadores da Aula 4, com a mesma semântica do PyTorch (torch.optim). Atualizam os vetores in-place.
export const NOMES = ['SGD', 'SGD+Momentum', 'RMSProp', 'Adam'];
export const LR_TRABALHO = { 'SGD': 1.0, 'SGD+Momentum': 0.1, 'RMSProp': 0.01, 'Adam': 0.01 }; // melhores taxas pela validação (tab_grade_lr)

const campos = ['W1', 'b1', 'W2', 'b2'];

export function criarOtimizador(nome, lr, P) {
  const est = () => campos.map((c) => new Float64Array(P[c].length));
  let t = 0;
  if (nome === 'SGD') {
    return { nome, passo(Pm, g) { for (const c of campos) { const w = Pm[c], d = g[c]; for (let i = 0; i < w.length; i++) w[i] -= lr * d[i]; } } };
  }
  if (nome === 'SGD+Momentum') { // PyTorch: buf = mu*buf + g ; p -= lr*buf  (mu = 0,9)
    const mu = 0.9, buf = est();
    return { nome, passo(Pm, g) { campos.forEach((c, k) => { const w = Pm[c], d = g[c], b = buf[k]; for (let i = 0; i < w.length; i++) { b[i] = mu * b[i] + d[i]; w[i] -= lr * b[i]; } }); } };
  }
  if (nome === 'RMSProp') { // PyTorch: sq = a*sq + (1-a)*g^2 ; p -= lr*g/(sqrt(sq)+eps)  (alpha = 0,9; eps = 1e-8)
    const a = 0.9, eps = 1e-8, sq = est();
    return { nome, passo(Pm, g) { campos.forEach((c, k) => { const w = Pm[c], d = g[c], s = sq[k]; for (let i = 0; i < w.length; i++) { s[i] = a * s[i] + (1 - a) * d[i] * d[i]; w[i] -= lr * d[i] / (Math.sqrt(s[i]) + eps); } }); } };
  }
  if (nome === 'Adam') { // betas 0,9/0,999; eps 1e-8; com correção de viés (como torch.optim.Adam)
    const b1 = 0.9, b2 = 0.999, eps = 1e-8, m = est(), v = est();
    return {
      nome,
      passo(Pm, g) {
        t++;
        const c1 = 1 - Math.pow(b1, t), c2s = Math.sqrt(1 - Math.pow(b2, t)), alfa = lr / c1;
        campos.forEach((c, k) => {
          const w = Pm[c], d = g[c], mm = m[k], vv = v[k];
          for (let i = 0; i < w.length; i++) {
            mm[i] = b1 * mm[i] + (1 - b1) * d[i];
            vv[i] = b2 * vv[i] + (1 - b2) * d[i] * d[i];
            w[i] -= alfa * mm[i] / (Math.sqrt(vv[i]) / c2s + eps);
          }
        });
      },
    };
  }
  throw new Error('otimizador desconhecido: ' + nome);
}
