// Treino mini-batch com parada antecipada por restauração da melhor época de validação (como injecao.treinar_mlp).
import { mulberry32, permutacao } from './rng.js';
import { inicializar, copiar, gradientes, novosGradientes, perdaMedia, prever, nParametros } from './mlp.js';
import { criarOtimizador } from './otimizadores.js';
import { metricas } from './metricas.js';
import { conjuntosDeAvaliacao } from './dados.js';

// cfg: {otimizador, lr, H, batch, epocas, semente}; dados: {Xtr, ytr, Xva, yva, ...}
// hooks: {aoFimEpoca(info) -> Promise|void (pode ser assíncrono para ceder o laço), parar() -> bool}
export async function treinar(cfg, dados, hooks = {}) {
  const { otimizador, lr, H, batch, epocas, semente } = cfg;
  const { Xtr, ytr, Xva, yva } = dados;
  const rng = mulberry32(semente);
  const P = inicializar(dados.V, H, rng);
  const opt = criarOtimizador(otimizador, lr, P);
  const g = novosGradientes(P);
  const curva = { treino: [], val: [], val_f1: [] };
  let melhor = Infinity, melhorEp = 0, melhorP = copiar(P), ep = 0;
  const t0 = performance.now();
  for (ep = 1; ep <= epocas; ep++) {
    const perm = permutacao(Xtr.length, rng);
    for (let i = 0; i < perm.length; i += batch) {
      gradientes(P, Xtr, ytr, perm.subarray(i, Math.min(i + batch, perm.length)), g);
      opt.passo(P, g);
    }
    const lt = perdaMedia(P, Xtr, ytr), lv = perdaMedia(P, Xva, yva);
    const f1 = metricas(yva, prever(P, Xva)).f1;
    curva.treino.push(lt); curva.val.push(lv); curva.val_f1.push(f1);
    if (lv < melhor - 1e-9) { melhor = lv; melhorEp = ep; melhorP = copiar(P); }
    if (hooks.aoFimEpoca) await hooks.aoFimEpoca({ ep, epocas, treino: lt, val: lv, val_f1: f1, melhorEp, melhor, ms: performance.now() - t0 });
    if (hooks.parar && hooks.parar()) { break; }
  }
  return { melhor: melhorP, ultima: P, melhorEp, curva, epocasFeitas: curva.val.length, ms: performance.now() - t0, parametros: nParametros(dados.V, H) };
}

// Métricas de um modelo nos conjuntos de avaliação do laboratório (en, pt, bil, neuralchemy, xtram1, br).
export function avaliarConjuntos(P, dados, limiar = 0.5) {
  const r = {};
  for (const [k, X, y] of conjuntosDeAvaliacao(dados)) r[k] = metricas(y, prever(P, X), limiar);
  return r;
}
