// Números da comparação de otimizadores (resultados.json) usados nos textos de #otimizadores e #resultados.
import { fmt } from './i18n.js';

export function parametrosRef(r) {
  const o = r.otimizadores, g = (n, lr) => r.grade_lr.find((x) => x.otimizador === n && x.lr === lr);
  return {
    rm: fmt(o.RMSProp.melhor_epoca, 1), ad: fmt(o.Adam.melhor_epoca, 1), cl: fmt((o.SGD.melhor_epoca + o['SGD+Momentum'].melhor_epoca) / 2, 0),
    vs: fmt(o.SGD.val_loss, 3), vm: fmt(o['SGD+Momentum'].val_loss, 3), vr: fmt(o.RMSProp.val_loss, 3), va: fmt(o.Adam.val_loss, 3),
    a01: fmt(g('Adam', 0.01).melhor_epoca, 0), r01: fmt(g('RMSProp', 0.01).melhor_epoca, 0),
    va01: fmt(g('Adam', 0.01).val_loss, 3), vr01: fmt(g('RMSProp', 0.01).val_loss, 3), va001: fmt(g('Adam', 0.001).val_loss, 3), vr001: fmt(g('RMSProp', 0.001).val_loss, 3),
  };
}
