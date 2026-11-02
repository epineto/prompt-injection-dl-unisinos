// Métricas de classificação binária (Aulas 2 e 5: perda não é métrica; com classes desbalanceadas a acurácia engana, usa-se F1).
export function confusao(y, p, limiar = 0.5) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < y.length; i++) {
    const pred = p[i] >= limiar ? 1 : 0;
    if (pred && y[i]) tp++; else if (pred && !y[i]) fp++; else if (!pred && y[i]) fn++; else tn++;
  }
  return { tp, fp, fn, tn };
}

// AUC pela estatística de Mann-Whitney com postos médios nos empates (igual a sklearn.roc_auc_score).
export function auc(y, p) {
  const n = y.length, ord = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b]);
  const posto = new Float64Array(n);
  for (let i = 0; i < n;) {
    let j = i;
    while (j + 1 < n && p[ord[j + 1]] === p[ord[i]]) j++;
    const m = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) posto[ord[k]] = m;
    i = j + 1;
  }
  let np = 0, soma = 0;
  for (let i = 0; i < n; i++) if (y[i] === 1) { np++; soma += posto[i]; }
  const nn = n - np;
  return np && nn ? (soma - np * (np + 1) / 2) / (np * nn) : NaN;
}

export function metricas(y, p, limiar = 0.5) {
  const c = confusao(y, p, limiar), n = y.length;
  const prec = c.tp + c.fp ? c.tp / (c.tp + c.fp) : 0, rec = c.tp + c.fn ? c.tp / (c.tp + c.fn) : 0;
  const f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0;
  return { ...c, acc: (c.tp + c.tn) / n, prec, rec, f1, auc: auc(y, p), fpr: c.fp + c.tn ? c.fp / (c.fp + c.tn) : 0 };
}
