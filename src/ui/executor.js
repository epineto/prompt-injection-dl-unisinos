// Controla um Web Worker de treino: uma execução por vez, com progresso e cancelamento.
let seq = 0;
export function iniciarTreino(cfg, aoEpoca) {
  const w = new Worker(new URL('../worker.js', import.meta.url), { type: 'module' });
  const id = ++seq;
  let encerrado = false;
  const promessa = new Promise((resolve, reject) => {
    w.onmessage = (ev) => {
      const m = ev.data;
      if (m.id !== id) return;
      if (m.tipo === 'epoca') aoEpoca && aoEpoca(m);
      else if (m.tipo === 'fim') { encerrado = true; w.terminate(); resolve(m.resultado); }
      else if (m.tipo === 'erro') { encerrado = true; w.terminate(); reject(new Error(m.mensagem)); }
    };
    w.onerror = (e) => { encerrado = true; w.terminate(); reject(new Error(e.message || 'erro no worker')); };
    w.postMessage({ tipo: 'treinar', id, cfg });
  });
  return { promessa, parar() { if (!encerrado) w.postMessage({ tipo: 'parar', id }); } };
}
