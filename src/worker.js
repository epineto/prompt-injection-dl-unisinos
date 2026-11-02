// Web Worker: treina o MLP no navegador sem travar a página. Mensagens:
//   entrada  {tipo:'treinar', id, cfg}  |  {tipo:'parar', id}
//   saída    {tipo:'epoca', id, ...}  |  {tipo:'fim', id, resultado}  |  {tipo:'erro', id, mensagem}
import { prepararDados } from './core/dados.js';
import { treinar, avaliarConjuntos } from './core/treino.js';

let brutos = null;
const prontos = new Map(); // língua de treino -> dados preparados
const parar = new Set();
const carregar = (treino = 'bil') => {
  brutos ??= Promise.all(['dataset', 'vocab'].map((n) => fetch(new URL(`../data/${n}.json`, import.meta.url)).then((r) => { if (!r.ok) throw new Error(n + ': HTTP ' + r.status); return r.json(); })));
  if (!prontos.has(treino)) prontos.set(treino, brutos.then(([ds, vocab]) => prepararDados(ds, vocab, 'tfidf', treino)));
  return prontos.get(treino);
};
const ceder = () => new Promise((r) => setTimeout(r, 0)); // deixa o worker receber a mensagem "parar" entre épocas

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m.tipo === 'parar') { parar.add(m.id); return; }
  if (m.tipo !== 'treinar') return;
  try {
    const dados = await carregar(m.cfg.treino || 'bil');
    const r = await treinar(m.cfg, dados, {
      aoFimEpoca: async (info) => { self.postMessage({ tipo: 'epoca', id: m.id, ...info }); await ceder(); },
      parar: () => parar.has(m.id),
    });
    const limiar = 0.5;
    self.postMessage({
      tipo: 'fim', id: m.id,
      resultado: {
        curva: r.curva, melhorEp: r.melhorEp, epocasFeitas: r.epocasFeitas, ms: r.ms, parametros: r.parametros, interrompido: parar.has(m.id),
        metricas: avaliarConjuntos(r.melhor, dados, limiar), metricasUltima: avaliarConjuntos(r.ultima, dados, limiar),
      },
    });
  } catch (e) {
    self.postMessage({ tipo: 'erro', id: m.id, mensagem: String(e && e.message || e) });
  } finally { parar.delete(m.id); }
};
