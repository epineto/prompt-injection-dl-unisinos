import { readFileSync } from 'node:fs';
const ler = (p) => JSON.parse(readFileSync(new URL('../' + p, import.meta.url), 'utf8'));
export const ref = ler('tests/reference.json');
export const dataset = ler('data/dataset.json');
export const vocab = ler('data/vocab.json');
export const modelo = ler('data/modelo.json');
export const resultados = ler('data/resultados.json');
export const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
