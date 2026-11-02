// Texto -> tokens -> TF-IDF, a mesma definição de reference/python/injecao.py (Aula 6).
// Python: re.compile(r"\w+").findall(texto.lower()). JavaScript: /[\p{L}\p{N}_]+/gu sobre texto.toLowerCase().
// Diferença conhecida: \w do Python aceita str.isalnum() (letras e números Unicode) e "_"; \p{L}\p{N}_ é o equivalente em JS.
// Marcas combinantes isoladas (categoria M) não casam em nenhum dos dois; a verificação em todos os textos está em tests/.
export const RE_TOKEN = /[\p{L}\p{N}_]+/gu;

export function tokenizar(texto, bigramas = false) {
  const t = texto.toLowerCase().match(RE_TOKEN) || [];
  if (bigramas) { const n = t.length; for (let i = 0; i < n - 1; i++) t.push(t[i] + ' ' + t[i + 1]); }
  return t;
}

// Tokens com posição no texto (para o destaque colorido). Casa-se sobre o texto em minúsculas, como no Python; se a conversão
// mudar o comprimento (ex.: "İ"), o texto exibido é o próprio texto em minúsculas. Devolve {base, tokens:[{ini, fim, token}]}.
export function tokensComPosicao(texto) {
  const baixo = texto.toLowerCase();
  const base = baixo.length === texto.length ? texto : baixo;
  const re = new RegExp(RE_TOKEN.source, 'gu');
  const tokens = [];
  let m;
  while ((m = re.exec(baixo)) !== null) tokens.push({ ini: m.index, fim: m.index + m[0].length, token: m[0] });
  return { base, tokens };
}

// Ajusta o vocabulário: df >= minDf, ordenado por (-df, termo), no máximo maxVocab termos; idf = ln((1+N)/(1+df)) + 1.
export function ajustar(textos, { maxVocab = 2000, minDf = 2, bigramas = false } = {}) {
  const df = new Map();
  for (const t of textos) for (const w of new Set(tokenizar(t, bigramas))) df.set(w, (df.get(w) || 0) + 1);
  const itens = [...df.entries()].filter(([, c]) => c >= minDf);
  itens.sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)); // desempate por ponto de código, como o sorted() do Python
  const sel = itens.slice(0, maxVocab);
  const n = textos.length;
  return { termos: sel.map(([w]) => w), idf: sel.map(([, c]) => Math.log((1 + n) / (1 + c)) + 1), n_documentos: n };
}

export function criarVetorizador(vocab) {
  const indice = new Map(vocab.termos.map((w, i) => [w, i]));
  return { termos: vocab.termos, idf: Float64Array.from(vocab.idf), indice, V: vocab.termos.length };
}

// Vetor esparso {idx, val} ordenado por índice, TF-IDF com normalização L2 (modo 'bow' = presença binária).
export function vetorizarTokens(tokens, vet, modo = 'tfidf') {
  const cont = new Map();
  for (const w of tokens) { const j = vet.indice.get(w); if (j !== undefined) cont.set(j, (cont.get(j) || 0) + 1); }
  const idx = Int32Array.from([...cont.keys()].sort((a, b) => a - b));
  const val = new Float64Array(idx.length);
  let s = 0;
  for (let k = 0; k < idx.length; k++) {
    const c = cont.get(idx[k]);
    val[k] = modo === 'bow' ? 1 : c * vet.idf[idx[k]];
    s += val[k] * val[k];
  }
  const nrm = Math.sqrt(s) || 1;
  for (let k = 0; k < val.length; k++) val[k] /= nrm;
  return { idx, val };
}

export const vetorizar = (texto, vet, modo = 'tfidf') => vetorizarTokens(tokenizar(texto), vet, modo);
