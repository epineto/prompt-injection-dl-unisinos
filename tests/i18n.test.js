import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import pt from '../src/i18n/pt.js';
import en from '../src/i18n/en.js';

test('dicionários PT e EN têm exatamente as mesmas chaves e nenhum valor vazio', () => {
  const a = Object.keys(pt).sort(), b = Object.keys(en).sort();
  assert.deepEqual(a.filter((k) => !(k in en)), [], 'faltam em EN');
  assert.deepEqual(b.filter((k) => !(k in pt)), [], 'faltam em PT');
  for (const k of a) { assert.ok(String(pt[k]).trim() !== '', 'PT vazio: ' + k); assert.ok(String(en[k]).trim() !== '', 'EN vazio: ' + k); }
});

test('todas as chaves data-i18n do HTML existem nos dicionários', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const chaves = new Set();
  for (const m of html.matchAll(/data-i18n(?:-html)?="([^"]+)"/g)) chaves.add(m[1]);
  for (const m of html.matchAll(/data-i18n-attr="([^"]+)"/g)) m[1].split(';').forEach((p) => chaves.add(p.split(':')[1]));
  const falta = [...chaves].filter((k) => !(k in pt));
  assert.deepEqual(falta, []);
});

test('o HTML tem as âncoras de seção exigidas', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  for (const id of ['intro', 'testar', 'treinar', 'otimizadores', 'backprop', 'resultados', 'sobre']) assert.match(html, new RegExp(`id="${id}"`));
});

test('toda chave t(\'…\') usada na interface existe nos dicionários (PT e EN)', async () => {
  const { readdirSync } = await import('node:fs');
  const dir = new URL('../src/ui/', import.meta.url);
  const usadas = new Set();
  for (const f of readdirSync(dir)) {
    const src = readFileSync(new URL(f, dir), 'utf8');
    for (const m of src.matchAll(/\bt\('([a-z0-9_.]+)'/g)) usadas.add(m[1]);
    for (const m of src.matchAll(/'((?:testar|treinar|otim|res|sobre|map|bp|intro|erro)\.[a-z0-9_.]+)'/g)) usadas.add(m[1]);
  }
  const falta = [...usadas].filter((k) => !(k in pt) || !(k in en)).filter((k) => !['sobre.a', 'sobre.lim', 'sobre.lic_', 'map.', 'testar.ex.'].includes(k));
  assert.deepEqual(falta, []);
});

test('nenhum número do trabalho monolíngue sobrevive nos textos da interface', () => {
  const proibidos = [/0,873/, /0\.873/, /31\.521/, /31,521/, /\b983 (termos|terms|pesos|weights|números|numbers)/, /436 amostras/, /436 training samples/];
  for (const [nome, d] of [['pt', pt], ['en', en]]) for (const [k, v] of Object.entries(d)) for (const re of proibidos) assert.ok(!re.test(v), `${nome}:${k} contém ${re}`);
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  for (const re of proibidos) assert.ok(!re.test(html), `index.html contém ${re}`);
});
