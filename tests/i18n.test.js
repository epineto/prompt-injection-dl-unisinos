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
