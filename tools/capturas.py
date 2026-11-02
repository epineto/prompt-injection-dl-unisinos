"""Gera as capturas de tela de docs/img com Playwright (servir antes: npm run serve). Uso: python tools/capturas.py [url_base]"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/index.html").split("#")[0]
OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
SECOES = ["intro", "testar", "treinar", "otimizadores", "backprop", "resultados", "sobre"]


def preparar(pg, secao):
    pg.goto(f"{BASE}#{secao}")
    pg.wait_for_selector('html[data-pronto="1"]')
    if secao == "treinar":
        pg.click("#tr-go"); pg.wait_for_selector("#tr-res:not([hidden])", timeout=60000)
    elif secao == "otimizadores":
        pg.click("#ot-go"); pg.wait_for_function("document.querySelector('#ot-tab table')", timeout=90000)
    elif secao == "backprop":
        pg.click("#bp-passo"); pg.click("#bp-treinar"); pg.wait_for_timeout(1200); pg.click("#bp-treinar")
    pg.wait_for_timeout(300)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ("pt", "en"):
        for nome, vp in (("desktop", {"width": 1280, "height": 900}), ("mobile", {"width": 390, "height": 800})):
            ctx = b.new_context(viewport=vp)
            pg = ctx.new_page()
            erros = []
            pg.on("console", lambda m, e=erros: e.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda x, e=erros: e.append(str(x)))
            pg.goto(BASE); pg.wait_for_selector('html[data-pronto="1"]')
            pg.evaluate("l => localStorage.setItem('dl-prompt-injection-lang', l)", lang)
            pg.reload(); pg.wait_for_selector('html[data-pronto="1"]')
            for s in SECOES:
                if nome == "mobile" and s not in ("intro", "testar", "treinar", "backprop"):
                    continue
                preparar(pg, s)
                pg.screenshot(path=str(OUT / f"{nome}-{lang}-{s}.png"), full_page=True)
            ctx.close()
            assert not erros, f"erros no console ({nome}, {lang}): {erros}"
            print(nome, lang, "console sem erros")
    b.close()
print("ok", OUT)
