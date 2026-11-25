#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown-it-py", "pillow", "pyyaml"]
# ///
"""Build a self-contained preview page for one article.

Usage:  uv run scripts/gen_preview.py articles/<slug> /tmp/<name>-preview.html
        python3 -m http.server 8793 --bind 127.0.0.1 --directory /tmp
        open  http://127.0.0.1:8793/<name>-preview.html

Renders the body the way scripts/publish.py does (same parser, same table format, same formula tags), embeds
every local image as a data URI, and typesets the formulas with KaTeX in the browser. A status line at the top
counts formulas rendered and images loaded, so a screenshot tells whether the page is fit to show.
"""
from __future__ import annotations

import base64
import mimetypes
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish  # noqa: E402

LOCAL_IMG_RE = re.compile(r'<img src="([^"]+)"([^>]*)>')

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
  body {{ margin: 0; background: #f6f6f6; color: #1a1a1a;
         font-family: "PingFang SC", "Noto Serif CJK SC", "Songti SC", serif; font-size: 16px; line-height: 1.85; }}
  #status {{ position: sticky; top: 0; z-index: 9; background: #fff8dc; border-bottom: 1px solid #e6d9a8; padding: 8px 16px;
            font: 13px/1.6 -apple-system, "Helvetica Neue", sans-serif; color: #6b5b1e; }}
  article {{ max-width: 720px; margin: 0 auto; padding: 24px 20px 80px; }}
  h2 {{ font-size: 22px; margin: 44px 0 14px; }}
  pre {{ background: #f2f2f2; border: 1px solid #e2e2e2; border-radius: 4px; padding: 12px 14px; overflow-x: auto;
         font: 13px/1.55 "SF Mono", Menlo, Consolas, monospace; }}
  code {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.9em; }}
  p code, li code, td code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
  img {{ max-width: 100%; }}
  table {{ border-collapse: collapse; margin: 12px 0; font-size: 14.5px; }}
  th, td {{ border: 1px solid #cfcfcf; padding: 5px 10px; text-align: left; }}
  .katex-display {{ margin: 14px 0; overflow-x: auto; }}
</style></head>
<body>
<div id="status">渲染中</div>
<article>
<h1 style="font-size:26px;margin:6px 0 18px">{title}</h1>
{body}
</article>
<script>
function report() {{
  var imgs = Array.from(document.querySelectorAll('article img:not([eeimg="1"])'));
  var loaded = imgs.filter(function (i) {{ return i.complete && i.naturalWidth > 0; }}).length;
  var done = document.querySelectorAll('article .katex').length;
  var total = +document.body.getAttribute('data-formulas') || 0;
  var text = '公式 ' + total + ' 个,已排版 ' + done + ' 个;图片 ' + imgs.length + ' 张,已加载 ' + loaded + ' 张';
  document.getElementById('status').textContent = text + (done === total && loaded === imgs.length ? '  ✓' : '  …');
}}
window.addEventListener('load', function () {{
  var imgs = Array.from(document.querySelectorAll('img[eeimg="1"]'));
  document.body.setAttribute('data-formulas', imgs.length);
  imgs.forEach(function (img) {{
    var tex = img.getAttribute('alt');
    var parent = img.parentNode;
    var alone = parent.tagName === 'P' && parent.querySelectorAll('img').length === 1 && parent.textContent.trim() === '';
    var out = document.createElement(alone ? 'div' : 'span');
    try {{ katex.render(tex, out, {{ displayMode: alone, throwOnError: true }}); }}
    catch (e) {{ out.textContent = '[KaTeX 报错] ' + tex; }}
    img.replaceWith(out);
  }});
  report();
  setTimeout(report, 1500);
}});
</script>
</body></html>
"""


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    meta, body = publish.split_frontmatter((root / "article.md").read_text(encoding="utf-8"))
    md = MarkdownIt("commonmark", {"typographer": True}).enable("table")   # same parser as scripts/publish.py
    html = publish.render_markdown_with_formulas(md, body)
    count = 0

    def embed(m: re.Match) -> str:
        nonlocal count
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = root / src
        Image.open(path).load()                               # must decode
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        count += 1
        return f'<img src="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"{m.group(2)}>'

    html = LOCAL_IMG_RE.sub(embed, html)
    out.write_text(PAGE.format(title=meta.get("title", ""), body=html), encoding="utf-8")
    referenced = len(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body))
    print(f"{out}: 嵌入图片 {count} 张 (文章引用 {referenced} 张),{out.stat().st_size / 1e6:.2f} MB")
    if count != referenced:
        raise SystemExit("嵌入的图片数和文章引用的对不上")


if __name__ == "__main__":
    main()
