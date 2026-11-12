#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyzhihu-cli>=0.2.4",
#     "markdown-it-py",
#     "pyyaml",
# ]
# ///
"""Publish a Markdown article in this repo to Zhihu.

Usage:
    uv run scripts/publish.py articles/<slug>/article.md [--dry-run] [--force]

Reads frontmatter (title/topics/status), renders Markdown to HTML, uploads
local images referenced in the body to Zhihu's image host, publishes via
pyzhihu-cli's client (reusing ~/.zhihu-cli/cookies.json login), then writes
status/url/id/published back into the frontmatter.

--dry-run: render HTML only, do not upload or publish.
--force:   allow republishing an article whose status is already published
           (creates a NEW article on Zhihu; it does not update the old one).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html as html_mod
import re
import sys
from pathlib import Path
from urllib.parse import quote

import yaml
from markdown_it import MarkdownIt

FRONTMATTER_KEYS = ["title", "topics", "status", "url", "id", "created", "published", "tags"]
IMG_RE = re.compile(r'<img src="([^"]+)"([^>]*)>')
FORMULA_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
FORMULA_INLINE_RE = re.compile(r"\$([^$\n]+?)\$")


def formula_img(tex: str) -> str:
    """Wrap raw LaTeX as a Zhihu native formula <img>.

    Verified format (survives server-side sanitization): LaTeX goes in the
    alt attribute, equation URL uses the full https:// scheme.
    """
    esc = html_mod.escape(tex, quote=True)
    return (f'<img eeimg="1" src="https://www.zhihu.com/equation?tex={quote(tex)}" '
            f'alt="{esc}"/>')


def convert_formulas(html: str) -> str:
    """Convert $$block$$ and $inline$ LaTeX in rendered HTML to Zhihu formula tags."""
    html = FORMULA_BLOCK_RE.sub(lambda m: formula_img(m.group(1).strip()), html)
    html = FORMULA_INLINE_RE.sub(lambda m: formula_img(m.group(1).strip()), html)
    return html


FORMULA_MD_RE = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+?\$")


def render_markdown_with_formulas(md: MarkdownIt, body: str) -> str:
    """Render markdown while protecting formulas from the parser.

    markdown-it treats '_' in LaTeX subscripts (x_{t+1}) as emphasis and tears
    formulas apart. Extract every formula first, render the markdown with
    plain placeholders, then substitute Zhihu formula <img> tags back in.
    """
    formulas: list[tuple[str, bool]] = []  # (tex, is_block)

    def protect(m: re.Match) -> str:
        raw = m.group(0)
        if raw.startswith("$$"):
            formulas.append((raw[2:-2], True))
            return f"\n\nXFBLOCKF{len(formulas)-1}FX\n\n"
        formulas.append((raw[1:-1], False))
        return f"XFINLINEF{len(formulas)-1}FX"

    html = md.render(FORMULA_MD_RE.sub(protect, body))
    html = re.sub(
        r"<p>\s*XFBLOCKF(\d+)FX\s*</p>",
        lambda m: "<p>" + formula_img(formulas[int(m.group(1))][0].strip()) + "</p>",
        html)
    html = re.sub(
        r"XFINLINEF(\d+)FX",
        lambda m: formula_img(formulas[int(m.group(1))][0].strip()),
        html)
    return html


def die(msg: str) -> None:
    print(f"✗ {msg}", file=sys.stderr)
    sys.exit(1)


def info(msg: str) -> None:
    print(f"› {msg}")


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split '---\nkey: val\n---\n\nbody' into (meta, body)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not m:
        die("no frontmatter found (expected '---' block at top of file)")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        die(f"frontmatter YAML parse error: {e}")
    return meta, text[m.end():]


def dump_frontmatter(meta: dict) -> str:
    """Serialize meta back to a compact '---' block, fixed key order."""

    def inline_list(v: list) -> str:
        return "[" + ", ".join(str(x) for x in v) + "]"

    lines = ["---"]
    for key in FRONTMATTER_KEYS:
        if key not in meta:
            continue
        v = meta[key]
        if isinstance(v, list):
            lines.append(f"{key}: {inline_list(v)}")
        elif v is None or v == "":
            lines.append(f"{key}:")
        else:
            lines.append(f"{key}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def upload_with_retry(client, img_path: Path, tries: int = 3, wait: float = 5.0) -> dict:
    """upload_image with retries — Zhihu's image processing occasionally stalls."""
    import time
    last: Exception | None = None
    for i in range(1, tries + 1):
        try:
            return client.upload_image(str(img_path), source="article")
        except Exception as e:  # noqa: BLE001 - retry any upload/processing failure
            last = e
            if i < tries:
                info(f"  upload attempt {i} failed ({e}); retrying in {wait:.0f}s")
                time.sleep(wait)
    raise last  # type: ignore[misc]


def zhihu_img_tag(up: dict, alt: str) -> str:
    """Build an <img> tag in Zhihu editor's format (mirrors client._build_img_html)."""
    src = up["src"]
    original = up.get("original_src", src)
    wm = up.get("watermark", "watermark")
    wm_src = up.get("watermark_src", "")
    w = up.get("width", 0)
    h = up.get("height", 0)
    caption = alt.replace('"', "&quot;")
    return (
        f'<img src="{src}" data-caption="{caption}" data-size="normal"'
        f' data-rawwidth="{w}" data-rawheight="{h}"'
        f' data-watermark="{wm}" data-original-src="{original}"'
        f' data-watermark-src="{wm_src}"'
        f' data-private-watermark-src=""/>'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish a repo article to Zhihu")
    ap.add_argument("article", help="path to article.md")
    ap.add_argument("--dry-run", action="store_true", help="render only, no upload/publish")
    ap.add_argument("--force", action="store_true", help="republish even if status is published")
    ap.add_argument("--column", default=None, help="column (zhuanlan) id to publish into")
    args = ap.parse_args()

    path = Path(args.article).resolve()
    if not path.is_file():
        die(f"file not found: {path}")
    article_dir = path.parent

    meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
    title = str(meta.get("title") or "").strip()
    column = args.column or meta.get("column") or None
    if not title:
        die("frontmatter 'title' is required")
    if meta.get("status") == "published" and not args.force:
        die("already published (status=published). Use --force to publish again as a NEW article.")
    topics = [str(t) for t in (meta.get("topics") or [])]

    md = MarkdownIt("commonmark", {"typographer": True}).enable("table")
    html = render_markdown_with_formulas(md, body)

    # --- collect local images referenced in the rendered html ---
    local_imgs: list[tuple[str, str]] = []  # (src, rest-of-tag attrs)
    for m in IMG_RE.finditer(html):
        src, rest = m.group(1), m.group(2)
        if not src.startswith(("http://", "https://", "data:", "//")):
            local_imgs.append((src, rest))
    alt_re = re.compile(r'alt="([^"]*)"')

    client = None
    if not args.dry_run:
        from zhihu_cli.auth import cookie_str_to_dict, get_cookie_string

        cookie = get_cookie_string()
        if not cookie:
            die("not logged in — run: zhihu login --qrcode")
        from zhihu_cli.client import ZhihuClient

        client = ZhihuClient(cookie_str_to_dict(cookie))

    # --- upload local images and swap tags ---
    if local_imgs:
        for src, rest in local_imgs:
            img_path = (article_dir / src).resolve()
            alt_m = alt_re.search(rest)
            alt = alt_m.group(1) if alt_m else ""
            if args.dry_run:
                info(f"[dry-run] would upload image: {src}")
                continue
            if not img_path.is_file():
                die(f"image not found: {img_path}")
            info(f"uploading image: {src}")
            up = upload_with_retry(client, img_path)
            new_tag = zhihu_img_tag(up, alt)
            old_tag = f'<img src="{src}"{rest}>'
            html = html.replace(old_tag, new_tag)
            info(f"  → {up['src']}")

    if args.dry_run:
        print("\n===== rendered HTML (dry run) =====\n")
        print(html)
        print(f"\n===== title: {title} | topics: {topics} =====")
        return

    # --- publish ---
    info(f"publishing: {title}")
    result = client.create_article(title=title, content=html, topic_ids=topics or None)
    aid = str(result.get("id") or "")
    if not aid:
        die(f"publish may have failed — no article id returned. Raw: {str(result)[:300]}")
    url = f"https://zhuanlan.zhihu.com/p/{aid}"

    # --- write back frontmatter ---
    meta["status"] = "published"
    meta["url"] = url
    meta["id"] = aid
    meta["published"] = _dt.date.today().isoformat()
    path.write_text(dump_frontmatter(meta) + "\n" + body, encoding="utf-8")

    client.close()

    print(f"\n✓ Published: {url}")
    if column:
        meta["column"] = column
        path.write_text(dump_frontmatter(meta) + "\n" + body, encoding="utf-8")
        print(f"  column: {column}")
    print(f"  frontmatter updated: {path}")


if __name__ == "__main__":
    main()
