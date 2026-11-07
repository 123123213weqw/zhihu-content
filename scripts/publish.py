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
import re
import sys
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

FRONTMATTER_KEYS = ["title", "topics", "status", "url", "id", "created", "published", "tags"]
IMG_RE = re.compile(r'<img src="([^"]+)"([^>]*)>')


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
    args = ap.parse_args()

    path = Path(args.article).resolve()
    if not path.is_file():
        die(f"file not found: {path}")
    article_dir = path.parent

    meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
    title = str(meta.get("title") or "").strip()
    if not title:
        die("frontmatter 'title' is required")
    if meta.get("status") == "published" and not args.force:
        die("already published (status=published). Use --force to publish again as a NEW article.")
    topics = [str(t) for t in (meta.get("topics") or [])]

    md = MarkdownIt("commonmark", {"typographer": True}).enable("table")
    html = md.render(body)

    # --- collect local images referenced in the rendered html ---
    local_imgs: list[tuple[str, str]] = []  # (src, rest-of-tag attrs)
    for m in IMG_RE.finditer(html):
        src, rest = m.group(1), m.group(2)
        if not src.startswith(("http://", "https://", "data:")):
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
            up = client.upload_image(str(img_path), source="article")
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
    print(f"  frontmatter updated: {path}")


if __name__ == "__main__":
    main()
