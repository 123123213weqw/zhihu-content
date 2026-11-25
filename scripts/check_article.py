#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown-it-py", "pillow"]
# ///
"""Publish-time sanity check for one article.

Usage:  uv run scripts/check_article.py articles/<slug>

Checks:
  - every image referenced in the body exists and decodes
  - no **bold** marker that markdown-it will not close (a closing ** preceded by
    punctuation and followed directly by text renders as literal asterisks)
  - house style: paragraphs of at most ~3 lines, at most one bold span a paragraph,
    no full-width punctuation other than the enumeration comma, no filler words
Run it before the KaTeX check and the dry run.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from PIL import Image

FILLER = [
    "值得注意的是", "值得一提", "本质上", "归根结底", "不仅仅是", "无疑", "令人", "让我们", "不妨",
    "你可能会想", "你也许会问", "赋能", "抓手", "闭环", "沉淀", "极大地", "有效地", "显著地", "首先",
    "其次", "深入", "深度解析", "全方位", "据我所知", "我实测", "多半", "基本上", "大致", "占位:", "<<", "{{",
]


def main() -> int:
    root = Path(sys.argv[1])
    text = (root / "article.md").read_text(encoding="utf-8")
    body = text.split("---", 2)[2]
    ok = True

    refs = re.findall(r"!\[.*?\]\((.*?)\)", body)
    for r in refs:
        im = Image.open(root / r)
        im.load()
    print(f"图片 {len(refs)} 张,全部存在并可解码")

    md = MarkdownIt("commonmark")
    unclosed = []
    in_code = False
    for n, line in enumerate(body.split("\n"), 1):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not in_code and "**" in line and "**" in md.render(line):
            unclosed.append((n, line[:60]))
    for n, line in unclosed:
        print(f"  没闭合的加粗:第 {n} 行(相对正文): {line}")
    if unclosed:
        ok = False
    print(f"加粗没闭合的行:{len(unclosed)}")

    paras, code_blocks, longest = [], 0, 0
    in_code = False
    for line in body.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
            code_blocks += int(in_code)
            continue
        if in_code:
            longest = max(longest, len(line))
            continue
        if not line.strip() or line.startswith(("|", "![", "$$", "#", "- ", "<", "*", "答:")) or re.match(r"^\d\. ", line):
            continue
        paras.append(line)
    cjk = lambda t: len(re.findall(r"[\u4e00-\u9fff]", t))
    long = [(cjk(p), p[:30]) for p in paras if cjk(p) > 105]
    multi_bold = sum(p.count("**") // 2 > 1 for p in paras)
    fullwidth = {c: body.count(c) for c in "，：；（）！“”" if body.count(c)}
    # the two ASCII placeholder markers are shift / template operators inside code, so only look at prose
    prose = re.sub(r"```.*?```", "", body, flags=re.S)
    flagged = {w: prose.count(w) if w in ("<<", "{{") else body.count(w) for w in FILLER if (prose.count(w) if w in ("<<", "{{") else body.count(w))}
    sections = re.findall(r"^## (.*)$", body, re.M)

    print(f"汉字 {cjk(body)} | 小节 {len(sections)} | 段落 {len(paras)} | 代码块 {code_blocks}(最长一行 {longest} 字符) | "
          f"公式 {body.count('$$') // 2}")
    print(f"超过 3 行的段落:{len(long)} {long[:3]}")
    print(f"一段里多于一处加粗:{multi_bold} | 破折号:{body.count('——')} | 全角标点:{fullwidth} | '我':{body.count('我')}")
    print(f"空话和缓冲词:{flagged or '无'}")
    print("小节:" + " / ".join(s.split(":")[0].split("、")[0] + "、" + s.split("、", 1)[-1].split(":")[0] for s in sections))
    if long or multi_bold or fullwidth or flagged:
        ok = False
    print("结论:" + ("通过" if ok else "有问题,看上面"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
