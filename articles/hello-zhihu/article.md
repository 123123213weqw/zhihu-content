---
title: 你好,知乎:我用命令行管理自己的内容
topics: []
status: draft
url:
id:
created: 2026-11-07
published:
tags: [test, workflow]
---

这是一篇示例文章,用来说明这个仓库的工作流。可以把它改成你的第一篇真文章,或者发布前删掉。

## 为什么用仓库管理内容

- 所有文章都是 Markdown 源文件,可版本控制
- 一条命令发布,不用打开浏览器进编辑器
- 发布状态(title、链接、日期)自动回填 frontmatter,`git log` 就是发布历史

## 支持的格式

正文是标准 Markdown:**加粗**、*斜体*、[链接](https://example.com)、行内代码 `code`、代码块:

```python
print("hello zhihu")
```

列表、引用也都支持:

> 写作是最长期的复利。

1. 写作
2. 发布
3. 复盘

图片放同目录 `images/` 下,这样引用(发布时自动上传替换):`![示例](images/xx.png)`
