# zhihu-content

知乎内容维护仓库:Markdown 写作 → 一条命令发布 → 仓库追踪发布状态。

## 目录结构

```
articles/<slug>/article.md   # 每篇文章一个目录(frontmatter + Markdown 正文)
articles/<slug>/images/      # 本地图片,发布时自动上传到知乎图床并内联替换
templates/article.md         # 新文章模板
topics.md                    # 常用话题 ID 备忘
scripts/publish.py           # 发布脚本
```

## 日常工作流

```bash
# 1. 新建文章:复制模板
cp -r templates/article.md articles/my-new-post/article.md   # 先 mkdir articles/my-new-post

# 2. 写作:编辑 article.md,图片放 images/ 里,正文用 ![](images/xx.png) 引用

# 3. 预览渲染(不发布)
uv run scripts/publish.py articles/my-new-post/article.md --dry-run

# 4. 正式发布(需先 zhihu login 扫码登录)
uv run scripts/publish.py articles/my-new-post/article.md

# 5. 发布成功后脚本自动回填 frontmatter(url/id/status/published),提交入库
git add -A && git commit -m "publish: my-new-post"
```

## frontmatter 字段

| 字段 | 说明 |
|---|---|
| `title` | 文章标题(必填) |
| `topics` | 话题 ID 列表,如 `[19551147]`,见 topics.md |
| `status` | `draft` / `published`,发布成功后自动回填 |
| `url` | 发布后自动回填的文章链接 |
| `id` | 发布后自动回填的文章 ID |
| `created` | 创建日期 |
| `published` | 发布日期,自动回填 |
| `tags` | 自用分类标签,不发布 |

## 环境要求

- 本机已安装 [uv](https://docs.astral.sh/uv/) 和 pyzhihu-cli(`uv tool install pyzhihu-cli`)
- 首次使用先登录:`zhihu login --qrcode`(扫码,Cookie 存 `~/.zhihu-cli/cookies.json`,不进仓库)
- 发布脚本用 PEP 723 内联声明依赖,`uv run` 自动解析,无需手动装包

## 注意

- 登录态(Cookie)只存在本机 `~/.zhihu-cli/`,**绝不提交到仓库**
- 已发布的文章重复执行会拒绝,需 `--force` 才重发(会在知乎生成新文章,不会覆盖旧的)
- 删除知乎文章:`zhihu delete-article <id>`(需确认)
