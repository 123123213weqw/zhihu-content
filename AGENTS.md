# zhihu-content 工作流规则

## 渲染验证(用户明确要求,最高优先级)

在把任何文章预览/发布给用户看之前,必须完成并通过渲染验证,不过关不许交付:

1. **公式验证**:提取文章里所有 `$$...$$` 与 `$...$`,用 KaTeX 服务端逐个真渲染(node + katex renderToString, throwOnError),零失败才算过。常见坑:
   - `\mathbb{1}` 不支持(数字不能进 `\mathbb`)→ 用 `\mathbf{1}`
   - `\text{中文}` 不可靠(KaTeX 字体无 CJK 字形)→ 中文移出公式,标签用英文
   - mathtext 不支持的符号(`\le`→`\leq`、`\snowflake` 等)
2. **图片验证**:文章引用的每张图必须存在、可解码(PIL open 校验)、预览 HTML 中 data URI 嵌入数 = 文章引用数。
3. **预览页验证**:构建预览后,用浏览器打开截图,确认公式已渲染(不是原始 `$` 文本)、图片全部显示,再交给用户。

## 其他

- 全部制图用服务器(WZU_V100)matplotlib 学术风格(serif + STIX),脚本入库可复现;不用 HTML 方式作图
- 文章数字必须有论文出处
- 发布走 scripts/publish.py(公式转知乎 eeimg+alt 格式,已实测)
