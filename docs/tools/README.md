# 六篇教程的 HTML 构建

直接阅读时打开 `../html/index.html`，无需 Node、服务器或联网。HTML、搜索、SVG 流程图、代码和补充材料均在 `docs/html/` 内；外部官方链接需要联网。

## 重建

源文档在 `docs/`。Node 依赖优先使用本目录安装，其次使用 `DEEPEP_DOCS_NODE_MODULES` 指定的目录，最后尝试本机 Codex 的 Node 依赖。需要一个 Chromium/Edge 浏览器来在构建阶段生成静态 SVG 图。

```bash
cd docs/tools
npm install
npx playwright install chromium
node build-docs.cjs
node check-docs.cjs
```

也可在仓库根目录运行 `node docs/tools/build-docs.cjs`。Windows 可自动使用标准路径的 Edge，其他路径可通过 `DEEPEP_DOCS_BROWSER` 指定。

`vendor/mermaid.min.js` 是固定版本 11.12.2 的 Mermaid 构建，仅在生成 SVG 时使用，许可见 `vendor/mermaid-LICENSE`。它来自 `https://cdn.jsdelivr.net/npm/mermaid@11.12.2/dist/mermaid.min.js`。生成后的阅读页面不需要载入 Mermaid，也不访问 CDN。

输出包括：目录和六篇全文、必要的本地材料页、带行号的源码快照、静态 SVG、主题 CSS、交互 JS、全文搜索索引、构建清单及校验报告。构建只生成文档；不会执行 CUDA、网络通信或部署命令。

页面的阅读检查卡片是围绕本文核心概念编写的学习辅助，原理和代码的完整说明仍在正文中。代码块里的省略号和伪代码保留 Markdown 原文的标注。

## 验证范围

`check-docs.cjs` 检查所有本地链接/片段、六篇正文完整性、SVG 生成结果，并用 headless 浏览器检查宽屏/手机布局、搜索、主题、字体、导航和控制台错误。截图与报告写入 `docs/html/qa/`。

这验证的是文档及浏览器阅读功能，不代表 CUDA/NVSHMEM/NCCL 集群测试通过。
