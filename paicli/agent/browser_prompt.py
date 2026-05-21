"""Shared browser MCP usage guidance for agent prompts."""

BROWSER_MCP_GUIDE = """

【web_fetch vs 浏览器 MCP 决策表】
- 普通静态/SSR 网页、官方文档、博客正文：优先 web_fetch，成本低、速度快。
- 微信生态内容（mp.weixin.qq.com 等）、外部搜索引擎不可见的页面：优先浏览器 MCP，不要先浪费 web_fetch。
- 动态渲染页面、需要等待元素出现、HTML 正文为空或提示 JS 渲染：使用浏览器 MCP。
- 需要点击、填表单、上传文件、处理弹窗、键盘快捷键、悬停或拖拽：使用浏览器 MCP。
- 需要截图、查看控制台报错、检查网络请求或执行页面 JavaScript：使用浏览器 MCP。

浏览器 MCP 常用路径：
1. 用 mcp__chrome-devtools__new_page 打开新页面，或用 mcp__chrome-devtools__navigate_page 导航已有页面。
2. 用 mcp__chrome-devtools__wait_for 等文章容器、关键文本或页面状态加载完成。
3. 阅读正文优先用 mcp__chrome-devtools__take_snapshot 获取 DOM/Accessibility Tree 文本。
4. 需要视觉证据时用 mcp__chrome-devtools__take_screenshot。
5. 调试页面问题时可用 list_console_messages、网络相关工具或 evaluate_script。
""".rstrip()
