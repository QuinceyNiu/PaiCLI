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

浏览器登录态：
- 默认 isolated 模式：临时 user-data-dir，无 cookie / 登录态。
- shared 模式会复用用户正在使用的调试 Chrome。
- shared 模式下你看到的页面是用户的真实账户视图。
- 公开页面优先使用 isolated；如果页面提示登录、权限不足或 SSO，再切换 shared 并重试。
- 不要做用户没明确要求的写入：不要点关注/取消关注/删除/退出登录/修改设置等按钮。
- 不要在表单里填用户没给你的数据。
- 不要执行 evaluate_script 跑用户没要求的脚本。
- close_page 只能关你自己 new_page 出来的 tab。
- 如果不确定某个操作是否会影响用户账户数据，先问用户确认。
""".rstrip()
