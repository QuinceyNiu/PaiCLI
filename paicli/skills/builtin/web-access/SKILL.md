---
name: web-access
description: |
  所有联网操作必须通过此 skill 处理，包括搜索、网页抓取、
  动态页面阅读、登录后页面访问和浏览器兜底
version: "1.0.0"
author: PaiCLI
tags: [web, browser, search]
---

# web-access Skill

## 浏览哲学

联网任务按四步执行：

1. 明确目标：先判断用户要的是正文摘要、事实核验、页面状态、登录后内容，还是需要交互操作。
2. 选择起点：优先使用成本最低、token 最少、对用户账户影响最小的工具。
3. 过程校验：拿到内容后检查标题、正文、时间、作者、关键段落是否符合目标。
4. 完成判断：信息充分再回答；正文为空、疑似登录墙、SPA 渲染或反爬时必须升级策略。

## 工具选择表

- 不知道访问哪个页面：先用 `web_search` 找候选链接。
- 已知普通 URL：先用 `web_fetch` 抓取正文。
- `web_fetch` 返回空正文、JS 渲染提示、登录墙、权限不足或内容明显不完整：切 Chrome DevTools MCP。
- 需要点击、滚动、等待元素、截图、查看控制台或网络请求：使用 Chrome DevTools MCP。
- `web_fetch` 和浏览器都拿不到公开正文：使用 Jina Reader 兜底，命令形态是 `curl https://r.jina.ai/http://example.com/path` 或 `curl https://r.jina.ai/http://example.com/path`，保留原始 URL 的协议和路径。

## 渐进式升级策略

1. 普通公开页面先试 `web_fetch`。
2. 如果失败，使用 Chrome DevTools MCP isolated 模式打开页面，优先 `navigate_page` / `new_page`，再 `wait_for`，然后 `take_snapshot` 读取 DOM/Accessibility Tree 文本。
3. 如果页面需要登录态、SSO、私有仓库、内网或用户真实账号视图，切换 shared 模式后重试。
4. 如果浏览器仍然失败，尝试 Jina Reader 或向用户说明站点限制。

## 站点经验

内置站点经验会被 PaiCLI 解压到 `~/.paicli/skills-cache/web-access/references/site-patterns/`。
处理特定站点前，优先用 `read_file` 读取对应文件，例如：

- `~/.paicli/skills-cache/web-access/references/site-patterns/mp.weixin.qq.com.md`
- `~/.paicli/skills-cache/web-access/references/site-patterns/zhuanlan.zhihu.com.md`
- `~/.paicli/skills-cache/web-access/references/site-patterns/x.com.md`
- `~/.paicli/skills-cache/web-access/references/site-patterns/xiaohongshu.com.md`
- `~/.paicli/skills-cache/web-access/references/site-patterns/github.com.md`
- `~/.paicli/skills-cache/web-access/references/site-patterns/juejin.cn.md`

这些 reference 描述站点架构、成功路径、失败模式和处理办法。不要把站点经验当成绝对规则；如果页面实际表现不同，以工具返回和页面状态为准。
