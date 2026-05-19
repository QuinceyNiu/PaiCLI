import unittest
from unittest.mock import Mock

import httpx

from paicli.web import (
    FetchResult,
    HtmlExtractor,
    NetworkPolicy,
    SearchProviderFactory,
    SearchResult,
    WebFetcher,
    WebToolProvider,
    ZhipuSearchProvider,
)


class WebSearchProviderTest(unittest.TestCase):
    def test_factory_prefers_explicit_provider_then_available_keys(self) -> None:
        self.assertEqual(
            SearchProviderFactory.pick_provider("serpapi", "glm", "", ""),
            "serpapi",
        )
        self.assertEqual(SearchProviderFactory.pick_provider("", "glm", "", ""), "zhipu")
        self.assertEqual(SearchProviderFactory.pick_provider("", "", "serp", ""), "serpapi")
        self.assertEqual(SearchProviderFactory.pick_provider("", "", "", "http://x"), "searxng")
        self.assertEqual(SearchProviderFactory.pick_provider("", "", "", ""), "zhipu")

    def test_zhipu_search_parses_search_result_array(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/paas/v4/web_search")
            payload = httpx.Response(200, json={
                "search_result": [
                    {
                        "title": "PaiCLI",
                        "link": "https://paicoding.com",
                        "content": "AI Agent CLI",
                    }
                ]
            })
            return payload

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = ZhipuSearchProvider("glm-key", client=client)

        results = provider.search("PaiCLI", 3)

        self.assertEqual(results, [SearchResult("PaiCLI", "https://paicoding.com", "AI Agent CLI")])

    def test_zhipu_search_normalizes_unsupported_engine_to_default(self) -> None:
        provider = ZhipuSearchProvider("glm-key", search_engine="search_sogou_pro")

        self.assertEqual(provider.search_engine, "search_std")

    def test_zhipu_search_reports_missing_key_without_throwing(self) -> None:
        provider = ZhipuSearchProvider("")

        self.assertFalse(provider.is_ready())
        self.assertIn("GLM_API_KEY", provider.unavailable_hint())


class WebFetchTest(unittest.TestCase):
    def test_network_policy_rejects_localhost_and_rate_limits(self) -> None:
        policy = NetworkPolicy(max_requests=2, window_seconds=60)

        self.assertIn("localhost", policy.validate("http://localhost:8080/admin"))
        self.assertIsNone(policy.validate("https://example.com/docs"))
        self.assertIsNone(policy.validate("https://example.com/blog"))
        self.assertIn("请求过于频繁", policy.validate("https://example.com/news"))

    def test_web_fetcher_reads_bounded_body_and_reports_truncation(self) -> None:
        body = b"<html><body>" + b"x" * 20 + b"</body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=body,
            )

        fetcher = WebFetcher(
            policy=NetworkPolicy(max_requests=10),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_bytes=18,
        )

        raw = fetcher.fetch("https://example.com")

        self.assertTrue(raw.truncated)
        self.assertLessEqual(len(raw.text.encode("utf-8")), 18)

    def test_html_extractor_removes_noise_and_prefers_article(self) -> None:
        html = """
        <html>
          <head><title>Demo</title><style>.x{}</style></head>
          <body>
            <nav><a href="/">home</a></nav>
            <article>
              <h1>标题</h1>
              <p>这是一段正文内容，包含 <strong>重点</strong> 和 <a href="/a">链接</a>。</p>
              <pre><code>print("pai")</code></pre>
            </article>
            <footer>footer</footer>
          </body>
        </html>
        """

        result = HtmlExtractor().extract("https://example.com/post", html)

        self.assertEqual(result.title, "Demo")
        self.assertIn("# 标题", result.markdown)
        self.assertIn("**重点**", result.markdown)
        self.assertIn("[链接](https://example.com/a)", result.markdown)
        self.assertIn('print("pai")', result.markdown)
        self.assertNotIn("home", result.markdown)
        self.assertNotIn("footer", result.markdown)

    def test_html_extractor_returns_known_boundary_message_for_empty_body(self) -> None:
        result = HtmlExtractor().extract("https://example.com", "<html><body><script>app()</script></body></html>")

        self.assertIn("未提取到正文", result.markdown)


class WebToolProviderTest(unittest.TestCase):
    def test_web_tool_provider_registers_search_and_fetch_tools(self) -> None:
        search_provider = Mock()
        search_provider.is_ready.return_value = True
        search_provider.search.return_value = [
            SearchResult("标题", "https://example.com", "摘要")
        ]
        fetcher = Mock()
        fetcher.fetch.return_value = Mock(url="https://example.com", status_code=200, text="<article><p>hello</p></article>", truncated=False)
        extractor = Mock()
        extractor.extract.return_value = FetchResult("https://example.com", "Example", "hello", False)

        provider = WebToolProvider(
            search_provider_factory=lambda: search_provider,
            fetcher_factory=lambda: fetcher,
            extractor_factory=lambda: extractor,
        )

        tools = {tool.name: tool for tool in provider.get_tools()}

        self.assertIn("web_search", tools)
        self.assertIn("web_fetch", tools)
        self.assertIn("实时信息", tools["web_search"].description)
        self.assertIn("JS 渲染", tools["web_fetch"].description)
        self.assertIn("https://example.com", tools["web_search"].executor({"query": "pai", "top_k": "1"}))
        self.assertIn("hello", tools["web_fetch"].executor({"url": "https://example.com", "max_chars": "20"}))


if __name__ == "__main__":
    unittest.main()
