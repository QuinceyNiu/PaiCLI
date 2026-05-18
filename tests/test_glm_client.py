import json
import unittest

from paicli.llm.glm_client import FunctionCall, GLMClient, Message, Tool, ToolCall


class GLMClientRequestTest(unittest.TestCase):
    def test_builds_chat_request_with_messages_and_tools(self) -> None:
        client = GLMClient(api_key="test-key")
        messages = [
            Message.system("You are PaiCli."),
            Message.user("Write a file."),
            Message.assistant(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(
                            name="write_file",
                            arguments=json.dumps(
                                {"path": "hello.txt", "content": "hello"},
                                ensure_ascii=False,
                            ),
                        ),
                    )
                ],
            ),
            Message.tool("call_1", "ok"),
        ]
        tools = [
            Tool(
                name="write_file",
                description="Write content to a file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件路径"},
                        "content": {"type": "string", "description": "文件内容"},
                    },
                    "required": ["path", "content"],
                },
            )
        ]

        payload = client.build_chat_payload(messages, tools)

        self.assertEqual(payload["model"], "glm-5.1")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "You are PaiCli."})
        self.assertEqual(payload["messages"][2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(payload["messages"][2]["tool_calls"][0]["function"]["name"], "write_file")
        self.assertEqual(payload["messages"][3]["tool_call_id"], "call_1")
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["function"]["parameters"]["required"], ["path", "content"])

    def test_build_chat_payload_replaces_lone_surrogates_before_json_encoding(self) -> None:
        client = GLMClient(api_key="test-key")
        messages = [
            Message.user("读取 pyproject.toml\udce8了解依赖"),
            Message.assistant(
                "",
                [
                    ToolCall(
                        id="call_\udce8",
                        function=FunctionCall(
                            name="read_file",
                            arguments='{"path": "README.md\udce8"}',
                        ),
                    )
                ],
            ),
            Message.tool("call_\udce8", "结果\udce8"),
        ]
        tools = [
            Tool(
                name="read_file",
                description="读取\udce8文件",
                parameters={"type": "object", "properties": {"path": {"description": "路径\udce8"}}},
            )
        ]

        payload = client.build_chat_payload(messages, tools)

        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.assertIn("�", encoded.decode("utf-8"))

    def test_parses_tool_call_response(self) -> None:
        client = GLMClient(api_key="test-key")
        response = client.parse_chat_response({
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{\"path\": \"README.md\"}",
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        })

        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.message.tool_calls[0].id, "call_2")
        self.assertEqual(response.message.tool_calls[0].function.name, "read_file")
        self.assertEqual(response.usage.total_tokens, 15)


if __name__ == "__main__":
    unittest.main()
