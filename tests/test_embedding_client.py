import os
import unittest
from unittest.mock import patch

from paicli.rag import EmbeddingClient


class RecordingTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post_json(self, url, payload, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self.response


class SequenceTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post_json(self, url, payload, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class EmbeddingClientTest(unittest.TestCase):
    def test_default_configuration_uses_ollama_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_PROVIDER": "ollama",
                "EMBEDDING_MODEL": "nomic-embed-text:latest",
                "EMBEDDING_BASE_URL": "http://localhost:11434",
            },
            clear=True,
        ):
            transport = RecordingTransport({"embedding": [0.1, 0.2]})
            client = EmbeddingClient(transport=transport)

            vector = client.embed("hello")

        self.assertEqual(vector, [0.1, 0.2])
        self.assertEqual(transport.calls[0]["url"], "http://localhost:11434/api/embeddings")
        self.assertEqual(
            transport.calls[0]["payload"],
            {"model": "nomic-embed-text:latest", "prompt": "hello"},
        )

    def test_ollama_falls_back_to_new_embed_endpoint_when_legacy_endpoint_is_missing(self) -> None:
        transport = SequenceTransport(
            [
                RuntimeError("404 Not Found"),
                {"embeddings": [[0.4, 0.6]]},
            ]
        )
        client = EmbeddingClient(
            provider="ollama",
            model="nomic-embed-text:latest",
            base_url="http://localhost:11434",
            transport=transport,
        )

        vector = client.embed("hello")

        self.assertEqual(vector, [0.4, 0.6])
        self.assertEqual(transport.calls[0]["url"], "http://localhost:11434/api/embeddings")
        self.assertEqual(transport.calls[1]["url"], "http://localhost:11434/api/embed")
        self.assertEqual(
            transport.calls[1]["payload"],
            {"model": "nomic-embed-text:latest", "input": "hello"},
        )

    def test_openai_compatible_provider_uses_embeddings_endpoint_and_api_key(self) -> None:
        transport = RecordingTransport({"data": [{"embedding": [1, 2, 3]}]})
        client = EmbeddingClient(
            provider="glm",
            model="embedding-3",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key="secret",
            transport=transport,
        )

        vector = client.embed("MemoryManager")

        self.assertEqual(vector, [1.0, 2.0, 3.0])
        self.assertEqual(transport.calls[0]["url"], "https://open.bigmodel.cn/api/paas/v4/embeddings")
        self.assertEqual(
            transport.calls[0]["payload"],
            {"model": "embedding-3", "input": "MemoryManager"},
        )
        self.assertEqual(transport.calls[0]["headers"]["Authorization"], "Bearer secret")

    def test_embed_truncates_input_to_max_chars_before_request(self) -> None:
        transport = RecordingTransport({"embedding": [0.3]})
        client = EmbeddingClient(
            provider="ollama",
            model="nomic-embed-text:latest",
            base_url="http://localhost:11434",
            max_input_chars=5,
            transport=transport,
        )

        client.embed("0123456789")

        self.assertEqual(transport.calls[0]["payload"]["prompt"], "01234")

    def test_unknown_provider_falls_back_to_ollama_shape(self) -> None:
        transport = RecordingTransport({"embedding": [0.5]})
        client = EmbeddingClient(
            provider="unknown",
            model="nomic-embed-text:latest",
            base_url="http://localhost:11434",
            transport=transport,
        )

        vector = client.embed("hello")

        self.assertEqual(vector, [0.5])
        self.assertEqual(transport.calls[0]["url"], "http://localhost:11434/api/embeddings")

    def test_local_provider_keeps_deterministic_offline_embedding(self) -> None:
        client = EmbeddingClient(provider="local", dimensions=8)

        first = client.embed("Agent run")
        second = client.embed("Agent run")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)


if __name__ == "__main__":
    unittest.main()
