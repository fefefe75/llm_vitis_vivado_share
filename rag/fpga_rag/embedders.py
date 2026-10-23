"""Turn text into vectors. Three backends, all local:

* ``fastembed``: runs the model in-process (ONNX, CPU). Nothing to start, the
  model is downloaded once from Hugging Face and cached.
* ``openai``: any server exposing ``POST /v1/embeddings`` (LM Studio, Ollama,
  llama.cpp server, vLLM...). Use it when a GPU server already runs.
* ``hash``: feature hashing of the words. No model, no network, no semantics:
  it only exists for the tests and for a machine with no model at all.

Every backend returns L2-normalised float32 vectors, so cosine similarity is a
dot product.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Protocol, Sequence

import numpy as np

from .config import Settings, private_dir


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


class HashEmbedder:
    """Bag of words hashed into ``dim`` buckets. Deterministic, lexical only."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    """In-process ONNX model through the ``fastembed`` package."""

    def __init__(self, model: str, query_prefix: str, doc_prefix: str, cache_dir: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "fastembed n'est pas installe : pip install -e 'rag[local]' "
                "(ou FPGA_RAG_BACKEND=openai avec un serveur LM Studio/Ollama)"
            ) from exc
        self.name = model
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        # The attention cost grows with the square of the sequence length and is
        # multiplied by the batch: 16 long chunks at once exhaust the RAM of an
        # 8 to 16 GB machine and the kernel kills the process. Small default,
        # both knobs overridable (see config.py).
        self._batch = max(1, int(os.environ.get("FPGA_RAG_EMBED_BATCH") or 4))
        kwargs = {}
        threads = os.environ.get("FPGA_RAG_EMBED_THREADS")
        if threads:
            kwargs["threads"] = int(threads)
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir, **kwargs)
        self.dim = int(self.embed_query("dimension probe").shape[0])

    def _run(self, texts: Sequence[str]) -> np.ndarray:
        return _normalise(np.stack(list(self._model.embed(list(texts), batch_size=self._batch))))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._run([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._run([self.query_prefix + text])[0]


class OpenAIEmbedder:
    """``POST {base_url}/v1/embeddings`` on a local OpenAI-compatible server."""

    def __init__(self, base_url: str, model: str, query_prefix: str, doc_prefix: str,
                 api_key: str = "", batch: int = 16, timeout: int = 300) -> None:
        self.url = base_url.rstrip("/") + "/v1/embeddings"
        self.name = model
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.api_key = api_key
        self.batch = batch
        self.timeout = timeout
        self.dim = int(self.embed_query("dimension probe").shape[0])

    def _post(self, inputs: list[str]) -> np.ndarray:
        payload = json.dumps({"model": self.name, "input": inputs}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(4):
            req = urllib.request.Request(self.url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())["data"]
                data.sort(key=lambda d: d.get("index", 0))
                return np.asarray([d["embedding"] for d in data], dtype=np.float32)
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 403, 404):
                    raise RuntimeError(
                        f"{self.url} a refuse la requete (HTTP {exc.code}) : "
                        f"le modele '{self.name}' est-il charge ?"
                    ) from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"serveur d'embeddings injoignable ({self.url}) : {last}. "
            "Le serveur (LM Studio, Ollama...) tourne-t-il ?"
        )

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        out = [self._post([self.doc_prefix + t for t in texts[i:i + self.batch]])
               for i in range(0, len(texts), self.batch)]
        return _normalise(np.concatenate(out))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._post([self.query_prefix + text]))[0]


def make_embedder(settings: Settings) -> Embedder:
    """Build the embedder chosen by the settings."""
    if settings.backend == "hash":
        match = re.search(r"(\d+)$", settings.model)
        return HashEmbedder(int(match.group(1)) if match else 512)
    if settings.backend == "openai":
        return OpenAIEmbedder(settings.base_url, settings.model, settings.query_prefix,
                              settings.doc_prefix, settings.api_key)
    return FastEmbedEmbedder(settings.model, settings.query_prefix, settings.doc_prefix,
                             cache_dir=str(private_dir(settings.home) / "models"))
