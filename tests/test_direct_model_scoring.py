from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import types

import pytest
import torch

from competition_packs.qwen3_27b_shared import shared


class _FakeTokenizer:
    def __call__(self, text: str, **_kwargs):
        del text
        return {
            "input_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }


class _FakeModel:
    def __init__(self, vocab_size: int = 5) -> None:
        self.vocab_size = vocab_size
        self.parameter = torch.nn.Parameter(torch.zeros(2))
        self.calls: list[dict[str, object]] = []

    def parameters(self):
        return iter([self.parameter])

    def to(self, _device):
        return self

    def eval(self) -> None:
        return None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert "labels" not in kwargs
        input_ids = kwargs["input_ids"]
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros((batch_size, sequence_length, self.vocab_size), dtype=torch.float32)
        return types.SimpleNamespace(logits=logits, loss=torch.tensor(1e-6))


def test_direct_model_scoring_computes_ppl_from_logits(monkeypatch) -> None:
    tokenizer_calls: list[tuple[str, dict[str, object]]] = []
    fake_model = _FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            tokenizer_calls.append((model_id, kwargs))
            return _FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(_model_dir: str, **_kwargs):
            return fake_model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM, AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setenv("AUTORESEARCH_EVAL_DEVICE", "cpu")
    monkeypatch.setenv("AUTORESEARCH_EVAL_LOCAL_FILES_ONLY", "1")
    monkeypatch.setenv("AUTORESEARCH_EVAL_MAX_TOKENS", "8")
    monkeypatch.setenv("AUTORESEARCH_EVAL_BATCH_SIZE", "2")
    monkeypatch.setenv("AUTORESEARCH_EVAL_TOKENIZER_ID", "validator/tokenizer")

    score = shared._score_transformer_model(
        model_dir=Path("artifact-model"),
        documents=["heldout text"],
        tokenizer_id="fallback/tokenizer",
    )

    assert score["tokenizer_id"] == "validator/tokenizer"
    assert tokenizer_calls == [
        (
            "validator/tokenizer",
            {"trust_remote_code": False, "local_files_only": True},
        )
    ]
    assert score["token_count"] == 3
    assert score["heldout_cross_entropy_nats"] == pytest.approx(math.log(fake_model.vocab_size))
    assert score["heldout_ppl"] == pytest.approx(float(fake_model.vocab_size))
    assert fake_model.calls
    assert all("labels" not in call for call in fake_model.calls)


def test_default_validator_tokenizer_is_self_contained(monkeypatch) -> None:
    fake_model = _FakeModel(vocab_size=257)

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(_model_dir: str, **_kwargs):
            return fake_model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM),
    )
    monkeypatch.setenv("AUTORESEARCH_EVAL_DEVICE", "cpu")
    monkeypatch.delenv("AUTORESEARCH_EVAL_TOKENIZER_ID", raising=False)

    score = shared._score_transformer_model(
        model_dir=Path("artifact-model"),
        documents=["abc"],
    )

    assert score["tokenizer_id"] == shared.VALIDATOR_TOKENIZER_ID
    assert score["token_count"] == 2
    assert score["heldout_cross_entropy_nats"] == pytest.approx(math.log(fake_model.vocab_size))
    assert score["heldout_ppl"] == pytest.approx(float(fake_model.vocab_size))


def test_direct_eval_stub_file_is_not_materialized(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTORESEARCH_ALLOW_DIRECT_EVAL_STUB", "1")
    artifact_path = tmp_path / "stub.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact_type": "autoresearch-direct-eval-stub",
                "heldout_ppl": 1.000001,
                "parameter_count": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Hugging Face model directory"):
        shared._materialize_model_artifact(artifact_path, tmp_path / "work")
