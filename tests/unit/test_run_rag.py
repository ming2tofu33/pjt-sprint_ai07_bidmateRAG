from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path


def test_run_rag_passes_history_and_retrieval_config(monkeypatch, capsys) -> None:
    from scripts import run_rag

    captured: dict = {}

    class FakePipeline:
        def answer(self, question: str, **kwargs):
            captured["question"] = question
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                answer="테스트 답변",
                cost_usd=0.001,
                debug={
                    "original_query": question,
                    "rewritten_query": "국민연금공단 차세대 ERP 사업의 평가기준",
                    "memory_summary": "이전 대화 요약",
                    "memory_slots": {"발주기관": "국민연금공단", "관심속성": "평가기준"},
                    "rewrite_cost_usd": 0.0003,
                    "generation_cost_usd": 0.001,
                    "total_cost_usd": 0.0013,
                },
            )

    runtime = SimpleNamespace(
        experiment=SimpleNamespace(retrieval_top_k=7),
        project=SimpleNamespace(default_retrieval_top_k=5),
        provider=SimpleNamespace(scenario="scenario_b", provider="openai"),
    )
    embedder = SimpleNamespace(provider_name="openai", model_name="text-embedding-3-small")

    def fake_build_runtime_pipeline(**kwargs):
        captured["build_kwargs"] = kwargs
        return FakePipeline(), runtime, embedder, object()

    monkeypatch.setattr(run_rag, "build_runtime_pipeline", fake_build_runtime_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_rag.py",
            "--question",
            "평가기준은?",
            "--provider-config",
            "configs/providers/openai_gpt5mini.yaml",
            "--retrieval-config",
            "configs/retrieval.yaml",
            "--history-json",
            '[{"role":"user","content":"국민연금공단 차세대 ERP 사업 알려줘"}]',
        ],
    )

    run_rag.main()
    out = capsys.readouterr().out

    assert captured["build_kwargs"]["retrieval_config_path"] == "configs/retrieval.yaml"
    assert captured["question"] == "평가기준은?"
    assert captured["kwargs"]["chat_history"] == [
        {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}
    ]
    assert captured["kwargs"]["top_k"] == 7
    assert "원본 질문: 평가기준은?" in out
    assert "재작성 질문: 국민연금공단 차세대 ERP 사업의 평가기준" in out
    assert "메모리 요약: 이전 대화 요약" in out
    assert "메모리 슬롯:" in out
    assert "총 비용(USD): 0.001300" in out
    assert out.strip().endswith("테스트 답변")


def test_load_history_supports_utf8_bom(tmp_path: Path) -> None:
    from scripts.run_rag import _load_history

    history_path = tmp_path / "history.json"
    history_path.write_text(
        '\ufeff[{"role":"user","content":"국민연금공단 차세대 ERP 사업 알려줘"}]',
        encoding="utf-8",
    )

    history = _load_history(history_json=None, history_file=str(history_path))

    assert history == [{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}]
