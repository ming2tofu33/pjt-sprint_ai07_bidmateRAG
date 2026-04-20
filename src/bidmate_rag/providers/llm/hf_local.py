"""Local Hugging Face generation provider."""

from __future__ import annotations

import time
from uuid import uuid4
from pathlib import Path
from bidmate_rag.config.prompts import build_rag_user_prompt
from bidmate_rag.generation.context_builder import build_numbered_context_block
from bidmate_rag.providers.llm.base import BaseLLMProvider, RewriteResponse
from bidmate_rag.schema import GenerationResult, RetrievedChunk
import re

class HFLocalLLM(BaseLLMProvider):
    def __init__(self, model_name: str, 
                 provider_name: str = "huggingface", 
                 generator=None,
                 adapter_path: str | Path | None = None,) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self._generator = generator
        self.adapter_path = Path(adapter_path) if adapter_path else None
        

    def _get_generator(self):
        """transformers pipeline을 생성한다. 4bit 양자화로 VRAM 절약 + 속도 향상."""
        if self._generator is not None:
            return self._generator
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "transformers is required for the local HF provider. "
                "Install the ml dependency group."
            ) from exc
        # 4bit 양자화 로드 시도 (bitsandbytes 필요)
        try:
            import torch
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.adapter_path and self.adapter_path.exists():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(self.adapter_path))
                model = model.merge_and_unload()
            self._generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        except (ImportError, Exception):
            # bitsandbytes 없으면 일반 로드로 폴백
            self._generator = pipeline("text-generation", model=self.model_name)
        return self._generator

    def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[dict],
        generation_config: dict,
        system_prompt: str,
    ) -> GenerationResult:
        context, used_indices = build_numbered_context_block(
            context_chunks,
            max_chars=generation_config.get("max_context_chars", 8000),
            question=question,
        )
        # LLM이 실제로 본 청크만 유지 — 본문 [n]과 Citation 카드 매칭 일치.
        visible_chunks = [context_chunks[i] for i in used_indices]
        prompt = build_rag_user_prompt(
            question,
            context,
            rewritten_query=generation_config.get("rewritten_query"),
            memory_summary=generation_config.get("memory_summary"),
            memory_slots=generation_config.get("memory_slots"),
        )
        generator = self._get_generator()
        tokenizer = generator.tokenizer

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # chat template 직접 적용
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 입력 토큰 수 측정
        input_tokens = len(tokenizer.encode(input_text))

        # 지연 시간 측정
        start = time.time()
        response = generator(
            input_text,
            max_new_tokens=generation_config.get("max_new_tokens", 512),
            do_sample=False,
            return_full_text=False,
            repetition_penalty=1.3,
        )
        latency_ms = (time.time() - start) * 1000

        generated_text = response[0]["generated_text"].strip() if response else ""
        generated_text = re.sub(r'<[^>]+>', '', generated_text).strip()

        # 출력 토큰 수 측정
        output_tokens = len(tokenizer.encode(generated_text)) if generated_text else 0

        return GenerationResult(
            question_id=generation_config.get("question_id", f"q-{uuid4().hex[:8]}"),
            question=question,
            scenario=generation_config.get("scenario", "scenario_a"),
            run_id=generation_config.get("run_id", f"run-{uuid4().hex[:8]}"),
            embedding_provider=generation_config.get("embedding_provider", ""),
            embedding_model=generation_config.get("embedding_model", ""),
            llm_provider=self.provider_name,
            llm_model=self.model_name,
            answer=generated_text,
            retrieved_chunk_ids=[chunk.chunk.chunk_id for chunk in visible_chunks],
            retrieved_doc_ids=[chunk.chunk.doc_id for chunk in visible_chunks],
            retrieved_chunks=visible_chunks,
            latency_ms=latency_ms,
            token_usage={
                "prompt": input_tokens,
                "completion": output_tokens,
                "total": input_tokens + output_tokens,
            },
            cost_usd=0.0,
            context=context,
        )

    def rewrite(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        timeout: int | None = None,
    ) -> RewriteResponse:
        """로컬 HF pipeline으로 짧은 텍스트 생성.

        timeout은 무시된다 — transformers pipeline은 동기 실행이라 강제 종료 불가.
        기본 max_tokens가 OpenAI 대비 낮은 이유: 로컬 모델은 reasoning 토큰이
        없으므로 256이면 한 줄 재작성에 충분.
        """
        generator = self._get_generator()
        tokenizer = generator.tokenizer
        prompt_tokens = len(tokenizer.encode(prompt))
        response = generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_full_text=False,
        )
        text = (response[0].get("generated_text") or "").strip() if response else ""
        completion_tokens = len(tokenizer.encode(text)) if text else 0
        return RewriteResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
