"""PEFT-ready utility helpers."""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    bias: str = "none"

@dataclass
class TrainingConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    logging_steps: int = 10
    save_steps: int = 100
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    use_qlora: bool = False

def load_base_model(base_model: str, use_qlora: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    return model, tokenizer

def apply_lora(model, lora_config: LoRAConfig, use_qlora: bool = False):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"학습 가능 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model

def load_sft_dataset(jsonl_path: str | Path):
    from datasets import Dataset

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    dataset = Dataset.from_list(records)
    logger.info(f"데이터셋 로딩 완료: {len(dataset)}개 샘플")
    return dataset

def run_training(
    base_model: str,
    train_jsonl: str | Path,
    output_dir: str | Path,
    method: str = "lora",
    lora_config: Optional[LoRAConfig] = None,
    training_config: Optional[TrainingConfig] = None,
) -> Path:
    from transformers import TrainingArguments
    from trl import SFTTrainer

    use_qlora = method == "qlora"
    lora_cfg = lora_config or LoRAConfig()
    train_cfg = training_config or TrainingConfig(use_qlora=use_qlora)

    model, tokenizer = load_base_model(base_model, use_qlora=use_qlora)
    model = apply_lora(model, lora_cfg, use_qlora=use_qlora)
    dataset = load_sft_dataset(train_jsonl)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        fp16=train_cfg.fp16,
        report_to="none",
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=train_cfg.max_seq_length,
        dataset_text_field="text",
    )

    logger.info(f"학습 시작 ({method.upper()}): {base_model}")
    trainer.train()

    adapter_path = Path(output_dir)
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"어댑터 저장 완료: {adapter_path}")
    return adapter_path
def load_adapter_model(base_model: str, adapter_path: str | Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()
    return model, tokenizer

def build_sft_record(example: dict) -> dict[str, str]:
    instruction = example.get("instruction", "").strip()
    output = example.get("output", "").strip()
    return {"text": f"### Instruction:\n{instruction}\n\n### Response:\n{output}"}


def build_training_artifact_name(base_model: str, method: str) -> str:
    return f"{base_model.replace('/', '_')}-{method}"


def default_adapter_dir(output_root: str | Path, base_model: str, method: str) -> Path:
    return Path(output_root) / build_training_artifact_name(base_model, method)
