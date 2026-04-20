from bidmate_rag.training.peft import build_sft_record, build_training_artifact_name


def test_build_sft_record_formats_instruction_and_output() -> None:
    record = build_sft_record({"instruction": "요약해줘", "output": "요약 결과"})

    assert "### Instruction:" in record["text"]
    assert "요약해줘" in record["text"]
    assert "### Response:" in record["text"]
    assert "요약 결과" in record["text"]


def test_build_training_artifact_name_is_stable() -> None:
    name = build_training_artifact_name("Qwen/Qwen2.5-3B-Instruct", "lora")

    assert name == "Qwen_Qwen2.5-3B-Instruct-lora"
