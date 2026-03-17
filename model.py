from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import config

_model = None
_processor = None
_ft_model = None
_ft_processor = None


def _bnb_cfg():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        quantization_config=_bnb_cfg(),
        device_map="auto",
    )
    _processor = AutoProcessor.from_pretrained(
        config.MODEL_ID,
        min_pixels=config.MIN_PIXELS,
        max_pixels=config.MAX_PIXELS,
    )
    return _model, _processor


def _run_inference(model, processor, messages: list) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
    )
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def infer(messages: list) -> str:
    return _run_inference(*load_model(), messages)


def load_ft_model(adapter_path: str):
    global _ft_model, _ft_processor
    if _ft_model is not None:
        return _ft_model, _ft_processor
    from peft import PeftModel
    base_model, processor = load_model()
    _ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    _ft_processor = processor
    return _ft_model, _ft_processor


def infer_ft(messages: list, adapter_path: str) -> str:
    return _run_inference(*load_ft_model(adapter_path), messages)
