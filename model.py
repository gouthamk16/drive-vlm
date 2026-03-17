from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info
import config

_model = None
_processor = None
_ft_model = None
_ft_processor = None


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    model, processor = FastVisionModel.from_pretrained(
        config.MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing=False,
        min_pixels=config.MIN_PIXELS,
        max_pixels=config.MAX_PIXELS,
    )
    FastVisionModel.for_inference(model)
    _model, _processor = model, processor
    return model, processor


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
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=512)
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
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    FastVisionModel.for_inference(ft_model)
    _ft_model, _ft_processor = ft_model, processor
    return ft_model, processor


def infer_ft(messages: list, adapter_path: str) -> str:
    return _run_inference(*load_ft_model(adapter_path), messages)
