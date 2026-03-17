from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info
import config

_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    model, processor = FastVisionModel.from_pretrained(
        config.MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing=False,
    )
    FastVisionModel.for_inference(model)
    _model, _processor = model, processor
    return model, processor


def infer(messages: list) -> str:
    model, processor = load_model()
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
    # strip input tokens from output
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]
