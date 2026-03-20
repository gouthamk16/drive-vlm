import sys
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.nn.utils.rnn import pad_sequence

import config
from train.dataset import load_drivelm, train_val_split, synthetic_dataset

LORA = LoraConfig(
    r=16,
    lora_alpha=32,
    # All attention + MLP projections in the language model — vision encoder stays frozen
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

TRAIN_ARGS = dict(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=10,
    dataloader_num_workers=0,         # Windows: no fork-based workers
    report_to="none",
    remove_unused_columns=False,
)


class VLMCollator:
    """
    Collates multimodal DriveLM samples for Qwen2.5-VL.
    Labels are -100 everywhere except the assistant response tokens,
    so the loss is only computed on what the model should generate.
    """

    def __init__(self, processor, max_seq=2048):
        self.processor = processor
        self.max_seq = max_seq
        tok = processor.tokenizer
        # Token sequence that marks the start of each assistant turn
        self._asst_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self._end_id = tok.encode("<|im_end|>", add_special_tokens=False)[0]
        self._pad_id = tok.pad_token_id or tok.eos_token_id

    def _make_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return labels with -100 on all non-assistant tokens."""
        ids = input_ids.tolist()
        labels = [-100] * len(ids)
        n = len(self._asst_ids)
        in_asst = False
        asst_start = 0
        for i in range(len(ids)):
            if ids[i : i + n] == self._asst_ids:
                in_asst = True
                asst_start = i + n   # skip the header itself
            if in_asst and i >= asst_start:
                if ids[i] == self._end_id:
                    in_asst = False
                else:
                    labels[i] = ids[i]
        return torch.tensor(labels, dtype=torch.long)

    def __call__(self, samples):
        all_ids, all_masks, all_labels = [], [], []
        all_pixels, all_grids = [], []
        all_vid_pixels, all_vid_grids = [], []

        for s in samples:
            msgs = s["messages"]
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            img_inputs, vid_inputs = process_vision_info(msgs)

            enc = self.processor(
                text=[text],
                images=img_inputs or None,
                videos=vid_inputs or None,
                return_tensors="pt",
                padding=False,
            )

            ids = enc["input_ids"][0][: self.max_seq]
            mask = enc["attention_mask"][0][: self.max_seq]

            all_ids.append(ids)
            all_masks.append(mask)
            all_labels.append(self._make_labels(ids))

            if "pixel_values" in enc:
                all_pixels.append(enc["pixel_values"])
            if "image_grid_thw" in enc:
                all_grids.append(enc["image_grid_thw"])
            if "pixel_values_videos" in enc:
                all_vid_pixels.append(enc["pixel_values_videos"])
            if "video_grid_thw" in enc:
                all_vid_grids.append(enc["video_grid_thw"])

        batch = {
            "input_ids":      pad_sequence(all_ids,    batch_first=True, padding_value=self._pad_id),
            "attention_mask": pad_sequence(all_masks,  batch_first=True, padding_value=0),
            "labels":         pad_sequence(all_labels, batch_first=True, padding_value=-100),
        }
        if all_pixels:
            batch["pixel_values"]        = torch.cat(all_pixels,     dim=0)
        if all_grids:
            batch["image_grid_thw"]      = torch.cat(all_grids,      dim=0)
        if all_vid_pixels:
            batch["pixel_values_videos"] = torch.cat(all_vid_pixels, dim=0)
        if all_vid_grids:
            batch["video_grid_thw"]      = torch.cat(all_vid_grids,  dim=0)
        return batch


def _load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        config.MODEL_ID,
        min_pixels=config.MIN_PIXELS,
        max_pixels=config.MAX_PIXELS,
    )
    return model, processor


def train(smoke_test: bool = False):
    model, processor = _load_model()

    if smoke_test:
        # Use synthetic data so pipeline can be validated without HF auth
        train_ds = synthetic_dataset(32)
        val_ds   = synthetic_dataset(4)
    else:
        ds = load_drivelm("train")
        train_ds, val_ds = train_val_split(ds)

    args = dict(TRAIN_ARGS)
    if smoke_test:
        args.update(max_steps=10, save_strategy="no", output_dir="checkpoints/smoke",
                    eval_strategy="no")
    else:
        args.update(num_train_epochs=1, save_strategy="epoch",
                    output_dir="checkpoints", eval_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**args),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=VLMCollator(processor),
    )
    trainer.train()

    if not smoke_test:
        model.save_pretrained("checkpoints/lora-final")
        processor.save_pretrained("checkpoints/lora-final")
        print("saved to checkpoints/lora-final")


if __name__ == "__main__":
    train(smoke_test="--smoke" in sys.argv)
