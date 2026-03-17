from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from train.dataset import load_drivelm, train_val_split
import config

QLORA = dict(
    r=16,
    lora_alpha=32,
    # target_modules: Unsloth defaults — covers all attention projection layers
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

_TRAIN_BASE = dict(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # effective batch = 16
    max_seq_length=2048,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    fp16=not is_bf16_supported(),
    bf16=is_bf16_supported(),
    logging_steps=10,
    report_to="none",
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
)


def train(smoke_test: bool = False):
    model, processor = FastVisionModel.from_pretrained(
        config.MODEL_ID,
        load_in_4bit=True,
        min_pixels=config.MIN_PIXELS,
        max_pixels=config.MAX_PIXELS,
    )
    model = FastVisionModel.get_peft_model(model, **QLORA)

    ds = load_drivelm("train")
    train_ds, val_ds = train_val_split(ds)

    cfg_kwargs = dict(_TRAIN_BASE)
    if smoke_test:
        train_ds = train_ds.select(range(16))
        cfg_kwargs.update(max_steps=10, save_strategy="no", output_dir="checkpoints/smoke")
    else:
        cfg_kwargs.update(num_train_epochs=1, save_strategy="epoch", output_dir="checkpoints")

    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(**cfg_kwargs),
    )
    trainer.train()

    if not smoke_test:
        model.save_pretrained("checkpoints/lora-final")
        processor.save_pretrained("checkpoints/lora-final")
        print("saved to checkpoints/lora-final")


if __name__ == "__main__":
    import sys
    train(smoke_test="--smoke" in sys.argv)
