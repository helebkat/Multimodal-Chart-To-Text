import argparse
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer,
    set_seed,
)
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_example(
    example: Dict[str, Any],
    processor: DonutProcessor,
    max_target_length: int = 64,
    task_prompt: str = "<s_answer>",
) -> Dict[str, Any]:
    """
    Preprocess a single ChartQA example:
      - Convert image to RGB PIL
      - Encode image with DonutProcessor
      - Tokenize answer text as labels
    Returns plain Python lists so HF Datasets can store them easily.
    """
    image = example["image"]

    # Ensure proper PIL RGB image
    if isinstance(image, Image.Image):
        image = image.convert("RGB")

    raw_label = example["label"]
    if isinstance(raw_label, list) and len(raw_label) > 0:
        answer = str(raw_label[0])
    else:
        answer = str(raw_label)

    # Target text: start token + answer
    target_text = f"{task_prompt} {answer}"

    # Encode image + prompt (we need pixel_values)
    enc = processor(
        images=image,
        text=task_prompt,
        return_tensors="pt",
    )
    # Tensor shape: (1, C, H, W) → take [0] and convert to list
    pixel_values = enc.pixel_values[0].tolist()

    # Tokenize labels
    tokenized = processor.tokenizer(
        target_text,
        add_special_tokens=True,
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids[0].tolist()
    pad_token_id = processor.tokenizer.pad_token_id

    labels = [tid if tid != pad_token_id else -100 for tid in input_ids]

    return {
        "pixel_values": pixel_values,  # list
        "labels": labels,              # list[int]
    }


def data_collator(batch):
    """
    Custom collator that converts lists into tensors and stacks them.
    """
    pixel_tensors = []
    label_tensors = []

    for item in batch:
        pv = item["pixel_values"]
        lb = item["labels"]

        # Convert lists to tensors if needed
        if not torch.is_tensor(pv):
            pv = torch.tensor(pv, dtype=torch.float32)
        if not torch.is_tensor(lb):
            lb = torch.tensor(lb, dtype=torch.long)

        pixel_tensors.append(pv)
        label_tensors.append(lb)

    pixel_values = torch.stack(pixel_tensors)  # (B, C, H, W)
    labels = torch.stack(label_tensors)       # (B, T)

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ahmed-masry/unichart-base-960")
    parser.add_argument("--output_dir", type=str, default="outputs/unichart-ft-small")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_samples", type=int, default=2000)
    parser.add_argument("--val_samples", type=int, default=400)

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--max_target_length", type=int, default=64)

    args = parser.parse_args()
    print("Training args:", args)
    print("Using device:", DEVICE)

    set_seed(args.seed)

    # 1. Load dataset
    raw_dataset = load_dataset("HuggingFaceM4/ChartQA")
    train_ds = raw_dataset["train"]
    val_ds = raw_dataset["val"]

    # Subsample
    train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.train_samples, len(train_ds))))
    val_ds = val_ds.shuffle(seed=args.seed).select(range(min(args.val_samples, len(val_ds))))

    print("Train subset size:", len(train_ds))
    print("Val subset size  :", len(val_ds))

    # 2. Load processor & model
    processor = DonutProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    pad_token_id = processor.tokenizer.pad_token_id
    decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_answer>")

    model.config.pad_token_id = pad_token_id
    model.config.decoder_start_token_id = decoder_start_token_id

    model.to(DEVICE)

    # 3. Preprocess datasets (single-example map)
    def _preprocess_single(example):
        return preprocess_example(
            example,
            processor=processor,
            max_target_length=args.max_target_length,
            task_prompt="<s_answer>",
        )

    print("Preprocessing train dataset...")
    train_proc = train_ds.map(
        _preprocess_single,
        remove_columns=train_ds.column_names,
    )

    print("Preprocessing val dataset...")
    val_proc = val_ds.map(
        _preprocess_single,
        remove_columns=val_ds.column_names,
    )

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_proc,
        eval_dataset=val_proc,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training done.")

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to: {args.output_dir}")


if __name__ == "__main__":
    main()
