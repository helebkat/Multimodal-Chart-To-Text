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
) -> Dict[str, Any]:
    """
    Preprocess a single ChartQA example for UniChart-ChartQA.

    We build a decoder target of the form:
        "<chartqa> {question} <s_answer> {answer}"

    and use that as labels. The model learns to generate this sequence
    conditioned on the image; at inference time we will feed the prompt
    "<chartqa> {question} <s_answer>" and let it generate the answer.
    """
    image = example["image"]
    if isinstance(image, Image.Image):
        image = image.convert("RGB")

    question = example["query"]
    raw_label = example["label"]
    if isinstance(raw_label, list) and len(raw_label) > 0:
        answer = str(raw_label[0])
    else:
        answer = str(raw_label)

    tokenizer = processor.tokenizer

    target_text = f"<chartqa> {question} <s_answer> {answer}"

    tokenized = tokenizer(
        target_text,
        add_special_tokens=False,
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids[0].tolist()
    pad_token_id = tokenizer.pad_token_id

    # Standard seq2seq training: pad tokens → -100 so they're ignored in loss
    labels = [tid if tid != pad_token_id else -100 for tid in input_ids]

    # Encode image → pixel_values
    pixel_values = processor(
        image,
        return_tensors="pt",
    ).pixel_values[0].tolist()

    return {
        "pixel_values": pixel_values,  # list[float]
        "labels": labels,              # list[int]
    }


def data_collator(batch):
    """
    Collate function that converts lists to tensors and stacks them.
    """
    pixel_tensors = []
    label_tensors = []

    for item in batch:
        pv = item["pixel_values"]
        lb = item["labels"]

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


def freeze_encoder(model: VisionEncoderDecoderModel):
    """
    Freeze the vision encoder so only the decoder is trained.

    This is our 'parameter-efficient' fine-tuning:
      - encoder (Swin) = frozen
      - decoder (MBart) = trainable
    """
    for p in model.encoder.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Total params: {total_params} | "
        f"Trainable params: {trainable_params} | "
        f"Trainable%: {100.0 * trainable_params / total_params:.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="ahmed-masry/unichart-chartqa-960",
        help="ChartQA-tuned UniChart checkpoint",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/unichart-qa-small")
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

    train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.train_samples, len(train_ds))))
    val_ds = val_ds.shuffle(seed=args.seed).select(range(min(args.val_samples, len(val_ds))))

    print("Train subset size:", len(train_ds))
    print("Val subset size  :", len(val_ds))

    # 2. Load processor & ChartQA model
    processor = DonutProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    pad_token_id = processor.tokenizer.pad_token_id
    model.config.pad_token_id = pad_token_id
    decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_answer>")
    model.config.decoder_start_token_id = decoder_start_token_id

    # Freeze encoder → parameter-efficient fine-tune
    freeze_encoder(model)

    model.to(DEVICE)

    # 3. Preprocess datasets
    def _prep(example):
        return preprocess_example(
            example,
            processor=processor,
            max_target_length=args.max_target_length,
        )

    print("Preprocessing train dataset...")
    train_proc = train_ds.map(
        _prep,
        remove_columns=train_ds.column_names,
    )

    print("Preprocessing val dataset...")
    val_proc = val_ds.map(
        _prep,
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
        save_total_limit=2,
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

    print("Starting QA fine-tuning (encoder frozen)...")
    trainer.train()
    print("Training done.")

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to: {args.output_dir}")


if __name__ == "__main__":
    main()
