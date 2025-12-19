import argparse
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

from .utils.metrics import compute_text_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_answer(seq: str) -> str:
    """
    Extract the answer portion from a generated sequence of the form:
      "<chartqa> ... <s_answer> ANSWER ..."

    We:
      - split on "<s_answer>"
      - drop prompt and tokens before it
      - clean up some special tokens
    """
    s = seq.replace("\n", " ").strip()

    if "<s_answer>" in s:
        s = s.split("<s_answer>", 1)[1]

    # Remove some common special tokens / tags if present
    for tok in ["</s>", "<pad>", "<eos>", "<unk>"]:
        s = s.replace(tok, " ")

    # Normalize spaces
    s = " ".join(s.split())
    return s


def generate_answers(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    dataset,
    num_samples: int,
) -> Tuple[List[str], List[str]]:
    """
    Generate predictions + gold answers for the first num_samples of dataset.
    """
    model.eval()
    preds: List[str] = []
    labels: List[str] = []

    tokenizer = processor.tokenizer

    for i in range(num_samples):
        example = dataset[i]
        image = example["image"]

        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        question = example["query"]
        raw_label = example["label"]
        if isinstance(raw_label, list) and len(raw_label) > 0:
            gold = str(raw_label[0])
        else:
            gold = str(raw_label)

        # Build decoder prompt
        prompt_text = f"<chartqa> {question} <s_answer>"

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(DEVICE)

        # Encode image
        enc = processor(
            image,
            return_tensors="pt",
        )
        pixel_values = enc.pixel_values.to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=prompt_ids,
                max_length=64,
                num_beams=3,
            )

        seq = processor.batch_decode(outputs, skip_special_tokens=False)[0]
        pred = extract_answer(seq)

        preds.append(pred)
        labels.append(gold)

    return preds, labels


def accuracy(preds: List[str], labels: List[str]) -> float:
    """
    Simple exact-match accuracy, case-insensitive, stripped.
    """
    correct = 0
    for p, l in zip(preds, labels):
        if p.strip().lower() == l.strip().lower():
            correct += 1
    return 100.0 * correct / len(labels) if labels else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="ahmed-masry/unichart-chartqa-960",
    )
    parser.add_argument(
        "--ft_model_dir",
        type=str,
        default="outputs/unichart-qa-small",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=50,
    )

    args = parser.parse_args()
    print("Eval args:", args)
    print("Using device:", DEVICE)

    # 1. Load dataset
    dataset = load_dataset("HuggingFaceM4/ChartQA")
    val_ds = dataset["val"]
    n = min(args.num_eval_samples, len(val_ds))
    print(f"Evaluating on {n} validation examples")

    # 2. Processor
    processor = DonutProcessor.from_pretrained(args.base_model_name)

    # 3. Base model
    print("Loading base model...")
    base_model = VisionEncoderDecoderModel.from_pretrained(args.base_model_name)
    pad_token_id = processor.tokenizer.pad_token_id
    decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_answer>")
    base_model.config.pad_token_id = pad_token_id
    base_model.config.decoder_start_token_id = decoder_start_token_id
    base_model.to(DEVICE)

    # 4. Fine-tuned model
    print("Loading fine-tuned model...")
    ft_model = VisionEncoderDecoderModel.from_pretrained(args.ft_model_dir)
    ft_model.config.pad_token_id = pad_token_id
    ft_model.config.decoder_start_token_id = decoder_start_token_id
    ft_model.to(DEVICE)

    # 5. Generate predictions
    print("\nGenerating predictions with BASE model...")
    base_preds, labels = generate_answers(base_model, processor, val_ds, n)

    print("Generating predictions with FINE-TUNED model...")
    ft_preds, _ = generate_answers(ft_model, processor, val_ds, n)

    # 6. Metrics
    print("\nComputing metrics...")

    base_metrics = compute_text_metrics(base_preds, labels)
    ft_metrics = compute_text_metrics(ft_preds, labels)

    base_acc = accuracy(base_preds, labels)
    ft_acc = accuracy(ft_preds, labels)

    # 7. Print results
    print("\n=== BASE MODEL (unichart-chartqa-960) ===")
    print(f"Accuracy: {base_acc:.2f}%")
    print(f"BLEU:     {base_metrics['bleu']:.2f}")
    print(f"ROUGE-L:  {base_metrics['rougeL']:.4f}")

    print("\n=== FINE-TUNED MODEL (ours) ===")
    print(f"Accuracy: {ft_acc:.2f}%")
    print(f"BLEU:     {ft_metrics['bleu']:.2f}")
    print(f"ROUGE-L:  {ft_metrics['rougeL']:.4f}")

    print("\nSample predictions (first 5):")
    for i in range(min(5, n)):
        print(f"\nExample {i+1}:")
        print(f"  Gold : {labels[i]}")
        print(f"  Base : {base_preds[i]}")
        print(f"  FT   : {ft_preds[i]}")


if __name__ == "__main__":
    main()
