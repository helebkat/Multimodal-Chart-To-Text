import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("Using device:", DEVICE)

    # 1. Load UniChart model + processor
    model_name = "ahmed-masry/unichart-base-960"
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    # 2. Load one ChartQA example (default config)
    dataset = load_dataset("HuggingFaceM4/ChartQA")
    print("Available splits:", list(dataset.keys()))

    example = dataset["val"][0]
    image = example["image"]

    print("\nExample non-image fields:")
    for k, v in example.items():
        if k != "image":
            print(f"{k}: {v}")

    # 3. Build summarization-like prompt for UniChart
    # UniChart summarization prompt pattern
    task_prompt = "<summarize_chart> <s_answer>"

    # IMPORTANT: use keyword arguments (images=..., text=...)
    inputs = processor(
        images=image,
        text=task_prompt,
        return_tensors="pt"
    )

    pixel_values = inputs.pixel_values.to(DEVICE)

    # 4. Generate a textual output
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=pixel_values,
            max_length=256,
            num_beams=3
        )

    # 5. Decode
    sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print("\nRaw model output:")
    print(sequence)

if __name__ == "__main__":
    main()
