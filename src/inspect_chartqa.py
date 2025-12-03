from datasets import load_dataset
from pathlib import Path
from PIL import Image

def main():
    dataset = load_dataset("HuggingFaceM4/ChartQA")
    print("Available splits:", list(dataset.keys()))
    train = dataset["train"]

    print("Train example keys:", train.column_names)

    example = train[0]
    print("\nExample 0 keys and values (non-image):")
    for k, v in example.items():
        if k != "image":
            print(f"{k}: {v}")

    img: Image.Image = example["image"]
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "example_chart_0.png"
    img.save(img_path)
    print(f"\nSaved example image to: {img_path}")

if __name__ == "__main__":
    main()
