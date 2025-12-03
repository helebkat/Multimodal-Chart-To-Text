from datasets import load_dataset
from pathlib import Path

def load_chartqa_splits():
    """
    Loads the ChartQA dataset and returns small train/val subsets
    for quick experimentation.
    Uses the default config of HuggingFaceM4/ChartQA.
    """
    dataset = load_dataset("HuggingFaceM4/ChartQA")

    # Print available splits just to be sure
    print("Available splits:", list(dataset.keys()))

    train = dataset["train"]
    val = dataset["val"]      # <-- IMPORTANT: 'val', not 'validation'
    test = dataset["test"]

    print("Full dataset sizes:")
    print("  train:", len(train))
    print("  val  :", len(val))
    print("  test :", len(test))

    # Small subsets so we can iterate quickly
    small_train = train.shuffle(seed=42).select(range(min(1000, len(train))))
    small_val = val.shuffle(seed=42).select(range(min(200, len(val))))

    print("\nSubset sizes:")
    print("  small_train:", len(small_train))
    print("  small_val  :", len(small_val))

    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)

    return small_train, small_val, test

if __name__ == "__main__":
    load_chartqa_splits()
