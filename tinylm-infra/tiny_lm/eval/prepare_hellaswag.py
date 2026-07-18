# tiny_lm/eval/prepare_hellaswag.py

from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "hellaswag"
CACHE_DIR = DATA_DIR / "hf_cache"
SAVE_DIR = DATA_DIR / "validation"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading HellaSwag to cache dir: {CACHE_DIR}")

    ds = load_dataset(
        "Rowan/hellaswag",
        split="validation",
        cache_dir=str(CACHE_DIR),
    )

    print(ds)
    print("First example:")
    print(ds[0])

    print(f"Saving processed dataset to: {SAVE_DIR}")
    ds.save_to_disk(str(SAVE_DIR))

    print("Done.")

if __name__ == "__main__":
    main()