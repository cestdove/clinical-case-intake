from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path("datasets")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Downloading Medical Abstracts dataset...")

dataset = load_dataset(
    "TimSchopf/medical_abstracts",
    split="train",
)

df = dataset.to_pandas()

output_file = OUTPUT_DIR / "clinical_cases.csv"
df.to_csv(output_file, index=False)

print(f"Dataset saved to {output_file}")
print(df.head())
print(f"\nTotal samples: {len(df)}")