from pathlib import Path    # to set the file paths

from datasets import load_dataset   # to import the datasets

OUTPUT_DIR = Path("datasets")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Downloading Medical Abstracts dataset...")

# splitting the dataset already into 2 csv 
dataset = load_dataset("TimSchopf/medical_abstracts") 
df_train = dataset["train"].to_pandas()
df_test = dataset["test"].to_pandas()


train_file = OUTPUT_DIR / "clinical_cases_train.csv"
test_file = OUTPUT_DIR / "clinical_cases_test.csv"

# convert all to csv format 
df_train.to_csv(train_file, index=False)
df_test.to_csv(test_file, index=False)

# check first rows 
print(f"Datasets saved to {train_file} and {test_file}")
print(df_train.head())
print(df_test.head())

# total samples for each 
print(f"\nTrain dataframe total samples: {len(df_train)}")
print(f"\nTest dataframe total samples : {len(df_test)}")