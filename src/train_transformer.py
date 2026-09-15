# useful libs
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# train/test load
df_train = pd.read_csv("datasets/clinical_cases_train.csv")
df_test = pd.read_csv("datasets/clinical_cases_test.csv")

# shape of dfs
print(df_train.shape)
print(df_test.shape)

# tokenizer chosen automatically for BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# listing the dfs to make them compatible with the tokenizer 
X_train = df_train["medical_abstract"].tolist()
y_train = df_train["condition_label"].tolist()

X_test = df_test["medical_abstract"].tolist()
y_test = df_test["condition_label"].tolist()


