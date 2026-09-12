import pandas as pd 
import joblib 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# loading the two datasets already splitted 
df_train = pd.read_csv("datasets/clinical_cases_train.csv")
df_test = pd.read_csv("datasets/clinical_cases_test.csv")

# setting train/test input and target
X_train = df_train["medical_abstract"]
y_train = df_train["condition_label"]

X_test = df_test["medical_abstract"]
y_test = df_test["condition_label"]


# tf-idf vectorizer 
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# setting up the svm classifier model 
model = LinearSVC()

# fitting the model 
model.fit(X_train_vec, y_train)


# evalutation -------------------------
predictions = model.predict(X_test_vec)

print("\nAccuracy: ")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# adding a confusion matrix 
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt 

cm = confusion_matrix(y_test, predictions)
print(f"Confusion Matrix: \n {cm}") # just printing the raw one 

# labels for graphical comfort 
labels = [
    "1 \n Neoplasms",
    "2 \n Digestive",
    "3 \n Nervous",
    "4 \n Cardiovascular",
    "5 \n General"
]

plt.figure(figsize=(11, 8)) # resizing it because of long text labels
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted", fontweight="bold", fontsize=12)
plt.ylabel("Actual", fontweight="bold", fontsize=12)
plt.title("Confusion Matrix", fontweight="bold", fontsize=16)
plt.show()

# -------------------------------------


# saving the model into the dir 
joblib.dump(model, "models/clinical_classifier.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print("\nModel saved.")