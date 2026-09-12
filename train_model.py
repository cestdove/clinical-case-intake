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


# evalutation 
predictions = model.predict(X_test_vec)

print("\nAccuracy: ")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# saving the model into the dir 
joblib.dump(model, "models/clinical_classifier.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print("\nModel saved.")