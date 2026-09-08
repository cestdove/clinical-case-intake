import pandas as pd 
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# -- Load dataset --

df = pd.read_csv("datasets/clinical_cases.csv")

X = df["medical_abstract"]
y = df["condition_label"]


# -- Train / Test split --

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -- TF-IDF -- 

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -- Train model --

model = LinearSVC()

model.fit(X_train_vec, y_train)

# -- Evaluation -- 

predictions = model.predict(X_test_vec)

print("\nAccuracy: ")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# -- Save 

joblib.dump(model, "models/clinical_classifier.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print("\nModel saved.")