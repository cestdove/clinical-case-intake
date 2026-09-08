# 2_pipeline_ml.py
# Pipeline di Machine Learning con classificazione mediante Support Vector Machine (SVM)

import matplotlib.pyplot as plt  # per visualizzazione dei dati
import numpy as np  # per operazioni numeriche
import pandas as pd  # per la manipolazione dei dati nel dataframe
import seaborn as sns  # data visualization avanzata
from nltk.corpus import stopwords  # lista delle stopword
from sklearn.feature_extraction.text import TfidfVectorizer  # per il vettorizzatore TF-IDF
from sklearn.linear_model import LogisticRegression # per il benchmark di regressione logistica
from sklearn.model_selection import StratifiedKFold, cross_val_score # per la cross-validation a 5-fold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # metriche di performance
from sklearn.model_selection import train_test_split  # per lo split del dataset in train e test set
from sklearn.svm import LinearSVC  # modello scelto di SVM

# --- 0. CONFIGURAZIONE STOPWORD ITALIANE ---
# SCARICARE IL PACCHETTO SE NON E' STATO ANCORA FATTO
# nltk.download('stopwords') # eseguire questa riga togliendo il commento alla prima esecuzione

# Controllo dell'errore su assenza o presenza pacchetto NLTK essenziale a proseguire con efficacia
try:
    italian_stopwords = stopwords.words('italian')  # Lista stopword italiane
except LookupError:
    print("ERRORE: corpus NLTK 'stopwords' non trovato.")
    print('Esegui: python -c "import nltk; nltk.download(\'stopwords\')"')
    exit()


# --- 1. CARICAMENTO DATI E PREPARAZIONE ---
print("Caricamento dataset...")
try:  # Altro controllo dell'errore sulla presenza o assenza del CSV e quindi del dataframe pandas
    df = pd.read_csv('data/tickets.csv')
except FileNotFoundError:  # Avvisa se il file non esiste con consiglio
    print("ERRORE: File 'data/tickets.csv' non trovato. Esegui prima lo script di generazione del dataset.")
    exit()  # Interruzione preventiva del programma
print(f"Dataset caricato con successo. Numero di record: {len(df)}") # quando il file esiste si procede

required_columns = {'title', 'body', 'category', 'priority'}
missing_columns = required_columns.difference(df.columns) # Ennesimo controllo d'errore, sulle colonne essenziali
if missing_columns:
    print(f"ERRORE: colonne mancanti nel dataset: {sorted(missing_columns)}")
    exit()


# Combina titolo e corpo in una colonna unica di testo
df['full_text'] = df['title'].astype(str) + " " + df['body'].astype(str)

X = df['full_text']  # Testo integrale, o FEATURE X
y_categ = df['category']  # Etichette / label di categoria o primo TARGET y
y_prior = df['priority']  # Stesso discorso, per la priorità d'urgenza


# --- 2. PREPROCESSING E VETTORIZZAZIONE TF-IDF ---
print("\nVettorizzazione TF-IDF in corso...")

# Configurazione del vettorizzatore
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # cattura di unigrammi e bigrammi
    min_df=5,  # si ignorano i termini presenti in meno di 5 documenti
    max_df=0.85,  # si ignorano i termini presenti in più del 85% di documenti
    stop_words=italian_stopwords,
    lowercase=True  # conversione di tutti i caratteri in minuscolo per miglior gestione dei casi simili
)
print("Preprocessing del testo completato.")

# Trasformazione del testo in una matrice TF-IDF
X_vect = tfidf_vectorizer.fit_transform(X)  # matrice dei pesi
print(f"Matrice vettoriale creata. Numero di feature (termini): {X_vect.shape[1]}")  # numero delle feature

print("\n############## VALIDAZIONE INCROCIATA ##############")
# uso di cross-validation stratificata per preservare la distribuzione delle classi in ogni fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def stampa_risultati_cv(nome_task, X_data, y_data):
    """Compara empiricamente due modelli lineari senza influenzare i risultati finali"""
    print(f"\n{nome_task}")
    modelli = {
        # LinearSVC come baseline
        'LinearSVC': LinearSVC(random_state=42, dual='auto'),
        # LogisticRegression come benchmark
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42)
    }

    for nome_modello, modello in modelli.items():
        # Accuratezza lungo i 5 fold per ogni modello
        scores = cross_val_score(modello, X_data, y_data, cv=cv, scoring='accuracy')
        print(f"\nModello: {nome_modello}")
        print(f"Accuracy per fold: {[round(score, 4) for score in scores]}")
        print(f"Accuracy media: {scores.mean():.4f}")
        print(f"Deviazione standard: {scores.std():.4f}")


stampa_risultati_cv("Cross-validation - Categoria", X_vect, y_categ)
stampa_risultati_cv("Cross-validation - Priorita'", X_vect, y_prior)


print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# --- 3. MODELLO 1 - CLASSIFICAZIONE DI CATEGORIA ---
print("\n############## CLASSIFICAZIONE CATEGORIA ##############")

# Effetua uno split in set di train e test mantenendo la distribuzione delle classi
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_vect,
    y_categ,
    test_size=0.2,
    random_state=42,
    stratify=y_categ
)

print("\nDistribuzione category - train:")
print(y_train_c.value_counts().sort_index())
print("\nDistribuzione category - test:")
print(y_test_c.value_counts().sort_index())

# Crea e addestra modello SVM per classificazione di categoria
svm_categ = LinearSVC(random_state=42, dual='auto')  # creazione
svm_categ.fit(X_train_c, y_train_c)  # training su train set
y_pred_categ = svm_categ.predict(X_test_c)  # predizioni su test set

acc1 = accuracy_score(y_test_c, y_pred_categ)  # calcolo dell'accuracy
target_categ = np.sort(y_categ.unique()).tolist()  # lista ordinata di categorie uniche

print(f"Accuracy Totale: {acc1:.4f}")  # accuracy globale
print("\nDettagli per classe:")
print(classification_report(y_test_c, y_pred_categ, zero_division=0, target_names=target_categ))  # stampa report


cm = confusion_matrix(y_test_c, y_pred_categ, labels=target_categ)  # matrice di confusione
sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_categ, yticklabels=target_categ, cmap="magma_r")  # la rende una heatmap
plt.xlabel("Predetto"); plt.ylabel("Vero"); plt.title("Matrice di confusione - Categoria")  # etichette del grafico
plt.show()  # mostra il grafico


print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")  # Separator between the two models

# --- 4. MODEL 2: STIMA DELLA PRIORITA' ---
print("\n############## STIMA PRIORITÀ ##############")

# Train-test split per priorità
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_vect,
    y_prior,
    test_size=0.2,
    random_state=42,
    stratify=y_prior
)

print("\nDistribuzione priority - train:")
print(y_train_p.value_counts().sort_index())
print("\nDistribuzione priority - test:")
print(y_test_p.value_counts().sort_index())

# Crea modello SVM per priorità
svm_prior = LinearSVC(random_state=42, dual='auto')
svm_prior.fit(X_train_p, y_train_p)
y_pred_prior = svm_prior.predict(X_test_p)

acc2 = accuracy_score(y_test_p, y_pred_prior)  # accuracy
target_prior = np.sort(y_prior.unique()).tolist()  # lista

print(f"Accuracy Totale: {acc2:.4f}")  # accuracy
print("\nDettagli per classe:")
print(classification_report(y_test_p, y_pred_prior, zero_division=0, target_names=target_prior))  # report

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

cm = confusion_matrix(y_test_p, y_pred_prior, labels=target_prior)
# Calcola la matrice di confusione senza matplotlib o seaborn display
#print("\nMATRICE DI CONFUSIONE:")
#print(pd.DataFrame(cm, index=[f'Vero: {c}' for c in target_prior],
                      #  columns=[f'Predetto: {c}' for c in target_prior]))
# Visualizza la matrice di confusione con seaborn
sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_prior, yticklabels=target_prior, cmap="magma_r")  # CM heatmap
plt.xlabel("Predetto"); plt.ylabel("Vero"); plt.title("Matrice di confusione - Priorità")
plt.show()  # Display the plot


print("\nPipeline ML completata con successo.")


# --- 5. SALVATAGGIO VETTORIZZATORE E MODELLI PER LA DASHBOARD ---
import joblib  # Libreria per salvataggio modelli
joblib.dump(svm_categ, 'models/svm_categ.joblib')  # salva modello  di categoria
joblib.dump(svm_prior, 'models/svm_prior.joblib')  # salva modello di priorita'
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')  # salva TF-IDF
print("Modelli e vettorizzatore salvati nella cartella 'models/'.")
