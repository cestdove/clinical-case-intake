# Report del Project Work: Ticket Triage automatizzato

## 1. Obiettivo

Questo progetto sviluppa un prototipo per il triage automatico di ticket di supporto aziendali.

L'obiettivo è quello di assegnare automaticamente a ogni ticket:

- una `category` tra `Amministrazione`, `Commerciale`, e `Tecnico`
- una `priority` tra `Alta`, `Media`, e `Bassa`

Il sistema è designato come uno strumento di supporto operazionale per la prima fase di instradamento, non come una sostituzione del giudizio umano.

## 2. Scenario di utilizzo 

Il flusso di utilizzo inteso è il seguente:

1. Un utente genera un ticket avente un `title` e un `body` come campi indipendenti.
2. Il testo viene trasformato in delle feature numeriche.
3. Un primo classificatore predice il dipartimento più adeguato.
4. Un secondo classificatore stima la priorità operazionale.
5. Le predizioni sono mostrate nella dashboard e possono essere esportate in CSV.

Questo approccio è utile quando il problema maggiore non è la stessa risoluzione del ticket, ma il suo corretto instradamento iniziale.

## 3. Repository Architecture

Il progetto è organizzato in tre componenti maggiori:

- `1_generatore.py`: genera il dataset sintetico 
- `2_pipeline_ml.py`: prepara i dati, compara i modelli lineari con cross-validation, addestra i modelli finali, li valuta, e salva gli artefatti
- `3_dashboard.py`: offre un'interfaccia su Streamlit per l'inferenza di un batch di ticket, le comparazioni e la spiegabilità

Artefatti e file di supporto:

- `data/tickets.csv`: dataset attuale
- `models/svm_categ.joblib`: modello di category 
- `models/svm_prior.joblib`: modello di priority 
- `models/tfidf_vectorizer.joblib`: vettorizzatore salvato
- `requirements.txt`: dipendenze del progetto

## 4. Dataset sintetico 

Il dataset non proviene da ticket reali, ma è stato generato artificialmente tramite `1_generatore.py`.

Ogni record contiene:

- `id`
- `title`
- `body`
- `category`
- `priority`

Lo script genera 3000 ticket e salva l'output ottenuto in `data/tickets.csv`.

### 4.1 Logica generativa 

Il generatore usa:

- vocabolari distinti per problemi amministrativi, tecnici e di vendita
- template linguistici standard e di urgenza critica
- segnali di testo che possono indicare priorità maggiore
- segnali contestuali che rendono la stima di priorità meno triviale
- una porzione di falsi allarmi per simulare ticket scritti con urgenza ma non sempre reale
- rumore di testo per simulare scrittura imperfetta
- rumore di etichetta per simulare l'errore umano della classificazione manuale

Si aggiunge che il generatore accorcia alcune azioni nel `title` per rendere l'oggetto più realistico, lasciando invece il `body` più descrittivo.

Il testo prodotto non è puramente randomico. Esso, aspira ad imitare situazioni plausibili come typos, capitalizzazione inconsistente, enfasi stile e-mail, e richieste che sembrano urgenti senza esserlo realmente. 

### 4.2 Vantaggi e limiti del dataset sintetico

Utilizzare dati sintetici è utile per costruire una pipeline completa, ma introduce dei limiti metodologici:

- i pattern linguistici sono più regolari di quelli trovati in ticket reali
- i vocabolari di classe rimangono ancora relativamente separati
- il rumore è controllato e non riflette la piena variabilità di un vero ambiente operazionale

Conseguentemente, i risultati osservati vogliono risultare un proof-of-concept o un progetto accademico, ma non sono abbastanza vari da stimare affidabilmente performance di produzione. 

## 5. Pipeline di Machine Learning

La pipeline implementata in `2_pipeline_ml.py` segue una struttura standard per la classificazione supervisionata di testo.

### 5.1 Preparazione dei dati

Il testo di ogni ticket è costruito con la concatenazione di:

- `title`
- `body`

in una nuova colonna chiamata `full_text`.

Prima dell'addestramento, la pipeline verifica anche la presenza delle colonne obbligatorie:

- `title`
- `body`
- `category`
- `priority`

### 5.2 Estrazione delle feature con TF-IDF

Il testo è trasformato con `TfidfVectorizer`, avente la seguente configurazione:

- `ngram_range=(1, 2)`
- `min_df=5`
- `max_df=0.85`
- Stopword italiane da `nltk.corpus.stopwords`
- `lowercase=True`

Questa scelta è coerente con il problema perché: 

- i bigrammi catturano espressioni utili di dominio
- rappresentazioni sparse lavorano bene su testi brevi
- il preprocessing rimane facile e spiegabile

### 5.3 Modelli utilizzati 

Il progetto tiene due modelli finali separati basati su `LinearSVC`:

- un modello per `category`
- l'altro modello per `priority`

La scelta di `LinearSVC` é appropriata in questa fase perché:

- lavora bene con feature sparse ad alta dimensionalità
- é rapido nell'addestramento e nell'inferenza
- offre buon bilanciamento tra performance ed interpretabilità

Oltre all'addestramento del modello finale, la pipeline offre un piccolo benchmark con `LogisticRegression` durante la cross-validation. Questo confronto non cambia gli artefatti salvati, ma aiuta a verificare se la baseline selezionata rimane competitiva su entrambi i problemi di classificazione.

## 6. Valutazione del modello

La pipeline combina due livelli di valutazione: 

- cross-validation
- holdout evaluation

### 6.1 Cross-Validation

Per entrambe le task, si sceglie un `StratifiedKFold` a 5-fold con:

- `shuffle=True`
- `random_state=42`

Per ognuna, la pipeline compara `LinearSVC` e `LogisticRegression` e mostra:

- fold accuracy
- accuracy media
- deviazione standard

Questa scelta migliora la robustezza della valutazione comparata con un singolo train/test split, perché riduce la dipendenza da una partizione, favorevole o meno, del dataset.

### 6.2 Holdout Evaluation

In aggiunta alla cross-validation, la pipeline mantiene una valutazione separata con:

- `train_test_split(test_size=0.2, random_state=42, stratify=...)`

per categoria e priorità.

Per ogni task, vengono prodotti i seguenti output:

- distribuzione delle classi nel training set
- distribuzione delle classi nel test set
- test accuracy
- `classification_report`
- confusion matrix visualizzata con `seaborn`

Questo approccio è utile per due ragioni:

- la cross-validation offre una stima più stabile della qualità media
- lo split di holdout permette una lettura più immediata degli errori e della confusione di classe

## 7. Interpretazione dei risultati 

Nel contesto attuale, è normale aspettarsi che:

- la classificazione di `category` raggiunge performance molto alte
- mentre per la `priority` appare più complicato

La ragione di ciò è strutturale:

- le categorie hanno vocabolari separati rigidamente l'uno dall'altro
- la priorità dipende più da segnali sottili complessi da cogliere per la macchina
- il generatore introduce volutamente del rumore di priorità per evitare testo lineare

Dunque, uno stacco tra i risultati attesi delle due task non dovrebbe essere interpretata come un malfunzionamento della pipeline, ma come una conseguenza della natura dei dati.

## 8. Dashboard su Streamlit 

La dashboard implementata nello script `3_dashboard.py` fa uso diretto di:

- `models/tfidf_vectorizer.joblib`
- `models/svm_categ.joblib`
- `models/svm_prior.joblib`
- `data/tickets.csv`

Se il modello o il dataset risultano assenti, l'applicazione riporta l'errore ed interrompe l'esecuzione.

### 8.1 Esportazione del Batch di ticket

La prima tab:

- legge automaticamente il file `data/tickets.csv`
- esegue le predizioni su tutti i ticket
- aggiunge la colonna `predicted_category`
- aggiunge anche `predicted_priority`
- mostra un'anteprima in forma tabulare 
- permette il download del CSV integrale 

### 8.2 Confronto Reale vs Predetto 

La seconda tap compara, sul batch corrente:

- distribuzioni di confronto reale e predetto per categoria
- medesima vista per la priorità
- diagrammi a barre per category e priority
- matrice di confusione per category
- stesso grafico per priority

Questa sezione non rimpiazza le valutazioni offline durante l'addestramento, ma rende più facile da ispezionare il comportamento del modello sul dataset locale caricato.

### 8.3 Spiegabilità globale

La terza tab espone una spiegazione globale del modello:

- estrae i coefficienti da `LinearSVC`
- selezione le 5 feature più influenti per ogni classe
- mostra i token e i pesi associati
- visualizza la distribuzione dei pesi dei token più rilevanti per ogni classe

Questa spiegabilità non è localizzata in un singolo ticket, ma è sufficiente a:

- verificare che il modello impari pattern plausibili
- discutere il comportamento del classificatore durante una presentazione espositiva
- identificare feature estremamente dominanti o spurie

## 9. Punti di forza del progetto

I maggiori punti di forza del modello sono:

- pipeline end-to-end completa
- struttura semplice e leggibile
- uso appropriato dei modelli per la classificazione del testo
- separazione chiara tra training e inferenza
- presenza di spiegabilità globale
- dashboard con esportazione di un batch
- cross-validation per una stima di performance più robusta

Per un progetto accademico o di natura dimostrativa, questa combinazione è più convincente di una soluzione più sofisticata ma meno controllabile.

## 10. Limiti tecnici e metodologici

### 10.1 Validità esterna del dataset

La limitazione principale è l'assenza di reali ticket resi anonimi.

Finché il modello non è validato su dati reali:

- la robustezza operazionale non può essere stimata efficacemente
- la piena variabilità linguistica del dominio non può essere osservata
- abbreviazioni, ambiguità e rumore tipici di un vero ambiente aziendale non possono essere misurati credibilmente

### 10.2 Benchmark ancora limitata

La pipeline include una comparazione utile tra `LinearSVC` e `LogisticRegression`, ma il benchmark resta limitato.

Alcune estensioni possibili sono prese in considerazione: 

- uso di metriche aggiuntive oltre all'accuratezza
- comparazione sistemica tra diverse configurazioni degli iperparametri
- introduzione di baseline aggiuntive come `MultinomialNB`

Questo punto è specialmente rilevante per il problema prioritario, che risulta intrinsecamente più instabile.

### 10.3 Explainability globale

La spiegabilità offerta è globale e non legata ad un singolo ticket.

Come conseguenza:

- permette l'interpretazione del comportamento medio del modello
- non giustifica ogni singola predizione individuale in modo preciso

### 10.4 Interfaccia focalizzata su batch locale 

La dashboard è funzionale, ma ancora orientata verso un uso unicamente dimostrativo:

- non accetta caricamento di CSV manuale
- non offre analisi per singolo ticket
- non mostra confidenza esplicita o punteggio di margine 

### 10.5 Preprocessing linguistico minimo

Il preprocessing è intenzionalmente leggero e non include: 

- lemmatizzazione 
- normalizzazione dei sinonimi di dominio
- gestione di abbreviazioni aziendali
- 
Questa scelta mantiene il sistema semplice e spiegabile, ma riduce la capacita di generalizzazione su varianti linguistiche più eterogenee.

## 11. Possibili miglioramenti futuri

I futuri sviluppi più probabili, in ordine di valore pratico, sono:

1. Integrazione di reali ticket anonimizzati.
2. Estensione della comparazione corrente tra `LinearSVC` e `LogisticRegression` con un benchmark più strutturato, magari con `MultinomialNB`.
3. Introduce an uncertainty threshold for ambiguous cases.
4. Aggiunta di upload CSV disponibile e inferenza per singolo ticket nella dashboard.
5. Migliore modularizzazione del codice con separazione dei processi più efficace e introduzione di funzioni proprietarie riutilizzabili lungo tutto il sistema senza loro ridefinizione.

## 12. Conclusione

Il progetto il questione vuole mostrare in maniera credibile che un sistema di triage automatizzato basato su `TF-IDF` e modelli lineari può essere implementato con una pipeline semplice e leggibile. 

La soluzione proposta è tecnicamente coerente e ricopre la generazione dei dati, la fase di training, valutazione, salvataggio dei modelli, e analisi dei dati attraverso una dashboard interattiva.

Il prossimo passaggio più importante non è quello di rendere i modelli più complessi, ma:

- validarli su dati reali
- compararli con un piccolo numero di benchmark ben scelti
- gestire casi esplicitamente incerti
