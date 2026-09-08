# 3_dashboard.py
# Dashboard intuitiva per spiegabilità del modello

import os  # utilities sistema operativo
import joblib  # per persistenza dei modelli
import matplotlib.pyplot as plt  # per i grafici
import numpy as np  # per calcolo numerico
import pandas as pd  # per manipolare i dati
import seaborn as sns  # per grafici avanzati come heatmap etc
import streamlit as st  # per dashboard interattive sul web

# --- PERCORSI DEI MODELLI E DEL CSV ---
# La dashboard lavora sempre e solo in locale coi file già prodotti dalla pipeline
MODELS_PATH = 'models/'  # Cartella contenente i modelli salvati
INPUT_FILE_PATH = 'data/tickets.csv'  # path del file CSV in input dei ticket


# --- 0. CARICAMENTO FUNZIONI DEL MODELLO ---
@st.cache_resource  # Cache di modello e vettorizzatore per evitare ricaricamenti dal disco
def load_models():
    """Carica modelli e vettorizzatore dalla cartella."""

    try:  # Prova a caricare i modelli
        tfidf = joblib.load(os.path.join(MODELS_PATH, 'tfidf_vectorizer.joblib'))  # TF-IDF vectorizer
        model_categ = joblib.load(os.path.join(MODELS_PATH, 'svm_categ.joblib'))  # modello di category
        model_prior = joblib.load(os.path.join(MODELS_PATH, 'svm_prior.joblib'))  # modello di priority
        return tfidf, model_categ, model_prior  # ritorna gli oggetti caricati
    except FileNotFoundError:  # errore se i file sono assenti
        st.error(f"ERRORE: Modelli non trovati in {MODELS_PATH}. Esegui prima lo script 2_pipeline_ml.py.")
        return None, None, None  # restituisce None se i file sono assenti


# --- 1. IMPORTANZA GLOBALE DELLE FEATURE / COEFFICIENTI ---
def get_top_features_per_class(model, vectorizer, top_n=5):
    """Estrae le keyword più influenti per classe."""

    # LinearSVM mostra un vettore di coefficienti per classe
    feature_names = vectorizer.get_feature_names_out()  # nomi feature da TF-IDF
    classes = model.classes_  # classi dei modelli
    coefs = model.coef_  # coefficienti dei modelli

    importance_data = {}  # Dizionario per raccogliere l'importanza delle feature

    for i, class_name in enumerate(classes):  # si itera su ogni classe
        class_coefs = coefs[i]  # coefficienti della classe corrente

        # Si mantengono solo i valori più alti
        # Argsort ritorna gli indici dal più piccolo al più grande quindi si taglia l'ultimo
        # e si invertono le top_n posizioni
        top_indices = class_coefs.argsort()[-top_n:][::-1]  # indici delle migliori N feature
        top_features = feature_names[top_indices]  # nomi delle N feature migliori
        top_scores = class_coefs[top_indices]  # i rispettivi coefficienti

        # Formattazione leggibile dei coefficienti con il simbolo +
        formatted_scores = [f"{score:+.4f}" for score in top_scores]

        # Creazione dataframe per classe corrente per visualizzare meglio
        df = pd.DataFrame({
            'Parola Chiave': top_features,  # nomi delle keyword o token
            'Peso Numerico': top_scores,  # valori grezzi da usare per mappare
            'Coefficiente (Peso)': formatted_scores  # coefficienti formattati
        })

        importance_data[class_name] = df  # salvataggio dataframe nel dizionario

    return importance_data


def plot_token_distribution(importance_data, title, class_label):
    """Mostra i token più influenti e curva Gaussiana."""

    plot_rows = []
    for class_name, df_class in importance_data.items():
        # Copia ogni tabella di classe e gli associa l'etichetta per essere visto univocamente
        df_plot = df_class.copy()
        df_plot[class_label] = class_name
        plot_rows.append(df_plot)

    if not plot_rows:
        st.info("Nessun token disponibile per il grafico di distribuzione.")
        return

    # Ordina i coefficienti dal token più debole al token più forte
    df_plot = pd.concat(plot_rows, ignore_index=True).sort_values('Peso Numerico', ascending=True)
    # Applica la stessa palette di colore su entrambi i chart
    palette = sns.color_palette('viridis', n_colors=max(1, df_plot[class_label].nunique()))

    # Mostra il ranking di token esatto, affiancato da una distribuzione visiva a livello di classe
    fig, (ax_bar, ax_gauss) = plt.subplots(
        1,
        2,
        figsize=(15, max(4, len(df_plot) * 0.42)),
        gridspec_kw={'width_ratios': [1.4, 1]}
    )
    sns.barplot(
        data=df_plot,
        x='Peso Numerico',
        y='Parola Chiave',
        hue=class_label,
        dodge=False,
        palette=palette,
        ax=ax_bar
    )
    ax_bar.set_title(title)
    ax_bar.set_xlabel("Peso del token")
    ax_bar.set_ylabel("Token")
    ax_bar.legend(title="")

    # Costruisce un asse delle x condiviso dai coefficienti osservati prima di disegnare la curva Gaussiana per ogni classe
    all_scores = df_plot['Peso Numerico'].to_numpy()
    x_min = float(all_scores.min())
    x_max = float(all_scores.max())
    if x_min == x_max:
        # si espande di poco il range per avere l'asse delle x orizzontali ancora visibile con matplot
        x_min -= 0.1
        x_max += 0.1
    x_values = np.linspace(x_min, x_max, 300)

    for color, (class_name, df_class) in zip(palette, importance_data.items()):
        scores = df_class['Peso Numerico'].to_numpy(dtype=float)
        if len(scores) == 0:
            continue

        # Riassume i coefficienti specifici per classe con curva di Gauss che attraversa
        mean = float(scores.mean())
        std = float(scores.std(ddof=0))
        if std < 1e-6:

            std = 1e-6    # protegge la formula di densità quando i coefficienti collassano su un valore unico

        gaussian = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std) ** 2)
        ax_gauss.plot(x_values, gaussian, label=class_name, color=color, linewidth=2)
        # Un marker rende il centro di ogni distribuzione immediatamente visibile
        ax_gauss.axvline(mean, color=color, linestyle='--', linewidth=1, alpha=0.7)

    ax_gauss.set_title(f"Curva gaussiana - {class_label}")
    ax_gauss.set_xlabel("Peso del token")
    ax_gauss.set_ylabel("Densita'")
    ax_gauss.legend(title="")
    ax_gauss.grid(alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --- 2. Classificazione e logica batch ticket ---
def run_batch_classification(tfidf, svm_categ, svm_prior, show_status=True):
    """Esegue classificazione completa lungo tutto il dataset."""

    # Controlla l'esistenza del file siccome è usato su più tab
    if not os.path.exists(INPUT_FILE_PATH):
        st.error(f"ERRORE: File di input non trovato. Assicurati che il file esista al percorso: `{INPUT_FILE_PATH}`")  # mostra errore
        return None
    try:
        df_batch = pd.read_csv(INPUT_FILE_PATH)  # legge il CSV
        if 'title' not in df_batch.columns or 'body' not in df_batch.columns:  # controllo colonne obbligate
            st.error("Il file CSV caricato automaticamente non contiene le colonne obbligatorie 'title' e 'body'.")
            return None

        if show_status:   # messaggio a video in caso di successo
            st.success(f"File di input letto automaticamente da `{INPUT_FILE_PATH}`. Trovati {len(df_batch)} ticket da elaborare.")

        # Stessa preparazione del testo della pipeline
        df_batch['full_text'] = df_batch['title'].astype(str) + " " + df_batch['body'].astype(str)  # title + body
        # Ancora stesso lowercase + strip del preprocessing
        X_batch = tfidf.transform(df_batch['full_text'].apply(lambda x: x.lower().strip()))  # vettorizzazione

        # Entrambi i modelli partono dalla stessa matrice TF-IDF ma cercano due target differenti
        df_batch['predicted_category'] = svm_categ.predict(X_batch)  # predice category
        df_batch['predicted_priority'] = svm_prior.predict(X_batch)  # predice priority

        # Rimuove colonna di full_text
        df_output = df_batch.drop(columns=['full_text'], errors='ignore')

        return df_output  # ritorna dataframe con predizioni

    except Exception as e:  # controllo errori durante elaborazione file
        st.error(f"Si è verificato un errore durante l'elaborazione del file: {e}")
        return None


# --- 3. SETUP DASHBOARD STREAMLIT ---
# Divide l'interfaccia per obiettivo dell'utente
st.set_page_config(page_title="Ticket Triage dashboard", layout="wide")
st.title("Dashboard")
st.markdown("### Triage automatico dei ticket con TF-IDF e LinearSVM")

# Carica le risorse una volta prima che vengano usate dalle tab
tfidf_vectorizer, svm_categ, svm_prior = load_models()

if tfidf_vectorizer is None:

    st.stop()   # interrompe preventivamente il processo con assenza dei modelli

# Nomi tab
tab_batch, tab_compare, tab_explain = st.tabs([
    "**Esportazione CSV batch**",
    "**Confronto reale vs predetto**",
    "**Le 5 parole più influenti**"
])


# === TAB 1: ESPORTAZIONE BATCH CSV ===
with tab_batch:
    st.header("Esportazione CSV con le predizioni per un batch di ticket")

    # Tab di caricamento file, aggiunta predizioni, possibilità di download
    df_results = run_batch_classification(tfidf_vectorizer, svm_categ, svm_prior, show_status=True)

    if df_results is not None:
        st.markdown("---")
        st.success("Classificazione Batch Completata! Tutte le predizioni sono state aggiunte.")

        col_output, col_download = st.columns([3, 1])

        with col_output:
            st.caption("Anteprima del risultato finale con le nuove colonne di predizione:")
            # Mostra solo le ultime righe del file come anteprima
            st.dataframe(df_results.tail(10), use_container_width=True)

        with col_download:   # download file mostrato con aggiunta di predizioni
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Scarica CSV Risultati Finali",
                data=csv_data,
                file_name=f"triage_risultati_batch.csv",
                mime='text/csv',
                use_container_width=True
            )


# === TAB 2: COMPARAZIONE REALE VS PREDETTO ===
with tab_compare:
    st.header("Confronto tra valori effettivi e predetti")

    # Riusa la stessa pipeline di predizione senza banner di successo per evitare messaggi duplicati
    df_results = run_batch_classification(tfidf_vectorizer, svm_categ, svm_prior, show_status=False)

    if df_results is not None:
        if not {'category', 'priority'}.issubset(df_results.columns):
            st.info("Il file corrente non contiene le colonne effettive `category` e `priority` necessarie per il confronto.")
        else:
            st.caption("Confronto sulle distribuzioni e matrici di confusione del batch corrente.")

            # Aggrega le tabelle per migliore comparazione tra distribuzioni reali e predette
            category_compare = pd.DataFrame({
                'Effettiva': df_results['category'].value_counts().sort_index(),
                'Predetta': df_results['predicted_category'].value_counts().sort_index()
            }).fillna(0)

            # Usa interfaccia user-friendly invece di un sorting puramente alfabetico
            priority_order = ['Alta', 'Media', 'Bassa']
            priority_compare = pd.DataFrame({
                'Effettiva': df_results['priority'].value_counts().reindex(priority_order, fill_value=0),
                'Predetta': df_results['predicted_priority'].value_counts().reindex(priority_order, fill_value=0)
            })

            col_cat_cmp, col_prio_cmp = st.columns(2)

            with col_cat_cmp:
                st.caption("Categorie: effettive vs predette")
                # Per comprendere se si sovrastima o sottostima una categoria
                fig_cat, ax_cat = plt.subplots(figsize=(7, 4))
                category_compare.plot(kind='bar', ax=ax_cat, color=['#4C78A8', '#F58518'])
                ax_cat.set_xlabel("Categoria")
                ax_cat.set_ylabel("Numero ticket")
                ax_cat.legend(title="")
                ax_cat.tick_params(axis='x', rotation=0)
                st.pyplot(fig_cat)
                plt.close(fig_cat)

            with col_prio_cmp:
                st.caption("Priorita': effettive vs predette")
                # Mantiene un ordine fisso
                fig_prio, ax_prio = plt.subplots(figsize=(7, 4))
                priority_compare.plot(kind='bar', ax=ax_prio, color=['#54A24B', '#E45756'])
                ax_prio.set_xlabel("Priorita'")
                ax_prio.set_ylabel("Numero ticket")
                ax_prio.legend(title="")
                ax_prio.tick_params(axis='x', rotation=0)
                st.pyplot(fig_prio)
                plt.close(fig_prio)

            category_labels = sorted(df_results['category'].dropna().unique().tolist())
            # Si re-indicizzano gli assi per avere lo stesso ordine di categoria
            # e crosstab costruisce la matrice di confusione grezza contando le coppie reale / predetto
            category_cm = pd.crosstab(
                df_results['category'],
                df_results['predicted_category'],
                rownames=['Reale'],
                colnames=['Predetta'],
                dropna=False
            ).reindex(index=category_labels, columns=category_labels, fill_value=0)
            # Ordine esplicito per maggiore leggibilità della confusion matrix
            priority_cm = pd.crosstab(
                df_results['priority'],
                df_results['predicted_priority'],
                rownames=['Reale'],
                colnames=['Predetta'],
                dropna=False
            ).reindex(index=priority_order, columns=priority_order, fill_value=0)

            st.markdown("---")
            col_cm_cat, col_cm_prio = st.columns(2)

            with col_cm_cat:
                st.caption("Matrice di confusione - Categoria")
                # Heatmap che mostra i casi corretti sulla diagonale centrale
                fig_cm_cat, ax_cm_cat = plt.subplots(figsize=(6, 4))
                sns.heatmap(category_cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm_cat)
                ax_cm_cat.set_xlabel("Predetta")
                ax_cm_cat.set_ylabel("Reale")
                st.pyplot(fig_cm_cat)
                plt.close(fig_cm_cat)

            with col_cm_prio:
                st.caption("Matrice di confusione - Priorita'")
                # Risulta utile specialmente per la priorità, dove risultano più sbagli
                fig_cm_prio, ax_cm_prio = plt.subplots(figsize=(6, 4))
                sns.heatmap(priority_cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm_prio)
                ax_cm_prio.set_xlabel("Predetta")
                ax_cm_prio.set_ylabel("Reale")
                st.pyplot(fig_cm_prio)
                plt.close(fig_cm_prio)


# === TAB 3: ANALISI IMPORTANZA DEL MODELLO ===
with tab_explain:
    st.header('I Token più "pesanti": cosa guida le predizioni')
    st.markdown("Categoria prevista, priorità suggerita e 5 parole più influenti.")

    st.markdown("---")

    # Mostra prima la priorità
    st.subheader("Parole chiave per la priorità")
    top_prior_features = get_top_features_per_class(svm_prior, tfidf_vectorizer, top_n=5)

    # Mantiene l'ordine semantico e non alfabetto delle etichette di priorità
    col_order = ['Alta', 'Media', 'Bassa']
    cols = st.columns(len(col_order))

    for col, priority in zip(cols, col_order):
        if priority in top_prior_features:
            # Mostra solo le priorità realmente presenti nell'output del modello addestrato
            with col:
                st.markdown(f"#### Priorità {priority}")
                st.dataframe(
                    top_prior_features[priority][['Parola Chiave', 'Coefficiente (Peso)']],
                    hide_index=True,
                    use_container_width=True
                )

    st.caption("Distribuzione dei token piu' pesanti per la priorita'.")
    plot_token_distribution(
        top_prior_features,
        title="Distribuzione dei pesi dei token - Priorita'",
        class_label='Priorita'
    )

    st.markdown("---")

    # Finalmente mostra la categoria, con pattern più facili da apprendere per il modello SVM
    st.subheader("Parole Chiave per la categoria")
    top_categ_features = get_top_features_per_class(svm_categ, tfidf_vectorizer, top_n=5)

    # Le categorie provengono direttamente dal modello addestrato quindi la UI si adatta alle classi salvate
    classes = list(top_categ_features.keys())
    cols = st.columns(len(classes))

    for col, category in zip(cols, classes):
        # Ogni colonna ospita una tabella di spiegabilità specifica per classe
        with col:
            st.markdown(f"#### Categoria {category}")
            st.dataframe(
                top_categ_features[category][['Parola Chiave', 'Coefficiente (Peso)']],
                hide_index=True,
                use_container_width=True
            )

    st.caption("Distribuzione dei token piu' pesanti per la categoria.")
    plot_token_distribution(
        top_categ_features,
        title="Distribuzione dei pesi dei token - Categoria",
        class_label='Categoria'
    )
