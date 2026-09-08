# 1_generatore.py
# Generatore ticket di supporto con iniezione di rumore

import pandas as pd
import random

# Seed per rendere il dataset riproducibile
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def prima_maiuscola(testo: str) -> str:
    """Capitalizza solo la prima lettera"""
    testo = testo.strip()  # Rimuove spazi se presenti
    if not testo:
        return testo
    return testo[0].upper() + testo[1:]


def azione_per_titolo(azione: str) -> str:
    """Accorcia le azioni per i titoli dei ticket."""
    replacements = [
        ("risolvere il problema di ", ""),
        ("risolvere il problema con ", ""),
        ("risolvere l'errore di ", ""),
        ("risolvere l'errore ", ""),
        ("verificare la scadenza del ", "scadenza del "),
        ("verificare la scadenza della ", "scadenza della "),
        ("verificare il ", ""),
        ("verificare la ", ""),
        ("ripristinare la connessione al ", "connessione al "),
        ("ripristinare la connessione alla ", "connessione alla "),
        ("ripristinare l'accesso alla ", "accesso alla "),
        ("ripristinare l'accesso al ", "accesso al "),
        ("ripristinare il ", ""),
        ("ripristinare la ", ""),
        ("configurare il ", ""),
        ("configurare la ", ""),
        ("aggiornare il ", ""),
        ("aggiornare la ", ""),
        ("installare il ", ""),
        ("installare la ", ""),
        ("sostituire il ", ""),
        ("sostituire la ", ""),
        ("mappare la ", ""),
        ("mappare il ", ""),
        ("calcolare il ", ""),
        ("calcolare la ", ""),
        ("registrare il ", ""),
        ("registrare la ", ""),
        ("emettere il ", ""),
        ("emettere la ", ""),
        ("approvare la ", ""),
        ("approvare il ", ""),
        ("inviare il ", ""),
        ("inviare la ", ""),
        ("chiudere la ", ""),
        ("pianificare la ", ""),
        ("rinnovare il ", ""),
        ("rinnovare la ", ""),
        ("sbloccare il ", ""),
        ("sbloccare la ", ""),
    ]

    testo = " ".join(azione.split())
    testo_lower = testo.lower()

    # Si provano in ordine diversi prefissi noti e, appena combaciano, si riscrive l'inizio
    # della frase in modo da tenere i titoli brevi e puliti senza cambiare il resto
    for old, new in replacements:
        if testo_lower.startswith(old):
            resto = testo[len(old):]
            testo = new + resto
            break

    testo = testo.strip(" ,;:-")
    return testo


# --- 1. SETUP E CONFIGURAZIONE ---
DATASET_SIZE = 3000  # Numero ticket da generare

CRITICAL_TICKET_RATIO = 0.42  # Bilancia le classi


# --- 2. DIZIONARI CON AZIONI ---

SALUTI = [
    "Buongiorno,", "Ciao,", "Salve,", "Buonasera,", "Ciao a tutti,",
    "Gentile supporto,", "Team,", "Scusate il disturbo,", "Per favore,", ""
]

# Frasi terminative dei ticket generali per rumore.
RACCORDO = [
    "Vi invio questa mail per un aggiornamento.", "In allegato trovate i dettagli.",
    "Fatemi sapere se avete bisogno di altro.", "Resto a disposizione per ulteriori chiarimenti.",
    "Grazie per la vostra attenzione.", "Attendo un vostro riscontro.",
    "Potete farmi sapere le tempistiche?", "Scrivo per segnalare quanto segue.",
    "Spero si possa risolvere in fretta.", "Vi prego di darci priorità se possibile.", "Grazie in anticipo per l'aiuto.",
    "Vi ringrazio per la collaborazione.", "Resto in attesa di una vostra risposta.",
    "Per qualsiasi domanda, non esitate a contattarmi.", "Spero di ricevere presto notizie da voi.", "Vi auguro una buona giornata.",
    "Grazie per il vostro supporto continuo.", "Resto a disposizione per ulteriori informazioni.",
]

# Le azioni iniziano sempre col tempo verbale infinito per comodità di generazione
# Ogni categoria ha un proprio lessico tipico per permettere al modello di apprendere pattern
KEYWORDS = {
    'Amministrazione': {
        'azione': [
            # AZIONI AMBIGUE
            'verificare errore contabile generato dal gestionale CRM',
            'risolvere il problema con la fatturazione automatica del nuovo software',
            'sbloccare il pagamento fermo nel database',
            'allineare le provvigioni dei venditori sul foglio Excel',
            'correggere l\'anagrafica cliente errata a sistema',
            'registrare l\'incasso del contratto chiuso ieri sul portale',
            'sistemare il sistema di vendita che ha calcolato l\'IVA in modo errato',
            'riconciliare le fatture generate dalla piattaforma e-commerce',
            'verificare il centro di costo assegnato al nuovo server',
            'approvare la nota spese caricata dall\'app mobile',
            'stornare la fattura del cliente VIP dal gestionale',
            'scaricare il report delle vendite mensili per il bilancio',
            'contabilizzare i rimborsi per i disservizi di rete',
            'allineare il budget IT con le fatture dei fornitori cloud',
            'verificare perché il software di tesoreria non quadra con i pagamenti POS',
            # AZIONI TIPICHE
            'verificare il pagamento della fattura', 'approvare il budget trimestrale',
            'emettere la nota di credito', 'controllare la busta paga di questo mese',
            'sbloccare la transazione in sospeso', 'registrare il bonifico in entrata',
            'stornare l\'importo errato', 'autorizzare la nota spese del trasfertista',
            'inviare il modello F24 al commercialista', 'calcolare la ritenuta d\'acconto',
            'fornire il riepilogo del bilancio', 'liquidare i rimborsi arretrati',
            'sollecitare il pagamento del cliente insoluto', 'contabilizzare le fatture estere',
            "verificare la quadratura contabile del mese", "aggiornare il piano dei conti",
            "inserire la fattura elettronica nel sistema", "generare il report IVA trimestrale",
            "verificare la scadenza del DURC", "calcolare il TFR da liquidare",
            "approvare la richiesta di anticipo spese", "registrare la fattura del fornitore",
            "verificare la regolarità contributiva dell'azienda", "emettere il mandato di pagamento",
            "aggiornare il registro dei beni ammortizzabili", "calcolare l'IRAP da versare",
            "verificare la scadenza del pagamento del fornitore", "liquidare le fatture in sospeso",
            "approvare la richiesta di rimborso spese", "registrare la fattura del cliente",
            "verificare la regolarità fiscale dell'azienda", "emettere il mandato di incasso",
            "aggiornare il registro dei cespiti", "calcolare l'IVA da versare",
            "verificare la scadenza del pagamento del cliente", "liquidare le fatture in attesa",
            "approvare la richiesta di anticipo spese del dipendente", "registrare la fattura del fornitore estero",
            "verificare la regolarità contributiva del dipendente", "emettere il mandato di pagamento per il fornitore"
        ]
    },
    'Tecnico': {
        'azione': [
            # AZIONI AMBIGUE
            'riparare il PC dell\'ufficio pagamenti che non si avvia',
            'ripristinare il database delle fatture irraggiungibile',
            'risolvere il crash del gestionale CRM nell\'inserimento lead',
            'sbloccare la stampante per il contratto del cliente',
            'risolvere l\'errore 404 sul portale di emissione note di credito',
            'ripristinare il software HR per le buste paga',
            'stabilizzare la VPN che si disconnette scaricando il bilancio',
            'sbloccare il server di posta che non invia le offerte commerciali',
            'ripristinare il file Excel delle provvigioni cancellato per sbaglio',
            'collegare il tablet del direttore vendite al Wi-Fi',
            'risolvere l\'errore di sincronizzazione tra il CRM e il database contabile',
            'sbloccare l\'account del nuovo responsabile vendite',
            'sistemare la stampante dell\'amministrazione che inceppa i moduli F24',
            'sbloccare il gestionale degli ordini fermo al login',
            'ripristinare l\'accesso alla cartella condivisa dei contratti',
            # AZIONI TIPICHE
            'risolvere il problema di accesso alla VPN', 'riavviare il server principale',
            'resettare la password dell\'account email', 'configurare il nuovo firewall',
            'ripristinare il backup del database', 'sbloccare l\'utenza aziendale bloccata',
            'installare la patch di sicurezza', 'sostituire il monitor guasto alla postazione 4',
            'ottimizzare le query lente sul DB', 'mappare la cartella condivisa di rete',
            'risolvere l\'errore di login sull\'applicativo', 'sostituire il toner della stampante',
            'verificare i log di errore del sistema', 'aggiornare il certificato SSL scaduto',
            'configurare il nuovo access point Wi-Fi', 'ripristinare la connessione al server di posta',
            'installare l\'antivirus su tutti i PC', 'sostituire la tastiera guasta alla postazione 2',
            'ottimizzare le prestazioni del server', 'mappare la nuova stampante di rete',
            'risolvere il problema di sincronizzazione del calendario', 'sostituire la cartuccia della stampante',
            'verificare i log di sicurezza del sistema', 'aggiornare il firmware del router',
            'configurare il nuovo switch di rete', 'ripristinare la connessione al server FTP',
            'installare il software di monitoraggio', 'sostituire il mouse guasto alla postazione 5',
            'ottimizzare le prestazioni del database', 'mappare la nuova unità di rete',
            'risolvere il problema di sincronizzazione dei contatti', 'sostituire la batteria del laptop',
            'verificare i log di sistema per errori critici', 'aggiornare il software di backup',
            'configurare il nuovo server di posta', 'ripristinare la connessione al server di autenticazione',
            'installare il software di gestione remota', 'sostituire la scheda di rete guasta alla postazione 3',
            'ottimizzare le prestazioni del sistema operativo', 'mappare la nuova stampante multifunzione',
            'risolvere il problema di sincronizzazione dei file', 'sostituire il disco rigido guasto alla postazione 1',
            'verificare i log di sistema per errori di sicurezza', 'aggiornare il software antivirus'
        ]
    },
    'Commerciale': {
        'azione': [
            # AZIONI AMBIGUE
            'risolvere il problema tecnico nell\'inserimento del nuovo lead',
            'inviare il preventivo urgente aggirando il blocco VPN',
            'verificare se la fattura insoluta blocca l\'ordine del cliente VIP',
            'richiedere lo sblocco amministrativo per applicare lo sconto',
            'inviare la brochure aggirando il limite di dimensione email',
            'chiudere la trattativa bloccata dall\'errore di sistema',
            'supportare il cliente che non riesce a firmare digitalmente il contratto',
            'sollecitare l\'approvazione del budget per confermare l\'ordine',
            'chiedere l\'emissione urgente della fattura proforma per il lead',
            'forzare il CRM a salvare i dati della campagna di marketing',
            'verificare con l\'IT il tracciamento delle vendite sul sito web',
            'rinegoziare i termini di pagamento bloccati dall\'amministrazione',
            'correggere il software di preventivazione che calcola un margine errato',
            'recuperare la cronologia del cliente dal vecchio database vendite',
            'confermare la vendita nonostante il disservizio del portale',
            # AZIONI TIPICHE
            'approvare lo sconto straordinario per il cliente', 'inviare il preventivo aggiornato',
            'rinnovare il contratto in scadenza', 'inserire il nuovo lead nel CRM',
            'firmare l\'NDA con il nuovo partner', 'calcolare le provvigioni di fine mese',
            'chiudere la trattativa in corso su Salesforce', 'aggiornare il listino prezzi per il Q3',
            'pianificare la nuova campagna di vendita', 'verificare il target di fatturato',
            'disdire l\'abbonamento come richiesto dal cliente', 'inviare il forecast aggiornato',
            'rinnovare la licenza del software di vendita', 'inserire il nuovo contatto nel CRM',
            'firmare il contratto con il nuovo cliente', 'calcolare le provvigioni di fine trimestre',
            'chiudere la trattativa in corso su HubSpot', 'aggiornare il listino prezzi per il Q4',
            'pianificare la nuova campagna di marketing', 'verificare il target di fatturato per il prossimo anno',
            'disdire l\'abbonamento come richiesto dal cliente VIP', 'inviare il forecast aggiornato al management',
            'rinnovare la licenza del software di gestione clienti', 'inserire il nuovo lead qualificato nel CRM',
            'firmare il contratto con il nuovo partner commerciale', 'calcolare le provvigioni di fine anno',
            'chiudere la trattativa in corso su Pipedrive', 'aggiornare il listino prezzi per il prossimo anno',
            'pianificare la nuova campagna di acquisizione clienti', 'verificare il target di fatturato per il trimestre in corso',
            'disdire l\'abbonamento come richiesto dal cliente importante', 'inviare il forecast aggiornato al team di vendita',
            'rinnovare la licenza del software di analisi dei dati commerciali', 'inserire il nuovo contatto qualificato nel CRM',
            'firmare il contratto con il nuovo cliente strategico', 'calcolare le provvigioni di fine anno per il top seller',
            'chiudere la trattativa in corso su Zoho CRM', 'aggiornare il listino prezzi per il trimestre in corso',
            'pianificare la nuova campagna di retention clienti', 'verificare il target di fatturato per il mese in corso',
            'disdire l\'abbonamento come richiesto dal cliente chiave', 'inviare il forecast aggiornato al direttore commerciale',
            'rinnovare la licenza del software di gestione delle vendite', 'inserire il nuovo lead qualificato nel CRM',
            'firmare il contratto con il nuovo cliente chiave', 'calcolare le provvigioni di fine anno per il top performer',
            'chiudere la trattativa in corso su Microsoft Dynamics CRM', 'aggiornare il listino prezzi per il mese in corso'
        ]
    }
}

AGGETTIVI_CRITICITA = {
    'ALTA_CRITICITA': ['critica', 'vitale', 'improrogabile', 'bloccante', 'urgente', 'seria', 'grave', 'paralizzante', 'disastrosa', 'catastrofica'],
    'MEDIA_CRITICITA': ['importante', 'necessaria', 'rilevante', 'fondamentale', 'prioritaria', 'significativa', 'essenziale', 'seria',],
}

# Gli aggettivi espliciti aggravano l'overfitting
# A questo scopo si introducono segnali contestuali per indurre rumore
SEGNALI_CONTESTUALI = {
    'Alta': [
        "il cliente e' fermo",
        "non riusciamo a lavorare",
        "il reparto e' bloccato",
        "serve entro oggi",
        "la produzione e' ferma",
        "l'ordine non parte",
        "il pagamento risulta bloccato",
        "non possiamo chiudere la pratica"
    ],
    'Media': [
        "sta rallentando il lavoro",
        "serve un riscontro in giornata",
        "ci crea ritardo operativo",
        "va gestito appena possibile",
        "impatta il flusso del team",
        "rischiamo di andare fuori tempo"
    ]
}

# Qui si vuole simulare l'ambiguità di linguaggio che può trarre il modello in confusione
FALSI_ALLARMI = [
    "non e' un'urgenza reale ma vorrei tenerlo alto in coda",
    "non blocca il lavoro ma preferirei una verifica rapida",
    "non siamo fermi, pero' vorrei chiuderlo presto",
    "non e' critico, e' solo per evitare ritardi piu' avanti"
]

# --- 3. TEMPLATE PER FRASI ---
# Basici template completabili con le azioni etc esposte in precedenza
TEMPLATE = {
    'STANDARD': {
        'titolo': [
            "Richiesta: {azione}",
            "Assistenza per {azione}",
            "Necessità di {azione}",
            "Info su come {azione}",
            "Task: {azione}",
            "Supporto per {azione}",
            "Domanda: come {azione}?"
        ],
        'descrizione': [
            "{saluto} vi scrivo per chiedere di {azione}. {raccordo}",
            "{saluto} avrei bisogno di {azione} quando avete un attimo. {raccordo}",
            "{saluto} come posso procedere per {azione}? {raccordo}",
            "{saluto} ci sarebbe da {azione} per favore. {raccordo}",
            "{saluto} apro questo ticket perché devo {azione}. {raccordo}",
            "{saluto} vi contatto per {azione}. {raccordo}",
            "{saluto} avrei una richiesta: mi serve {azione}. {raccordo}",
            "{saluto} potete aiutarmi a {azione}? {raccordo}",
            "{saluto} ho bisogno di supporto per {azione}. {raccordo}",
            "{saluto} vorrei sapere come {azione}. {raccordo}",
            "{saluto} è necessario {azione}. {raccordo}",
            "{saluto} mi serve una mano per {azione}. {raccordo}",
            "{saluto} potete darmi indicazioni su come {azione}? {raccordo}",
            "{saluto} vorrei chiedervi di {azione}. {raccordo}"
        ]
    },
    'CRITICO': {
        'titolo': [
            "URGENTE: {azione}",
            "Problema {aggettivo}: {azione}",
            "Blocco operativo: {azione}",
            "CRITICO: {azione}",
            "Situazione {aggettivo}: {azione}",
            "Allarme: {azione}",
            "Emergenza: {azione} il prima possibile",
            "CRITICO: serve {azione} con massima urgenza",
            "Blocco totale: {azione}"
        ],
        'descrizione': [
            "{saluto} è assolutamente {aggettivo} {azione} il prima possibile. {raccordo}",
            "{saluto} la situazione è {aggettivo}, vi chiedo di {azione} immediatamente. {raccordo}",
            "{saluto} siamo fermi. È {aggettivo} poter {azione} per continuare a lavorare. {raccordo}",
            "{saluto} ticket escalato: serve {azione} con massima urgenza. È una questione {aggettivo}. {raccordo}",
            "{saluto} il cliente si lamenta, è {aggettivo} riuscire a {azione} entro oggi. {raccordo}",
            "{saluto} situazione critica: dobbiamo {azione} subito. {raccordo}",
            "{saluto} è un blocco totale, serve {azione} immediatamente. {raccordo}",
            "{saluto} è un'emergenza, dobbiamo {azione} ora. {raccordo}",
            "{saluto} è una situazione {aggettivo}, per favore fate in modo di {azione} con urgenza. {raccordo}",
            "{saluto} vi prego di {azione} immediatamente, la situazione è {aggettivo}. {raccordo}"
        ]
    }
}


# --- 4. ASSEGNAMENTO PRIORITA' PER KEYWORD ---
def assegna_priorita(titolo: str, descrizione: str) -> str:
    """Assegna la priorità (Alta/Media/Bassa) analizzando le keyword."""
    # La priorità è ottenuta dell'intero testo
    # Sia soggetto che corpo contribuiscono all'urgenza
    testo = (titolo + " " + descrizione).lower()

    # Alta priorità
    PRIORITY_HIGH_KEYWORDS = AGGETTIVI_CRITICITA['ALTA_CRITICITA']
    # Media priorità
    PRIORITY_MEDIUM_KEYWORDS = AGGETTIVI_CRITICITA['MEDIA_CRITICITA']
    # Segnali contestuali integrati per rendere l'addestramento meno triviale
    PRIORITY_HIGH_CONTEXT = [segnale.lower() for segnale in SEGNALI_CONTESTUALI['Alta']]
    PRIORITY_MEDIUM_CONTEXT = [segnale.lower() for segnale in SEGNALI_CONTESTUALI['Media']]

    # Il  controllo parte sempre dai segnali più forti e scorre verso i medi solo se necessario
    # Se non vi sono KW di Alta o Media priorità, il ticket ha priorità bassa
    if any(p in testo for p in PRIORITY_HIGH_KEYWORDS) or any(p in testo for p in PRIORITY_HIGH_CONTEXT):
        return 'Alta'
    elif any(p in testo for p in PRIORITY_MEDIUM_KEYWORDS) or any(p in testo for p in PRIORITY_MEDIUM_CONTEXT):
        return 'Media'
    return 'Bassa'


def aggiungi_rumore(testo: str, probabilita: float = 0.2) -> str:
    """
    Applica diversi tipi di rumore al testo
    con probabilità compresa in 0-1 che il testo viene modificato.
    """
    # Alcuni ticket restano puliti
    if random.random() > probabilita:
        return testo

    # Il tipo di rumore è scelto randomicamente
    tipo_rumore = random.choice(['maiuscolo', 'minuscolo', 'typo_inversione', 'typo_mancante', 'punteggiatura', 'prefissi_mail'])

    if tipo_rumore == 'maiuscolo':
        # Utente arrabbiato o CAPS LOCK attivo
        return testo.upper()

    elif tipo_rumore == 'minuscolo':
        # testo tutto in minuscolo, forse per fretta
        return testo.lower()

    elif tipo_rumore == 'typo_inversione' and len(testo) > 5:
        # Inversione di due caratteri adiacenti come "problema" -> "prbolema" per fretta o distrazione
        idx = random.randint(1, len(testo) - 3)
        return testo[:idx] + testo[idx+1] + testo[idx] + testo[idx+2:]

    elif tipo_rumore == 'typo_mancante' and len(testo) > 5:
        # Rimozione di un carattere come "fattura" -> "fatura" per stesse ragioni
        idx = random.randint(1, len(testo) - 2)
        return testo[:idx] + testo[idx+1:]

    elif tipo_rumore == 'punteggiatura':
        # Eccesso di punteggiatura emotiva
        return testo + random.choice(['!!!', '??', '... !!!', ' !!'])

    return testo


# --- 5. LOOP GENERATIVO PRINCIPALE ---
# Costruzione di ticket completi mischiando vocabolario, template, urgenza e rumore
dati = []
categorie = list(KEYWORDS.keys())

for i in range(1, DATASET_SIZE + 1):
    # La vera categoria è scelta prima di applicare il rumore
    category = random.choice(categorie)
    role_lexicon = KEYWORDS[category]

    # 1. Scelta degli elementi lessicali
    azione = random.choice(role_lexicon['azione'])
    # Il titolo usa un azione breve, cioè la normale ma abbreviata, per realismo d'oggetto
    azione_breve = azione_per_titolo(azione)
    raccordo = random.choice(RACCORDO)
    saluto = random.choice(SALUTI)
    # Optional description fragment used to add operational context to the priority.
    # It starts empty and is filled only in some cases so tickets do not all look too similar.
    contesto_priorita = ""

    # 2. Scelta del template standard / critica
    # Vengono generati molti ticket critici per bilanciare le classi
    if random.random() < CRITICAL_TICKET_RATIO:
        crit_type = random.choice(list(AGGETTIVI_CRITICITA.keys()))
        aggettivo = random.choice(AGGETTIVI_CRITICITA[crit_type])
        # Se il ticket è critico si potrebbe aggiungere più contesto coerente col caso
        priorita_attesa = 'Alta' if crit_type == 'ALTA_CRITICITA' else 'Media'
        # Ma non tutti i ticket hanno lo stesso livello di dettaglio
        if random.random() < 0.65:
            contesto_priorita = " " + random.choice(SEGNALI_CONTESTUALI[priorita_attesa]) + "."

        # Titolo e corpo provengono da template separati per dare variazione senza perdere consistenza
        title_template = random.choice(TEMPLATE['CRITICO']['titolo'])
        body_template = random.choice(TEMPLATE['CRITICO']['descrizione'])

        # Riempe i ticket critici con azione, aggettivo e contesto operazionale
        title_text = title_template.format(azione=azione_breve, aggettivo=aggettivo)
        body_text = body_template.format(azione=azione, aggettivo=aggettivo, raccordo=raccordo, saluto=saluto) + contesto_priorita

    else:
        # Lo stesso avviene con i ticket di tipo standard
        title_template = random.choice(TEMPLATE['STANDARD']['titolo'])
        body_template = random.choice(TEMPLATE['STANDARD']['descrizione'])

        # Anche per l'arricchimento, limitato però ad azione, raccordo e saluto
        title_text = title_template.format(azione=azione_breve)
        body_text = body_template.format(azione=azione, raccordo=raccordo, saluto=saluto)

        # SI aggiunge un poco di contesto via segnali testuali moderati anche in caso di urgenza bassa
        if random.random() < 0.18: # Limitatamente per un poco di rumore
            body_text += " " + random.choice(SEGNALI_CONTESTUALI['Media']) + "."

    # Si aggiungono i falsi allarmi per rumore d'urgenza
    # Rari intenzionalmente, si vuole confondere il modello rischiando di predire urgenza più alta
    if random.random() < 0.07:
        body_text += " " + random.choice(FALSI_ALLARMI) + "."

    # 3. Assegnamento della priorità logica
    # Anche qui, prima si deriva la priorità, poi opzionalmente si aggiunge l'errore umano
    priority = assegna_priorita(title_text, body_text)

    # --- SIMULAZIONE D'ERRORE UMANO (LABEL NOISE) ---
    # 1. Errore di categoria con scelta di dipartimento sbagliato
    if random.random() < 0.08:
        categorie_errate = [c for c in categorie if c != category]  # Esclude la categoria corretta
        category = random.choice(categorie_errate)  # Sceglie a caso un'altra categoria

    # 2. Errore di priorità con scelta di urgenza sbagliata
    if random.random() < 0.15:
        priority = random.choice(['Alta', 'Media', 'Bassa'])  # Scelta randomica di dipartimento
    # ---------------------------------------------

    # 4. Aggiunta del rumore di testo
    # Di nuovo, il rumore si applica dopo che le etichette sono già state scelte
    noisy_title = aggiungi_rumore(prima_maiuscola(title_text), probabilita=0.15)
    noisy_body = aggiungi_rumore(prima_maiuscola(body_text), probabilita=0.25)

    # Finalmente si aggiungono record del futuro dataframe
    dati.append({
        'id': i,
        'title': noisy_title,
        'body': noisy_body,
        'category': category,
        'priority': priority
    })


# --- 6. SALVATAGGIO FINALE ---
# Conversione della lista di ticket in un dataframe pandas per semplicità gestionale
df = pd.DataFrame(dati)
# Si riordinano le colonne per mantenere ordine e coerenza
df = df[['id', 'title', 'body', 'category', 'priority']]

# Salvataggio esplicito su CSV con virgola separatrice
df.to_csv('data/tickets.csv', index=False, sep=',')

print(f"\nGenerazione del dataset completata con successo.")
print(f"Numero totale di ticket generati: {len(df)}")
print(f"Disponibile su: data/tickets.csv")

print("\nREPORT DISTRIBUZIONE DELLE ETICHETTE")

print("\nPer categoria:")
print(df['category'].value_counts().sort_index())  # Stampa conteggio ticket per categoria in ordine alfabetico

print("\n" + "#" * 43 + "\n")  # Separatore visivo tra le due distribuzioni nella console

print("Per priorità:")
print(df['priority'].value_counts().sort_index())  # Avviene lo stesso ma per la priorità
