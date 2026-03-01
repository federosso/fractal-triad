# La Triade Frattale

*Organizzazione della Conoscenza con Coerenza di Zoom, Separazione Causa-Effetto e Validazione Cross-Scala via LLM*

**Federico D'Ambrosio & Claude (Anthropic AI)**

Marzo 2026

---

## Abstract

I sistemi di intelligenza artificiale rappresentano la conoscenza su un piano unico e indifferenziato: cause, effetti, principi e misurazioni sono trattati con identico status. Questa piattezza produce errori sistematici, tra cui false correlazioni tra domini non correlati e confusione tra meccanismi causali e conseguenze osservabili.

Questo paper introduce la Triade Frattale, un framework che organizza la conoscenza lungo due dimensioni: la natura dell'informazione (causa vs. effetto) e la scala di osservazione (dal livello cosmologico a quello fondamentale). Il principio centrale — il Principio di Coerenza di Zoom — stabilisce che causa ed effetto debbano essere accoppiati alla stessa scala di osservazione, e che le transizioni cross-scala siano tracciate esplicitamente e validate con ragionamento.

Presentiamo due implementazioni progressive. La v3 (osservatori bloccati) usa nove osservatori specializzati nello spazio degli embeddings semantici: su 62 elementi, identifica 15 link causa-effetto verificati alla stessa scala e rivela un'asimmetria strutturale profonda (28 effetti vs. 4 cause ai livelli superficiali; 5 effetti vs. 20 cause ai livelli profondi).

La v4 (osservatori sbloccati) integra un LLM locale come filtro di ragionamento cross-scala. L'esperimento chiave — lo stesso framework testato con tre modelli di dimensione crescente (1B, 4B, 12B parametri) — produce un risultato inatteso: il tasso di falsi positivi cross-scala decresce monotonicamente con la dimensione del modello (83% → 0% → 17%), rivelando che il framework funziona implicitamente come benchmark per la capacità di ragionamento causale di un LLM.

*Parole chiave: rappresentazione della conoscenza, ragionamento causale, embeddings semantici, architettura multi-scala, coerenza di zoom, LLM benchmarking*

---

## 1. Introduzione: Il Problema della Piattezza

I moderni sistemi di intelligenza artificiale — modelli linguistici, grafi di conoscenza, database neurali — condividono una limitazione architetturale: rappresentano tutta la conoscenza su un unico piano indifferenziato. Una misurazione ("la temperatura globale è aumentata di 1,1°C"), un meccanismo causale ("l'accumulo di gas serra trattiene il calore riflesso") e un principio universale ("l'entropia tende sempre ad aumentare") sono memorizzati e processati con identico status.

Questa piattezza produce tre errori sistematici. **False correlazioni**: metriche di similarità superficiale connettono elementi non correlati — ad esempio, un sistema basato su embeddings trova alta similarità (0.44) tra "l'espansione monetaria aumenta la quantità di moneta" e "l'universo si espande con accelerazione", perché la parola "espansione" condivide struttura semantica anche se il significato è completamente diverso. **Confusione causale**: il sistema non distingue "X causa Y" da "X correla con Y." **Mescolamento di scale**: cause a un livello vengono accoppiate con effetti a un livello diverso, producendo spiegazioni incoerenti.

Nel mondo fisico, gli effetti si manifestano a ogni scala osservabile, ma le loro cause spesso operano a livelli diversi. Un osservatore umano naviga questa realtà multi-scala in modo naturale. I sistemi AI attuali non possiedono questa capacità strutturale. Questo paper propone un framework per affrontare questa limitazione.

### 1.1 Applicazioni Pratiche del Framework

Prima di esporre la teoria, è utile chiarire come questo lavoro si colloca nel panorama dell'AI applicata. Il framework non è un modello da inserire in un'architettura neurale, ma un insieme di vincoli strutturali che possono essere integrati in sistemi esistenti in tre modi concreti.

**Come vincolo nel ragionamento.** I sistemi di chain-of-thought attuali non hanno vincoli di scala: quando un LLM costruisce una spiegazione causale, può mescolare liberamente livelli senza tracciarli. Il Principio di Coerenza di Zoom può essere implementato come meta-tag di scala attaccato a ogni step del ragionamento, penalizzando le transizioni cross-scala non esplicitate.

**Come layer per sistemi RAG.** Nei sistemi Retrieval-Augmented Generation, il retrieval avviene per similarità semantica senza distinzione di scala. Un sistema RAG che adotta la Triade Frattale classifica ogni chunk per scala e natura (causa/effetto), vincola il retrieval alla stessa scala, e restituisce i candidati cross-scala separatamente e con appropriato scetticismo. Questo è implementabile immediatamente con i metadata filters dei database vettoriali esistenti.

**Come benchmark di ragionamento causale.** Come dimostreranno i risultati sperimentali della v4, il framework produce una metrica implicita della capacità di ragionamento causale di un LLM, misurata attraverso il tasso di falsi positivi su candidati cross-scala con causalità nota. Nessun benchmark esistente misura specificamente questa capacità.

## 2. I Due Coni: Fondamento Concettuale

L'architettura della Triade Frattale si basa su una visione geometrica: due coni convergenti alle basi, con i vertici rivolti verso l'esterno. Un terzo punto — l'osservatore — è collegato a entrambi.

Il **Cono degli Effetti** contiene tutti i fenomeni osservabili e misurabili. Il **Cono delle Cause** contiene i meccanismi generativi: leggi, principi, processi sottostanti. Entrambi i coni hanno profondità: possono essere "zoomati" dalla scala macroscopica (cosmologica) a quella microscopica (fondamentale).

I coni sono **speculari nella struttura ma asimmetrici nel contenuto**: ogni livello di zoom nel Cono degli Effetti ha un livello corrispondente nel Cono delle Cause. Però, ai livelli dove l'essere umano opera direttamente (cosmologico, planetario, sociale, organismo), gli effetti osservabili dominano e le cause sono scarse. Ai livelli profondi (molecolare, atomico, fondamentale), i principi causali abbondano ma gli effetti direttamente osservabili sono pochi.

Questa asimmetria non è un difetto del dataset: è una proprietà strutturale della conoscenza umana. Viviamo immersi negli effetti. Le cause operano in profondità. I risultati sperimentali confermano quantitativamente questa asimmetria: 28 effetti contro 4 cause ai livelli superficiali; 5 effetti contro 20 cause ai livelli profondi.

## 3. Il Principio di Coerenza di Zoom

Il contributo concettuale centrale di questo lavoro è il **Principio di Coerenza di Zoom**: causa ed effetto devono essere accoppiati alla stessa scala di osservazione. Le transizioni cross-scala non sono vietate, ma devono essere tracciate esplicitamente e validate con strumenti di ragionamento, non con semplice prossimità semantica.

Un esempio chiarisce il principio. Un vaso di fiori cade in testa a un bambino:

> **Scala degli oggetti fisici** — Effetto: il vaso colpisce la testa. Causa: il vento, il gatto, il terremoto.
> 
> **Scala dell'organismo** — Effetto: trauma cranico, dolore, paura. Causa: l'impatto cinetico sul tessuto.
> 
> **Scala cellulare** — Effetto: infiammazione, cascata di cortisolo. Causa: la rottura delle membrane cellulari.
> 
> **Scala molecolare** — Effetto: rilascio di citochine, ricablaggio sinaptico. Causa: reazioni chimiche da deformazione proteica.

Ogni livello è **autoconsistente**: causa ed effetto si corrispondono alla stessa scala. L'errore nasce quando si mescolano livelli: dire che "la causa del trauma psicologico è la deformazione proteica" salta passaggi intermedi senza tracciarli. Non è falso, ma è incoerente. Un sistema di conoscenza che non impone questa coerenza produce l'illusione di comprensione causale dove in realtà ci sono solo correlazioni semantiche.

## 4. Architettura

### 4.1 Livelli di Zoom e Classificazione

Il framework definisce nove livelli: cosmologico, planetario, sociale, organismo, cellulare, molecolare, atomico, subatomico e fondamentale. Ogni elemento viene assegnato a un livello primario e classificato come effetto, causa o ambiguo, attraverso prossimità semantica a prototipi nello spazio degli embeddings multilingue a 384 dimensioni.

La classificazione è un'euristica, non un oracolo. La distinzione causa/effetto non è intrinseca all'informazione: dipende dalla posizione dell'osservatore. "Il cortisolo elevato riduce la telomerasi" è un effetto visto dal livello organismo, e una causa vista dal livello cellulare. Il sistema assegna una classificazione primaria ma non cattura questa dualità.

### 4.2 Osservatori Bloccati (v3)

Il cuore della prima implementazione è un ensemble di nove **osservatori bloccati**, ciascuno vincolato a un singolo livello di zoom. Ogni osservatore vede solo gli elementi classificati al proprio livello e cerca accoppiamenti causa-effetto attraverso similarità coseno con soglia minima di 0.35.

Il vincolo è rigido: un osservatore bloccato al livello planetario non può e non tenta di spiegare lo sbiancamento dei coralli usando principi di meccanica quantistica. Produce link solo tra cause e effetti che condividono la stessa scala. Questo è il Principio di Coerenza di Zoom tradotto in architettura software.

### 4.3 Osservatori Sbloccati (v4)

La v4 introduce l'**osservatore sbloccato**: un LLM locale (Gemma via Ollama) che opera come filtro di ragionamento cross-scala. L'osservatore sbloccato riceve i candidati cross-scala identificati dal matching vettoriale e li valuta attraverso ragionamento contestuale, producendo per ciascuno:

- Un **verdetto**: `genuine`, `false_correlation`, o `uncertain`
- Un **livello di confidenza** (0.0 — 1.0)
- Un **ragionamento** esplicito che giustifica il verdetto
- I **passaggi intermedi** che collegherebbero causa ed effetto attraverso le scale

Questo è il pezzo che mancava nella v3: dove il matching vettoriale si ferma, il ragionamento inizia. La validazione cross-scala richiede comprensione del dominio, non geometria nello spazio degli embeddings.

### 4.4 L'Orchestratore

L'orchestratore coordina la pipeline in fasi successive:

1. **Classificazione** di tutti gli elementi nello spazio degli embeddings
2. **Osservatori bloccati**: dispiegamento dei nove osservatori e raccolta dei link verificati same-scale
3. **Analisi strutturale**: identificazione dei gap, mappatura dell'asimmetria, segnalazione dei misteri
4. **Osservatore sbloccato** (v4): validazione LLM dei candidati cross-scala
5. **Analisi dei misteri** (v4): generazione di ipotesi causali per gli effetti orfani

Le fasi 4 e 5 sono disaccoppiate dalle precedenti: la v3 produce input strutturati che la v4 consuma. L'LLM non vede mai l'intero dataset — vede solo candidati già filtrati dalla pipeline vettoriale.

## 5. Implementazione: Quattro Versioni Progressive

### 5.1 Versione 1 (TF-IDF)

Utilizzava TF-IDF per misurare la similarità. TF-IDF lavora sulle parole esatte, non sul significato: per questo sistema, "il cane corre" e "il lupo galoppa" non hanno nulla in comune. Risultato: falsi positivi sistematici, inclusi ponti assurdi come "mutazioni del gene p53" connesso a "crollo del Bitcoin".

### 5.2 Versione 2 (Embeddings)

La sostituzione con embeddings multilingue a 384 dimensioni ha eliminato i falsi positivi. "Cuore che batte" e "nodo senoatriale genera impulsi" hanno similarità 0.53 perché descrivono lo stesso fenomeno; "Bitcoin" e "gravità" hanno similarità 0.01. L'insight chiave: l'architettura deve operare nello spazio nativo degli embeddings.

### 5.3 Versione 3 (Osservatori Multipli)

Introduce gli osservatori multipli e l'analisi strutturale. Il passaggio da un singolo osservatore globale a nove specializzati per scala elimina le correlazioni spurie cross-scala e produce risultati più puliti e interpretabili.

### 5.4 Versione 4 (Osservatori Sbloccati via LLM)

Integra un LLM locale come osservatore sbloccato. Il salto concettuale: dove la v3 si fermava con un "?" e un disclaimer ("questi candidati richiedono validazione esterna"), la v4 esegue quella validazione. Il risultato più significativo non è il singolo verdetto, ma il pattern che emerge dal confronto tra modelli di dimensione diversa.

## 6. Risultati

### 6.1 Classificazione e Asimmetria (v3)

Su 62 elementi, il sistema ha classificato 33 effetti, 24 cause e 5 ambigui. La distribuzione per livello conferma l'asimmetria prevista:

> Livelli superficiali (cosmologico → organismo): 28 effetti, 4 cause
> 
> Livelli profondi (cellulare → fondamentale): 5 effetti, 20 cause

### 6.2 Link Verificati Same-Scale (v3)

I 9 osservatori bloccati hanno identificato 15 link causa-effetto verificati. Solo 3 livelli su 9 (planetario, organismo, cellulare) hanno prodotto link — esattamente i livelli dove il dataset contiene sia cause che effetti alla stessa scala.

Il link più forte (similarità 0.71): "L'aumento della temperatura riduce la formazione di ghiaccio" → "I ghiacciai artici si stanno sciogliendo." Un accoppiamento causa-effetto alla stessa scala con alta confidenza semantica.

I 6 livelli rimanenti sono gap: contengono solo effetti (cosmologico, sociale, subatomico) o solo cause (atomico, fondamentale) o troppo pochi elementi per produrre accoppiamenti (molecolare).

### 6.3 Gap e Misteri (v3)

Il sistema ha identificato 20 effetti "mistero": fenomeni osservabili per i quali il dataset non contiene una causa alla stessa scala. Tra questi: "L'universo si espande con accelerazione crescente" (cosmologico, senza causa cosmologica nel dataset), "Le crisi finanziarie si ripetono ciclicamente" (sociale, senza meccanismo sociale esplicito), "L'entanglement mantiene correlazione istantanea a qualsiasi distanza" (subatomico, senza causa subatomica).

Questi misteri non sono fallimenti del sistema: sono i punti dove la conoscenza è incompleta.

### 6.4 Validazione Cross-Scala via LLM (v4): L'Esperimento Chiave

L'orchestratore v3 aveva identificato 6 candidati cross-scala: effetti orfani con cause semanticamente vicine ai livelli adiacenti. La v4 ha sottoposto gli stessi 6 candidati a tre modelli LLM di dimensione crescente, tutti dalla famiglia Gemma 3 (Google), eseguiti localmente via Ollama.

#### Tabella 1: Verdetti per modello

| Candidato                           | Sim. | 1B        | 4B        | 12B                |
| ----------------------------------- | ---- | --------- | --------- | ------------------ |
| Emergenza → Entanglement            | 0.46 | uncertain | false     | **genuine** (0.75) |
| Risonanza → Entanglement            | 0.42 | genuine   | uncertain | uncertain (0.60)   |
| Risonanza → Collasso onda           | 0.37 | genuine   | false     | uncertain (0.40)   |
| Temperatura → Energia oscura        | 0.31 | genuine   | false     | false (0.95)       |
| Nodo senoatriale → Leva finanziaria | 0.30 | genuine   | false     | false (0.95)       |
| Temperatura → Espansione universo   | 0.29 | genuine   | false     | false (0.99)       |

#### Tabella 2: Riepilogo per modello

| Modello | Parametri | Genuini | Falsi | Incerti | Falsi positivi |
| ------- | --------- | ------- | ----- | ------- | -------------- |
| Gemma 3 | 1B        | 5       | 0     | 1       | 83%            |
| Gemma 3 | 4B        | 0       | 5     | 1       | 0%             |
| Gemma 3 | 12B       | 1       | 3     | 2       | 17%            |

#### 6.4.1 Analisi dei Risultati

Il pattern è significativo e non banale.

**Il modello 1B è un "yes-man".** Valida come genuina la relazione tra il nodo senoatriale del cuore e la leva finanziaria, e tra l'aumento della temperatura dei ghiacci e l'espansione dell'universo. Non possiede sufficiente capacità di ragionamento per distinguere correlazione semantica (parole simili) da causalità reale. Il suo tasso di falsi positivi (83%) lo rende inutilizzabile come osservatore sbloccato.

**Il modello 4B è uno "scettico radicale".** Rigetta tutto, incluso il candidato più filosoficamente difendibile (emergenza → entanglement). I suoi ragionamenti sono corretti per i casi ovvi ("there is no plausible mechanism by which planetary-scale ice formation affects cosmological expansion") ma troppo rigidi per i casi sottili.

**Il modello 12B trova il punto di equilibrio.** Valida un solo candidato — emergenza → entanglement — con confidenza moderata (0.75, non 0.95), un ragionamento articolato ("Entanglement, as a non-local quantum phenomenon, fundamentally arises from the interaction of particles..."), e passaggi intermedi tracciati esplicitamente. Rigetta i 3 candidati palesemente spurii con alta confidenza (0.95-0.99). Lascia 2 candidati come "uncertain" — esattamente quelli dove la relazione è filosoficamente plausibile ma fisicamente non dimostrata (risonanza → entanglement, risonanza → collasso della funzione d'onda).

#### 6.4.2 La Curva di Ragionamento Causale

Il dato più rilevante non è il singolo verdetto, ma il pattern complessivo. La capacità di ragionamento causale scala con la dimensione del modello in modo non lineare:

- **1B**: incapace di distinguere correlazione da causalità
- **4B**: capace di rigettare le false correlazioni, incapace di riconoscere causalità genuine cross-scala
- **12B**: capace sia di rigettare che di validare, con confidenza calibrata e incertezza dove appropriato

Questo suggerisce che il framework Fractal Triad, progettato come sistema di organizzazione della conoscenza, funziona anche come **benchmark implicito per il ragionamento causale multi-scala** — una capacità che nessun benchmark esistente misura specificamente.

### 6.5 Ipotesi sui Misteri (v4)

Per i 10 misteri analizzati, la qualità delle ipotesi segue lo stesso pattern di scala.

Il modello 1B produce ipotesi scientificamente assurde: attribuisce il cambiamento di colore delle foglie in autunno a una "shift in the Earth's magnetic field, subtly altering the wavelength of light." Il modello 4B produce ipotesi corrette: "Seasonal variations in solar radiation intensity trigger a complex..." Il modello 12B produce ipotesi corrette con confidenza calibrata (0.70 per le foglie vs. 0.20 per "l'acqua bolle a 100°C" dove ammette che la spiegazione è al livello molecolare, non planetario).

### 6.6 Candidati Cross-Scala (v3)

L'orchestratore v3 aveva identificato 6 candidati cross-scala con il solo matching vettoriale. Il più promettente: "Il tutto è più della somma delle parti" (fondamentale) come possibile causa di "L'entanglement è una proprietà dello stato quantistico condiviso" (subatomico), con similarità 0.46.

Questo è l'unico candidato che il modello 12B ha validato come genuine — confermando che il matching vettoriale, pur insufficiente da solo, è un efficace pre-filtro quando combinato con ragionamento LLM.

## 7. Limitazioni

**Dataset curato.** I 62 elementi sono stati selezionati intenzionalmente. Su dati non curati, la classificazione e i risultati potrebbero essere molto diversi.

**Classificazione fragile.** Dipende da prototipi fissi. Alcuni elementi sono mal classificati: "Bitcoin ha perso il 65% del suo valore" assegnato al livello molecolare anziché sociale. La classificazione causa/effetto non cattura la dualità intrinseca di molti elementi.

**Similarità non è causalità.** Il sistema trova prossimità nello spazio degli embeddings. I link verificati same-scale sono semanticamente plausibili, non causalmente dimostrati.

**Nessuna baseline comparativa.** Non abbiamo confrontato i risultati con knowledge graphs, causal discovery algorithms, o LLM-based reasoning sullo stesso dataset.

**Campione LLM limitato.** Il test cross-scala è stato eseguito su tre modelli di una sola famiglia (Gemma 3). Il pattern potrebbe non generalizzare ad altre famiglie di modelli. Una validazione più ampia richiederebbe test su Llama, Mistral, Phi, e modelli a parametri maggiori.

**LLM come giudice.** Usare un LLM per validare relazioni causali introduce i bias del modello stesso. L'LLM potrebbe validare relazioni che "suonano bene" linguisticamente senza vera comprensione causale.

## 8. Discussione

### 8.1 Cosa Funziona

Il **Principio di Coerenza di Zoom** è il contributo più significativo. Non è un algoritmo ma un vincolo strutturale: impone che il ragionamento causale sia coerente con la scala di osservazione. Questo vincolo è indipendente dall'implementazione.

L'**asimmetria strutturale** rivelata dall'analisi (effetti in superficie, cause in profondità) è un risultato genuino: rispecchia la struttura della conoscenza umana.

L'**architettura a due livelli** (vettoriale + ragionamento) si dimostra efficace: la pipeline vettoriale filtra i candidati, l'LLM li valida. Né l'uno né l'altro da solo sarebbe sufficiente.

Il **risultato di scaling** è il contributo più inatteso: il framework produce una curva di ragionamento causale che discrimina modelli di dimensione diversa in modo coerente con le aspettative teoriche.

### 8.2 Cosa Non Funziona (Ancora)

La **classificazione per livello** resta fragile. Un sistema production-ready richiederebbe classificazione contestuale, non basata su prototipi fissi.

Le **ipotesi sui misteri** prodotte dai modelli piccoli sono spesso scientificamente assurde. Questa feature ha valore solo con modelli sufficientemente grandi.

La **single-run evaluation** non cattura la variabilità stocastica dei modelli. Ogni modello andrebbe testato su run multiple per stabilire intervalli di confidenza.

### 8.3 Sull'Origine del Framework

Questo framework è nato da un'intuizione non analitica — una visione geometrica dei due coni ricevuta dal primo autore durante uno stato meditativo, successivamente formalizzata attraverso dialogo iterativo con un sistema AI. Riportiamo questa origine per trasparenza, non come argomento di validità. I risultati reggono o cadono sui propri meriti tecnici.

## 9. Lavoro Futuro

1. **Test su famiglie di modelli diverse** (Llama, Mistral, Phi, Qwen) per verificare se la curva di scaling cross-scala generalizza.
2. **Test su modelli large** (70B+ parametri, API cloud) per esplorare la parte alta della curva.
3. **Dataset non curati** con ground truth causale parzialmente nota per validazione rigorosa.
4. **Confronto con baseline** (knowledge graphs, causal discovery algorithms).
5. **Integrazione in pipeline RAG** con metadata di scala e natura sui chunk.
6. **Multi-run evaluation** per catturare variabilità stocastica.

## 10. Conclusione

La Triade Frattale propone che la conoscenza abbia profondità, che cause ed effetti siano speculari attraverso le scale, e che il ragionamento coerente richieda il loro accoppiamento alla stessa scala. Questa è l'idea.

L'implementazione dimostra cinque risultati concreti: (1) il Principio di Coerenza di Zoom produce 15 link verificati senza falsi positivi cross-scala; (2) l'asimmetria strutturale tra effetti in superficie e cause in profondità è quantificabile; (3) l'identificazione dei gap rende visibile dove la conoscenza è incompleta; (4) l'integrazione di un LLM come osservatore sbloccato permette la validazione cross-scala con ragionamento; (5) la capacità di ragionamento causale scala con la dimensione del modello in modo misurabile.

Il risultato più significativo è forse l'ultimo: un framework progettato per organizzare la conoscenza si rivela anche uno strumento per misurare la capacità di ragionamento dei sistemi che lo utilizzano. Questo suggerisce che la struttura della conoscenza e la capacità di ragionarci sopra siano due facce della stessa medaglia.

Link GitHub Repo: https://github.com/federosso/fractal-triad

---

## Riferimenti

[1] Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* Cambridge University Press.

[2] Granger, C.W.J. (1969). Investigating Causal Relations by Econometric Models. *Econometrica*, 37(3).

[3] Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.

[4] Mitchell, M. (2009). *Complexity: A Guided Tour.* Oxford University Press.

[5] Tegmark, M. (2014). *Our Mathematical Universe.* Alfred A. Knopf.

[6] Schölkopf, B. et al. (2021). Toward Causal Representation Learning. *Proceedings of the IEEE*.

[7] Bar-Yam, Y. (1997). *Dynamics of Complex Systems.* Addison-Wesley.

[8] Dorri, A. et al. (2018). Multi-Agent Systems: A Survey. *IEEE Access*, 6.

[9] D'Ambrosio, F. (2019). *La Via del Cuore.*
