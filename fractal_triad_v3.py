"""
FRACTAL TRIAD v3: Zoom-Coherent Knowledge Organization
=========================================================

Proof-of-concept for a knowledge organization framework based on
two core principles:

1. ZOOM COHERENCE: cause and effect should be matched at the same
   scale of observation. Mixing scales produces incoherent explanations.

2. MULTI-SCALE OBSERVATION: one observer per scale identifies
   cause-effect links with high precision. An orchestrator detects
   gaps (scales where links are missing) and maps the structural
   asymmetry between observable effects and known causes.

Architecture:
  - 9 Locked Observers (one per zoom level) → VERIFIED LINKS
  - 1 Orchestrator → classifies, detects gaps, reports mysteries

The full design envisions unlocked observers (spanning adjacent
scales) and LLM-based cross-scale reasoning, described in the paper
as future work.

Federico Piantoni & Claude (Anthropic AI) — February 2026
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass
from fastembed import TextEmbedding
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ============================================================
# CONFIGURATION
# ============================================================

ZOOM_LEVELS = [
    "cosmico", "planetario", "sociale", "organismo",
    "cellulare", "molecolare", "atomico", "subatomico", "fondamentale"
]
ZOOM_DEPTH = {level: i for i, level in enumerate(ZOOM_LEVELS)}

LOCKED_THRESHOLD = 0.35  # minimum cosine similarity for a verified link


# ============================================================
# EMBEDDING ENGINE
# ============================================================

class EmbeddingEngine:
    """
    All classification happens in embedding space — semantic meaning,
    not surface words. The architecture must operate in its native
    representation.
    """

    def __init__(self):
        print("  Inizializzazione spazio vettoriale...")
        self.model = TextEmbedding(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        self._cache = {}

        # Zoom level prototypes — rich descriptions of each scale
        zoom_descriptions = [
            "universo cosmologia big bang galassia spaziotempo espansione cosmica "
            "radiazione cosmica energia oscura materia oscura buco nero stelle "
            "supernova nebulosa redshift costante di Hubble",

            "terra pianeta clima atmosfera oceano ghiaccio ecosistema geologia "
            "biosfera temperatura globale riscaldamento climatico ghiacciai "
            "barriera corallina specie animali piante biodiversità estinzione "
            "api impollinatori foreste meteorologia stagioni",

            "economia società mercato finanza inflazione commercio politica "
            "cultura crisi prezzo moneta debito banca investimento occupazione "
            "disuguaglianza guerra popolazione migrazione istruzione",

            "corpo umano organo cuore cervello salute malattia stress trauma "
            "esperienza coscienza percezione dolore emozione meditazione "
            "comportamento psicologia sonno invecchiamento morte nascita",

            "cellula tessuto tumore apoptosi mitosi neurone sinapsi batterio "
            "telomero staminale membrana cellulare divisione cellulare "
            "invecchiamento cellulare sistema immunitario cortisolo telomerasi "
            "infiammazione citochina",

            "molecola proteina dna rna gene enzima neurotrasmettitore serotonina "
            "dopamina legame chimico reazione chimica pesticida neonicotinoide "
            "mutazione genetica p53 antibiotico resistenza batterica clorofilla",

            "atomo elettrone protone neutrone legame covalente orbitale ione "
            "reticolo cristallino struttura atomica legame idrogeno acqua "
            "stato solido liquido gas energia cinetica molecolare",

            "particella quantistica funzione d'onda entanglement spin fotone "
            "bosone campo quantistico sovrapposizione collasso della funzione "
            "d'onda dualismo onda-particella principio di indeterminazione "
            "non-località correlazione quantistica",

            "principio fondamentale legge universale conservazione energia "
            "entropia informazione emergenza evoluzione auto-organizzazione "
            "complessità simmetria risonanza caos ordine termodinamica "
            "azione reazione minimo energetico"
        ]
        self.zoom_embeddings = np.array(list(self.model.embed(zoom_descriptions)))

        # Cause/Effect prototypes
        effect_prototypes = [
            "si osserva un fenomeno misurabile nel mondo fisico",
            "un dato empirico mostra una variazione quantificabile",
            "si è registrato un cambiamento visibile e misurabile",
            "un evento si manifesta producendo conseguenze osservabili",
            "le misurazioni mostrano che qualcosa è aumentato o diminuito",
            "un fenomeno è stato rilevato e documentato",
        ]
        cause_prototypes = [
            "un meccanismo genera e produce un fenomeno attraverso un processo",
            "un principio fondamentale determina e regola il comportamento",
            "una forza o legge causa e spinge una trasformazione",
            "un processo interno attiva e controlla una reazione a cascata",
            "la ragione per cui qualcosa accade è un processo sottostante",
            "il fattore che determina il cambiamento è un meccanismo specifico",
        ]
        self.effect_embs = np.array(list(self.model.embed(effect_prototypes)))
        self.cause_embs = np.array(list(self.model.embed(cause_prototypes)))

        print(f"  Spazio: {self.zoom_embeddings.shape[1]} dimensioni")

    def embed(self, texts):
        new_texts = [t for t in texts if t not in self._cache]
        if new_texts:
            for t, e in zip(new_texts, self.model.embed(new_texts)):
                self._cache[t] = e
        return np.array([self._cache[t] for t in texts])

    def similarity_matrix(self, embs_a, embs_b):
        a_n = embs_a / (np.linalg.norm(embs_a, axis=1, keepdims=True) + 1e-10)
        b_n = embs_b / (np.linalg.norm(embs_b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_n, b_n.T)

    def classify_zoom(self, embedding):
        emb_n = embedding / (np.linalg.norm(embedding) + 1e-10)
        zoom_n = self.zoom_embeddings / (
            np.linalg.norm(self.zoom_embeddings, axis=1, keepdims=True) + 1e-10
        )
        sims = np.dot(zoom_n, emb_n)
        idx = int(np.argmax(sims))
        return {
            "primary_zoom": ZOOM_LEVELS[idx],
            "depth": idx,
            "zoom_confidence": round(float(sims[idx]), 4),
        }

    def classify_nature(self, embedding):
        emb_n = embedding / (np.linalg.norm(embedding) + 1e-10)
        eff_n = self.effect_embs / (
            np.linalg.norm(self.effect_embs, axis=1, keepdims=True) + 1e-10
        )
        cau_n = self.cause_embs / (
            np.linalg.norm(self.cause_embs, axis=1, keepdims=True) + 1e-10
        )
        eff_s = float(np.max(np.dot(eff_n, emb_n)))
        cau_s = float(np.max(np.dot(cau_n, emb_n)))

        margin = 0.02
        if eff_s > cau_s + margin:
            nature = "effect"
        elif cau_s > eff_s + margin:
            nature = "cause"
        else:
            nature = "ambiguous"

        return {
            "nature": nature,
            "effect_score": round(eff_s, 4),
            "cause_score": round(cau_s, 4),
        }


# ============================================================
# DATASET (62 items, curated for demonstration)
# ============================================================

KNOWLEDGE_DATASET = [
    # --- Cosmological ---
    {"id": 1,  "text": "L'universo si espande con accelerazione crescente"},
    {"id": 2,  "text": "L'energia oscura genera una pressione negativa che spinge lo spaziotempo ad espandersi"},
    {"id": 3,  "text": "La temperatura della radiazione cosmica di fondo è 2.725 Kelvin"},
    {"id": 4,  "text": "Il raffreddamento del plasma primordiale 380.000 anni dopo il Big Bang ha rilasciato i fotoni che osserviamo oggi come radiazione cosmica"},
    {"id": 5,  "text": "La luce si piega passando vicino a masse enormi"},
    {"id": 6,  "text": "La curvatura dello spaziotempo in presenza di massa-energia devia la traiettoria dei fotoni"},

    # --- Planetary ---
    {"id": 7,  "text": "La temperatura media globale è aumentata di 1.1°C rispetto al periodo preindustriale"},
    {"id": 8,  "text": "L'accumulo di gas serra nell'atmosfera trattiene il calore riflesso dalla superficie terrestre"},
    {"id": 9,  "text": "I ghiacciai artici si stanno sciogliendo a un ritmo senza precedenti"},
    {"id": 10, "text": "L'aumento della temperatura riduce la formazione di ghiaccio stagionale e accelera la fusione dei ghiacci permanenti"},
    {"id": 11, "text": "Le barriere coralline stanno sbiancando in tutto il mondo"},
    {"id": 12, "text": "L'acidificazione degli oceani per assorbimento di CO2 dissolve le strutture calcaree dei coralli"},
    {"id": 13, "text": "Gli eventi meteorologici estremi sono aumentati del 35% negli ultimi 30 anni"},
    {"id": 14, "text": "L'aumento dell'energia termica negli oceani intensifica i fenomeni convettivi e ciclonici"},
    {"id": 15, "text": "Le api stanno diminuendo in numero globalmente"},
    {"id": 16, "text": "I pesticidi neonicotinoidi interferiscono con il sistema nervoso degli impollinatori"},
    {"id": 37, "text": "Le foglie cambiano colore in autunno"},
    {"id": 38, "text": "La riduzione di clorofilla espone i pigmenti carotenoidi e antociani già presenti nella foglia"},
    {"id": 45, "text": "L'acqua bolle a 100 gradi Celsius al livello del mare"},

    # --- Social/Economic ---
    {"id": 17, "text": "L'inflazione negli USA ha raggiunto il 9.1% nel giugno 2022"},
    {"id": 18, "text": "L'espansione monetaria massiccia aumenta la quantità di moneta in circolazione riducendone il potere d'acquisto"},
    {"id": 19, "text": "I prezzi delle case sono aumentati del 40% in molte città dal 2020"},
    {"id": 20, "text": "La domanda che supera l'offerta in un mercato con vincoli di costruzione spinge i prezzi verso l'alto"},
    {"id": 21, "text": "Le crisi finanziarie si ripetono ciclicamente nella storia"},
    {"id": 22, "text": "L'eccessiva leva finanziaria crea fragilità sistemica che amplifica gli shock"},
    {"id": 23, "text": "Il Bitcoin ha perso il 65% del suo valore nel 2022"},
    {"id": 24, "text": "Il tightening monetario riduce la liquidità disponibile per asset speculativi"},

    # --- Organism ---
    {"id": 25, "text": "Il cuore umano batte circa 100.000 volte al giorno"},
    {"id": 26, "text": "Il nodo senoatriale genera impulsi elettrici ritmici che regolano la contrazione cardiaca"},
    {"id": 27, "text": "Le persone depresse mostrano ridotta attività nella corteccia prefrontale"},
    {"id": 28, "text": "Lo squilibrio nei neurotrasmettitori serotonina e dopamina altera i circuiti di ricompensa e motivazione"},
    {"id": 29, "text": "I bambini esposti a traumi mostrano iperattivazione dell'amigdala in età adulta"},
    {"id": 30, "text": "Il trauma precoce ricabla i circuiti neurali della risposta allo stress durante le finestre critiche di sviluppo"},
    {"id": 31, "text": "La meditazione regolare riduce i livelli di cortisolo nel sangue"},
    {"id": 32, "text": "L'attenzione focalizzata attiva la corteccia prefrontale e regola l'asse ipotalamo-ipofisi-surrene"},
    {"id": 33, "text": "Lo stress cronico accelera l'invecchiamento cellulare"},
    {"id": 34, "text": "Il cortisolo elevato riduce l'attività della telomerasi accorciando i telomeri"},
    {"id": 35, "text": "Le esperienze di pre-morte riportano pattern simili in culture diverse"},
    {"id": 36, "text": "La coscienza potrebbe essere un fenomeno fondamentale e non emergente"},

    # --- Cellular ---
    {"id": 39, "text": "Le cellule tumorali si moltiplicano in modo incontrollato"},
    {"id": 40, "text": "Mutazioni nel gene p53 disattivano il meccanismo di apoptosi cellulare"},
    {"id": 41, "text": "I batteri sviluppano resistenza agli antibiotici"},
    {"id": 42, "text": "La pressione selettiva favorisce i mutanti resistenti nella popolazione batterica"},
    {"id": 43, "text": "Lo stress cronico accorcia i telomeri nelle cellule immunitarie"},

    # --- Molecular ---
    {"id": 44, "text": "Il DNA contiene le istruzioni per costruire proteine"},
    {"id": 46, "text": "L'energia cinetica delle molecole supera la forza dei legami idrogeno permettendo il passaggio allo stato gassoso"},

    # --- Atomic ---
    {"id": 47, "text": "Il ghiaccio galleggia sull'acqua"},
    {"id": 48, "text": "Le molecole d'acqua formano legami idrogeno che creano una struttura cristallina meno densa allo stato solido"},
    {"id": 49, "text": "Le particelle subatomiche mostrano comportamento sia ondulatorio che corpuscolare"},

    # --- Subatomic ---
    {"id": 50, "text": "L'atto di osservazione collassa la funzione d'onda determinando lo stato della particella"},
    {"id": 51, "text": "Due particelle entangled mantengono correlazione istantanea a qualsiasi distanza"},
    {"id": 52, "text": "L'entanglement è una proprietà dello stato quantistico condiviso che trascende la località spaziale"},

    # --- Fundamental principles ---
    {"id": 53, "text": "L'energia non può essere creata né distrutta, solo trasformata"},
    {"id": 54, "text": "Ogni azione genera una reazione uguale e contraria"},
    {"id": 55, "text": "L'entropia di un sistema isolato tende sempre ad aumentare"},
    {"id": 56, "text": "I sistemi complessi tendono all'auto-organizzazione al bordo del caos"},
    {"id": 57, "text": "L'informazione non può essere distrutta, solo trasformata o dispersa"},
    {"id": 58, "text": "Il tutto è più della somma delle parti — proprietà emergenti nascono dall'interazione"},
    {"id": 59, "text": "Ciò che si osserva dipende dal punto di osservazione"},
    {"id": 60, "text": "La risonanza tra sistemi produce amplificazione e sincronizzazione"},
    {"id": 61, "text": "L'evoluzione procede attraverso variazione, selezione e replicazione"},
    {"id": 62, "text": "Ogni sistema tende al minimo energetico compatibile con i suoi vincoli"},
]


# ============================================================
# LOCKED OBSERVER
# ============================================================

@dataclass
class LockedLink:
    cause: dict
    effect: dict
    zoom_level: str
    similarity: float

@dataclass
class LockedReport:
    zoom_level: str
    depth: int
    effects: list
    causes: list
    ambiguous: list
    verified_links: list
    orphan_effects: list
    orphan_causes: list


class LockedObserver:
    """
    Observer LOCKED to a single zoom level.
    Sees ONLY cause-effect at its level. Maximum precision.
    """

    def __init__(self, engine, zoom_level, items):
        self.engine = engine
        self.zoom_level = zoom_level
        self.depth = ZOOM_DEPTH[zoom_level]
        self.effects = [i for i in items
                        if i["nature"] == "effect" and i["primary_zoom"] == zoom_level]
        self.causes = [i for i in items
                       if i["nature"] == "cause" and i["primary_zoom"] == zoom_level]
        self.ambiguous = [i for i in items
                          if i["nature"] == "ambiguous" and i["primary_zoom"] == zoom_level]

    def observe(self):
        verified = []
        matched_eff = set()
        matched_cau = set()

        if self.effects and self.causes:
            eff_embs = np.array([e["embedding"] for e in self.effects])
            cau_embs = np.array([c["embedding"] for c in self.causes])
            sim = self.engine.similarity_matrix(eff_embs, cau_embs)

            for i, effect in enumerate(self.effects):
                for j, cause in enumerate(self.causes):
                    if sim[i][j] > LOCKED_THRESHOLD:
                        verified.append(LockedLink(
                            cause={"id": cause["id"], "text": cause["text"]},
                            effect={"id": effect["id"], "text": effect["text"]},
                            zoom_level=self.zoom_level,
                            similarity=round(float(sim[i][j]), 4)
                        ))
                        matched_eff.add(effect["id"])
                        matched_cau.add(cause["id"])

        verified.sort(key=lambda x: x.similarity, reverse=True)

        return LockedReport(
            zoom_level=self.zoom_level,
            depth=self.depth,
            effects=self.effects,
            causes=self.causes,
            ambiguous=self.ambiguous,
            verified_links=verified,
            orphan_effects=[e for e in self.effects if e["id"] not in matched_eff],
            orphan_causes=[c for c in self.causes if c["id"] not in matched_cau],
        )


# ============================================================
# ORCHESTRATOR
# ============================================================

class Orchestrator:
    """
    Pipeline:
    1. Classify all items (zoom level + cause/effect nature)
    2. Deploy 9 locked observers → verified links
    3. Identify gaps and structural asymmetries
    4. Map cross-scale candidates (without pretending to validate them)
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.engine = EmbeddingEngine()
        self.classified = []

    def run(self):
        self._header()
        self._classify_all()

        # ═══ PHASE 1: Locked observers ═══
        print("\n" + "=" * 70)
        print("  FASE 1: OSSERVATORI BLOCCATI (uno per livello)")
        print("=" * 70)

        reports = {}
        for level in ZOOM_LEVELS:
            observer = LockedObserver(self.engine, level, self.classified)
            reports[level] = observer.observe()
            self._print_observer(reports[level])

        # ═══ PHASE 2: Gap analysis ═══
        print("\n" + "=" * 70)
        print("  FASE 2: ANALISI STRUTTURALE — GAP E ASIMMETRIE")
        print("=" * 70)

        gaps = self._analyze_gaps(reports)

        # ═══ PHASE 3: Cross-scale candidates ═══
        print("\n" + "=" * 70)
        print("  FASE 3: CANDIDATI CROSS-SCALA (richiedono validazione esterna)")
        print("=" * 70)

        self._cross_scale_candidates(reports)

        # ═══ FINAL REPORT ═══
        self._final_report(reports, gaps)

        # ═══ VISUAL REPORT ═══
        self._generate_visual_report(reports, gaps)

    def _header(self):
        print("\n" + "◆" * 70)
        print("  FRACTAL TRIAD v3: Zoom-Coherent Knowledge Organization")
        print()
        print('  "Causa ed effetto si corrispondono alla stessa scala."')
        print('  "Dove manca il link, c\'è un gap da esplorare."')
        print("◆" * 70 + "\n")

    def _classify_all(self):
        print("─" * 70)
        print("  FASE 0: Classificazione nello spazio vettoriale")
        print("─" * 70)

        texts = [item["text"] for item in self.dataset]
        embeddings = self.engine.embed(texts)

        for i, item in enumerate(self.dataset):
            emb = embeddings[i]
            zoom = self.engine.classify_zoom(emb)
            nature = self.engine.classify_nature(emb)
            self.classified.append({**item, "embedding": emb, **zoom, **nature})

        n_e = sum(1 for i in self.classified if i["nature"] == "effect")
        n_c = sum(1 for i in self.classified if i["nature"] == "cause")
        n_a = sum(1 for i in self.classified if i["nature"] == "ambiguous")
        print(f"  {len(self.classified)} items → Effetti:{n_e} Cause:{n_c} Ambigui:{n_a}")

        print(f"\n  Distribuzione per livello:")
        for level in ZOOM_LEVELS:
            items_at = [i for i in self.classified if i["primary_zoom"] == level]
            if items_at:
                ee = sum(1 for i in items_at if i["nature"] == "effect")
                cc = sum(1 for i in items_at if i["nature"] == "cause")
                aa = sum(1 for i in items_at if i["nature"] == "ambiguous")
                print(f"    {level:>13}: {len(items_at):>2} (E:{ee} C:{cc} A:{aa})")
                for item in items_at:
                    conf = item.get("zoom_confidence", 0)
                    tag = item["nature"][0].upper()
                    print(f"      [{tag}] (conf={conf:.3f}) {item['text'][:60]}")

    def _print_observer(self, report):
        n_e = len(report.effects)
        n_c = len(report.causes)
        n_a = len(report.ambiguous)
        n_links = len(report.verified_links)
        n_oe = len(report.orphan_effects)
        n_oc = len(report.orphan_causes)

        bar = "█" * (report.depth + 1)
        status = "✓" if n_links > 0 else ("⚠" if n_e + n_c > 0 else "·")

        print(f"  {bar} [{report.zoom_level:>13}] {status} "
              f"E:{n_e} C:{n_c} A:{n_a} | link:{n_links} | "
              f"orfani E:{n_oe} C:{n_oc}")

        for link in report.verified_links:
            print(f"      ✓ sim={link.similarity}  "
                  f"C:{link.cause['text'][:38]}... → "
                  f"E:{link.effect['text'][:38]}...")

    def _analyze_gaps(self, reports):
        gaps = {}

        for level, report in reports.items():
            total = len(report.effects) + len(report.causes) + len(report.ambiguous)
            n_links = len(report.verified_links)
            n_orphans = len(report.orphan_effects) + len(report.orphan_causes)
            n_e = len(report.effects)
            n_c = len(report.causes)

            issues = []

            if total > 0 and n_links == 0:
                issues.append("nessun link verificato")
            if n_orphans > n_links * 2 and n_links > 0:
                issues.append(f"troppi orfani ({n_orphans}) vs link ({n_links})")
            if n_e > 0 and n_c == 0:
                issues.append(f"{n_e} effetti senza cause → cause mancanti a questo zoom")
            elif n_c > 0 and n_e == 0:
                issues.append(f"{n_c} cause senza effetti → effetti non osservati a questo zoom")

            if issues:
                gaps[level] = issues
                for issue in issues:
                    print(f"  ⚠ [{level:>13}] {issue}")

        if not gaps:
            print("  Nessun gap critico rilevato.")

        # Asymmetry
        total_eff = sum(len(r.effects) for r in reports.values())
        total_cau = sum(len(r.causes) for r in reports.values())

        shallow = ["cosmico", "planetario", "sociale", "organismo"]
        deep = ["cellulare", "molecolare", "atomico", "subatomico", "fondamentale"]
        eff_shallow = sum(len(reports[l].effects) for l in shallow)
        cau_shallow = sum(len(reports[l].causes) for l in shallow)
        eff_deep = sum(len(reports[l].effects) for l in deep)
        cau_deep = sum(len(reports[l].causes) for l in deep)

        print(f"\n  ─── ASIMMETRIA STRUTTURALE ───")
        print(f"  Globale: {total_eff} effetti vs {total_cau} cause")
        print(f"  Livelli superficiali (cosm→org): E:{eff_shallow} C:{cau_shallow}")
        print(f"  Livelli profondi (cell→fond):     E:{eff_deep} C:{cau_deep}")
        print(f"  → La superficie è ricca di osservazioni, povera di spiegazioni.")
        print(f"  → La profondità è ricca di principi, povera di effetti osservabili.")

        return gaps

    def _cross_scale_candidates(self, reports):
        """
        For each orphan effect, show the closest cause at an adjacent level.
        These are NOT validated — they are shown as pointers for future
        exploration with proper reasoning (LLM or domain expert).

        The similarity score is a HINT of semantic proximity, not proof
        of causality. The "monetary expansion → universe expansion" type
        of false positive is exactly why this step requires reasoning,
        not just vector matching.
        """
        print()
        candidates_found = 0

        for i, level in enumerate(ZOOM_LEVELS):
            report = reports[level]
            if not report.orphan_effects:
                continue

            # Gather causes at adjacent levels
            adjacent = []
            if i > 0:
                adjacent.append(ZOOM_LEVELS[i - 1])
            if i < len(ZOOM_LEVELS) - 1:
                adjacent.append(ZOOM_LEVELS[i + 1])

            adj_causes = []
            for adj in adjacent:
                for c in reports[adj].causes:
                    adj_causes.append({**c, "from_zoom": adj})
                # Also include orphan causes
                for c in reports[adj].orphan_causes:
                    if c["id"] not in {ac["id"] for ac in adj_causes}:
                        adj_causes.append({**c, "from_zoom": adj})

            if not adj_causes:
                continue

            eff_embs = np.array([e["embedding"] for e in report.orphan_effects])
            cau_embs = np.array([c["embedding"] for c in adj_causes])
            sim = self.engine.similarity_matrix(eff_embs, cau_embs)

            level_cands = []
            for ei, eff in enumerate(report.orphan_effects):
                best_j = int(np.argmax(sim[ei]))
                best_sim = float(sim[ei][best_j])
                if best_sim > 0.25:
                    level_cands.append({
                        "effect": eff,
                        "cause": adj_causes[best_j],
                        "sim": round(best_sim, 4)
                    })

            if level_cands:
                level_cands.sort(key=lambda x: x["sim"], reverse=True)
                print(f"  [{level}] {len(level_cands)} effetti orfani con "
                      f"possibili cause ai livelli adiacenti:")
                for cand in level_cands[:5]:
                    c = cand["cause"]
                    e = cand["effect"]
                    print(f"    ? hint={cand['sim']} [{c['from_zoom']}→{level}]")
                    print(f"      C: {c['text'][:60]}")
                    print(f"      E: {e['text'][:60]}")
                candidates_found += len(level_cands)

        if candidates_found == 0:
            print("  Nessun candidato cross-scala individuato.")
        else:
            print(f"\n  ⚠ {candidates_found} candidati cross-scala identificati.")
            print(f"    Questi NON sono link verificati.")
            print(f"    La similarità semantica è un hint, non una prova di causalità.")
            print(f"    Validazione richiede ragionamento contestuale (LLM o esperto).")

    def _final_report(self, reports, gaps):
        total_links = sum(len(r.verified_links) for r in reports.values())
        total_items = len(self.classified)
        n_e = sum(1 for i in self.classified if i["nature"] == "effect")
        n_c = sum(1 for i in self.classified if i["nature"] == "cause")
        n_a = sum(1 for i in self.classified if i["nature"] == "ambiguous")
        levels_with = sum(1 for r in reports.values() if r.verified_links)

        explained_ids = set()
        for r in reports.values():
            for link in r.verified_links:
                explained_ids.add(link.effect["id"])

        mysteries = [i for i in self.classified
                     if i["nature"] == "effect" and i["id"] not in explained_ids]

        print("\n" + "◆" * 70)
        print("  REPORT FINALE")
        print("◆" * 70)

        print(f"""
  ╔═══════════════════════════════════════════════════════════════╗
  ║          TRIADE FRATTALE v3 — ZOOM COHERENCE                 ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║                                                               ║
  ║  Dataset:                                          {total_items:>3}         ║
  ║  Effetti / Cause / Ambigui:              {n_e:>3} / {n_c:>3} / {n_a:>3}       ║
  ║                                                               ║
  ║  Link verificati (stesso zoom):                    {total_links:>3}         ║
  ║  Livelli con link / totali:                      {levels_with} / 9         ║
  ║  Gap (livelli problematici):                       {len(gaps):>3}         ║
  ║  Misteri (effetti senza causa same-zoom):          {len(mysteries):>3}         ║
  ║                                                               ║
  ╚═══════════════════════════════════════════════════════════════╝""")

        print(f"\n  ─── TUTTI I LINK VERIFICATI ───")
        for level in ZOOM_LEVELS:
            for link in reports[level].verified_links:
                print(f"    [{level:>13}] sim={link.similarity}")
                print(f"      C: {link.cause['text'][:65]}")
                print(f"      E: {link.effect['text'][:65]}")

        print(f"\n  ─── GAP ───")
        for level, issues in sorted(gaps.items(), key=lambda x: ZOOM_DEPTH[x[0]]):
            for issue in issues:
                print(f"    ⚠ [{level:>13}] {issue}")

        print(f"\n  ─── MISTERI (effetti senza spiegazione alla stessa scala) ───")
        for m in mysteries:
            print(f"    ? [{m['primary_zoom']:>13}] {m['text'][:65]}")

        print(f"""
  ─── NOTE METODOLOGICHE ───
  • Proof-of-concept su dataset curato ({total_items} items).
  • I link verificati usano SOLO matching alla stessa scala (Zoom Coherence).
  • I candidati cross-scala sono hint semantici, non relazioni causali.
  • L'architettura completa prevede osservatori sbloccati con
    ragionamento LLM per la validazione cross-scala (lavoro futuro).""")

    def _generate_visual_report(self, reports, gaps):
        """Generate a visual dashboard as PNG."""

        # --- Color palette ---
        BG = '#0a0e1a'
        FG = '#e0e0e0'
        GOLD = '#d4a84b'
        CYAN = '#5bbce4'
        RED = '#e05555'
        GREEN = '#55e088'
        PURPLE = '#9b7fd4'
        GRID = '#1a2035'

        plt.rcParams.update({
            'figure.facecolor': BG, 'axes.facecolor': BG,
            'text.color': FG, 'axes.labelcolor': FG,
            'xtick.color': FG, 'ytick.color': FG,
            'font.family': 'sans-serif', 'font.size': 9,
        })

        fig = plt.figure(figsize=(18, 22))
        fig.suptitle('FRACTAL TRIAD v3 — Visual Report',
                      fontsize=20, fontweight='bold', color=GOLD, y=0.98)
        fig.text(0.5, 0.965, 'Zoom-Coherent Knowledge Organization',
                 ha='center', fontsize=11, color=FG, alpha=0.7)

        gs = GridSpec(4, 2, hspace=0.35, wspace=0.3,
                      left=0.08, right=0.92, top=0.94, bottom=0.03)

        levels_short = ['COS', 'PLA', 'SOC', 'ORG', 'CEL', 'MOL', 'ATO', 'SUB', 'FON']

        # ═══════════════════════════════════════════════
        # PANEL 1: Cause-Effect Distribution per level
        # ═══════════════════════════════════════════════
        ax1 = fig.add_subplot(gs[0, 0])
        n_eff = [len(reports[l].effects) for l in ZOOM_LEVELS]
        n_cau = [len(reports[l].causes) for l in ZOOM_LEVELS]
        n_amb = [len(reports[l].ambiguous) for l in ZOOM_LEVELS]
        x = np.arange(len(ZOOM_LEVELS))
        w = 0.25

        ax1.bar(x - w, n_eff, w, color=CYAN, alpha=0.85, label='Effetti')
        ax1.bar(x, n_cau, w, color=GOLD, alpha=0.85, label='Cause')
        ax1.bar(x + w, n_amb, w, color=PURPLE, alpha=0.85, label='Ambigui')
        ax1.set_xticks(x)
        ax1.set_xticklabels(levels_short, fontsize=8)
        ax1.set_ylabel('Conteggio')
        ax1.set_title('Distribuzione Causa / Effetto per Livello', color=GOLD, fontsize=12, pad=10)
        ax1.legend(fontsize=8, loc='upper right', framealpha=0.3)
        ax1.grid(axis='y', color=GRID, alpha=0.5)
        ax1.set_axisbelow(True)

        # ═══════════════════════════════════════════════
        # PANEL 2: Verified links heatmap
        # ═══════════════════════════════════════════════
        ax2 = fig.add_subplot(gs[0, 1])
        link_counts = [len(reports[l].verified_links) for l in ZOOM_LEVELS]
        orphan_e = [len(reports[l].orphan_effects) for l in ZOOM_LEVELS]
        orphan_c = [len(reports[l].orphan_causes) for l in ZOOM_LEVELS]

        ax2.barh(x, link_counts, 0.6, color=GREEN, alpha=0.85, label='Link verificati')
        for i, (lc, oe, oc) in enumerate(zip(link_counts, orphan_e, orphan_c)):
            label = f'{lc} link'
            if oe > 0:
                label += f' | {oe} eff. orfani'
            if oc > 0:
                label += f' | {oc} cau. orfani'
            ax2.text(max(link_counts) * 0.05 + lc, i, f' {label}',
                     va='center', fontsize=8, color=FG, alpha=0.8)

        ax2.set_yticks(x)
        ax2.set_yticklabels(levels_short, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel('Link verificati')
        ax2.set_title('Link Verificati e Orfani per Livello', color=GOLD, fontsize=12, pad=10)
        ax2.grid(axis='x', color=GRID, alpha=0.5)
        ax2.set_axisbelow(True)

        # ═══════════════════════════════════════════════
        # PANEL 3: Structural asymmetry (the two cones)
        # ═══════════════════════════════════════════════
        ax3 = fig.add_subplot(gs[1, :])

        shallow = ["cosmico", "planetario", "sociale", "organismo"]
        deep = ["cellulare", "molecolare", "atomico", "subatomico", "fondamentale"]
        eff_per_level = {l: len(reports[l].effects) for l in ZOOM_LEVELS}
        cau_per_level = {l: len(reports[l].causes) for l in ZOOM_LEVELS}

        # Draw the bicone representation
        max_val = max(max(eff_per_level.values()), max(cau_per_level.values()), 1)

        for i, level in enumerate(ZOOM_LEVELS):
            y = i
            ew = eff_per_level[level] / max_val * 4
            cw = cau_per_level[level] / max_val * 4

            # Effect bar (left)
            ax3.barh(y, -ew, 0.7, color=CYAN, alpha=0.8)
            if eff_per_level[level] > 0:
                ax3.text(-ew - 0.15, y, str(eff_per_level[level]),
                         ha='right', va='center', fontsize=9, color=CYAN, fontweight='bold')

            # Cause bar (right)
            ax3.barh(y, cw, 0.7, color=GOLD, alpha=0.8)
            if cau_per_level[level] > 0:
                ax3.text(cw + 0.15, y, str(cau_per_level[level]),
                         ha='left', va='center', fontsize=9, color=GOLD, fontweight='bold')

            # Level label (center)
            ax3.text(0, y, ZOOM_LEVELS[i].upper(), ha='center', va='center',
                     fontsize=9, fontweight='bold', color=FG,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=GRID, alpha=0.9))

            # Separator for shallow/deep
            if level == "organismo":
                ax3.axhline(y + 0.5, color=RED, alpha=0.4, linestyle='--', linewidth=1)
                ax3.text(-4.8, y + 0.5, '── superficie / profondità ──',
                         fontsize=7, color=RED, alpha=0.6, va='center')

        ax3.set_yticks([])
        ax3.set_xlim(-5.5, 5.5)
        ax3.invert_yaxis()
        ax3.set_title('Asimmetria Strutturale: Effetti ← → Cause',
                       color=GOLD, fontsize=12, pad=10)

        eff_patch = mpatches.Patch(color=CYAN, alpha=0.8, label='Effetti (← sinistra)')
        cau_patch = mpatches.Patch(color=GOLD, alpha=0.8, label='Cause (destra →)')
        ax3.legend(handles=[eff_patch, cau_patch], fontsize=9, loc='upper right', framealpha=0.3)
        ax3.grid(axis='x', color=GRID, alpha=0.3)
        ax3.axvline(0, color=FG, alpha=0.15, linewidth=1)

        # ═══════════════════════════════════════════════
        # PANEL 4: Verified links detail
        # ═══════════════════════════════════════════════
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        all_links = []
        for level in ZOOM_LEVELS:
            for link in reports[level].verified_links:
                all_links.append(link)
        all_links.sort(key=lambda l: l.similarity, reverse=True)

        ax4.set_title(f'Link Verificati ({len(all_links)} totali)',
                       color=GOLD, fontsize=12, pad=10)

        if all_links:
            y_pos = 0.95
            row_h = 0.058
            for i, link in enumerate(all_links):
                if y_pos < 0.02:
                    break
                sim_color = GREEN if link.similarity > 0.5 else (GOLD if link.similarity > 0.4 else RED)
                # Similarity bar
                bar_w = link.similarity * 0.12
                ax4.barh(y_pos, bar_w, row_h * 0.6, left=0.0, color=sim_color, alpha=0.6)
                ax4.text(0.13, y_pos, f'{link.similarity:.2f}', va='center',
                         fontsize=8, color=sim_color, fontweight='bold')
                ax4.text(0.17, y_pos, f'[{link.zoom_level}]', va='center',
                         fontsize=8, color=PURPLE)
                cause_txt = link.cause['text'][:50] + ('…' if len(link.cause['text']) > 50 else '')
                effect_txt = link.effect['text'][:50] + ('…' if len(link.effect['text']) > 50 else '')
                ax4.text(0.28, y_pos + 0.012, f'C: {cause_txt}', va='center',
                         fontsize=7.5, color=GOLD, alpha=0.9)
                ax4.text(0.28, y_pos - 0.012, f'E: {effect_txt}', va='center',
                         fontsize=7.5, color=CYAN, alpha=0.9)
                y_pos -= row_h

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

        # ═══════════════════════════════════════════════
        # PANEL 5: Mysteries
        # ═══════════════════════════════════════════════
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis('off')

        explained_ids = set()
        for r in reports.values():
            for link in r.verified_links:
                explained_ids.add(link.effect["id"])
        mysteries = [i for i in self.classified
                     if i["nature"] == "effect" and i["id"] not in explained_ids]

        ax5.set_title(f'Misteri — {len(mysteries)} effetti senza causa alla stessa scala',
                       color=RED, fontsize=12, pad=10)

        if mysteries:
            y_pos = 0.95
            row_h = 0.045
            for m in mysteries:
                if y_pos < 0.02:
                    break
                ax5.text(0.02, y_pos, '?', va='center', fontsize=10, color=RED, fontweight='bold')
                ax5.text(0.05, y_pos, f'[{m["primary_zoom"]}]', va='center',
                         fontsize=8, color=PURPLE)
                txt = m['text'][:80] + ('…' if len(m['text']) > 80 else '')
                ax5.text(0.18, y_pos, txt, va='center', fontsize=8, color=FG, alpha=0.85)
                y_pos -= row_h

        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)

        # Save
        out_path = 'fractal_triad_report.png'
        fig.savefig(out_path, dpi=150, facecolor=BG, edgecolor='none')
        plt.close(fig)
        print(f"\n  ★ Report visuale salvato in: {out_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    orchestrator = Orchestrator(KNOWLEDGE_DATASET)
    orchestrator.run()
