"""
FRACTAL TRIAD v4: Unlocked Observers via LLM
=========================================================

Extends v3 with LLM-based cross-scale reasoning.

v3 identifies WHERE knowledge is incomplete (gaps, mysteries,
cross-scale candidates). v4 uses a local LLM (Gemma via Ollama)
to REASON about those gaps — acting as the "unlocked observer"
described in the paper as future work.

Architecture:
  v3 (unchanged):
    - 9 Locked Observers → VERIFIED same-scale links
    - Orchestrator → classification, gaps, mysteries

  v4 (new):
    - Unlocked Observer (LLM) → validates cross-scale candidates
    - Mystery Analyst (LLM) → hypothesizes missing causes
    - Enhanced report with LLM reasoning

Requirements:
  - Ollama running locally (http://localhost:11434)
  - Model: gemma2:2b (or any model available in Ollama)
  - All v3 dependencies (fastembed, numpy, matplotlib)

Federico D'Ambrosio & Claude (Anthropic AI) — March 2026
"""

import json
import time
import requests
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# IMPORT v3 COMPONENTS
# ============================================================

from fractal_triad_v3 import (
    EmbeddingEngine,
    LockedObserver,
    LockedReport,
    LockedLink,
    KNOWLEDGE_DATASET,
    ZOOM_LEVELS,
    ZOOM_DEPTH,
    LOCKED_THRESHOLD,
)


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
#OLLAMA_MODEL = "gemma3:1b"
#OLLAMA_MODEL = "gemma3:4b"
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_TIMEOUT = 120  # seconds per request
CROSS_SCALE_SIM_THRESHOLD = 0.25  # minimum similarity to send to LLM


# ============================================================
# OLLAMA CLIENT
# ============================================================

class OllamaClient:
    """Minimal client for Ollama REST API."""

    def __init__(self, base_url=OLLAMA_URL, model=OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self._check_connection()

    def _check_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            r = requests.get(
                self.base_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            # Check if our model is available (with or without :latest tag)
            model_base = self.model.split(":")[0]
            available = any(model_base in m for m in models)
            if available:
                print(f"  ✓ Ollama connesso — modello: {self.model}")
            else:
                print(f"  ⚠ Ollama connesso ma '{self.model}' non trovato.")
                print(f"    Modelli disponibili: {models}")
                print(f"    Esegui: ollama pull {self.model}")
                raise SystemExit(1)
        except requests.ConnectionError:
            print("  ✗ Ollama non raggiungibile su localhost:11434")
            print("    Assicurati che Ollama sia in esecuzione: ollama serve")
            raise SystemExit(1)

    def generate(self, prompt, temperature=0.3):
        """
        Send a prompt to Ollama and return the response text.
        Uses low temperature for more deterministic reasoning.
        """
        try:
            r = requests.post(
                self.base_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 512,
                    }
                },
                timeout=OLLAMA_TIMEOUT
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.Timeout:
            return '{"error": "timeout"}'
        except Exception as e:
            return f'{{"error": "{str(e)}"}}'

    def generate_json(self, prompt, temperature=0.3):
        """Generate and parse JSON response. Returns dict or None."""
        raw = self.generate(prompt, temperature)

        # Try to extract JSON from response
        # Gemma sometimes wraps JSON in markdown blocks
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": "json_parse_failed", "raw": raw[:200]}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class CrossScaleVerdict:
    """Result of LLM validation for a cross-scale candidate."""
    effect_text: str
    effect_zoom: str
    cause_text: str
    cause_zoom: str
    semantic_similarity: float
    verdict: str          # "genuine", "false_correlation", "uncertain"
    confidence: float     # 0.0 — 1.0
    reasoning: str
    intermediate_steps: list = field(default_factory=list)

@dataclass
class MysteryHypothesis:
    """LLM-generated hypothesis for an unexplained effect."""
    effect_text: str
    effect_zoom: str
    hypothesized_cause: str
    hypothesized_scale: str
    reasoning: str
    confidence: float


# ============================================================
# UNLOCKED OBSERVER (LLM-based)
# ============================================================

class UnlockedObserver:
    """
    The unlocked observer sees ACROSS scales.

    Unlike locked observers (pure vector matching at same scale),
    the unlocked observer uses LLM reasoning to validate or reject
    cross-scale causal candidates identified by v3.

    This is the piece that was missing: reasoning, not geometry.
    """

    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def validate_candidate(self, effect, cause, similarity):
        """
        Ask the LLM: is this cross-scale link genuine or spurious?
        """
        prompt = f"""You are a scientific reasoning system. Evaluate if there is a genuine causal relationship between these two items that operate at different scales of observation.

EFFECT (scale: {effect['primary_zoom']}):
"{effect['text']}"

CANDIDATE CAUSE (scale: {cause['from_zoom']}):
"{cause['text']}"

Semantic similarity score: {similarity}

IMPORTANT: High semantic similarity does NOT mean causation. Words like "expansion" can appear in both economics and cosmology without any causal link.

Respond ONLY with this JSON:
{{
  "verdict": "genuine" or "false_correlation" or "uncertain",
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence explaining your verdict",
  "intermediate_steps": ["step1 connecting cause to effect", "step2"] or []
}}"""

        result = self.llm.generate_json(prompt)

        if result and "error" not in result:
            return CrossScaleVerdict(
                effect_text=effect["text"],
                effect_zoom=effect["primary_zoom"],
                cause_text=cause["text"],
                cause_zoom=cause["from_zoom"],
                semantic_similarity=similarity,
                verdict=result.get("verdict", "uncertain"),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", "no reasoning provided"),
                intermediate_steps=result.get("intermediate_steps", []),
            )
        else:
            return CrossScaleVerdict(
                effect_text=effect["text"],
                effect_zoom=effect["primary_zoom"],
                cause_text=cause["text"],
                cause_zoom=cause["from_zoom"],
                semantic_similarity=similarity,
                verdict="error",
                confidence=0.0,
                reasoning=f"LLM error: {result.get('error', 'unknown') if result else 'no response'}",
            )

    def hypothesize_mystery(self, mystery):
        """
        For an unexplained effect (mystery), ask the LLM to hypothesize
        what kind of cause might explain it and at what scale.
        """
        prompt = f"""You are a scientific reasoning system. This observed phenomenon has no known cause at its own scale of observation.

UNEXPLAINED EFFECT (scale: {mystery['primary_zoom']}):
"{mystery['text']}"

Hypothesize what kind of cause could explain this phenomenon.

Respond ONLY with this JSON:
{{
  "hypothesized_cause": "brief description of the hypothesized cause",
  "hypothesized_scale": one of {json.dumps(ZOOM_LEVELS)},
  "reasoning": "one sentence explaining why",
  "confidence": 0.0 to 1.0
}}"""

        result = self.llm.generate_json(prompt)

        if result and "error" not in result:
            return MysteryHypothesis(
                effect_text=mystery["text"],
                effect_zoom=mystery["primary_zoom"],
                hypothesized_cause=result.get("hypothesized_cause", "unknown"),
                hypothesized_scale=result.get("hypothesized_scale", "unknown"),
                reasoning=result.get("reasoning", "no reasoning"),
                confidence=float(result.get("confidence", 0.5)),
            )
        else:
            return MysteryHypothesis(
                effect_text=mystery["text"],
                effect_zoom=mystery["primary_zoom"],
                hypothesized_cause="LLM error",
                hypothesized_scale="unknown",
                reasoning=str(result),
                confidence=0.0,
            )


# ============================================================
# ORCHESTRATOR v4
# ============================================================

class OrchestratorV4:
    """
    Extended pipeline:
      Phase 0-2: identical to v3 (classify, locked observers, gaps)
      Phase 3:   LLM validates cross-scale candidates (NEW)
      Phase 4:   LLM hypothesizes causes for mysteries (NEW)
      Phase 5:   enhanced report with LLM reasoning
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.engine = EmbeddingEngine()
        self.classified = []
        self.reports = {}
        self.gaps = {}
        self.cross_scale_verdicts = []
        self.mystery_hypotheses = []

    def run(self):
        self._header()

        # ═══ v3 PIPELINE (phases 0-2) ═══
        self._classify_all()
        self._run_locked_observers()
        self._analyze_gaps()

        # ═══ v4 NEW: LLM REASONING (phases 3-4) ═══
        print("\n" + "=" * 70)
        print("  CONNESSIONE LLM")
        print("=" * 70)
        llm = OllamaClient()
        observer = UnlockedObserver(llm)

        self._validate_cross_scale(observer)
        self._hypothesize_mysteries(observer)

        # ═══ REPORTS ═══
        self._final_report()
        self._generate_visual_report()

    def _header(self):
        print("\n" + "◆" * 70)
        print("  FRACTAL TRIAD v4: Unlocked Observers via LLM")
        print()
        print('  "Dove il vettore si ferma, il ragionamento inizia."')
        print("◆" * 70 + "\n")

    # ─────────────────────────────────────────────
    # PHASE 0: Classification (from v3)
    # ─────────────────────────────────────────────

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

    # ─────────────────────────────────────────────
    # PHASE 1: Locked Observers (from v3)
    # ─────────────────────────────────────────────

    def _run_locked_observers(self):
        print("\n" + "=" * 70)
        print("  FASE 1: OSSERVATORI BLOCCATI (uno per livello)")
        print("=" * 70)

        for level in ZOOM_LEVELS:
            observer = LockedObserver(self.engine, level, self.classified)
            self.reports[level] = observer.observe()
            r = self.reports[level]

            bar = "█" * (r.depth + 1)
            n_links = len(r.verified_links)
            status = "✓" if n_links > 0 else ("⚠" if len(r.effects) + len(r.causes) > 0 else "·")
            print(f"  {bar} [{r.zoom_level:>13}] {status} "
                  f"E:{len(r.effects)} C:{len(r.causes)} A:{len(r.ambiguous)} | "
                  f"link:{n_links} | orfani E:{len(r.orphan_effects)} C:{len(r.orphan_causes)}")

            for link in r.verified_links:
                print(f"      ✓ sim={link.similarity}  "
                      f"C:{link.cause['text'][:38]}... → "
                      f"E:{link.effect['text'][:38]}...")

    # ─────────────────────────────────────────────
    # PHASE 2: Gap Analysis (from v3)
    # ─────────────────────────────────────────────

    def _analyze_gaps(self):
        print("\n" + "=" * 70)
        print("  FASE 2: ANALISI STRUTTURALE — GAP E ASIMMETRIE")
        print("=" * 70)

        for level, report in self.reports.items():
            total = len(report.effects) + len(report.causes) + len(report.ambiguous)
            n_links = len(report.verified_links)
            n_orphans = len(report.orphan_effects) + len(report.orphan_causes)
            n_e = len(report.effects)
            n_c = len(report.causes)
            issues = []

            if total > 0 and n_links == 0:
                issues.append("nessun link verificato")
            if n_e > 0 and n_c == 0:
                issues.append(f"{n_e} effetti senza cause → cause mancanti")
            elif n_c > 0 and n_e == 0:
                issues.append(f"{n_c} cause senza effetti → effetti non osservati")

            if issues:
                self.gaps[level] = issues
                for issue in issues:
                    print(f"  ⚠ [{level:>13}] {issue}")

        # Asymmetry summary
        shallow = ["cosmico", "planetario", "sociale", "organismo"]
        deep = ["cellulare", "molecolare", "atomico", "subatomico", "fondamentale"]
        eff_s = sum(len(self.reports[l].effects) for l in shallow)
        cau_s = sum(len(self.reports[l].causes) for l in shallow)
        eff_d = sum(len(self.reports[l].effects) for l in deep)
        cau_d = sum(len(self.reports[l].causes) for l in deep)

        print(f"\n  ─── ASIMMETRIA STRUTTURALE ───")
        print(f"  Livelli superficiali: E:{eff_s} C:{cau_s}")
        print(f"  Livelli profondi:     E:{eff_d} C:{cau_d}")

    # ─────────────────────────────────────────────
    # PHASE 3: LLM Cross-Scale Validation (NEW)
    # ─────────────────────────────────────────────

    def _validate_cross_scale(self, observer: UnlockedObserver):
        print("\n" + "=" * 70)
        print("  FASE 3: OSSERVATORE SBLOCCATO — Validazione Cross-Scala (LLM)")
        print("=" * 70)

        # Collect cross-scale candidates (same logic as v3)
        candidates = []
        for i, level in enumerate(ZOOM_LEVELS):
            report = self.reports[level]
            if not report.orphan_effects:
                continue

            adjacent = []
            if i > 0:
                adjacent.append(ZOOM_LEVELS[i - 1])
            if i < len(ZOOM_LEVELS) - 1:
                adjacent.append(ZOOM_LEVELS[i + 1])

            adj_causes = []
            for adj in adjacent:
                for c in self.reports[adj].causes:
                    adj_causes.append({**c, "from_zoom": adj})
                for c in self.reports[adj].orphan_causes:
                    if c["id"] not in {ac["id"] for ac in adj_causes}:
                        adj_causes.append({**c, "from_zoom": adj})

            if not adj_causes:
                continue

            eff_embs = np.array([e["embedding"] for e in report.orphan_effects])
            cau_embs = np.array([c["embedding"] for c in adj_causes])
            sim = self.engine.similarity_matrix(eff_embs, cau_embs)

            for ei, eff in enumerate(report.orphan_effects):
                best_j = int(np.argmax(sim[ei]))
                best_sim = float(sim[ei][best_j])
                if best_sim > CROSS_SCALE_SIM_THRESHOLD:
                    candidates.append({
                        "effect": eff,
                        "cause": adj_causes[best_j],
                        "sim": round(best_sim, 4)
                    })

        candidates.sort(key=lambda x: x["sim"], reverse=True)

        if not candidates:
            print("  Nessun candidato cross-scala da validare.")
            return

        print(f"  {len(candidates)} candidati da sottoporre all'LLM...\n")

        for idx, cand in enumerate(candidates):
            eff = cand["effect"]
            cau = cand["cause"]
            sim = cand["sim"]

            print(f"  [{idx+1}/{len(candidates)}] "
                  f"{cau['from_zoom']} → {eff['primary_zoom']} (sim={sim})")
            print(f"    C: {cau['text'][:60]}")
            print(f"    E: {eff['text'][:60]}")

            verdict = observer.validate_candidate(eff, cau, sim)
            self.cross_scale_verdicts.append(verdict)

            # Verdict symbol
            sym = {"genuine": "✓", "false_correlation": "✗",
                   "uncertain": "?", "error": "⚠"}.get(verdict.verdict, "?")
            print(f"    → {sym} {verdict.verdict} "
                  f"(conf={verdict.confidence:.2f}): {verdict.reasoning[:70]}")
            if verdict.intermediate_steps:
                for step in verdict.intermediate_steps[:3]:
                    print(f"      ↳ {step[:65]}")
            print()

        # Summary
        genuine = sum(1 for v in self.cross_scale_verdicts if v.verdict == "genuine")
        false_c = sum(1 for v in self.cross_scale_verdicts if v.verdict == "false_correlation")
        uncertain = sum(1 for v in self.cross_scale_verdicts if v.verdict == "uncertain")
        errors = sum(1 for v in self.cross_scale_verdicts if v.verdict == "error")

        print(f"  ─── RIEPILOGO VALIDAZIONE CROSS-SCALA ───")
        print(f"  Genuini: {genuine} | False correlazioni: {false_c} | "
              f"Incerti: {uncertain} | Errori: {errors}")

    # ─────────────────────────────────────────────
    # PHASE 4: LLM Mystery Hypotheses (NEW)
    # ─────────────────────────────────────────────

    def _hypothesize_mysteries(self, observer: UnlockedObserver):
        print("\n" + "=" * 70)
        print("  FASE 4: ANALISI MISTERI — Ipotesi sulle cause mancanti (LLM)")
        print("=" * 70)

        # Collect mysteries: effects with no cause at same scale
        explained_ids = set()
        for r in self.reports.values():
            for link in r.verified_links:
                explained_ids.add(link.effect["id"])
        mysteries = [i for i in self.classified
                     if i["nature"] == "effect" and i["id"] not in explained_ids]

        if not mysteries:
            print("  Nessun mistero da analizzare.")
            return

        # Limit to avoid excessive LLM calls (mysteries can be many)
        max_mysteries = 10
        if len(mysteries) > max_mysteries:
            print(f"  {len(mysteries)} misteri trovati, analisi dei primi {max_mysteries}...\n")
            mysteries = mysteries[:max_mysteries]
        else:
            print(f"  {len(mysteries)} misteri da analizzare...\n")

        for idx, mystery in enumerate(mysteries):
            print(f"  [{idx+1}/{len(mysteries)}] [{mystery['primary_zoom']}] "
                  f"{mystery['text'][:60]}")

            hyp = observer.hypothesize_mystery(mystery)
            self.mystery_hypotheses.append(hyp)

            print(f"    → Ipotesi: {hyp.hypothesized_cause[:65]}")
            print(f"      Scala: {hyp.hypothesized_scale} | "
                  f"Conf: {hyp.confidence:.2f}")
            print(f"      {hyp.reasoning[:70]}")
            print()

    # ─────────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────────

    def _final_report(self):
        total_links = sum(len(r.verified_links) for r in self.reports.values())
        total_items = len(self.classified)
        n_e = sum(1 for i in self.classified if i["nature"] == "effect")
        n_c = sum(1 for i in self.classified if i["nature"] == "cause")
        n_a = sum(1 for i in self.classified if i["nature"] == "ambiguous")

        explained_ids = set()
        for r in self.reports.values():
            for link in r.verified_links:
                explained_ids.add(link.effect["id"])
        n_mysteries = sum(1 for i in self.classified
                          if i["nature"] == "effect" and i["id"] not in explained_ids)

        genuine = sum(1 for v in self.cross_scale_verdicts if v.verdict == "genuine")
        false_c = sum(1 for v in self.cross_scale_verdicts if v.verdict == "false_correlation")

        print("\n" + "◆" * 70)
        print("  REPORT FINALE — FRACTAL TRIAD v4")
        print("◆" * 70)

        print(f"""
  ╔═══════════════════════════════════════════════════════════════╗
  ║          TRIADE FRATTALE v4 — UNLOCKED OBSERVERS             ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║                                                               ║
  ║  Dataset:                                          {total_items:>3}         ║
  ║  Effetti / Cause / Ambigui:              {n_e:>3} / {n_c:>3} / {n_a:>3}       ║
  ║                                                               ║
  ║  ── v3: Osservatori Bloccati ──                               ║
  ║  Link verificati (stesso zoom):                    {total_links:>3}         ║
  ║  Gap (livelli problematici):                       {len(self.gaps):>3}         ║
  ║  Misteri (effetti senza causa):                    {n_mysteries:>3}         ║
  ║                                                               ║
  ║  ── v4: Osservatore Sbloccato (LLM) ──                       ║
  ║  Candidati cross-scala valutati:         {len(self.cross_scale_verdicts):>3}         ║
  ║  → Genuini:                                        {genuine:>3}         ║
  ║  → False correlazioni:                             {false_c:>3}         ║
  ║  Misteri con ipotesi LLM:               {len(self.mystery_hypotheses):>3}         ║
  ║                                                               ║
  ╚═══════════════════════════════════════════════════════════════╝""")

        # Detail: genuine cross-scale links
        genuine_verdicts = [v for v in self.cross_scale_verdicts if v.verdict == "genuine"]
        if genuine_verdicts:
            print(f"\n  ─── LINK CROSS-SCALA VALIDATI DALL'LLM ───")
            for v in genuine_verdicts:
                print(f"    ✓ [{v.cause_zoom} → {v.effect_zoom}] "
                      f"conf={v.confidence:.2f}")
                print(f"      C: {v.cause_text[:65]}")
                print(f"      E: {v.effect_text[:65]}")
                print(f"      Ragionamento: {v.reasoning[:70]}")
                if v.intermediate_steps:
                    print(f"      Passaggi: {' → '.join(s[:40] for s in v.intermediate_steps[:3])}")

        # Detail: false correlations caught
        false_verdicts = [v for v in self.cross_scale_verdicts
                          if v.verdict == "false_correlation"]
        if false_verdicts:
            print(f"\n  ─── FALSE CORRELAZIONI IDENTIFICATE DALL'LLM ───")
            for v in false_verdicts:
                print(f"    ✗ [{v.cause_zoom} → {v.effect_zoom}] "
                      f"sim={v.semantic_similarity} ma: {v.reasoning[:60]}")

        # Detail: mystery hypotheses
        good_hyp = [h for h in self.mystery_hypotheses if h.confidence >= 0.5]
        if good_hyp:
            print(f"\n  ─── IPOTESI PIÙ PROMETTENTI SUI MISTERI ───")
            for h in sorted(good_hyp, key=lambda x: x.confidence, reverse=True):
                print(f"    ? [{h.effect_zoom}] {h.effect_text[:55]}")
                print(f"      → Ipotesi ({h.hypothesized_scale}): "
                      f"{h.hypothesized_cause[:55]}")

        print(f"""
  ─── NOTE METODOLOGICHE ───
  • v3: matching vettoriale same-scale (preciso, nessun falso positivo cross-scala)
  • v4: ragionamento LLM cross-scala (esplora, ma soggetto ai limiti del modello)
  • Il modello LLM ({OLLAMA_MODEL}) è un filtro di ragionamento, non un oracolo.
  • I verdetti "genuine" richiedono ulteriore validazione da esperti di dominio.
  • Le ipotesi sui misteri sono spunti per esplorazione, non conclusioni.""")

    # ─────────────────────────────────────────────
    # VISUAL REPORT
    # ─────────────────────────────────────────────

    def _generate_visual_report(self):
        """Extended visual dashboard with LLM results."""

        BG = '#0a0e1a'
        FG = '#e0e0e0'
        GOLD = '#d4a84b'
        CYAN = '#5bbce4'
        RED = '#e05555'
        GREEN = '#55e088'
        PURPLE = '#9b7fd4'
        GRID = '#1a2035'
        ORANGE = '#e09955'

        plt.rcParams.update({
            'figure.facecolor': BG, 'axes.facecolor': BG,
            'text.color': FG, 'axes.labelcolor': FG,
            'xtick.color': FG, 'ytick.color': FG,
            'font.family': 'sans-serif', 'font.size': 9,
        })

        fig = plt.figure(figsize=(18, 28))
        fig.suptitle('FRACTAL TRIAD v4 — Unlocked Observers Report',
                      fontsize=20, fontweight='bold', color=GOLD, y=0.99)
        fig.text(0.5, 0.98, f'Zoom Coherence + LLM Cross-Scale Reasoning ({OLLAMA_MODEL})',
                 ha='center', fontsize=11, color=FG, alpha=0.7)

        gs = GridSpec(5, 2, hspace=0.35, wspace=0.3,
                      left=0.08, right=0.92, top=0.97, bottom=0.02)

        levels_short = ['COS', 'PLA', 'SOC', 'ORG', 'CEL', 'MOL', 'ATO', 'SUB', 'FON']

        # ═══════════════════════════════════════════════
        # PANEL 1: Cause-Effect Distribution (from v3)
        # ═══════════════════════════════════════════════
        ax1 = fig.add_subplot(gs[0, 0])
        n_eff = [len(self.reports[l].effects) for l in ZOOM_LEVELS]
        n_cau = [len(self.reports[l].causes) for l in ZOOM_LEVELS]
        n_amb = [len(self.reports[l].ambiguous) for l in ZOOM_LEVELS]
        x = np.arange(len(ZOOM_LEVELS))
        w = 0.25

        ax1.bar(x - w, n_eff, w, color=CYAN, alpha=0.85, label='Effetti')
        ax1.bar(x, n_cau, w, color=GOLD, alpha=0.85, label='Cause')
        ax1.bar(x + w, n_amb, w, color=PURPLE, alpha=0.85, label='Ambigui')
        ax1.set_xticks(x)
        ax1.set_xticklabels(levels_short, fontsize=8)
        ax1.set_ylabel('Conteggio')
        ax1.set_title('Distribuzione Causa / Effetto per Livello',
                       color=GOLD, fontsize=12, pad=10)
        ax1.legend(fontsize=8, loc='upper right', framealpha=0.3)
        ax1.grid(axis='y', color=GRID, alpha=0.5)
        ax1.set_axisbelow(True)

        # ═══════════════════════════════════════════════
        # PANEL 2: Structural Asymmetry (from v3)
        # ═══════════════════════════════════════════════
        ax2 = fig.add_subplot(gs[0, 1])
        eff_per_level = {l: len(self.reports[l].effects) for l in ZOOM_LEVELS}
        cau_per_level = {l: len(self.reports[l].causes) for l in ZOOM_LEVELS}
        max_val = max(max(eff_per_level.values()), max(cau_per_level.values()), 1)

        for i, level in enumerate(ZOOM_LEVELS):
            y = i
            ew = eff_per_level[level] / max_val * 4
            cw = cau_per_level[level] / max_val * 4

            ax2.barh(y, -ew, 0.7, color=CYAN, alpha=0.8)
            if eff_per_level[level] > 0:
                ax2.text(-ew - 0.15, y, str(eff_per_level[level]),
                         ha='right', va='center', fontsize=9, color=CYAN, fontweight='bold')
            ax2.barh(y, cw, 0.7, color=GOLD, alpha=0.8)
            if cau_per_level[level] > 0:
                ax2.text(cw + 0.15, y, str(cau_per_level[level]),
                         ha='left', va='center', fontsize=9, color=GOLD, fontweight='bold')
            ax2.text(0, y, ZOOM_LEVELS[i].upper(), ha='center', va='center',
                     fontsize=8, fontweight='bold', color=FG,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=BG,
                               edgecolor=GRID, alpha=0.9))
            if level == "organismo":
                ax2.axhline(y + 0.5, color=RED, alpha=0.4, linestyle='--')

        ax2.set_yticks([])
        ax2.set_xlim(-5.5, 5.5)
        ax2.invert_yaxis()
        ax2.set_title('Asimmetria: Effetti ← → Cause',
                       color=GOLD, fontsize=12, pad=10)
        ax2.axvline(0, color=FG, alpha=0.15, linewidth=1)

        # ═══════════════════════════════════════════════
        # PANEL 3: v3 Verified Links
        # ═══════════════════════════════════════════════
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')

        all_links = []
        for level in ZOOM_LEVELS:
            for link in self.reports[level].verified_links:
                all_links.append(link)
        all_links.sort(key=lambda l: l.similarity, reverse=True)

        ax3.set_title(f'Link Verificati Same-Scale — v3 ({len(all_links)} totali)',
                       color=GOLD, fontsize=12, pad=10)

        if all_links:
            y_pos = 0.95
            row_h = 0.058
            for link in all_links:
                if y_pos < 0.02:
                    break
                sim_color = GREEN if link.similarity > 0.5 else (GOLD if link.similarity > 0.4 else RED)
                bar_w = link.similarity * 0.12
                ax3.barh(y_pos, bar_w, row_h * 0.6, left=0.0, color=sim_color, alpha=0.6)
                ax3.text(0.13, y_pos, f'{link.similarity:.2f}', va='center',
                         fontsize=8, color=sim_color, fontweight='bold')
                ax3.text(0.17, y_pos, f'[{link.zoom_level}]', va='center',
                         fontsize=8, color=PURPLE)
                cause_txt = link.cause['text'][:50] + ('…' if len(link.cause['text']) > 150 else '')
                effect_txt = link.effect['text'][:50] + ('…' if len(link.effect['text']) > 150 else '')
                ax3.text(0.28, y_pos + 0.012, f'C: {cause_txt}', va='center',
                         fontsize=7.5, color=GOLD, alpha=0.9)
                ax3.text(0.28, y_pos - 0.012, f'E: {effect_txt}', va='center',
                         fontsize=7.5, color=CYAN, alpha=0.9)
                y_pos -= row_h

        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

        # ═══════════════════════════════════════════════
        # PANEL 4: LLM Cross-Scale Verdicts (NEW)
        # ═══════════════════════════════════════════════
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        ax4.set_title(f'Validazione Cross-Scala — LLM ({len(self.cross_scale_verdicts)} candidati)',
                       color=ORANGE, fontsize=12, pad=10)

        if self.cross_scale_verdicts:
            y_pos = 0.95
            row_h = 0.065
            for v in self.cross_scale_verdicts:
                if y_pos < 0.02:
                    break
                vcolor = {
                    "genuine": GREEN, "false_correlation": RED,
                    "uncertain": GOLD, "error": PURPLE
                }.get(v.verdict, FG)
                vsym = {
                    "genuine": "✓", "false_correlation": "✗",
                    "uncertain": "?", "error": "⚠"
                }.get(v.verdict, "?")

                ax4.text(0.02, y_pos, vsym, va='center', fontsize=11,
                         color=vcolor, fontweight='bold')
                ax4.text(0.05, y_pos, f'[{v.cause_zoom}→{v.effect_zoom}]',
                         va='center', fontsize=8, color=PURPLE)
                ax4.text(0.18, y_pos,
                         f'sim={v.semantic_similarity:.2f} | conf={v.confidence:.2f}',
                         va='center', fontsize=8, color=vcolor)

                cause_txt = v.cause_text[:45] + ('…' if len(v.cause_text) > 45 else '')
                effect_txt = v.effect_text[:45] + ('…' if len(v.effect_text) > 45 else '')
                ax4.text(0.38, y_pos + 0.012, f'C: {cause_txt}', va='center',
                         fontsize=7, color=GOLD, alpha=0.9)
                ax4.text(0.38, y_pos - 0.012, f'E: {effect_txt}', va='center',
                         fontsize=7, color=CYAN, alpha=0.9)
                y_pos -= row_h
        else:
            ax4.text(0.5, 0.5, 'Nessun candidato cross-scala valutato',
                     ha='center', va='center', fontsize=11, color=FG, alpha=0.5)

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

        # ═══════════════════════════════════════════════
        # PANEL 5: LLM Mystery Hypotheses (NEW)
        # ═══════════════════════════════════════════════
        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis('off')

        ax5.set_title(f'Ipotesi sui Misteri — LLM ({len(self.mystery_hypotheses)} analizzati)',
                       color=ORANGE, fontsize=12, pad=10)

        if self.mystery_hypotheses:
            y_pos = 0.95
            row_h = 0.065
            for h in sorted(self.mystery_hypotheses,
                            key=lambda x: x.confidence, reverse=True):
                if y_pos < 0.02:
                    break
                conf_color = GREEN if h.confidence >= 0.7 else (GOLD if h.confidence >= 0.4 else RED)

                ax5.text(0.02, y_pos, '💡', va='center', fontsize=9)
                ax5.text(0.05, y_pos, f'[{h.effect_zoom}]', va='center',
                         fontsize=8, color=PURPLE)
                ax5.text(0.15, y_pos, f'conf={h.confidence:.2f}', va='center',
                         fontsize=8, color=conf_color)

                eff_txt = h.effect_text[:50] + ('…' if len(h.effect_text) > 50 else '')
                hyp_txt = h.hypothesized_cause[:50] + ('…' if len(h.hypothesized_cause) > 50 else '')
                ax5.text(0.28, y_pos + 0.012, f'Effetto: {eff_txt}', va='center',
                         fontsize=7, color=CYAN, alpha=0.9)
                ax5.text(0.28, y_pos - 0.012,
                         f'Ipotesi ({h.hypothesized_scale}): {hyp_txt}', va='center',
                         fontsize=7, color=ORANGE, alpha=0.9)
                y_pos -= row_h
        else:
            ax5.text(0.5, 0.5, 'Nessuna ipotesi generata',
                     ha='center', va='center', fontsize=11, color=FG, alpha=0.5)

        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)

        # ═══════════════════════════════════════════════
        # PANEL 6: Summary stats
        # ═══════════════════════════════════════════════
        ax6 = fig.add_subplot(gs[4, :])
        ax6.axis('off')

        total_links = sum(len(r.verified_links) for r in self.reports.values())
        genuine = sum(1 for v in self.cross_scale_verdicts if v.verdict == "genuine")
        false_c = sum(1 for v in self.cross_scale_verdicts if v.verdict == "false_correlation")
        good_hyp = sum(1 for h in self.mystery_hypotheses if h.confidence >= 0.5)

        summary_text = (
            f"RIEPILOGO v4\n\n"
            f"Dataset: {len(self.classified)} items\n"
            f"Link same-scale verificati (v3): {total_links}\n"
            f"Candidati cross-scala valutati (v4 LLM): {len(self.cross_scale_verdicts)}\n"
            f"  → Genuini: {genuine}  |  False correlazioni: {false_c}\n"
            f"Misteri analizzati (v4 LLM): {len(self.mystery_hypotheses)}\n"
            f"  → Ipotesi promettenti (conf≥0.5): {good_hyp}\n\n"
            f"Modello: {OLLAMA_MODEL} via Ollama"
        )

        ax6.text(0.5, 0.5, summary_text, ha='center', va='center',
                 fontsize=12, color=FG, fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=1', facecolor='#111827',
                           edgecolor=GOLD, alpha=0.9))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)

        # Save
        out_path = 'fractal_triad_v4_report.png'
        fig.savefig(out_path, dpi=150, facecolor=BG, edgecolor='none')
        plt.close(fig)
        print(f"\n  ★ Report visuale salvato in: {out_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    orchestrator = OrchestratorV4(KNOWLEDGE_DATASET)
    orchestrator.run()
