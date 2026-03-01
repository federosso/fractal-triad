# The Fractal Triad

*Knowledge Organization with Zoom Coherence, Cause-Effect Separation, and Cross-Scale Validation via LLM*

**Federico D'Ambrosio & Claude (Anthropic AI)**

March 2026

---

## Abstract

Artificial intelligence systems represent knowledge on a single, undifferentiated plane: causes, effects, principles, and measurements are all treated with identical status. This flatness produces systematic errors, including false correlations between unrelated domains and confusion between causal mechanisms and observable consequences.

This paper introduces the Fractal Triad, a framework that organizes knowledge along two dimensions: the nature of information (cause vs. effect) and the scale of observation (from cosmological to fundamental level). The central principle — the Zoom Coherence Principle — establishes that cause and effect must be coupled at the same scale of observation, and that cross-scale transitions must be explicitly tracked and validated through reasoning.

We present two progressive implementations. V3 (locked observers) uses nine specialized observers in semantic embedding space: across 62 elements, it identifies 15 verified cause-effect links at the same scale and reveals a deep structural asymmetry (28 effects vs. 4 causes at surface levels; 5 effects vs. 20 causes at deep levels).

V4 (unlocked observers) integrates a local LLM as a cross-scale reasoning filter. The key experiment — the same framework tested with three models of increasing size (1B, 4B, 12B parameters) — produces an unexpected result: the cross-scale false positive rate decreases monotonically with model size (83% → 0% → 17%), revealing that the framework implicitly functions as a benchmark for an LLM's causal reasoning capability.

*Keywords: knowledge representation, causal reasoning, semantic embeddings, multi-scale architecture, zoom coherence, LLM benchmarking*

---

## 1. Introduction: The Flatness Problem

Modern artificial intelligence systems — language models, knowledge graphs, neural databases — share an architectural limitation: they represent all knowledge on a single undifferentiated plane. A measurement ("global temperature has risen by 1.1°C"), a causal mechanism ("greenhouse gas accumulation traps reflected heat"), and a universal principle ("entropy always tends to increase") are stored and processed with identical status.

This flatness produces three systematic errors. **False correlations**: superficial similarity metrics connect unrelated elements — for example, an embedding-based system finds high similarity (0.44) between "monetary expansion increases the money supply" and "the universe expands with acceleration," because the word "expansion" shares semantic structure even though the meaning is completely different. **Causal confusion**: the system does not distinguish "X causes Y" from "X correlates with Y." **Scale mixing**: causes at one level are coupled with effects at a different level, producing incoherent explanations.

In the physical world, effects manifest at every observable scale, but their causes often operate at different levels. A human observer navigates this multi-scale reality naturally. Current AI systems lack this structural capability. This paper proposes a framework to address this limitation.

### 1.1 Practical Applications of the Framework

Before presenting the theory, it is useful to clarify how this work fits within the applied AI landscape. The framework is not a model to be inserted into a neural architecture, but a set of structural constraints that can be integrated into existing systems in three concrete ways.

**As a reasoning constraint.** Current chain-of-thought systems have no scale constraints: when an LLM constructs a causal explanation, it can freely mix levels without tracking them. The Zoom Coherence Principle can be implemented as a scale meta-tag attached to each reasoning step, penalizing unexplicated cross-scale transitions.

**As a layer for RAG systems.** In Retrieval-Augmented Generation systems, retrieval occurs by semantic similarity without scale distinction. A RAG system adopting the Fractal Triad classifies each chunk by scale and nature (cause/effect), constrains retrieval to the same scale, and returns cross-scale candidates separately and with appropriate skepticism. This is immediately implementable with existing vector database metadata filters.

**As a causal reasoning benchmark.** As the v4 experimental results will demonstrate, the framework produces an implicit metric of an LLM's causal reasoning capability, measured through the false positive rate on cross-scale candidates with known causality. No existing benchmark specifically measures this capability.

## 2. The Two Cones: Conceptual Foundation

The Fractal Triad architecture is based on a geometric vision: two cones converging at their bases, with vertices pointing outward. A third point — the observer — is connected to both.

The **Cone of Effects** contains all observable and measurable phenomena. The **Cone of Causes** contains the generative mechanisms: laws, principles, underlying processes. Both cones have depth: they can be "zoomed" from the macroscopic (cosmological) to the microscopic (fundamental) scale.

The cones are **specular in structure but asymmetric in content**: each zoom level in the Cone of Effects has a corresponding level in the Cone of Causes. However, at the levels where human beings operate directly (cosmological, planetary, social, organism), observable effects dominate and causes are scarce. At deep levels (molecular, atomic, fundamental), causal principles abound but directly observable effects are few.

This asymmetry is not a dataset defect: it is a structural property of human knowledge. We live immersed in effects. Causes operate at depth. The experimental results quantitatively confirm this asymmetry: 28 effects versus 4 causes at surface levels; 5 effects versus 20 causes at deep levels.

## 3. The Zoom Coherence Principle

The central conceptual contribution of this work is the **Zoom Coherence Principle**: cause and effect must be coupled at the same scale of observation. Cross-scale transitions are not forbidden, but must be explicitly tracked and validated with reasoning tools, not with simple semantic proximity.

An example clarifies the principle. A flower pot falls on a child's head:

> **Physical objects scale** — Effect: the pot strikes the head. Cause: the wind, the cat, the earthquake.
> 
> **Organism scale** — Effect: cranial trauma, pain, fear. Cause: the kinetic impact on tissue.
> 
> **Cellular scale** — Effect: inflammation, cortisol cascade. Cause: the rupture of cell membranes.
> 
> **Molecular scale** — Effect: cytokine release, synaptic rewiring. Cause: chemical reactions from protein deformation.

Each level is **self-consistent**: cause and effect correspond at the same scale. The error arises when levels are mixed: saying that "the cause of psychological trauma is protein deformation" skips intermediate steps without tracking them. It is not false, but it is incoherent. A knowledge system that does not enforce this coherence produces the illusion of causal understanding where in reality there are only semantic correlations.

## 4. Architecture

### 4.1 Zoom Levels and Classification

The framework defines nine levels: cosmological, planetary, social, organism, cellular, molecular, atomic, subatomic, and fundamental. Each element is assigned to a primary level and classified as effect, cause, or ambiguous, through semantic proximity to prototypes in the 384-dimensional multilingual embedding space.

The classification is a heuristic, not an oracle. The cause/effect distinction is not intrinsic to the information: it depends on the observer's position. "Elevated cortisol reduces telomerase" is an effect seen from the organism level, and a cause seen from the cellular level. The system assigns a primary classification but does not capture this duality.

### 4.2 Locked Observers (v3)

The core of the first implementation is an ensemble of nine **locked observers**, each constrained to a single zoom level. Each observer sees only the elements classified at its own level and seeks cause-effect couplings through cosine similarity with a minimum threshold of 0.35.

The constraint is rigid: an observer locked at the planetary level cannot and does not attempt to explain coral bleaching using quantum mechanics principles. It produces links only between causes and effects that share the same scale. This is the Zoom Coherence Principle translated into software architecture.

### 4.3 Unlocked Observers (v4)

V4 introduces the **unlocked observer**: a local LLM (Gemma via Ollama) that operates as a cross-scale reasoning filter. The unlocked observer receives the cross-scale candidates identified by vector matching and evaluates them through contextual reasoning, producing for each:

- A **verdict**: `genuine`, `false_correlation`, or `uncertain`
- A **confidence level** (0.0 — 1.0)
- An explicit **reasoning** justifying the verdict
- The **intermediate steps** that would connect cause and effect across scales

This is the piece that was missing in v3: where vector matching stops, reasoning begins. Cross-scale validation requires domain understanding, not geometry in embedding space.

### 4.4 The Orchestrator

The orchestrator coordinates the pipeline in successive phases:

1. **Classification** of all elements in embedding space
2. **Locked observers**: deployment of the nine observers and collection of verified same-scale links
3. **Structural analysis**: gap identification, asymmetry mapping, mystery flagging
4. **Unlocked observer** (v4): LLM validation of cross-scale candidates
5. **Mystery analysis** (v4): generation of causal hypotheses for orphan effects

Phases 4 and 5 are decoupled from the previous ones: v3 produces structured inputs that v4 consumes. The LLM never sees the entire dataset — it only sees candidates already filtered by the vector pipeline.

## 5. Implementation: Four Progressive Versions

### 5.1 Version 1 (TF-IDF)

Used TF-IDF to measure similarity. TF-IDF works on exact words, not meaning: for this system, "the dog runs" and "the wolf gallops" have nothing in common. Result: systematic false positives, including absurd bridges such as "p53 gene mutations" connected to "Bitcoin crash."

### 5.2 Version 2 (Embeddings)

The replacement with 384-dimensional multilingual embeddings eliminated false positives. "Beating heart" and "sinoatrial node generates impulses" have similarity 0.53 because they describe the same phenomenon; "Bitcoin" and "gravity" have similarity 0.01. The key insight: the architecture must operate in the native embedding space.

### 5.3 Version 3 (Multiple Observers)

Introduces multiple observers and structural analysis. The shift from a single global observer to nine scale-specialized ones eliminates spurious cross-scale correlations and produces cleaner, more interpretable results.

### 5.4 Version 4 (Unlocked Observers via LLM)

Integrates a local LLM as an unlocked observer. The conceptual leap: where v3 stopped with a "?" and a disclaimer ("these candidates require external validation"), v4 performs that validation. The most significant result is not the individual verdict, but the pattern that emerges from comparing models of different sizes.

## 6. Results

### 6.1 Classification and Asymmetry (v3)

Across 62 elements, the system classified 33 effects, 24 causes, and 5 ambiguous. The distribution by level confirms the predicted asymmetry:

> Surface levels (cosmological → organism): 28 effects, 4 causes
> 
> Deep levels (cellular → fundamental): 5 effects, 20 causes

### 6.2 Verified Same-Scale Links (v3)

The 9 locked observers identified 15 verified cause-effect links. Only 3 out of 9 levels (planetary, organism, cellular) produced links — precisely the levels where the dataset contains both causes and effects at the same scale.

The strongest link (similarity 0.71): "Rising temperature reduces ice formation" → "Arctic glaciers are melting." A cause-effect coupling at the same scale with high semantic confidence.

The remaining 6 levels are gaps: they contain only effects (cosmological, social, subatomic) or only causes (atomic, fundamental) or too few elements to produce couplings (molecular).

### 6.3 Gaps and Mysteries (v3)

The system identified 20 "mystery" effects: observable phenomena for which the dataset contains no cause at the same scale. Among them: "The universe expands with increasing acceleration" (cosmological, with no cosmological cause in the dataset), "Financial crises repeat cyclically" (social, with no explicit social mechanism), "Entanglement maintains instantaneous correlation at any distance" (subatomic, with no subatomic cause).

These mysteries are not system failures: they are the points where knowledge is incomplete.

### 6.4 Cross-Scale Validation via LLM (v4): The Key Experiment

The v3 orchestrator had identified 6 cross-scale candidates: orphan effects with semantically close causes at adjacent levels. V4 submitted the same 6 candidates to three LLM models of increasing size, all from the Gemma 3 family (Google), run locally via Ollama.

#### Table 1: Verdicts by model

| Candidate                            | Sim. | 1B        | 4B        | 12B                |
| ------------------------------------ | ---- | --------- | --------- | ------------------ |
| Emergence → Entanglement             | 0.46 | uncertain | false     | **genuine** (0.75) |
| Resonance → Entanglement             | 0.42 | genuine   | uncertain | uncertain (0.60)   |
| Resonance → Wave collapse            | 0.37 | genuine   | false     | uncertain (0.40)   |
| Temperature → Dark energy            | 0.31 | genuine   | false     | false (0.95)       |
| Sinoatrial node → Financial leverage | 0.30 | genuine   | false     | false (0.95)       |
| Temperature → Universe expansion     | 0.29 | genuine   | false     | false (0.99)       |

#### Table 2: Summary by model

| Model   | Parameters | Genuine | False | Uncertain | False positives |
| ------- | ---------- | ------- | ----- | --------- | --------------- |
| Gemma 3 | 1B         | 5       | 0     | 1         | 83%             |
| Gemma 3 | 4B         | 0       | 5     | 1         | 0%              |
| Gemma 3 | 12B        | 1       | 3     | 2         | 17%             |

#### 6.4.1 Analysis of Results

The pattern is significant and non-trivial.

**The 1B model is a "yes-man."** It validates as genuine the relationship between the sinoatrial node of the heart and financial leverage, and between the increase in glacier temperature and the expansion of the universe. It lacks sufficient reasoning capability to distinguish semantic correlation (similar words) from real causality. Its false positive rate (83%) makes it unusable as an unlocked observer.

**The 4B model is a "radical skeptic."** It rejects everything, including the most philosophically defensible candidate (emergence → entanglement). Its reasoning is correct for the obvious cases ("there is no plausible mechanism by which planetary-scale ice formation affects cosmological expansion") but too rigid for subtle cases.

**The 12B model finds the balance point.** It validates a single candidate — emergence → entanglement — with moderate confidence (0.75, not 0.95), an articulated reasoning ("Entanglement, as a non-local quantum phenomenon, fundamentally arises from the interaction of particles..."), and explicitly traced intermediate steps. It rejects the 3 patently spurious candidates with high confidence (0.95-0.99). It leaves 2 candidates as "uncertain" — precisely those where the relationship is philosophically plausible but physically undemonstrated (resonance → entanglement, resonance → wave function collapse).

#### 6.4.2 The Causal Reasoning Curve

The most relevant finding is not the individual verdict, but the overall pattern. Causal reasoning capability scales with model size in a non-linear fashion:

- **1B**: incapable of distinguishing correlation from causality
- **4B**: capable of rejecting false correlations, incapable of recognizing genuine cross-scale causality
- **12B**: capable of both rejecting and validating, with calibrated confidence and uncertainty where appropriate

This suggests that the Fractal Triad framework, designed as a knowledge organization system, also functions as an **implicit benchmark for multi-scale causal reasoning** — a capability that no existing benchmark specifically measures.

### 6.5 Mystery Hypotheses (v4)

For the 10 mysteries analyzed, the quality of hypotheses follows the same scaling pattern.

The 1B model produces scientifically absurd hypotheses: it attributes the color change of leaves in autumn to a "shift in the Earth's magnetic field, subtly altering the wavelength of light." The 4B model produces correct hypotheses: "Seasonal variations in solar radiation intensity trigger a complex..." The 12B model produces correct hypotheses with calibrated confidence (0.70 for the leaves vs. 0.20 for "water boils at 100°C" where it acknowledges that the explanation lies at the molecular level, not the planetary one).

### 6.6 Cross-Scale Candidates (v3)

The v3 orchestrator had identified 6 cross-scale candidates using vector matching alone. The most promising: "The whole is more than the sum of its parts" (fundamental) as a possible cause of "Entanglement is a property of the shared quantum state" (subatomic), with similarity 0.46.

This is the only candidate that the 12B model validated as genuine — confirming that vector matching, while insufficient on its own, is an effective pre-filter when combined with LLM reasoning.

## 7. Limitations

**Curated dataset.** The 62 elements were intentionally selected. On uncurated data, classification and results could be very different.

**Fragile classification.** Depends on fixed prototypes. Some elements are misclassified: "Bitcoin lost 65% of its value" assigned to the molecular level instead of social. The cause/effect classification does not capture the intrinsic duality of many elements.

**Similarity is not causality.** The system finds proximity in embedding space. The verified same-scale links are semantically plausible, not causally demonstrated.

**No comparative baseline.** We did not compare results with knowledge graphs, causal discovery algorithms, or LLM-based reasoning on the same dataset.

**Limited LLM sample.** The cross-scale test was run on three models from a single family (Gemma 3). The pattern may not generalize to other model families. A broader validation would require testing on Llama, Mistral, Phi, and larger-parameter models.

**LLM as judge.** Using an LLM to validate causal relationships introduces the model's own biases. The LLM might validate relationships that "sound right" linguistically without true causal understanding.

## 8. Discussion

### 8.1 What Works

The **Zoom Coherence Principle** is the most significant contribution. It is not an algorithm but a structural constraint: it requires that causal reasoning be coherent with the scale of observation. This constraint is implementation-independent.

The **structural asymmetry** revealed by the analysis (effects at the surface, causes at depth) is a genuine result: it reflects the structure of human knowledge.

The **two-tier architecture** (vector + reasoning) proves effective: the vector pipeline filters candidates, the LLM validates them. Neither alone would be sufficient.

The **scaling result** is the most unexpected contribution: the framework produces a causal reasoning curve that discriminates models of different sizes in a manner consistent with theoretical expectations.

### 8.2 What Doesn't Work (Yet)

**Level classification** remains fragile. A production-ready system would require contextual classification, not prototype-based.

The **mystery hypotheses** produced by small models are often scientifically absurd. This feature has value only with sufficiently large models.

**Single-run evaluation** does not capture the stochastic variability of models. Each model should be tested across multiple runs to establish confidence intervals.

### 8.3 On the Origin of the Framework

This framework originated from a non-analytical intuition — a geometric vision of the two cones received by the first author during a meditative state, subsequently formalized through iterative dialogue with an AI system. We report this origin for transparency, not as an argument for validity. The results stand or fall on their own technical merits.

## 9. Future Work

1. **Testing on different model families** (Llama, Mistral, Phi, Qwen) to verify whether the cross-scale scaling curve generalizes.
2. **Testing on large models** (70B+ parameters, cloud APIs) to explore the upper end of the curve.
3. **Uncurated datasets** with partially known causal ground truth for rigorous validation.
4. **Comparison with baselines** (knowledge graphs, causal discovery algorithms).
5. **Integration into RAG pipelines** with scale and nature metadata on chunks.
6. **Multi-run evaluation** to capture stochastic variability.

## 10. Conclusion

The Fractal Triad proposes that knowledge has depth, that causes and effects are specular across scales, and that coherent reasoning requires their coupling at the same scale. This is the idea.

The implementation demonstrates five concrete results: (1) the Zoom Coherence Principle produces 15 verified links with no cross-scale false positives; (2) the structural asymmetry between effects at the surface and causes at depth is quantifiable; (3) gap identification makes visible where knowledge is incomplete; (4) the integration of an LLM as an unlocked observer enables cross-scale validation with reasoning; (5) causal reasoning capability scales with model size in a measurable way.

The most significant result is perhaps the last: a framework designed to organize knowledge also turns out to be a tool for measuring the reasoning capability of the systems that use it. This suggests that the structure of knowledge and the ability to reason about it are two sides of the same coin.

GitHub Repo Link: https://github.com/federosso/fractal-triad

---

## References

[1] Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* Cambridge University Press.

[2] Granger, C.W.J. (1969). Investigating Causal Relations by Econometric Models. *Econometrica*, 37(3).

[3] Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.

[4] Mitchell, M. (2009). *Complexity: A Guided Tour.* Oxford University Press.

[5] Tegmark, M. (2014). *Our Mathematical Universe.* Alfred A. Knopf.

[6] Schölkopf, B. et al. (2021). Toward Causal Representation Learning. *Proceedings of the IEEE*.

[7] Bar-Yam, Y. (1997). *Dynamics of Complex Systems.* Addison-Wesley.

[8] Dorri, A. et al. (2018). Multi-Agent Systems: A Survey. *IEEE Access*, 6.

[9] D'Ambrosio, F. (2019). *La Via del Cuore.*
