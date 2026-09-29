# DocuResearch — Evaluation Methodology

**Status:** Draft v1  
**Last updated:** 2026-09-21  
**Source:** PRFAQ §8, Architecture §9  
**Owner:** Product owner (curates dataset); Hermes assists with scaffolding

---

## 1. Purpose

This document defines what "good" means for each evaluation metric in the
DocuResearch system, and how each metric will be scored. It is the foundation
for validating architectural decisions and implementation quality throughout
the project.

The evaluation dataset (documents + questions + gold answers) provides the
test material; this methodology provides the scoring rules.

---

## 2. Metric Definitions

### 2.1 Retrieval Precision

**Definition:** The fraction of retrieved passages that are relevant to the query.

**Operational definition:**
- A passage is "relevant" if it contains information that helps answer the query,
  even if it is not the single best passage. A passage that mentions the query's
  topic but does not contain useful information is NOT relevant.
- Gold standard: the evaluation dataset records which passages are relevant for
  each question. The dataset curator marks the passage IDs that contain the
  answer or information needed to construct the answer.
- Formula: `precision = (number of retrieved passages that are in the gold
  relevant set) / (total number of retrieved passages)`
- If no passages are retrieved: precision is undefined (count as 0 or N/A
  depending on context).

**Scoring:**
- Binary pass/fail at the per-question level is not appropriate; precision is
  a continuous 0–1 value.
- Record the raw precision for each question; report mean/median across the
  dataset.

### 2.2 Retrieval Recall

**Definition:** The fraction of all relevant passages that were retrieved.

**Operational definition:**
- "All relevant passages" = the set of passage IDs marked as relevant in the
  gold dataset for that question.
- Formula: `recall = (number of retrieved passages that are in the gold
  relevant set) / (total number of gold relevant passages)`
- If there are no relevant passages (should not happen for valid questions):
  recall is 1.0 (vacuously true) — but this case should not arise in the dataset.

**Scoring:**
- Continuous 0–1 value per question.
- Record raw recall per question; report mean/median.

### 2.3 Citation Correctness

**Definition:** The fraction of citations in generated answers that point to
passages that actually support the claim made.

**Operational definition:**
- A citation is "correct" if the cited passage contains the claim or information
  that the answer attributes to it. The passage must *support* the claim — it is
  not sufficient for the passage to be topically related.
- A citation that points to a passage ID that does not exist in the content store
  is automatically incorrect.
- A citation whose excerpt (the passage text) does not match the content store's
  record for that passage ID is automatically incorrect.

**Scoring:**
- Per-citation binary: correct / incorrect.
- Per-answer: `citation_correctness = (number of correct citations) /
  (total number of citations in the answer)`
- If the answer has no citations: score is N/A (not applicable).

**Gold standard:**
- Each question in the dataset has a set of "gold passage IDs" that support the
  gold answer. A generated citation is correct if it references one of these
  passages AND the passage text supports the claim as stated in the answer.
- For multi-passage questions, all gold passages should be cited if the answer
  draws on all of them.

### 2.4 Answer Faithfulness

**Definition:** The fraction of generated answers that do not introduce claims
unsupported by the retrieved passages.

**Operational definition:**
- An answer is "faithful" if every factual claim in the answer is supported by
  at least one of the retrieved passages (not just topically related — actually
  stated or directly implied by the text).
- An answer is "unfaithful" if it contains at least one claim that cannot be
  traced to any retrieved passage.
- "Supported by retrieved passages" does NOT mean "in the documents overall" —
  it means in the passages the retrieval layer returned. If the retrieval layer
  failed to retrieve a relevant passage, and the generation layer used its prior
  knowledge to fill the gap, that is an unfaithful answer (and also a retrieval
  failure).

**Scoring:**
- Binary per answer: faithful / unfaithful.
- Faithfulness rate = (number of faithful answers) / (total number of answers).

**How to evaluate:**
- Compare each claim in the generated answer against the retrieved passages.
- If the answer makes a claim that is not present in any retrieved passage,
  mark as unfaithful.
- The evaluation runner should flag specific claims that are unsupported,
  not just produce a binary score.

### 2.5 Unsupported-Answer / Hallucination Rate

**Definition:** The rate at which the system produces an answer containing claims
not grounded in the retrieved passages.

**Operational definition:**
- This is the complement of faithfulness: `hallucination_rate = 1 - faithfulness_rate`.
- A "hallucination" here means any claim in the answer that is not supported by
  the retrieved passages. This includes:
  - Fabricated facts (claims with no basis in any document).
  - Claims attributed to a document when they are not in that document.
  - Inferences presented as facts without being labeled as inferences.

**Scoring:**
- Rate per dataset: fraction of answers that contain at least one hallucinated claim.
- Also track: fraction of individual claims that are hallucinated (for finer-grained analysis).

### 2.6 Abstention Accuracy

**Definition:** The rate at which the system correctly abstains when evidence is
insufficient, and does not abstain when evidence is sufficient.

**Operational definition:**
- "Correct abstention": the system says it cannot answer (or that evidence is
  insufficient) AND the gold label for the question is "insufficient evidence."
- "Correct non-abstention": the system produces an answer AND the gold label is
  one of: single-source, multi-passage, partial evidence, or conflicting evidence.
- "Incorrect abstention": the system abstains but the gold label indicates
  sufficient evidence exists.
- "Incorrect non-abstention": the system produces an answer but the gold label
  is "insufficient evidence."

**Scoring:**
- Binary per question: correct / incorrect abstention decision.
- `abstention_accuracy = (correct abstentions + correct non-abstentions) /
  (total questions)`
- Also report separately:
  - abstention sensitivity: of the insufficient-evidence questions, how many
    did the system correctly abstain on?
  - abstention specificity: of the sufficient-evidence questions, how many
    did the system correctly NOT abstain on?

### 2.7 Handling of Conflicting Evidence

**Definition:** Whether the system surfaces conflicts when sources disagree,
and does not silently choose one side.

**Operational definition:**
- For questions labeled "conflicting evidence" in the dataset:
  - "Correct handling": the answer names both (or all) conflicting sources and
    states what each says. The answer does not silently pick one side.
  - "Incorrect handling": the answer picks one side without acknowledging the
    conflict, or ignores one source entirely.
- For questions not labeled as conflicting: the system should not fabricate a
  conflict (that would be a hallucination).

**Scoring:**
- Binary per conflicting-evidence question: correctly surfaced / not correctly surfaced.
- Conflict handling rate = (number of correctly handled conflicting-evidence
  questions) / (total conflicting-evidence questions in the dataset).

**What "surfacing the conflict" means in an evaluable way:**
- The answer text must mention both sources by name (or by document identifier).
- The answer must state what each source claims about the conflicting point.
- The evidence quality narration must use the "conflicting" category.

### 2.8 Response Latency

**Definition:** End-to-end time from query submission to answer delivery.

**Operational definition:**
- Measured from the moment the query is submitted to the moment the complete
  answer (including citations) is available.
- For component-level evaluation: measure retrieval latency and generation
  latency separately.
- Record in seconds, with millisecond precision.

**Scoring:**
- Report mean, median, p95, and max latency across the dataset.
- No pass/fail threshold at this stage — establish baseline, set targets later.

---

## 3. Question Types

The dataset must include the following question types, as specified in the
PRFAQ §8 and Architecture §9:

| Type | Description | Gold data required |
|------|-------------|-------------------|
| **Single-source** | One passage directly supports the answer. | One gold passage ID + gold answer text. |
| **Multi-passage synthesis** | The answer draws on more than one passage. | Two or more gold passage IDs + gold answer text. |
| **Partial evidence** | The documents support part of the answer but not all. | Gold passage IDs for the supported part + gold answer that distinguishes supported from unsupported. |
| **Insufficient evidence** | The documents do not contain enough to answer definitively. | No gold passage supports a definitive answer; record the closest relevant passage if one exists. Gold answer states what is and is not supported. |
| **Conflicting evidence** | Two or more sources disagree on a point. | Gold passage IDs for each conflicting claim + gold answer that surfaces the conflict, naming sources and their claims. |

Each question in the dataset must be labeled with its type.

---

## 4. What Constitutes a "Relevant Passage" (for retrieval precision/recall)

A passage is **relevant** to a query if:

1. The passage contains information that is needed to answer the query (not
   just topically related).
2. The passage is part of the minimal set of passages required to construct
   a complete answer.
3. For single-source questions: the one passage that contains the answer.
4. For multi-passage questions: all passages that together provide the
   information needed for the complete answer.

A passage is **not relevant** if:

1. It mentions a keyword from the query but does not contain useful information
   for answering.
2. It is topically adjacent but does not contribute to the answer.
3. It contains the same keywords but in a different context.

The gold dataset must record, for each question, the exact set of passage IDs
that are relevant. This is the basis for computing precision and recall.

---

## 5. What Constitutes a "Correct Citation"

A citation is **correct** if:

1. The cited passage ID exists in the content store.
2. The passage text (the excerpt) matches the content store's record for that
   passage ID exactly.
3. The passage contains the claim or information that the answer attributes to
   it — the passage *supports* the claim.

A citation is **incorrect** if:

1. The cited passage ID does not exist.
2. The excerpt does not match the content store.
3. The passage is topically related but does not actually support the claim
   as stated in the answer.

---

## 6. Partial-Evidence Labeling and Evaluation

For questions labeled "partial evidence":

- The gold answer must clearly distinguish:
  - What the documents **do** support (with passage references).
  - What the documents **do not** support (and what would be needed to support it).
- The system's evidence-quality narration is evaluated against the label:
  - **Correct:** The system says the evidence is partial, identifies what is
    supported and what is not, and does not present unsupported inferences as
    facts.
  - **Incorrect:** The system presents the answer as fully supported when it is
    only partially supported, or presents unsupported inferences as facts.

---

## 7. Conflicting-Evidence Labeling and Evaluation

For questions labeled "conflicting evidence":

- The gold answer must identify:
  - The two (or more) conflicting sources.
  - What each source claims.
  - The passages that contain each claim.
- "Correct handling" means:
  - The answer names both sources.
  - The answer states what each source says.
  - The answer does not pick a winner.
  - The evidence quality narration uses the "conflicting" category.
- "Incorrect handling" means:
  - The answer picks one side without acknowledging the other.
  - The answer ignores one of the conflicting sources.
  - The answer fabricates a conflict that isn't in the data.

---

## 8. Evaluation Runner Capabilities

The evaluation runner (story 0.4) must be able to:

1. Load questions from the dataset.
2. Submit each question to the system (or to individual components).
3. Collect the system's answer, citations, evidence quality, and abstention flag.
4. Compare against gold answers and gold passage IDs.
5. Compute: retrieval precision, retrieval recall, citation correctness,
   answer faithfulness (binary), abstention classification.
6. Record response latency for each question.
7. Produce a readable results summary.

The runner must be structured so that new questions and new metrics can be
added without rewriting the core logic.

---

## 9. Acceptance Criteria Mapping

This methodology document addresses all 7 acceptance criteria from Story 0.1:

| AC | Addressed in section |
|----|---------------------|
| Define each metric in operational terms | §2.1–§2.8 |
| Each metric has a defined scoring approach | §2 (each subsection) |
| What constitutes a "relevant passage" | §4 |
| What constitutes a "correct citation" | §5 |
| How partial-evidence cases are labeled and evaluated | §6 |
| How conflicting-evidence cases are labeled and evaluated | §7 |
| Question types that must be represented | §3 |
| Methodology reviewed and accepted before epics 2–6 | This document (living artifact) |
