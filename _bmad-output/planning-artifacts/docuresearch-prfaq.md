# DocuResearch — PRFAQ

**Document owner:** Product  
**Status:** Draft v1 (finalized decisions incorporated)  
**Last updated:** 2026-09-21  
**Codename:** DocuResearch  

---

## 1. Problem Statement

Technical professionals — engineers, architects, consultants, compliance reviewers, and similar roles — routinely work with complex documents: technical specifications, standards, manuals, regulatory guidance, and long-form research. Their job is not to read these documents cover to cover; it is to find a specific piece of information quickly and be able to prove exactly where it came from.

Today, that job is harder than it should be:

- **Search is blunt.** Ctrl+F gets you to a keyword, not to the answer. The keyword might appear in five places, only one of which is relevant, and the user still has to read each one.
- **Verification is manual.** When an answer matters — for a design decision, a compliance claim, a safety assessment — the user must trace it back to the source themselves. There is no link between the answer they got and the passage that supports it.
- **Multi-document work multiplies the problem.** When the question spans several documents, the user must search each one, reconcile results by hand, and remember which document said what.
- **AI summarization tools solve a different problem.** They produce a synthesized answer, but they do not reliably tell the user where each claim came from, whether the evidence was complete, or whether two sources disagree. For a verification-oriented user, that missing traceability is a dealbreaker, not a nice-to-have.

The gap is not "better search" or "better summarization." It is a tool built around the actual job: **find the specific information, and show your work.**

---

## 2. Proposed Solution

DocuResearch is an AI research assistant for technical documents. A user uploads one or more documents (starting with text-based PDFs, Markdown, and plain text), asks a question, and receives an answer that is:

- **Grounded in the source material**, not generated from the model's prior knowledge.
- **Cited to the specific document and, where possible, the page or section** where the supporting passage appears.
- **Interactive**: each citation is a link the user can follow to inspect the actual passage from the original document, not just a page number in a footnote.
- **Honest about evidence quality**: the system communicates whether the evidence is sufficient, partial, or conflicting — in plain language, not a numerical confidence score.

The product does not try to be a general-purpose chatbot or a document summarizer. Its default posture is: *tell me what the documents say, tell me where they say it, and tell me when you are guessing.*

### What "interactive citation" means in practice

When the answer references a source, the user sees the document name and, where available, a page or section identifier. Clicking or expanding that citation shows the exact passage from the original document — the same words the system used to construct the answer. This is the core trust mechanism: the user is never asked to take the system's word for what a document contains.

### Evidence quality communication

The system narrates evidence quality naturally:

- **Sufficient evidence**: "The documents support this directly. [citation(s)]."
- **Partial evidence**: "The documents support part of this. [what is supported] appears in [source]; the rest is a reasonable inference from that passage, but is not stated explicitly."
- **Insufficient evidence**: "The documents do not contain enough information to answer this definitively. [closest relevant passage, if any]."
- **Conflicting evidence**: "The documents disagree on this point. [Source A] says X; [Source B] says Y."

The system does not fabricate an answer to avoid saying "I don't know." Abstention is a feature, not a failure.

### Conversational follow-up

After an initial answer, the user can ask follow-up questions in context — refining the scope, asking for clarification on a specific passage, or narrowing to a particular document. The conversation stays grounded in the uploaded documents; the system does not drift into general knowledge.

### Multi-document and scoped search

Users can search across all uploaded documents or restrict a query to a specific document. This matters when a user knows the answer is in one document and does not want noise from the others, or when comparing what different documents say about the same question.

---

## 3. Why Now

Three trends converge:

1. **AI tools are everywhere, but trust is the bottleneck.** Technical professionals are asked to use AI assistants, but for verification-heavy work, an ungrounded answer is worse than no answer. A tool that makes its evidence inspectable directly addresses the trust gap.

2. **The documents are not going away.** Specifications, standards, manuals, and regulatory guidance are still primary-source materials that professionals must engage with. The problem is not that the documents exist; it is that engaging with them efficiently is hard.

3. **The alternative is worse than it looks.** Without a tool like this, the user either spends significant time reading and searching manually, or relies on memory and informal notes — both of which degrade as document complexity and volume increase.

---

## 4. Target Users

### Primary MVP user

**Technical professionals** — people whose job involves finding and verifying specific information in complex documents. This includes engineers reviewing specifications, consultants checking regulatory guidance, architects cross-referencing standards, and similar roles. The common thread is that an unverified answer is not an acceptable answer for their work.

### Secondary / future users (out of MVP scope)

- Academic researchers
- Students
- Financial or business analysts
- General professionals doing document research

These users are not the MVP focus, but the product should not actively exclude them. The prioritization is in the design attention and evaluation: the primary user's verification needs drive the feature set.

---

## 5. Core Job to Be Done

> *Find specific information in complex documents quickly, and know exactly where it came from so I can verify it.*

This is a single job with two inseparable parts: retrieval speed and source verification. A tool that does one without the other fails the job. The product is not "an AI that answers questions about your documents" — it is "a tool that helps me find and verify what my documents say."

---

## 6. Functional Scope

### MVP — included

- **Document upload and ingestion** for text-based PDF, Markdown, and plain text (TXT).
- **Single-document and multi-document support**: users can upload multiple documents and search across all of them or within one.
- **Query interface**: a question or search term, with the option to restrict to a specific document.
- **Grounded answers with citations**: every answer identifies the source document and, where possible, the page or section. Citations are interactive — the user can inspect the supporting passage.
- **Conversational follow-up**: the user can ask follow-up questions in context.
- **Evidence-quality communication**: the system says, in plain language, whether the evidence is sufficient, partial, or insufficient, and surfaces conflicting information when sources disagree.
- **No fabricated answers**: when evidence is insufficient, the system abstains or states what is and is not supported, rather than generating a plausible-sounding answer.

### MVP — explicitly deferred

- **DOCX support**: later extension.
- **Scanned PDFs and OCR**: later extension. MVP covers text-based PDFs only.
- **Dedicated contradiction-detection feature**: the system surfaces conflicts when they arise naturally from the evidence, but a separate contradiction-detection mode or report is not an MVP requirement.
- **Authentication, billing, multi-tenancy, enterprise administration**: out of MVP scope (single-user, local-first).
- **Numerical confidence scores**: out of scope. Evidence quality is communicated in language.

---

## 7. Value Proposition

### For the user

- **Speed**: find the relevant passage without reading the whole document.
- **Verification**: see the actual source passage, not a paraphrased claim.
- **Traceability**: know which document and which page or section the answer came from.
- **Honesty**: the system tells you when it does not know, when evidence is partial, and when sources conflict — instead of masking uncertainty with a confident-sounding answer.
- **Multi-document coherence**: ask a question across all your documents and get an answer that distinguishes what each one says.

### For the product/business

- **Differentiation from general-purpose AI chatbots**: the interactive citation and evidence-quality communication are the core moat. A chatbot that generates answers without inspectable sources is a different product.
- **Portfolio/learning value**: the architecture is model-agnostic, starting with a free model through Hermes for the agent layer, so the product is not locked into a single LLM provider. This is both a cost advantage and a learning opportunity.
- **Clear expansion path**: DOCX, OCR, evaluation datasets, and multi-user features are natural next steps that build on the same core architecture.

---

## 8. Success Metrics and Early Evaluation Dataset

The product should eventually be evaluated against a curated dataset. The metrics that matter are:

- **Retrieval precision and recall**: did the system retrieve the right passages, and did it retrieve all the passages needed to answer the question?
- **Citation correctness**: do the citations point to passages that actually support the claims made?
- **Answer faithfulness**: is the answer consistent with the source material, or does it introduce claims not supported by the documents?
- **Unsupported-answer rate**: how often does the system generate an answer that is not grounded in the documents?
- **Abstention accuracy**: when the system says it cannot answer, is it right?
- **Handling of conflicting evidence**: when sources disagree, does the system surface the conflict rather than silently choosing one?
- **Latency**: how long does the user wait for an answer? Speed is part of the core job.

### Early action: begin evaluation dataset design in parallel with implementation

Reliability and verifiability are core to DocuResearch's value proposition, so the evaluation methodology should not be deferred until after implementation. An early project action is to begin designing and curating a representative evaluation dataset in parallel with the product and architecture work.

The dataset eventually needs to cover all the metrics above. The early effort does not need to build the complete dataset before architecture or implementation begins; instead, the goals are to:

- **Establish the evaluation methodology**: define what a "good" retrieval, citation, answer, abstention, and conflict-handling look like in measurable terms, and agree on how each will be scored.
- **Start curating representative documents**: select documents that exercise the MVP formats (text-based PDF, Markdown, TXT), vary in length and structure, and include cases relevant to the primary user's verification job.
- **Start curating representative questions**: include questions with clear, single-source answers; questions requiring multi-passage synthesis; questions where evidence is partial; questions where evidence is insufficient; and questions where sources conflict.
- **Record expected answers and expected source locations**: each curated question should have a gold answer and the passage(s) that support it, so the dataset can be used to evaluate retrieval, citation, and faithfulness.

The intent is that architectural decisions — retrieval strategy, citation granularity, evidence-quality heuristics — can be validated against the dataset as it grows, rather than being evaluated for the first time after a full release. The dataset is also the foundation for the evaluation metrics in this section, so starting it early reduces the risk that the architecture is optimized for the wrong thing.

These are evaluation metrics, not necessarily MVP release criteria. The MVP should be built with them in mind — the architecture should make each measurable — but a full evaluation dataset is a future step, not an MVP gate.

---

## 9. Competitive Alternatives

### Current alternatives users have today

- **Ctrl+F and manual reading.** Slow, no cross-document synthesis, no verification aid. Still the default for many professionals.
- **General-purpose AI chatbots.** Fast answers, but no reliable source tracing, no interactive citation, and a tendency to generate plausible-sounding answers without making evidence quality visible. For verification-heavy work, these are often used with skepticism or not at all.
- **Document search tools.** Good at finding keywords, bad at answering the question "what does this document say about X?" and bad at verification.
- **Manual summarization and note-taking.** Accurate but slow, does not scale with document volume, and relies on the user's own memory and organization.

### What DocuResearch does differently

- Interactive citations that show the actual source passage, not a page number the user must go look up.
- Explicit evidence-quality narration: sufficient, partial, insufficient, conflicting.
- An explicit abstention posture: better to say "the documents don't say" than to fabricate.
- A verification-first design stance rather than a speed-first or summary-first stance.

---

## 10. Risks and Mitigations

### Risk: Citation accuracy is hard, and failures are visible

If a citation points to the wrong passage, or a claimed source passage does not actually say what the answer claims, the user loses trust immediately — possibly faster than they would with a chatbot that gives no citation at all.

**Mitigation:** The architecture must make citation generation a first-class, constrained step, not a free-form text generation. Evaluation against an answer/citation dataset is the long-term fix; in the MVP, the interactive citation mechanism (showing the actual passage) gives the user a direct verification path, which limits the damage of an inaccurate citation.

### Risk: Evidence-quality communication is subjective and easy to get wrong

Saying "the evidence is partial" when it is actually sufficient, or "conflicting" when it is not, degrades trust in the opposite direction.

**Mitigation:** Start with clear, conservative rules: err toward saying "I don't have enough" rather than overclaiming. Refine the language through evaluation and user feedback. Do not present a numerical confidence score that implies a precision the system does not have.

### Risk: Model-agnostic architecture may underperform with a free tier model

Starting with a free model through Hermes is a cost and learning advantage, but the quality of grounded answers and citations depends on the model's ability to reason over retrieved passages.

**Mitigation:** The architecture is model-agnostic by design. If a free model proves insufficient for citation quality or evidence narration, swapping to a paid model is a configuration change, not a rewrite. The MVP should establish the pipeline and evaluation approach first; model upgrades follow measured need.

### Risk: Scope creep from "eventually, everyone is a user"

The secondary user list (researchers, students, analysts) is large and tempting. Building for all of them at once dilutes the verification-first focus that distinguishes the product.

**Mitigation:** The primary user's job — find and verify — is the design center. Secondary users are explicitly deferred. Features that serve secondary users but do not serve the primary user's verification job are out of scope until the primary job is solid.

### Risk: MVP scope is thin — is there enough for a meaningful release?

Text-based PDF, Markdown, and TXT with search, citations, and conversational follow-up is a focused scope. It may feel narrow.

**Mitigation:** Narrow is correct for an MVP whose differentiation is citation quality and evidence communication, not format breadth. DOCX and OCR are expansion points that add users without changing the core job. A thin MVP that does the core job well is more valuable than a broad MVP that does it poorly.

---

## 11. Open Questions

These are decisions that have been made for the PRFAQ but may need validation during implementation or early user testing:

1. **What is the minimum viable citation granularity?** Page-level citations are the aspiration; section-level or paragraph-level may be the reality for some document types. The MVP should aim for the most specific locator the ingestion pipeline can reliably produce, and the interactive passage inspection is the fallback that makes granularity less critical.

2. **How does the system decide when evidence is "partial" versus "sufficient"?** This judgment drives the evidence-quality narration, and the boundary is fuzzy. The MVP can start with conservative, rule-based heuristics and refine through evaluation.

3. **What does "interactive citation" look like in a local-first, single-user MVP?** The implementation may be a clickable reference that displays the passage inline, a side panel, or a highlight in the original document view. The interaction model should be prototyped early enough to inform the evaluation of citation usefulness.

4. **Which retrieval approach gives the best precision for citation grounding?** The MVP needs a retrieval strategy that returns passages suitable for citation, not just relevant passages. This may be a layered approach (keyword/ BM25 for candidate retrieval, then ranking for citation suitability). The specific approach should be validated against the evaluation metrics.

5. **What is the evaluation dataset, and who builds it?** The metrics in Section 8 require a curated set of questions, documents, and gold answers with verified citations. This is a significant effort and may be the longest lead-time item for meaningful reliability measurement.

---

## 12. Summary

DocuResearch is a research assistant for technical documents, built around a single job: find specific information quickly and verify exactly where it came from. The MVP serves technical professionals working with text-based PDFs, Markdown, and plain text, with interactive citations, evidence-quality narration, honest abstention, and conversational follow-up across single or multiple documents. DOCX, OCR, and advanced contradiction detection are deferred. The product differentiates from general-purpose AI chatbots through inspectable sources and explicit evidence communication, and its long-term reliability will be measured against an evaluation dataset covering retrieval precision/recall, citation correctness, answer faithfulness, unsupported-answer rate, abstention accuracy, and latency.
