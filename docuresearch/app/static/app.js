// DocuResearch UI — Epic 6 (stories 6.1, 6.2).
//
// Citation mechanism (story 6.1 decision): each citation is a numbered reference
// placed inline at the end of the answer text. Activating it expands an inline
// panel showing the document name, location, and the passage text fetched from
// the local content store (GET /api/v1/citations/{id}), not the model's output.
//
// Multi-document answers (story 6.2): every reference shows its document name,
// and each document gets its own colour within an answer.
//
// All model- and document-derived text is inserted with textContent, never as
// HTML, so document content cannot inject markup or scripts.
"use strict";

const API = "/api/v1";

const state = {
  sessionId: null,
  documents: [],
};

const QUALITY_LABELS = {
  sufficient: "Supported by the documents",
  partial: "Partly supported",
  insufficient: "Not enough information in the documents",
  conflicting: "Sources disagree",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
}

async function api(path, options = {}) {
  const resp = await fetch(API + path, options);
  let body = null;
  try {
    body = await resp.json();
  } catch {
    body = null;
  }
  if (!resp.ok) {
    const summary = body && body.error ? body.error : `Request failed (${resp.status})`;
    const detail = body && body.detail ? `: ${body.detail}` : "";
    throw new Error(summary + detail);
  }
  return body;
}

function setMessage(id, text, isError = false) {
  const node = document.getElementById(id);
  node.textContent = text;
  node.classList.toggle("error", isError);
}

function formatSeconds(seconds) {
  return `${seconds.toFixed(1)} s`;
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

async function loadDocuments() {
  const { documents } = await api("/documents");
  state.documents = documents;
  renderDocuments();
  renderScopeOptions();
}

function renderDocuments() {
  const list = document.getElementById("document-list");
  list.replaceChildren();
  document.getElementById("document-empty").hidden = state.documents.length > 0;

  for (const doc of state.documents) {
    const details = [doc.format.toUpperCase()];
    if (doc.page_count) details.push(`${doc.page_count} page${doc.page_count === 1 ? "" : "s"}`);
    list.append(
      el("li", { class: "document" },
        el("div", {},
          el("span", { class: "document-name", text: doc.name }),
          el("span", { class: "document-meta", text: details.join(" · ") }),
        ),
        el("button", {
          type: "button",
          class: "link danger",
          "aria-label": `Remove ${doc.name}`,
          text: "Remove",
          onclick: () => removeDocument(doc),
        }),
      ),
    );
  }
}

function renderScopeOptions() {
  const select = document.getElementById("scope");
  const current = select.value;
  select.replaceChildren(el("option", { value: "", text: "All documents" }));
  for (const doc of state.documents) {
    select.append(el("option", { value: doc.document_id, text: doc.name }));
  }
  select.value = state.documents.some((d) => d.document_id === current) ? current : "";
}

async function uploadDocument(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  if (!data.get("name")) data.delete("name");
  const button = form.querySelector("button");
  button.disabled = true;
  setMessage("upload-message", "Uploading and indexing…");
  try {
    const doc = await api("/documents/upload", { method: "POST", body: data });
    setMessage("upload-message", `Added ${doc.name} (${doc.passage_count} passages).`);
    form.reset();
    await loadDocuments();
  } catch (err) {
    setMessage("upload-message", err.message, true);
  } finally {
    button.disabled = false;
  }
}

async function removeDocument(doc) {
  if (!window.confirm(`Remove "${doc.name}" and all its passages?`)) return;
  try {
    await api(`/documents/${encodeURIComponent(doc.document_id)}`, { method: "DELETE" });
    setMessage("upload-message", `Removed ${doc.name}.`);
    await loadDocuments();
  } catch (err) {
    setMessage("upload-message", err.message, true);
  }
}

// ---------------------------------------------------------------------------
// Questions and answers
// ---------------------------------------------------------------------------

function currentScope() {
  const documentId = document.getElementById("scope").value;
  return documentId ? { mode: "specific", document_id: documentId } : { mode: "all" };
}

async function askQuestion(event) {
  event.preventDefault();
  const textarea = document.getElementById("question");
  const question = textarea.value.trim();
  if (!question) return;

  const button = document.getElementById("ask-button");
  button.disabled = true;
  setMessage("ask-message", "Searching the documents and drafting an answer…");
  const scope = currentScope();
  try {
    if (!state.sessionId) {
      const session = await api("/conversation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ initial_document_scope: scope }),
      });
      state.sessionId = session.session_id;
    }
    const result = await api("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: state.sessionId, document_scope: scope }),
    });
    appendTurn(question, result.answer);
    textarea.value = "";
    setMessage("ask-message", "");
  } catch (err) {
    setMessage("ask-message", err.message, true);
  } finally {
    button.disabled = false;
    textarea.focus();
  }
}

function newConversation() {
  state.sessionId = null;
  document.getElementById("thread").replaceChildren();
  document.getElementById("thread-empty").hidden = false;
  setMessage("ask-message", "Started a new conversation.");
}

function appendTurn(question, answer) {
  document.getElementById("thread-empty").hidden = true;
  const thread = document.getElementById("thread");
  const item = el("li", { class: "turn" },
    el("p", { class: "question", text: question }),
    renderAnswer(answer),
  );
  thread.append(item);
  item.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderAnswer(answer) {
  const card = el("article", { class: `answer quality-${answer.evidence_quality}` });

  card.append(
    el("p", { class: "evidence" },
      el("span", { class: "badge", text: QUALITY_LABELS[answer.evidence_quality] }),
      el("span", { class: "narrative", text: answer.evidence_quality_narrative }),
    ),
  );

  const text = el("p", { class: "answer-text", text: answer.answer_text });
  const panels = el("div", { class: "citation-panels" });

  // One colour per distinct document within this answer (story 6.2).
  const docColours = new Map();
  answer.citations.forEach((citation, index) => {
    if (!docColours.has(citation.document_name)) {
      docColours.set(citation.document_name, docColours.size % 6);
    }
    const ref = renderCitationRef(citation, index + 1, docColours.get(citation.document_name),
      panels);
    text.append(" ", ref);
  });
  card.append(text, panels);

  if (answer.is_abstention) {
    card.append(el("p", {
      class: "abstention",
      text: "DocuResearch could not find enough support in your documents to answer this "
        + "definitively, so it did not guess.",
    }));
  }
  if (answer.citations.length === 0 && !answer.is_abstention) {
    card.append(el("p", { class: "abstention", text: "This answer has no citations." }));
  }

  card.append(el("p", {
    class: "timing",
    text: `Answered in ${formatSeconds(answer.total_latency_seconds)}`,
  }));
  return card;
}

function renderCitationRef(citation, number, colour, panels) {
  const panelId = `citation-${crypto.randomUUID()}`;
  const button = el("button", {
    type: "button",
    class: `citation-ref doc-colour-${colour}`,
    "aria-expanded": "false",
    "aria-controls": panelId,
    title: `${citation.document_name}, ${citation.location_label}`,
  },
  el("span", { class: "ref-number", text: `[${number}]` }),
  el("span", { class: "ref-source", text: `${citation.document_name} · ${citation.location_label}` }),
  );

  const panel = el("section", {
    id: panelId,
    class: `citation-panel doc-colour-${colour}`,
    "aria-label": `Source ${number}`,
    hidden: "",
  });
  panels.append(panel);

  button.addEventListener("click", async () => {
    const expanding = panel.hidden;
    panel.hidden = !expanding;
    button.setAttribute("aria-expanded", String(expanding));
    if (expanding && !panel.dataset.loaded) {
      await loadPassage(citation, number, panel);
    }
  });
  return button;
}

async function loadPassage(citation, number, panel) {
  panel.replaceChildren(el("p", { class: "loading", text: "Loading passage…" }));
  try {
    // Always display the content store's record, not the text carried in the answer.
    const passage = await api(`/citations/${encodeURIComponent(citation.passage_id)}`);
    panel.replaceChildren(
      el("header", {},
        el("span", { class: "ref-number", text: `[${number}]` }),
        el("span", { class: "panel-doc", text: passage.document_name }),
        el("span", { class: "panel-location", text: passage.location_label }),
      ),
      el("blockquote", { class: "passage-text", text: passage.text }),
      el("p", { class: "provenance", text: "Exact text stored from the original document." }),
    );
    panel.dataset.loaded = "true";
  } catch (err) {
    panel.replaceChildren(el("p", {
      class: "error",
      text: `This passage could not be loaded (${err.message}). `
        + "The document may have been removed since this answer was given.",
    }));
  }
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

async function checkHealth() {
  const status = document.getElementById("llm-status");
  try {
    const health = await api("/health");
    status.textContent = health.llm_configured
      ? "Ready"
      : "No language model configured: you can upload and browse documents, "
        + "but questions are disabled.";
    status.classList.toggle("warning", !health.llm_configured);
    document.getElementById("ask-button").disabled = !health.llm_configured;
  } catch {
    status.textContent = "Cannot reach the DocuResearch server.";
    status.classList.add("warning");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("upload-form").addEventListener("submit", uploadDocument);
  document.getElementById("ask-form").addEventListener("submit", askQuestion);
  document.getElementById("new-conversation").addEventListener("click", newConversation);
  document.getElementById("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      document.getElementById("ask-form").requestSubmit();
    }
  });
  checkHealth();
  loadDocuments().catch((err) => setMessage("upload-message", err.message, true));
});
