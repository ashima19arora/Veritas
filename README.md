# VERITAS — Offline Multi-Agent Research & Fact-Verification System

Secure offline multi-agent research system powered by Ollama.
Every answer is researched, independently verified, checked for contradictions, and cited — before you ever see it.

**Veritas** is Latin for _truth_. That's the whole premise of this project: an AI system that doesn't just answer confidently, but actually checks whether it's earned the right to.

---

## Built for

Innova Hack Chapter-1 — Domain 3 (Gen AI), Problem Statement 1

> Generative AI tools are powerful researchers but often struggle with hallucination and unverified claims. A system where multiple AI agents check and challenge each other can produce far more trustworthy output than a single model working alone.
>
> Build a multi-agent pipeline where one agent researches a given topic, another cross-verifies claims against multiple sources, a third detects contradictions or hallucinations, and a final agent compiles a citation-backed report — complete with a confidence score for each claim.

---

## What it does

Most AI tools give you an answer and ask you to trust it. VERITAS doesn't work that way.

Upload documents, ask a question, and VERITAS answers using only what's actually in those documents — nothing invented, nothing pulled from the internet, nothing running through a cloud API. But that final answer isn't the whole story. Behind it, four separate passes happen every single time: the system researches an answer, then independently re-checks that answer against fresh evidence, then checks whether the source documents even agree with each other, and only then writes the version you actually see — complete with a confidence score and exact page citations.

Nothing here is trusted blindly, not even by the system's own earlier steps. And unlike most AI tools, none of this happens behind a curtain — every stage's work is visible in the interface, live, as it happens. You're not told to trust the answer. You can watch it get earned.

It runs entirely offline. No document, no query, nothing ever leaves the machine it's running on — which matters most for exactly the kind of documents people are usually most nervous about sharing: internal reports, research papers, anything sensitive enough that uploading it to a cloud AI isn't really an option.

---

## Architecture — four jobs, one model

VERITAS doesn't run four different AI models. It runs one — you choose which (Gemma or Phi, whichever is installed and selected) — but asks that single model to do four different jobs, one after another, each with its own instructions and its own responsibility.

![VERITAS Pipeline](images/pipeline_diagram.png)

- **Researcher** — reads the retrieved document chunks and drafts a first-pass answer. Fast, direct, not yet trusted by anything downstream.
- **Verifier** — puts on a skeptic's hat. Doesn't take the Researcher's draft at face value — independently searches the documents again, from scratch, using the draft itself as the query, and checks every claim against fresh evidence it found on its own.
- **Contradiction Checker** — runs at the same time as the Verifier, looking for a different failure mode entirely: do the source documents themselves disagree with each other? (Genuinely useful when comparing, say, a 2021 policy against its 2026 update.)
- **Synthesizer** — the last word. Reads the draft and the Verifier's findings, removes or caveats anything that didn't hold up, flags any claim that quietly combined two unrelated facts into something that sounds specific but isn't, and writes the final answer — with a confidence score and source citations attached. The Contradiction Checker's findings are shown to you directly, in their own section — kept separate rather than folded into this prose, so a borderline flag never quietly reshapes the wording of the actual answer.

No single stage's output is final until something else has checked it.

---

## Screenshots

**Main interface**

![Main UI](images/main_ui.png)

**A full 4-stage answer, expanded**

![Pipeline expanded](images/pipeline_expanded.png)

**Live tech stack, shown in the caption of every answer**

![Caption detail](images/caption_closeup.png)

**Correctly declining an out-of-scope question, instead of guessing**

![Guardrail decline](images/guardrail_decline.png)

---

## Tech stack

- **LLM inference:** Ollama (fully local — Gemma 3, Phi-3, or Llama 3.2, selectable)
- **Embeddings:** bge-small-en-v1.5 / bge-large-en-v1.5 / all-MiniLM-L6-v2 (selectable, local weights)
- **Vector search:** FAISS / Chroma / Qdrant (selectable, all pre-built together)
- **Reranking:** FlashRank cross-encoder
- **Orchestration:** LangChain
- **Interface:** Streamlit
- **Formats supported:** PDF, DOCX, DOC, TXT

---

## Features

- **Fully offline** — no internet connection required at any point after initial setup. No document content or query ever leaves the machine.
- **Multi-agent verification** — not a single LLM call. Four distinct passes, each checking the work of the one before it.
- **Full transparency** — every stage's output is visible in the UI as it runs, not hidden behind a spinner. You see the Researcher's draft, the Verifier's independent check, the Contradiction Checker's findings, and the Synthesizer's final pass, separately.
- **Objective confidence scoring** — not just the model's self-reported opinion. An independent embedding-similarity check compares each claim against its source evidence mathematically, and can override an overconfident self-score.
- **Multiple embedding models and vector engines** — all combinations pre-built together on rebuild, so switching between them mid-session is instant, with no reprocessing wait.
- **Source citations** — every answer ends with the exact document and page number it was drawn from, generated from the retrieved chunks directly, not asked of the LLM (so citations can't be hallucinated).
- **Contradiction detection with real logic** — not just an LLM asked "any conflicts?" — a structured comparison that distinguishes a genuine factual conflict from a paraphrase, a general-rule-plus-exception, or two unrelated facts sharing a keyword.

---

## How to run

### Option A — Docker

```bash
# Build the image
docker build -t veritas .

# Run the container (Ollama must be running on the host machine)
docker run -p 8501:8501 veritas
```

Then open `http://localhost:8501` in your browser. Ollama itself runs on your host machine, not inside the container — the app connects to it over the network.

### Option B — Manual setup

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is installed and running, with at least one model pulled
ollama pull gemma3:4b

# Run the app
streamlit run streamlit_app.py
```

Embedding model weights should be placed in a local `models/` folder (see `requirements.txt` for the exact folder names expected) so the app runs fully offline from the first launch, with no download step.

---

## Known limitations

- Response time scales with the number of stages and the size of the chosen models — a full 4-stage answer on larger models can take noticeably longer than a single-pass system.
- No timeout is currently set on the Ollama connection — if the Ollama service itself hangs (rare, but possible), the app will wait rather than time out cleanly.
- Comparison-style questions across many documents at once retrieve broadly, which can occasionally surface less-relevant source chunks alongside the useful ones.

---

## Future scope

- **Additional specialized modes** — task-specific pipelines beyond general Q&A and document comparison, built for distinct use cases rather than duplicating existing functionality.
- **Desktop quick-access app** — a lightweight version triggerable by a global hotkey, for instant access without keeping a browser tab open.

---

## License

_(to be decided)_
