import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Chroma.*")
warnings.filterwarnings("ignore", message=".*langchain.*")
try:
    from langchain._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["FAISS_NO_AVX2"] = "1"

import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)

import time
import re
import numpy as np
import concurrent.futures
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import ollama as ollama_client
from flashrank import Ranker, RerankRequest

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DOCS_DIR       = "docs"
VECTOR_DB_ROOT = "vectorstore_index"
EMBED_MODEL_FOLDER = "bge-small-en-v1.5"   # matches folder name inside models/
CHUNK_SIZE     = 750
CHUNK_OVERLAP  = 100

VECTOR_DB_DIR = os.path.join(VECTOR_DB_ROOT, EMBED_MODEL_FOLDER, "faiss")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

GREETINGS = ["hi", "hello", "hey", "how are you", "good morning", "good evening"]
FAREWELLS = ["bye", "goodbye", "exit", "quit", "see you", "take care", "thanks", "thank you"]
GUARDRAIL_PHRASE = "I cannot find sufficient verified data within the loaded documents."

st.set_page_config(page_title="Offline Fact-Verification", layout="centered")

st.markdown("""
<style>
    [data-testid="stSidebarHeader"] { min-height: 0rem; padding-top: 0.1rem; padding-bottom: 0.1rem; }
    [data-testid="stSidebarUserContent"] { padding-top: 0.25rem; }
    [class*="st-key-uploader_wrap"] { display: flex; flex-direction: column; align-items: center; margin-bottom: -1.25rem; }
    [class*="st-key-uploader_wrap"] [data-testid="stFileUploaderDropzoneInstructions"] { text-align: center; }
    [class*="st-key-delete_"] + div [data-testid="stHorizontalBlock"],
    [data-testid="stHorizontalBlock"]:has([class*="st-key-delete_"]) { gap: 0.4rem; margin-bottom: -0.6rem; }
    [class*="st-key-delete_"] button { font-weight: 700; font-size: 1.05rem; padding: 0rem 0.3rem; min-height: 1.5rem; color: #888888; }
    [class*="st-key-delete_"] button:hover { color: #e03131; }
    .system-caption { font-size: 0.82rem; color: #6b6b6b; margin-top: -0.3rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SESSION STATE — ported exactly from VEDA client.py
# ---------------------------------------------------------------------------
if "messages"          not in st.session_state: st.session_state.messages          = []
if "generating"        not in st.session_state: st.session_state.generating        = False
if "pending_question"  not in st.session_state: st.session_state.pending_question  = None
if "pending_deletes"   not in st.session_state: st.session_state.pending_deletes   = set()
if "saved_uploads"     not in st.session_state: st.session_state.saved_uploads     = set()
if "uploader_version"  not in st.session_state: st.session_state.uploader_version  = 0
if "just_rebuilt"      not in st.session_state: st.session_state.just_rebuilt      = False
if "force_rebuild"     not in st.session_state: st.session_state.force_rebuild     = False
if "processing_task"   not in st.session_state: st.session_state.processing_task   = None
if "processing_files"  not in st.session_state: st.session_state.processing_files  = []
if "processing_doc"    not in st.session_state: st.session_state.processing_doc    = None
if "last_rebuilt_time" not in st.session_state: st.session_state.last_rebuilt_time = None
if "vectorstore"       not in st.session_state: st.session_state.vectorstore       = None
if "upload_toast"      not in st.session_state: st.session_state.upload_toast      = None
if "delete_toast"      not in st.session_state: st.session_state.delete_toast      = None

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
def display_name_for(filename, max_len=26):
    if len(filename) <= max_len: return filename
    return filename[:max_len - 3] + "..."

def _slug(name):
    return "".join(c if c.isalnum() else "_" for c in name)

def get_ollama_models():
    try:
        return [m.model for m in ollama_client.list().models] or []
    except Exception:
        return []

# ---------------------------------------------------------------------------
# EMBEDDINGS — ported exactly from server.py's load_embeddings().
# Checks local models/ folder FIRST (fully offline, no HuggingFace Hub call).
# Falls back to Hub namespace only if local folder missing.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embeddings():
    try:
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"   # torch/CUDA check failed — fall back safely rather than crashing
    local_path = os.path.join("models", EMBED_MODEL_FOLDER)
    try:
        if os.path.exists(local_path):
            return HuggingFaceEmbeddings(
                model_name=local_path,
                model_kwargs={"device": device},
                encode_kwargs={"batch_size": 8}
            )
        namespace = f"BAAI/{EMBED_MODEL_FOLDER}" if "bge" in EMBED_MODEL_FOLDER \
                    else f"sentence-transformers/{EMBED_MODEL_FOLDER}"
        return HuggingFaceEmbeddings(
            model_name=namespace,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 8}
        )
    except Exception as e:
        st.error(f"Could not load embedding model: {e}. Check that models/{EMBED_MODEL_FOLDER} contains valid model files.")
        raise

def index_exists():
    return os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss"))

def load_and_chunk_documents():
    """Ported exactly from server.py's load_and_chunk_documents() — skips hidden/Mac files, skips unreadable files (e.g. password-protected PDFs, corrupted files) and reports which ones."""
    chunks = []
    skipped = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for fname in os.listdir(DOCS_DIR):
        if fname.startswith("."):
            continue
        fpath = os.path.join(DOCS_DIR, fname)
        ext = fname.lower().split(".")[-1]
        try:
            if ext == "pdf":              docs = PyPDFLoader(fpath).load()
            elif ext in ("docx", "doc"): docs = Docx2txtLoader(fpath).load()
            elif ext == "txt":            docs = TextLoader(fpath, encoding="utf-8").load()
            else: continue
            if not docs or not any(d.page_content.strip() for d in docs):
                skipped.append(fname)   # loaded but empty (e.g. scanned/image-only PDF, no extractable text)
                continue
            chunks.extend(splitter.split_documents(docs))
        except Exception:
            skipped.append(fname)   # unreadable — password-protected, corrupted, or unsupported encoding
    return chunks, skipped

def rebuild_index():
    """Rebuild FAISS index from all docs in DOCS_DIR. Returns True on success, False if no docs (matches server.py's rebuild_all_indexes contract)."""
    chunks, skipped = load_and_chunk_documents()
    if skipped:
        st.warning(f"Could not read {len(skipped)} file(s): {', '.join(skipped)} — may be password-protected, corrupted, or contain no extractable text.")
    if not chunks:
        # no docs left — wipe stale index so nothing stale can be queried (last-doc-delete bug fix from VEDA)
        import shutil
        shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        st.session_state.vectorstore = None
        return False
    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(VECTOR_DB_DIR)
    st.session_state.vectorstore = vs
    return True

def load_vectorstore():
    if st.session_state.vectorstore is not None:
        return st.session_state.vectorstore
    if index_exists():
        try:
            embeddings = get_embeddings()
            vs = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
            st.session_state.vectorstore = vs
            return vs
        except Exception as e:
            st.error(f"Saved index appears corrupted ({e}). Wiping it — please re-upload documents and rebuild.")
            import shutil
            shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)
            return None
    return None

# ---------------------------------------------------------------------------
# LAYER 2 — MULTI-AGENT PIPELINE
# Same Ollama model, 4 different prompts/jobs, called in sequence.
# Each stage has its own guardrail, same ideology as Layer 1's proven prompt.
# ---------------------------------------------------------------------------

@st.cache_resource
def get_flashrank_engine():
    """Tiny local cross-encoder — re-scores candidates by true semantic relevance, not just vector distance."""
    return Ranker()

def retrieve_chunks(vectorstore, query, k=15, top_n=6):
    """Two-stage retrieval, matches VEDA exactly: bi-encoder search (k=15 candidates)
    -> FlashRank cross-encoder rerank -> top_n most relevant chunks, in true-relevance order."""
    candidates = vectorstore.similarity_search(query, k=k)
    if not candidates:
        return []
    try:
        flash_ranker = get_flashrank_engine()
        payload = [{"id": i, "text": c.page_content} for i, c in enumerate(candidates)]
        ranked = flash_ranker.rerank(RerankRequest(query=query, passages=payload))
        result_docs = []
        for r in ranked[:top_n]:
            idx = r.get("id")
            if idx is not None and idx < len(candidates):
                result_docs.append(candidates[idx])
        return result_docs if result_docs else candidates[:top_n]
    except Exception:
        # FlashRank failed for any reason (e.g. model download issue) — fall back to plain similarity order
        return candidates[:top_n]

def extract_sources(chunks):
    """Builds a clean 'doc (pg N)' list from chunk metadata, deduplicated, in order of first appearance."""
    sources = []
    seen = set()
    for c in chunks:
        doc = c.metadata.get("source", "")
        page = c.metadata.get("page", None)
        doc_name = os.path.basename(doc) if doc else "unknown"
        key = (doc_name, page)
        if key not in seen:
            seen.add(key)
            sources.append(f"{doc_name} (pg {page + 1})" if page is not None else doc_name)
    return sources

def compute_grounding_score(claim_text, source_chunks):
    """OBJECTIVE confidence signal — independent of the LLM's own self-grading.
    Embeds the claim text and each source chunk (reusing the already-loaded embedding
    model), measures cosine similarity, returns the best match. Low similarity here
    means the claim drifted from what the source actually says, even if the LLM
    confidently graded itself SUPPORTED — this is real math, not the model's opinion."""
    if not source_chunks:
        return 0.0
    try:
        embeddings = get_embeddings()
        claim_vec = np.array(embeddings.embed_query(claim_text))
        claim_norm = np.linalg.norm(claim_vec)
        if claim_norm == 0:
            return 0.0
        best_similarity = 0.0
        for chunk in source_chunks:
            chunk_vec = np.array(embeddings.embed_query(chunk.page_content))
            chunk_norm = np.linalg.norm(chunk_vec)
            if chunk_norm == 0:
                continue
            similarity = float(np.dot(claim_vec, chunk_vec) / (claim_norm * chunk_norm))
            best_similarity = max(best_similarity, similarity)
        return round(best_similarity, 3)
    except Exception:
        return None   # grounding check failed for any reason — don't block the pipeline, just skip this signal

def run_researcher(vectorstore, question, model_name):
    """Stage 1 — retrieve + draft an answer/claims from context. Same prompt as Layer 1."""
    chunks = retrieve_chunks(vectorstore, question, k=15, top_n=8)
    context = "\n\n---\n\n".join([c.page_content for c in chunks])

    prompt_template = """You are an elite, highly accurate document analysis system.
Review the provided context carefully to address the query.

Operational Rules:
1. Rely ONLY on facts directly mentioned in the Context below. Do not extrapolate or hallucinate.
2. If the context contains relevant information, answer it directly and completely. Do not append disclaimers after a complete answer.
3. If and ONLY IF the context contains NO relevant information at all, respond with: "I cannot find sufficient verified data within the loaded documents." — nothing else.
4. Keep your analysis concise, structured, and objective.
5. If the Question asks about a specific sub-category (e.g. "AI projects", "cybersecurity projects") and the Context only states a rule generally (not tied to that sub-category), answer with the GENERAL rule and explicitly note that no sub-category-specific rule was found — do NOT combine the general rule with an unrelated detail (like a budget threshold or approval type) to imply a specific rule exists when it doesn't.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = OllamaLLM(model=model_name, temperature=0.2)
    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({"context": context, "question": question})
    return draft, chunks

def run_verifier(vectorstore, draft_answer, model_name):
    """Stage 2 — independently re-retrieves evidence for the draft's claims, checks support, scores confidence."""
    # Use the draft itself as the search query — an independent re-check, not reusing Researcher's chunks
    fresh_chunks = retrieve_chunks(vectorstore, draft_answer, k=15, top_n=7)
    fresh_context = "\n\n---\n\n".join([c.page_content for c in fresh_chunks])

    prompt_template = """You are an expert fact-verifier and proofreader. You did NOT write the claim below — your job is to independently check it.

Operational Rules:
1. Compare the Claim against the Evidence below ONLY. Do not use outside knowledge.
2. For each distinct factual statement in the Claim, judge: SUPPORTED, PARTIALLY SUPPORTED, or NOT SUPPORTED by the Evidence.
3. The overall confidence score MUST reflect the WEAKEST verdict among all statements — it is not an average and not based only on the statements that passed. If even one statement is NOT SUPPORTED, the overall score must be 0.4 or lower. If any statement is only PARTIALLY SUPPORTED (and none are NOT SUPPORTED), the overall score must be between 0.4 and 0.7. Only give 0.8-1.0 if EVERY statement is fully SUPPORTED. The score MUST be a decimal between 0.0 and 1.0 ONLY (e.g. 0.75) — never a whole number, never a percentage, never a number above 1.0.
4. If the Evidence contains nothing relevant to judge the Claim, say so explicitly and give confidence 0.0.
5. Be concise — short verdict per statement, then the overall score. No extra commentary.

Evidence:
{evidence}

Claim to verify:
{claim}

Verification:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["evidence", "claim"])
    llm = OllamaLLM(model=model_name, temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    verification = chain.invoke({"evidence": fresh_context, "claim": draft_answer})

    # OBJECTIVE signal — independent of the LLM's self-grading above.
    grounding_score = compute_grounding_score(draft_answer, fresh_chunks)
    if grounding_score is not None:
        verification += f"\n\n[Automated grounding check — embedding similarity between claim and evidence: {grounding_score:.2f}]"

    return verification, fresh_chunks, grounding_score

def run_contradiction_check(researcher_chunks, model_name):
    """Stage 3 — checks the Researcher's own retrieved chunks against each other for internal conflicts."""
    if len(researcher_chunks) < 2:
        return "Not enough distinct chunks retrieved to check for contradictions."

    numbered = "\n\n".join([f"[Chunk {i+1}]\n{c.page_content}" for i, c in enumerate(researcher_chunks)])

    prompt_template = """You are a strict contradiction-detection system. You will be given several document excerpts, numbered.

A REAL contradiction means: two chunks state DIFFERENT NUMBERS OR VALUES about the EXACT SAME subject and scope that CANNOT both be true at the same time.

Do NOT flag as a contradiction:
- The same fact worded differently in two chunks (e.g. "start() creates a new thread" and "calling start() creates a new thread, then runs it" — these AGREE, just one has more detail)
- Two chunks stating the SAME NUMBER, even if scope/wording differs (e.g. "team size is 4" and "team size is 4 for projects under ₹1 crore" — SAME number, 4 = 4, this is NOT a conflict)
- A general rule plus a specific exception with a DIFFERENT number for the exception case ONLY (e.g. "minimum team size is 4" and "minimum team size is 5 for projects over ₹2 crore" — the second is a special case, NOT a conflict)
- Two chunks about different topics that happen to share a keyword
- A chunk elaborating on or adding detail to another chunk's claim
- Two DIFFERENT specific entities each correctly having a different value by design (e.g. "Bootstrap class loader = most trusted" and "Extension class loader = medium trust" are TWO DIFFERENT class loaders, each with its own correct trust level — this is a hierarchy working as intended, NOT a conflict. Only flag this kind of comparison if the SAME single entity is given two different values in different chunks.)

Before flagging anything, double-check: are the two numbers/values ACTUALLY different? If they are the same number, it is NOT a contradiction, no matter how the scope is worded.

ONLY flag as a contradiction: two chunks giving genuinely DIFFERENT VALUES for the exact same specific thing under the exact same scope (e.g. one chunk says the deadline is 45 days, another chunk says the deadline for that SAME scenario is 60 days — different numbers, same scope — that's a real conflict).

Steps:
1. Extract concrete factual claims (dates, numbers, named facts, stated rules) from each chunk, noting the exact scope each claim applies to.
2. Compare claims only within the SAME scope across chunks. Verify the numbers are actually different before flagging.
3. Decide ONCE: either contradictions exist, or they don't.
4. If NONE exist, your ENTIRE response must be exactly: "No contradictions detected across the retrieved chunks." — nothing else, no lists, no explanation.
5. If real contradictions exist, your ENTIRE response must be ONLY the list, in this format: "Chunk X vs Chunk Y: [conflicting values, same scope] — [why they cannot both be true]" — do NOT add the "No contradictions detected" line anywhere if you are listing real conflicts.

Chunks:
{chunks}

Contradiction Analysis:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["chunks"])
    llm = OllamaLLM(model=model_name, temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"chunks": numbered})

    # Safety net: if the model listed real contradictions (contains "Chunk") but also
    # appended the "no contradictions" line, strip that trailing contradictory sentence.
    no_conflict_line = "No contradictions detected across the retrieved chunks."
    if "Chunk" in result and no_conflict_line in result:
        result = result.replace(no_conflict_line, "").strip()

    return result

def run_synthesizer(question, draft_answer, verification, contradiction_report, model_name, grounding_score=None):
    """Stage 4 — compiles everything into one final, clean, cited report for the user."""
    grounding_note = (
        f"\n\nAutomated grounding score (independent embedding-similarity check, 0.0-1.0, "
        f"higher = better math-verified match between claim and evidence): {grounding_score:.2f}"
        if grounding_score is not None else ""
    )
    prompt_template = """You are a synthesis agent. Compile the material below into ONE final, clean answer for the user.

Operational Rules:
1. Base your final answer on the Draft Answer, adjusted per the Verification findings.
2. If Verification flagged something as NOT SUPPORTED, remove or clearly caveat it in the final answer — do not present unsupported claims as fact.
3. If the Contradiction Analysis found conflicts, add a short "Note" at the end flagging this to the user.
4. Check the Draft Answer for FABRICATED CONNECTIONS: if it links two separately-true facts together in a way not explicitly stated in the Evidence (e.g. combining a general rule with an unrelated threshold or condition), flag this explicitly in your Note — even if each individual fact was separately marked SUPPORTED.
5. If the Original Question asks about something more specific than what the documents cover (e.g. asks about a sub-category that the documents only address generally), say so plainly instead of implying a specific rule exists.
6. An automated grounding score (independent embedding-similarity math, not the LLM's opinion) is provided below if available. If this score is LOW (below 0.4) but the Verification's self-reported confidence was High, trust the grounding score more — lower your final confidence level and add a note that the automated check found weaker support than the self-assessment suggested.
7. ALWAYS end with a line in this EXACT format: "Confidence Level: [High/Medium/Low] (X.X)" — where X.X is the numeric score from Verification, adjusted per rule 6 if applicable. Never omit the number. If Verification was skipped, omit this line entirely.
8. Keep it concise, well-structured, and objective. Do not repeat the raw verification text — synthesize it.

Original Question:
{question}

Draft Answer:
{draft}

Verification Findings:
{verification}{grounding_note}

Contradiction Analysis:
{contradiction}

Final Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "draft", "verification", "contradiction", "grounding_note"])
    llm = OllamaLLM(model=model_name, temperature=0.2)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": question, "draft": draft_answer,
        "verification": verification, "contradiction": contradiction_report,
        "grounding_note": grounding_note,
    })

def extract_and_strip_confidence(answer_text):
    """Pulls the 'Confidence Level: X (Y.Y)' line out of the answer body so it can be
    shown in the caption instead, and returns the cleaned answer text plus the extracted string.
    Also normalizes/clamps the score if the model outputs it on the wrong scale (e.g. 8.2 instead of 0.82).
    Tolerant of the model omitting the High/Medium/Low word and just giving a number."""
    match = re.search(r"Confidence Level:\s*(High|Medium|Low)?\s*\(?([\d.]+)?\)?", answer_text, re.IGNORECASE)
    if not match or (not match.group(1) and not match.group(2)):
        return answer_text.strip(), None
    level = match.group(1)
    score = match.group(2)
    if score:
        try:
            score_val = float(score)
            if score_val > 1.0:
                # model likely wrote a 0-10 or 0-100 scale value by mistake — normalize down
                score_val = score_val / 10 if score_val <= 10 else score_val / 100
            score_val = max(0.0, min(1.0, score_val))   # hard clamp, never outside 0.0-1.0
            score = f"{score_val:.2f}"
            if not level:
                # derive level word from the number if the model didn't give one
                level = "High" if score_val >= 0.8 else "Medium" if score_val >= 0.4 else "Low"
        except ValueError:
            score = None
    level = level.capitalize() if level else "Unknown"
    confidence_str = f"{level} ({score})" if score else level
    cleaned = answer_text[:match.start()].rstrip(" \n-—:")
    return cleaned, confidence_str

# ---------------------------------------------------------------------------
# DOC LIST FRAGMENT — ported exactly from client.py's doc_list_section()
# ---------------------------------------------------------------------------
@st.fragment
def doc_list_section():
    all_docs = sorted(f for f in os.listdir(DOCS_DIR) if not f.startswith("."))
    visible = [d for d in all_docs if d not in st.session_state.pending_deletes]
    st.write("")
    st.write(f"**Loaded documents ({len(visible)})**")
    if visible:
        outer = st.container(height=260, border=True) if len(visible) >= 8 else st.container(border=True)
        with outer:
            for doc_name in visible:
                slug = _slug(doc_name)
                name_col, del_col = st.columns([7, 1], gap="small", vertical_alignment="center")
                with name_col:
                    st.write(display_name_for(doc_name))
                with del_col:
                    if st.button("x", key=f"delete_{slug}", help=f"Delete {doc_name}",
                                 type="tertiary", disabled=st.session_state.generating):
                        st.session_state.pending_deletes.add(doc_name)
                        st.session_state.processing_doc  = doc_name
                        st.session_state.processing_task = "delete"
                        st.session_state.generating      = True
                        st.rerun()
    else:
        st.caption("No documents yet — upload PDF, Word, or TXT files above.")

# ---------------------------------------------------------------------------
# SIDEBAR — ported exactly from client.py, controls disabled while generating
# ---------------------------------------------------------------------------
with st.sidebar:
    available_llms = get_ollama_models()
    if not available_llms:
        st.warning("Ollama not reachable — start Ollama and refresh.")
        selected_model = st.selectbox("Select Language Model", ["(no models found)"], disabled=True)
    else:
        default_idx = available_llms.index("gemma3:4b") if "gemma3:4b" in available_llms else 0
        selected_model = st.selectbox(
            "Select Language Model", available_llms, index=default_idx,
            disabled=st.session_state.generating
        )

    st.header("Document Management")

    with st.container(key="uploader_wrap"):
        new_uploads = st.file_uploader(
            "Add new documents", type=["pdf", "docx", "doc", "txt"],
            accept_multiple_files=True,
            key=f"doc_uploader_{st.session_state.uploader_version}",
            disabled=st.session_state.generating
        )

    # Queue files for upload — actual save + rebuild happens on main page (Rerun 2),
    # so the spinner appears in the chat area, not the sidebar.
    if new_uploads and not st.session_state.generating:
        files_to_queue = [(f.name, f.read()) for f in new_uploads
                           if f.name not in st.session_state.saved_uploads]
        if files_to_queue:
            st.session_state.processing_files = files_to_queue
            st.session_state.processing_task  = "upload"
            st.session_state.generating       = True
            st.rerun()

    doc_list_section()
    st.write("")

    if st.session_state.force_rebuild:
        with st.spinner("Rebuilding index..."):
            try:
                ok = rebuild_index()
                if ok:
                    st.session_state.just_rebuilt = True
                    st.session_state.last_rebuilt_time = time.time()
                    st.session_state.pending_deletes = set()
                else:
                    st.warning("No documents found to index.")
            except Exception as e:
                st.error(f"Index rebuild failed: {e}. Check your documents and try again.")
        st.session_state.force_rebuild = False
        st.rerun()

    if st.session_state.generating:
        st.button("Rebuild Vector Index", use_container_width=True, disabled=True,
                   help="Wait for current operation to finish.")
    else:
        if st.button("Rebuild Vector Index", use_container_width=True):
            st.session_state.force_rebuild = True
            st.rerun()

    if st.session_state.last_rebuilt_time is not None:
        elapsed = int((time.time() - st.session_state.last_rebuilt_time) / 60)
        if elapsed == 0:
            st.caption("Last rebuilt: just now")
        elif elapsed == 1:
            st.caption("Last rebuilt: 1 min ago")
        else:
            st.caption(f"Last rebuilt: {elapsed} mins ago")
    elif not index_exists():
        st.caption("No index found — upload documents and click Rebuild")
    else:
        st.caption("Last rebuilt: unknown — consider rebuilding")

# ---------------------------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------------------------
col1, col2 = st.columns([4, 1.5])
with col1:
    st.title("VERITAS")
    st.markdown(
        '<div class="system-caption">Secure offline multi-agent research system powered by Ollama</div>',
        unsafe_allow_html=True
    )
with col2:
    st.write("")
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

if st.session_state.get("upload_toast"): st.toast(st.session_state.pop("upload_toast"))
if st.session_state.get("delete_toast"): st.toast(st.session_state.pop("delete_toast"))

# ---------------------------------------------------------------------------
# MAIN PAGE PROCESSING — two-rerun pattern ported exactly from client.py.
# Runs AFTER the header so the spinner appears in the chat area.
# try/except/finally guarantees generating always resets, even on error.
# ---------------------------------------------------------------------------
if st.session_state.generating and st.session_state.processing_task == "upload":
    files = st.session_state.processing_files
    ok = False
    try:
        with st.spinner(f"Uploading {len(files)} file(s) and rebuilding index — please wait..."):
            for fname, fbytes in files:
                with open(os.path.join(DOCS_DIR, fname), "wb") as f:
                    f.write(fbytes)
            ok = rebuild_index()
    except Exception as e:
        st.error(f"Upload/rebuild failed: {e}")
    finally:
        st.session_state.processing_files = []
        st.session_state.processing_task  = None
        st.session_state.generating       = False
    for fname, _ in files:
        st.session_state.saved_uploads.add(fname)
    st.session_state.uploader_version += 1
    if ok:
        st.session_state.just_rebuilt = True
        st.session_state.last_rebuilt_time = time.time()
        st.session_state.upload_toast = f"Added {len(files)} file(s)"
    st.rerun()

elif st.session_state.generating and st.session_state.processing_task == "delete":
    doc_name = st.session_state.processing_doc
    ok = False
    try:
        with st.spinner(f"Removing {doc_name} and rebuilding index — please wait..."):
            fpath = os.path.join(DOCS_DIR, doc_name)
            if os.path.exists(fpath):
                os.remove(fpath)
            ok = rebuild_index()   # rebuild_index() itself wipes stale index if no docs remain
    except Exception as e:
        st.error(f"Delete/rebuild failed: {e}")
    finally:
        st.session_state.processing_doc  = None
        st.session_state.processing_task = None
        st.session_state.generating      = False
    st.session_state.just_rebuilt = True
    st.session_state.last_rebuilt_time = time.time()
    st.session_state.saved_uploads.discard(doc_name)
    st.session_state.uploader_version += 1
    st.session_state.pending_deletes.discard(doc_name)
    st.session_state.delete_toast = f"Removed {doc_name}"
    remaining = [f for f in os.listdir(DOCS_DIR) if not f.startswith(".")]
    if not remaining:
        # last doc deleted — clear everything so re-uploads of any filename work cleanly
        st.session_state.saved_uploads = set()
        st.session_state.pending_deletes = set()
    st.rerun()

# ---------------------------------------------------------------------------
# CHAT HISTORY — replays persisted stage boxes for past multi-agent answers
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑\u200d💻" if message["role"] == "user" else "🖥️"):
        if message.get("stages"):
            stages = message["stages"]
            with st.expander("Stage 1 — Researcher", expanded=False):
                st.write(stages["draft_answer"])
            if stages.get("skipped_verification"):
                st.caption("Stages 2+3 skipped — Researcher found no relevant information to verify.")
            else:
                with st.expander("Stage 2+3 — Verifier & Contradiction check", expanded=False):
                    st.write("**Verification:**")
                    st.write(stages["verification"])
                    st.write("**Contradiction check:**")
                    st.write(stages["contradiction_report"])
        st.write(message["content"])
        if "latency" in message:
            if message.get("guardrail_triggered"):
                st.caption(
                    f"{message['latency']:.2f}s | LLM: {message['engine']} | "
                    f"Correctly declined — no relevant information found"
                )
            else:
                conf_part = f" | Confidence: {message['confidence_str']}" if message.get("confidence_str") else ""
                st.caption(
                    f"{message['latency']:.2f}s | LLM: {message['engine']} | "
                    f"Embed: {message.get('embed', EMBED_MODEL_FOLDER)}{conf_part}"
                )

# ---------------------------------------------------------------------------
# CHAT INPUT — greetings/farewells + RAG answer, ported exactly from client.py
# ---------------------------------------------------------------------------
question = st.chat_input(
    "Ask a question regarding the loaded documents...",
    disabled=st.session_state.generating
)

if not st.session_state.generating and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    if st.session_state.generating:
        st.session_state.pending_question = question
        st.info("Still generating — your question has been queued.")
    else:
        with st.chat_message("user", avatar="🧑\u200d💻"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        q_lower = question.lower().strip()
        with st.chat_message("assistant", avatar="🖥️"):
            if q_lower in GREETINGS:
                answer = "Hello. I am your local offline research assistant. How can I help you today?"
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif q_lower in FAREWELLS:
                answer = "Goodbye! Feel free to return whenever you have more questions."
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            else:
                vectorstore = load_vectorstore()
                response_area = st.empty()

                if vectorstore is None:
                    response_area.warning("No index built yet. Upload documents and click Rebuild Vector Index in the sidebar.")
                else:
                    if st.session_state.get("just_rebuilt"):
                        st.session_state.just_rebuilt = False

                    st.session_state.generating = True
                    try:
                        start = time.time()

                        with st.status("Researching...", expanded=False) as status1:
                            draft_answer, researcher_chunks = run_researcher(vectorstore, question, selected_model)
                            st.write(draft_answer)
                            status1.update(label="Stage 1 — Researcher: draft complete", state="complete")

                        guardrail_triggered = GUARDRAIL_PHRASE in draft_answer

                        if guardrail_triggered:
                            # No relevant context found — skip Verifier/Contradiction, nothing to check.
                            final_answer = draft_answer
                            verification = None
                            contradiction_report = None
                            st.caption("Stages 2+3 skipped — no relevant information found to verify.")
                        else:
                            # Stage 2 (Verifier) and Stage 3 (Contradiction check) both only need
                            # Researcher's output, not each other — run them concurrently.
                            with st.status("Verifying claims + checking contradictions...", expanded=False) as status23:
                                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                                    future_verify = executor.submit(run_verifier, vectorstore, draft_answer, selected_model)
                                    future_contradict = executor.submit(run_contradiction_check, researcher_chunks[:5], selected_model)
                                    verification, verifier_chunks, grounding_score = future_verify.result()
                                    contradiction_report = future_contradict.result()
                                st.write("**Verification:**")
                                st.write(verification)
                                st.write("**Contradiction check:**")
                                st.write(contradiction_report)
                                status23.update(label="Stage 2+3 — Verifier & Contradiction check: complete", state="complete")

                            with st.status("Synthesizing final report...", expanded=True) as status4:
                                final_answer = run_synthesizer(question, draft_answer, verification, contradiction_report, selected_model, grounding_score)
                                status4.update(label="Stage 4 — Synthesizer: report ready", state="complete")

                        if guardrail_triggered:
                            confidence_str = None
                            sources_list = []
                        else:
                            final_answer, confidence_str = extract_and_strip_confidence(final_answer)
                            # Safety override — if the objective math check found weak grounding but the LLM
                            # still self-reported High, force it down. Don't just trust the prompt to comply.
                            if grounding_score is not None and grounding_score < 0.35 and confidence_str and "High" in confidence_str:
                                confidence_str = f"Low ({grounding_score:.2f} grounding — overridden from self-reported High)"
                            sources_list = extract_sources(researcher_chunks[:3])
                            if sources_list:
                                final_answer += "\n\n**Sources:** " + ", ".join(sources_list)

                        response_area.markdown(final_answer)

                        latency = time.time() - start
                        if guardrail_triggered:
                            st.caption(f"{latency:.2f}s | LLM: {selected_model} | Correctly declined — no relevant information found")
                        else:
                            conf_part = f" | Confidence: {confidence_str}" if confidence_str else ""
                            st.caption(f"{latency:.2f}s total | LLM: {selected_model} | 4-stage pipeline | Embed: {EMBED_MODEL_FOLDER}{conf_part}")

                        st.session_state.messages.append({
                            "role": "assistant", "content": final_answer,
                            "latency": latency, "engine": selected_model, "embed": EMBED_MODEL_FOLDER,
                            "guardrail_triggered": guardrail_triggered,
                            "confidence_str": confidence_str,
                            "stages": {
                                "draft_answer": draft_answer,
                                "verification": verification,
                                "contradiction_report": contradiction_report,
                                "skipped_verification": guardrail_triggered,
                            },
                        })

                    except Exception as e:
                        error_message = f"Something went wrong while generating an answer: {e}"
                        response_area.markdown(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

                    finally:
                        st.session_state.generating = False