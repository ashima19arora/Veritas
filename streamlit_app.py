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
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import ollama as ollama_client

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

st.set_page_config(page_title="VERITAS", layout="centered")

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
    .system-caption { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.8rem; color: #888; }
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
    import torch as _torch
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    local_path = os.path.join("models", EMBED_MODEL_FOLDER)
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

def index_exists():
    return os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss"))

def load_and_chunk_documents():
    """Ported exactly from server.py's load_and_chunk_documents() — skips hidden/Mac files, skips unreadable files."""
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
            chunks.extend(splitter.split_documents(docs))
        except Exception:
            skipped.append(fname)
    return chunks

def rebuild_index():
    """Rebuild FAISS index from all docs in DOCS_DIR. Returns True on success, False if no docs (matches server.py's rebuild_all_indexes contract)."""
    chunks = load_and_chunk_documents()
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
        embeddings = get_embeddings()
        vs = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
        st.session_state.vectorstore = vs
        return vs
    return None

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
            ok = rebuild_index()
            if ok:
                st.session_state.just_rebuilt = True
                st.session_state.last_rebuilt_time = time.time()
                st.session_state.pending_deletes = set()
            else:
                st.warning("No documents found to index.")
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
        f'<div class="system-caption">Offline Research & Fact-Verification System — {selected_model} | '
        f'Embed: {EMBED_MODEL_FOLDER} | Engine: FAISS</div>',
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
# CHAT HISTORY
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑\u200d💻" if message["role"] == "user" else "🖥️"):
        st.write(message["content"])
        if "latency" in message:
            st.caption(
                f"{message['latency']:.2f}s | LLM: {message['engine']} | "
                f"Embed: {message.get('embed', EMBED_MODEL_FOLDER)}"
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
                status_area   = st.empty()

                if vectorstore is None:
                    response_area.warning("No index built yet. Upload documents and click Rebuild Vector Index in the sidebar.")
                else:
                    if st.session_state.get("just_rebuilt"):
                        response_area.markdown("_Thinking... (first question after rebuild — slightly slower)_")
                        st.session_state.just_rebuilt = False
                    else:
                        response_area.markdown("_Thinking... (Scanning Vector Nodes)_")

                    st.session_state.generating = True
                    try:
                        start = time.time()

                        raw_docs = vectorstore.similarity_search(question, k=6)
                        context = "\n\n---\n\n".join([d.page_content for d in raw_docs])

                        # PROMPT — kept exact, word-for-word from VEDA's proven standard-mode prompt
                        prompt_template = """You are an elite, highly accurate document analysis system.
Review the provided context carefully to address the query.

Operational Rules:
1. Rely ONLY on facts directly mentioned in the Context below. Do not extrapolate or hallucinate.
2. If the context contains relevant information, answer it directly and completely. Do not append disclaimers after a complete answer.
3. If and ONLY IF the context contains NO relevant information at all, respond with: "I cannot find sufficient verified data within the loaded documents." — nothing else.
4. Keep your analysis concise, structured, and objective.

Context:
{context}

Question:
{question}

Answer:"""
                        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                        llm = OllamaLLM(model=selected_model, temperature=0.2)
                        chain = prompt | llm | StrOutputParser()

                        accumulated = ""
                        for chunk in chain.stream({"context": context, "question": question}):
                            accumulated += chunk
                            response_area.markdown(accumulated)
                            status_area.caption(f"Reasoning via {selected_model}... {time.time() - start:.0f}s")

                        if not accumulated:
                            accumulated = "_(No response received from Ollama. Check that the model is running.)_"
                            response_area.markdown(accumulated)

                        latency = time.time() - start
                        status_area.caption(f"{latency:.2f}s | LLM: {selected_model} | Embed: {EMBED_MODEL_FOLDER}")

                        st.session_state.messages.append({
                            "role": "assistant", "content": accumulated,
                            "latency": latency, "engine": selected_model, "embed": EMBED_MODEL_FOLDER,
                        })

                    except Exception as e:
                        error_message = f"Something went wrong while generating an answer: {e}"
                        response_area.markdown(error_message)
                        status_area.empty()
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

                    finally:
                        st.session_state.generating = False