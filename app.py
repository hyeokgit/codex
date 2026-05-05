import os
import tempfile
from typing import List

import chromadb
import fitz  # PyMuPDF
import google.generativeai as genai
import streamlit as st
from sentence_transformers import SentenceTransformer

# -----------------------------
# Configuration
# -----------------------------
CHROMA_DIR = "./chroma_data"
COLLECTION_NAME = "local_docs"
EMBED_MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5


@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        full_text = []
        for page in doc:
            full_text.append(page.get_text("text"))
        return "\n".join(full_text).strip()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def index_pdf(filename: str, file_bytes: bytes):
    text = extract_text_from_pdf(file_bytes)
    if not text:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다.")

    chunks = chunk_text(text)
    embedder = get_embedder()
    vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()

    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

    collection = get_chroma_collection()
    collection.upsert(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)

    return len(chunks)


def retrieve_context(question: str, k: int = TOP_K) -> List[str]:
    embedder = get_embedder()
    q_vector = embedder.encode([question], normalize_embeddings=True).tolist()[0]

    collection = get_chroma_collection()
    result = collection.query(query_embeddings=[q_vector], n_results=k)

    docs = result.get("documents", [[]])[0]
    return docs


def ask_gemini(question: str, contexts: List[str]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경 변수를 설정해주세요.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    context_text = "\n\n".join(contexts)
    prompt = f"""
너는 로컬 RAG 시스템의 답변 어시스턴트다.
아래 [문맥]만 근거로 답하고, 근거가 부족하면 모른다고 답해.

[문맥]
{context_text}

[질문]
{question}
""".strip()

    response = model.generate_content(prompt)
    return response.text


def main():
    st.set_page_config(page_title="Local-First RAG", layout="wide")
    st.title("📚 Local-First RAG (Gemini + Local Embedding + Chroma)")
    st.caption("파일은 로컬에서 처리하고, 정제된 문맥만 Gemini API로 전송합니다.")

    with st.sidebar:
        st.header("설정")
        st.write(f"Embedding: `{EMBED_MODEL_NAME}`")
        st.write(f"Vector DB: `ChromaDB` ({CHROMA_DIR})")
        st.write("LLM: `gemini-2.5-flash` (API 호출)")

    uploaded_files = st.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("로컬 인덱싱 시작"):
        if not uploaded_files:
            st.warning("업로드된 파일이 없습니다.")
        else:
            with st.spinner("PDF 처리 및 임베딩 생성 중..."):
                total_chunks = 0
                for f in uploaded_files:
                    chunk_count = index_pdf(f.name, f.read())
                    total_chunks += chunk_count
            st.success(f"인덱싱 완료: {len(uploaded_files)}개 파일, {total_chunks}개 청크 저장")

    st.divider()
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기") and question.strip():
        with st.spinner("로컬 검색 + Gemini 답변 생성 중..."):
            contexts = retrieve_context(question, TOP_K)
            if not contexts:
                st.error("검색된 문맥이 없습니다. 먼저 문서를 인덱싱하세요.")
                return

            answer = ask_gemini(question, contexts)

        st.subheader("답변")
        st.write(answer)

        with st.expander("검색된 문맥 보기"):
            for i, c in enumerate(contexts, start=1):
                st.markdown(f"**Context {i}**")
                st.write(c)


if __name__ == "__main__":
    main()
