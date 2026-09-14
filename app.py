import os
import re
import html
from urllib.parse import urlparse, parse_qs

import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    IpBlocked,
    RequestBlocked,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ============================================================
# CHANGED: MISTRAL → HUGGING FACE
# ============================================================

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

from langchain_core.prompts import PromptTemplate


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()


# ============================================================
# API KEY
# ============================================================

# CHANGED: MISTRAL_API_KEY → HF_TOKEN

def get_api_key():

    key = os.getenv("HF_TOKEN")

    if key:
        return key

    try:
        key = st.secrets.get("HF_TOKEN")

        if key:
            return key

    except Exception:
        pass

    return None


HF_TOKEN = get_api_key()

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0f0f0f;
        color: white;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    header {
        background: transparent !important;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background: #181818;
        border-right: 1px solid #292929;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 25px;
    }

    .sidebar-logo {
        width: 44px;
        height: 31px;
        background: #ff0000;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.35);
    }

    .sidebar-play {
        width: 0;
        height: 0;
        border-top: 7px solid transparent;
        border-bottom: 7px solid transparent;
        border-left: 11px solid white;
        margin-left: 3px;
    }

    .sidebar-title {
        font-size: 19px;
        font-weight: 800;
    }

    .sidebar-heading {
        color: #ff4444 !important;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .sidebar-text {
        color: #aaaaaa !important;
        font-size: 13px;
        line-height: 1.6;
    }

    .pipeline-item {
        background: #222222;
        border: 1px solid #303030;
        border-radius: 8px;
        padding: 9px 11px;
        margin: 6px 0;
        color: #dddddd !important;
        font-size: 12px;
    }

    .pipeline-item:hover {
        border-color: #ff0000;
    }

    .tech-box {
        background: #222222;
        border: 1px solid #303030;
        border-radius: 12px;
        padding: 12px;
    }

    .tech-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid #303030;
        font-size: 11px;
    }

    .tech-row:last-child {
        border-bottom: none;
    }

    .tech-label {
        color: #999999 !important;
    }

    .tech-value {
        color: #ff4444 !important;
        font-weight: 700;
        text-align: right;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        text-align: center;
        padding: 15px 20px 40px;
    }

    .youtube-logo {
        width: 130px;
        height: 88px;
        background: #ff0000;
        border-radius: 24px;
        margin: 0 auto 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        box-shadow:
            0 0 30px rgba(255, 0, 0, 0.45),
            0 0 80px rgba(255, 0, 0, 0.15);
        animation: floatLogo 3s ease-in-out infinite;
    }

    .youtube-logo::before {
        content: "";
        position: absolute;
        inset: -8px;
        border: 2px solid rgba(255, 0, 0, 0.3);
        border-radius: 30px;
        animation: pulseLogo 2s ease-in-out infinite;
    }

    .play-icon {
        width: 0;
        height: 0;
        border-top: 19px solid transparent;
        border-bottom: 19px solid transparent;
        border-left: 31px solid white;
        margin-left: 7px;
    }

    .hero-title {
        font-size: clamp(35px, 5vw, 58px);
        font-weight: 900;
        letter-spacing: -2px;
        color: white;
        margin-bottom: 8px;
    }

    .hero-title span {
        color: #ff0000;
    }

    .hero-subtitle {
        color: #aaaaaa;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 15px;
        border-radius: 30px;
        background: #17251c;
        border: 1px solid #285333;
        color: #67d98a;
        font-size: 12px;
        font-weight: 700;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        animation: statusPulse 1.5s infinite;
    }


    /* ========================================================
       SECTIONS
       ======================================================== */

    .section-card {
        background: #181818;
        border: 1px solid #292929;
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 22px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);
    }

    .youtube-card {
        border-top: 3px solid #ff0000;
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: white;
        margin-bottom: 7px;
    }

    .section-description {
        font-size: 13px;
        color: #999999;
        line-height: 1.6;
        margin-bottom: 18px;
    }


    /* ========================================================
       INPUT
       ======================================================== */

    .stTextInput input {
        background: #101010 !important;
        color: white !important;
        border: 1px solid #363636 !important;
        border-radius: 12px !important;
    }

    .stTextInput input:focus {
        border-color: #ff0000 !important;
        box-shadow: 0 0 0 1px #ff0000 !important;
    }

    .stTextInput label {
        color: #bbbbbb !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;
        min-height: 43px;
        background: #ff0000 !important;
        color: white !important;
        border: none !important;
        border-radius: 11px !important;
        font-weight: 800 !important;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #cc0000 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 0, 0, 0.25);
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    [data-testid="stChatInput"] {
        background: #181818 !important;
        border: 1px solid #363636 !important;
        border-radius: 15px !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #101010 !important;
        color: white !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #777777 !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        box-shadow: 0 0 0 1px #ff0000 !important;
        border-color: #ff0000 !important;
    }

    [data-testid="stChatInput"] button {
        background: #ff0000 !important;
        color: white !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: #cc0000 !important;
    }


    /* ========================================================
       INFO CARDS
       ======================================================== */

    .info-card {
        min-height: 135px;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #292929;
        transition: 0.25s ease;
        margin-bottom: 15px;
    }

    .info-card:hover {
        transform: translateY(-5px);
        border-color: #ff0000;
        box-shadow: 0 10px 30px rgba(255, 0, 0, 0.12);
    }

    .blue-info {
        background: #111c2b;
    }

    .green-info {
        background: #11251a;
    }

    .purple-info {
        background: #1d1628;
    }

    .orange-info {
        background: #291e12;
    }

    .info-icon {
        font-size: 24px;
        margin-bottom: 12px;
    }

    .info-title {
        color: #999999;
        font-size: 12px;
        font-weight: 700;
    }

    .info-value {
        color: white;
        font-size: 19px;
        font-weight: 800;
        margin-top: 5px;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    .question-box {
        background: #202020;
        border: 1px solid #303030;
        border-radius: 15px;
        padding: 17px;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .answer-box {
        background: #181818;
        border: 1px solid #292929;
        border-left: 4px solid #ff0000;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        color: #e5e5e5;
        line-height: 1.7;
        animation: fadeIn 0.4s ease;
    }

    .chat-label {
        color: #ff4444;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .source-box {
        background: #111111;
        border: 1px solid #292929;
        border-radius: 10px;
        padding: 15px;
        color: #aaaaaa;
        font-size: 12px;
        line-height: 1.6;
    }

    [data-testid="stExpander"] {
        background: #151515;
        border: 1px solid #292929;
        border-radius: 12px;
    }

    [data-testid="stProgressBar"] > div > div > div {
        background: #ff0000 !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .custom-footer {
        text-align: center;
        color: #666666;
        font-size: 12px;
        padding: 40px 0 10px;
        border-top: 1px solid #222222;
        margin-top: 40px;
    }

    .custom-footer span {
        color: #ff0000;
    }


    /* ========================================================
       ANIMATIONS
       ======================================================== */

    @keyframes floatLogo {

        0%, 100% {
            transform: translateY(0);
        }

        50% {
            transform: translateY(-9px);
        }
    }

    @keyframes pulseLogo {

        0%, 100% {
            transform: scale(1);
            opacity: 0.45;
        }

        50% {
            transform: scale(1.08);
            opacity: 0.1;
        }
    }

    @keyframes statusPulse {

        0% {
            box-shadow:
                0 0 0 0 rgba(34, 197, 94, 0.5);
        }

        70% {
            box-shadow:
                0 0 0 8px rgba(34, 197, 94, 0);
        }

        100% {
            box-shadow:
                0 0 0 0 rgba(34, 197, 94, 0);
        }
    }

    @keyframes fadeIn {

        from {
            opacity: 0;
            transform: translateY(8px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():

    defaults = {
        "vector_store": None,
        "transcript": "",
        "chunks": [],
        "video_id": "",
        "chat_history": [],
        "processing_status": "Waiting",
        "embeddings_ready": False,
        "ai_ready": False,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# ============================================================
# YOUTUBE VIDEO ID
# ============================================================

def extract_video_id(url):

    if not url:
        return None

    url = url.strip()

    if re.fullmatch(
        r"[A-Za-z0-9_-]{11}",
        url
    ):
        return url

    try:

        parsed = urlparse(url)

        hostname = parsed.netloc.lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        if hostname in [
            "youtube.com",
            "m.youtube.com"
        ]:

            query = parse_qs(
                parsed.query
            )

            if "v" in query:

                video_id = query["v"][0]

                if re.fullmatch(
                    r"[A-Za-z0-9_-]{11}",
                    video_id
                ):
                    return video_id

            parts = parsed.path.strip(
                "/"
            ).split("/")

            if (
                len(parts) >= 2
                and parts[0] in [
                    "embed",
                    "shorts"
                ]
            ):

                video_id = parts[1]

                if re.fullmatch(
                    r"[A-Za-z0-9_-]{11}",
                    video_id
                ):
                    return video_id

        if hostname == "youtu.be":

            video_id = parsed.path.strip(
                "/"
            ).split("/")[0]

            if re.fullmatch(
                r"[A-Za-z0-9_-]{11}",
                video_id
            ):
                return video_id

    except Exception:

        return None

    return None


# ============================================================
# TRANSCRIPT
# ============================================================

def fetch_transcript(video_id):

    try:

        transcript_data = (
            YouTubeTranscriptApi().fetch(
                video_id
            )
        )

        transcript = " ".join(
            item.text
            for item in transcript_data
        )

        if not transcript.strip():

            return (
                None,
                "The transcript is empty."
            )

        return transcript, None

    except TranscriptsDisabled:

        return (
            None,
            "Transcripts are disabled for this video."
        )

    except NoTranscriptFound:

        return (
            None,
            "No transcript was found for this video."
        )

    except IpBlocked:

        return (
            None,
            "YouTube is blocking transcript requests "
            "from this server."
        )

    except RequestBlocked:

        return (
            None,
            "YouTube blocked the transcript request."
        )

    except Exception as e:

        return (
            None,
            "Transcript could not be retrieved. "
            f"Error: {type(e).__name__}: {e}"
        )


# ============================================================
# VECTOR STORE
# ============================================================

def create_vector_store(transcript):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )

    chunks = splitter.create_documents(
        [transcript]
    )

    # ========================================================
    # CHANGED:
    # MistralAIEmbeddings → HuggingFaceEmbeddings
    # ========================================================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    return vector_store, chunks


# ============================================================
# RETRIEVER
# ============================================================

def get_retriever(vector_store):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        },
    )


# ============================================================
# HUGGING FACE LLM
# ============================================================

def get_llm():

    # ========================================================
    # DeepSeek-V4.1-Flash through Hugging Face
    # ========================================================

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.2,
        max_new_tokens=1024,
    )

    llm = ChatHuggingFace(
        llm=llm_endpoint
    )

    return llm


# ============================================================
# LLM ANSWER
# ============================================================

def generate_answer(question, results):

    context = "\n\n".join(
        document.page_content
        for document in results
    )

    prompt = PromptTemplate(

        template="""
You are a helpful AI assistant answering questions
about a YouTube video.

Use ONLY the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Answer clearly and concisely.
- Use only the provided context.
- Do not make up information.
- Do not use outside knowledge.
- If the answer is not available in the context, say:
  "I don't know based on the provided context."

Answer:
""",

        input_variables=[
            "context",
            "question"
        ],
    )

    final_prompt = prompt.format(
        context=context,
        question=question,
    )

    # ========================================================
    # CHANGED:
    # ChatMistralAI → Hugging Face DeepSeek
    # ========================================================

    llm = get_llm()

    response = llm.invoke(
        final_prompt
    )

    return response.content


# ============================================================
# HERO
# ============================================================

def render_hero():

    st.html(
        """
        <div class="hero">

            <div class="youtube-logo">

                <div class="play-icon"></div>

            </div>

            <div class="hero-title">
                YouTube <span>AI Assistant</span>
            </div>

            <div class="hero-subtitle">
                Ask questions. Understand videos.
                Get intelligent answers.
            </div>

            <div class="status">

                <span class="status-dot"></span>

                AI System Ready

            </div>

        </div>
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.html(
            """
            <div class="sidebar-brand">

                <div class="sidebar-logo">
                    <div class="sidebar-play"></div>
                </div>

                <div class="sidebar-title">
                    YouTube AI
                </div>

            </div>
            """
        )

        st.html(
            """
            <div class="sidebar-heading">
                About
            </div>

            <div class="sidebar-text">
                An AI-powered YouTube Transcript Q&A
                application using Retrieval Augmented
                Generation.
            </div>
            """
        )

        st.html(
            """
            <div class="sidebar-heading">
                RAG Pipeline
            </div>

            <div class="pipeline-item">
                1. YouTube Transcript
            </div>

            <div class="pipeline-item">
                2. Text Splitting
            </div>

            <div class="pipeline-item">
                3. Hugging Face Embeddings
            </div>

            <div class="pipeline-item">
                4. FAISS Vector Store
            </div>

            <div class="pipeline-item">
                5. Similarity Retrieval
            </div>

            <div class="pipeline-item">
                6. DeepSeek-V4.1-Flash
            </div>

            <div class="pipeline-item">
                7. Final Answer
            </div>
            """
        )

        st.html(
            """
            <div class="sidebar-heading">
                Model Configuration
            </div>

            <div class="tech-box">

                <div class="tech-row">

                    <span class="tech-label">
                        LLM
                    </span>

                    <span class="tech-value">
                        DeepSeek-V4.1-Flash
                    </span>

                </div>

                <div class="tech-row">

                    <span class="tech-label">
                        Embedding
                    </span>

                    <span class="tech-value">
                        all-MiniLM-L6-v2
                    </span>

                </div>

                <div class="tech-row">

                    <span class="tech-label">
                        Vector DB
                    </span>

                    <span class="tech-value">
                        FAISS
                    </span>

                </div>

            </div>
            """
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # CLEAR CHAT
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
        ):

            st.session_state.chat_history = []

            st.rerun()

        # ----------------------------------------------------
        # RESET VIDEO
        # ----------------------------------------------------

        if st.button(
            "🔄 Reset Video",
            use_container_width=True,
        ):

            st.session_state.vector_store = None
            st.session_state.transcript = ""
            st.session_state.chunks = []
            st.session_state.video_id = ""
            st.session_state.chat_history = []

            st.session_state.processing_status = "Waiting"

            st.session_state.embeddings_ready = False
            st.session_state.ai_ready = False

            st.rerun()


# ============================================================
# VIDEO PROCESSING
# ============================================================

def render_video_section():

    st.html(
        """
        <div class="section-card youtube-card">

            <div class="section-title">
                🎬 Process YouTube Video
            </div>

            <div class="section-description">
                Paste a YouTube video URL and convert its
                transcript into an AI-searchable knowledge base.
            </div>

        </div>
        """
    )

    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
        key="video_url_input",
    )

    col1, col2 = st.columns(
        [1, 3]
    )

    with col1:

        process_button = st.button(
            "🚀 Process Video",
            use_container_width=True,
        )

    with col2:

        if st.session_state.video_id:

            video_id = html.escape(
                st.session_state.video_id
            )

            st.html(
                f"""
                <div style="
                    padding:11px 15px;
                    background:#191919;
                    border:1px solid #292929;
                    border-radius:11px;
                    color:#999999;
                    font-size:12px;
                ">

                    Current Video ID:

                    <strong style="
                        color:#ff4444;
                    ">
                        {video_id}
                    </strong>

                </div>
                """
            )

    if not process_button:
        return

    if not video_url.strip():

        st.warning(
            "Please paste a YouTube video URL."
        )

        return

    video_id = extract_video_id(
        video_url
    )

    if not video_id:

        st.error(
            "Invalid YouTube URL. Please enter a valid URL."
        )

        return

    # ========================================================
    # CHANGED:
    # MISTRAL_API_KEY → HF_TOKEN
    # ========================================================

    if not HF_TOKEN:

        st.error(
            "HF_TOKEN was not found."
        )

        st.info(
            "For Streamlit Cloud, add HF_TOKEN "
            "under App Settings → Secrets."
        )

        return

    progress = st.progress(0)

    status = st.empty()

    try:

        # ----------------------------------------------------
        # 1. TRANSCRIPT
        # ----------------------------------------------------

        status.info(
            "📥 Fetching YouTube transcript..."
        )

        progress.progress(20)

        transcript, error = fetch_transcript(
            video_id
        )

        if error:

            progress.empty()
            status.empty()

            st.error(error)

            return

        # ----------------------------------------------------
        # 2. CHUNKING
        # ----------------------------------------------------

        status.info(
            "✂️ Splitting transcript into chunks..."
        )

        progress.progress(40)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
        )

        chunks = splitter.create_documents(
            [transcript]
        )

        # ----------------------------------------------------
        # 3. EMBEDDINGS
        # ----------------------------------------------------

        status.info(
            "🧠 Creating Hugging Face embeddings..."
        )

        progress.progress(60)

        # ====================================================
        # CHANGED:
        # MistralAIEmbeddings → HuggingFaceEmbeddings
        # ====================================================

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ----------------------------------------------------
        # 4. FAISS
        # ----------------------------------------------------

        status.info(
            "📚 Building FAISS vector database..."
        )

        progress.progress(80)

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        # ----------------------------------------------------
        # 5. SAVE
        # ----------------------------------------------------

        st.session_state.vector_store = vector_store
        st.session_state.transcript = transcript
        st.session_state.chunks = chunks
        st.session_state.video_id = video_id
        st.session_state.processing_status = "Loaded"
        st.session_state.embeddings_ready = True
        st.session_state.ai_ready = True

        # New video = new conversation
        st.session_state.chat_history = []

        progress.progress(100)

        status.success(
            "✅ Video processed successfully!"
        )

    except Exception as e:

        progress.empty()
        status.empty()

        st.error(
            "Something went wrong while processing the video."
        )

        st.exception(e)


# ============================================================
# INFO CARDS
# ============================================================

def render_info_cards():

    transcript_status = (
        "Loaded"
        if st.session_state.transcript
        else "Waiting"
    )

    chunk_count = len(
        st.session_state.chunks
    )

    embedding_status = (
        "Ready"
        if st.session_state.embeddings_ready
        else "Waiting"
    )

    ai_status = (
        "Ready"
        if st.session_state.ai_ready
        else "Waiting"
    )

    columns = st.columns(4)

    cards = [

        (
            columns[0],
            "📄",
            "Transcript",
            transcript_status,
            "blue-info",
        ),

        (
            columns[1],
            "🧩",
            "Chunks",
            str(chunk_count),
            "green-info",
        ),

        (
            columns[2],
            "🤖",
            "Embeddings",
            embedding_status,
            "purple-info",
        ),

        (
            columns[3],
            "⚡",
            "AI Response",
            ai_status,
            "orange-info",
        ),
    ]

    for column, icon, title, value, css_class in cards:

        with column:

            st.html(
                f"""
                <div class="info-card {css_class}">

                    <div class="info-icon">
                        {icon}
                    </div>

                    <div class="info-title">
                        {html.escape(title)}
                    </div>

                    <div class="info-value">
                        {html.escape(value)}
                    </div>

                </div>
                """
            )


# ============================================================
# CHAT HISTORY
# ============================================================

def render_chat_history():

    if not st.session_state.chat_history:

        st.html(
            """
            <div style="
                text-align:center;
                color:#555555;
                padding:30px;
                font-size:13px;
            ">
                💭 Your conversation will appear here.
            </div>
            """
        )

        return

    st.html(
        """
        <div class="section-title">
            💬 Conversation
        </div>
        """
    )

    for chat in st.session_state.chat_history:

        question = html.escape(
            str(chat["question"])
        )

        answer = html.escape(
            str(chat["answer"])
        ).replace(
            "\n",
            "<br>"
        )

        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        st.html(
            f"""
            <div class="question-box">

                <div class="chat-label">
                    🧑 YOU
                </div>

                <div style="color:white;">
                    {question}
                </div>

            </div>
            """
        )

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.html(
            f"""
            <div class="answer-box">

                <div class="chat-label">
                    🤖 AI ASSISTANT
                </div>

                <div>
                    {answer}
                </div>

            </div>
            """
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        with st.expander(
            "📚 View Retrieved Context"
        ):

            for index, source in enumerate(
                chat["sources"]
            ):

                st.markdown(
                    f"**Source {index + 1}**"
                )

                safe_source = html.escape(
                    str(source)
                ).replace(
                    "\n",
                    "<br>"
                )

                st.html(
                    f"""
                    <div class="source-box">
                        {safe_source}
                    </div>
                    """
                )


# ============================================================
# ASK QUESTION
# ============================================================

def render_question_section():

    # --------------------------------------------------------
    # Don't show chat input until a video is processed.
    # --------------------------------------------------------

    if st.session_state.vector_store is None:

        return

    # --------------------------------------------------------
    # Native Streamlit Chat Input
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about this video..."
    )

    if question is None:

        return

    question = question.strip()

    if not question:

        return

    # ========================================================
    # CHANGED:
    # MISTRAL_API_KEY → HF_TOKEN
    # ========================================================

    if not HF_TOKEN:

        st.error(
            "HF_TOKEN was not found."
        )

        return

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    try:

        with st.spinner(
            "🧠 Searching transcript and generating answer..."
        ):

            retriever = get_retriever(
                st.session_state.vector_store
            )

            results = retriever.invoke(
                question
            )

            answer = generate_answer(
                question,
                results
            )

            sources = [
                document.page_content
                for document in results
            ]

            # ------------------------------------------------
            # SAVE CHAT
            # ------------------------------------------------

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                }
            )

        # ----------------------------------------------------
        # RERUN
        # ----------------------------------------------------

        st.rerun()

    except Exception as e:

        st.error(
            "Unable to generate the answer."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    st.html(
        """
        <div class="custom-footer">

            Built with <span>♥</span>
            using Streamlit + LangChain + Hugging Face + FAISS

        </div>
        """
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    render_sidebar()

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    render_hero()

    # --------------------------------------------------------
    # VIDEO PROCESSING
    # --------------------------------------------------------

    render_video_section()

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # INFO CARDS
    # --------------------------------------------------------

    render_info_cards()

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    render_chat_history()

    # --------------------------------------------------------
    # EXTRA BOTTOM SPACE
    # --------------------------------------------------------

    if st.session_state.vector_store is not None:

        st.markdown(
            """
            <div style="height:100px;"></div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    render_question_section()

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    render_footer()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

