<div align="center">

# 🎥 YouTube AI Assistant

### 🤖 Ask Questions. Understand Videos. Get Intelligent Answers.
live:-https://readmemd-pnt3rj954vhyu8efywognn.streamlit.app/

An AI-powered YouTube Transcript Q&A application built using  
**RAG, LangChain, Mistral AI, FAISS, and Streamlit.**

<p>
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-rag-pipeline">RAG Pipeline</a>
</p>

</div>

---

## 📸 Project Preview

<p align="center">
  <img src="assets/Screenshot.png" alt="YouTube AI Assistant" width="900">
</p>

---


</p>

The application allows users to paste a YouTube video URL and ask questions directly about the video's content.

---

## 📌 Overview

**YouTube AI Assistant** is a Retrieval-Augmented Generation (RAG) application that allows users to interact with YouTube videos using natural language.

Instead of watching an entire video to find a specific piece of information, users can simply enter the YouTube URL and ask questions.

The application:

**YouTube Video → Transcript → Text Chunks → Embeddings → FAISS → Relevant Context → Mistral AI → Answer**

This project demonstrates how modern **Generative AI, Vector Search, Embeddings, and RAG** can be combined to build a practical AI application.

---

## 🚀 Features

- 🎥 YouTube video URL processing
- 📝 Automatic transcript extraction
- ✂️ Intelligent transcript chunking
- 🧠 Mistral AI embeddings
- 🔎 FAISS vector similarity search
- 🤖 Mistral AI-powered answers
- 💬 Interactive question-answer interface
- 📚 Context-aware responses
- ⚡ Fast retrieval using vector search
- 🎨 Modern YouTube-inspired UI
- 🌙 Dark-themed interface
- 🔐 Secure API key management using environment variables
- 🛡️ Answers generated using retrieved video context

---

## 🏗️ Architecture

```text
                 ┌──────────────────────┐
                 │   YouTube Video URL  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Extract Video ID   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Fetch Transcript    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Plain Text         │
                 │   Transcript         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Text Chunking       │
                 │  1500 / 300          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Mistral AI Embedding │
                 │    mistral-embed     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    FAISS Vector      │
                 │       Store          │
                 └──────────┬───────────┘
                            │
                       User Question
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Similarity Search   │
                 │       Top K = 5      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Relevant Transcript  │
                 │       Context        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Mistral AI      │
                 │ mistral-small-latest │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     Final Answer     │
                 └──────────────────────┘
```

---

## 🧠 How It Works

The application uses a **Retrieval-Augmented Generation (RAG)** architecture.

### 1️⃣ Enter YouTube URL

The user enters the URL of a YouTube video.

Example:

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

The application extracts the video ID from the URL.

---

### 2️⃣ Fetch Transcript

The application uses `youtube-transcript-api` to retrieve the available transcript.

The transcript is then converted into plain text.

```text
Transcript Chunks
        ↓
Plain Text
```

---

### 3️⃣ Text Chunking

Long transcripts are divided into smaller chunks using:

**RecursiveCharacterTextSplitter**

Current configuration:

```text
Chunk Size   : 1500
Chunk Overlap: 300
```

The overlap helps maintain contextual continuity between adjacent chunks.

---

### 4️⃣ Generate Embeddings

Each text chunk is converted into a vector representation using:

```text
Mistral AI
Model: mistral-embed
```

These embeddings allow the system to compare the semantic meaning of the user's question with transcript chunks.

---

### 5️⃣ Store in FAISS

The generated embeddings are stored in a **FAISS vector store**.

FAISS enables efficient similarity search over the transcript embeddings.

---

### 6️⃣ Retrieve Relevant Context

When the user asks a question, the system searches the FAISS vector store.

The retriever configuration is:

```text
Search Type: Similarity
Top K: 5
```

The five most relevant transcript chunks are retrieved.

---

### 7️⃣ Generate Answer

The retrieved context is passed to:

```text
Mistral AI
Model: mistral-small-latest
```

The model generates the final response based on the relevant transcript context.

---

# 🧩 RAG Pipeline

The complete RAG workflow can be summarized as:

```text
YouTube URL
     ↓
Extract Video ID
     ↓
Fetch Transcript
     ↓
Convert to Plain Text
     ↓
RecursiveCharacterTextSplitter
     ↓
Text Chunks
     ↓
Mistral Embeddings
     ↓
FAISS Vector Store
     ↓
User Question
     ↓
Similarity Search
     ↓
Top 5 Relevant Chunks
     ↓
Context + Question
     ↓
Mistral LLM
     ↓
Final Answer
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| 🐍 **Python** | Core programming language |
| 🎨 **Streamlit** | Web application and UI |
| 🦜 **LangChain** | RAG pipeline and LLM integration |
| 🤖 **Mistral AI** | Embeddings and AI-generated responses |
| 🔎 **FAISS** | Vector similarity search |
| 📜 **YouTube Transcript API** | YouTube transcript extraction |
| 🔐 **python-dotenv** | Environment variable management |

---

## ⚙️ RAG Configuration

| Component | Configuration |
|-----------|---------------|
| Text Splitter | RecursiveCharacterTextSplitter |
| Chunk Size | 1500 |
| Chunk Overlap | 300 |
| Embedding Model | `mistral-embed` |
| Vector Store | FAISS |
| Search Type | Similarity |
| Retrieved Documents | Top 5 |
| LLM | `mistral-small-latest` |

---

## 📂 Project Structure

```text
YouTube-AI-Assistant/
│
├── assets/
│   └── screenshot.png
│
├── app.py
├── youtube.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── demo.mp4
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/jatin-kumar210/YouTube-AI-Assistant.git
```

---

## 2. Navigate to the Project

```bash
cd YouTube-AI-Assistant
```

---

## 3. Create a Virtual Environment

```bash
python -m venv .venv
```

---

## 4. Activate the Virtual Environment

### Windows

```powershell
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory.

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

> ⚠️ **Important:** Never upload your `.env` file to GitHub.

The project includes `.env.example` so users know which environment variable is required.

---

# ▶️ Usage

## Start the Application

Run the following command:

```bash
streamlit run app.py
```

Streamlit will start the application locally.

You can then open:

```text
http://localhost:8501
```

in your browser.

---

## 📌 How to Use

### Step 1 — Open the Application

Launch the Streamlit application.

### Step 2 — Enter YouTube URL

Paste a valid YouTube video URL.

### Step 3 — Process the Video

Click the **Process Video** button.

The application will:

```text
Fetch Transcript
       ↓
Process Text
       ↓
Create Chunks
       ↓
Generate Embeddings
       ↓
Create FAISS Index
```

### Step 4 — Ask Questions

Once processing is complete, enter your question.

For example:

```text
What is this video about?
```

### Step 5 — Get AI Answer

The system retrieves the most relevant transcript sections and generates an answer using Mistral AI.

---

# 💬 Example Questions

You can ask questions such as:

```text
What is this video about?

Summarize the main points of this video.

Explain the main concept in simple words.

What are the key points discussed?

What examples were mentioned?

What did the speaker conclude?

Explain this topic in detail.

What are the important takeaways from the video?
```

---

# 🎯 Use Cases

This application can be useful for:

- 🎓 Students studying educational videos
- 📚 Researchers analyzing long videos
- 💼 Professionals extracting information from presentations
- 🧑‍💻 Developers learning from technical tutorials
- 📰 Researchers analyzing interviews and discussions
- ⏱️ Anyone who wants quick information from long videos

---

# ⚠️ Limitations

- The application depends on transcript availability.
- Some YouTube videos may not have accessible transcripts.
- YouTube may restrict transcript requests.
- Certain videos may not be supported because of transcript restrictions.
- AI responses depend on the quality of the retrieved transcript context.
- Mistral AI API usage may be subject to account limits.

---

# 🔮 Future Improvements

Some planned improvements include:

- 🌍 Multi-language transcript support
- 🎙️ Automatic speech-to-text for videos without transcripts
- 📄 PDF and document support
- 🧠 Advanced semantic chunking
- 💾 Persistent vector database
- 📊 YouTube video analytics
- 🔗 Timestamp-based answers
- 🎬 Video chapter summarization
- ☁️ Cloud deployment
- 👥 Multi-user support

---

# 🔐 Security

The application uses environment variables to store API credentials.

```text
.env
```

is excluded from Git using `.gitignore`.

### Never commit:

```text
API Keys
Passwords
Tokens
Credentials
.env files
```

Use `.env.example` when sharing the project.

---

# 📦 Requirements

The project dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# 🧪 Example Workflow

```text
                USER
                 │
                 ▼
        Paste YouTube URL
                 │
                 ▼
        Process Video
                 │
                 ▼
       Fetch Transcript
                 │
                 ▼
          Split Text
                 │
                 ▼
        Create Embeddings
                 │
                 ▼
        Store in FAISS
                 │
                 ▼
          Ask Question
                 │
                 ▼
       Retrieve Context
                 │
                 ▼
          Mistral AI
                 │
                 ▼
         AI Generated Answer
```

---

# 🌟 Why This Project?

Traditional video learning requires users to watch an entire video to find specific information.

This project makes video content **queryable**.

Instead of:

```text
Watch 60-minute video
        ↓
Find required information
```

Users can simply:

```text
Paste Video
     ↓
Ask Question
     ↓
Get Relevant Answer
```

This demonstrates a practical implementation of **Retrieval-Augmented Generation (RAG)** using modern AI technologies.

---

# 👨‍💻 Author

<div align="center">

## Jatin Kumar

**B.Tech — Artificial Intelligence & Machine Learning**

Interested in:

**Artificial Intelligence • Machine Learning • Generative AI • RAG • Python • Data Science**

</div>

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps support the project and motivates further development.

---

<div align="center">

### 🚀 Built with Python • LangChain • Mistral AI • FAISS • Streamlit

**YouTube AI Assistant**

</div>
