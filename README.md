# 🤖 OOPSMate AI Advanced

### Your Personal AI-Powered Learning Assistant

OOPSMate AI Advanced is an AI-powered learning assistant designed to help students understand complex topics, learn from PDF study materials, practice with quizzes, and evaluate their knowledge through AI-generated feedback.

Built using Python, Streamlit, LangChain, Sentence Transformers, and Google Gemini, this project combines document-based question answering with personalized learning support.

## ✨ Features

- 📄 **PDF Document Upload** — Upload study materials in PDF format.
- 🔍 **Smart Content Retrieval** — Retrieve relevant information from uploaded documents using semantic similarity.
- 🤖 **AI-Powered Question Answering** — Ask questions and receive context-based answers.
- 🧠 **Beginner-Friendly Explanations** — Simplify difficult concepts into easier explanations.
- 📝 **Interactive Quizzes** — Generate three questions based on the uploaded study material.
- 🎯 **AI-Based Evaluation** — Get scores and feedback on quiz answers.
- 📊 **Performance Summary** — View your score, percentage, and learning feedback.
- ⚡ **Interactive Web Interface** — Learn through a simple and user-friendly Streamlit application.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| Google Gemini | AI-generated answers and feedback |
| LangChain | Prompt templates and AI chains |
| Sentence Transformers | Text embeddings |
| NumPy | Vector similarity calculations |
| PyMuPDF | PDF text extraction |
| python-dotenv | Environment variable management |

## 🔄 How It Works

OOPSMate AI Advanced uses a Retrieval-Augmented Generation (RAG) approach to answer questions using uploaded study materials.

1. **Upload a PDF:** The user uploads a study document.
2. **Extract Text:** The application extracts text from the PDF.
3. **Split into Chunks:** The text is divided into smaller sections for processing.
4. **Generate Embeddings:** Sentence Transformers converts text chunks into numerical representations.
5. **Retrieve Relevant Content:** NumPy calculates similarity scores to identify relevant chunks.
6. **Generate Answers:** Google Gemini generates answers using the retrieved content.
7. **Simplify Learning:** Users can request easier explanations of complex concepts.
8. **Practice with Quizzes:** The application generates three questions based on the study material.
9. **Evaluate Performance:** Users receive AI-generated scores and feedback after submitting their answers.

## 📁 Project Structure

```text
OOPSMate-AI-Advanced/
│
├── .venv/                # Python virtual environment
├── .env                  # API key configuration (private)
├── .gitignore            # Git ignore rules
├── langchain_demo.py     # LangChain demonstration
├── rag_app.py            # Main Streamlit application
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## ⚙️ Installation and Setup

Follow these steps to run the application on your local machine.

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd OOPSMate-AI-Advanced
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 5. Configure Your API Key

Create a `.env` file in the project root directory and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_google_api_key
```

Replace `your_google_api_key` with your actual API key.

**Important:** Never share your API key or upload your `.env` file to GitHub.

### 6. Run the Application

Start the Streamlit application using:

```bash
python -m streamlit run rag_app.py
```

Open the local URL displayed in your terminal to access the application.

## 🌐 Deployment

You can deploy OOPSMate AI Advanced using Streamlit Community Cloud.

1. Push the project to a GitHub repository.
2. Connect your GitHub account to Streamlit Community Cloud.
3. Select your repository and the `main` branch.
4. Set the main application file to `rag_app.py`.
5. Open the application's **Advanced Settings**.
6. Add your API key under Streamlit Secrets:

```toml
GOOGLE_API_KEY = "your_google_api_key"
```

7. Save the settings and deploy the application.

Keep your API key private and monitor your API usage and applicable limits.

## 🔐 Security and Privacy

- Store API keys in environment variables or Streamlit Secrets.
- Exclude `.env` and other sensitive files from Git version control.
- Never publish passwords, API keys, or private credentials.
- Be careful when uploading confidential documents to a publicly accessible application.

## 🎯 Project Objectives

The main objective of OOPSMate AI Advanced is to make learning more interactive, accessible, and personalized.

By combining document-based question answering, simplified explanations, quizzes, and automated feedback, the application helps students understand study materials and assess their knowledge.

## 🚀 Future Enhancements

- [ ] Support multiple PDF documents.
- [ ] Add conversation history.
- [ ] Introduce topic-wise quizzes.
- [ ] Add different quiz difficulty levels.
- [ ] Provide downloadable performance reports.
- [ ] Improve document retrieval accuracy.
- [ ] Add support for additional document formats.
- [ ] Track learning progress over time.

## 👩‍💻 Author

**Shivani Pandey**

**Project:** OOPSMate AI Advanced

---

⭐ If you find this project interesting, consider giving the repository a star!