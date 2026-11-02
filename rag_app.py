import os
import streamlit as st
import pymupdf
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="OOPSMate AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 OOPSMate AI")
st.caption("Your personal AI learning assistant")


if not API_KEY:
    st.error("GOOGLE_API_KEY missing. Please check your .env file.")
    st.stop()


# ==========================================
# 2. LOAD MODELS
# ==========================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_gemini():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=API_KEY
    )


embedding_model = load_embedding_model()
llm = load_gemini()


# ==========================================
# 3. SESSION STATE
# ==========================================

defaults = {
    "chunks": [],
    "embeddings": None,
    "document_name": None,
    "question": "",
    "answer": "",
    "context": "",
    "stage": "ask",
    "simpler_answer": "",
    "quiz_questions": [],
    "quiz_answers": [],
    "quiz_feedback": [],
    "quiz_index": 0,
    "quiz_finished": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# 4. PDF PROCESSING
# ==========================================

def extract_pdf_text(pdf_file):
    text = ""

    with pymupdf.open(
        stream=pdf_file.getvalue(),
        filetype="pdf"
    ) as document:

        for page in document:
            text += page.get_text() + "\n"

    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    return splitter.split_text(text)


def create_embeddings(chunks):
    return embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )


def retrieve_chunks(question, chunks, embeddings, top_k=3):
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    similarities = embeddings @ question_embedding

    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [chunks[i] for i in top_indices]


# ==========================================
# 5. LANGCHAIN PROMPTS
# ==========================================

answer_prompt = ChatPromptTemplate.from_template(
    """
    You are OOPSMate AI, a friendly study assistant.

    Answer the student's question using the provided context.

    Rules:
    - Use simple English.
    - Give examples when helpful.
    - Do not invent facts not supported by the context.
    - If the context does not contain the answer, say so.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


simplify_prompt = ChatPromptTemplate.from_template(
    """
    You are a patient teacher explaining a topic to a beginner.

    The student did not understand the previous explanation.

    Explain the same topic again using:
    1. Very simple English.
    2. Short sentences.
    3. An everyday-life analogy.
    4. A small example from the provided study material.
    5. A short recap at the end.

    Do not change the topic.

    Study context:
    {context}

    Student's original question:
    {question}

    Previous explanation:
    {answer}

    Give a simpler explanation:
    """
)


quiz_prompt = ChatPromptTemplate.from_template(
    """
    You are a teacher creating a quiz from study material.

    Create exactly 3 questions about the topic discussed.

    Use the supplied context and the original question.
    The questions must test understanding, not just memorization.
    Use simple English.
    Do not include answers or hints.
    Each question must be on a separate line and numbered 1, 2, 3.

    Context:
    {context}

    Topic/question:
    {question}

    Previous explanation:
    {answer}
    """
)


answer_chain = answer_prompt | llm | StrOutputParser()
simplify_chain = simplify_prompt | llm | StrOutputParser()
quiz_chain = quiz_prompt | llm | StrOutputParser()


evaluation_prompt = ChatPromptTemplate.from_template(
    """
    You are a fair and helpful teacher evaluating a student's answer.

    Study materiIt helps with code reuse and organization. Bike and Car inherit common features like vehicleId and baseFare from Vehicle, so we don't need to write the same code repeatedly. This makes the program easier to maintain and update.al:
    {context}

    Question:
    {question}

    Student's answer:
    {student_answer}

    Evaluate the answer based on correctness and understanding.

    Give a score from 0 to 10.

    Follow this exact format:

    Score: X/10

    What was correct:
    Explain what the student understood correctly.

    What was missing or incorrect:
    Explain any mistakes or missing points.

    How to improve:
    Give a simple suggestion.

    Be fair. Give partial marks when the answer is partly correct.
    Use simple English.
    """
)

evaluation_chain = evaluation_prompt | llm | StrOutputParser()


# ==========================================
# 6. UPLOAD PDF
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your study PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.session_state.document_name != uploaded_file.name:

        # Reset progress when a different PDF is uploaded.
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.document_name = None
        st.session_state.question = ""
        st.session_state.answer = ""
        st.session_state.context = ""
        st.session_state.stage = "ask"
        st.session_state.simpler_answer = ""
        st.session_state.quiz_questions = []
        st.session_state.quiz_answers = []
        st.session_state.quiz_feedback = []
        st.session_state.quiz_index = 0
        st.session_state.quiz_finished = False

        with st.spinner("Reading your PDF..."):

            pdf_text = extract_pdf_text(uploaded_file)

        if not pdf_text.strip():
            st.error(
                "No readable text found in this PDF. "
                "Scanned PDFs may need OCR."
            )
            st.stop()

        with st.spinner("Creating document embeddings..."):

            chunks = split_text(pdf_text)
            embeddings = create_embeddings(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.document_name = uploaded_file.name

        st.success(
            f"PDF ready! Created {len(chunks)} text chunks."
        )

if not st.session_state.chunks:
    st.info("Upload a PDF to begin.")
    st.stop()


# ==========================================
# 7. ASK QUESTION AND GET RAG ANSWER
# ==========================================

if st.session_state.stage == "ask":

    st.subheader("Step 1: Ask a question")

    question = st.text_input(
        "What do you want to learn from this PDF?",
        key="question_input"
    )

    if st.button("Get Answer", type="primary"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            relevant_chunks = retrieve_chunks(
                question,
                st.session_state.chunks,
                st.session_state.embeddings
            )

            context = "\n\n".join(relevant_chunks)

            with st.spinner("Finding the answer..."):

                try:
                    answer = answer_chain.invoke({
                        "context": context,
                        "question": question
                    })

                    st.session_state.question = question
                    st.session_state.context = context
                    st.session_state.answer = answer
                    st.session_state.simpler_answer = ""
                    st.session_state.stage = "understanding"

                    st.rerun()

                except Exception as error:
                    st.error(f"Unable to generate answer: {error}")


# ==========================================
# 8. UNDERSTANDING CHECK
# ==========================================

if st.session_state.stage == "understanding":

    st.subheader("Step 2: Understand the concept")

    st.markdown("### 🤖 AI Answer")
    st.markdown(st.session_state.answer)

    st.divider()

    st.write("**Did you understand this explanation?**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, I understood", type="primary"):

            st.session_state.stage = "quiz"
            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = []
            st.session_state.quiz_feedback = []
            st.session_state.quiz_index = 0
            st.session_state.quiz_finished = False

            with st.spinner("Creating 3 questions..."):

                try:
                    quiz_text = quiz_chain.invoke({
                        "context": st.session_state.context,
                        "question": st.session_state.question,
                        "answer": st.session_state.answer
                    })

                    questions = []

                    for line in quiz_text.splitlines():
                        line = line.strip()

                        if line and line[0].isdigit():
                            cleaned = line.lstrip("0123456789.):- ")
                            if cleaned:
                                questions.append(cleaned)

                    if len(questions) != 3:
                        st.session_state.quiz_questions = []
                        st.session_state.stage = "understanding"
                        st.error(
                            "AI did not return exactly 3 questions. "
                            "Please click Yes again to retry."
                        )

                    else:
                        st.session_state.quiz_questions = questions

                except Exception as error:
                    st.session_state.stage = "understanding"
                    st.error(f"Unable to create quiz: {error}")

            st.rerun()

    with col2:
        if st.button("❌ No, explain again"):

            with st.spinner("Preparing a simpler explanation..."):

                try:
                    simpler_answer = simplify_chain.invoke({
                        "context": st.session_state.context,
                        "question": st.session_state.question,
                        "answer": st.session_state.answer
                    })

                    st.session_state.simpler_answer = simpler_answer
                    st.session_state.stage = "simpler"

                    st.rerun()

                except Exception as error:
                    st.error(f"Unable to simplify: {error}")


# ==========================================
# 9. SIMPLER EXPLANATION
# ==========================================

if st.session_state.stage == "simpler":

    st.subheader("Step 3: Let's try a simpler explanation")

    st.markdown(st.session_state.simpler_answer)

    st.info(
        "Read the explanation. When you feel ready, choose an option below."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ I understand now", type="primary"):

            st.session_state.answer = st.session_state.simpler_answer
            st.session_state.stage = "understanding"
            st.rerun()

    with col2:
        if st.button("🔄 Explain even more simply"):

            with st.spinner("Simplifying again..."):

                try:
                    simpler_answer = simplify_chain.invoke({
                        "context": st.session_state.context,
                        "question": st.session_state.question,
                        "answer": st.session_state.simpler_answer
                    })

                    st.session_state.simpler_answer = simpler_answer
                    st.rerun()

                except Exception as error:
                    st.error(f"Unable to simplify: {error}")




# ==========================================
# 10. QUIZ: COLLECT ANSWERS, THEN EVALUATE
# ==========================================

if st.session_state.stage == "quiz":

    questions = st.session_state.quiz_questions

    if not questions:
        st.warning("No quiz questions are available.")

        if st.button("Back to understanding check"):
            st.session_state.stage = "understanding"
            st.rerun()

        st.stop()

    # Initialize quiz state
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    # ------------------------------------------
    # FINAL REPORT
    # ------------------------------------------

    if st.session_state.quiz_finished:

        st.subheader("🎉 Your Learning Report")

        feedback_list = st.session_state.quiz_feedback

        total_questions = len(feedback_list)

        total_score = sum(
            item["score"] for item in feedback_list
        )

        max_score = total_questions * 10

        percentage = (
            (total_score / max_score) * 100
            if max_score > 0 else 0
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Your Final Score",
                f"{total_score:g}/{max_score}"
            )

        with col2:
            st.metric(
                "Your Percentage",
                f"{percentage:.1f}%"
            )

        if percentage >= 80:
            st.success(
                "Excellent work! You understand this topic well."
            )

        elif percentage >= 50:
            st.info(
                "Good effort! Review your feedback to improve."
            )

        else:
            st.warning(
                "Keep practicing! Review the explanation "
                "and try again."
            )

        st.divider()

        st.subheader("📝 Detailed Feedback")

        for index, item in enumerate(feedback_list, start=1):

            with st.expander(f"Question {index}", expanded=True):

                st.write(item["question"])

                st.markdown("**Your Answer:**")
                st.write(item["answer"])

                st.markdown(f"**Score: {item['score']}/10**")

                st.markdown("**AI Feedback:**")
                st.markdown(item["feedback"])

        if st.button(
            "🔄 Start a New Learning Session",
            type="primary"
        ):

            st.session_state.question = ""
            st.session_state.answer = ""
            st.session_state.context = ""
            st.session_state.simpler_answer = ""

            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = []
            st.session_state.quiz_feedback = []
            st.session_state.quiz_index = 0

            st.session_state.quiz_finished = False
            st.session_state.quiz_submitted = False
            st.session_state.stage = "ask"

            st.rerun()

        st.stop()

    # ------------------------------------------
    # ALL ANSWERS COLLECTED: FINAL SUBMISSION
    # ------------------------------------------

    if st.session_state.quiz_submitted:

        st.subheader("✅ All Questions Completed!")

        st.success(
            "Your answers have been saved. "
            "Submit your test to see your marks and feedback."
        )

        st.write(
            f"Answers completed: "
            f"{len(st.session_state.quiz_answers)} "
            f"of {len(questions)}"
        )

        if st.button(
            "📊 Submit Final Test & View Results",
            type="primary"
        ):

            feedback_list = []

            with st.spinner(
                "AI is evaluating all your answers. Please wait..."
            ):

                try:
                    import re

                    for index, question in enumerate(questions):

                        student_answer = (
                            st.session_state.quiz_answers[index]
                        )

                        feedback = evaluation_chain.invoke({
                            "context": st.session_state.context,
                            "question": question,
                            "student_answer": student_answer
                        })

                        # Extract score, for example: Score: 8/10
                        match = re.search(
                            r"Score\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
                            feedback,
                            re.IGNORECASE
                        )

                        if match:
                            score = float(match.group(1))
                            score = max(0, min(10, score))
                        else:
                            score = 0

                        feedback_list.append({
                            "question": question,
                            "answer": student_answer,
                            "score": score,
                            "feedback": feedback
                        })

                    # Save results only after evaluation finishes
                    st.session_state.quiz_feedback = feedback_list
                    st.session_state.quiz_finished = True

                    st.rerun()

                except Exception as error:
                    st.error(
                        f"Unable to evaluate your test: {error}"
                    )

        st.stop()

    # ------------------------------------------
    # SHOW ONE QUESTION AT A TIME
    # ------------------------------------------

    index = st.session_state.quiz_index

    st.subheader("Step 4: Test Your Understanding")

    st.progress((index + 1) / len(questions))

    st.write(f"Question {index + 1} of {len(questions)}")

    st.info(questions[index])

    with st.form(key=f"quiz_form_{index}"):

        student_answer = st.text_area(
            "✍️ Write your answer:",
            placeholder="Explain in your own words...",
            height=150,
            key=f"student_answer_{index}"
        )

        submitted = st.form_submit_button(
            "Submit Answer & Continue",
            type="primary"
        )

    if submitted:

        if not student_answer.strip():

            st.warning(
                "Please write your answer before continuing."
            )

        else:

            # Save the answer without evaluating it
            st.session_state.quiz_answers.append(
                student_answer.strip()
            )

            if index + 1 < len(questions):

                # Move directly to the next question
                st.session_state.quiz_index += 1
                st.rerun()

            else:

                # All answers collected; wait for final submission
                st.session_state.quiz_submitted = True
                st.rerun()