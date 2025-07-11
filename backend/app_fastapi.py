import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, BSHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import groq
from fastapi.responses import JSONResponse
import json
import re
from typing import List, Any
import logging
import threading
import demjson3

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
CHROMA_DIR = os.path.join(DATA_DIR, 'chroma_index')
QUIZZES_DIR = os.path.join(DATA_DIR, 'quizzes')
VECTORSTORES_DIR = os.path.join(DATA_DIR, 'vectorstores')

if not os.path.exists(QUIZZES_DIR):
    os.makedirs(QUIZZES_DIR)
if not os.path.exists(VECTORSTORES_DIR):
    os.makedirs(VECTORSTORES_DIR)

quiz_lock = threading.Lock()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_vectorstore_for_file(filename: str):
    path = os.path.join(DATA_DIR, filename)
    file_base = os.path.splitext(filename)[0]
    vs_dir = os.path.join(VECTORSTORES_DIR, file_base)
    if not os.path.exists(vs_dir):
        os.makedirs(vs_dir)
    # Load and split
    if filename.endswith('.pdf'): loader = PyPDFLoader(path)
    elif filename.endswith('.txt'): loader = TextLoader(path)
    elif filename.endswith('.csv'): loader = CSVLoader(path)
    elif filename.endswith('.docx'): loader = Docx2txtLoader(path)
    elif filename.endswith('.html'): loader = BSHTMLLoader(path)
    else: raise Exception('Unsupported file type')
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(texts, embeddings, persist_directory=vs_dir)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.pdf', '.txt', '.docx', '.csv', '.html']:
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    build_vectorstore_for_file(filename)
    return {"filename": filename, "message": "File uploaded and indexed."}

@app.get("/list_pdfs")
async def list_pdfs():
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    return {"pdfs": pdfs}

def validate_quiz_format(quiz: Any) -> bool:
    if not isinstance(quiz, list): return False
    for q in quiz:
        if not isinstance(q, dict): return False
        if not all(k in q for k in ("question", "options", "correct_answers")): return False
        if not isinstance(q["options"], list) or len(q["options"]) != 4: return False
        if not isinstance(q["correct_answers"], list): return False
        if not (1 <= len(q["correct_answers"]) <= 2): return False
        if not all(opt in ["A", "B", "C", "D"] for opt in q["correct_answers"]): return False
    return True

FORBIDDEN_WORDS = [
    "module", "objective", "goal", "purpose", "guide", "section", "introduction", "overview", "author", "copyright",
    "learning outcome", "table of contents", "contents", "about this", "how to use", "document", "metadata"
]

def flatten(l):
    for el in l:
        if isinstance(el, list):
            yield from flatten(el)
        else:
            yield el

def filter_and_deduplicate_questions(questions):
    seen = set()
    unique = []
    flat_questions = list(flatten(questions))
    for q in flat_questions:
        if (
            isinstance(q, dict)
            and 'question' in q
            and isinstance(q.get('options', []), list)
            and all(isinstance(opt, str) for opt in q.get('options', []))
            and not any(word in q['question'].lower() for word in FORBIDDEN_WORDS)
            and not any(word in " ".join(q.get('options', [])).lower() for word in FORBIDDEN_WORDS)
            and q['question'] not in seen
        ):
            seen.add(q['question'])
            unique.append(q)
        else:
            logging.warning(f"Filtered out malformed, non-technical, or duplicate question: {q}")
    return unique

def get_quiz_path(filename: str) -> str:
    return os.path.join(QUIZZES_DIR, f"{filename.replace('/','_').replace('\\','_')}.quiz.json")

def save_quiz_to_file(quiz: list, filename: str):
    with quiz_lock:
        with open(get_quiz_path(filename), 'w', encoding='utf-8') as f:
            json.dump({'quiz': quiz}, f, ensure_ascii=False, indent=2)

def load_quiz_from_file(filename: str) -> dict:
    path = get_quiz_path(filename)
    if not os.path.exists(path): return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if 'quiz' in data else {}
    except Exception:
        return {}

def retrieve_relevant_chunks(pdf_file: str, top_k: int = 5) -> str:
    file_base = os.path.splitext(pdf_file)[0]
    vs_dir = os.path.join(VECTORSTORES_DIR, file_base)
    if not os.path.exists(vs_dir):
        raise Exception(f"No vectorstore found for {pdf_file}. Please upload and index it first.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=vs_dir, embedding_function=embeddings)
    query = "Generate quiz questions about technical concepts from this document."
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n".join([d.page_content for d in docs])

def fix_llm_json(s):
    # Escape unescaped double quotes inside string values (not perfect, but helps for most LLM output)
    def replacer(match):
        inner = match.group(2)
        fixed = inner.replace('"', '\"')
        return f'{match.group(1)}"{fixed}"'
    # This regex finds values between : " and the next "
    return re.sub(r'(:\s*")([^\"]*?)(")', replacer, s)

def clean_extra_quotes(s):
    # Remove double double-quotes at the start of any string value for 'question'
    return re.sub(r'("question":\s*)""', r'\1"', s)

def parse_llm_json(answer):
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        try:
            return demjson3.decode(answer)
        except Exception as e:
            logging.error(f"demjson3 also failed: {e}\nRaw LLM output: {answer}")
            raise Exception(f"LLM output is not valid JSON: {e}")

def generate_quiz_llm(selected_pdf: str, mode: str = 'strict') -> list:
    context = retrieve_relevant_chunks(selected_pdf, top_k=5)
    if not context.strip():
        raise Exception("No relevant content found.")
    example_json = '''[
      {
        "question": "What is the capital of France?",
        "options": ["Berlin", "Madrid", "Paris", "Rome"],
        "correct_answers": ["C"]
      }
    ]'''
    if mode == 'strict':
        prompt = (
            "You are an expert quiz generator specialized in IT infrastructure. "
            "Create exactly 10 HARD multiple-choice questions ONLY based on the following text. "
            "If you cannot generate 10 valid technical questions, generate as many as you can, but only if there is not enough technical content. Do not invent content. "
            "DO NOT ask about:\n"
            "- Copyrights, authors, contributors, metadata, document/module titles\n"
            "- Company names, disclaimers, or publishing dates\n"
            "\nEach question must:\n"
            "- Have exactly 4 options (A, B, C, D)\n"
            "- Have 1 or 2 correct answers (e.g., ['A'] or ['A','C'])\n"
            "- Be in clear, professional English\n"
            "- Target experienced professionals\n"
            "\nRespond ONLY with a JSON array (no text around):\n"
            f"{example_json}\n\n"
            f"Text:\n{context}"
        )
    else:  # flexible mode for update
        prompt = (
            "You are an expert quiz generator specialized in IT infrastructure. "
            "Create up to 10 HARD multiple-choice questions ONLY based on the following text. "
            "If you cannot generate 10 valid technical questions, generate as many as you can based on the provided text. Do not invent content. "
            "DO NOT ask about:\n"
            "- Copyrights, authors, contributors, metadata, document/module titles\n"
            "- Company names, disclaimers, or publishing dates\n"
            "\nEach question must:\n"
            "- Have exactly 4 options (A, B, C, D)\n"
            "- Have 1 or 2 correct answers (e.g., ['A'] or ['A','C'])\n"
            "- Be in clear, professional English\n"
            "- Target experienced professionals\n"
            "\nRespond ONLY with a JSON array (no text around):\n"
            f"{example_json}\n\n"
            f"Text:\n{context}"
        )
    client = groq.Client(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()
    answer = re.sub(r'```json|```', '', answer).strip()
    answer = fix_llm_json(answer)
    answer = clean_extra_quotes(answer)
    try:
        quiz = parse_llm_json(answer)
    except Exception as e:
        logging.error(f"Quiz generation failed: {e}")
        raise Exception(f"Quiz generation failed: {e}")
    if not isinstance(quiz, list):
        raise Exception("Quiz is not a list")
    return quiz

def auto_index_all_files():
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.pdf', '.txt', '.docx', '.csv', '.html'))]
    for filename in files:
        file_base = os.path.splitext(filename)[0]
        vs_dir = os.path.join(VECTORSTORES_DIR, file_base)
        if not os.path.exists(vs_dir):
            try:
                build_vectorstore_for_file(filename)
                print(f"Indexed: {filename}")
            except Exception as e:
                print(f"Failed to index {filename}: {e}")

def generate_and_save_quiz(filename: str):
    quiz = generate_quiz_llm(selected_pdf=filename)
    save_quiz_to_file(quiz, filename)
    return quiz

def auto_generate_all_quizzes():
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    for pdf in pdfs:
        quiz_path = get_quiz_path(pdf)
        if not os.path.exists(quiz_path):
            try:
                print(f"Generating quiz for {pdf}...")
                generate_and_save_quiz(pdf)
            except Exception as e:
                print(f"Failed to generate quiz for {pdf}: {e}")

# Call on startup
auto_index_all_files()
auto_generate_all_quizzes()

# Also call before quiz generation to catch new files
@app.post("/generate_quiz")
async def generate_quiz(filename: str = Query(...)):
    auto_index_all_files()
    try:
        quiz = generate_quiz_llm(selected_pdf=filename, mode='strict')
        filtered = filter_and_deduplicate_questions(quiz)
        filtered = filtered[:10]
        if len(filtered) < 3:
            raise Exception("Not enough valid technical questions could be generated from this document.")
        save_quiz_to_file(filtered, filename)
        if len(filtered) < 10:
            logging.warning(f"Only {len(filtered)} valid questions generated for {filename}.")
        return {'quiz': filtered}
    except Exception as e:
        logging.error(f"Quiz generation failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get_quiz")
async def get_quiz(filename: str = Query(...)):
    data = load_quiz_from_file(filename)
    if data.get('quiz'):
        return data['quiz']
    return JSONResponse(status_code=404, content={"error": "Quiz not found"})

@app.post("/generate_all_quizzes")
async def generate_all_quizzes():
    auto_generate_all_quizzes()
    return {"status": "All quizzes generated for PDFs without a quiz."}

@app.post("/update_quiz")
async def update_quiz(filename: str = Query(...)):
    # Load existing quiz
    existing = load_quiz_from_file(filename).get('quiz', [])
    # Always preserve the first 10 questions
    preserved = existing[:10]
    # Generate up to 10 new questions (flexible)
    new_questions = generate_quiz_llm(selected_pdf=filename, mode='flexible')
    # Filter out any new questions that are duplicates of the preserved ones
    existing_questions_set = set(q['question'] for q in preserved if 'question' in q)
    unique_new_questions = [q for q in new_questions if 'question' in q and q['question'] not in existing_questions_set]
    # Only take up to 10 new unique questions
    unique_new_questions = unique_new_questions[:10]
    # Combine preserved and new unique questions
    combined = preserved + unique_new_questions
    # Optionally, cap the total at 50 questions
    combined = combined[:50]
    if len(combined) == 0:
        raise Exception("No valid technical questions could be generated from this document.")
    if not validate_quiz_format(combined):
        raise Exception("Invalid quiz format")
    save_quiz_to_file(combined, filename)
    return {"quiz": combined}

