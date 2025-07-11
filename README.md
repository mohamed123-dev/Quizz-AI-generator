# Quizz-AI-generator

A web application that generates AI-powered quizzes. The project consists of a FastAPI backend and a simple HTML/JavaScript/CSS frontend. Data is managed locally, with support for custom datasets.

## Features
-Generate quizzes dynamically using LLaMA via Groq
-Fast and lightweight backend with FastAPI
-Simple, user-friendly frontend interface
-Local data storage and indexing with Chroma
-Support for adding your own custom datasets

## Folder Structure
```
Quizz-AI-generator/
  backend/
    app_fastapi.py         # FastAPI backend application
    data/
      chroma_index/       # Chroma index database
  frontend/
    index.html            # Main frontend page
    script.js             # Frontend logic
    style.css             # Frontend styles
  data/
    put_your_data_here    # Placeholder for custom data
  requirements.txt        # Python dependencies
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd Quizz-AI-generator
```

### 2. Backend Setup (FastAPI)
1. Install Python 3.8+
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```
   uvicorn backend.app_fastapi:app --reload
   ```
   The backend will be available at `http://127.0.0.1:8000`.

### 3. Frontend Setup
No build step is required. Simply open `frontend/index.html` in your browser. Make sure the backend is running for full functionality.


## Requirements
- Python 3.8+
- See `requirements.txt` for Python dependencies
- Modern web browser (for frontend)

## Custom Data
- Place your custom data in the `data/put_your_data_here` directory.
- The backend will use this data for quiz generation (if implemented).



## Credits
-FastAPI
-Chroma
-Groq for lightning-fast LLaMA API integration

---
Feel free to customize this README with more details, screenshots, or usage examples as your project evolves.
