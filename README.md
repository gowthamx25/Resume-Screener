# Resume Screener – AI-Powered Resume Analysis System

An end-to-end AI-based Resume Screening system that automatically extracts candidate information from resumes, evaluates skill match against job requirements, and recommends suitable roles.

This project combines **Transformer-based Named Entity Recognition (NER)** with **rule-based skill matching** to provide structured candidate insights from unstructured PDF resumes.

---

## 🚀 Key Features

- Upload single or multiple resumes (PDF)
- Extract candidate name, email, and skills
- Transformer-based Resume NER model
- SpaCy-based fallback for robust name extraction
- Skill matching against job requirements or model resume
- Resume scoring & classification:
  - Selected
  - Shortlisted
  - Rejected
- Job role recommendations based on detected skills
- Automatic CSV report generation

---

## 🧠 System Architecture

1. Resume Upload (PDF)
2. Text Extraction
3. Resume NER Inference (Transformer)
4. Skill Detection & Matching
5. Match Percentage Calculation
6. Candidate Status Decision
7. Job Recommendation
8. Report Generation

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **NLP**: HuggingFace Transformers, SpaCy
- **ML Framework**: PyTorch
- **PDF Processing**: PyPDF2, pdfplumber
- **Data Handling**: Pandas
- **Frontend**: HTML (Jinja2 Templates)

---

## 📂 Project Structure

Resume-Screener/

├── app.py

├── requirements.txt

├── templates/

│ └── index.html

├── outputs/

├── .gitignore

└── README.md

---

## ⚠️ Large Files & Privacy Notice

This repository intentionally excludes the following directories:

- **`resume_ner_model/`**  
  Contains trained Transformer-based NER model checkpoints.  
  These files are large and are excluded to keep the repository lightweight.

- **`uploads/`**  
  Stores user-uploaded resumes (PDFs).  
  Excluded to ensure data privacy and prevent sensitive data exposure.

These folders are created and used **locally at runtime**.

---

## 🔧 Setup Instructions

### 1️⃣ Clone Repository
git clone https://github.com/gowthamx25/Resume-Screener.git

cd Resume-Screener
### 2️⃣ Install Dependencies
pip install -r requirements.txt 

python -m spacy download en_core_web_sm
### 3️⃣ Add Trained Model
Place the trained NER model locally at:

resume_ner_model/checkpoint-66
### 4️⃣ Run Application

python app.py
Access the app at:
http://127.0.0.1:5000
### 📊 Output
Parsed resume details displayed in UI

CSV report generated automatically in outputs/

### Includes:

- Candidate name
- Email
- kills found
- Match percentage
- Status
- Recommended roles

## 📌 Use Cases
- Automated resume screening for recruiters
- Academic resume analysis
- Skill-gap analysis
- Entry-level hiring automation

### 🧑‍💻 Author

Gowtham S

AI & Data Science | Applied ML & MLOps

GitHub: https://github.com/gowthamx25
