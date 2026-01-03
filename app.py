from flask import Flask, render_template, request, send_file
import PyPDF2
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import uuid
import spacy

app = Flask(__name__)

# ------------------ Load trained Resume NER model ------------------
model_path = "./resume_ner_model/checkpoint-66"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()
id2label = model.config.id2label

# Load SpaCy for better name extraction
nlp = spacy.load("en_core_web_sm")

# ------------------ Helper Functions ------------------
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def clean_text(text):
    text = re.sub(r"\s+", " ", text)  
    text = re.sub(r",\s*", ", ", text)  
    return text.strip()

def reconstruct_entities(tokens, predictions):
    entities = {"Name": [], "Email_Address": [], "Skills": []}
    current_entity = None
    buffer = []

    for token, pred_id in zip(tokens, predictions):
        label = id2label.get(pred_id, "O")
        if label.startswith("B-"):
            if current_entity and buffer and current_entity in entities:
                entities[current_entity].append("".join(buffer))
            buffer = [token.replace("##", "")]
            current_entity = label.split("-")[1]
            if current_entity in ["PER", "PERSON", "Candidate"]:
                current_entity = "Name"
            elif current_entity in ["EMAIL", "Mail", "Email"]:
                current_entity = "Email_Address"
            elif current_entity in ["SKILL", "Skill", "Technology"]:
                current_entity = "Skills"
        elif label.startswith("I-") and current_entity:
            buffer.append(token.replace("##", ""))
        else:
            if current_entity and buffer and current_entity in entities:
                entities[current_entity].append("".join(buffer))
            buffer = []
            current_entity = None

    if current_entity and buffer and current_entity in entities:
        entities[current_entity].append("".join(buffer))

    for key in entities:
        entities[key] = ", ".join(entities[key]) if entities[key] else "Not Found"

    return entities

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    name_match = re.search(r"Name[:\-\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)", text)
    if name_match:
        return name_match.group(1).strip()
    lines = text.split("\n")
    for line in lines[:5]:
        possible = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", line)
        if possible:
            return possible[0].strip()
    return "Not Found"

def ner_predict(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    tokens_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    entities = reconstruct_entities(tokens_list, predictions)
    if not entities.get("Name") or entities["Name"] == "Not Found":
        entities["Name"] = extract_name(text)
    return entities

def extract_email(text):
    # Find all possible emails
    matches = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if matches:
        # Return the first one, or join if multiple
        return ", ".join(set(matches))
    return "Not Found"


def match_skills(text, required_skills):
    text_lower = text.lower()
    skills_found = [skill for skill in required_skills if skill.lower() in text_lower]
    return skills_found

def calculate_match_percentage(required_skills, skills_found):
    if not required_skills:
        return 0
    return round((len(skills_found) / len(required_skills)) * 100, 2)

def save_report(results, output_path="outputs"):
    os.makedirs(output_path, exist_ok=True)
    report_file = os.path.join(output_path, f"resume_report_{uuid.uuid4().hex}.csv")
    pd.DataFrame(results).to_csv(report_file, index=False)
    return report_file

# ------------------ Job Recommendation ------------------
job_database = {
    "Python": ["Python Developer", "Data Scientist", "ML Engineer"],
    "Java": ["Java Developer", "Backend Engineer"],
    "C++": ["C++ Developer", "Embedded Engineer"],
    "Machine Learning": ["ML Engineer", "AI Researcher", "Data Scientist"],
    "AI": ["AI Engineer", "Research Scientist"],
    "DVC": ["MLOps Engineer", "Data Engineer"],
    "Git": ["DevOps Engineer", "Software Engineer"],
    "Docker": ["DevOps Engineer", "Cloud Engineer"],
    "HTML": ["Frontend Developer", "Web Designer"],
    "JavaScript": ["Frontend Developer", "Full Stack Developer"],
    "JSP": ["Java Developer", "Web Developer"],
    "Servlet": ["Java Web Developer"],
    "DBMS": ["Database Administrator", "Backend Developer"],
    "SQL": ["Database Engineer", "BI Developer"],
    "Oracle": ["Oracle DBA", "Data Engineer"],
    "Spring": ["Java Spring Developer", "Backend Engineer"]
}

master_skills = list(job_database.keys())  # master skills list

def find_skills_in_resume(text):
    text_lower = text.lower()
    return [skill for skill in master_skills if skill.lower() in text_lower]

def recommend_jobs(skills_found):
    recommended = []
    for skill in skills_found:
        recommended.extend(job_database.get(skill, []))

    if recommended:
        unique_jobs = list(dict.fromkeys(recommended))  # remove duplicates
        return unique_jobs[:3]  # ✅ only top 3 jobs
    else:
        if skills_found:
            return [f"Based on your skills ({', '.join(skills_found)}), consider entry-level roles."]
        else:
            return ["No skills detected. Please update your resume with relevant skills."]

# ------------------ Routes ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    report_file = None

    if request.method == "POST":
        mode = request.form.get("mode")
        requirement_type = request.form.get("requirement_type")

        uploaded_files = []
        required_skills = []

        # Skills requirement
        if requirement_type == "skills":
            required_skills = request.form.get("skills", "").split(",")
            required_skills = [s.strip() for s in required_skills if s.strip()]
        elif requirement_type == "model_resume":
            model_resume = request.files.get("model_resume")
            if model_resume and model_resume.filename.endswith(".pdf"):
                model_resume_path = os.path.join("uploads", "model_resume.pdf")
                os.makedirs("uploads", exist_ok=True)
                model_resume.save(model_resume_path)
                model_resume_text = clean_text(extract_text_from_pdf(model_resume_path))

                # ✅ Use skill matcher instead of NER for model resume
                required_skills = find_skills_in_resume(model_resume_text)


        # Resume input
        if mode == "file":
            file = request.files.get("resume_file")
            if file and file.filename.endswith(".pdf"):
                uploaded_files.append(file)
        elif mode == "folder":
            folder_files = request.files.getlist("resume_folder")
            for file in folder_files:
                if file.filename.endswith(".pdf"):
                    uploaded_files.append(file)

        # Process resumes
        for uploaded_file in uploaded_files:
    # Keep original filename (may contain subfolders from webkitdirectory)
            filename = uploaded_file.filename if hasattr(uploaded_file, "filename") else os.path.basename(uploaded_file.name)

            # Normalize path separators
            safe_filename = filename.replace("\\", "/")

            # Build full path under uploads/
            file_path = os.path.join("uploads", safe_filename)

            # Make sure subfolders exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())


            text = clean_text(extract_text_from_pdf(file_path))
            entities = ner_predict(text)
            email = extract_email(text)

            all_skills_in_resume = find_skills_in_resume(text)
            skills_found = match_skills(text, required_skills)
            match_percent = calculate_match_percentage(required_skills, skills_found)

            status = "Rejected"
            if match_percent >= 60:
                status = "Selected"
            elif 30 <= match_percent < 60:
                status = "Shortlisted"

            recommended = recommend_jobs(all_skills_in_resume)

            results.append({
                "File": filename,
                "Name": entities.get("Name", "Not Found"),
                "Email": email,
                "Skills Found": ", ".join(all_skills_in_resume) if all_skills_in_resume else "Not Found",
                "Skills Matched": ", ".join(skills_found) if skills_found else "Not Found",
                "Matched %": match_percent,
                "Status": status,
                "Recommended Jobs": recommended[:3]# ✅ list of top 3 jobs
            })

        if results:
            report_file = save_report(results)

    return render_template("index.html", results=results, report_file=report_file)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_file(filename, as_attachment=True)

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(debug=True)
