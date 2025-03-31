from flask import Flask, request, render_template, send_from_directory
import os
import docx2txt
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
import json

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PREVIOUS_RESUMES'] = 'previous_resumes.json'  # File to store previous resumes

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Ensure previous_resumes.json file exists and is valid
if not os.path.exists(app.config['PREVIOUS_RESUMES']):
    with open(app.config['PREVIOUS_RESUMES'], 'w') as file:
        json.dump([], file)  # Initialize with an empty list

# Load BERT model for embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


# Load previous resumes from file
def load_previous_resumes():
    try:
        with open(app.config['PREVIOUS_RESUMES'], 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


# Save previous resumes to file
def save_previous_resumes(resumes):
    with open(app.config['PREVIOUS_RESUMES'], 'w') as file:
        json.dump(resumes, file)


# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()


# Function to extract text from a DOCX file
def extract_text_from_docx(file_path):
    return docx2txt.process(file_path).strip()


# Function to extract text from a TXT file
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


# Function to determine file type and extract text
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    return ""


# Function to check if the text is a resume
def is_resume(text):
    resume_keywords = [
        "experience", "education", "skills", "summary", "objective",
        "work history", "projects", "certifications", "achievements",
        "contact information", "phone", "email", "linkedin"
    ]
    keyword_count = sum(1 for keyword in resume_keywords if keyword.lower() in text.lower())
    return keyword_count >= 3


# Function to extract meaningful keywords
def extract_meaningful_keywords(text):
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only alphabets and spaces
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase

    # Extract words
    words = re.findall(r'\b\w+\b', text)

    # Filter out stopwords and short words
    meaningful_keywords = {
        word for word in words
        if word not in stop_words and len(word) > 2 and word.isalpha()
    }

    return meaningful_keywords


# Function to display extracted keywords
def display_extracted_keywords(text, source):
    keywords = extract_meaningful_keywords(text)
    print(f"Extracted keywords from {source}: {keywords}")
    return keywords


# Check if a section exists using regex
def has_section(text, section):
    pattern = rf'(?i)\b{section}\b.*(\n|\r)'
    return bool(re.search(pattern, text))


# Generate resume improvement suggestions
def generate_improvements(resume_text, job_description):
    suggestions = []
    sections = ['experience', 'education', 'skills']
    for section in sections:
        if not has_section(resume_text, section):
            suggestions.append(f"Consider adding a dedicated '{section.capitalize()}' section.")

    job_keywords = extract_meaningful_keywords(job_description)
    resume_keywords = extract_meaningful_keywords(resume_text)
    missing_keywords = job_keywords - resume_keywords

    if missing_keywords:
        suggestions.append(f"Consider including these relevant keywords: {', '.join(list(missing_keywords)[:5])}.")

    leadership_phrases = ["led a team", "managed a team", "supervised", "team lead", "project lead"]
    if not any(phrase in resume_text.lower() for phrase in leadership_phrases):
        suggestions.append("Highlight any leadership experience you may have.")

    if len(resume_text.split()) < 300:
        suggestions.append("Your resume seems too short; consider adding more details about your experience.")
    if 'certification' not in resume_text.lower():
        suggestions.append("Consider including any relevant certifications to strengthen your profile.")
    if 'projects' not in resume_text.lower():
        suggestions.append("Mention any key projects you have worked on to showcase your experience.")
    if 'achievements' not in resume_text.lower():
        suggestions.append("Highlight any achievements or awards to stand out from other candidates.")

    return suggestions[:6]


@app.route("/")
def home():
    return render_template('matchresume.html')


@app.route("/matcher", methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        if not job_description or not resume_files:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        # Extract and display keywords from the job description
        job_keywords = display_extracted_keywords(job_description, "Job Description")

        resume_texts = []
        resume_names = []
        resume_suggestions = {}

        for resume_file in resume_files:
            if resume_file.filename == '':
                continue

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(file_path)
            resume_text = extract_text(file_path)

            if resume_text:
                if not is_resume(resume_text):
                    resume_suggestions[resume_file.filename] = ["This file does not appear to be a valid resume."]
                    continue

                # Extract and display keywords from the resume
                resume_keywords = display_extracted_keywords(resume_text, f"Resume: {resume_file.filename}")

                resume_texts.append(resume_text)
                resume_names.append(resume_file.filename)
                resume_suggestions[resume_file.filename] = generate_improvements(resume_text, job_description)

        if not resume_texts:
            return render_template('matchresume.html', message="No valid resumes were processed.")

        job_embedding = bert_model.encode(job_description)
        resume_embeddings = bert_model.encode(resume_texts)

        similarities = cosine_similarity([job_embedding], resume_embeddings)[0]
        similarity_scores = [(resume_names[i], float(similarities[i])) for i in range(len(resume_names))]
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        top_resumes = [x[0] for x in similarity_scores]
        top_scores = [round(x[1] * 100, 2) for x in similarity_scores]
        top_suggestions = [resume_suggestions[x[0]] for x in similarity_scores]

        previous_resumes = load_previous_resumes()
        for i, resume_name in enumerate(top_resumes):
            previous_resumes.append({
                "filename": resume_name,
                "score": top_scores[i],
                "suggestions": top_suggestions[i]
            })
        save_previous_resumes(previous_resumes)

        better_resumes = [
            {
                "filename": resume['filename'],
                "score": resume['score'],
                "suggestions": resume['suggestions'],
                "file_path": os.path.join(app.config['UPLOAD_FOLDER'], resume['filename'])
            }
            for resume in previous_resumes if resume['score'] > top_scores[0]
        ]

        return render_template('matchresume.html', message="Top matching resumes:", top_resumes=top_resumes,
                               similarity_scores=top_scores, resume_suggestions=top_suggestions,
                               better_resumes=better_resumes)

    return render_template('matchresume.html')


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route("/previous_resumes")
def previous_resumes():
    previous_resumes = load_previous_resumes()
    return render_template('previous_resumes.html', previous_resumes=previous_resumes)


if __name__ == '__main__':
    app.run(debug=True)